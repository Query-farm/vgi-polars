# Copyright 2026 Query Farm LLC - https://query.farm

"""The `register_io_source` callback that backs `VgiTable.scan()`.

Design Principle 1 (load-bearing — see README/CLAUDE.md): pushdown is an
optimization, never a correctness delegation. Verified empirically: Polars does
NOT re-verify a predicate an `io_source` claims to have handled (an io_source
that silently ignores `predicate` gets all rows back, unfiltered, in the final
`.collect()` result). VGI worker implementations are written and tested against
DuckDB's own client, which *does* always re-verify — so a VGI table function may
declare `filter_pushdown`/`projection_pushdown` and still apply either only
approximately. Given neither side can be trusted alone, this module:

  1. Attempts filter/projection pushdown only when the resolved scan function's
     `FunctionInfo` declares support for it (an optimization).
  2. ALWAYS applies the complete, original `with_columns` / `predicate` locally
     to every batch before yielding it, regardless of what was pushed or what
     the worker claims to have done.
  3. ALWAYS truncates to `n_rows` locally rather than trusting a downstream
     `head()`/`slice()` operator to do it — for the same reason: once Polars
     believes the source honored a pushdown, the operator that would otherwise
     enforce it is elided from the physical plan.

A partial or entirely-failed pushdown translation is therefore only ever a
performance loss, never a correctness one.

**Splits.** When the resolved scan function advertises `FunctionInfo.
supports_splits`, the scan is driven through `Client.table_function_plan()` +
per-split `Client.table_function(split_tokens=...)` redemption instead of one
whole-scan call — see `_iter_splits_sequential`'s docstring for why this is
*sequential* (one split at a time, in the order the worker returned them), not
parallel: Polars' `register_io_source` gets scan parallelism only from
independent generator *instances* for repeated plan occurrences (self-join/
concat/collect_all) — there is no mechanism here for cooperatively driving one
generator's work across threads, so redeeming splits one at a time client-side
concurrency gains nothing. It's still worth doing over skipping splits
entirely: splits are the client-scoped decomposition the worker actually
tested and tuned against (bounded per-split cost, exact-count/statistics
metadata, replayability), and sequential-but-correct beats leaving that path
dark.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl
import pyarrow as pa
from vgi.arguments import Arguments

from vgi_polars._filter_translate import translate_predicate
from vgi_polars._polars_compat import arrow_to_df
from vgi_polars._result_cache import get_default_cache, parse_cache_control
from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from vgi.catalog.catalog_interface import FunctionInfo, ScanFunctionResult
    from vgi.client.client import Client

    from vgi_polars.catalog import VgiCatalog


class _ScanSource(Protocol):
    """What `make_io_source` needs from its `table` argument.

    `VgiTable` satisfies this structurally, as does `_multi_branch.py`'s
    duck-typed `_BranchTable` (one `ScanBranch` masquerading as a table for
    exactly this function) — making the duck-typing explicit and
    mypy-checkable, rather than accepting `Any`.
    """

    _catalog: VgiCatalog
    schema_name: str
    name: str
    at_unit: str | None
    at_value: str | None

    def _scan_function_get(self) -> ScanFunctionResult: ...
    def _function_info_get(self) -> FunctionInfo | None: ...
    def scan_function_schema(self) -> str: ...
    def required_filters(self) -> list[list[str]]: ...


# `hasattr`-style capability guard for `batch_metadata_callback` (new — see
# CLAUDE.md's "Table-function result cache" section), computed once. Mirrors
# the splits/multi-branch degradation pattern elsewhere in this module: an
# installed vgi-python that predates the parameter just never gets result
# caching, no `TypeError`.
try:
    import inspect

    from vgi.client.client import Client as _RuntimeClient

    _table_function_params = inspect.signature(_RuntimeClient.table_function).parameters
    _SUPPORTS_RESULT_CACHE = "batch_metadata_callback" in _table_function_params
    # `table_function(at_unit=..., at_value=...)` is equally new (same
    # vgi-python fix, see CLAUDE.md's "Time travel" section) — passing it
    # unconditionally on an older install raises `TypeError` on *every*
    # scan, not just a time-travel one, since it's an unconditional kwarg
    # below. Checked separately from `_SUPPORTS_RESULT_CACHE` even though
    # both landed in the same vgi-python commit, so this module doesn't
    # silently break if that ever changes.
    _SUPPORTS_TABLE_FUNCTION_AT_CLAUSE = "at_unit" in _table_function_params
except Exception:  # noqa: BLE001 - never let a capability probe break import
    _SUPPORTS_RESULT_CACHE = False
    _SUPPORTS_TABLE_FUNCTION_AT_CLAUSE = False


def _canonical_arguments(arguments: Arguments) -> tuple[Any, ...] | None:
    """A hashable snapshot of `arguments`' values for use in a cache key.

    Returns `None` if any value isn't hashable (e.g. a struct/list-typed argument
    decodes to an unhashable `dict`/`list` via `.as_py()`) — caching is just skipped
    for that call, never a correctness issue (mirrors `_scalar.py`'s
    `_dedup_positions` unhashable fallback).
    """
    try:
        positional = tuple(s.as_py() if s is not None else None for s in arguments.positional)
        named = tuple(sorted((k, v.as_py()) for k, v in (arguments.named or {}).items()))
        hash((positional, named))
    except TypeError:
        return None
    return positional, named


def _check_required_filters(
    required_filters: list[list[str]], predicate: pl.Expr | None, schema_name: str, name: str
) -> None:
    """Cost-safety guard, not a correctness one — see `VgiTable.required_filters`'s docstring.

    `required_filters` is an AND-of-OR-groups of column names; a group entry may
    be a dotted struct-subfield path (e.g. `"s.a"`), which `root_names()` can't
    see through (it reports the top-level `Column` reference, `"s"`, not the
    subfield actually touched inside a `.struct.field(...)` chain). Rather than
    parse struct-access expressions to verify the exact subfield, this takes the
    conservative approximation the plan calls for: a dotted requirement is
    satisfied if the predicate references the requirement's top-level column at
    all (i.e. `"s.a"` is satisfied by any predicate touching `"s"`) — this can
    pass a predicate that doesn't actually touch the required subfield, but it
    never *blocks* a query that does, and it's a safety net against the common
    case (no filter at all), not a full pushdown-translatability check.
    """
    if not required_filters:
        return
    referenced = set(predicate.meta.root_names()) if predicate is not None else set()

    def _group_satisfied(group: list[str]) -> bool:
        for entry in group:
            top_level = entry.split(".", 1)[0]
            if top_level in referenced:
                return True
        return False

    for group in required_filters:
        if not _group_satisfied(group):
            raise VgiPolarsError(
                f"{schema_name}.{name} requires a filter on one of {group!r} — "
                "none of these columns are referenced by the query's predicate"
            )


class _RemainingBudget:
    """Mutable `n_rows` tracker shared across possibly-many `table_function` calls.

    Shared across one whole-scan call, or one per split, so an early exit
    partway through one call can also stop the caller from starting the next one.
    """

    __slots__ = ("exhausted", "remaining")

    def __init__(self, n_rows: int | None) -> None:
        self.remaining = n_rows
        self.exhausted = False


def _process_batches(
    gen: Iterator[pa.RecordBatch],
    expected_names: list[str],
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    budget: _RemainingBudget,
) -> Iterator[pl.DataFrame]:
    """Convert, rename, re-filter/re-select, and `n_rows`-truncate one `table_function` generator's batches.

    Shared by the whole-scan and per-split paths so both get identical Design
    Principle 1 treatment.
    """
    for batch in gen:
        df = arrow_to_df(batch)
        # The resolved scan function's own column names are an implementation
        # detail that need not match the catalog's declared TableInfo.columns
        # names (observed live against vgi-fixture-worker's `data.numbers`:
        # the table declares column "value", the function it resolves to
        # emits "n"). `register_io_source(schema=...)` promised callers the
        # declared names, so every batch is renamed positionally to match
        # before with_columns/predicate (which reference the declared names)
        # are applied.
        if list(df.columns) != expected_names and len(df.columns) == len(expected_names):
            df = df.rename(dict(zip(df.columns, expected_names, strict=True)))
        if with_columns is not None:
            df = df.select(with_columns)
        if predicate is not None:
            df = df.filter(predicate)
        if budget.remaining is not None:
            if df.height >= budget.remaining:
                if df.height:
                    yield df.head(budget.remaining)
                budget.exhausted = True
                return
            budget.remaining -= df.height
        if df.height:
            yield df


def _tee_batches(gen: Iterator[pa.RecordBatch], sink: list[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
    """Pass batches through unchanged while also appending each to `sink`.

    Lets the result-cache path capture the raw worker output alongside driving
    the normal `_process_batches` local-refilter pipeline, without duplicating
    the RPC.
    """
    for batch in gen:
        sink.append(batch)
        yield batch


def _iter_splits_sequential(
    client: Client,
    *,
    function_name: str,
    schema_name: str,
    arguments: Arguments,
    projection_ids: list[int] | None,
    pushdown_filters: bytes | None,
) -> Iterator[tuple[Any, bytes | None, bytes | None]]:
    """Yield `(split, execution_id, init_opaque_data)` for every split of this scan.

    Drains `PlanResponse.next_cursors` pagination sequentially via a queue
    rather than a single resume pointer: a plan response may hand back *more
    than one* continuation cursor at once (parallel enumeration branches —
    see `Client.table_function_plan`'s docstring), and this consumer is
    single-threaded, so it just visits each branch in turn instead of fanning
    out. `execution_id`/`init_opaque_data` are carried per plan response (not
    assumed stable across pages) and threaded back into each of that page's
    splits' redemption, for a worker whose splits share cross-process state
    via `BoundStorage`.
    """
    queue: list[bytes | None] = [None]
    while queue:
        cursor = queue.pop(0)
        plan = client.table_function_plan(
            function_name=function_name,
            schema_name=schema_name,
            arguments=arguments,
            projection_ids=projection_ids,
            pushdown_filters=pushdown_filters,
            cursor=cursor,
        )
        for split in plan.splits:
            yield split, plan.execution_id, plan.init_opaque_data
        if plan.next_cursors:
            queue.extend(plan.next_cursors)


#: The callback shape `polars.io.plugins.register_io_source` expects.
IoSource = Callable[[list[str] | None, "pl.Expr | None", "int | None", "int | None"], Iterator[pl.DataFrame]]


def make_io_source(table: _ScanSource, arrow_schema: pa.Schema) -> IoSource:
    column_names = list(arrow_schema.names)

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        _check_required_filters(table.required_filters(), predicate, table.schema_name, table.name)

        # NOT table._catalog.client — this generator instance can run
        # concurrently with another instance of the *same* scan (self-join,
        # concat, collect_all), confirmed live, so exchange-mode calls need a
        # per-thread connection. See catalog.py's "Thread safety" docstring.
        client = table._catalog._exchange_client()
        scan_fn = table._scan_function_get()
        function_info = table._function_info_get()
        function_name = scan_fn.function_name
        schema_name = table.scan_function_schema()

        # Projection pushdown: only sent when the worker function declares
        # support for it (an optimization; `.select(with_columns)` below is
        # applied unconditionally regardless).
        projection_ids: list[int] | None = None
        if with_columns is not None and function_info is not None and function_info.projection_pushdown:
            try:
                projection_ids = [column_names.index(c) for c in with_columns]
            except ValueError:
                projection_ids = None

        # Filter pushdown: only attempted when the worker function declares
        # support, AND only when projection isn't *also* being pushed down for
        # this call — VGI's filter `column_index` is relative to the scan's
        # (possibly already-projected) output schema, and reconciling that
        # against a pushed projection is real complexity deferred to Tier 2
        # (see the design plan). Skipping it here costs a pushdown
        # opportunity, never correctness: the local `.filter()` below still
        # runs unconditionally.
        pushdown_filters: bytes | None = None
        if (
            predicate is not None
            and function_info is not None
            and function_info.filter_pushdown
            and projection_ids is None
        ):
            pushdown_filters = translate_predicate(predicate, column_names)

        arguments = Arguments(
            positional=tuple(scan_fn.positional_arguments),
            named=dict(scan_fn.named_arguments) if scan_fn.named_arguments else None,
        )
        expected_names = [column_names[i] for i in projection_ids] if projection_ids is not None else column_names
        budget = _RemainingBudget(n_rows)

        try:
            # `hasattr` guards against an installed vgi-python that predates
            # `Client.table_function_plan`/`table_function(split_tokens=...)`
            # (both new — see CLAUDE.md's "Splits" section) — an older pinned
            # release (e.g. this repo's own CI, pinned to a released tag,
            # until it's bumped to one that includes these methods) falls
            # back to the ordinary whole-scan path instead of an
            # `AttributeError`. Never a correctness issue either way.
            #
            # `table.at_unit is None` also gates the split path off:
            # `TableFunctionPlanRequest` carries no `at_unit`/`at_value` field
            # at all, so planning has no way to honor a requested AT clause —
            # taking the split path anyway would silently serve the live
            # scan instead of the requested version. The plain path below
            # threads `at_unit`/`at_value` correctly, so a time-travel scan
            # of a split-capable table just loses the split decomposition,
            # never correctness.
            if (
                function_info is not None
                and function_info.supports_splits
                and table.at_unit is None
                and hasattr(client, "table_function_plan")
            ):
                # See module docstring's "Splits" section: sequential, not
                # parallel — Polars gives this generator no mechanism to
                # exploit split-level concurrency, but redeeming splits one at
                # a time in order is still sound, replayable, and exercises
                # the decomposition the worker actually tuned for.
                for split, split_execution_id, split_init_opaque_data in _iter_splits_sequential(
                    client,
                    function_name=function_name,
                    schema_name=schema_name,
                    arguments=arguments,
                    projection_ids=projection_ids,
                    pushdown_filters=pushdown_filters,
                ):
                    gen = client.table_function(
                        function_name=function_name,
                        schema_name=schema_name,
                        arguments=arguments,
                        projection_ids=projection_ids,
                        pushdown_filters=pushdown_filters,
                        split_tokens=[split.token],
                        split_execution_id=split_execution_id,
                        split_init_opaque_data=split_init_opaque_data,
                    )
                    try:
                        yield from _process_batches(gen, expected_names, with_columns, predicate, budget)
                    finally:
                        gen.close()
                    if budget.exhausted:
                        return
            else:
                # Result cache: only attempted for a whole, untruncated scan
                # (`n_rows is None`) — a LIMIT-truncated call never drains
                # its generator to EOS, so the raw batches captured below
                # would be a partial result; caching that under the
                # full-scan key would silently under-serve a later
                # untruncated repeat. See `_result_cache.py`'s module
                # docstring for the rest of this minimal slice's scope.
                cache_key = None
                if _SUPPORTS_RESULT_CACHE and n_rows is None:
                    canon_args = _canonical_arguments(arguments)
                    if canon_args is not None:
                        cache_key = (
                            table._catalog.attach_opaque_data,
                            function_name,
                            schema_name,
                            canon_args,
                            tuple(projection_ids) if projection_ids is not None else None,
                            pushdown_filters,
                            table.at_unit,
                            table.at_value,
                        )

                cached_batches = get_default_cache().get(cache_key) if cache_key is not None else None
                if cached_batches is not None:
                    yield from _process_batches(iter(cached_batches), expected_names, with_columns, predicate, budget)
                    return

                captured_ttl: list[float] = []

                def _on_batch_metadata(metadata: Any, _captured: list[float] = captured_ttl) -> None:
                    if not _captured:
                        ttl = parse_cache_control(metadata)
                        if ttl is not None:
                            _captured.append(ttl)

                extra_kwargs: dict[str, Any] = {}
                if cache_key is not None:
                    extra_kwargs["batch_metadata_callback"] = _on_batch_metadata
                if _SUPPORTS_TABLE_FUNCTION_AT_CLAUSE:
                    extra_kwargs["at_unit"] = table.at_unit
                    extra_kwargs["at_value"] = table.at_value
                elif table.at_unit is not None or table.at_value is not None:
                    raise VgiPolarsError(
                        "time travel requires a newer vgi-python (Client.table_function predates "
                        "at_unit/at_value on the installed version) — see CLAUDE.md's Time travel section"
                    )

                gen = client.table_function(
                    function_name=function_name,
                    schema_name=schema_name,
                    arguments=arguments,
                    projection_ids=projection_ids,
                    pushdown_filters=pushdown_filters,
                    **extra_kwargs,
                )
                raw_batches: list[pa.RecordBatch] = []
                source = _tee_batches(gen, raw_batches) if cache_key is not None else gen
                try:
                    yield from _process_batches(source, expected_names, with_columns, predicate, budget)
                    # Reached only on a full drain to EOS (never-partial —
                    # an early `return` inside _process_batches for a
                    # truncated scan, or an exception, skips this commit).
                    if cache_key is not None and captured_ttl:
                        get_default_cache().put(cache_key, raw_batches, captured_ttl[0])
                finally:
                    gen.close()
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e

    return io_source
