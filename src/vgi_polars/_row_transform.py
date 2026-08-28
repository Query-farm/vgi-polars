# Copyright 2026 Query Farm LLC - https://query.farm

"""Blended row-transform (`RowTransformFunction`) -> `LazyFrame.map_batches` bridge.

A blended function's positional args ARE its per-row input columns (real typed
args, no `TABLE`-placeholder field) — one worker registration serves a literal
call (`f(52, 13)`), an uncorrelated column join (`FROM t, f(t.x, t.y)`), and a
correlated `LATERAL` join, discriminated on the wire by `FunctionInfo.
input_from_args` (`function_type` is the same `TABLE` value an ordinary
table-in-out function uses — see `_table_in_out.py`). vgi-polars only reaches
this through `Client.table_in_out_function` (exchange mode): the caller
synthesizes the input batch itself — an N-row batch built from `lf`'s columns
for the column/LATERAL shape ("slice 1"), or a 1-row batch of literals for
the bare-literal-call shape (`call(lf=None, ...)`, "slice 3", below).

**The literal-call shape needs no `parent_row` at all.** There is no outer
`lf` to stamp columns from — the result IS the worker's own output columns,
whatever cardinality it emits for the one synthesized input row (1->1, 1->N,
or 1->0). So `_make_literal_call` below never passes `parent_row_callback`
and never decodes provenance; it just concatenates whatever `out_batches`
comes back. This is a real simplification over the column path, not a
shortcut — there is nothing to gather, so there is nothing that can be
gathered wrong. Routed via `pl.defer`, not a hand-rolled
`register_io_source`: a literal call has no scan cost to optimize, no
splits, and a shape that's exactly `pl.defer`'s `Callable[[], DataFrame]`
contract (`pl.defer` is itself built on `register_io_source` as the
single-shot special case). Still goes through `catalog._exchange_client()`,
not `catalog.client` — `pl.defer` inherits `register_io_source`'s
concurrent-instances hazard even though its own docs don't mention it.

**Every call passes `has_finalize=False`.** A blended function never has a
FINALIZE stage (enforced at `resolve_metadata()` — a `RowTransformFunction`
overriding `finish()`/`finalize()` is a `TypeError` at registration).
`Client.table_in_out_function` used to send a FINALIZE-phase `init()`
unconditionally regardless; the Python fixture worker happens to no-op that
gracefully, but a real non-Python worker SDK was confirmed live to reject it
outright ("a blended row-transform function has no FINALIZE phase"),
breaking every call against that worker. `_has_finalize_kwarg` (module-level,
capability-guarded the same way `_SUPPORTS_PARENT_ROW_CALLBACK` is) is spread
into every `table_in_out_function()` call below so the FINALIZE `init()` is
never sent at all, not just tolerated when it no-ops.

**Row provenance is a correctness requirement, not an optimization.** Unlike
scan pushdown (`_source.py`'s Design Principle 1 — always locally
re-verified, so a wrong worker claim only ever costs performance), a blended
function's output cardinality can differ from its input's (1->N fan-out,
1->0 filter), and there is no local fallback that lets this bridge
independently recompute which output row came from which input row. A wrong
or malformed `vgi_rpc.parent_row` therefore corrupts data (wrong outer-column
values stamped onto a row) rather than merely under-optimizing, so every
index it produces is validated before use — mirroring the DuckDB C++
extension's own reference decoder (`vgi_lateral_batch_operator.cpp`'s
`DecodeParentRow`) and its client-side vgi-python counterpart
(`vgi.protocol._decode_parent_rows`, threaded through
`Client.table_in_out_function`'s `parent_row_callback`).

**Gather safety (Polars-specific, non-negotiable).** `pl.DataFrame.gather()`
treats a *negative* index as Python-style "from the end," not as an error —
confirmed live: `df.gather([-1])` silently returns the *last* row, no
exception (pyarrow's `.take()`, by contrast, rejects `-1` outright). A
corrupted or buggy parent-row index would therefore silently gather the
*wrong* row instead of raising, if `.gather()`'s own bounds checking were
relied on for the lower bound. `_validate_parent_row` below runs an explicit
`< 0` check on every composed outer-row index before `.gather()` is ever
called — never rely on `.gather()` to catch a negative index.

`DataFrame.gather()` needs Polars >= 1.41.1 (confirmed by bisecting PyPI
releases — 1.41.0 itself was yanked; 1.40.1 predates the method, 1.41.1 has
it), hence this package's `polars>=1.41.1` floor.

**Outer-column policy: gather every column of `lf`, never exclude columns
"consumed" as args.** This matches what a real `FROM t, f(t.x)` correlated
join does (`t.x` stays present in the result) and avoids a fragile
`pl.Expr.meta.root_names()`-based exclusion that would misbehave for a
derived arg expression (`f(t.x + 1)` would wrongly drop `t.x`, even though a
real join never touches it). A worker output column colliding with an outer
column name is still an error, though — `VgiPolarsError`, not a silent
override.

**Per-chunk dedup (`dedup=True`, default), group-and-replicate composition.**
Same `FunctionStability != VOLATILE` gate and `_dedup_positions` helper
`_scalar.py` uses (imported from there — an established cross-module
private-import convention in this codebase). The wrinkle here vs. `_scalar.py`:
a deduped input batch means a 1->N worker output row's `parent_row` index is
expressed in terms of the *deduped/shipped* batch, not the original chunk —
and, because the map is 1:N rather than `_scalar.py`'s always-1:1, recovering
the true result needs more than a one-hop index rewrite. A naive
`distinct_positions[parent_rows[i]]` composition would only ever recover the
FIRST original row for each distinct value, silently dropping every other
duplicate row's output — a real row-loss bug, not a missed optimization. The
correct composition groups each worker output row by which shipped row
produced it, then replicates that whole group once per ORIGINAL row that
collapsed to it, so every duplicate outer row still gets its own copy of the
shared worker output, gathered with its own outer columns.

**Same-name overloads are not resolved by arity here.** `_function_info()`
picks the first `FunctionInfo` whose name matches (`next((i for i in infos if
i.name == name), None)`) — a pre-existing limitation shared with
`_scalar.py`/`_table_in_out.py`, not new to this module. A worker registering
`geo_encode` as both a 2-arg and a 3-arg overload is only reachable here as
the first-registered one; picking the right overload for a given call's arg
count is unimplemented and out of scope for slice 1.

**Literal-call precedence rule.** `lf is not None` always takes the column
path, even when every arg passed is a plain literal (broadcast to every row
via `pl.lit`) — `lf`'s presence, not "are any args exprs," decides which path
runs. `lf is None` is the bare-literal-call shape: every positional arg must
then be a plain value, never a `pl.Expr` (there is no frame to evaluate one
against) — checked explicitly rather than left to fail confusingly inside
`pl.lit`/Arrow conversion.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import polars as pl
import pyarrow as pa
from vgi.arguments import Arguments
from vgi.catalog.catalog_interface import FunctionInfo, FunctionStability, FunctionType, SchemaObjectType

from vgi_polars._arguments import to_scalar
from vgi_polars._polars_compat import arrow_to_df
from vgi_polars._scalar import _dedup_positions
from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from vgi_polars.catalog import VgiCatalog

#: The callable `VgiCatalog.row_transform_function` returns — see `make_row_transform_function`.
RowTransformFunctionCall = Callable[..., pl.LazyFrame]

_VGI_ARG_KEY = b"vgi_arg"
_VGI_NAMED_VALUE = b"named"


class _HasFinalizeKwargs(TypedDict, total=False):
    """The `**_has_finalize_kwarg` spread's precise shape, for mypy's `**kwargs` unpacking check.

    A plain `dict[str, bool]` spread can't be checked against
    `table_in_out_function`'s heterogeneously-typed keyword parameters (mypy
    would have to assume the dict could supply any of them) — a `TypedDict`
    with `total=False` is what makes `**_has_finalize_kwarg` verifiable as
    "zero or one `has_finalize: bool`, nothing else."
    """

    has_finalize: bool


# `parent_row_callback` is a new vgi-python parameter (see this package's plan
# doc / vgi-python's protocol.py `_decode_parent_rows`) — probe for it once at
# import time rather than letting a raw TypeError surface from inside a
# map_batches callback on an older install. Mirrors `_source.py`'s
# `_SUPPORTS_RESULT_CACHE` capability-probe convention exactly.
try:
    from vgi.client.client import Client as _RuntimeClient

    _table_in_out_params = inspect.signature(_RuntimeClient.table_in_out_function).parameters
    _SUPPORTS_PARENT_ROW_CALLBACK = "parent_row_callback" in _table_in_out_params
    # `has_finalize=False` skips the FINALIZE init() entirely for a function
    # that never has one -- every blended function, by construction. Found
    # live against a real (non-Python) worker SDK: it rejects an unadvertised
    # FINALIZE init() outright rather than the Python fixture worker's
    # graceful no-op, so table_in_out_function() ALWAYS sending one broke
    # interop with that SDK. `_has_finalize_kwarg` is spread into every
    # table_in_out_function() call below via `**_has_finalize_kwarg` so an
    # older vgi-python (predating the parameter) degrades to the old
    # always-finalize behavior instead of a raw TypeError.
    _has_finalize_kwarg: _HasFinalizeKwargs = {"has_finalize": False} if "has_finalize" in _table_in_out_params else {}
except Exception:  # noqa: BLE001 - never let a capability probe break import
    _SUPPORTS_PARENT_ROW_CALLBACK = False
    _has_finalize_kwarg = {}


def _is_named_field(field: pa.Field[Any]) -> bool:
    md = field.metadata or {}
    return md.get(_VGI_ARG_KEY) == _VGI_NAMED_VALUE


def _validate_parent_row(indices: list[int], *, bound: int) -> None:
    """Range-check every composed outer-row index before it's ever gathered.

    Explicit `< 0` check first — see module docstring's "Gather safety"
    section for why this can't be left to `pl.DataFrame.gather()` itself.
    Defense in depth: `vgi.protocol._decode_parent_rows` already range-checks
    against the shipped (possibly deduped) batch on the vgi-python side, but
    this bridge composes a *further* index (`distinct_positions[parent_rows[i]]`)
    that vgi-python never sees, so it gets its own check against the true bound.
    """
    for idx in indices:
        if idx < 0 or idx >= bound:
            raise VgiPolarsError(f"parent_row-derived index {idx} out of range [0, {bound}) -- refusing to gather")


def _make_literal_call(
    *,
    catalog: VgiCatalog,
    schema_name: str,
    name: str,
    args: tuple[Any, ...],
    positional_fields: list[pa.Field[Any]],
    arguments: Arguments | None,
    settings: dict[str, Any] | None,
    worker_out_schema: pl.Schema,
) -> pl.LazyFrame:
    """The bare-literal-call shape (`fn(None, 52.0, 13.0)`) via `pl.defer`.

    No `parent_row` decoding here at all — see module docstring's "The
    literal-call shape needs no parent_row" section: there's no outer frame
    to gather from, so the result is simply whatever the worker emits for
    the one synthesized input row, concatenated as-is.
    """

    def literal_fn() -> pl.DataFrame:
        row = dict(zip((f.name for f in positional_fields), args, strict=True))
        input_batch = pa.RecordBatch.from_pylist([row], schema=pa.schema(positional_fields))
        try:
            # A fresh per-call exchange client, same rationale as the column
            # path's bridge_fn — pl.defer's function is the concurrent-
            # instances hazard register_io_source has, so nothing here may
            # be shared across calls.
            out_batches = list(
                catalog._exchange_client().table_in_out_function(
                    function_name=name,
                    schema_name=schema_name,
                    input=iter([input_batch]),
                    arguments=arguments,
                    settings=settings,
                    **_has_finalize_kwarg,
                )
            )
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        if not out_batches:
            return pl.DataFrame(schema=worker_out_schema)
        return arrow_to_df(pa.Table.from_batches(out_batches))

    return pl.defer(literal_fn, schema=worker_out_schema)


def make_row_transform_function(catalog: VgiCatalog, schema_name: str, name: str) -> RowTransformFunctionCall:
    """Return a callable for the blended row-transform function `schema_name.name`.

    The returned callable has signature `fn(lf: pl.LazyFrame | None = None,
    *args, settings=None, dedup=True, **named_args) -> pl.LazyFrame`. Each
    positional `arg` is either a `pl.Expr` (a column reference into `lf`,
    requires `lf`) or a plain literal (broadcast to every row when `lf` is
    given). `lf=None` is the bare-literal-call shape (`fn(None, 52.0,
    13.0)`) — every positional arg must then be a plain value, never a
    `pl.Expr`. The `FunctionInfo` is resolved on first use and cached.
    """
    cache: dict[str, FunctionInfo] = {}

    def _function_info() -> FunctionInfo:
        if "info" not in cache:
            try:
                # Catalog-metadata call — the shared client, not the per-thread
                # exchange one; see catalog.py's "Thread safety" docstring.
                infos = catalog.client.schema_contents(
                    attach_opaque_data=catalog.attach_opaque_data,
                    name=schema_name,
                    type=SchemaObjectType.TABLE_FUNCTION,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
            info = next((i for i in infos if i.name == name), None)
            if info is None:
                raise VgiPolarsError(f"row-transform function not found: {schema_name}.{name}")
            if info.function_type is not FunctionType.TABLE or not info.input_from_args:
                raise VgiPolarsError(
                    f"{schema_name}.{name} is not a blended row-transform function "
                    f"(function_type={info.function_type.value}, input_from_args={info.input_from_args}) "
                    "-- use table_in_out_function() instead"
                )
            cache["info"] = info
        return cache["info"]

    def call(
        lf: pl.LazyFrame | None = None,
        *args: Any,
        settings: dict[str, Any] | None = None,
        dedup: bool = True,
        **named_args: Any,
    ) -> pl.LazyFrame:
        if not _SUPPORTS_PARENT_ROW_CALLBACK:
            raise VgiPolarsError(
                f"{schema_name}.{name}: row-transform functions need a newer vgi-python "
                "(Client.table_in_out_function predates parent_row_callback on the installed version)"
            )

        info = _function_info()
        arg_schema = pa.ipc.read_schema(pa.py_buffer(info.arguments)) if info.arguments else pa.schema([])
        positional_fields = [f for f in arg_schema if not _is_named_field(f)]
        named_fields = {f.name: f for f in arg_schema if _is_named_field(f)}

        if len(args) != len(positional_fields):
            raise VgiPolarsError(
                f"{schema_name}.{name} expects {len(positional_fields)} positional argument(s), got {len(args)}"
            )
        unknown = sorted(set(named_args) - set(named_fields))
        if unknown:
            raise VgiPolarsError(f"{schema_name}.{name} has no named argument(s): {unknown}")
        for k, v in named_args.items():
            if isinstance(v, pl.Expr):
                raise VgiPolarsError(
                    f"{schema_name}.{name}: named argument '{k}' is a bind-time parameter — "
                    "pass a plain Python value, not a pl.Expr"
                )
        if lf is None:
            for i, (v, f) in enumerate(zip(args, positional_fields, strict=True)):
                if isinstance(v, pl.Expr):
                    raise VgiPolarsError(
                        f"{schema_name}.{name}: positional argument {i} ('{f.name}') is a pl.Expr but "
                        "no `lf` was given -- a literal call needs a plain Python value for every "
                        "positional argument"
                    )

        arguments = None
        if named_args:
            arguments = Arguments(named={k: to_scalar(v, named_fields[k].type) for k, v in named_args.items()})

        # Output schema is resolved via a real, eager probe bind — never
        # trusted from the static FunctionInfo.output_schema (same rationale
        # as _table_in_out.py). A blended function's declared arg schema is
        # fully self-described independent of any `lf`, so the probe batch
        # is built straight from `positional_fields`, not from `lf`'s schema
        # — this is what makes probe-bind call-shape-agnostic, backing both
        # the literal and column entry points below with one bind.
        probe_batch = pa.RecordBatch.from_pylist([], schema=pa.schema(positional_fields))
        bound: dict[str, pa.Schema] = {}
        try:
            list(
                catalog._exchange_client().table_in_out_function(
                    function_name=name,
                    schema_name=schema_name,
                    input=iter([probe_batch]),
                    arguments=arguments,
                    settings=settings,
                    bind_result_callback=lambda r: bound.__setitem__("schema", r.output_schema),
                    **_has_finalize_kwarg,
                )
            )
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        if "schema" not in bound:
            raise VgiPolarsError(f"{schema_name}.{name}: worker never returned a bind response")
        worker_out_schema = arrow_to_df(bound["schema"].empty_table()).schema

        if lf is None:
            return _make_literal_call(
                catalog=catalog,
                schema_name=schema_name,
                name=name,
                args=args,
                positional_fields=positional_fields,
                arguments=arguments,
                settings=settings,
                worker_out_schema=worker_out_schema,
            )

        arg_exprs = [
            (v if isinstance(v, pl.Expr) else pl.lit(v)).alias(f.name)
            for v, f in zip(args, positional_fields, strict=True)
        ]

        lf_schema = lf.collect_schema()
        collisions = sorted(set(worker_out_schema) & set(lf_schema))
        if collisions:
            raise VgiPolarsError(
                f"{schema_name}.{name}: worker output column(s) {collisions} collide with `lf`'s own "
                "column(s) of the same name"
            )
        combined_schema = pl.Schema({**dict(lf_schema.items()), **dict(worker_out_schema.items())})

        dedup_safe = dedup and info.stability != FunctionStability.VOLATILE

        def bridge_fn(df: pl.DataFrame) -> pl.DataFrame:
            # A map_batches chunk isn't guaranteed to be one single Arrow
            # batch internally (confirmed live: an unrechunked multi-source
            # pl.concat produced 2 batches for one chunk under
            # streamable=False) -- parent_row's "index into THE input batch"
            # contract depends on there being exactly one, so this is
            # asserted, not assumed.
            df = df.rechunk()
            input_rows = df.height

            # with_columns(), not select(): a select() of ONLY literal
            # expressions (e.g. every positional arg passed as a plain
            # literal, not a pl.Expr) collapses to a single row -- Polars
            # only broadcasts a scalar-length expression against an existing
            # frame's row count, which with_columns() provides and a bare
            # select() of literals-only does not. Confirmed live: this bug
            # silently dropped every row past the first for an all-literal
            # call.
            input_table = df.with_columns(arg_exprs).select([f.name for f in positional_fields]).to_arrow()
            casted = [
                input_table.column(i).cast(positional_fields[i].type).combine_chunks()
                for i in range(len(positional_fields))
            ]
            input_batches = pa.table(casted, names=[f.name for f in positional_fields]).to_batches()
            if not input_batches:
                input_batches = [pa.RecordBatch.from_pylist([], schema=pa.schema(positional_fields))]
            if len(input_batches) != 1:
                raise VgiPolarsError(
                    f"{schema_name}.{name}: input chunk produced {len(input_batches)} Arrow batches "
                    "after rechunk(), expected exactly 1 -- parent_row provenance indexes into a "
                    "single batch"
                )
            input_batch = input_batches[0]

            # inverse[i] = which shipped/distinct row original row i collapsed
            # to, in the SAME index space `parent_rows` (below) uses to refer
            # to shipped rows -- both are "position within the shipped batch".
            inverse: list[int] | None = None
            shipped_batch = input_batch
            if dedup_safe:
                dedup_result = _dedup_positions(input_batch)
                if dedup_result is not None:
                    distinct_positions, inverse = dedup_result
                    shipped_batch = input_batch.take(pa.array(distinct_positions, type=pa.int64()))

            parent_rows_by_batch: list[list[int]] = []
            try:
                # map_batches(streamable=True) calls bridge_fn concurrently
                # from multiple threads (confirmed live, even for a single
                # large LazyFrame) — every piece of per-call state above and
                # below must stay local to this call, never hoisted to the
                # enclosing closure, or concurrent calls interleave their
                # bookkeeping and silently misattribute outer-column values.
                out_batches = list(
                    catalog._exchange_client().table_in_out_function(
                        function_name=name,
                        schema_name=schema_name,
                        input=iter([shipped_batch]),
                        arguments=arguments,
                        settings=settings,
                        parent_row_callback=parent_rows_by_batch.append,
                        **_has_finalize_kwarg,
                    )
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e

            if not out_batches:
                return pl.DataFrame(schema=combined_schema)

            worker_output_table = pa.Table.from_batches(out_batches)
            parent_rows = [p for sub in parent_rows_by_batch for p in sub]

            if inverse is not None:
                # Dedup ran: `parent_rows[k]` indexes into the SHIPPED
                # (deduped) batch, not the original chunk. A naive one-hop
                # "distinct_positions[parent_rows[k]]" composition would only
                # ever recover the FIRST original row for each distinct
                # value, silently dropping every other duplicate row's
                # output entirely -- a real row-loss bug, not just a missed
                # optimization. The correct mapping groups each worker
                # output row by which shipped row produced it, then
                # replicates that whole group once per ORIGINAL row that
                # collapsed to it, so every duplicate gets its own copy of
                # the shared worker output, gathered with ITS OWN outer
                # columns (not the representative row's).
                groups: dict[int, list[int]] = {}
                for output_pos, shipped_idx in enumerate(parent_rows):
                    groups.setdefault(shipped_idx, []).append(output_pos)
                outer_indices: list[int] = []
                output_row_positions: list[int] = []
                for original_idx, shipped_idx in enumerate(inverse):
                    for output_pos in groups.get(shipped_idx, ()):
                        outer_indices.append(original_idx)
                        output_row_positions.append(output_pos)
                _validate_parent_row(outer_indices, bound=input_rows)
                worker_output_table = worker_output_table.take(pa.array(output_row_positions, type=pa.int64()))
            else:
                outer_indices = parent_rows
                _validate_parent_row(outer_indices, bound=input_rows)

            outer_df = df.gather(pl.Series(outer_indices, dtype=pl.Int64))
            worker_output_df = arrow_to_df(worker_output_table)
            return outer_df.hstack(worker_output_df)

        return lf.map_batches(bridge_fn, streamable=True, schema=combined_schema)

    return call
