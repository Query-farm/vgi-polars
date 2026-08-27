# Copyright 2026 Query Farm LLC - https://query.farm

"""Table-in-out (streaming/buffered map-a-table) -> `LazyFrame.map_batches` bridge.

Two **separate** vgi-python client methods for two **separate** worker base
classes, dispatched on `FunctionInfo.function_type` (both discoverable via the
same `schema_contents(type=TABLE_FUNCTION)` catalog listing, confirmed live):

- `FunctionType.TABLE` (`TableInOutGenerator`/`RowTransformFunction`, streaming
  INPUT/FINALIZE phases) — driven by `Client.table_in_out_function`, bridged via
  `LazyFrame.map_batches(fn, streamable=True)`. Confirmed (original planning
  session) that Polars really does chunk-batch and allow cardinality change
  under `streamable=True`.
- `FunctionType.TABLE_BUFFERING` (`TableBufferingFunction`, Sink+Source — the
  worker needs to see the *whole* input before producing output) — driven by
  `Client.table_buffering_function`, bridged via `LazyFrame.map_batches(fn,
  streamable=False)` (whole-input materialize — the correct match).

No pushdown concept here the way there is for scans (Design Principle 1 in
`_source.py`) — the worker sees the whole input either way, so there's nothing
to push down partially; the only correctness concern is the RPC drive loop and
thread safety.

**Bind-time arguments** (e.g. `accumulate('events', ttl=..., max_row_size=...)`).
Some table-in-out functions take extra arguments beyond the table input itself
— confirmed live against `main.accumulate`'s declared `FunctionInfo.arguments`
schema, which carries per-field wire metadata distinguishing three kinds:
  - `metadata[b"vgi_type"] == b"table"` — the table-input slot itself (`data`
    for `accumulate`). Never appears in `Arguments` — it's supplied via the
    exchange's `input` batches instead.
  - `metadata[b"vgi_arg"] == b"named"` — a named argument (`ttl`,
    `max_row_size`, `result` for `accumulate`).
  - anything else — a positional argument (`name` for `accumulate`, wire
    position 0; the table slot's own wire position doesn't consume a slot in
    `Arguments.positional`, exactly like a scalar function's dense
    const-only index space — see `_scalar.py`'s module docstring for the
    general pattern).
`call()`'s `*args`/`**named_args` map onto exactly these two non-table groups,
in their declared order/name.

`Client.table_in_out_function`'s `input` iterator must yield at least one batch
(raises `ClientError` otherwise) — `call()` synthesizes a zero-row batch
matching the input schema when Polars hands it an empty chunk, so a caller
never has to special-case that. `Client.table_buffering_function` explicitly
allows empty input (a buffered aggregation can still produce a result over
zero rows), so no such synthesis is needed there.

**Output schema is resolved via a real, eager probe bind — never trusted from
the static `FunctionInfo.output_schema`.** Confirmed live, two different ways,
that the static declaration can be wrong: it's an empty placeholder for a
function whose output mirrors its input (`echo`'s `on_bind` sets output_schema
= the actual bound input schema, unknowable until real data flows), and it's
non-empty but *incorrect* for `accumulate` (declares `{"x": Int64}`, silently
omitting the `_timestamp` column its real `bind()` response actually adds).
So `call()` always does one extra exchange call up front — a zero-row batch
shaped like `lf`'s own schema, via `bind_result_callback` — to learn the real,
argument-bound output schema before handing `map_batches` a `schema=`. This
mirrors what the DuckDB C++ extension itself does (always binds for real
before planning a query) rather than trusting a static catalog hint.

**No `secrets=` parameter** — confirmed `Client.table_in_out_function` and
`Client.table_buffering_function` both hardcode `secrets=None` internally with
no public parameter (unlike `Client.scalar_function` — see `_scalar.py`). A
real vgi-python API gap, not a vgi-polars design choice.

**`has_finalize` is forwarded from the catalog, never hardcoded.** Every
`table_in_out_function()` call here spreads `has_finalize_kwarg`, built from
`FunctionInfo.has_finalize` (the function's own declared value — a `TABLE`
function may or may not implement finalize; only the caller's catalog
metadata knows which). This isn't a blended-function-only concern —
`_row_transform.py` needs `has_finalize=False` unconditionally because a
blended function structurally never has one, but a plain `TableInOutGenerator`
with no finalize override (`echo`, `filter_by_setting`, `repeat_inputs`, ...)
has exactly the same shape and can hit the same interop hazard a strict
worker SDK exposes: sending an unadvertised FINALIZE-phase `init()`. The
Python fixture worker no-ops that gracefully either way, so this bridge's own
test suite can't distinguish "forwarded correctly" from "never mattered
locally" — the value is in not depending on every worker SDK tolerating a
request it never declared support for. `has_finalize_kwarg` is `{}` (the old,
unconditional-FINALIZE behavior) for a buffered function (no FINALIZE-phase
concept exists there at all) or on a vgi-python predating the parameter.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import polars as pl
import pyarrow as pa
from vgi.arguments import Arguments
from vgi.catalog.catalog_interface import FunctionInfo, FunctionType, SchemaObjectType

from vgi_polars._arguments import to_scalar
from vgi_polars._polars_compat import arrow_to_df
from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from vgi_polars.catalog import VgiCatalog

#: The callable `VgiCatalog.table_in_out_function` returns — see `make_table_in_out_function`.
TableInOutFunction = Callable[..., pl.LazyFrame]

_VGI_TYPE_KEY = b"vgi_type"
_VGI_TABLE_VALUE = b"table"
_VGI_ARG_KEY = b"vgi_arg"
_VGI_NAMED_VALUE = b"named"


class _HasFinalizeKwargs(TypedDict, total=False):
    """`**kwargs` spread shape for the `has_finalize` capability guard below — see `_row_transform.py`."""

    has_finalize: bool


# `has_finalize` (vgi-python >= 0.29.6) lets a caller that already knows a
# function's FunctionInfo.has_finalize skip the FINALIZE-phase init() when
# it's False, rather than sending one every worker SDK must tolerate. This
# was added for _row_transform.py's blended functions (which are ALWAYS
# has_finalize=False), but the same interop hazard applies to an ordinary
# no-finalize TableInOutGenerator (echo, filter_by_setting, repeat_inputs,
# ...) called against a worker SDK that rejects an unadvertised FINALIZE
# init() outright rather than no-op'ing it (confirmed live against the
# vgi-open-meteo TypeScript worker for the blended case) -- so this bridge
# forwards the catalog's own per-function has_finalize too, not just a
# hardcoded value. Capability-guarded the same way _SUPPORTS_RESULT_CACHE is.
try:
    from vgi.client.client import Client as _RuntimeClient

    _SUPPORTS_HAS_FINALIZE = "has_finalize" in inspect.signature(_RuntimeClient.table_in_out_function).parameters
except Exception:  # noqa: BLE001 - never let a capability probe break import
    _SUPPORTS_HAS_FINALIZE = False


def _is_table_field(field: pa.Field[Any]) -> bool:
    md = field.metadata or {}
    return md.get(_VGI_TYPE_KEY) == _VGI_TABLE_VALUE


def _is_named_field(field: pa.Field[Any]) -> bool:
    md = field.metadata or {}
    return md.get(_VGI_ARG_KEY) == _VGI_NAMED_VALUE


def make_table_in_out_function(catalog: VgiCatalog, schema_name: str, name: str) -> TableInOutFunction:
    """Return a callable for the table-in-out function `schema_name.name`.

    The returned callable has signature `fn(lf: pl.LazyFrame, *args,
    settings=None, **named_args) -> pl.LazyFrame`. `*args`/`**named_args` are
    the function's declared bind-time arguments beyond the table input itself
    (see module docstring). The `FunctionInfo` (argument/output schema +
    streaming-vs-buffering dispatch) is resolved on first use and cached.
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
                raise VgiPolarsError(f"table-in-out function not found: {schema_name}.{name}")
            if info.function_type not in (FunctionType.TABLE, FunctionType.TABLE_BUFFERING):
                # Defensive: schema_contents(type=TABLE_FUNCTION) only ever
                # lists TABLE/TABLE_BUFFERING functions today (confirmed —
                # a scalar function name just comes back "not found" above,
                # never reaches here), so this is currently unreachable.
                # Kept in case that catalog-RPC classification ever widens.
                raise VgiPolarsError(
                    f"{schema_name}.{name} is a {info.function_type.value} function, not a table-in-out one"
                )
            cache["info"] = info
        return cache["info"]

    def call(lf: pl.LazyFrame, *args: Any, settings: dict[str, Any] | None = None, **named_args: Any) -> pl.LazyFrame:
        info = _function_info()
        is_buffering = info.function_type == FunctionType.TABLE_BUFFERING
        # table_buffering_function has no has_finalize parameter at all (a
        # buffered function's Sink+Source shape has no FINALIZE-phase concept
        # to skip) -- only ever build/spread this for the streaming method.
        has_finalize_kwarg: _HasFinalizeKwargs = (
            {"has_finalize": info.has_finalize} if not is_buffering and _SUPPORTS_HAS_FINALIZE else {}
        )

        arg_schema = pa.ipc.read_schema(pa.py_buffer(info.arguments)) if info.arguments else pa.schema([])
        positional_fields = [f for f in arg_schema if not _is_table_field(f) and not _is_named_field(f)]
        named_fields = {f.name: f for f in arg_schema if not _is_table_field(f) and _is_named_field(f)}

        if len(args) != len(positional_fields):
            raise VgiPolarsError(
                f"{schema_name}.{name} expects {len(positional_fields)} positional argument(s), got {len(args)}"
            )
        unknown = sorted(set(named_args) - set(named_fields))
        if unknown:
            raise VgiPolarsError(f"{schema_name}.{name} has no named argument(s): {unknown}")

        arguments = None
        if args or named_args:
            arguments = Arguments(
                positional=tuple(to_scalar(v, f.type) for v, f in zip(args, positional_fields, strict=True)),
                named={k: to_scalar(v, named_fields[k].type) for k, v in named_args.items()} or None,
            )

        # FunctionInfo.output_schema (the STATIC catalog declaration) cannot
        # be trusted in general — confirmed live two different ways: it's an
        # empty placeholder for a function whose output mirrors its input
        # (`echo`'s `on_bind` sets output_schema = the actual bound input
        # schema, unknowable until real data flows), and it's non-empty but
        # WRONG for `accumulate` (declares `{"x": Int64}`, omits the
        # `_timestamp` column its real bind() response actually adds).
        # The only reliable source is a real bind — same as the DuckDB C++
        # extension itself, which always binds before it can plan a query.
        # So: eagerly probe-bind here (once, at call() time, not per batch)
        # with a zero-row input batch shaped like `lf`'s own schema, via
        # `bind_result_callback`, and use ITS output_schema.
        exchange_client = catalog._exchange_client()
        probe_batch = pa.RecordBatch.from_pylist([], schema=lf.collect_schema().to_arrow())
        bound: dict[str, pa.Schema] = {}
        try:
            # Explicit branches, not a polymorphic `method` variable: mypy
            # can't verify has_finalize_kwarg is always empty when calling
            # table_buffering_function (which has no such parameter at all)
            # from a shared call expression's static type alone -- only this
            # branch structure lets it see each call site's real signature.
            if is_buffering:
                list(
                    exchange_client.table_buffering_function(
                        function_name=name,
                        schema_name=schema_name,
                        input=iter([probe_batch]),
                        arguments=arguments,
                        settings=settings,
                        bind_result_callback=lambda r: bound.__setitem__("schema", r.output_schema),
                    )
                )
            else:
                list(
                    exchange_client.table_in_out_function(
                        function_name=name,
                        schema_name=schema_name,
                        input=iter([probe_batch]),
                        arguments=arguments,
                        settings=settings,
                        bind_result_callback=lambda r: bound.__setitem__("schema", r.output_schema),
                        **has_finalize_kwarg,
                    )
                )
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        if "schema" not in bound:
            raise VgiPolarsError(f"{schema_name}.{name}: worker never returned a bind response")
        pl_out_schema = arrow_to_df(bound["schema"].empty_table()).schema

        def bridge_fn(df: pl.DataFrame) -> pl.DataFrame:
            table = df.to_arrow()
            batches = table.to_batches()
            if not batches and not is_buffering:
                # table_in_out_function requires at least one input batch
                # (raises ClientError on an empty iterator); table_buffering_
                # function explicitly allows empty input, so only synthesize
                # here for the streaming case.
                batches = [pa.RecordBatch.from_pylist([], schema=table.schema)]

            try:
                # map_batches(streamable=True) calls bridge_fn concurrently
                # from multiple threads (confirmed live) — must use a
                # per-thread client, never one shared across calls.
                exchange_client = catalog._exchange_client()
                # Explicit branches -- see the probe-bind's identical comment
                # above for why a shared polymorphic `method` variable can't
                # be spread with has_finalize_kwarg and still type-check.
                if is_buffering:
                    out_batches = list(
                        exchange_client.table_buffering_function(
                            function_name=name,
                            schema_name=schema_name,
                            input=iter(batches),
                            arguments=arguments,
                            settings=settings,
                        )
                    )
                else:
                    out_batches = list(
                        exchange_client.table_in_out_function(
                            function_name=name,
                            schema_name=schema_name,
                            input=iter(batches),
                            arguments=arguments,
                            settings=settings,
                            **has_finalize_kwarg,
                        )
                    )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e

            if not out_batches:
                return pl.DataFrame(schema=pl_out_schema)
            out_table = pa.Table.from_batches(out_batches)
            return arrow_to_df(out_table)

        return lf.map_batches(bridge_fn, streamable=not is_buffering, schema=pl_out_schema)

    return call
