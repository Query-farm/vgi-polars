# Copyright 2026 Query Farm LLC - https://query.farm

"""Scalar-function -> `pl.Expr.map_batches` bridge.

A VGI scalar function call is an out-of-process RPC regardless of how it's
invoked, so this rides Polars' own `map_batches` escape hatch (a Python UDF,
not a compiled expression plugin) rather than pretending otherwise — see the
package README for the rationale.

Two argument kinds, both declared on `FunctionInfo.arguments` (a `pa.Schema`,
one field per declared parameter, in order): a **constant** parameter (a
`ConstParam` in vgi-python — its field carries `metadata[b"vgi_const"] ==
b"true"`) is bound once per call, not exchanged per row, so the caller passes
a plain Python value for it; every other field is a per-row **array**
parameter, exchanged through the batch. Confirmed empirically against
`vgi-fixture-worker`'s `main.multiply(value, factor)` — `value` is an array
param, `factor` is const (`multiply(price, 2)` in the worker's own docstring
example) — the two use *separate, densely-numbered* index spaces server-side
(`vgi/scalar_function.py`'s `column_index`/`const_index` counters), so the
`Arguments.positional` tuple built here holds only the const values, in their
own relative declared order — not padded to the full parameter count.

Multiple array arguments are combined via `pl.struct(*exprs)` (Polars' own
idiom for a multi-column `map_batches`), unnested back into per-argument
Arrow arrays cast to the function's declared types, and exchanged 1:1 through
`Client.scalar_function`.

**`ANY`-typed (polymorphic) arguments** are the one declared-type exception.
A worker function that accepts more than one concrete type for a parameter
(e.g. `vgi-ical`'s `is_valid_ical(input)`, documented as "a `VARCHAR` file
path, or a `BLOB` of bytes") advertises that argument's declared Arrow type
as `null` — a placeholder, not a real type to cast to (see
`_arguments.is_any_type_field`). Casting real data to `null` is nonsensical
and pyarrow correctly refuses it (`ArrowNotImplementedError: Unsupported
cast from ... to null`), which is what calling such a function used to raise
before this was handled — confirmed live against `vgi-ical`. `_resolve_array_field`
below sends the column's own natural Arrow type for exactly these fields,
skipping the cast, while every ordinarily-typed field still casts to its
declared type as before.

**Scoped secrets.** The returned callable takes an optional keyword-only
`secrets: dict[str, Any] | None` (`{name: value}`, value may be a nested dict
for a struct-typed secret), threaded straight through to
`Client.scalar_function`'s own `secrets` parameter — e.g.
`multiply(pl.col("value"), 2, secrets={"vgi_example": {"key": "..."}})`.
**Table scans cannot get this** — confirmed `Client.table_function` (and the
table-in-out/buffering methods) hardcode `secrets=None` internally with no
public parameter, even though the underlying `BindRequest` wire type supports
it end to end. That's a real vgi-python API gap, not a vgi-polars design
choice; don't try to route around it here.

**Per-chunk input dedup** (`dedup=True`, the default; the client-side
mirror of the DuckDB C++ extension's `vgi_exchange_input_dedup` setting — see
the extension's CLAUDE.md). Before shipping a chunk's array-argument rows to
the worker, `_apply` deduplicates them to their distinct value tuples (a
low-cardinality column of 2048 rows reaches the worker as its distinct
count), then scatters the worker's output back to every original row. Scoped
to **within one chunk/call only** — no cross-chunk/cross-query cache (that
would be the C++ side's separate `vgi_result_cache_per_value`, a genuinely
stateful cache with its own TTL/eviction concerns, deliberately out of scope
here; see CLAUDE.md's Scope section). Gated on `FunctionInfo.stability !=
VOLATILE` (a `FunctionStability.VOLATILE` function — e.g. one wrapping
`random()` — must never be deduped, since identical inputs are not
guaranteed to produce identical outputs even within one call); `CONSISTENT`
and `CONSISTENT_WITHIN_QUERY` are both safe here precisely because the dedup
window never crosses a call boundary. Falls back to shipping the whole batch
unmodified — never incorrectly — whenever a row's values aren't hashable
(e.g. a struct- or list-typed array argument decodes to an unhashable
`dict`/`list` via `pl.DataFrame.rows()`) or when dedup wouldn't reduce the
row count at all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
from vgi.arguments import Arguments
from vgi.catalog.catalog_interface import FunctionInfo, FunctionStability, SchemaObjectType

from vgi_polars._arguments import is_any_type_field, is_const_field, to_scalar
from vgi_polars._polars_compat import arrow_to_df, arrow_to_series
from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from vgi_polars.catalog import VgiCatalog

#: The callable `VgiCatalog.scalar_function` returns — see `make_scalar_function`.
ScalarFunctionCall = Callable[..., pl.Expr]


def _resolve_array_field(array: pa.Array[Any], declared_field: pa.Field[Any]) -> tuple[pa.Array[Any], pa.Field[Any]]:
    """Cast `array` to `declared_field`'s type -- unless the field is `ANY`-typed.

    An `ANY`-typed field's *declared* type is always `pa.null()`, a placeholder
    (see `_arguments.is_any_type_field`) -- there is no real type to cast to, so
    the array's own natural type is sent as-is, and the returned field carries
    that same resolved type instead of the placeholder.

    Args:
        array: The column's data, already converted to Arrow.
        declared_field: The function's declared schema field for this argument.

    Returns:
        `(array, field)` ready to feed into `pa.RecordBatch.from_arrays` --
        `array` cast to `field.type` when the declared type was concrete;
        `array` unchanged, `field` retyped to `array.type`, when it was `ANY`.

    """
    if is_any_type_field(declared_field):
        return array, declared_field.with_type(array.type)
    return array.cast(declared_field.type), declared_field


def _dedup_positions(batch: pa.RecordBatch) -> tuple[list[int], list[int]] | None:
    """Return `(distinct_positions, inverse)` if deduping `batch`'s rows is possible and worthwhile.

    Deduping is possible if every row's values are hashable, and worthwhile if there are
    fewer distinct rows than total rows. Returns `None` otherwise, in which case the caller
    ships the batch unmodified. See module docstring's "Per-chunk input dedup" section.
    """
    if batch.num_rows == 0:
        return None
    try:
        rows = arrow_to_df(batch).rows()
        seen: dict[tuple[Any, ...], int] = {}
        distinct_positions: list[int] = []
        inverse: list[int] = []
        for i, row in enumerate(rows):
            idx = seen.get(row)
            if idx is None:
                idx = len(distinct_positions)
                seen[row] = idx
                distinct_positions.append(i)
            inverse.append(idx)
    except TypeError:
        return None  # an unhashable cell (struct/list-typed argument) — skip dedup
    if len(distinct_positions) == len(rows):
        return None  # already all-distinct — dedup would only add overhead
    return distinct_positions, inverse


def make_scalar_function(catalog: VgiCatalog, schema_name: str, name: str) -> ScalarFunctionCall:
    """Return a callable for the scalar function `schema_name.name`.

    Pass a `pl.Expr` for each array parameter and a plain Python value for each constant
    parameter, in the function's declared argument order. The `FunctionInfo`
    (argument/output schema) is resolved on first use and cached.
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
                    type=SchemaObjectType.SCALAR_FUNCTION,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
            info = next((i for i in infos if i.name == name), None)
            if info is None:
                raise VgiPolarsError(f"scalar function not found: {schema_name}.{name}")
            cache["info"] = info
        return cache["info"]

    def call(*args: Any, secrets: dict[str, Any] | None = None, dedup: bool = True) -> pl.Expr:
        info = _function_info()
        arg_schema = pa.ipc.read_schema(pa.py_buffer(info.arguments))
        out_schema = pa.ipc.read_schema(pa.py_buffer(info.output_schema))
        if len(args) != len(arg_schema.names):
            raise VgiPolarsError(f"{schema_name}.{name} expects {len(arg_schema.names)} argument(s), got {len(args)}")

        const_fields = [(i, f) for i, f in enumerate(arg_schema) if is_const_field(f)]
        array_fields = [(i, f) for i, f in enumerate(arg_schema) if not is_const_field(f)]

        for i, f in const_fields:
            if isinstance(args[i], pl.Expr):
                raise VgiPolarsError(
                    f"{schema_name}.{name}: argument {i} ('{f.name}') is a constant parameter — "
                    "pass a plain Python value, not a pl.Expr"
                )
        if not array_fields:
            raise VgiPolarsError(
                f"{schema_name}.{name} has no non-constant (per-row) arguments — not supported by "
                "vgi-polars's map_batches bridge"
            )

        # Dense, const-only positional order — see module docstring.
        const_arguments = Arguments(positional=tuple(to_scalar(args[i], f.type) for i, f in const_fields))
        array_exprs = [args[i] if isinstance(args[i], pl.Expr) else pl.lit(args[i]) for i, _ in array_fields]
        array_schema = pa.schema([f for _, f in array_fields])

        out_field = out_schema.field(0)
        return_dtype = arrow_to_series(pa.array([], type=out_field.type)).dtype
        dedup_safe = dedup and info.stability != FunctionStability.VOLATILE

        def _apply(struct_series: pl.Series) -> pl.Series:
            cols = struct_series.struct.unnest()
            resolved = [
                _resolve_array_field(cols.to_series(i).to_arrow(), array_schema.field(i))
                for i in range(len(array_fields))
            ]
            batch = pa.RecordBatch.from_arrays(
                [array for array, _ in resolved], schema=pa.schema([field for _, field in resolved])
            )

            inverse: list[int] | None = None
            if dedup_safe:
                dedup_result = _dedup_positions(batch)
                if dedup_result is not None:
                    distinct_positions, inverse = dedup_result
                    batch = batch.take(pa.array(distinct_positions, type=pa.int64()))

            try:
                # `map_batches(streamable=True)` calls `_apply` concurrently
                # from multiple threads (confirmed live) — must use a
                # per-thread client, never one shared across calls.
                out_batches = list(
                    catalog._exchange_client().scalar_function(
                        function_name=name,
                        schema_name=schema_name,
                        input=iter([batch]),
                        arguments=const_arguments,
                        secrets=secrets,
                    )
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
            if not out_batches:
                return pl.Series(name=out_field.name, values=[], dtype=return_dtype)
            out_table = pa.Table.from_batches(out_batches)
            result = arrow_to_df(out_table)[out_table.column_names[0]]
            if inverse is not None:
                result = result[pl.Series(inverse)]
            return result

        return pl.struct(*array_exprs).map_batches(_apply, return_dtype=return_dtype)

    return call
