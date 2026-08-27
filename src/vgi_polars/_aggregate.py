# Copyright 2026 Query Farm LLC - https://query.farm

"""Aggregate-function -> eager `pl.DataFrame` bridge.

`AggregateClientMixin.aggregate_function` already does the hard part —
client-side group-id hashing/allocation and the full bind -> update ->
finalize -> destructor drive loop — confirmed via its exact signature and a
passing vgi-python test this session:

    def aggregate_function(
        self, *, function_name: str, schema_name: str,
        input: Iterable[pa.RecordBatch] = (), group_by: Sequence[str] = (),
        arguments: Arguments | None = None, settings=None, secrets=None,
        input_schema: pa.Schema | None = None,
        finalize_chunk_size: int = 2048,
    ) -> pa.RecordBatch

It wants `Iterable[pa.RecordBatch]`, not a `pl.DataFrame`, and returns a
**single** `pa.RecordBatch` (group keys + result columns) — a genuinely
**eager** call, not a lazy/streaming operator. Polars has no plugin hook for a
custom groupby-aggregation the way it does for scalar `map_batches`, so
`cat.aggregate_function(schema, name)` returns a plain eager function, not a
Polars expression:

    cat.aggregate_function("main", "vgi_sum")(df, group_by=["cat"]) -> pl.DataFrame

Every non-`group_by` column of `df` becomes an aggregate **value** column, in
schema order — vgi-python's own convention (confirmed in `aggregate.py`'s
`_assign_group_ids`/`update` drive loop).

**Bind-time constant arguments** (e.g. `vgi_percentile(value, 0.5)`'s
`percentile`) use the *identical* `ConstParam` wire convention as scalar
functions — confirmed live (`main.vgi_percentile`'s `percentile` field carries
`metadata[b"vgi_const"] == b"true"`, same as `main.multiply`'s `factor`) — so
this reuses `_arguments.is_const_field`/`to_scalar` directly rather than
reimplementing the split. Unlike table-in-out (`_table_in_out.py`), there is
no separate positional-vs-named distinction observed for aggregates; const
args are passed positionally, in declared order, same as scalar functions.

No output-schema-resolution dance needed the way `_table_in_out.py` has to do
— `aggregate_function` returns the real batch directly, converted straight to
a `pl.DataFrame`; there's no lazy `map_batches(schema=...)` to pre-declare.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
from vgi.arguments import Arguments
from vgi.catalog.catalog_interface import SchemaObjectType

from vgi_polars._arguments import is_const_field, to_scalar
from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vgi_polars.catalog import VgiCatalog


def make_aggregate_function(catalog: VgiCatalog, schema_name: str, name: str):
    """Return a callable `fn(df: pl.DataFrame, *args, group_by=(), settings=None,
    secrets=None) -> pl.DataFrame` for the aggregate function `schema_name.name`.
    `*args` are the function's `ConstParam` arguments, in declared order (e.g.
    `vgi_percentile(df, 0.5, group_by=["cat"])`). Every non-`group_by` column
    of `df` becomes a value column, in schema order. The `FunctionInfo`
    (argument schema) is resolved on first use and cached."""
    cache: dict[str, object] = {}

    def _function_info():
        if "info" not in cache:
            try:
                # Catalog-metadata call — the shared client, not the per-thread
                # exchange one; see catalog.py's "Thread safety" docstring.
                infos = catalog.client.schema_contents(
                    attach_opaque_data=catalog.attach_opaque_data,
                    name=schema_name,
                    type=SchemaObjectType.AGGREGATE_FUNCTION,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
            info = next((i for i in infos if i.name == name), None)
            if info is None:
                raise VgiPolarsError(f"aggregate function not found: {schema_name}.{name}")
            cache["info"] = info
        return cache["info"]

    def call(
        df: pl.DataFrame,
        *args: Any,
        group_by: Sequence[str] = (),
        settings: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        info = _function_info()
        arg_schema = pa.ipc.read_schema(pa.py_buffer(info.arguments)) if info.arguments else pa.schema([])
        const_fields = [f for f in arg_schema if is_const_field(f)]

        if len(args) != len(const_fields):
            raise VgiPolarsError(
                f"{schema_name}.{name} expects {len(const_fields)} constant argument(s), got {len(args)}"
            )
        arguments = (
            Arguments(positional=tuple(to_scalar(v, f.type) for v, f in zip(args, const_fields, strict=True)))
            if args
            else None
        )

        input_batches = df.to_arrow().to_batches()
        try:
            # Not a map_batches callback — this bridge is eager, called
            # directly on the calling thread, never concurrently. Still uses
            # the per-thread exchange client for consistency/safety if a
            # future caller wraps it in something concurrent.
            out_batch = catalog._exchange_client().aggregate_function(
                function_name=name,
                schema_name=schema_name,
                input=iter(input_batches),
                group_by=list(group_by),
                arguments=arguments,
                settings=settings,
                secrets=secrets,
            )
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e

        return pl.from_arrow(pa.Table.from_batches([out_batch]))

    return call
