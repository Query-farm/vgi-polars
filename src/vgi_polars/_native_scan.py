# Copyright 2026 Query Farm LLC - https://query.farm

"""Native scan-function delegation -- `Client.table_function()` bypassed entirely.

`ScanFunctionResult.function_name` can name a reader the CALLING engine
should run itself, not a VGI-hosted function. Mirrors the DuckDB C++
extension's own resolution order
(`~/Development/vgi/src/storage/vgi_table_entry.cpp`'s `GetScanFunctionImpl`):
`table_scan_function_get`'s `function_name` is looked up in the calling
engine's OWN function catalog *before* ever being treated as a VGI RPC
target. `read_parquet`/`iceberg_scan`/etc. are DuckDB built-ins a worker can
delegate to so the actual read runs entirely client-side, with the client's
own native reader doing row-group pruning, cloud range reads, and so on — no
worker round-trip for the data at all. `ScanFunctionResult`'s own docstring
(`vgi/catalog/catalog_interface.py`) documents this as a first-class,
general wire concept, not specific to any one worker: "This enables catalogs
to delegate scanning to any DuckDB function (e.g., `read_parquet`,
`iceberg_scan`, or a custom VGI table function) with appropriate arguments."

This module is the Polars-native equivalent of that resolution step:
`read_parquet` -> `pl.scan_parquet`, using `ScanFunctionResult`'s
positional/named arguments (already decoded to `pa.Scalar` by vgi-python's
`Client`) to build the call.

**Confirmed live, not hypothetical.** Attaching vgi-polars to a real
`read_parquet`-delegating worker (`vgi-overture-maps-typescript`,
`https://vgi-overture.rusty-bb6.workers.dev`) and calling
`Client.table_function(function_name="read_parquet", ...)` against it — the
*old* behavior, before this module existed — raised
`FunctionNotFoundError: Unknown function 'read_parquet'` from the worker.
`FunctionRegistry` there is deliberately empty; native delegation's entire
point is that the worker never implements a matching function at all, so
there was nothing this package could have reached any other way.

**Required extensions and out-of-band settings are NOT expressible here.**
`ScanFunctionResult.required_extensions` (`["parquet", "httpfs"]` for
`read_parquet`) are DuckDB `LOAD` statements with no Polars equivalent —
silently ignored (Polars needs no such install step; parquet + cloud reads
are built in). S3 credentials/region are genuinely out-of-band on the wire
too — the DuckDB reference worker's own README requires a separate
`SET s3_region=...` the worker never conveys either, since VGI's wire
protocol has no field for it. `scan(storage_options=...)` is a plain
passthrough to `pl.scan_parquet`, not something this module can infer from
the `ScanFunctionResult` alone.

**Required-filters cost-safety is NOT enforced here — by necessity, not
oversight.** The io_source path (`_source.py`'s `_check_required_filters`)
can enforce this because Polars calls its generator back with the *resolved*
predicate at collect time. A natively-delegated scan instead returns a bare
`pl.LazyFrame` immediately from `VgiTable.scan()` — there's no callback, no
hook, nothing to inspect before `.collect()` runs. Rather than silently skip
the safety check the way a bug would, `table.py`'s `scan()` refuses outright
(raises `VgiPolarsError`) whenever a natively-delegated table declares
`required_filters`, unless the caller explicitly passes
`acknowledge_required_filters=True` — forcing a conscious "yes, I'm
responsible for my own filters here" rather than an accidental full-bucket
read silently going out over the wire.

**Only `read_parquet` today.** Extending to other native delegation targets
(`read_csv` -> `pl.scan_csv`, `iceberg_scan` -> a Polars Iceberg reader if
one's configured, `delta_scan` -> `polars-deltalake`, ...) is a per-function
mapping to add here as a real need arises — deliberately not solved
generically; there's no cross-engine standard for "translate any function
call by name," and guessing wrong would silently misread data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from vgi_polars.errors import VgiPolarsError

if TYPE_CHECKING:
    from vgi.catalog.catalog_interface import ScanFunctionResult

#: Named arguments `_scan_parquet_native` knows how to translate onto
#: `pl.scan_parquet`'s own keyword parameters. Anything else raises rather
#: than silently dropping a knob the worker asked for.
_KNOWN_READ_PARQUET_NAMED_ARGS = {"hive_partitioning"}


def _scan_parquet_native(
    scan_fn: ScanFunctionResult,
    *,
    schema_name: str,
    table_name: str,
    storage_options: dict[str, str] | None,
) -> pl.LazyFrame:
    """Build a native `pl.scan_parquet(...)` from a `read_parquet`-delegating `ScanFunctionResult`.

    Args:
        scan_fn: The worker's `table_scan_function_get` response, with
            `function_name == "read_parquet"`.
        schema_name: The table's schema — only used to name it in errors.
        table_name: The table's name — only used to name it in errors.
        storage_options: Passed straight through to `pl.scan_parquet` (e.g.
            `{"aws_region": "us-west-2", "skip_signature": "true"}` for a
            public, unauthenticated S3 bucket) — see module docstring's
            "out-of-band settings" section for why this can't be inferred.

    Returns:
        A `pl.LazyFrame` reading the delegated path directly — no VGI
        round-trip for the data.

    Raises:
        VgiPolarsError: If the `ScanFunctionResult` doesn't carry a usable
            path (arg 0, a string), or carries a named argument this module
            doesn't know how to translate.

    """
    if not scan_fn.positional_arguments:
        raise VgiPolarsError(
            f"{schema_name}.{table_name}: worker delegated to read_parquet with no positional "
            "arguments (expected the file/glob path as argument 0)"
        )
    path = scan_fn.positional_arguments[0].as_py()
    if not isinstance(path, str):
        raise VgiPolarsError(
            f"{schema_name}.{table_name}: read_parquet's first argument is a "
            f"{type(path).__name__}, expected a path/glob string"
        )

    kwargs: dict[str, Any] = {}
    unknown = sorted(set(scan_fn.named_arguments or {}) - _KNOWN_READ_PARQUET_NAMED_ARGS)
    if unknown:
        raise VgiPolarsError(
            f"{schema_name}.{table_name}: worker's read_parquet delegation passed named "
            f"argument(s) {unknown} vgi-polars doesn't know how to translate to pl.scan_parquet "
            f"(known: {sorted(_KNOWN_READ_PARQUET_NAMED_ARGS)})"
        )
    for name in _KNOWN_READ_PARQUET_NAMED_ARGS:
        value = (scan_fn.named_arguments or {}).get(name)
        if value is not None:
            kwargs[name] = value.as_py()

    return pl.scan_parquet(path, storage_options=storage_options, **kwargs)


#: `ScanFunctionResult.function_name` -> the Polars-native builder that
#: satisfies it directly, bypassing `register_io_source`/`Client.
#: table_function` entirely. See module docstring.
NATIVE_SCAN_HANDLERS = {
    "read_parquet": _scan_parquet_native,
}
