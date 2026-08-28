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

**`read_parquet` is confirmed live against a real worker; `read_csv` and
`iceberg_scan` are not — built conservatively from DuckDB's/Polars' own
documented signatures, not verified against an actual delegating worker the
way `read_parquet` was.** `read_csv`'s argument map is deliberately empty:
DuckDB's `read_csv` has a large, commonly-used named-argument surface
(`delim`, `header`, `columns`, ...), and guessing which of those a real
worker sends — and which Polars kwarg each should become — without a live
example to verify against would risk exactly the silent-misread failure
mode this module's philosophy refuses to accept. `iceberg_scan` maps only
`snapshot_from_id` -> `pl.scan_iceberg`'s `snapshot_id` (a safe, unambiguous
1:1 correspondence; DuckDB's `iceberg_scan` also has
`snapshot_from_timestamp`/`version`/`allow_moved_paths`, none of which
`pl.scan_iceberg` has an equivalent for). Any named argument outside a
target's map raises `VgiPolarsError` rather than silently dropping it —
extend the map for a given target once a real worker confirms what it
actually sends, the same way `read_parquet`'s `hive_partitioning` entry was
added from vgi-python's own `rff_hive` fixture and confirmed again live
against Overture. Other native delegation targets (`delta_scan` ->
`polars-deltalake`, ...) aren't mapped at all yet — add here as a real need
arises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import polars as pl

from vgi_polars.errors import VgiPolarsError

if TYPE_CHECKING:
    from vgi.catalog.catalog_interface import ScanFunctionResult

NativeScanHandler = Callable[..., pl.LazyFrame]


def _make_native_scan_handler(
    *,
    duckdb_function_name: str,
    reader: Callable[..., pl.LazyFrame],
    reader_call_name: str,
    arg_name_map: dict[str, str],
) -> NativeScanHandler:
    """Build a `NATIVE_SCAN_HANDLERS` entry: wire-arg validation + translation, one native reader.

    Args:
        duckdb_function_name: The `ScanFunctionResult.function_name` this
            handler answers for (e.g. `"read_parquet"`) — used only to name
            it in error messages.
        reader: The Polars-native `pl.scan_*` function to call.
        reader_call_name: `reader`'s name, for error messages (e.g.
            `"pl.scan_parquet"`).
        arg_name_map: `{wire named-argument name: reader's keyword parameter
            name}` for every named argument this handler knows how to
            translate. A wire named argument outside this map raises rather
            than being silently dropped — see module docstring.

    Returns:
        A handler function matching `NATIVE_SCAN_HANDLERS`'s call
        signature (`scan_fn`, `schema_name`, `table_name`, `storage_options`).

    """
    known_wire_names = set(arg_name_map)

    def handler(
        scan_fn: ScanFunctionResult,
        *,
        schema_name: str,
        table_name: str,
        storage_options: dict[str, str] | None,
    ) -> pl.LazyFrame:
        if not scan_fn.positional_arguments:
            raise VgiPolarsError(
                f"{schema_name}.{table_name}: worker delegated to {duckdb_function_name} with no "
                "positional arguments (expected the file/glob path as argument 0)"
            )
        path = scan_fn.positional_arguments[0].as_py()
        if not isinstance(path, str):
            raise VgiPolarsError(
                f"{schema_name}.{table_name}: {duckdb_function_name}'s first argument is a "
                f"{type(path).__name__}, expected a path/glob string"
            )

        kwargs: dict[str, Any] = {}
        unknown = sorted(set(scan_fn.named_arguments or {}) - known_wire_names)
        if unknown:
            raise VgiPolarsError(
                f"{schema_name}.{table_name}: worker's {duckdb_function_name} delegation passed "
                f"named argument(s) {unknown} vgi-polars doesn't know how to translate to "
                f"{reader_call_name} (known: {sorted(known_wire_names)})"
            )
        for wire_name, reader_kwarg in arg_name_map.items():
            value = (scan_fn.named_arguments or {}).get(wire_name)
            if value is not None:
                kwargs[reader_kwarg] = value.as_py()

        return reader(path, storage_options=storage_options, **kwargs)

    return handler


#: Bound to a module-level name (not just an entry in `NATIVE_SCAN_HANDLERS`)
#: because `tests/test_native_scan.py` imports it directly for its pure-unit
#: test slice (no worker involved).
_scan_parquet_native = _make_native_scan_handler(
    duckdb_function_name="read_parquet",
    reader=pl.scan_parquet,
    reader_call_name="pl.scan_parquet",
    arg_name_map={"hive_partitioning": "hive_partitioning"},
)

_scan_csv_native = _make_native_scan_handler(
    duckdb_function_name="read_csv",
    reader=pl.scan_csv,
    reader_call_name="pl.scan_csv",
    arg_name_map={},
)

_scan_iceberg_native = _make_native_scan_handler(
    duckdb_function_name="iceberg_scan",
    reader=pl.scan_iceberg,
    reader_call_name="pl.scan_iceberg",
    arg_name_map={"snapshot_from_id": "snapshot_id"},
)

#: `ScanFunctionResult.function_name` -> the Polars-native builder that
#: satisfies it directly, bypassing `register_io_source`/`Client.
#: table_function` entirely. See module docstring.
NATIVE_SCAN_HANDLERS: dict[str, NativeScanHandler] = {
    "read_parquet": _scan_parquet_native,
    "read_csv": _scan_csv_native,
    "iceberg_scan": _scan_iceberg_native,
}
