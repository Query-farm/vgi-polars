<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-polars/main/docs/vgi-logo.png?v=1" alt="VGI logo" width="260">
  &nbsp;&nbsp;+&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-polars/main/docs/polars-logo.svg?v=1" alt="Polars logo" width="180">
</p>

# vgi-polars

[![PyPI](https://img.shields.io/pypi/v/vgi-polars.svg)](https://pypi.org/project/vgi-polars/)
[![Python](https://img.shields.io/pypi/pyversions/vgi-polars.svg)](https://pypi.org/project/vgi-polars/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Query-farm/vgi-polars/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-polars/actions/workflows/ci.yml)

A [Polars](https://pola.rs) client for [VGI](https://github.com/Query-farm/vgi-python)
(Vector Gateway Interface). Lets a `polars.LazyFrame`/`polars.DataFrame` scan a VGI
catalog's tables and call its scalar, table-in-out, and aggregate functions — the same
role the [VGI DuckDB extension](https://github.com/Query-farm/vgi) plays for DuckDB,
but for Polars, with no DuckDB dependency at all.

This is not a new VGI protocol implementation. It's a thin adapter over
[vgi-python](https://github.com/Query-farm/vgi-python)'s existing pure-Python,
Arrow-native reference client (`vgi.client.Client`) — the same wire-protocol code the
DuckDB extension speaks — combined with Polars'
[`polars.io.plugins.register_io_source`](https://docs.pola.rs/api/python/stable/reference/api/polars.io.plugins.register_io_source.html)
extension point, which was purpose-built for exactly this "external source with
pushdown" shape.

## Installation

```bash
pip install vgi-polars
```

HTTP-transport support (talking to a VGI worker over `http://`/`https://`, and the
Orchard remote-secret-provider path) needs an extra:

```bash
pip install "vgi-polars[http]"
```

Subprocess and TCP transports need no extra; the `launch:` transport (a shared,
launcher-managed worker over AF_UNIX) needs `vgi-polars[launch]`.

**Requirements:** Python 3.13+, [Polars](https://pola.rs) 2.0 (1.41.1 and later 1.x
releases remain supported and tested), and vgi-python 0.42.1+, which speaks the
current VGI protocol — a worker must speak the same protocol version.

On Polars 2.0 an error raised while a scan runs (a worker error, a
`required_filters` violation) reaches your `.collect()` call as the original
`vgi_polars.VgiPolarsError`; Polars 1.x wraps it in `polars.exceptions.ComputeError`
with the same message.

## Quick start

```python
import polars as pl
import vgi_polars as vp

with vp.attach("path/to/my-vgi-worker", name="my_catalog") as cat:
    print(cat.schemas())
    print(cat.tables("main"))

    t = cat.table("main", "events")
    print(t.schema)                # polars.Schema, no scan performed
    print(t.scan().filter(pl.col("value") > 90).collect())

    my_fn = cat.scalar_function("main", "my_function")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    print(df.with_columns(my_fn(pl.col("a"), pl.col("b")).alias("result")))
```

`attach()` auto-detects transport from the location string's scheme — a bare command
is a subprocess worker, `http://`/`https://` is HTTP, `tcp://host:port` is raw
Arrow-IPC framing over TCP, `launch:<command>` is a launcher-managed worker shared by
every process that names the same command:

```python
cat = vp.attach("http://localhost:8080", name="my_catalog")
```

Any VGI worker — written in Python, Rust, Go, Java, or TypeScript — speaks to
vgi-polars unchanged; VGI is a cross-language protocol, not a Python-specific one.
See [vgi-python](https://github.com/Query-farm/vgi-python) for reference worker
implementations and the protocol documentation.

## Live example: earthquakes

No local worker needed — this attaches over HTTPS to a live, public VGI worker
serving the [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/)'s
rolling 30-day feed as an ordinary table (a weather example follows below).
More live example workers at [query.farm/vgi](https://query.farm/vgi/).

```python
import polars as pl
import vgi_polars as vp

cat = vp.attach("https://vgi-earthquakes.rusty-bb6.workers.dev", name="earthquakes")
recent = cat.table("main", "recent")

print(
    recent.scan()
    .filter(pl.col("mag") >= 5)
    .sort("mag", descending=True)
    .head(8)
    .select("time", pl.col("mag").round(1), "place")
    .collect()
)
```

```text
shape: (8, 3)
┌─────────────────────────────┬─────┬─────────────────────────────────┐
│ time                        ┆ mag ┆ place                           │
│ ---                         ┆ --- ┆ ---                             │
│ datetime[μs, UTC]           ┆ f64 ┆ str                             │
╞═════════════════════════════╪═════╪═════════════════════════════════╡
│ 2026-08-14 21:58:21.564 UTC ┆ 7.7 ┆ 68 km NNW of Ende, Indonesia    │
│ 2026-08-10 12:34:28.125 UTC ┆ 7.4 ┆ 5 km S of San José del Palmar,… │
│ …                           ┆ …   ┆ …                               │
└─────────────────────────────┴─────┴─────────────────────────────────┘
```

The `.filter()`/`.sort()`/`.head()`/`.select()` chain runs entirely against the
`LazyFrame` `.scan()` returns — nothing is fetched until `.collect()`. If the worker
declares filter/projection pushdown support, `mag >= 5` and the column selection are
sent to it to reduce what crosses the wire; either way, the *complete* original
predicate and projection are always re-applied locally after scanning too (see
[Pushdown is an optimization, never a correctness delegation](#pushdown-is-an-optimization-never-a-correctness-delegation)
below), so the result is identical whether or not pushdown happened to work.

## Live example: weather

This worker exposes no plain catalog tables at all — every function
(`geocoding`, `forecast_hourly`, ...) is a *blended row-transform* function,
callable either as a bare literal call or joined against an existing
`LazyFrame`/`DataFrame` (DuckDB's `FROM t, LATERAL f(t.x)`, for Polars). Both
shapes chain together below: a literal `geocoding` call resolves a place name
to coordinates, then `forecast_hourly` is called *against that result*,
correlating each output row back to the place that produced it.

Try it live: [Colab notebook](https://colab.research.google.com/drive/1eGmuXmpaLQ4MfStPTWlDXabfX5aAZf8a?usp=sharing).

```python
import polars as pl
import vgi_polars as vp

cat = vp.attach("https://vgi-open-meteo.rusty-bb6.workers.dev", name="open_meteo")
geocoding = cat.row_transform_function("main", "geocoding")
forecast_hourly = cat.row_transform_function("main", "forecast_hourly")

# Literal call: fn(None, ...) -- no LazyFrame, just arguments.
place = geocoding(None, "Glen Allen, VA", count=1, country_code="US")

# Correlated call: fn(lf, pl.col(...), ...) -- joined against `place`'s
# result, one forecast per input row (here, just the one place).
forecast = forecast_hourly(
    place,
    pl.col("latitude"),
    pl.col("longitude"),
    forecast_days=1,
    temperature_unit="fahrenheit",
).collect()

print(forecast.select("name", "time", pl.col("temperature_2m").round(1).alias("temp_f"), "weather_code").head(6))
```

```text
shape: (6, 4)
┌────────────┬─────────────────────────┬────────┬──────────────┐
│ name       ┆ time                    ┆ temp_f ┆ weather_code │
│ ---        ┆ ---                     ┆ ---    ┆ ---          │
│ str        ┆ datetime[μs, UTC]       ┆ f64    ┆ i32          │
╞════════════╪═════════════════════════╪════════╪══════════════╡
│ Glen Allen ┆ 2026-08-27 00:00:00 UTC ┆ 76.4   ┆ 0            │
│ Glen Allen ┆ 2026-08-27 01:00:00 UTC ┆ 73.8   ┆ 1            │
│ Glen Allen ┆ 2026-08-27 02:00:00 UTC ┆ 73.1   ┆ 0            │
│ Glen Allen ┆ 2026-08-27 03:00:00 UTC ┆ 72.7   ┆ 0            │
│ Glen Allen ┆ 2026-08-27 04:00:00 UTC ┆ 71.4   ┆ 2            │
│ Glen Allen ┆ 2026-08-27 05:00:00 UTC ┆ 71.0   ┆ 0            │
└────────────┴─────────────────────────┴────────┴──────────────┘
```

`place` is itself a `LazyFrame` — `forecast_hourly` never sees raw coordinates,
it sees the result of another VGI call, and every one of `place`'s own columns
(`name`, `country`, `population`, ...) rides along onto each forecast row for
free. Swap in `forecast_daily`, `historical_hourly`, `marine_hourly`, or any
of the worker's other functions the same way — one registration serves the
literal, column, and correlated-join call shapes uniformly.

## Catalog metadata caching

Listing and lookup calls (`schemas()`, `tables()`, `functions()`, `function_info()`,
a table's schema, a function's metadata) are answered from one snapshot of the whole
catalog, loaded lazily on first use. When the worker advertises it, the snapshot is a
single `catalog_contents` request; otherwise it is built from the per-schema listing
requests. The snapshot is checked on every read, by the same rules the VGI DuckDB
extension uses:

- a catalog whose worker declares its metadata **frozen** is loaded once and never
  rechecked;
- a snapshot with an **etag** is revalidated with one conditional request (unchanged
  keeps it, changed replaces it);
- otherwise the catalog's **version** is polled, and a changed version reloads it;
- a worker that reports version 0 and no etag can't say whether anything changed, so
  after the first load each read asks the worker directly, per schema.

So listings never go staler than the worker's current state, and a frozen catalog
costs no requests after the first. A table or function missing from the snapshot is
still looked up directly, and time-travel (`at_unit`) lookups bypass the snapshot.
`cat.clear_cache()` drops it, for example after redeploying a frozen catalog's
worker.

## Pushdown is an optimization, never a correctness delegation

This is the single design principle vgi-polars won't compromise on, so it's worth
stating plainly: **a worker's pushdown support is never trusted for correctness, only
for performance.**

Polars' `register_io_source` extension point — the mechanism `.scan()` is built on —
does **not** re-verify a predicate or projection an `io_source` claims to have
applied. An `io_source` that silently ignores the `predicate` it's handed still gets
every row back, unfiltered, in the final `.collect()` result; Polars has no fallback
check. This was confirmed empirically, not assumed: a `register_io_source` callback
that received a filter and did nothing with it produced unfiltered results with no
warning or error anywhere in the pipeline.

VGI workers, meanwhile, are written and tested against the DuckDB extension, which
*always* re-verifies a pushed-down predicate against DuckDB's own query engine — so a
worker can declare `filter_pushdown`/`projection_pushdown` support and still apply
either only approximately (e.g. a worker that pushes an equality filter but silently
ignores a range filter it doesn't know how to translate) and never notice, because
DuckDB was always going to catch the difference downstream. Polars won't.

Two systems that each individually assume "the other side will catch what I miss"
add up to neither side catching anything. So vgi-polars breaks that: every scan
**always** applies the complete, original `with_columns` selection, `predicate`, and
row-limit truncation locally, after fetching, regardless of what was pushed down or
what the worker claims to have handled. A partial or entirely-failed pushdown
translation is therefore only ever a performance loss — sending more rows/columns
than strictly necessary — never a correctness one.

Filters go to the worker in VGI Filter Encoding v2, one **advisory** predicate per
translatable conjunct of the predicate's top-level `AND`: comparisons between a column
and a literal, `is_null()`/`is_not_null()`, and `is_in([...])`. Advisory is the
encoding's term for exactly this posture: the worker may use a predicate to skip data
but must keep every row that satisfies it, because the client filters again. Literals
keep their exact Polars type (a nanosecond timestamp stays nanoseconds), and a
comparison the worker could not bind (say, a string column against a number) is not
sent. To name columns the way the worker's scan function does, the first pushed-down
`collect()` of a `scan()` binds the function once to learn its output schema.

## Status

**Implemented:**

- Catalog attach/detach with versioning introspection
- Schema and table discovery, incl. per-column statistics
- Table scan (eager + lazy) with best-effort projection/filter pushdown, incl. `is_in`
- `required_filters` cost-safety enforcement
- Sequential split-scan redemption
- Transparent multi-branch-table scanning (`pl.concat` under the hood)
- A minimal in-memory/TTL result cache
- Time-travel scans (`AT` clauses)
- Scalar function calls with scoped secrets and per-chunk input dedup
- Streaming and buffered table-in-out functions
- Blended (row-transform) table functions (`cat.row_transform_function(schema, name)`) —
  a worker function called with caller-supplied literal or column arguments and no
  separate table input, both the correlated-join shape (DuckDB's `SELECT * FROM t,
  LATERAL geocode(t.place)` equivalent: `geocode(lf, pl.col("place"))`) and the bare
  literal-call shape (`geocode(None, 'some place')`)
- Aggregate functions
- Native scan-function delegation (`read_parquet` -> `pl.scan_parquet`, `read_csv` ->
  `pl.scan_csv`, `iceberg_scan` -> `pl.scan_iceberg`) — a worker that ships no data of
  its own and instead tells the caller to run a native reader itself (VGI's
  `ScanFunctionResult` mechanism; see e.g.
  [vgi-overture-maps](https://github.com/Query-farm/vgi-overture-maps-typescript),
  a pure-metadata Overture Maps catalog). Real Polars-native pushdown (row-group
  pruning, cloud range reads), not anything vgi-polars hand-rolls
- Catalog metadata snapshot with revalidation (see
  [Catalog metadata caching](#catalog-metadata-caching))
- Subprocess, HTTP, TCP, and `launch:` (shared AF_UNIX worker) transports

**Not implemented:**

- Writes
- Companion-catalog federation
- Per-table time-travel discovery
- The `container://`/`github://` transport schemes (a substantially larger effort — a
  from-scratch Python transport layer, not an extension of the existing scheme table)

## Development

```bash
git clone https://github.com/Query-farm/vgi-polars.git
git clone https://github.com/Query-farm/vgi-python.git   # test fixture workers only
(cd vgi-python && uv sync --extra http)
cd vgi-polars
uv sync
VGI_PYTHON=../vgi-python uv run pytest -v
```

`vgi-python` itself is an ordinary PyPI dependency (pinned in `uv.lock`). The
vgi-python checkout is only needed for the integration tests' worker fixtures
(`vgi-fixture-worker`, `vgi-fixture-http`, ...), which are deliberately not in the
published wheel. `VGI_PYTHON` (default `~/Development/vgi-python`) points the tests at
that checkout's `.venv`; check out a release tag that satisfies vgi-polars'
`vgi-python` floor so the fixtures speak the same protocol. `VGI_TEST_WORKER`
overrides the worker binary path directly.

To develop against an unreleased vgi-python, overlay your checkout without touching
the lock: `uv run --with-editable ../vgi-python pytest -v`.

`uv run mypy src/`, `uv run ruff check src/ tests/`, and `uv run ruff format --check
src/ tests/` mirror what CI runs; `tests/test_docstrings.py` runs `pydoclint` as part
of the normal `pytest` run rather than as a separate step.

## License

Apache License, Version 2.0 — see [LICENSE](LICENSE).
