<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-polars/main/docs/vgi-logo.png?v=1" alt="VGI logo" width="260">
  &nbsp;&nbsp;+&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-polars/main/docs/polars-logo.svg?v=1" alt="Polars logo" width="130">
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

Subprocess and TCP transports need no extra.

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
Arrow-IPC framing over TCP:

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
rolling 30-day feed as an ordinary table. More live example workers (weather,
volcanoes, and others) at [query.farm/vgi](https://query.farm/vgi/).

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
│ …                            ┆ …   ┆ …                               │
└─────────────────────────────┴─────┴─────────────────────────────────┘
```

The `.filter()`/`.sort()`/`.head()`/`.select()` chain runs entirely against the
`LazyFrame` `.scan()` returns — nothing is fetched until `.collect()`. If the worker
declares filter/projection pushdown support, `mag >= 5` and the column selection are
sent to it to reduce what crosses the wire; either way, the *complete* original
predicate and projection are always re-applied locally after scanning too (see
[Pushdown is an optimization, never a correctness delegation](#pushdown-is-an-optimization-never-a-correctness-delegation)
below), so the result is identical whether or not pushdown happened to work.

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
- Aggregate functions
- Subprocess, HTTP, and TCP transports

**Not implemented:**

- Writes
- Companion-catalog federation
- Per-table time-travel discovery
- Blended (row-transform) table functions — a worker function called with caller-supplied
  literal or column arguments and no separate table input (DuckDB's `SELECT * FROM
  t, LATERAL geocode(t.place)`-style correlated join, or a bare `geocode('some place')`
  literal call). vgi-polars can only scan pre-registered catalog *tables*
  (`cat.table(schema, name)`) and drive *table-in-out* functions that transform an
  existing `LazyFrame`/`DataFrame` — there's no bridge yet for a table-producing
  function invoked directly with its own arguments.
- The `container://`/`github://` transport schemes (a substantially larger effort — a
  from-scratch Python transport layer, not an extension of the existing scheme table).
  `launch:`/`unix://` (a launcher-managed shared worker) is implemented in the
  underlying `vgi-python` client (`Client.from_launch`, v0.29.4+) but not yet wired
  into `attach()`'s scheme detection here.

## Development

```bash
git clone https://github.com/Query-farm/vgi-polars.git
git clone https://github.com/Query-farm/vgi-python.git   # sibling checkout — see below
cd vgi-polars
uv sync
uv run pytest -v
```

`vgi-python` is pulled from that local sibling checkout (`[tool.uv.sources]` in
`pyproject.toml`, path `../vgi-python`) rather than the published PyPI release,
because this repo tracks vgi-python's client-side surface as both projects develop
together — the same pattern `vgi-spark`'s `settings.gradle.kts` uses to
composite-build a sibling `vgi-java` checkout. Integration tests need a
`vgi-fixture-worker` binary from that checkout; `VGI_PYTHON` (default
`~/Development/vgi-python`) picks which one, and `VGI_TEST_WORKER` overrides the
binary path directly if you need something other than the default venv location.

`uv run mypy src/`, `uv run ruff check src/ tests/`, and `uv run ruff format --check
src/ tests/` mirror what CI runs; `tests/test_docstrings.py` runs `pydoclint` as part
of the normal `pytest` run rather than as a separate step.

## Learn more

[CLAUDE.md](CLAUDE.md) is this repo's deep-dive doc — full architecture, every
non-obvious behavior discovered while building this (with the live evidence behind
each one), and a more detailed, evidence-backed version of the Status section above.
Not required reading to use the package; it's there for contributors and for anyone
who wants the "why," not just the "what."

## License

Apache License, Version 2.0 — see [LICENSE](LICENSE).
