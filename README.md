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

## Design principle: pushdown is an optimization, never a correctness delegation

Polars does **not** re-verify a predicate/projection an `io_source` claims to have
applied — confirmed empirically (see CLAUDE.md). VGI worker implementations, meanwhile,
are written and tested against the DuckDB extension, which *does* always re-verify, so
a worker may declare `filter_pushdown`/`projection_pushdown` and still apply either only
approximately. Since neither side can be trusted alone, vgi-polars always applies the
complete, original filter/projection/limit locally after scanning, regardless of what
was pushed down. Pushdown here only ever affects performance, never correctness.

## Status

**Implemented:**

- Catalog attach/detach with versioning introspection
- Schema and table discovery, incl. per-column statistics
- Table scan (eager + lazy) with best-effort projection/filter pushdown
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
- `is_in` filter pushdown
- Companion-catalog federation
- Per-table time-travel discovery
- The `container://`/`github://`/`launch:` transport schemes (a substantially
  larger effort — a from-scratch Python transport layer, not an extension of the
  existing scheme table)

See [CLAUDE.md](CLAUDE.md) for the full architecture, every non-obvious behavior
discovered while building this, and an evidence-backed breakdown of every scoped-out
item above.

## Development

```bash
git clone https://github.com/Query-farm/vgi-polars.git
git clone https://github.com/Query-farm/vgi-python.git   # sibling checkout, see CLAUDE.md
cd vgi-polars
uv sync
uv run pytest -v
```

vgi-polars tracks vgi-python's client-side surface as both develop together — see
[CLAUDE.md](CLAUDE.md)'s "Build / Test" section for why a sibling `vgi-python`
checkout is used locally and in CI rather than the published package, and for the
full test-running/fixture-worker setup.

## License

Apache License, Version 2.0 — see [LICENSE](LICENSE).
