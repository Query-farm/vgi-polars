# Copyright 2026 Query Farm LLC - https://query.farm

"""The minimal in-memory result cache (`_result_cache.py`) — against the real
`cacheable_numbers`/`cache_no_store` fixtures (`vgi/_test_fixtures/table/
cache.py`), which needed a vgi-python fix first: `Client._table_function_
parallel` discarded `AnnotatedBatch.custom_metadata` unconditionally, so a
worker's `vgi.cache.*` cacheability advertisement never reached
`table_function()`'s public generator at all. See CLAUDE.md's "Table-function
result cache" section for the full scope (in-memory/TTL/producer-mode only —
deliberately not the C++ extension's much larger disk-tier/revalidation/
per-partition feature)."""

from __future__ import annotations

import inspect

import pyarrow as pa
import pytest
from vgi.client.client import Client

import vgi_polars as vp
from vgi_polars import _result_cache

# `Client.table_function(batch_metadata_callback=...)` is a new upstream
# addition (see this module's docstring). Skipped, not failed, against an
# older installed vgi-python — mirrors `test_splits.py`'s
# `requires_split_support` pattern. Applied to every test in this file: even
# the no_store/n_rows/flush ones would pass vacuously without the fix (no
# caching ever happening also satisfies "never cached"), which isn't the
# same as actually exercising the property each test names.
requires_result_cache = pytest.mark.skipif(
    "batch_metadata_callback" not in inspect.signature(Client.table_function).parameters,
    reason="installed vgi-python predates Client.table_function(batch_metadata_callback=...)",
)


@pytest.fixture(autouse=True)
def _flush_cache_before_and_after():
    """The cache is a process-global singleton (see module docstring's
    "identity scoping" note) — isolate each test from ones that ran before
    or will run after it in the same pytest process."""
    _result_cache.flush()
    yield
    _result_cache.flush()


class _CachedTable:
    """Duck-typed stand-in for `VgiTable`, resolving `cacheable_numbers`/
    `cache_no_store` directly as bare functions (schema `data`) — mirrors
    `test_errors.py`'s `_FakeTableForScanFunction` pattern; neither fixture
    is wrapped in a catalog `Table` entry."""

    def __init__(self, catalog: vp.VgiCatalog, function_name: str, n: int) -> None:
        self._catalog = catalog
        self.schema_name = "data"
        self.name = function_name
        self.at_unit = None
        self.at_value = None
        self._fn = function_name
        self._n = n

    def _scan_function_get(self):
        from vgi.catalog.catalog_interface import ScanFunctionResult

        return ScanFunctionResult(
            function_name=self._fn, positional_arguments=[], named_arguments={"n": pa.scalar(self._n)}
        )

    def _function_info_get(self):
        return None

    def scan_function_schema(self) -> str:
        return "data"

    def required_filters(self) -> list[list[str]]:
        return []


def _spy_table_function(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch the exchange client's `table_function` to count real calls."""
    exchange_client = catalog._exchange_client()
    real = exchange_client.table_function
    calls: list[int] = []

    def spying(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(exchange_client, "table_function", spying)
    return calls


@requires_result_cache
def test_cacheable_result_is_served_from_cache_on_repeat(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    from vgi_polars._source import make_io_source

    calls = _spy_table_function(catalog, monkeypatch)
    t = _CachedTable(catalog, "cacheable_numbers", n=5)
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))

    first = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    second = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))

    assert sum(df.height for df in first) == 5
    assert sum(df.height for df in second) == 5
    assert len(calls) == 1, f"expected the second scan to be served from cache, saw {len(calls)} worker calls"


@requires_result_cache
def test_local_refilter_still_applies_to_a_cache_hit(catalog: vp.VgiCatalog) -> None:
    """Design Principle 1 holds for cached results too — a different
    predicate on the second (cache-hit) call still gets the right rows."""
    import polars as pl

    from vgi_polars._source import make_io_source

    t = _CachedTable(catalog, "cacheable_numbers", n=10)
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))

    list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))  # populate
    filtered = list(io_source(with_columns=None, predicate=pl.col("n") >= 7, n_rows=None, batch_size=None))
    rows = sorted(v for df in filtered for v in df["n"].to_list())
    assert rows == [7, 8, 9]


@requires_result_cache
def test_no_store_is_never_cached(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    from vgi_polars._source import make_io_source

    calls = _spy_table_function(catalog, monkeypatch)
    t = _CachedTable(catalog, "cache_no_store", n=4)
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))

    list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))

    assert len(calls) == 2, "no_store must never be served from cache"


@requires_result_cache
def test_n_rows_truncated_scan_is_never_cached(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    """A LIMIT-truncated scan never drains its generator to EOS, so it must
    never commit a partial result to the cache under the full-scan key."""
    from vgi_polars._source import make_io_source

    calls = _spy_table_function(catalog, monkeypatch)
    t = _CachedTable(catalog, "cacheable_numbers", n=20)
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))

    truncated = list(io_source(with_columns=None, predicate=None, n_rows=3, batch_size=None))
    assert sum(df.height for df in truncated) == 3

    full = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    assert sum(df.height for df in full) == 20
    assert len(calls) == 2, "the truncated scan must not have populated the cache for the full scan"


@requires_result_cache
def test_flush_clears_all_entries(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    from vgi_polars._source import make_io_source

    calls = _spy_table_function(catalog, monkeypatch)
    t = _CachedTable(catalog, "cacheable_numbers", n=3)
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))

    list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    _result_cache.flush()
    list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))

    assert len(calls) == 2, "flush() must force the next scan to hit the worker again"
