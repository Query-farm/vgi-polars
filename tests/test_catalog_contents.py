# Copyright 2026 Query Farm LLC - https://query.farm

"""`VgiCatalog` honours `supports_catalog_contents` through its catalog snapshot.

Runs vgi-polars against vgi-python's `contents_*` fixture catalogs (served by
`vgi-fixture-worker`) and asserts which catalog RPCs were actually sent: one
`catalog_contents` when advertised, the per-schema RPCs when it is not
advertised or fails, `if_none_match` revalidation for an etag-carrying catalog,
and the version-0 rule. Listings are checked against what the per-schema RPCs
return for the same catalog. See catalog.py's "Catalog snapshot" docstring.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pyarrow as pa
import pytest
from vgi.catalog.catalog_interface import SchemaObjectType
from vgi.client.client import Client

import vgi_polars as vp

# Names from vgi-python's vgi/_test_fixtures/catalog_contents.py.
CATALOG_PROBE = "contents_probe"
CATALOG_BROKEN = "contents_broken"
CATALOG_LEGACY = "contents_legacy"
CATALOG_MEMORY = "contents_memory"
CATALOG_REVAL = "contents_reval"

requires_load_catalog = pytest.mark.skipif(
    not hasattr(Client, "load_catalog"), reason="installed vgi-python predates Client.load_catalog"
)
pytestmark = requires_load_catalog

Call = tuple[str, str | None]


@pytest.fixture(params=["subprocess", "http"])
def location(request: pytest.FixtureRequest, worker_location: str) -> str:
    """`vgi-fixture-worker` over subprocess, and `vgi-fixture-http` over HTTP."""
    if request.param == "http":
        return str(request.getfixturevalue("http_worker_base_url"))
    return worker_location


_PER_SCHEMA_RPCS = {
    "catalog_schema_contents_tables",
    "catalog_schema_contents_views",
    "catalog_schema_contents_functions",
    "catalog_schema_contents_macros",
    "catalog_schema_contents_indexes",
}


class _RecordingProxy:
    """Wraps a VgiProtocol proxy, recording each `catalog_*` method sent (and its `if_none_match`)."""

    def __init__(self, proxy: Any, calls: list[Call], lock: threading.Lock) -> None:
        self._proxy = proxy
        self._calls = calls
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._proxy, name)
        if not name.startswith("catalog_") or not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self._calls.append((name, kwargs.get("if_none_match")))
            return attr(*args, **kwargs)

        return call


def _record(cat: vp.VgiCatalog) -> list[Call]:
    """Record every catalog RPC `cat`'s catalog-metadata client sends; returns the live list."""
    calls: list[Call] = []
    lock = threading.Lock()
    client = cat.client
    original = client._catalog_connect

    @contextmanager
    def recording() -> Iterator[Any]:
        with original() as proxy:
            yield _RecordingProxy(proxy, calls, lock)

    client._catalog_connect = recording  # type: ignore[method-assign]
    return calls


def _names(calls: list[Call]) -> list[str]:
    return [name for name, _ in calls]


def _listings(cat: vp.VgiCatalog) -> dict[str, Any]:
    """Every listing vgi-polars offers, for every schema."""
    out: dict[str, Any] = {"schemas": cat.schemas()}
    for schema in out["schemas"]:
        out[schema] = {
            "tables": cat.tables(schema),
            "views": cat.views(schema),
            "macros": cat.macros(schema),
            "functions": cat.functions(schema),
        }
    return out


def _per_schema_listings(cat: vp.VgiCatalog) -> dict[str, Any]:
    """The same listings, straight from `catalog_schemas` + the per-schema RPCs (no snapshot)."""
    client, attach = cat.client, cat.attach_opaque_data

    def names(schema: str, *types: SchemaObjectType) -> list[str]:
        return [i.name for t in types for i in client.schema_contents(attach_opaque_data=attach, path=[schema], type=t)]

    out: dict[str, Any] = {"schemas": [".".join(s.path) for s in client.schemas(attach_opaque_data=attach)]}
    for schema in out["schemas"]:
        out[schema] = {
            "tables": names(schema, SchemaObjectType.TABLE),
            "views": names(schema, SchemaObjectType.VIEW),
            "macros": names(schema, SchemaObjectType.SCALAR_MACRO, SchemaObjectType.TABLE_MACRO),
            "functions": names(
                schema,
                SchemaObjectType.SCALAR_FUNCTION,
                SchemaObjectType.TABLE_FUNCTION,
                SchemaObjectType.AGGREGATE_FUNCTION,
            ),
        }
    return out


def _expected_static(listings: dict[str, Any]) -> None:
    """The static two-schema catalog every `contents_*` static fixture serves."""
    assert listings["schemas"] == ["main", "extra"]
    main = listings["main"]
    assert main["tables"] == ["ten"]
    assert main["views"] == ["answer"]
    assert main["macros"] == ["contents_triple", "contents_range"]
    assert main["functions"] == ["double", "sequence", "vgi_sum"]  # scalar, table, aggregate
    assert listings["extra"]["tables"] == ["five"]


def test_probe_uses_one_catalog_contents_call(location: str) -> None:
    """Advertised + version-frozen: one `catalog_contents` answers every listing and lookup, forever."""
    with vp.attach(location, name=CATALOG_PROBE) as cat:
        assert cat._attach_result.supports_catalog_contents
        assert cat.catalog_version_frozen
        calls = _record(cat)

        listings = _listings(cat)
        assert calls == [("catalog_contents", None)]
        _expected_static(listings)

        # Lookups the bridges/table handles do come from the same snapshot:
        # no catalog_table_get, no per-schema function listing.
        assert cat.function_info("main", "vgi_sum").name == "vgi_sum"
        assert cat.table("main", "ten").schema.names() == ["n"]
        assert "catalog_table_get" not in _names(calls)
        assert not _PER_SCHEMA_RPCS & set(_names(calls))
        assert _names(calls).count("catalog_contents") == 1

        calls.clear()
        assert _listings(cat) == listings
        assert calls == []  # frozen: never revalidated

        calls.clear()
        assert _per_schema_listings(cat) == listings


def test_legacy_never_sends_catalog_contents(location: str) -> None:
    """Not advertised: `catalog_schemas` + per-schema RPCs, and never a `catalog_contents`."""
    with vp.attach(location, name=CATALOG_LEGACY) as cat:
        assert not cat._attach_result.supports_catalog_contents
        calls = _record(cat)

        listings = _listings(cat)
        names = _names(calls)
        assert "catalog_contents" not in names
        assert names[0] == "catalog_schemas"
        assert names.count("catalog_schemas") == 1
        assert set(names[1:]) <= _PER_SCHEMA_RPCS
        _expected_static(listings)

        calls.clear()
        cat.clear_cache()
        assert _listings(cat) == listings
        assert "catalog_contents" not in _names(calls)

        assert _per_schema_listings(cat) == listings


def test_broken_falls_back_to_per_schema(location: str) -> None:
    """Advertised but failing: one `catalog_contents` attempt, then the per-schema RPCs."""
    with vp.attach(location, name=CATALOG_BROKEN) as cat:
        assert cat._attach_result.supports_catalog_contents
        calls = _record(cat)

        listings = _listings(cat)
        names = _names(calls)
        assert names[:2] == ["catalog_contents", "catalog_schemas"]
        assert names.count("catalog_contents") == 1
        assert set(names[2:]) <= _PER_SCHEMA_RPCS
        _expected_static(listings)

        calls.clear()
        assert _listings(cat) == listings
        assert calls == []  # the fallback snapshot is kept (frozen catalog)

        assert _per_schema_listings(cat) == listings


def test_reval_revalidates_with_if_none_match(location: str) -> None:
    """Etag catalog: each read sends `if_none_match`; `not_modified` keeps it, DDL replaces it."""
    with vp.attach(location, name=CATALOG_REVAL) as cat:
        assert not cat.catalog_version_frozen
        calls = _record(cat)

        assert cat.schemas() == ["main"]
        assert calls == [("catalog_contents", None)]
        etag = cat._snapshot.etag if cat._snapshot is not None else None
        assert etag is not None and etag.startswith("gen-")

        calls.clear()
        assert cat.views("main") == []
        assert calls == [("catalog_contents", etag)]
        assert cat._snapshot is not None and cat._snapshot.not_modified

        # DDL (through the catalog's own client) is seen by the very next read.
        cat.client.view_create(
            attach_opaque_data=cat.attach_opaque_data, schema_path=["main"], name="v1", definition="SELECT 1 AS x"
        )
        calls.clear()
        assert cat.views("main") == ["v1"]
        assert calls == [("catalog_contents", etag)]
        new_etag = cat._snapshot.etag if cat._snapshot is not None else None
        assert new_etag not in (None, etag)
        assert not cat._snapshot.not_modified

        calls.clear()
        assert cat.views("main") == ["v1"]
        assert calls == [("catalog_contents", new_etag)]

        # A read spanning several kinds (functions: scalar+table+aggregate) is one check.
        calls.clear()
        assert cat.functions("main") == []
        assert calls == [("catalog_contents", new_etag)]

        assert _per_schema_listings(cat) == _listings(cat)


def test_memory_version_zero_reloads_per_schema(location: str) -> None:
    """Version 0, no etag, not frozen: first read uses `catalog_contents`, later reads the per-schema RPCs."""
    with vp.attach(location, name=CATALOG_MEMORY) as cat:
        assert cat.catalog_version == 0
        calls = _record(cat)

        assert cat.schemas() == ["main"]
        assert calls == [("catalog_contents", None)]

        cat.client.view_create(
            attach_opaque_data=cat.attach_opaque_data, schema_path=["main"], name="v1", definition="SELECT 1 AS x"
        )
        calls.clear()
        assert cat.views("main") == ["v1"]
        assert calls == [("catalog_schema_contents_views", None)]

        calls.clear()
        assert cat.schemas() == ["main"]
        assert calls == [("catalog_schemas", None)]


def test_clear_cache_reloads(location: str) -> None:
    """`clear_cache()` drops the snapshot: the next read issues one fresh `catalog_contents`."""
    with vp.attach(location, name=CATALOG_PROBE) as cat:
        calls = _record(cat)
        first = cat.schemas()
        cat.clear_cache()
        assert cat.schemas() == first
        assert calls == [("catalog_contents", None), ("catalog_contents", None)]


def test_concurrent_first_reads_share_one_load(location: str) -> None:
    """Concurrent readers of a fresh catalog share one `catalog_contents`, all seeing the full listing."""
    with vp.attach(location, name=CATALOG_PROBE) as cat:
        calls = _record(cat)
        results: list[list[str]] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def reader() -> None:
            try:
                tables = cat.tables("main")
                with lock:
                    results.append(tables)
            except BaseException as e:  # noqa: BLE001 - collected for the assertion
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert results == [["ten"]] * 8
        assert calls == [("catalog_contents", None)]


def test_scan_of_snapshot_table(location: str) -> None:
    """A table resolved from the snapshot still scans correctly (its TableInfo is the real one)."""
    with vp.attach(location, name=CATALOG_PROBE) as cat:
        calls = _record(cat)
        df = cat.table("main", "ten").read()
        assert "catalog_table_get" not in _names(calls)
        assert sorted(df.to_series(0).to_list()) == list(range(10))
        assert df.to_arrow().schema.field(0).type == pa.int64()
