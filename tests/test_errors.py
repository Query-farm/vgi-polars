# Copyright 2026 Query Farm LLC - https://query.farm

"""vgi-python's two unrelated exception types (`ClientError` for exchange
RPCs, `CatalogClientError` for everything through `_catalog_connect()`) are
never leaked to a vgi-polars caller — every call site re-raises
`VgiPolarsError`. See `errors.py`'s `VGI_CLIENT_ERRORS`."""

from __future__ import annotations

import pyarrow as pa
import pytest
from vgi.catalog.catalog_interface import ScanFunctionResult
from vgi.client.catalog_mixin import CatalogClientError
from vgi.client.client import ClientError

import vgi_polars as vp
from vgi_polars._source import make_io_source
from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError


def test_vgi_client_errors_covers_both_unrelated_exception_types() -> None:
    # ClientError and CatalogClientError are NOT related by inheritance (both
    # subclass plain Exception independently) — confirms the tuple actually
    # needs both entries, not just one that happens to be a base class.
    assert not issubclass(ClientError, CatalogClientError)
    assert not issubclass(CatalogClientError, ClientError)
    assert VGI_CLIENT_ERRORS == (ClientError, CatalogClientError)


def test_bad_worker_command_raises_vgipolars_error() -> None:
    with pytest.raises(VgiPolarsError):
        vp.attach("/no/such/executable/at/all", name="example")


def test_unknown_catalog_raises_vgipolars_error(worker_location: str) -> None:
    # Routes through CatalogClientError (CatalogClientMixin._catalog_connect).
    with pytest.raises(VgiPolarsError):
        vp.attach(worker_location, name="no_such_catalog_xyz")


def test_unknown_table_raises_vgipolars_error(catalog: vp.VgiCatalog) -> None:
    with pytest.raises(VgiPolarsError, match="not found"):
        _ = catalog.table("data", "does_not_exist_xyz").schema


class _FakeTableForScanFunction:
    """Duck-typed stand-in for `VgiTable`, implementing only what
    `make_io_source` actually calls on `table`, so a scan-function *call*
    (`generator_exception`) can be exercised directly without it being
    wrapped in a catalog `Table()` entry — `generator_exception` is a bare
    schema function in vgi-fixture-worker, not a catalog table, and
    vgi-polars' public API has no "scan an arbitrary function with args"
    entry point (a real, documented gap — see CLAUDE.md)."""

    def __init__(self, catalog: vp.VgiCatalog, function_name: str, fail_after: int) -> None:
        self._catalog = catalog
        self.schema_name = "main"
        self.name = function_name
        self.at_unit = None
        self.at_value = None
        self._fn = function_name
        self._fail_after = fail_after

    def _scan_function_get(self) -> ScanFunctionResult:
        return ScanFunctionResult(
            function_name=self._fn,
            positional_arguments=[pa.scalar(self._fail_after)],
            named_arguments={},
        )

    def _function_info_get(self):
        return None  # no pushdown metadata needed for this test

    def scan_function_schema(self) -> str:
        return "main"

    def required_filters(self) -> list[list[str]]:
        return []


def test_mid_stream_worker_error_raises_vgipolars_error_not_silent_truncation(catalog: vp.VgiCatalog) -> None:
    """`generator_exception(fail_after)` yields `fail_after` good batches
    then raises — proves a worker failure *partway through* a scan surfaces
    as `VgiPolarsError` (via `_source.py`'s try/except around the exchange
    generator loop), not a silently truncated result."""
    fake_table = _FakeTableForScanFunction(catalog, "generator_exception", fail_after=2)
    io_source = make_io_source(fake_table, pa.schema([pa.field("n", pa.int64())]))

    with pytest.raises(VgiPolarsError, match="Intentional failure"):
        list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
