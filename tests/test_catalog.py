# Copyright 2026 Query Farm LLC - https://query.farm

"""Catalog attach/detach and introspection."""

from __future__ import annotations

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars.catalog import _detect_transport
from vgi_polars.errors import VgiPolarsError


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("http://localhost:8080", "http"),
        ("https://example.com/vgi", "http"),
        ("tcp://localhost:9000", "tcp"),
        ("uv run --project ~/vgi-python vgi-fixture-worker", "subprocess"),
        ("/usr/local/bin/my-worker", "subprocess"),
    ],
)
def test_detect_transport(location: str, expected: str) -> None:
    assert _detect_transport(location) == expected


def test_schemas(catalog: vp.VgiCatalog) -> None:
    schemas = catalog.schemas()
    assert "main" in schemas
    assert "data" in schemas


def test_tables(catalog: vp.VgiCatalog) -> None:
    tables = catalog.tables("data")
    assert "numbers" in tables


def test_functions(catalog: vp.VgiCatalog) -> None:
    functions = catalog.functions("main")
    assert "multiply" in functions


def test_function_info_comment(catalog: vp.VgiCatalog) -> None:
    """`multiply` declares a Meta.comment; `upper_case` doesn't (comment is None)."""
    info = catalog.function_info("main", "multiply")
    assert info.name == "multiply"
    assert info.comment == "fixture function for scalar bind-parameter tests"
    assert info.description == "Multiplies a value by a constant factor"

    uncommented = catalog.function_info("main", "upper_case")
    assert uncommented.comment is None


def test_function_info_not_found_raises(catalog: vp.VgiCatalog) -> None:
    with pytest.raises(VgiPolarsError, match="function not found"):
        catalog.function_info("main", "does_not_exist_xyz")


def test_table_schema_no_scan(catalog: vp.VgiCatalog) -> None:
    t = catalog.table("data", "numbers")
    assert t.schema == pl.Schema({"value": pl.Int64})


def test_table_statistics(catalog: vp.VgiCatalog) -> None:
    stats = catalog.table("data", "numbers").statistics()
    assert stats, "expected at least one column's statistics"
    value_stats = next(s for s in stats if s.column_name == "value")
    assert value_stats.min is not None
    assert value_stats.max is not None
    assert value_stats.min.as_py() == 0
    assert value_stats.max.as_py() == 99


def test_table_not_found_raises(catalog: vp.VgiCatalog) -> None:
    t = catalog.table("data", "does_not_exist_xyz")
    with pytest.raises(VgiPolarsError, match="not found"):
        _ = t.schema


def test_attach_unknown_catalog_raises(worker_location: str) -> None:
    with pytest.raises(VgiPolarsError):
        vp.attach(worker_location, name="does_not_exist_xyz")


def test_context_manager_detaches(worker_location: str) -> None:
    with vp.attach(worker_location, name="example") as cat:
        assert cat.schemas()
    # Detach is idempotent — calling it again (implicitly, or explicitly) must not raise.
    cat.detach()
