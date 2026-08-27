# Copyright 2026 Query Farm LLC - https://query.farm

"""Eager `.read()` and lazy `.scan()` — including the observed column-renaming
gotcha (a table's declared schema name need not match the resolved scan
function's own output column name; `_source.py` renames positionally to the
declared schema so downstream `with_columns`/`predicate` — which reference the
declared names — resolve correctly)."""

from __future__ import annotations

import polars as pl

import vgi_polars as vp


def test_read_eager(catalog: vp.VgiCatalog) -> None:
    df = catalog.table("data", "numbers").read()
    assert df.schema == pl.Schema({"value": pl.Int64})
    assert df.height == 100
    assert sorted(df["value"].to_list()) == list(range(100))


def test_scan_lazy_collect(catalog: vp.VgiCatalog) -> None:
    lf = catalog.table("data", "numbers").scan()
    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert df.height == 100


def test_scan_cross_schema_function(catalog: vp.VgiCatalog) -> None:
    """`data.filter_echo_table` resolves to a scan function registered only
    in schema `main` — confirms VgiTable finds it there rather than failing
    (see `table.py`'s `_resolve_scan_function`)."""
    t = catalog.table("data", "filter_echo_table")
    assert t.scan_function_schema() == "main"
    df = t.read()
    assert set(df.columns) == {"n", "s", "pushed_filters"}
