# Copyright 2026 Query Farm LLC - https://query.farm

"""Projection: correct result regardless of whether the scan function honors pushdown.

Whether or not the resolved scan function actually honors pushed
`projection_ids`, Design Principle 1 (`_source.py`) applies
`.select(with_columns)` locally regardless.
"""

from __future__ import annotations

import polars as pl

import vgi_polars as vp


def test_select_subset_of_columns(catalog: vp.VgiCatalog) -> None:
    t = catalog.table("data", "departments")
    assert set(t.schema.names()) == {"id", "name", "budget"}

    out = t.scan().select("name").collect()
    assert out.columns == ["name"]
    assert out.height > 0


def test_select_after_filter(catalog: vp.VgiCatalog) -> None:
    t = catalog.table("data", "numbers")
    out = t.scan().filter(pl.col("value") > 90).select("value").collect()
    assert out.columns == ["value"]
    assert sorted(out["value"].to_list()) == list(range(91, 100))
