# Copyright 2026 Query Farm LLC - https://query.farm

"""Aggregate functions via the eager `pl.DataFrame` bridge (`_aggregate.py`).

`AggregateClientMixin.aggregate_function` already drives the whole bind/
update/finalize/destructor loop; this is a thin conversion layer, not a lazy
Polars expression (Polars has no groupby-aggregation plugin hook).
"""

from __future__ import annotations

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars.errors import VgiPolarsError


def test_grouped_sum(catalog: vp.VgiCatalog) -> None:
    vgi_sum = catalog.aggregate_function("main", "vgi_sum")
    df = pl.DataFrame({"cat": ["a", "a", "b"], "value": [1, 2, 30]})
    out = vgi_sum(df, group_by=["cat"])
    result = dict(zip(out["cat"].to_list(), out["result"].to_list(), strict=True))
    assert result == {"a": 3, "b": 30}


def test_ungrouped_sum_is_one_row(catalog: vp.VgiCatalog) -> None:
    """No `group_by` -> one global-aggregate row, matching SQL semantics."""
    vgi_sum = catalog.aggregate_function("main", "vgi_sum")
    df = pl.DataFrame({"value": [1, 2, 3, 4]})
    out = vgi_sum(df)
    assert out.height == 1
    assert out["result"][0] == 10


def test_const_arg_aggregate(catalog: vp.VgiCatalog) -> None:
    """`vgi_percentile(value, 0.5)`.

    `percentile` is a `ConstParam`, same wire convention (`vgi_const`
    metadata) as a scalar function's const args.
    """
    percentile = catalog.aggregate_function("main", "vgi_percentile")
    df = pl.DataFrame({"cat": ["a", "a", "a"], "value": [1.0, 2.0, 3.0]})
    out = percentile(df, 0.5, group_by=["cat"])
    assert out["result"][0] == pytest.approx(2.0)


def test_wrong_const_arg_count_raises(catalog: vp.VgiCatalog) -> None:
    percentile = catalog.aggregate_function("main", "vgi_percentile")
    df = pl.DataFrame({"value": [1.0, 2.0]})
    with pytest.raises(VgiPolarsError, match="expects 1 constant argument"):
        percentile(df)


def test_unknown_aggregate_function_raises(catalog: vp.VgiCatalog) -> None:
    fn = catalog.aggregate_function("main", "does_not_exist_xyz")
    with pytest.raises(VgiPolarsError, match="not found"):
        fn(pl.DataFrame({"value": [1]}))
