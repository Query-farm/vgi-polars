# Copyright 2026 Query Farm LLC - https://query.farm

"""Multi-branch tables (`VgiTable.scan()` → `_multi_branch.scan_multi_branch`).

Tested against the real `data.multi_branch_*` fixtures
(`vgi/_test_fixtures/worker.py`), which needed a vgi-python fix first:
`Client` had no `table_scan_branches_get` at all, so a multi-branch table
was invisible to any non-DuckDB caller — worse, scanning one through the
old `table_scan_function_get`-only path silently returned only its first
branch. See CLAUDE.md's "Multi-branch tables" section.
"""

from __future__ import annotations

import pytest
from vgi.client.client import Client

import vgi_polars as vp
from vgi_polars._multi_branch import parse_branch_filter
from vgi_polars.errors import VgiPolarsError

# `Client.table_scan_branches_get` is a new upstream addition (see this
# module's docstring). Skipped, not failed, against an older installed
# vgi-python — mirrors `test_splits.py`'s `requires_split_support` pattern.
requires_branches_support = pytest.mark.skipif(
    not hasattr(Client, "table_scan_branches_get"),
    reason="installed vgi-python predates Client.table_scan_branches_get",
)


@requires_branches_support
def test_two_branch_table_unions_both_arms(catalog: vp.VgiCatalog) -> None:
    """multi_branch_numbers: two sequence(50) arms, no branch_filter — union size 100."""
    out = catalog.table("data", "multi_branch_numbers").scan().collect()
    assert out.height == 100
    assert sorted(out["n"].to_list()) == sorted(list(range(50)) * 2)


@requires_branches_support
def test_branch_filter_makes_overlapping_arms_disjoint(catalog: vp.VgiCatalog) -> None:
    """multi_branch_filtered_numbers: two sequence(100) arms with complementary branch_filters.

    ('n < 50' / 'n >= 50') carving the range in half — total 100 rows, no
    duplicates, proving branch_filter is actually applied (not just a
    no-op union that would give 200 rows).
    """
    out = catalog.table("data", "multi_branch_filtered_numbers").scan().collect()
    assert out.height == 100
    assert sorted(out["n"].to_list()) == list(range(100))


def test_single_branch_table_takes_the_unchanged_scan_path(catalog: vp.VgiCatalog) -> None:
    """An ordinary table (one branch under the hood, via the legacy-RPC fallback).

    Must scan exactly as before — no multi-branch overhead visible in the
    result.
    """
    out = catalog.table("data", "numbers").scan().collect()
    assert out.height > 0


def test_catalog_table_branch_raises_clearly(catalog: vp.VgiCatalog) -> None:
    """`source_table`-discriminated branches (companion-catalog federation) are explicitly out of scope.

    Must raise `VgiPolarsError`, never silently mis-scan or crash with an
    unrelated error.
    """
    from vgi.catalog.catalog_interface import ScanBranch

    from vgi_polars._multi_branch import scan_multi_branch

    table = catalog.table("data", "multi_branch_numbers")
    branch = ScanBranch(function_name="", positional_arguments=[], named_arguments={}, source_table="ducklake_tbl")
    with pytest.raises(VgiPolarsError, match="catalog-table branch"):
        scan_multi_branch(table, [branch])


def test_format_branch_raises_clearly(catalog: vp.VgiCatalog) -> None:
    from vgi.catalog.catalog_interface import ScanBranch

    from vgi_polars._multi_branch import scan_multi_branch

    table = catalog.table("data", "multi_branch_numbers")
    branch = ScanBranch(
        function_name="",
        positional_arguments=[],
        named_arguments={},
        format_name="parquet",
        format_locations=["s3://bucket/data.parquet"],
    )
    with pytest.raises(VgiPolarsError, match="format branch"):
        scan_multi_branch(table, [branch])


def test_zero_branches_scans_empty(catalog: vp.VgiCatalog) -> None:
    from vgi_polars._multi_branch import scan_multi_branch

    table = catalog.table("data", "multi_branch_numbers")
    out = scan_multi_branch(table, []).collect()
    assert out.height == 0
    assert out.columns == table.schema.names()


# --- parse_branch_filter unit tests -----------------------------------------


def test_parse_branch_filter_single_comparison() -> None:
    import polars as pl

    df = pl.DataFrame({"n": [1, 49, 50, 99]})
    out = df.filter(parse_branch_filter("n < 50"))
    assert out["n"].to_list() == [1, 49]


def test_parse_branch_filter_and_chain() -> None:
    import polars as pl

    df = pl.DataFrame({"n": [1, 25, 60, 99]})
    out = df.filter(parse_branch_filter("n >= 10 AND n < 50"))
    assert out["n"].to_list() == [25]


def test_parse_branch_filter_string_literal() -> None:
    import polars as pl

    df = pl.DataFrame({"country": ["US", "FR", "US"]})
    out = df.filter(parse_branch_filter("country = 'US'"))
    assert out.height == 2


def test_parse_branch_filter_unsupported_raises() -> None:
    with pytest.raises(VgiPolarsError, match="not understood"):
        parse_branch_filter("n < 50 OR n > 100")
