# Copyright 2026 Query Farm LLC - https://query.farm

"""`required_filters` cost-safety enforcement (`_source.py::_check_required_filters`)
and the ordinary-column behavior of a `row_id`-flavored table (`rff_rowid`).

`TableInfo.required_filters` is purely declarative on the wire — vgi-python does
no client-side enforcement itself (the DuckDB C++ extension's optimizer does it
there instead). Without a check here, `.scan().collect()` against a
`required_filters` table with no matching predicate would trigger a full,
possibly enormous, unfiltered remote scan; Design Principle 1 (see `_source.py`)
keeps that *correct* but says nothing about *cost*. These tests exercise the
`rff_*` fixture matrix already used by `~/Development/vgi`'s own
`required_filters_*.test` sqllogictests (schema `data`, see
`vgi/_test_fixtures/table/required_filters.py`)."""

from __future__ import annotations

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars._source import _check_required_filters
from vgi_polars.errors import VgiPolarsError


def test_required_filters_reports_declared_groups(catalog: vp.VgiCatalog) -> None:
    assert catalog.table("data", "rff_simple").required_filters() == [["a"]]
    assert catalog.table("data", "rff_or").required_filters() == [["a", "b"]]
    assert catalog.table("data", "rff_struct").required_filters() == [["s.a"], ["s.b"]]
    assert catalog.table("data", "rff_multi").required_filters() == [["top"], ["s.a"]]
    assert catalog.table("data", "rff_none").required_filters() == []


def test_check_required_filters_raises_vgipolars_error_directly() -> None:
    """Unit-level check of the raw exception type `_check_required_filters`
    itself raises, bypassing Polars' Python-source engine (which wraps any
    exception raised inside `io_source` in `polars.exceptions.ComputeError` —
    see the next test's comment)."""
    with pytest.raises(VgiPolarsError, match=r"requires a filter on one of \['a'\]"):
        _check_required_filters([["a"]], None, "data", "rff_simple")
    # No-op cases: satisfied group, and no requirement at all.
    _check_required_filters([["a"]], pl.col("a") > 1, "data", "rff_simple")
    _check_required_filters([], None, "data", "rff_none")


def test_scan_without_required_filter_raises(catalog: vp.VgiCatalog) -> None:
    # `io_source` raises `VgiPolarsError`, but a Python-source generator runs
    # inside Polars' own execution engine — an exception raised there surfaces
    # to `.collect()` wrapped in `polars.exceptions.ComputeError` (the original
    # message is preserved as text, not chained; see `test_errors.py`'s
    # `_FakeTableForScanFunction` pattern for asserting the raw `VgiPolarsError`
    # by calling `make_io_source`'s callable directly instead).
    t = catalog.table("data", "rff_simple")
    with pytest.raises(pl.exceptions.ComputeError, match="requires a filter"):
        t.scan().collect()


def test_scan_with_required_filter_succeeds(catalog: vp.VgiCatalog) -> None:
    t = catalog.table("data", "rff_simple")
    out = t.scan().filter(pl.col("a") > 1).collect()
    assert sorted(out["a"].to_list()) == [2, 3]


def test_scan_or_group_satisfied_by_either_column(catalog: vp.VgiCatalog) -> None:
    """`rff_or` declares a single OR-group `("a", "b")` — a filter on just `b`
    (not `a`) must still satisfy it."""
    t = catalog.table("data", "rff_or")
    out = t.scan().filter(pl.col("b") > 10).collect()
    assert sorted(out["a"].to_list()) == [2, 3]


def test_scan_or_group_unsatisfied_by_unrelated_column_raises(catalog: vp.VgiCatalog) -> None:
    """rff_or shares its columns with rff_simple's schema (a, b) — a predicate
    that references neither still fails the requirement even though the table
    has other columns to filter on in principle (there are none here, but this
    proves the check isn't accidentally satisfied by "some predicate exists")."""
    t = catalog.table("data", "rff_or")
    with pytest.raises(pl.exceptions.ComputeError, match="requires a filter"):
        t.scan().filter(pl.lit(True)).collect()


def test_scan_no_required_filters_table_needs_no_predicate(catalog: vp.VgiCatalog) -> None:
    t = catalog.table("data", "rff_none")
    out = t.scan().collect()
    assert sorted(out["a"].to_list()) == [1, 2, 3]


def test_scan_struct_subfield_requirement_conservative_approximation(catalog: vp.VgiCatalog) -> None:
    """`rff_struct` requires filters on both `s.a` and `s.b` — two singleton
    groups keyed on the *same* top-level column `s`. `root_names()` can't see
    through a `.struct.field(...)` access to know which subfield was touched, so
    the check conservatively treats a predicate referencing top-level `s` at all
    as satisfying every group keyed under `s` (documented in
    `_check_required_filters`'s docstring: never blocks a legitimate query, not a
    full pushdown-translatability check)."""
    t = catalog.table("data", "rff_struct")
    out = t.scan().filter(pl.col("s").struct.field("a") > 1).collect()
    assert sorted(out["other"].to_list()) == [200, 300]


def test_scan_multi_requires_every_and_group(catalog: vp.VgiCatalog) -> None:
    """`rff_multi` declares two AND'd singleton groups (`top`, `s.a`) — a
    predicate satisfying only one must still raise."""
    t = catalog.table("data", "rff_multi")
    with pytest.raises(pl.exceptions.ComputeError, match="requires a filter"):
        t.scan().filter(pl.col("top") > 100).collect()

    out = t.scan().filter((pl.col("top") > 100) & (pl.col("s").struct.field("a") > 0)).collect()
    assert out["top"].to_list() == [200]


def test_rff_rowid_row_id_is_an_ordinary_column(catalog: vp.VgiCatalog) -> None:
    """No special DuckDB-side "virtual, hidden from SELECT *" handling exists in
    vgi-polars (there's no SQL planner to hide it from) — `row_id` just shows up
    as a normal declared schema column, queryable and filterable like any other.
    `rff_rowid` requires filters on all four `bbox.*` subfields (four AND'd
    singleton groups, all keyed under top-level `bbox`), satisfied here by a
    single predicate touching `bbox`."""
    t = catalog.table("data", "rff_rowid")
    assert "row_id" in t.schema.names()
    out = t.scan().filter(pl.col("bbox").struct.field("xmin") >= 5.0).select(["row_id", "other"]).collect()
    assert sorted(out["row_id"].to_list()) == [5, 6, 7, 8, 9]
    assert sorted(out["other"].to_list()) == [50, 60, 70, 80, 90]
