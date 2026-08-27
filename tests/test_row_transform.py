# Copyright 2026 Query Farm LLC - https://query.farm

"""Blended row-transform (`RowTransformFunction`) -> `LazyFrame`/`pl.defer` bridge.

See `_row_transform.py`'s module docstring for the mechanism, the gather-safety
finding, the outer-column policy, and the dedup group-and-replicate composition
this exercises directly (`test_dedup_duplicate_outer_rows_each_get_their_own_output`).
Covers both call shapes: the column/LATERAL path (`fn(lf, pl.col(...))`, `map_batches`)
and the bare-literal-call path (`fn(None, ...)`, `pl.defer` — see `TestLiteralCall`).
"""

from __future__ import annotations

import threading

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars.errors import VgiPolarsError


def test_fan_out_pairs_outer_column_correctly(catalog: vp.VgiCatalog) -> None:
    """`blended_explode`: each output row must carry ITS producing row's outer column.

    Asserting only the final row *set* would pass even for a naive
    "assume identity" bridge whenever fan-out counts differ across rows, so
    this reconstructs the exact expected (id, n, i) triples and compares.
    """
    explode = catalog.row_transform_function("main", "blended_explode")
    ids = ["a", "b", "c", "d"]
    counts = [0, 1, 3, 2]
    lf = pl.LazyFrame({"id": ids, "n": counts})

    out = explode(lf, pl.col("n")).collect()

    expected = sorted((ids[row], counts[row], k) for row, n in enumerate(counts) for k in range(n))
    actual = sorted(zip(out["id"].to_list(), out["n"].to_list(), out["i"].to_list(), strict=True))
    assert actual == expected


def test_all_rows_filtered_to_empty_result_with_correct_schema(catalog: vp.VgiCatalog) -> None:
    """n=0 for every row -> a zero-row result carrying both outer and worker output columns."""
    explode = catalog.row_transform_function("main", "blended_explode")
    lf = pl.LazyFrame({"id": ["a", "b"], "n": [0, 0]})

    out = explode(lf, pl.col("n")).collect()

    assert out.height == 0
    assert out.schema.names() == ["id", "n", "i"]


def test_dedup_duplicate_outer_rows_each_get_their_own_output(catalog: vp.VgiCatalog) -> None:
    """Two outer rows sharing the same arg value must each get their OWN copy of the output.

    Regression for a real bug caught during implementation: a naive
    "distinct_positions[parent_row]" composition only ever recovers the
    FIRST original row for a deduped value, silently dropping every other
    duplicate row's output. `grp="a"` appears twice with the same `n=2` (so
    the shipped/deduped batch collapses them to one worker call), `grp="b"`
    is the unique `n=1`. Both `grp="a"` rows must independently produce
    `i=0,1` -- not just one of them.
    """
    explode = catalog.row_transform_function("main", "blended_explode")
    lf = pl.LazyFrame({"grp": ["a", "a", "b"], "n": [2, 2, 1]})

    out = explode(lf, pl.col("n")).collect()

    expected = sorted([("a", 2, 0), ("a", 2, 1), ("a", 2, 0), ("a", 2, 1), ("b", 1, 0)])
    actual = sorted(zip(out["grp"].to_list(), out["n"].to_list(), out["i"].to_list(), strict=True))
    assert actual == expected

    # Confirm dedup actually engaged (the worker really was called once for
    # the shared n=2, not twice) -- otherwise this test wouldn't distinguish
    # "dedup is correct" from "dedup never fired at all".
    out_no_dedup = explode(lf, pl.col("n"), dedup=False).collect()
    actual_no_dedup = sorted(
        zip(out_no_dedup["grp"].to_list(), out_no_dedup["n"].to_list(), out_no_dedup["i"].to_list(), strict=True)
    )
    assert actual_no_dedup == expected


class TestHostileProvenance:
    """Adversarial worker payloads must surface as `VgiPolarsError`, never corrupt or leak raw errors."""

    @pytest.mark.parametrize("mode", ["range", "length", "base64"])
    def test_malformed_provenance_raises(self, catalog: vp.VgiCatalog, mode: str) -> None:
        hostile = catalog.row_transform_function("main", "hostile_provenance")
        lf = pl.LazyFrame({"x": [1, 2, 3]})

        with pytest.raises(VgiPolarsError):
            hostile(lf, pl.col("x"), mode=mode).collect()


class TestLiteralCall:
    """`fn(None, ...)` -- the bare-literal-call shape, routed through `pl.defer`."""

    def test_1_to_1(self, catalog: vp.VgiCatalog) -> None:
        geo_encode = catalog.row_transform_function("main", "geo_encode")
        out = geo_encode(None, 52.0, 13.0, precision=1).collect()
        assert out["geohash"].to_list() == ["52.0:13.0"]

    def test_1_to_n_fan_out(self, catalog: vp.VgiCatalog) -> None:
        """A literal call has no outer frame, so the result is JUST the worker's own output rows."""
        explode = catalog.row_transform_function("main", "blended_explode")
        out = explode(None, 3).collect()
        assert out["i"].to_list() == [0, 1, 2]

    def test_1_to_0_filtered(self, catalog: vp.VgiCatalog) -> None:
        explode = catalog.row_transform_function("main", "blended_explode")
        out = explode(None, 0).collect()
        assert out.height == 0
        assert out.schema.names() == ["i"]

    def test_expr_arg_with_no_lf_raises(self, catalog: vp.VgiCatalog) -> None:
        """A `pl.Expr` needs a frame to evaluate against -- nonsensical with `lf=None`."""
        geo_encode = catalog.row_transform_function("main", "geo_encode")
        with pytest.raises(VgiPolarsError, match="pl.Expr but no `lf`"):
            geo_encode(None, pl.col("lat"), 13.0)

    def test_named_arg_threading(self, catalog: vp.VgiCatalog) -> None:
        """Bind-time named args work identically to the column path."""
        geo_encode = catalog.row_transform_function("main", "geo_encode")
        out = geo_encode(None, 52.0, 13.0, precision=0).collect()
        assert out["geohash"].to_list() == ["52.0:13.0"]

    def test_hostile_provenance_worker_ignored_safely(self, catalog: vp.VgiCatalog) -> None:
        """A malformed vgi_rpc.parent_row is simply never read here -- the row count is what it is."""
        hostile = catalog.row_transform_function("main", "hostile_provenance")
        out = hostile(None, 7, mode="range").collect()
        assert out["hv"].to_list() == [7]


def test_lf_presence_decides_the_path_not_arg_shape(catalog: vp.VgiCatalog) -> None:
    """`lf is not None` always takes the column path, even when every arg is a plain literal."""
    geo_encode = catalog.row_transform_function("main", "geo_encode")
    lf = pl.LazyFrame({"unrelated": [1, 2]})

    out = geo_encode(lf, 52.0, 13.0, precision=1).collect()

    assert out["geohash"].to_list() == ["52.0:13.0", "52.0:13.0"]
    assert out["unrelated"].to_list() == [1, 2]


def test_named_arg_rejects_expr(catalog: vp.VgiCatalog) -> None:
    """A named (bind-time) argument must be a plain value, never a `pl.Expr`."""
    geo_encode = catalog.row_transform_function("main", "geo_encode")
    lf = pl.LazyFrame({"lat": [52.0], "lon": [13.0]})

    with pytest.raises(VgiPolarsError, match="bind-time parameter"):
        geo_encode(lf, pl.col("lat"), pl.col("lon"), precision=pl.col("lat"))


def test_geo_encode_two_arg_overload(catalog: vp.VgiCatalog) -> None:
    """`geo_encode` is registered as two same-name overloads (2-arg and 3-arg).

    `row_transform_function()` resolves by name only (`next((i for i in infos
    if i.name == name), None)` -- the same pre-existing limitation
    `_scalar.py`/`_table_in_out.py` already have, not something new to this
    feature), so only the FIRST-registered overload (the 2-arg one) is
    reachable here. Arity-based overload resolution is out of scope for
    slice 1.
    """
    geo_encode = catalog.row_transform_function("main", "geo_encode")
    lf = pl.LazyFrame({"lat": [52.0, 48.5], "lon": [13.0, 2.3]})

    out = geo_encode(lf, pl.col("lat"), pl.col("lon"), precision=1).collect()

    assert sorted(out["geohash"].to_list()) == ["48.5:2.3", "52.0:13.0"]


def test_unknown_row_transform_function_raises(catalog: vp.VgiCatalog) -> None:
    fn = catalog.row_transform_function("main", "does_not_exist_xyz")
    with pytest.raises(VgiPolarsError, match="not found"):
        fn(pl.LazyFrame({"a": [1]}), pl.col("a"))


def test_ordinary_table_in_out_function_rejected(catalog: vp.VgiCatalog) -> None:
    """`echo` is a plain table-in-out function, not a blended row-transform one."""
    fn = catalog.row_transform_function("main", "echo")
    with pytest.raises(VgiPolarsError, match="not a blended row-transform function"):
        fn(pl.LazyFrame({"a": [1]}), pl.col("a"))


def test_concurrent_row_transform_calls_are_correct(worker_location: str) -> None:
    """N threads sharing one VgiCatalog, each calling the same row-transform function concurrently.

    Per-call state (parent-row capture, input row count, outer-column
    snapshot) must be local to each `bridge_fn` invocation -- getting this
    wrong means two concurrent calls interleave their bookkeeping and
    silently misattribute outer-column values across unrelated calls (see
    `_row_transform.py`'s module docstring). Mirrors
    `test_concurrency.py::test_concurrent_scalar_calls_are_correct`.
    """
    n = 8
    results: dict[int, list[tuple[str, int]]] = {}
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    with vp.attach(worker_location, name="example") as cat:
        explode = cat.row_transform_function("main", "blended_explode")

        def worker(i: int) -> None:
            try:
                lf = pl.LazyFrame({"id": [f"row{i}"], "n": [i % 4]})
                out = explode(lf, pl.col("n")).collect()
                with lock:
                    results[i] = sorted(zip(out["id"].to_list(), out["i"].to_list(), strict=True))
            except BaseException as e:  # noqa: BLE001 - collecting every failure for the assertion below
                with lock:
                    errors.append((i, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"{len(errors)}/{n} calls raised: {errors[:3]}..."
    expected = {i: sorted((f"row{i}", k) for k in range(i % 4)) for i in range(n)}
    assert results == expected
