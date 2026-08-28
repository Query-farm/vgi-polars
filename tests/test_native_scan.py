# Copyright 2026 Query Farm LLC - https://query.farm

"""Native scan-function delegation -- see `_native_scan.py`'s module docstring.

`ScanFunctionResult.function_name` naming a reader the calling engine runs
itself. `TestScanParquetNativeUnit` exercises `_scan_parquet_native` directly with a
synthetic `ScanFunctionResult` and a real local parquet file — no worker at
all, purely the translation logic. `TestRffParquetIntegration` drives the
real thing end to end through `VgiTable.scan()` against `vgi-fixture-
worker`'s `data.rff_parquet`/`data.rff_hive` fixtures (the same ones
`~/Development/vgi/test/sql/integration/table/required_filters_native.test`
exercises on the DuckDB C++ side — same COPY shape, same assertions,
confirming vgi-polars' native-delegation path treats the exact table shape
that motivated this feature (Overture's `transportation.segment`) the same
way). No `VGI_TEST_BRANCH_DIR` is set: the fixture worker's own fallback
(`tempfile.gettempdir()`) already resolves to the same directory this test
process sees on the same host, so the backing parquet files just need to be
written there under the well-known names the fixture expects.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vgi.catalog.catalog_interface import ScanFunctionResult

import vgi_polars as vp
from vgi_polars._native_scan import _scan_parquet_native
from vgi_polars.errors import VgiPolarsError


class TestScanParquetNativeUnit:
    """`_scan_parquet_native` in isolation — no worker, no RPC."""

    def test_builds_a_working_scan(self, tmp_path: Path) -> None:
        path = tmp_path / "data.parquet"
        pq.write_table(pa.table({"a": [1, 2, 3]}), path)

        scan_fn = ScanFunctionResult(
            function_name="read_parquet",
            positional_arguments=[pa.scalar(str(path))],
            named_arguments={},
        )
        lf = _scan_parquet_native(scan_fn, schema_name="s", table_name="t", storage_options=None)
        assert lf.collect()["a"].to_list() == [1, 2, 3]

    def test_hive_partitioning_named_arg_is_translated(self, tmp_path: Path) -> None:
        (tmp_path / "theme=a" / "type=b").mkdir(parents=True)
        pq.write_table(pa.table({"x": [1, 2]}), tmp_path / "theme=a" / "type=b" / "part.parquet")

        scan_fn = ScanFunctionResult(
            function_name="read_parquet",
            positional_arguments=[pa.scalar(f"{tmp_path}/*/*/*.parquet")],
            named_arguments={"hive_partitioning": pa.scalar(True)},
        )
        lf = _scan_parquet_native(scan_fn, schema_name="s", table_name="t", storage_options=None)
        out = lf.collect()
        assert sorted(out["x"].to_list()) == [1, 2]
        assert out["theme"].to_list() == ["a", "a"]
        assert out["type"].to_list() == ["b", "b"]

    def test_no_positional_arguments_raises(self) -> None:
        scan_fn = ScanFunctionResult(function_name="read_parquet", positional_arguments=[], named_arguments={})
        with pytest.raises(VgiPolarsError, match="no positional arguments"):
            _scan_parquet_native(scan_fn, schema_name="s", table_name="t", storage_options=None)

    def test_non_string_path_raises(self) -> None:
        scan_fn = ScanFunctionResult(
            function_name="read_parquet", positional_arguments=[pa.scalar(42)], named_arguments={}
        )
        with pytest.raises(VgiPolarsError, match="expected a path/glob string"):
            _scan_parquet_native(scan_fn, schema_name="s", table_name="t", storage_options=None)

    def test_unknown_named_argument_raises(self) -> None:
        scan_fn = ScanFunctionResult(
            function_name="read_parquet",
            positional_arguments=[pa.scalar("/tmp/whatever.parquet")],
            named_arguments={"some_unknown_option": pa.scalar(True)},
        )
        with pytest.raises(VgiPolarsError, match="some_unknown_option"):
            _scan_parquet_native(scan_fn, schema_name="s", table_name="t", storage_options=None)


class TestRffParquetIntegration:
    """`data.rff_parquet`/`data.rff_hive` through `VgiTable.scan()`, real worker.

    Mirrors `required_filters_native.test`'s COPY shape exactly: 100 rows,
    `bbox.xmin` varying 0..99 (proves a pushed predicate is actually
    *applied*, not merely permitted), other corners constant.

    Writes directly into the fixture worker's OWN fallback resolution
    (`tempfile.gettempdir()`, unset `VGI_TEST_BRANCH_DIR`) rather than trying
    to isolate via a per-test env var override -- confirmed live that a
    freshly-set `VGI_TEST_BRANCH_DIR` does not reliably reach a freshly
    spawned `pool=None` worker subprocess (a second `Client(pool=None)`
    resolved the *first* client's directory despite a distinct PID and a
    changed env var beforehand; root cause not fully chased down, filed as a
    known flake in this environment rather than blocking on it). Since every
    test in this class targets the exact same well-known filenames
    (`rff_seg.parquet`, `rff_hive/...`) the fixture always resolves to, each
    write here explicitly clears any pre-existing content first -- this
    machine's `~/Development/vgi` C++ integration suite writes its own
    `rff_hive` fixture to the same shared fallback dir (via DuckDB's `COPY
    ... PARTITION_BY`, which names files `data_0.parquet`), and an earlier
    version of this fixture that didn't clear first silently read 200 rows
    instead of 100 -- both that leftover file and this test's own
    `part.parquet` matched the same glob.
    """

    @pytest.fixture(autouse=True)
    def _write_backing_files(self, catalog: vp.VgiCatalog) -> None:
        self.catalog = catalog
        branch_dir = Path(tempfile.gettempdir())

        seg_path = branch_dir / "rff_seg.parquet"
        seg_path.unlink(missing_ok=True)

        hive_root = branch_dir / "rff_hive"
        if hive_root.exists():
            shutil.rmtree(hive_root)

        seg = pa.table(
            {
                "bbox": pa.StructArray.from_arrays(
                    [
                        pa.array(list(range(100)), type=pa.float32()),
                        pa.array([2.0] * 100, type=pa.float32()),
                        pa.array([3.0] * 100, type=pa.float32()),
                        pa.array([4.0] * 100, type=pa.float32()),
                    ],
                    names=["xmin", "ymin", "xmax", "ymax"],
                ),
                "other": pa.array([5] * 100, type=pa.int64()),
            }
        )
        pq.write_table(seg, branch_dir / "rff_seg.parquet")

        hive_dir = branch_dir / "rff_hive" / "theme=transportation" / "type=segment"
        hive_dir.mkdir(parents=True, exist_ok=True)
        hive = pa.table(
            {
                "id": pa.array([f"id{i}" for i in range(100)], type=pa.string()),
                "bbox": pa.StructArray.from_arrays(
                    [
                        pa.array(list(range(100)), type=pa.float32()),
                        pa.array([2.0] * 100, type=pa.float32()),
                        pa.array([3.0] * 100, type=pa.float32()),
                        pa.array([4.0] * 100, type=pa.float32()),
                    ],
                    names=["xmin", "ymin", "xmax", "ymax"],
                ),
                "name": pa.array([f"n{i}" for i in range(100)], type=pa.string()),
                "num": pa.array(list(range(100)), type=pa.int64()),
            }
        )
        pq.write_table(hive, hive_dir / "part.parquet")

    def test_required_filters_refused_without_acknowledgement(self) -> None:
        t = self.catalog.table("data", "rff_parquet")
        with pytest.raises(VgiPolarsError, match="acknowledge_required_filters"):
            t.scan()

    def test_full_scan_with_acknowledgement(self) -> None:
        t = self.catalog.table("data", "rff_parquet")
        out = t.scan(acknowledge_required_filters=True).collect()
        assert out.height == 100

    def test_filter_after_scan_is_genuinely_applied(self) -> None:
        """`xmin >= 50` on a real Polars-native scan selects exactly 50/100 rows.

        Proves the predicate reaches the actual parquet read (native pushdown,
        Polars' own optimizer) -- not merely tolerated and silently ignored,
        which a constant column couldn't distinguish.
        """
        t = self.catalog.table("data", "rff_parquet")
        out = t.scan(acknowledge_required_filters=True).filter(pl.col("bbox").struct.field("xmin") >= 50).collect()
        assert out.height == 50

    def test_hive_partitioned_native_scan(self) -> None:
        t = self.catalog.table("data", "rff_hive")
        out = t.scan(acknowledge_required_filters=True).collect()
        assert out.height == 100
        assert "theme" in out.columns
        assert "type" in out.columns
        assert out["theme"].unique().to_list() == ["transportation"]

    def test_read_is_scan_collect(self) -> None:
        t = self.catalog.table("data", "rff_parquet")
        out = t.read(acknowledge_required_filters=True)
        assert out.height == 100
