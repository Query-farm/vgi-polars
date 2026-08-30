# Copyright 2026 Query Farm LLC - https://query.farm

"""The core catalog/scan/scalar paths over the AF_UNIX launcher transport.

`vp.attach("launch:<argv>", ...)` is the Polars-side counterpart to the
DuckDB C++ extension's `launch:<argv>` LOCATION scheme and vgi-python's own
`Client.from_launch` (v0.29.4+) -- every caller across the machine pointing
at the same worker argv shares one warm worker process, spawned-or-reused via
`vgi_rpc.launcher`'s per-command-hash flock, self-terminating after its idle
timeout. Smoke coverage, not a duplicate of the full suite -- see
`test_http_transport.py`'s module docstring for the same rationale.

`launch_catalog` (`conftest.py`) uses an isolated `state_dir` so this can't
collide with a launcher-managed worker left running by unrelated local
`launch:` activity.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import vgi_polars as vp
from vgi_polars.catalog import _detect_transport


def test_launch_scheme_auto_detected() -> None:
    assert _detect_transport("launch:uv run vgi-fixture-worker") == "launch"


def test_schemas_over_launch(launch_catalog: vp.VgiCatalog) -> None:
    assert "data" in launch_catalog.schemas()


def test_scan_over_launch(launch_catalog: vp.VgiCatalog) -> None:
    out = launch_catalog.table("data", "numbers").scan().filter(pl.col("value") > 95).collect()
    assert sorted(out["value"].to_list()) == [96, 97, 98, 99]


def test_scalar_function_over_launch(launch_catalog: vp.VgiCatalog) -> None:
    multiply = launch_catalog.scalar_function("main", "multiply")
    df = pl.DataFrame({"value": [1, 2, 3]})
    out = df.with_columns(multiply(pl.col("value"), 3).alias("product"))
    assert out["product"].to_list() == [3, 6, 9]


def test_two_catalogs_share_one_warm_worker(worker_location: str, launch_state_dir: str) -> None:
    """The whole point of the launcher over plain subprocess: two attaches sharing one worker.

    Proven the same way vgi-python's own `test_launch_shares_one_worker_across_clients`
    does -- by checking exactly one tracked worker exists in the shared state
    dir after both attach, not by any client-visible process identity (there
    is no public API for that here either, by design).
    """
    with (
        vp.attach(f"launch:{worker_location}", name="example", state_dir=launch_state_dir, idle_timeout=30.0) as cat_a,
        vp.attach(f"launch:{worker_location}", name="example", state_dir=launch_state_dir, idle_timeout=30.0) as cat_b,
    ):
        assert "data" in cat_a.schemas()
        assert "data" in cat_b.schemas()
        meta_files = list(Path(launch_state_dir).glob("*.meta"))
        assert len(meta_files) == 1, f"expected one shared launcher-tracked worker, found {meta_files}"
