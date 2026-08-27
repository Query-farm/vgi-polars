# Copyright 2026 Query Farm LLC - https://query.farm

"""The core catalog/scan/scalar paths over TCP transport (raw Arrow-IPC
framing, no auth/encryption — loopback only), mirroring the subprocess- and
HTTP-transport tests. Smoke coverage, not a duplicate of the full suite — see
`test_http_transport.py`'s module docstring for the same rationale.

`tcp_worker_base_url` (session-scoped, `conftest.py`) is backed by a
launcher-style reused-or-spawned warm worker (`_launch_tcp_worker`) rather
than a fresh subprocess per test — the TCP analogue of `vgi_rpc.launcher`'s
unix-socket design, since `vgi.client.Client` has no unix-socket transport to
pair the real launcher module with (see CLAUDE.md's transport-gap note)."""

from __future__ import annotations

import polars as pl

import vgi_polars as vp


def test_schemas_over_tcp(tcp_catalog: vp.VgiCatalog) -> None:
    assert "data" in tcp_catalog.schemas()


def test_scan_over_tcp(tcp_catalog: vp.VgiCatalog) -> None:
    out = tcp_catalog.table("data", "numbers").scan().filter(pl.col("value") > 95).collect()
    assert sorted(out["value"].to_list()) == [96, 97, 98, 99]


def test_scalar_function_over_tcp(tcp_catalog: vp.VgiCatalog) -> None:
    multiply = tcp_catalog.scalar_function("main", "multiply")
    df = pl.DataFrame({"value": [1, 2, 3]})
    out = df.with_columns(multiply(pl.col("value"), 3).alias("product"))
    assert out["product"].to_list() == [3, 6, 9]


def test_required_filters_over_tcp(tcp_catalog: vp.VgiCatalog) -> None:
    """The required-filters cost-safety check is transport-agnostic (purely
    client-side, before any RPC) — a smoke check it also works over TCP."""
    out = tcp_catalog.table("data", "rff_simple").scan().filter(pl.col("a") > 1).collect()
    assert sorted(out["a"].to_list()) == [2, 3]
