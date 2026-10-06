# Copyright 2026 Query Farm LLC - https://query.farm

"""The core catalog/scan/scalar paths over HTTP transport, mirroring the subprocess-transport tests.

VGI's client-side surface is transport-agnostic, so these are deliberately
a small subset (smoke coverage), not a duplicate of the full subprocess
suite.
"""

from __future__ import annotations

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars.errors import VgiPolarsError


def test_schemas_over_http(http_catalog: vp.VgiCatalog) -> None:
    assert "data" in http_catalog.schemas()


def test_scan_over_http(http_catalog: vp.VgiCatalog) -> None:
    out = http_catalog.table("data", "numbers").scan().filter(pl.col("value") > 95).collect()
    assert sorted(out["value"].to_list()) == [96, 97, 98, 99]


def test_scalar_function_over_http(http_catalog: vp.VgiCatalog) -> None:
    multiply = http_catalog.scalar_function("main", "multiply")
    df = pl.DataFrame({"value": [1, 2, 3]})
    out = df.with_columns(multiply(pl.col("value"), 3).alias("product"))
    assert out["product"].to_list() == [3, 6, 9]


def test_exchange_borrows_share_one_httpx_client_over_http(http_catalog: vp.VgiCatalog) -> None:
    """Two borrowed exchange Clients over plain (non-OAuth) HTTP share one httpx2.Client.

    Confirms catalog.py's client_factory() sharing actually happens, not
    just that the code exists — each fresh per-call Client would otherwise
    pay its own fresh TLS/TCP handshake instead of reusing keep-alive
    connections, defeating the point of per-call borrowing over HTTP.
    """
    with http_catalog._exchange_client() as client_a:
        httpx_a = client_a._get_or_create_httpx_client()
    with http_catalog._exchange_client() as client_b:
        httpx_b = client_b._get_or_create_httpx_client()
    assert httpx_a is httpx_b


def test_bearer_auth_with_correct_token(http_bearer_worker_base_url: str, http_bearer_token: str) -> None:
    with vp.attach(http_bearer_worker_base_url, name="example", bearer_token=http_bearer_token) as cat:
        assert "data" in cat.schemas()


def test_bearer_auth_without_token_rejected(http_bearer_worker_base_url: str) -> None:
    with pytest.raises(VgiPolarsError):
        vp.attach(http_bearer_worker_base_url, name="example")


def test_bearer_auth_with_wrong_token_rejected(http_bearer_worker_base_url: str) -> None:
    with pytest.raises(VgiPolarsError):
        vp.attach(http_bearer_worker_base_url, name="example", bearer_token="wrong-token")
