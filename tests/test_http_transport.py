# Copyright 2026 Query Farm LLC - https://query.farm

"""The core catalog/scan/scalar paths over HTTP transport, mirroring the
subprocess-transport tests — VGI's client-side surface is transport-agnostic,
so these are deliberately a small subset (smoke coverage), not a duplicate of
the full subprocess suite."""

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


def test_bearer_auth_with_correct_token(http_bearer_worker_base_url: str, http_bearer_token: str) -> None:
    with vp.attach(http_bearer_worker_base_url, name="example", bearer_token=http_bearer_token) as cat:
        assert "data" in cat.schemas()


def test_bearer_auth_without_token_rejected(http_bearer_worker_base_url: str) -> None:
    with pytest.raises(VgiPolarsError):
        vp.attach(http_bearer_worker_base_url, name="example")


def test_bearer_auth_with_wrong_token_rejected(http_bearer_worker_base_url: str) -> None:
    with pytest.raises(VgiPolarsError):
        vp.attach(http_bearer_worker_base_url, name="example", bearer_token="wrong-token")
