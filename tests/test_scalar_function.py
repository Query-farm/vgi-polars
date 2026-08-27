# Copyright 2026 Query Farm LLC - https://query.farm

"""Scalar function calls via the `pl.Expr.map_batches` bridge — including the
array-argument + constant-argument mix (`main.multiply(value, factor)`:
`value` is a per-row array param, `factor` is a `ConstParam` bound at call
time, not exchanged per row — see `_scalar.py`'s module docstring)."""

from __future__ import annotations

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars.errors import VgiPolarsError


def test_multiply_array_and_const_arg(catalog: vp.VgiCatalog) -> None:
    multiply = catalog.scalar_function("main", "multiply")
    df = pl.DataFrame({"value": [1, 2, 3, 4]})
    out = df.with_columns(multiply(pl.col("value"), 2).alias("product"))
    assert out["product"].to_list() == [2, 4, 6, 8]


def test_multiply_wrong_arg_count_raises(catalog: vp.VgiCatalog) -> None:
    multiply = catalog.scalar_function("main", "multiply")
    with pytest.raises(VgiPolarsError, match="expects 2 argument"):
        multiply(pl.col("value"))


def test_multiply_const_arg_as_expr_raises(catalog: vp.VgiCatalog) -> None:
    multiply = catalog.scalar_function("main", "multiply")
    with pytest.raises(VgiPolarsError, match="constant parameter"):
        multiply(pl.col("value"), pl.col("value"))


def test_unknown_scalar_function_raises(catalog: vp.VgiCatalog) -> None:
    fn = catalog.scalar_function("main", "does_not_exist_xyz")
    with pytest.raises(VgiPolarsError, match="not found"):
        fn(pl.col("value"))


def test_secrets_kwarg_reaches_the_exchange_call(catalog: vp.VgiCatalog, monkeypatch) -> None:
    """`secrets=` on the returned callable threads straight through to
    `Client.scalar_function`'s own `secrets` parameter. No existing
    vgi-fixture-worker scalar function combines a `Secret()` param with a
    regular array param (the two that use `Secret()` at all,
    `return_secret_value`/`secret_field`, take zero array args — a call shape
    vgi-polars' bridge doesn't support at all, independent of secrets), so
    this verifies the wiring directly rather than forcing an awkward fixture
    match; see `_scalar.py`'s module docstring for the confirmed API gap."""
    multiply = catalog.scalar_function("main", "multiply")

    exchange_client = catalog._exchange_client()
    real_scalar_function = exchange_client.scalar_function
    seen_secrets = []

    def spying_scalar_function(*, secrets=None, **kwargs):
        seen_secrets.append(secrets)
        return real_scalar_function(secrets=secrets, **kwargs)

    monkeypatch.setattr(exchange_client, "scalar_function", spying_scalar_function)

    df = pl.DataFrame({"value": [1, 2]})
    my_secret = {"vgi_example": {"secret_string": "s3cr3t"}}
    out = df.with_columns(multiply(pl.col("value"), 2, secrets=my_secret).alias("product"))

    assert out["product"].to_list() == [2, 4]
    assert seen_secrets == [my_secret]
