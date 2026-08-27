# Copyright 2026 Query Farm LLC - https://query.farm

"""Plain-Python-value -> `Arguments` conversion (`_arguments.py`).

Pure unit tests — no worker needed."""

from __future__ import annotations

import pyarrow as pa
import pytest

from vgi_polars._arguments import build_arguments, to_scalar


def test_to_scalar_none() -> None:
    result = to_scalar(None, pa.int64())
    assert result.is_valid is False
    assert result.type == pa.int64()


def test_to_scalar_plain_value_inferred_type() -> None:
    result = to_scalar(5)
    assert result.as_py() == 5
    assert result.type == pa.int64()


def test_to_scalar_plain_value_explicit_type() -> None:
    result = to_scalar(5, pa.int32())
    assert result.as_py() == 5
    assert result.type == pa.int32()


def test_to_scalar_existing_scalar_passthrough() -> None:
    existing = pa.scalar(7, type=pa.int64())
    assert to_scalar(existing) is existing


def test_to_scalar_existing_scalar_cast() -> None:
    existing = pa.scalar(7, type=pa.int64())
    result = to_scalar(existing, pa.int32())
    assert result.type == pa.int32()
    assert result.as_py() == 7


def test_build_arguments_positional_only() -> None:
    args = build_arguments([1, "x", None])
    assert args.positional[0].as_py() == 1
    assert args.positional[1].as_py() == "x"
    assert args.positional[2].is_valid is False
    assert args.named is None


def test_build_arguments_with_named() -> None:
    args = build_arguments([1], {"factor": 2})
    assert args.positional[0].as_py() == 1
    assert args.named["factor"].as_py() == 2


def test_build_arguments_with_types() -> None:
    args = build_arguments(
        [1],
        {"factor": 2},
        positional_types=[pa.int32()],
        named_types={"factor": pa.float64()},
    )
    assert args.positional[0].type == pa.int32()
    assert args.named["factor"].type == pa.float64()


def test_build_arguments_mismatched_types_length_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_arguments([1, 2], positional_types=[pa.int32()])
