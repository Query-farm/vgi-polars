# Copyright 2026 Query Farm LLC - https://query.farm

"""Wire-robustness regression tests.

A worker that violates the protocol (incompatible version, an
unrecognized wire-enum value) must surface as a clean `VgiPolarsError`,
never a hang or a raw traceback leaking vgi-python's own exception types.
Mirrors the `protocol_version/` and `bad_enum.test` categories of
`~/Development/vgi`'s sqllogictest suite (protocol-general behavior, not
DuckDB-specific) and vgi-python's own
`tests/conformance/test_protocol_version.py`.
"""

from __future__ import annotations

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars.errors import VgiPolarsError


def test_incompatible_protocol_version_raises_at_attach(bad_protocol_worker_location: str) -> None:
    """The version mismatch is enforced at every RPC dispatch boundary.

    So it surfaces immediately on `catalog_attach` — before any
    scan/scalar call.
    """
    with pytest.raises(VgiPolarsError, match="protocol_version"):
        vp.attach(bad_protocol_worker_location, name="example")


def test_unrecognized_wire_enum_raises_cleanly(bad_enum_worker_location: str) -> None:
    """`double`'s corrupted `null_handling` enum value fails catalog-metadata parsing.

    Fails `schema_contents` parsing the moment its FunctionInfo is loaded
    — triggered here by resolving the scalar-function bridge, which
    fetches FunctionInfo on first call.
    """
    with vp.attach(bad_enum_worker_location, name="example") as cat:
        double = cat.scalar_function("main", "double")
        with pytest.raises(VgiPolarsError, match="NullHandling"):
            df = pl.DataFrame({"value": [1, 2, 3]})
            df.with_columns(double(pl.col("value")).alias("doubled"))


def test_unrecognized_wire_enum_poisons_the_whole_schema_listing(bad_enum_worker_location: str) -> None:
    """Surprising, worth documenting rather than assuming otherwise.

    Only `double`'s metadata is corrupted, but `schema_contents(type=
    SCALAR_FUNCTION)` fetches every scalar function in a schema in ONE bulk
    RPC (`vgi.client.catalog_mixin`) and vgi-python deserializes the whole
    response together — so a single corrupted function's metadata fails
    the ENTIRE listing, taking down resolution of `multiply` too even
    though `multiply` itself is perfectly healthy. Not a vgi-polars bug;
    inherited directly from vgi-python's bulk catalog-RPC granularity. If
    this ever changes upstream (a per-function metadata RPC), this test
    should start failing and needs updating, not silently keep "passing"
    against a changed assumption.
    """
    with vp.attach(bad_enum_worker_location, name="example") as cat:
        multiply = cat.scalar_function("main", "multiply")
        with pytest.raises(VgiPolarsError, match="NullHandling"):
            df = pl.DataFrame({"value": [1, 2, 3]})
            df.with_columns(multiply(pl.col("value"), 2).alias("product"))
