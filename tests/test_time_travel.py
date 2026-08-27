# Copyright 2026 Query Farm LLC - https://query.farm

"""Time travel via `VgiCatalog.table(..., at_unit=..., at_value=...)`.

Exercised against the real `data.versioned_data` fixture (`vgi/_test_fixtures/table/
versioned.py`), which needed a vgi-python fix first: `Client.table_get`/
`Client.table_function` didn't accept `at_unit`/`at_value` at all, even
though the underlying wire RPCs always carried them and
`Client.table_scan_function_get` already exposed them. See CLAUDE.md's
"Time travel" section.
"""

from __future__ import annotations

import inspect

import pytest
from vgi.client.catalog_mixin import CatalogClientMixin

import vgi_polars as vp

# `Client.table_get(at_unit=..., at_value=...)` is a new upstream addition
# (see this module's docstring). Skipped, not failed, against an older
# installed vgi-python — mirrors `test_splits.py`'s `requires_split_support`
# pattern.
requires_time_travel = pytest.mark.skipif(
    "at_unit" not in inspect.signature(CatalogClientMixin.table_get).parameters,
    reason="installed vgi-python predates Client.table_get(at_unit=..., at_value=...)",
)


@requires_time_travel
def test_schema_at_a_past_version_differs_from_live(catalog: vp.VgiCatalog) -> None:
    v1 = catalog.table("data", "versioned_data", at_unit="VERSION", at_value="1")
    live = catalog.table("data", "versioned_data")
    assert v1.schema.names() == ["id"]  # version 1: (id int64) only
    assert live.schema.names() == ["id", "score"]  # current: version 3


@requires_time_travel
def test_scan_at_a_past_version_returns_that_versions_rows(catalog: vp.VgiCatalog) -> None:
    v1 = catalog.table("data", "versioned_data", at_unit="VERSION", at_value="1")
    out = v1.scan().collect()
    assert sorted(out["id"].to_list()) == [1, 2, 3]  # version 1 data: 3 rows
    assert out.columns == ["id"]


@requires_time_travel
def test_scan_live_is_unaffected_by_a_separately_requested_version(catalog: vp.VgiCatalog) -> None:
    """Two different `VgiTable` instances (one versioned, one live) never share memoized state.

    Each `VgiTable` resolves and scans independently, with its own schema/scan-function state.
    """
    v1 = catalog.table("data", "versioned_data", at_unit="VERSION", at_value="1")
    live = catalog.table("data", "versioned_data")

    v1_out = v1.scan().collect()
    live_out = live.scan().collect()

    assert sorted(v1_out["id"].to_list()) == [1, 2, 3]
    assert sorted(live_out["id"].to_list()) == [1, 2, 3, 4]  # version 3 data: 4 rows
    assert live_out.columns == ["id", "score"]


def test_no_at_clause_defaults_to_none(catalog: vp.VgiCatalog) -> None:
    t = catalog.table("data", "versioned_data")
    assert t.at_unit is None
    assert t.at_value is None
