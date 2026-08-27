# Copyright 2026 Query Farm LLC - https://query.farm

"""Catalog-versioning introspection (`VgiCatalog.catalog_version`,
`resolved_data_version`, `resolved_implementation_version`, ...) and
attach-option/version validation — against the `versioned` fixture catalog
(`vgi/_test_fixtures/versioned.py`), which validates `data_version_spec`/
`implementation_version` at ATTACH time and echoes back what it resolved.

"Attach-option validation" itself needs no new vgi-polars code: `attach()`
already threads `options`/`data_version_spec`/`implementation_version`
straight through to `Client.catalog_attach`, and a worker-side rejection
(unsatisfiable version, unknown catalog, ...) already surfaces as a clean
`VgiPolarsError` via the existing `VGI_CLIENT_ERRORS` wrapping — the tests
below confirm that behavior end to end rather than adding a redundant
client-side re-validation layer (VGI's design has the *worker* validate its
own options; see CLAUDE.md's Scope section)."""

from __future__ import annotations

import pytest

import vgi_polars as vp
from vgi_polars.errors import VgiPolarsError


def test_catalog_version_is_exposed(catalog: vp.VgiCatalog) -> None:
    assert isinstance(catalog.catalog_version, int)


def test_versioned_attach_with_no_spec_resolves_default(versioned_worker_location: str) -> None:
    with vp.attach(versioned_worker_location, name="versioned") as cat:
        assert cat.resolved_data_version == "1.2.0"  # DEFAULT_DATA_VERSION
        # This fixture always echoes its own IMPLEMENTATION_VERSION, even
        # when the client didn't request one — not every worker will (the
        # protocol allows `None` for "no opinion"), so this only asserts the
        # value round-trips faithfully, not that a specific fixture's choice
        # to always echo is the general contract.
        assert cat.resolved_implementation_version == "1.0.0"
        assert cat.supports_time_travel is False
        assert cat.supports_transactions is False
        assert cat.catalog_version_frozen is True
        assert cat.comment == "Example catalog demonstrating data_version_spec validation and cookie stickiness"


def test_versioned_attach_with_satisfiable_spec_resolves_it(versioned_worker_location: str) -> None:
    with vp.attach(
        versioned_worker_location,
        name="versioned",
        data_version_spec="1.1.0",
        implementation_version="1.0.0",
    ) as cat:
        assert cat.resolved_data_version == "1.1.0"
        assert cat.resolved_implementation_version == "1.0.0"


def test_versioned_attach_with_unsatisfiable_data_version_raises(versioned_worker_location: str) -> None:
    with pytest.raises(VgiPolarsError, match="Unsupported data_version_spec"):
        vp.attach(versioned_worker_location, name="versioned", data_version_spec="9.9.9")


def test_versioned_attach_with_unsatisfiable_implementation_version_raises(versioned_worker_location: str) -> None:
    with pytest.raises(VgiPolarsError, match="Unsupported implementation_version"):
        vp.attach(versioned_worker_location, name="versioned", implementation_version="9.9.9")


def test_attach_with_unknown_catalog_name_raises(versioned_worker_location: str) -> None:
    """Same worker binary, wrong attach `name` — the worker rejects it and the
    failure surfaces cleanly, exactly like an unrecognized attach option
    would (this fixture happens not to validate options; the mechanism that
    surfaces a rejection is identical either way, see module docstring)."""
    with pytest.raises(VgiPolarsError, match="Unknown catalog"):
        vp.attach(versioned_worker_location, name="no_such_catalog")


def test_unrecognized_attach_option_does_not_crash(worker_location: str) -> None:
    """The `example` fixture worker (unlike `versioned`) doesn't validate its
    options at all — an unrecognized one is silently ignored, a valid worker
    choice VGI's design permits (the worker owns option validation, not the
    client). Confirms vgi-polars doesn't second-guess that by rejecting
    client-side."""
    with vp.attach(worker_location, name="example", options={"no_such_option_xyz": "value"}) as cat:
        assert "data" in cat.schemas()
