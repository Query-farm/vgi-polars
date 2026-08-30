# Copyright 2026 Query Farm LLC - https://query.farm

"""`secrets=` for table scans.

`Client.table_function` hardcoded `secrets=None` on the wire until the
companion vgi-python fix (see CLAUDE.md's Scope section) -- before that,
`secrets` only worked for `scalar_function`/`aggregate_function`. This
exercises `VgiTable.scan(secrets=...)`/`make_io_source(secrets=...)` against
`secret_demo` (vgi-python's `vgi/_test_fixtures/table/settings.py`), a bare
table function not wrapped in a catalog `Table` entry -- same
`_FakeSplitTable`/`_FakeTableForScanFunction` duck-typed stand-in pattern
`test_splits.py`/`test_errors.py` already use for exactly this reason.
"""

from __future__ import annotations

import inspect

import pyarrow as pa
import pytest
from vgi.catalog.catalog_interface import FunctionInfo, ScanFunctionResult, SchemaObjectType
from vgi.client.client import Client

import vgi_polars as vp
from vgi_polars._source import make_io_source
from vgi_polars.errors import VgiPolarsError

# `table_function(secrets=...)` is a new upstream addition (vgi-python
# >=0.31.2) -- an installed vgi-python that predates it can't run the
# happy-path test. Skipped, not failed, mirroring test_splits.py's
# `requires_split_support`.
_HAS_SECRETS_SUPPORT = "secrets" in inspect.signature(Client.table_function).parameters
requires_table_function_secrets = pytest.mark.skipif(
    not _HAS_SECRETS_SUPPORT,
    reason="installed vgi-python predates Client.table_function(secrets=...)",
)

_SECRET_SCHEMA = pa.schema(
    [pa.field("key", pa.string()), pa.field("value", pa.string()), pa.field("arrow_type", pa.string())]
)
# SecretDemoFunction.initial_state resolves via `params.secrets.of_type("vgi_example")`,
# which filters on the secret payload's own `type` field -- must equal the
# requested secret_type, distinct from the secret dict's *key* below.
_SECRET_VALUE = {"type": "vgi_example", "provider": "config", "secret_string": "s3cr3t"}


class _FakeSecretTable:
    def __init__(self, catalog: vp.VgiCatalog, function_name: str = "secret_demo") -> None:
        self._catalog = catalog
        self.schema_name = "main"
        self.name = function_name
        self.at_unit = None
        self.at_value = None
        self._fn = function_name

    def _scan_function_get(self) -> ScanFunctionResult:
        return ScanFunctionResult(function_name=self._fn, positional_arguments=[], named_arguments={})

    def _function_info_get(self) -> FunctionInfo | None:
        infos = self._catalog.client.schema_contents(
            attach_opaque_data=self._catalog.attach_opaque_data, name="main", type=SchemaObjectType.TABLE_FUNCTION
        )
        return next((i for i in infos if i.name == self._fn), None)

    def scan_function_schema(self) -> str:
        return "main"

    def required_filters(self) -> list[list[str]]:
        return []


@requires_table_function_secrets
def test_secret_reaches_table_function(catalog: vp.VgiCatalog) -> None:
    """A client-supplied secret reaches secret_demo's process() via a plain table scan."""
    t = _FakeSecretTable(catalog)
    io_source = make_io_source(t, _SECRET_SCHEMA, secrets={"vgi_example": _SECRET_VALUE})
    dfs = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    rows = {row["key"]: row["value"] for df in dfs for row in df.to_dicts()}
    assert rows == {"type": "vgi_example", "provider": "config", "secret_string": "s3cr3t"}


def test_no_secret_yields_no_rows(catalog: vp.VgiCatalog) -> None:
    """Without a secret, secret_demo's process() emits nothing (state.keys is empty)."""
    t = _FakeSecretTable(catalog)
    io_source = make_io_source(t, _SECRET_SCHEMA)
    dfs = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    assert dfs == []


def test_secrets_raises_on_old_vgi_python(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pinned vgi-python predating table_function(secrets=...) fails loudly, not silently drops it."""
    import vgi_polars._source as source_module

    monkeypatch.setattr(source_module, "_SUPPORTS_TABLE_FUNCTION_SECRETS", False)
    t = _FakeSecretTable(catalog)
    io_source = make_io_source(t, _SECRET_SCHEMA, secrets={"vgi_example": _SECRET_VALUE})
    with pytest.raises(VgiPolarsError, match="secrets requires a newer vgi-python"):
        list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))


def test_scan_rejects_secrets_for_native_delegation(catalog: vp.VgiCatalog) -> None:
    """secrets= on a natively-delegated scan (no worker RPC to attach it to) raises rather than silently dropping it.

    `data.rff_parquet` is the real native-delegation fixture `test_native_scan.py`'s
    `TestRffParquetIntegration` already exercises end to end (resolves to
    `read_parquet`, bypassing the worker entirely).
    """
    t = catalog.table("data", "rff_parquet")
    with pytest.raises(VgiPolarsError, match="never calls the worker"):
        t.scan(secrets={"vgi_example": _SECRET_VALUE})
