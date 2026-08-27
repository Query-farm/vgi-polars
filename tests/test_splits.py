# Copyright 2026 Query Farm LLC - https://query.farm

"""Split-aware scanning (`_source.py`'s `_iter_splits_sequential` +
`function_info.supports_splits` gate) — against the real `split_sequence`/
`split_zero` fixtures (`vgi/_test_fixtures/table/splits.py`), which needed
new upstream `vgi.client.Client` methods (`table_function_plan`,
`table_function(split_tokens=...)`) that didn't exist before this: splits
were entirely unreachable from `Client`, not just unsupported by vgi-polars.
See CLAUDE.md's "Splits" section.

Neither `split_sequence` nor `split_zero` is wrapped in a catalog `Table`
entry (they're bare functions used by the DuckDB extension's own multi-branch/
splits SQL suite) — mirrors `test_errors.py`'s `_FakeTableForScanFunction`
pattern: a duck-typed stand-in implementing only what `make_io_source` calls
on `table`, resolving a real `FunctionInfo` from the catalog rather than a
mock, since the point here is exercising the real `supports_splits` flag."""

from __future__ import annotations

from typing import Any

import polars as pl
import pyarrow as pa
import pytest
from vgi.arguments import Arguments
from vgi.catalog.catalog_interface import ScanFunctionResult, SchemaObjectType
from vgi.client.client import Client

import vgi_polars as vp
from vgi_polars._source import make_io_source

# `table_function_plan` is a new upstream addition (see module docstring) —
# an installed vgi-python that predates it (e.g. a pin not yet bumped past
# the release that includes it) can't run any test here that needs REAL
# split RPC support. Skipped, not failed: CI stays green against the current
# pin and these tests start running for real the moment it's bumped, with no
# further vgi-polars change needed. `test_missing_table_function_plan_falls_
# back_cleanly` is the one exception — it's written to behave correctly
# whether or not the capability is present, so it always runs.
_HAS_SPLIT_SUPPORT = hasattr(Client, "table_function_plan")
requires_split_support = pytest.mark.skipif(
    not _HAS_SPLIT_SUPPORT,
    reason="installed vgi-python predates Client.table_function_plan",
)


class _FakeSplitTable:
    def __init__(
        self,
        catalog: vp.VgiCatalog,
        function_name: str,
        named_arguments: dict[str, Any] | None = None,
        positional_arguments: list[Any] | None = None,
    ) -> None:
        self._catalog = catalog
        self.schema_name = "main"
        self.name = function_name
        self.at_unit = None
        self.at_value = None
        self._fn = function_name
        self._named_arguments = named_arguments or {}
        self._positional_arguments = positional_arguments or []

    def _scan_function_get(self) -> ScanFunctionResult:
        return ScanFunctionResult(
            function_name=self._fn,
            positional_arguments=self._positional_arguments,
            named_arguments=self._named_arguments,
        )

    def _function_info_get(self):
        infos = self._catalog.client.schema_contents(
            attach_opaque_data=self._catalog.attach_opaque_data,
            name="main",
            type=SchemaObjectType.TABLE_FUNCTION,
        )
        return next(i for i in infos if i.name == self._fn)

    def scan_function_schema(self) -> str:
        return "main"

    def required_filters(self) -> list[list[str]]:
        return []


def _args(**kwargs: int) -> dict[str, Any]:
    return {k: pa.scalar(v) for k, v in kwargs.items()}


def test_function_info_reports_supports_splits(catalog: vp.VgiCatalog) -> None:
    t = _FakeSplitTable(catalog, "split_sequence", _args(n=5, splits=2))
    assert t._function_info_get().supports_splits is True

    plain = _FakeSplitTable(catalog, "sequence", {})
    assert plain._function_info_get().supports_splits is False


@requires_split_support
def test_split_scan_matches_plain_sequence(catalog: vp.VgiCatalog) -> None:
    t = _FakeSplitTable(catalog, "split_sequence", _args(n=23, splits=4))
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))
    dfs = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    rows = sorted(v for df in dfs for v in df["n"].to_list())
    assert rows == list(range(23))


@requires_split_support
def test_split_zero_yields_no_rows(catalog: vp.VgiCatalog) -> None:
    t = _FakeSplitTable(catalog, "split_zero", _args(n=10, splits=4))
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))
    dfs = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    assert dfs == []


@requires_split_support
def test_split_scan_respects_local_predicate(catalog: vp.VgiCatalog) -> None:
    """Design Principle 1 holds across splits too — the local re-filter
    applies uniformly regardless of which split a row came from."""
    t = _FakeSplitTable(catalog, "split_sequence", _args(n=20, splits=5))
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))
    dfs = list(io_source(with_columns=None, predicate=pl.col("n") >= 15, n_rows=None, batch_size=None))
    rows = sorted(v for df in dfs for v in df["n"].to_list())
    assert rows == [15, 16, 17, 18, 19]


@requires_split_support
def test_split_scan_respects_n_rows_across_split_boundary(catalog: vp.VgiCatalog) -> None:
    """`n_rows` truncation must stop the *whole* multi-split scan early, not
    just the batch it happens to land in — `_RemainingBudget` is shared
    across every split's `table_function` call for exactly this reason."""
    t = _FakeSplitTable(catalog, "split_sequence", _args(n=40, splits=8))  # 5 rows/split
    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))
    dfs = list(io_source(with_columns=None, predicate=None, n_rows=7, batch_size=None))
    rows = [v for df in dfs for v in df["n"].to_list()]
    assert len(rows) == 7


@requires_split_support
def test_non_split_function_never_calls_plan(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    """`supports_splits=False` (the plain `sequence` function) must take the
    ordinary whole-scan path — `table_function_plan` is never called."""
    t = _FakeSplitTable(catalog, "sequence", positional_arguments=[pa.scalar(5)])
    exchange_client = catalog._exchange_client()
    calls: list[Any] = []
    real_plan = exchange_client.table_function_plan

    def spying_plan(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(exchange_client, "table_function_plan", spying_plan)

    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))
    dfs = list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))
    rows = sorted(v for df in dfs for v in df["n"].to_list())
    assert rows == list(range(5))
    assert calls == []


def test_missing_table_function_plan_falls_back_cleanly(catalog: vp.VgiCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates an installed vgi-python that predates `Client.
    table_function_plan` (e.g. this repo's own CI, pinned to a released tag
    until it's bumped past one that includes it — see CLAUDE.md's "Splits"
    section). The `hasattr` guard in `_source.py` takes the ordinary
    whole-scan path instead of raising `AttributeError` — every fixture in
    this module is deliberately split-*only* (see
    `vgi/_test_fixtures/table/splits.py`'s `_SplitBase.initial_state`), so
    that whole-scan attempt is correctly refused by the *worker* with its own
    clear message, not silently wrong data and not a client-side
    `AttributeError`.

    Deliberately NOT gated by `requires_split_support` — it's written to hold
    under both an installed vgi-python that already has `table_function_plan`
    (simulated absence via `monkeypatch.delattr`) and one that genuinely
    doesn't (nothing to patch — the scenario under test is already the
    ambient reality), so it always runs and is the one test in this module
    that proves the graceful-degradation path itself."""
    t = _FakeSplitTable(catalog, "split_sequence", _args(n=9, splits=3))
    exchange_client = catalog._exchange_client()
    if _HAS_SPLIT_SUPPORT:
        monkeypatch.delattr(type(exchange_client), "table_function_plan")

    io_source = make_io_source(t, pa.schema([pa.field("n", pa.int64())]))
    with pytest.raises(vp.VgiPolarsError, match="split-only"):
        list(io_source(with_columns=None, predicate=None, n_rows=None, batch_size=None))


@requires_split_support
def test_split_scan_end_to_end_via_arguments_helper(catalog: vp.VgiCatalog) -> None:
    """Sanity check that the fixture's own arg convention (named, via
    `Arguments`) matches what `_FakeSplitTable`/`ScanFunctionResult` build —
    catches an accidental positional/named mismatch independent of the
    io_source machinery."""
    args = Arguments(named=_args(n=6, splits=2))
    plan = catalog._exchange_client().table_function_plan(
        function_name="split_sequence", schema_name="main", arguments=args
    )
    assert len(plan.splits) == 2
