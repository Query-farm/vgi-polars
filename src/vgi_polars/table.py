# Copyright 2026 Query Farm LLC - https://query.farm

"""`VgiTable` — a lazy handle to one catalog table.

Schema and scan-function resolution are cheap, scan-free unary catalog RPCs
(`table_get` / `table_scan_function_get`) — see `_source.py`'s module docstring
for why that matters for `register_io_source(schema=...)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
from vgi.catalog.catalog_interface import (
    ColumnStatistics,
    FunctionInfo,
    ScanFunctionResult,
    SchemaObjectType,
    TableInfo,
)

from vgi_polars._polars_compat import arrow_to_df
from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from vgi_polars.catalog import VgiCatalog

__all__ = ["VgiTable"]

# `Client.table_get` predates `at_unit`/`at_value` (see CLAUDE.md's "Time
# travel" section) — an installed vgi-python without the fix raises
# `TypeError: unexpected keyword argument` on ANY call if these are passed
# unconditionally, even a `None, None` no-op time-travel request. Computed
# once; `table_scan_function_get` has always accepted both, no guard needed
# there.
try:
    import inspect

    from vgi.client.catalog_mixin import CatalogClientMixin as _RuntimeCatalogMixin

    _SUPPORTS_TABLE_GET_AT_CLAUSE = "at_unit" in inspect.signature(_RuntimeCatalogMixin.table_get).parameters
except Exception:  # noqa: BLE001 - never let a capability probe break import
    _SUPPORTS_TABLE_GET_AT_CLAUSE = False


class VgiTable:
    """A lazy handle to one table in an attached VGI catalog.

    Construct via `VgiCatalog.table(schema_name, name)`, not directly.
    """

    def __init__(
        self,
        *,
        catalog: VgiCatalog,
        schema_name: str,
        name: str,
        at_unit: str | None = None,
        at_value: str | None = None,
    ) -> None:
        """Wrap a resolved `(schema_name, name)` in `catalog`. Use `VgiCatalog.table(...)`, not this directly."""
        self._catalog = catalog
        self.schema_name = schema_name
        self.name = name
        # Time travel: immutable per-instance, not a mutable setter — a
        # `VgiTable` at one AT clause and one at another are different views
        # of the table (possibly different schemas, per `versioned_data` in
        # vgi-python's own fixtures) and must not share the memoized
        # `_table_info`/`_scan_function` below. Get a table at a different
        # version via `VgiCatalog.table(..., at_unit=..., at_value=...)`
        # again, not by mutating this one. See CLAUDE.md's "Time travel"
        # section — both `Client.table_get`/`table_function` needed a
        # vgi-python fix to expose `at_unit`/`at_value` at all.
        self.at_unit = at_unit
        self.at_value = at_value
        self._table_info: TableInfo | None = None
        self._scan_function: ScanFunctionResult | None = None
        self._function_info: FunctionInfo | None | Any = _UNRESOLVED
        self._scan_function_schema: str | None = None
        self._scan_branches: list[Any] | None = None

    def _table_get(self) -> TableInfo:
        if self._table_info is None:
            if not _SUPPORTS_TABLE_GET_AT_CLAUSE and (self.at_unit is not None or self.at_value is not None):
                raise VgiPolarsError(
                    "time travel requires a newer vgi-python (Client.table_get predates at_unit/"
                    "at_value on the installed version) — see CLAUDE.md's Time travel section"
                )
            # `dict[str, Any]`, not the narrower `dict[str, str | None]` a bare
            # literal would infer — splatted against `table_get`'s many
            # keyword-only params, the narrow inference makes mypy conflate
            # this with an unrelated param (`transaction_opaque_data`) it
            # could also (but never actually does) receive via **at_kwargs.
            at_kwargs: dict[str, Any] = (
                {"at_unit": self.at_unit, "at_value": self.at_value} if _SUPPORTS_TABLE_GET_AT_CLAUSE else {}
            )
            try:
                info = self._catalog.client.table_get(
                    attach_opaque_data=self._catalog.attach_opaque_data,
                    schema_name=self.schema_name,
                    name=self.name,
                    **at_kwargs,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
            if info is None:
                raise VgiPolarsError(f"table not found: {self.schema_name}.{self.name}")
            self._table_info = info
        return self._table_info

    def _scan_function_get(self) -> ScanFunctionResult:
        if self._scan_function is None:
            try:
                self._scan_function = self._catalog.client.table_scan_function_get(
                    attach_opaque_data=self._catalog.attach_opaque_data,
                    schema_name=self.schema_name,
                    name=self.name,
                    at_unit=self.at_unit,
                    at_value=self.at_value,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
        return self._scan_function

    def _lookup_table_function(self, schema_name: str, function_name: str) -> FunctionInfo | None:
        try:
            infos = self._catalog.client.schema_contents(
                attach_opaque_data=self._catalog.attach_opaque_data,
                name=schema_name,
                type=SchemaObjectType.TABLE_FUNCTION,
            )
        except VGI_CLIENT_ERRORS:
            return None
        return next((i for i in infos if i.name == function_name), None)

    def _resolve_scan_function(self) -> None:
        """Resolve the `FunctionInfo` and the schema the scan function actually lives in.

        Resolves both the `FunctionInfo` (pushdown capability flags) and the
        **schema the scan function actually lives in** — which is not
        necessarily this table's own schema. A worker registers function
        names per schema and may reuse a name across schemas (observed live
        against `vgi-fixture-worker`: `data.filter_echo_table` resolves to a
        scan function that is only registered in schema `main`), so calling
        `Client.table_function(schema_name=self.schema_name, ...)`
        unconditionally is wrong in general.

        Mirrors the DuckDB C++ extension's own resolution order
        (`vgi_table_entry.cpp`): try the table's own schema first, then the
        catalog's default schema. If neither lists it (e.g. it's a
        DuckDB-native function like `read_parquet` a worker delegates to —
        out of scope for vgi-polars, see CLAUDE.md), fall back to the
        table's own schema as the best remaining guess; `FunctionInfo` stays
        `None` in that case, which just disables pushdown (safe — see
        Design Principle 1 in `_source.py`), not the scan itself.
        """
        if self._function_info is not _UNRESOLVED:
            return
        scan_fn = self._scan_function_get()

        info = self._lookup_table_function(self.schema_name, scan_fn.function_name)
        if info is not None:
            self._function_info = info
            self._scan_function_schema = self.schema_name
            return

        default_schema = self._catalog.default_schema
        if default_schema != self.schema_name:
            info = self._lookup_table_function(default_schema, scan_fn.function_name)
            if info is not None:
                self._function_info = info
                self._scan_function_schema = default_schema
                return

        self._function_info = None
        self._scan_function_schema = self.schema_name

    def _function_info_get(self) -> FunctionInfo | None:
        """The `FunctionInfo` for the resolved scan function, if discoverable.

        Used to check `projection_pushdown`/`filter_pushdown` opt-in flags.
        `None` means "couldn't find it, assume no pushdown support" — a safe
        default given `_source.py` always re-verifies locally regardless.
        """
        self._resolve_scan_function()
        return self._function_info

    def scan_function_schema(self) -> str:
        """The schema to call the resolved scan function in.

        See `_resolve_scan_function`'s docstring for why this can differ from
        `self.schema_name`.
        """
        self._resolve_scan_function()
        assert self._scan_function_schema is not None
        return self._scan_function_schema

    @property
    def arrow_schema(self) -> pa.Schema:
        """The table's schema as a `pyarrow.Schema` (no scan)."""
        return pa.ipc.read_schema(pa.py_buffer(self._table_get().columns))

    @property
    def schema(self) -> pl.Schema:
        """The table's schema as a `polars.Schema` (no scan)."""
        return arrow_to_df(self.arrow_schema.empty_table()).schema

    def _scan_branches_get(self) -> list[Any]:
        """The table's scan branches (memoized).

        One `ScanBranch` for an ordinary single-source table, more for a
        multi-branch one. Uses `Client.table_scan_branches_get`, which
        transparently falls back to wrapping `table_scan_function_get`'s
        single result as one branch for a worker that predates the branches
        RPC — so this is always safe to call, not just for tables known in
        advance to be multi-branch. See `_multi_branch.py`'s module docstring
        for why calling this unconditionally (rather than only when a table
        is already known to be multi-branch) matters: `table_scan_function_get`
        alone silently returns only the *first* branch of a multi-branch
        table, which was a real, silent correctness gap before this method
        existed.
        """
        if self._scan_branches is None:
            try:
                result = self._catalog.client.table_scan_branches_get(
                    attach_opaque_data=self._catalog.attach_opaque_data,
                    schema_name=self.schema_name,
                    name=self.name,
                    at_unit=self.at_unit,
                    at_value=self.at_value,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
            self._scan_branches = list(result.branches)
        return self._scan_branches

    def scan(
        self,
        *,
        storage_options: dict[str, str] | None = None,
        acknowledge_required_filters: bool = False,
        secrets: dict[str, Any] | None = None,
    ) -> pl.LazyFrame:
        """A lazy, pushdown-aware scan of this table.

        `secrets`: pre-resolved secret values (`{name: value}`, value may be
        a nested dict for a struct-typed secret) forwarded to the worker on
        every `table_function`/`table_function_plan` RPC this scan issues —
        the table-scan analogue of `scalar_function`'s/`aggregate_function`'s
        own `secrets=` kwarg (see their docstrings), previously a real
        vgi-python API gap: `Client.table_function` hardcoded `secrets=None`
        despite the exchange-mode methods already exposing it. Requires
        vgi-python with the fix (`Client.table_function(secrets=...)`); an
        older install raises `VgiPolarsError` rather than silently dropping
        it. Skips the result cache when given (see `_source.py`), since the
        cache key has no secrets dimension. Not applied to a natively-
        delegated scan (`read_parquet`-style) — that path never calls the
        worker at all, so there is no RPC to attach secrets to; use
        `storage_options` for that case instead.

        Transparently handles a multi-branch table (`pl.concat` of one scan
        per branch, each branch's `branch_filter` applied — see
        `_multi_branch.py`) — the common, single-branch case takes the
        unchanged single-scan path with no multi-branch overhead beyond the
        one extra (cheap, memoized, unary) `table_scan_branches_get` catalog
        call. `hasattr` guards against an installed vgi-python that predates
        `Client.table_scan_branches_get` (see CLAUDE.md's "Multi-branch
        tables" section) — an older pinned release falls back to the
        single-scan path via the legacy `table_scan_function_get` (branch 0
        only for a genuinely multi-branch table — the same silent limitation
        this feature fixes, but no worse than before this method existed,
        and never an `AttributeError`).

        **Native scan-function delegation** (`_native_scan.py`): when the
        resolved scan function names something Polars can satisfy directly
        (currently just `read_parquet` -> `pl.scan_parquet`) rather than a
        VGI-hosted function, this returns that native `LazyFrame` straight
        away — no `register_io_source`, no worker round-trip for the data at
        all. `storage_options` is a plain passthrough for that path (cloud
        credentials/region are genuinely out-of-band on the wire, the same
        way DuckDB's own reference worker needs a separate `SET
        s3_region=...`). If the table also declares `required_filters`, this
        raises `VgiPolarsError` unless `acknowledge_required_filters=True` —
        the io_source path's cost-safety check has no equivalent hook here
        (see `_native_scan.py`'s module docstring for why), so refusing by
        default beats silently dropping a real safety guard.
        """
        if hasattr(self._catalog.client, "table_scan_branches_get"):
            branches = self._scan_branches_get()
            if len(branches) != 1:
                from vgi_polars._multi_branch import scan_multi_branch

                return scan_multi_branch(self, branches, secrets=secrets)

        from vgi_polars._native_scan import NATIVE_SCAN_HANDLERS

        scan_fn = self._scan_function_get()
        native_handler = NATIVE_SCAN_HANDLERS.get(scan_fn.function_name)
        if native_handler is not None:
            if secrets is not None:
                raise VgiPolarsError(
                    f"{self.schema_name}.{self.name}: natively delegates to "
                    f"{scan_fn.function_name!r}, which never calls the worker — secrets has no "
                    "RPC to attach to and would be silently dropped. Use storage_options for "
                    "native-scan credentials instead."
                )
            required = self.required_filters()
            if required and not acknowledge_required_filters:
                raise VgiPolarsError(
                    f"{self.schema_name}.{self.name}: natively delegates to "
                    f"{scan_fn.function_name!r} and declares required_filters {required} that "
                    "vgi-polars cannot enforce for a native scan (no hook to inspect the "
                    "eventual predicate before .collect()). Pass scan(acknowledge_required_"
                    "filters=True) once you've applied the equivalent filter(s) yourself, or "
                    "you WILL trigger a full, possibly enormous, unfiltered remote read."
                )
            return native_handler(
                scan_fn,
                schema_name=self.schema_name,
                table_name=self.name,
                storage_options=storage_options,
            )

        from polars.io.plugins import register_io_source

        from vgi_polars._source import make_io_source

        arrow_schema = self.arrow_schema
        return register_io_source(make_io_source(self, arrow_schema, secrets=secrets), schema=self.schema)

    def read(
        self,
        *,
        storage_options: dict[str, str] | None = None,
        acknowledge_required_filters: bool = False,
        secrets: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """An eager, full scan of this table. See `scan()` for the parameters."""
        return self.scan(
            storage_options=storage_options,
            acknowledge_required_filters=acknowledge_required_filters,
            secrets=secrets,
        ).collect()

    def statistics(self) -> list[ColumnStatistics]:
        """Per-column statistics (min/max/null presence/distinct count/...), if the worker advertises them.

        A plain catalog-metadata RPC, no scan. Returns vgi-python's own
        `ColumnStatistics` dataclass directly rather than reinventing an
        equivalent; `pa.Scalar`-typed `.min`/`.max`, call `.as_py()` for a
        plain Python value.
        """
        try:
            return self._catalog.client.table_column_statistics(
                attach_opaque_data=self._catalog.attach_opaque_data,
                schema_name=self.schema_name,
                name=self.name,
            )
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e

    def required_filters(self) -> list[list[str]]:
        """AND-of-OR-groups of column names a scan predicate must reference at least one of, per group.

        Purely declarative on the wire (`TableInfo.required_filters`);
        vgi-python does no enforcement itself (by design, the DuckDB C++
        extension's optimizer does it there). `_source.py` enforces it here,
        before scanning, as a cost-safety guard: without it,
        `.scan().collect()` with no matching filter on a `required_filters`
        table would trigger a full, possibly enormous, unfiltered remote scan
        — Design Principle 1 keeps that *correct*, not *safe from accidental
        cost*.
        """
        return list(self._table_get().required_filters)


class _Unresolved:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<unresolved>"


_UNRESOLVED = _Unresolved()
