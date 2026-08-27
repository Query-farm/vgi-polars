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
    """A lazy handle to one table in an attached VGI catalog. Construct via
    `VgiCatalog.table(schema_name, name)`, not directly."""

    def __init__(
        self,
        *,
        catalog: VgiCatalog,
        schema_name: str,
        name: str,
        at_unit: str | None = None,
        at_value: str | None = None,
    ) -> None:
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
            at_kwargs = {"at_unit": self.at_unit, "at_value": self.at_value} if _SUPPORTS_TABLE_GET_AT_CLAUSE else {}
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
        """Resolve both the `FunctionInfo` (pushdown capability flags) and the
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
        """The `FunctionInfo` for the resolved scan function, if discoverable —
        used to check `projection_pushdown`/`filter_pushdown` opt-in flags.
        `None` means "couldn't find it, assume no pushdown support" — a safe
        default given `_source.py` always re-verifies locally regardless."""
        self._resolve_scan_function()
        return self._function_info

    def scan_function_schema(self) -> str:
        """The schema to call the resolved scan function in — see
        `_resolve_scan_function`'s docstring for why this can differ from
        `self.schema_name`."""
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
        return pl.from_arrow(self.arrow_schema.empty_table()).schema

    def _scan_branches_get(self) -> list[Any]:
        """The table's scan branches (memoized) — one `ScanBranch` for an
        ordinary single-source table, more for a multi-branch one. Uses
        `Client.table_scan_branches_get`, which transparently falls back to
        wrapping `table_scan_function_get`'s single result as one branch for
        a worker that predates the branches RPC — so this is always safe to
        call, not just for tables known in advance to be multi-branch. See
        `_multi_branch.py`'s module docstring for why calling this
        unconditionally (rather than only when a table is already known to
        be multi-branch) matters: `table_scan_function_get` alone silently
        returns only the *first* branch of a multi-branch table, which was a
        real, silent correctness gap before this method existed."""
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

    def scan(self) -> pl.LazyFrame:
        """A lazy, pushdown-aware scan of this table. Transparently handles a
        multi-branch table (`pl.concat` of one scan per branch, each branch's
        `branch_filter` applied — see `_multi_branch.py`) — the common,
        single-branch case takes the unchanged single-scan path with no
        multi-branch overhead beyond the one extra (cheap, memoized, unary)
        `table_scan_branches_get` catalog call. `hasattr` guards against an
        installed vgi-python that predates `Client.table_scan_branches_get`
        (see CLAUDE.md's "Multi-branch tables" section) — an older pinned
        release falls back to the single-scan path via the legacy
        `table_scan_function_get` (branch 0 only for a genuinely multi-branch
        table — the same silent limitation this feature fixes, but no worse
        than before this method existed, and never an `AttributeError`)."""
        if hasattr(self._catalog.client, "table_scan_branches_get"):
            branches = self._scan_branches_get()
            if len(branches) != 1:
                from vgi_polars._multi_branch import scan_multi_branch

                return scan_multi_branch(self, branches)

        from polars.io.plugins import register_io_source

        from vgi_polars._source import make_io_source

        arrow_schema = self.arrow_schema
        return register_io_source(make_io_source(self, arrow_schema), schema=self.schema)

    def read(self) -> pl.DataFrame:
        """An eager, full scan of this table."""
        return self.scan().collect()

    def statistics(self) -> list[ColumnStatistics]:
        """Per-column statistics (min/max/null presence/distinct count/...),
        if the worker advertises them — a plain catalog-metadata RPC, no scan.
        Returns vgi-python's own `ColumnStatistics` dataclass directly rather
        than reinventing an equivalent; `pa.Scalar`-typed `.min`/`.max`, call
        `.as_py()` for a plain Python value."""
        try:
            return self._catalog.client.table_column_statistics(
                attach_opaque_data=self._catalog.attach_opaque_data,
                schema_name=self.schema_name,
                name=self.name,
            )
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e

    def required_filters(self) -> list[list[str]]:
        """AND-of-OR-groups of column names a scan predicate must reference
        at least one of, per group — purely declarative on the wire (`TableInfo.
        required_filters`); vgi-python does no enforcement itself (by design,
        the DuckDB C++ extension's optimizer does it there). `_source.py`
        enforces it here, before scanning, as a cost-safety guard: without it,
        `.scan().collect()` with no matching filter on a `required_filters`
        table would trigger a full, possibly enormous, unfiltered remote scan
        — Design Principle 1 keeps that *correct*, not *safe from accidental
        cost*."""
        return list(self._table_get().required_filters)


class _Unresolved:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<unresolved>"


_UNRESOLVED = _Unresolved()
