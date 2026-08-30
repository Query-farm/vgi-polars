# Copyright 2026 Query Farm LLC - https://query.farm

"""Multi-branch table scanning — `pl.concat` of one scan per `ScanBranch`.

A multi-branch table composes a logical scan from N physical sources
(canonical case: a hot VGI tier + a cold Iceberg/Parquet tier). The DuckDB
C++ extension rewrites the placeholder scan into `LogicalSetOperation(
UNION_ALL, ...)`, one arm per branch; this module is the Polars-side
equivalent, `pl.concat` over one lazy scan per branch.

This needed a vgi-python fix first: `Client` had no `table_scan_branches_get`
method at all — a multi-branch table was invisible to any non-DuckDB caller,
not just unsupported. Without it, `VgiTable.scan()` silently used
`table_scan_function_get`'s single-branch view (branch 0 only) for a
multi-branch table — a real, silent correctness gap for exactly these tables,
not a missing feature that failed loudly. See CLAUDE.md's "Multi-branch
tables" section.

**Scoped to function branches only** (the common case — a worker naming a
DuckDB/VGI table function per source). Catalog-table branches (companion
catalog federation, e.g. a DuckLake arm) and format branches (`read_parquet`/
`read_csv`-style declarative readers) are NOT yet supported — see the raises
below. Both need real design work this session didn't reach: catalog-table
branches require attaching (or reusing) a companion catalog, which has no
established Polars-side concept at all; format branches need a resolver from
`format_name` to an actual reader the same way the C++ extension's own
`ResolveFormatBranchFunction` does.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import polars as pl

from vgi_polars.errors import VgiPolarsError

if TYPE_CHECKING:
    from vgi.catalog.catalog_interface import FunctionInfo, ScanBranch, ScanFunctionResult

    from vgi_polars.catalog import VgiCatalog
    from vgi_polars.table import VgiTable

__all__ = ["parse_branch_filter", "scan_multi_branch"]

# Mirrors the DuckDB C++ extension's own minimal v1.0 branch_filter binder
# scope (see ~/Development/vgi's docs/multi_branch.md): an AND-chain of
# "col OP const" comparisons. OR is out of scope here, same as there.
_COMPARISON_RE = re.compile(r"^(\w+)\s*(<=|>=|<>|!=|=|<|>)\s*(.+)$")
_AND_SPLIT_RE = re.compile(r"(?i)\s+AND\s+")
_OPS: dict[str, Any] = {
    "=": lambda c, v: c == v,
    "<>": lambda c, v: c != v,
    "!=": lambda c, v: c != v,
    "<": lambda c, v: c < v,
    "<=": lambda c, v: c <= v,
    ">": lambda c, v: c > v,
    ">=": lambda c, v: c >= v,
}


def _parse_literal(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.upper() in ("TRUE", "FALSE"):
        return text.upper() == "TRUE"
    raise VgiPolarsError(f"branch_filter: literal not understood: {text!r}")


def parse_branch_filter(sql: str) -> pl.Expr:
    """Parse a `ScanBranch.branch_filter` SQL string into a `pl.Expr`.

    `branch_filter` is not an optional pushdown hint the way an ordinary scan
    predicate is (see Design Principle 1 in `_source.py`) — it's a
    worker-declared partition boundary that MUST be applied, or overlapping
    branches produce duplicate/wrong rows in the unioned result (that's the
    whole point of `branch_filter`: making overlapping physical sources,
    e.g. a hot tier with a day of overlap against the cold tier, disjoint at
    scan time). There is no local "always re-verify against the original"
    fallback for it the way there is for a caller's own predicate — an
    unparseable `branch_filter` raises rather than silently scanning the
    branch unconstrained, which would double-count overlapping rows.
    """
    conjuncts = _AND_SPLIT_RE.split(sql.strip())
    expr: pl.Expr | None = None
    for conjunct in conjuncts:
        match = _COMPARISON_RE.match(conjunct.strip())
        if not match:
            raise VgiPolarsError(
                f"branch_filter expression not understood: {conjunct!r} — only AND-chains of "
                "'col OP const' comparisons are supported"
            )
        col, op, literal_text = match.groups()
        clause = _OPS[op](pl.col(col), _parse_literal(literal_text))
        expr = clause if expr is None else expr & clause
    assert expr is not None  # sql is non-empty; _AND_SPLIT_RE always yields >= 1 conjunct
    return expr


class _BranchTable:
    """Duck-typed adapter so one `ScanBranch` can drive `_source.py`'s `make_io_source`.

    It behaves exactly like an ordinary `VgiTable` — the same pattern `test_errors.py`'s
    `_FakeTableForScanFunction` and `test_splits.py`'s `_FakeSplitTable` use in this repo's
    own test suite. `required_filters` returns empty: enforced once at the outer
    multi-branch table level (`VgiTable.required_filters()`), not redundantly per branch.
    """

    def __init__(self, *, catalog: VgiCatalog, schema_name: str, name: str, branch: ScanBranch) -> None:
        self._catalog = catalog
        self.schema_name = schema_name
        self.name = name
        # No per-branch time travel (yet) — always the live view; a branch
        # inherits AT-clause behavior from the outer VgiTable in a future
        # extension, not per-branch state today. Typed `str | None` (not
        # bare `None`) to match `_ScanSource`'s attribute, which every other
        # implementer (`VgiTable`) genuinely varies at runtime.
        self.at_unit: str | None = None
        self.at_value: str | None = None
        self._branch = branch

    def _scan_function_get(self) -> ScanFunctionResult:
        from vgi.catalog.catalog_interface import ScanFunctionResult

        return ScanFunctionResult(
            function_name=self._branch.function_name,
            positional_arguments=list(self._branch.positional_arguments),
            named_arguments=dict(self._branch.named_arguments),
        )

    def _function_info_get(self) -> FunctionInfo | None:
        from vgi.catalog.catalog_interface import SchemaObjectType

        from vgi_polars.errors import VGI_CLIENT_ERRORS

        try:
            infos = self._catalog.client.schema_contents(
                attach_opaque_data=self._catalog.attach_opaque_data,
                name=self.schema_name,
                type=SchemaObjectType.TABLE_FUNCTION,
            )
        except VGI_CLIENT_ERRORS:
            return None
        return next((i for i in infos if i.name == self._branch.function_name), None)

    def scan_function_schema(self) -> str:
        return self.schema_name

    def required_filters(self) -> list[list[str]]:
        return []


def scan_multi_branch(
    table: VgiTable, branches: list[ScanBranch], *, secrets: dict[str, Any] | None = None
) -> pl.LazyFrame:
    """Build a `pl.concat` of one scan per branch, each with its `branch_filter` (if any) applied.

    Zero branches is legal (a fully-pruned multi-branch scan prunes to nothing, matching the
    C++ extension's own handling) and scans as an empty result with the table's declared
    schema.
    """
    from polars.io.plugins import register_io_source

    from vgi_polars._source import make_io_source

    if not branches:
        return pl.DataFrame(schema=table.schema).lazy()

    lazy_frames: list[pl.LazyFrame] = []
    for branch in branches:
        if branch.source_table is not None:
            raise VgiPolarsError(
                f"{table.schema_name}.{table.name}: a catalog-table branch "
                f"(source_table={branch.source_table!r}) is not yet supported by vgi-polars — "
                "only function branches are. See CLAUDE.md's Scope section."
            )
        if branch.format_name is not None:
            raise VgiPolarsError(
                f"{table.schema_name}.{table.name}: a format branch (format_name="
                f"{branch.format_name!r}) is not yet supported by vgi-polars. See CLAUDE.md's "
                "Scope section."
            )
        branch_table = _BranchTable(
            catalog=table._catalog, schema_name=table.schema_name, name=table.name, branch=branch
        )
        lf = register_io_source(make_io_source(branch_table, table.arrow_schema, secrets=secrets), schema=table.schema)
        if branch.branch_filter:
            lf = lf.filter(parse_branch_filter(branch.branch_filter))
        lazy_frames.append(lf)
    return pl.concat(lazy_frames, how="vertical")
