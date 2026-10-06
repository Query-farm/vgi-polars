# Copyright 2026 Query Farm LLC - https://query.farm

"""Best-effort translation of a Polars predicate `Expr` into VGI Filter Encoding v2.

This is purely an optimization (see `errors.py`'s module docstring and
`_source.py`'s "Design Principle 1" comment): `vgi_polars` always re-applies the
*complete, original* predicate locally after scanning, regardless of what gets
translated here. So this module is free to be conservative — an un-translatable
predicate, or one of its conjuncts, is simply not pushed, never an error.

**Wire format.** vgi-python >= 0.40 accepts only Filter Encoding v2
(`vgi.filters.v2`, spec: vgi-python's `docs/protocol/vgi-filter-encoding-v2-spec.md`,
consumer: `vgi/filter_v2.py`): a one-row Arrow batch whose first field is a
non-nullable `filter_spec` JSON document (a typed expression tree), followed by
typed `value_N` payload fields the tree's literals reference by index. The batch
is built with vgi-python's own `build_filter_batch`/`value_payload` helpers, so
the envelope (field order, nullability, schema metadata) is exactly the one the
consumer validates.

How the v2 document is shaped, and why:

- **One predicate per translated conjunct, all `advisory`.** The spec's producer
  rules (section 8.3): a client that keeps its own exact filter sends `advisory`,
  and a partial translation of a top-level `AND` may send each non-throwing
  conjunct as its own advisory predicate. That is exactly vgi-polars' posture —
  the local re-filter always runs — and the spec names it for Polars (section
  10.4). A worker may prune with an advisory predicate but must keep every row
  that satisfies it, so pushing a subset of the conjuncts is always safe.
- **No evaluation context (`vgi.none.v1`).** Every node emitted here — a
  column/literal comparison, `IS [NOT] NULL`, a literal `IN` list — is a
  context-independent core operation. String comparison is binary in Polars,
  which is the uncollated case the spec allows under `vgi.none.v1` (5.4).
- **Columns are named by the bind output schema.** v2 references a column by its
  index in the scan function's *unprojected bind output* and validates the
  name at that index; a mismatch fails the whole request (it is malformed, not
  merely unsupported). The catalog's declared column names are not necessarily
  the function's (`data.numbers` declares `value`; its function emits `n`), so
  the caller passes the bound schema (`_source.py` binds once to learn it) and
  the declared names the predicate uses; they correspond by position.
- **Literals keep Polars' resolved type.** The predicate an `io_source`
  receives has been type-resolved against the scan schema, so each literal
  carries a concrete dtype (`{"Scalar": {"Int32": 5}}`). Payloads are built
  with that exact Arrow type — a nanosecond `Datetime` stays nanoseconds, a
  `Decimal(38, 2)` stays a decimal — rather than via a Python value, which
  would silently truncate sub-microsecond precision. Truncating a literal is
  not a harmless pushdown miss: `t < 1.0000005 µs` truncated to `t < 1.000000
  µs` is stricter than the original, so a worker applying it drops rows the
  local filter can never get back.
- **Only type-compatible comparisons.** A conjunct is pushed only when the
  literal's type belongs to the same family as the bound column's (numeric with
  numeric, string with string, a naive timestamp with a naive timestamp, ...).
  The consumer binds every predicate under DuckDB's rules and rejects one that
  does not bind, which would fail the scan instead of merely skipping the
  pushdown.

Supported grammar (unchanged from the v1 encoding this replaced): a top-level
chain of `AND`ed comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) between a bare
column reference and a scalar literal (either side), `is_null()`/
`is_not_null()`, and `col.is_in([...])` (default `nulls_equal=False` only — see
`_translate_is_in`). `OR`, string/function predicates, casts, and anything else
are left untranslated and fall through to local filtering.

The Polars expression AST (`Expr.meta.serialize(format="json")`) shapes this
module reads are the same in Polars 1.41+ and 2.0.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Literal

import polars as pl
import pyarrow as pa
from vgi import FilterPayload, build_filter_batch, serialize_filter_batch, value_payload
from vgi.filter_v2 import DUCKDB_STANDARD_V1, FILTER_ENCODING

_COMPARISON_OPS = {
    "Eq": "eq",
    "NotEq": "ne",
    "Lt": "lt",
    "LtEq": "le",
    "Gt": "gt",
    "GtEq": "ge",
}
# Op to use when the column/literal sides of a BinaryExpr are swapped
# (`5 < col("a")` instead of `col("a") > 5`).
_FLIPPED_OPS = {
    "Eq": "Eq",
    "NotEq": "NotEq",
    "Lt": "Gt",
    "LtEq": "GtEq",
    "Gt": "Lt",
    "GtEq": "LtEq",
}

_TimeUnit = Literal["ms", "us", "ns"]
_TIME_UNITS: dict[str, _TimeUnit] = {"Milliseconds": "ms", "Microseconds": "us", "Nanoseconds": "ns"}

#: Concretely-typed (`Scalar`) literal keys -> the Arrow type of the payload.
_SCALAR_TYPES: dict[str, pa.DataType] = {
    "Int8": pa.int8(),
    "Int16": pa.int16(),
    "Int32": pa.int32(),
    "Int64": pa.int64(),
    "UInt8": pa.uint8(),
    "UInt16": pa.uint16(),
    "UInt32": pa.uint32(),
    "UInt64": pa.uint64(),
    "Float32": pa.float32(),
    "Float64": pa.float64(),
    "String": pa.string(),
    "Boolean": pa.bool_(),
}
#: Not-yet-type-resolved (`Dyn`) literal keys -> the Arrow type of the payload.
#: Only a freshly authored `Expr` carries these (e.g. one built directly in a
#: unit test); the predicate a real scan receives is always resolved.
_DYN_TYPES: dict[str, pa.DataType] = {
    "Int": pa.int64(),
    "Float": pa.float64(),
    "Str": pa.string(),
    "Bool": pa.bool_(),
}


def _flatten_and(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a chain of top-level `And` BinaryExprs into a flat list of conjunct AST nodes.

    Handles both right-leaning and left-leaning chains.
    """
    binary = node.get("BinaryExpr")
    if isinstance(binary, dict) and binary.get("op") == "And":
        return _flatten_and(binary["left"]) + _flatten_and(binary["right"])
    return [node]


def _temporal_scalar(key: str, value: Any) -> pa.Scalar[Any] | None:
    """Build the exact Arrow scalar for a Date/Datetime/Duration/Binary/Decimal literal, or `None`."""
    if key == "Date" and isinstance(value, int) and not isinstance(value, bool):
        return pa.scalar(value, type=pa.int32()).cast(pa.date32())
    if key == "Datetime" and isinstance(value, list) and len(value) == 3:
        ticks, unit, tz = value
        # A time-zone-aware literal is left untranslated: comparing it against
        # a column correctly depends on the zone, which DuckDB resolves from
        # session state — context this module deliberately never sends.
        if tz is not None or unit not in _TIME_UNITS or not isinstance(ticks, int):
            return None
        return pa.scalar(ticks, type=pa.int64()).cast(pa.timestamp(_TIME_UNITS[unit]))
    if key == "Duration" and isinstance(value, list) and len(value) == 2:
        amount, unit = value
        if unit not in _TIME_UNITS or not isinstance(amount, int):
            return None
        return pa.scalar(amount, type=pa.int64()).cast(pa.duration(_TIME_UNITS[unit]))
    if key == "Binary" and isinstance(value, list):
        return pa.scalar(bytes(value), type=pa.binary())
    if key == "Decimal" and isinstance(value, list) and len(value) == 3:
        unscaled, precision, scale = value
        if not (isinstance(unscaled, int) and isinstance(precision, int) and isinstance(scale, int)):
            return None
        if not 1 <= precision <= 38:
            return None
        return pa.scalar(Decimal(unscaled).scaleb(-scale), type=pa.decimal128(precision, scale))
    return None


def _literal_scalar(node: dict[str, Any]) -> pa.Scalar[Any] | None:
    """Build a typed Arrow scalar from a `Literal` AST node, or `None` if it isn't one we understand.

    Two shapes carry a literal, observed live: a freshly authored `Expr` keeps an
    un-type-resolved literal under `Dyn` (`{"Dyn": {"Int": 5}}`); the same
    predicate, once it has flowed through a real `LazyFrame` (which is what
    `register_io_source`'s callback receives), carries a concretely typed one
    under `Scalar` (`{"Scalar": {"Int32": 5}}`). Date/Datetime/Duration/Binary/
    Decimal literals always appear under `Scalar`.

    A NULL literal (and the IPC-encoded list literal `is_in` uses, handled by
    `_decode_is_in_values`) returns `None` — never raises.
    """
    literal = node.get("Literal")
    if not isinstance(literal, dict) or len(literal) != 1:
        return None
    ((kind, typed),) = literal.items()
    if kind not in ("Dyn", "Scalar") or not isinstance(typed, dict) or len(typed) != 1:
        return None
    ((key, value),) = typed.items()
    try:
        if kind == "Dyn":
            data_type = _DYN_TYPES.get(key)
        else:
            data_type = _SCALAR_TYPES.get(key)
            if data_type is None:
                return _temporal_scalar(key, value)
        if data_type is None or value is None or isinstance(value, (list, dict)):
            return None
        return pa.scalar(value, type=data_type)
    except (pa.ArrowException, ValueError, TypeError, OverflowError, ArithmeticError):
        # e.g. a `Dyn` int beyond int64 — just not pushed.
        return None


def _column_name(node: dict[str, Any]) -> str | None:
    name = node.get("Column")
    return name if isinstance(name, str) else None


def _translate_comparison(node: dict[str, Any]) -> tuple[str, str, pa.Scalar[Any]] | None:
    """Match `col OP literal` or `literal OP col` -> (column_name, v2_op, literal)."""
    binary = node.get("BinaryExpr")
    if not isinstance(binary, dict):
        return None
    op = binary.get("op")
    left, right = binary.get("left"), binary.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None

    col = _column_name(left)
    if col is not None and op in _COMPARISON_OPS:
        value = _literal_scalar(right)
        if value is not None:
            return col, _COMPARISON_OPS[op], value

    col = _column_name(right)
    if col is not None and op in _FLIPPED_OPS:
        value = _literal_scalar(left)
        if value is not None:
            return col, _COMPARISON_OPS[_FLIPPED_OPS[op]], value

    return None


def _translate_null_check(node: dict[str, Any]) -> tuple[str, bool] | None:
    """Match `col.is_null()` / `col.is_not_null()` -> (column_name, negated)."""
    func = node.get("Function")
    if not isinstance(func, dict):
        return None
    inputs = func.get("input")
    boolean = func.get("function", {}).get("Boolean") if isinstance(func.get("function"), dict) else None
    if not isinstance(inputs, list) or len(inputs) != 1 or boolean not in ("IsNull", "IsNotNull"):
        return None
    col = _column_name(inputs[0])
    if col is None:
        return None
    return col, boolean == "IsNotNull"


def _decode_is_in_values(node: dict[str, Any]) -> pa.Array[Any] | None:
    """Decode the RHS needle-list literal of an `is_in` `Function` node.

    Unlike every other literal shape, Polars serializes an `is_in([...])`
    needle list not as plain JSON but as a *complete Arrow IPC stream*,
    embedded as a raw list of ints (byte values) under `Literal.Scalar.List`
    (the raw bytes start with the Arrow IPC continuation marker `0xFFFFFFFF`).
    Decoding it yields a RecordBatch with one column and one row per candidate
    value (NOT a single row containing a list) — the flat array of candidates
    is `batch.column(0)`.

    Returns `None` on any unexpected/malformed shape, never raises.
    """
    literal = node.get("Literal")
    if not isinstance(literal, dict):
        return None
    scalar = literal.get("Scalar")
    if not isinstance(scalar, dict):
        return None
    raw_list = scalar.get("List")
    if not isinstance(raw_list, list):
        return None
    try:
        ipc_bytes = bytes(raw_list)
        batch = pa.ipc.open_stream(io.BytesIO(ipc_bytes)).read_next_batch()
    except Exception:  # noqa: BLE001 - deliberately never raises, see docstring
        return None
    if batch.num_columns != 1:
        return None
    return batch.column(0)


def _translate_is_in(node: dict[str, Any]) -> tuple[str, pa.Array[Any]] | None:
    """Match `col.is_in([...])` -> (column_name, candidate_values_array).

    Only translates the default `nulls_equal=False` case, whose semantics match
    SQL `IN` for filtering: a NULL tested value, or a non-match against a list
    containing NULL, is NULL in SQL and false in Polars — both drop the row.
    `nulls_equal=True` makes a NULL needle match a NULL value, which SQL `IN`
    cannot express; a worker applying the advisory predicate would drop rows
    the caller's local re-filter needs and can never get back, so that case is
    left untranslated.
    """
    func = node.get("Function")
    if not isinstance(func, dict):
        return None
    inputs = func.get("input")
    function = func.get("function")
    boolean = function.get("Boolean") if isinstance(function, dict) else None
    is_in = boolean.get("IsIn") if isinstance(boolean, dict) else None
    if not isinstance(inputs, list) or len(inputs) != 2 or not isinstance(is_in, dict):
        return None
    if is_in.get("nulls_equal"):
        return None

    col = _column_name(inputs[0])
    if col is None:
        return None

    values = _decode_is_in_values(inputs[1])
    if values is None:
        return None

    return col, values


def _is_numeric(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def _is_string(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t)


def _is_binary(t: pa.DataType) -> bool:
    return pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_binary_view(t)


def _is_naive_timestamp(t: pa.DataType) -> bool:
    return pa.types.is_timestamp(t) and t.tz is None


#: Type families within which a comparison binds under `vgi.duckdb.standard.v1`
#: with no cast the producer would have to spell out. Deliberately narrow.
_FAMILIES: tuple[Callable[[pa.DataType], bool], ...] = (
    _is_numeric,
    _is_string,
    _is_binary,
    pa.types.is_boolean,
    pa.types.is_date,
    _is_naive_timestamp,
    pa.types.is_duration,
)


def _compatible(column_type: pa.DataType, literal_type: pa.DataType) -> bool:
    """Whether a column of `column_type` may be compared with a literal of `literal_type`."""
    return any(family(column_type) and family(literal_type) for family in _FAMILIES)


class _V2Builder:
    """Accumulates v2 predicates and their `value_N` payloads."""

    def __init__(self) -> None:
        self.predicates: list[dict[str, Any]] = []
        self.payloads: list[FilterPayload] = []

    def literal(self, value: pa.Scalar[Any]) -> dict[str, Any]:
        index = len(self.payloads)
        self.payloads.append(value_payload(index, value))
        return {"node": "literal", "value_ref": index}

    def add(self, expression: dict[str, Any]) -> None:
        self.predicates.append(
            {
                "id": f"query:{len(self.predicates)}",
                "revision": 0,
                "mode": "advisory",
                "source": "query",
                "expression": expression,
            }
        )


def translate_predicate(
    predicate: pl.Expr, bound_schema: pa.Schema, column_names: list[str] | None = None
) -> bytes | None:
    """Best-effort translate `predicate` into VGI Filter Encoding v2 `pushdown_filters` IPC bytes.

    Args:
        predicate: The Polars predicate the scan received.
        bound_schema: The scan function's unprojected bind output schema — the
            names and types the worker validates column references against.
        column_names: The names `predicate` refers to columns by, positionally
            aligned with `bound_schema` (a catalog table's declared names may
            differ from its scan function's). Defaults to `bound_schema.names`.

    Returns:
        The serialized one-batch v2 snapshot, or `None` if no conjunct could be
        translated (never raises on an unsupported shape — that's just a
        pushdown miss, handled safely by the caller's unconditional local
        re-filter).
    """
    names = list(bound_schema.names) if column_names is None else list(column_names)
    if len(names) != len(bound_schema):
        return None
    try:
        raw = predicate.meta.serialize(format="json")
    except Exception:  # noqa: BLE001 - deliberately never raises, see docstring
        return None
    try:
        ast = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(ast, dict):
        return None

    def column_ref(col: str) -> tuple[dict[str, Any], pa.DataType] | None:
        # A name appearing twice can't be resolved to one index — not pushed.
        if names.count(col) != 1:
            return None
        index = names.index(col)
        field = bound_schema.field(index)
        return {"node": "column_ref", "column_index": index, "column_name": field.name}, field.type

    builder = _V2Builder()
    for conjunct in _flatten_and(ast):
        comparison = _translate_comparison(conjunct)
        if comparison is not None:
            col, op, value = comparison
            ref = column_ref(col)
            if ref is not None and _compatible(ref[1], value.type):
                builder.add({"node": "comparison", "op": op, "left": ref[0], "right": builder.literal(value)})
            continue

        null_check = _translate_null_check(conjunct)
        if null_check is not None:
            col, negated = null_check
            ref = column_ref(col)
            if ref is not None:
                builder.add({"node": "is_null", "expression": ref[0], "negated": negated})
            continue

        is_in = _translate_is_in(conjunct)
        if is_in is not None:
            col, values = is_in
            ref = column_ref(col)
            if ref is not None and _compatible(ref[1], values.type):
                # v2 literal IN set: ONE Arrow list scalar holding every candidate.
                offsets = pa.array([0, len(values)], type=pa.int32())
                candidates = pa.ListArray.from_arrays(offsets, values)[0]
                set_ref = builder.literal(candidates)["value_ref"]
                builder.add(
                    {
                        "node": "in",
                        "expression": ref[0],
                        "set": {"kind": "literal", "value_ref": set_ref},
                        "negated": False,
                    }
                )
            continue

        # Unsupported conjunct (OR, function/string predicate, ...) — just
        # skip it. Correctness is guaranteed by the caller's local re-filter
        # regardless of what does or doesn't get pushed.

    if not builder.predicates:
        return None

    document = {
        "encoding": FILTER_ENCODING,
        "semantics": DUCKDB_STANDARD_V1,
        "kind": "snapshot",
        "predicates": builder.predicates,
    }
    return serialize_filter_batch(build_filter_batch(document, builder.payloads))
