# Copyright 2026 Query Farm LLC - https://query.farm

"""Best-effort translation of a Polars predicate `Expr` into VGI's filter-pushdown wire format.

This is purely an optimization (see `errors.py`'s module docstring and
`_source.py`'s "Design Principle 1" comment): `vgi_polars` always re-applies the
*complete, original* predicate locally after scanning, regardless of what gets
translated here. So this module is free to be conservative — an un-translatable
predicate, or one of its conjuncts, is simply not pushed, never an error.

Supported grammar (deliberately small, matching this repo's own C++ client's
`branch_filter` binder restriction to "col OP const, AND"): a top-level chain of
`AND`ed comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) between a bare column
reference and a scalar literal, plus `is_null()`/`is_not_null()` and
`col.is_in([...])` (default `nulls_equal=False` only — see `_translate_is_in`).
`is_in`'s needle list doesn't serialize as plain JSON like other literals — it's a
complete Arrow IPC stream embedded as a raw byte list (see `meta.serialize()`) —
`_translate_is_in` decodes it via `pyarrow.ipc`. `OR`, string/function predicates,
and anything else are left untranslated and simply fall through to local
filtering.

VGI wire format reference: vgi-python's `docs/filter-pushdown.md` +
`vgi/table_filter_pushdown.py`. Recipe for building the IPC bytes taken from
`vgi-python/tests/transactor/test_transactor.py`.
"""

from __future__ import annotations

import datetime
import io
import json
from decimal import Decimal
from typing import Any

import polars as pl
import pyarrow as pa

_EPOCH_DATE = datetime.date(1970, 1, 1)
_EPOCH_DATETIME = datetime.datetime(1970, 1, 1)
_TIME_UNIT_TO_TIMEDELTA_KWARG = {
    "Milliseconds": "milliseconds",
    "Microseconds": "microseconds",
    "Nanoseconds": "microseconds",  # datetime has no ns resolution — sub-µs precision is lost, not incorrect
}
_NS_UNITS = {"Nanoseconds"}

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


def _flatten_and(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a chain of top-level `And` BinaryExprs into a flat list of conjunct AST nodes.

    Handles both right-leaning and left-leaning chains.
    """
    binary = node.get("BinaryExpr")
    if isinstance(binary, dict) and binary.get("op") == "And":
        return _flatten_and(binary["left"]) + _flatten_and(binary["right"])
    return [node]


def _timedelta_from_units(amount: int, unit: str) -> datetime.timedelta:
    if unit == "Nanoseconds":
        amount //= 1000  # datetime has no sub-microsecond resolution — truncate, not incorrect
        unit = "Microseconds"
    return datetime.timedelta(**{_TIME_UNIT_TO_TIMEDELTA_KWARG[unit]: amount})


def _literal_value(node: dict[str, Any]) -> tuple[Any, bool]:
    """Extract a Python value from a `Literal` AST node.

    The value is converted to the Python type pyarrow would infer the *correct* Arrow type
    from (so `pa.array([value])` downstream never silently mistypes a temporal/binary/
    decimal value as a bare int).

    Two distinct shapes carry a plain numeric/string/bool scalar, observed
    live: a freshly authored `Expr` (e.g. built directly in a test) keeps an
    un-type-resolved literal under `Dyn` (`{"Dyn": {"Int": 5}}`); the SAME
    predicate, once it has flowed through a real `LazyFrame`'s query engine
    (type-resolved against the registered schema — this is what
    `register_io_source`'s `io_source` callback actually receives), carries a
    concretely-typed literal under `Scalar` instead (`{"Scalar": {"Int64":
    5}}`). Both are handled identically — only the outer key differs. Date/
    Datetime/Duration/Binary/Decimal literals were observed identical either
    way (always under `Scalar`, never `Dyn` — presumably because those Python
    types are unambiguous at construction, unlike a bare `int`/`float`/`str`).

    Returns (value, ok) — ok is False for shapes we don't understand (e.g. the
    IPC-encoded list literal `is_in` uses, which `_translate_is_in`/
    `_decode_is_in_values` decode separately), never raises.
    """
    literal = node.get("Literal")
    if not isinstance(literal, dict):
        return None, False
    typed = literal.get("Dyn")
    if not isinstance(typed, dict):
        typed = literal.get("Scalar")
    if not isinstance(typed, dict):
        return None, False

    for key, value in typed.items():
        # {"Int": 5} / {"Int64": 5}, {"Float": 1.5} / {"Float64": 1.5},
        # {"Str": "x"} / {"String": "x"}, {"Bool": true} / {"Boolean": true}, ...
        # A bare int/float under a temporal/decimal key would be mistyped
        # below by the isinstance check alone — these keys are handled
        # explicitly first via the branches below, so falling through to
        # here only happens for genuinely plain scalars.
        if (isinstance(value, (int, float, str, bool)) or value is None) and key not in (
            "Date",
            "Datetime",
            "Duration",
            "Binary",
            "Decimal",
        ):
            return value, True
        try:
            if key == "Date" and isinstance(value, int):
                return _EPOCH_DATE + datetime.timedelta(days=value), True
            if key == "Datetime" and isinstance(value, list) and len(value) == 3:
                ts, unit, _tz = value
                # Timezone-aware literals aren't reconstructed (the tz name
                # alone isn't enough to build a correct offset without a
                # tzdata dependency) — left untranslated rather than risk a
                # silently-wrong comparison.
                if _tz is not None:
                    return None, False
                return _EPOCH_DATETIME + _timedelta_from_units(ts, unit), True
            if key == "Duration" and isinstance(value, list) and len(value) == 2:
                amount, unit = value
                return _timedelta_from_units(amount, unit), True
            if key == "Binary" and isinstance(value, list):
                return bytes(value), True
            if key == "Decimal" and isinstance(value, list) and len(value) == 3:
                unscaled, _precision, scale = value
                return Decimal(unscaled).scaleb(-scale), True
        except (KeyError, ValueError, OverflowError, OSError):
            return None, False
        return None, False
    return None, False


def _column_name(node: dict[str, Any]) -> str | None:
    name = node.get("Column")
    return name if isinstance(name, str) else None


def _translate_comparison(node: dict[str, Any]) -> tuple[str, str, Any] | None:
    """Match `col OP literal` or `literal OP col` -> (column_name, vgi_op, value)."""
    binary = node.get("BinaryExpr")
    if not isinstance(binary, dict):
        return None
    op = binary.get("op")
    left, right = binary.get("left"), binary.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None

    col = _column_name(left)
    if col is not None and op in _COMPARISON_OPS:
        value, ok = _literal_value(right)
        if ok:
            return col, _COMPARISON_OPS[op], value

    col = _column_name(right)
    if col is not None and op in _FLIPPED_OPS:
        value, ok = _literal_value(left)
        if ok:
            return col, _COMPARISON_OPS[_FLIPPED_OPS[op]], value

    return None


def _translate_null_check(node: dict[str, Any]) -> tuple[str, str] | None:
    """Match `col.is_null()` / `col.is_not_null()` -> (column_name, filter_type)."""
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
    return col, "is_null" if boolean == "IsNull" else "is_not_null"


def _decode_is_in_values(node: dict[str, Any]) -> pa.Array[Any] | None:
    """Decode the RHS needle-list literal of an `is_in` `Function` node.

    Unlike every other literal shape `_literal_value` handles, Polars
    serializes an `is_in([...])` needle list not as plain JSON but as a
    *complete Arrow IPC stream*, embedded as a raw list of ints (byte values)
    under `Literal.Scalar.List` (confirmed empirically — the raw bytes start
    with the Arrow IPC continuation marker `0xFFFFFFFF`). Decoding it yields a
    RecordBatch with one unnamed column and one row per candidate value (NOT
    a single row containing a list) — the flat array of candidates is
    `batch.column(0)`.

    Returns `None` on any unexpected/malformed shape (missing keys, a
    `List` that isn't a byte list, IPC bytes that don't decode, an
    unexpected number of columns), never raises.
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

    Only translates the default `nulls_equal=False` case. `nulls_equal=True`
    changes Polars' NULL semantics so that a NULL needle matches NULL haystack
    values — the wire format's plain "col IN (v1, v2, v3)" set-membership
    filter has no way to express that, and a worker applying it with ordinary
    SQL-style IN semantics would silently drop rows that should have matched
    (a real correctness risk, not merely a missed optimization, since a
    worker that auto-applies pushed filters never returns those rows for the
    caller's local re-filter to recover) — so that case is deliberately left
    untranslated rather than risk it.
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


def translate_predicate(predicate: pl.Expr, column_names: list[str]) -> bytes | None:
    """Best-effort translate `predicate` into VGI `pushdown_filters` IPC bytes.

    Returns `None` if nothing in the predicate could be translated (never raises
    on an unsupported shape — that's just a pushdown miss, handled safely by the
    caller's unconditional local re-filter).
    """
    try:
        raw = predicate.meta.serialize(format="json")
    except Exception:  # noqa: BLE001 - deliberately never raises, see docstring
        return None
    try:
        ast = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    specs: list[dict[str, Any]] = []
    value_arrays: list[pa.Array[Any]] = []

    for conjunct in _flatten_and(ast):
        comparison = _translate_comparison(conjunct)
        if comparison is not None:
            col, op, value = comparison
            if col not in column_names:
                continue
            value_ref = len(value_arrays)
            value_arrays.append(pa.array([value]))
            specs.append(
                {
                    "column_name": col,
                    "column_index": column_names.index(col),
                    "type": "constant",
                    "op": op,
                    "value_ref": value_ref,
                }
            )
            continue

        null_check = _translate_null_check(conjunct)
        if null_check is not None:
            col, filter_type = null_check
            if col not in column_names:
                continue
            specs.append(
                {
                    "column_name": col,
                    "column_index": column_names.index(col),
                    "type": filter_type,
                }
            )
            continue

        is_in = _translate_is_in(conjunct)
        if is_in is not None:
            col, values = is_in
            if col not in column_names:
                continue
            value_ref = len(value_arrays)
            # Wire spec: the value_ref column is a list-typed array with a
            # SINGLE row whose value is the list of candidates (e.g.
            # `_val_0: ["active", "pending", "review"]`) — not one row per
            # candidate, unlike every other value_ref column in this file.
            offsets = pa.array([0, len(values)], type=pa.int32())
            value_arrays.append(pa.ListArray.from_arrays(offsets, values))
            specs.append(
                {
                    "column_name": col,
                    "column_index": column_names.index(col),
                    "type": "in",
                    "value_ref": value_ref,
                }
            )
            continue

        # Unsupported conjunct (OR, function/string predicate, ...) — just
        # skip it. Correctness is guaranteed by the caller's local re-filter
        # regardless of what does or doesn't get pushed.

    if not specs:
        return None

    spec_field = pa.field("filter_spec", pa.string(), metadata={b"vgi_filter_version": b"1"})
    fields = [spec_field, *(pa.field(f"value_{i}", arr.type) for i, arr in enumerate(value_arrays))]
    batch = pa.RecordBatch.from_arrays(
        [pa.array([json.dumps(specs)]), *value_arrays],
        schema=pa.schema(fields),
    )

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()
