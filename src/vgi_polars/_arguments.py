# Copyright 2026 Query Farm LLC - https://query.farm

"""Plain-Python-value -> `vgi.arguments.Arguments` conversion.

vgi-python's `Arguments` dataclass (`vgi/arguments.py`) only accepts
`pyarrow.Scalar` values — every call site in vgi-python itself wraps values
manually with `pa.scalar(...)`; there is no plain-value constructor to reuse.
This module is that conversion layer for vgi-polars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
from vgi.arguments import Arguments

#: Field metadata a `ConstParam` argument carries on the wire — a bind-time
#: value (not a per-row one), used by both the scalar (`_scalar.py`) and
#: aggregate (`_aggregate.py`) bridges to split a declared argument schema
#: into const vs. per-row/value fields. Confirmed identical convention on
#: both `main.multiply`'s `factor` and `main.vgi_percentile`'s `percentile`.
VGI_CONST_KEY = b"vgi_const"
VGI_CONST_TRUE = b"true"


def is_const_field(field: pa.Field) -> bool:
    """Whether a declared argument field is a bind-time `ConstParam`."""
    md = field.metadata or {}
    return md.get(VGI_CONST_KEY) == VGI_CONST_TRUE


def to_scalar(value: Any, arrow_type: pa.DataType | None = None) -> pa.Scalar[Any] | None:
    """Convert a plain Python value (or an already-built `pa.Scalar`) to a `pa.Scalar`.

    Args:
        value: A Python value, `None`, or an existing `pa.Scalar`/`pa.Array` element.
        arrow_type: Optional target Arrow type (e.g. decoded from a function's
            declared argument schema) so the value binds to the declared type
            instead of pyarrow's inferred default (`int` -> `int64`, etc.).

    """
    if value is None:
        return pa.scalar(None, type=arrow_type)
    if isinstance(value, pa.Scalar):
        return value.cast(arrow_type) if arrow_type is not None else value
    return pa.scalar(value, type=arrow_type)


def build_arguments(
    positional: Sequence[Any] = (),
    named: Mapping[str, Any] | None = None,
    *,
    positional_types: Sequence[pa.DataType | None] | None = None,
    named_types: Mapping[str, pa.DataType] | None = None,
) -> Arguments:
    """Build an `Arguments` from plain Python values.

    Args:
        positional: Positional argument values.
        named: Named argument values.
        positional_types: Optional per-position Arrow types, parallel to `positional`.
        named_types: Optional per-name Arrow types.

    """
    if positional_types is not None and len(positional_types) != len(positional):
        raise ValueError("positional_types must be the same length as positional")

    pos_scalars = tuple(
        to_scalar(v, positional_types[i] if positional_types is not None else None)
        for i, v in enumerate(positional)
    )
    named_scalars = None
    if named:
        named_types = named_types or {}
        named_scalars = {k: to_scalar(v, named_types.get(k)) for k, v in named.items()}
    return Arguments(positional=pos_scalars, named=named_scalars)
