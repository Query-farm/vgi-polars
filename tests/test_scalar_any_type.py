# Copyright 2026 Query Farm LLC - https://query.farm

"""`ANY`-typed (polymorphic) scalar-function arguments — `_resolve_array_field`.

A declared argument accepting more than one concrete Arrow type (e.g.
`vgi-ical`'s `is_valid_ical(input)`, documented as "a `VARCHAR` file path, or
a `BLOB` of bytes") advertises its Arrow type on the wire as `pa.null()` — a
placeholder, never a real type to cast real data to. Calling such a function
used to crash with `pyarrow.lib.ArrowNotImplementedError: Unsupported cast
from ... to null`, confirmed live against the real `vgi-ical` worker (built
from source, attached as a subprocess, called via `catalog.scalar_function`)
before this fix; no fixture in vgi-fixture-worker declares an `AnyArrow`
argument, so these are unit tests of the resolution helper directly rather
than an end-to-end one against a live worker.
"""

from __future__ import annotations

import pyarrow as pa

from vgi_polars._arguments import VGI_TYPE_ANY, VGI_TYPE_KEY, is_any_type_field
from vgi_polars._scalar import _resolve_array_field


def _any_typed_field(name: str = "input") -> pa.Field:
    """An ANY-typed declared field, matching vgi-python's own wire encoding exactly."""
    return pa.field(name, pa.null(), metadata={VGI_TYPE_KEY: VGI_TYPE_ANY})


def test_is_any_type_field_detects_the_metadata_tag() -> None:
    assert is_any_type_field(_any_typed_field())
    assert not is_any_type_field(pa.field("x", pa.utf8()))
    assert not is_any_type_field(pa.field("x", pa.null()))  # null type alone isn't enough


def test_resolve_array_field_casts_a_concrete_declared_type() -> None:
    """A normal, concretely-typed field still casts real data to its declared type -- unchanged behavior."""
    array = pa.array([1, 2, 3], type=pa.int32())
    declared = pa.field("value", pa.int64())

    resolved_array, resolved_field = _resolve_array_field(array, declared)

    assert resolved_array.type == pa.int64()
    assert resolved_field is declared


def test_resolve_array_field_skips_the_cast_for_any_typed_string_data() -> None:
    """The bug this regresses: a VARCHAR value for an ANY-typed arg used to crash trying to cast to null."""
    array = pa.array(["/tmp/some/path.ics"], type=pa.large_string())
    declared = _any_typed_field()

    resolved_array, resolved_field = _resolve_array_field(array, declared)

    assert resolved_array is array  # untouched -- no cast attempted
    assert resolved_array.type == pa.large_string()
    assert resolved_field.type == pa.large_string()
    assert resolved_field.name == "input"


def test_resolve_array_field_skips_the_cast_for_any_typed_binary_data() -> None:
    """The other side of the same polymorphic argument: raw BLOB bytes instead of a path string."""
    array = pa.array([b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n"], type=pa.large_binary())
    declared = _any_typed_field()

    resolved_array, resolved_field = _resolve_array_field(array, declared)

    assert resolved_array is array
    assert resolved_array.type == pa.large_binary()
    assert resolved_field.type == pa.large_binary()
