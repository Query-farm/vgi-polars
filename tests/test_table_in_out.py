# Copyright 2026 Query Farm LLC - https://query.farm

"""Table-in-out (streaming + buffered map-a-table) via `LazyFrame.map_batches`
— see `_table_in_out.py`'s module docstring for the streaming-vs-buffering
dispatch and the input-dependent-output-schema handling."""

from __future__ import annotations

import polars as pl
import pytest

import vgi_polars as vp
from vgi_polars.errors import VgiPolarsError


def test_echo_streaming_passthrough(catalog: vp.VgiCatalog) -> None:
    echo = catalog.table_in_out_function("main", "echo")
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = echo(lf).collect()
    assert out.equals(lf.collect())


def test_echo_buffering_passthrough(catalog: vp.VgiCatalog) -> None:
    """`echo_buffering` is a `TableBufferingFunction` (Sink+Source) — same
    observable result as the streaming `echo`, different RPC drive loop
    underneath (`Client.table_buffering_function`, `streamable=False`)."""
    echo_buffering = catalog.table_in_out_function("main", "echo_buffering")
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = echo_buffering(lf).collect()
    assert out.equals(lf.collect())


def test_filter_by_setting(catalog: vp.VgiCatalog) -> None:
    """Rows pass through only where `value` >= the `threshold` setting."""
    filter_fn = catalog.table_in_out_function("main", "filter_by_setting")
    lf = pl.LazyFrame({"value": [1, 5, 3, 7, 2]})
    out = filter_fn(lf, settings={"threshold": 4}).collect()
    assert sorted(out["value"].to_list()) == [5, 7]


def test_echo_empty_input(catalog: vp.VgiCatalog) -> None:
    """The streaming path synthesizes a zero-row batch when Polars hands it
    an empty chunk (`Client.table_in_out_function` requires >=1 input batch)
    — confirms that synthesis doesn't break the empty-input case itself."""
    echo = catalog.table_in_out_function("main", "echo")
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.Int64), "b": pl.Series([], dtype=pl.Utf8)})
    out = echo(lf).collect()
    assert out.height == 0


def test_unknown_table_in_out_function_raises(catalog: vp.VgiCatalog) -> None:
    fn = catalog.table_in_out_function("main", "does_not_exist_xyz")
    with pytest.raises(VgiPolarsError, match="not found"):
        fn(pl.LazyFrame({"a": [1]}))


def test_accumulate_positional_and_named_args(accumulate_catalog: vp.VgiCatalog) -> None:
    """`accumulate('events', result='new')` — one positional bind-time arg
    (`name`, wire position 0, no `vgi_arg` metadata) plus a named one
    (`result`, `vgi_arg=named`); confirms the positional/named/table-slot
    split against real wire metadata, and that the real (not static-catalog)
    output schema — including the `_timestamp` column the static
    `FunctionInfo.output_schema` omits — is what `map_batches` actually gets."""
    accumulate = accumulate_catalog.table_in_out_function("main", "accumulate")
    lf = pl.LazyFrame({"x": [1, 2, 3]})
    out = accumulate(lf, "test_accumulate_positional_and_named_args", result="new").collect()
    assert out["x"].to_list() == [1, 2, 3]
    assert "_timestamp" in out.columns


def test_accumulate_wrong_positional_arg_count_raises(accumulate_catalog: vp.VgiCatalog) -> None:
    accumulate = accumulate_catalog.table_in_out_function("main", "accumulate")
    with pytest.raises(VgiPolarsError, match="expects 1 positional argument"):
        accumulate(pl.LazyFrame({"x": [1]}))


def test_accumulate_unknown_named_arg_raises(accumulate_catalog: vp.VgiCatalog) -> None:
    accumulate = accumulate_catalog.table_in_out_function("main", "accumulate")
    with pytest.raises(VgiPolarsError, match="no named argument"):
        accumulate(pl.LazyFrame({"x": [1]}), "events", not_a_real_option=1)


def test_scalar_function_name_not_found_as_table_in_out(catalog: vp.VgiCatalog) -> None:
    """`multiply` is a scalar function — `schema_contents(type=TABLE_FUNCTION)`
    (the catalog RPC `table_in_out_function` resolution uses) only ever lists
    TABLE/TABLE_BUFFERING functions to begin with, so a scalar function's name
    surfaces as "not found" here rather than a distinct "wrong kind" error;
    still a clear `VgiPolarsError`, not a confusing downstream RPC failure."""
    fn = catalog.table_in_out_function("main", "multiply")
    with pytest.raises(VgiPolarsError, match="not found"):
        fn(pl.LazyFrame({"value": [1]}))
