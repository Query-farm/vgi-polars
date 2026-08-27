# Copyright 2026 Query Farm LLC - https://query.farm

"""A narrowing helper for `pl.from_arrow`'s untypeable return.

`polars.from_arrow` is intentionally not `@overload`ed on its input type (see
its own source comment: "we cannot @overload the typing (Series vs DataFrame)
here, as pyarrow ..."), so it's always statically typed `DataFrame | Series`
even though every call site in this package passes a `pa.Table`/`pa.RecordBatch`/
`pa.Schema.empty_table()` and always gets a `DataFrame` back at runtime. `cast`
alone would silently hide a real mismatch, so this asserts instead — a cheap,
correct-by-construction check, not a workaround for an actual ambiguity.
"""

from __future__ import annotations

from typing import Any

import polars as pl

__all__ = ["arrow_to_df", "arrow_to_series"]


def arrow_to_df(data: Any) -> pl.DataFrame:
    """Call `pl.from_arrow(data)` and assert the result is a `DataFrame`, not a `Series`.

    Every call site here passes table-shaped Arrow data (`pa.Table`/
    `pa.RecordBatch`/an empty table from a `pa.Schema`), for which
    `pl.from_arrow` always returns a `DataFrame` — the `Series` half of its
    return type is only reachable for `pa.Array`/`pa.ChunkedArray` input,
    which this package never passes.
    """
    result = pl.from_arrow(data)
    assert isinstance(result, pl.DataFrame), f"expected a DataFrame from pl.from_arrow, got {type(result)!r}"
    return result


def arrow_to_series(data: Any) -> pl.Series:
    """Call `pl.from_arrow(data)` and assert the result is a `Series`, not a `DataFrame`.

    The mirror image of `arrow_to_df`, for the `pa.Array`/`pa.ChunkedArray`
    call sites (a bare Arrow array, not table-shaped data).
    """
    result = pl.from_arrow(data)
    assert isinstance(result, pl.Series), f"expected a Series from pl.from_arrow, got {type(result)!r}"
    return result
