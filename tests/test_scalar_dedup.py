# Copyright 2026 Query Farm LLC - https://query.farm

"""Per-chunk scalar input dedup (`_scalar.py`'s `_dedup_positions`).

The client-side mirror of the DuckDB C++ extension's `vgi_exchange_input_dedup`
setting. See `_scalar.py`'s module docstring for the full design; these tests
verify the three load-bearing properties: (1) it actually reduces the batch
shipped to the worker for a low-cardinality `CONSISTENT`/
`CONSISTENT_WITHIN_QUERY` call, (2) results still land on the correct
original row regardless, and (3) a `FunctionStability.VOLATILE` function is
never deduped — duplicate inputs must still each reach the worker, since
identical inputs aren't guaranteed to produce identical outputs.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa

import vgi_polars as vp
from vgi_polars._scalar import _dedup_positions


def test_dedup_positions_reduces_and_maps_back_correctly() -> None:
    batch = pa.RecordBatch.from_pydict({"a": [1, 2, 1, 3, 2, 1]})
    result = _dedup_positions(batch)
    assert result is not None
    distinct_positions, inverse = result
    assert distinct_positions == [0, 1, 3]  # first occurrence of 1, 2, 3
    assert inverse == [0, 1, 0, 2, 1, 0]


def test_dedup_positions_declines_when_all_distinct() -> None:
    batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3]})
    assert _dedup_positions(batch) is None


def test_dedup_positions_declines_on_unhashable_struct_cell() -> None:
    batch = pa.RecordBatch.from_pydict(
        {"p": [{"lat": 1.0, "lon": 2.0}, {"lat": 1.0, "lon": 2.0}]},
        schema=pa.schema([pa.field("p", pa.struct([("lat", pa.float64()), ("lon", pa.float64())]))]),
    )
    assert _dedup_positions(batch) is None


def test_dedup_positions_declines_on_empty_batch() -> None:
    assert _dedup_positions(pa.RecordBatch.from_pydict({"a": pa.array([], type=pa.int64())})) is None


def test_dedup_reduces_worker_batch_size_for_low_cardinality_input(catalog: vp.VgiCatalog, monkeypatch) -> None:
    """`query_seed` is safe to dedup within one call.

    It's `FunctionStability.CONSISTENT_WITHIN_QUERY`. Spy on the underlying
    exchange call to confirm the worker actually receives the deduped
    (smaller) batch, not the raw one.
    """
    query_seed = catalog.scalar_function("main", "query_seed")

    exchange_client = catalog._exchange_client()
    real_scalar_function = exchange_client.scalar_function
    seen_batch_sizes: list[int] = []

    def spying_scalar_function(*, input, **kwargs):
        batches = list(input)
        seen_batch_sizes.append(sum(b.num_rows for b in batches))
        return real_scalar_function(input=iter(batches), **kwargs)

    monkeypatch.setattr(exchange_client, "scalar_function", spying_scalar_function)

    # 8 rows, only 2 distinct values.
    df = pl.DataFrame({"value": [1, 1, 1, 1, 2, 2, 2, 2]})
    out = df.with_columns(query_seed(pl.col("value")).alias("result"))

    assert seen_batch_sizes == [2], f"expected the worker to see 2 distinct rows, saw {seen_batch_sizes}"
    assert out["result"].to_list() == [1001, 1001, 1001, 1001, 1002, 1002, 1002, 1002]


def test_dedup_can_be_disabled_explicitly(catalog: vp.VgiCatalog, monkeypatch) -> None:
    query_seed = catalog.scalar_function("main", "query_seed")

    exchange_client = catalog._exchange_client()
    real_scalar_function = exchange_client.scalar_function
    seen_batch_sizes: list[int] = []

    def spying_scalar_function(*, input, **kwargs):
        batches = list(input)
        seen_batch_sizes.append(sum(b.num_rows for b in batches))
        return real_scalar_function(input=iter(batches), **kwargs)

    monkeypatch.setattr(exchange_client, "scalar_function", spying_scalar_function)

    df = pl.DataFrame({"value": [1, 1, 1, 1]})
    df.with_columns(query_seed(pl.col("value"), dedup=False).alias("result"))

    assert seen_batch_sizes == [4], f"expected dedup=False to ship the full batch, saw {seen_batch_sizes}"


def test_volatile_function_never_deduped(catalog: vp.VgiCatalog, monkeypatch) -> None:
    """`random_int` is VOLATILE, so every duplicate row must still reach the worker.

    It's `FunctionStability.VOLATILE` — even with `dedup=True` (the default),
    every duplicate row must still reach the worker.
    """
    random_int = catalog.scalar_function("main", "random_int")

    exchange_client = catalog._exchange_client()
    real_scalar_function = exchange_client.scalar_function
    seen_batch_sizes: list[int] = []

    def spying_scalar_function(*, input, **kwargs):
        batches = list(input)
        seen_batch_sizes.append(sum(b.num_rows for b in batches))
        return real_scalar_function(input=iter(batches), **kwargs)

    monkeypatch.setattr(exchange_client, "scalar_function", spying_scalar_function)

    # 5 rows, identical (min, max) on every row — would dedup to 1 if this
    # function were (wrongly) treated as safe.
    df = pl.DataFrame({"lo": [1, 1, 1, 1, 1], "hi": [10, 10, 10, 10, 10]})
    out = df.with_columns(random_int(pl.col("lo"), pl.col("hi")).alias("result"))

    assert seen_batch_sizes == [5], f"expected VOLATILE to ship all 5 rows undeduped, saw {seen_batch_sizes}"
    assert out.height == 5
