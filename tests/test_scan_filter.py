# Copyright 2026 Query Farm LLC - https://query.farm

"""Filter pushdown translation, and the Design Principle 1 regression test.

Even when the (here, deliberately faked) worker completely ignores the
pushed-down predicate, `VgiTable.scan()` must still return the exactly
correct rows — because `_source.py` always re-applies the original predicate
locally, unconditionally. See CLAUDE.md's "Design Principle 1" section.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import polars as pl
import pyarrow as pa

import vgi_polars as vp
from vgi_polars._filter_translate import translate_predicate


def test_translate_predicate_flat_and() -> None:
    pred = (pl.col("n") > 3) & (pl.col("n") < 8)
    ipc_bytes = translate_predicate(pred, ["n", "s"])
    assert ipc_bytes is not None

    reader = pa.ipc.open_stream(ipc_bytes)
    batch = reader.read_next_batch()
    row = batch.to_pydict()
    assert row["filter_spec"] == [
        (
            '[{"column_name": "n", "column_index": 0, "type": "constant", "op": "gt", "value_ref": 0}, '
            '{"column_name": "n", "column_index": 0, "type": "constant", "op": "lt", "value_ref": 1}]'
        )
    ]
    assert row["value_0"] == [3]
    assert row["value_1"] == [8]


def test_translate_predicate_is_null() -> None:
    ipc_bytes = translate_predicate(pl.col("s").is_null(), ["n", "s"])
    assert ipc_bytes is not None
    row = pa.ipc.open_stream(ipc_bytes).read_next_batch().to_pydict()
    assert '"type": "is_null"' in row["filter_spec"][0]


def test_translate_predicate_unsupported_returns_none() -> None:
    # `OR` isn't part of the supported grammar (only a top-level AND chain
    # is) — translate_predicate must decline cleanly, not raise.
    assert translate_predicate((pl.col("n") > 3) | (pl.col("n") < 1), ["n"]) is None


def test_translate_predicate_is_in() -> None:
    ipc_bytes = translate_predicate(pl.col("status").is_in(["active", "pending", "review"]), ["status", "n"])
    assert ipc_bytes is not None

    batch = pa.ipc.open_stream(ipc_bytes).read_next_batch()
    row = batch.to_pydict()
    assert row["filter_spec"] == ['[{"column_name": "status", "column_index": 0, "type": "in", "value_ref": 0}]']
    # A single row whose value is the list of candidates, per the wire spec's
    # `_val_0: ["active", "pending", "review"]` example — not one row per
    # candidate value.
    assert row["value_0"] == [["active", "pending", "review"]]
    assert pa.types.is_list(batch.schema.field("value_0").type)


def test_translate_predicate_is_in_nulls_equal_declines() -> None:
    """`nulls_equal=True` changes NULL-matching semantics the plain wire IN filter can't express.

    Declines rather than risk a worker excluding rows that should have
    matched (falls back to local filtering, per Design Principle 1).
    """
    assert translate_predicate(pl.col("n").is_in([1, 2, 3], nulls_equal=True), ["n"]) is None


def test_translate_predicate_is_in_composed_with_and() -> None:
    pred = pl.col("a").is_in([1, 2, 3]) & (pl.col("b") > 5)
    ipc_bytes = translate_predicate(pred, ["a", "b"])
    assert ipc_bytes is not None

    row = pa.ipc.open_stream(ipc_bytes).read_next_batch().to_pydict()
    assert row["filter_spec"] == [
        (
            '[{"column_name": "a", "column_index": 0, "type": "in", "value_ref": 0}, '
            '{"column_name": "b", "column_index": 1, "type": "constant", "op": "gt", "value_ref": 1}]'
        )
    ]
    assert row["value_0"] == [[1, 2, 3]]
    assert row["value_1"] == [5]


def _pushed_value(pred: pl.Expr, columns: list[str]):
    ipc_bytes = translate_predicate(pred, columns)
    assert ipc_bytes is not None, f"expected {pred} to translate"
    row = pa.ipc.open_stream(ipc_bytes).read_next_batch().to_pydict()
    return row["value_0"][0]


def test_translate_predicate_date() -> None:
    d = datetime.date(2024, 1, 1)
    assert _pushed_value(pl.col("d") > d, ["d"]) == d


def test_translate_predicate_datetime() -> None:
    dt = datetime.datetime(2024, 1, 1, 12, 30, 0)
    assert _pushed_value(pl.col("dt") > dt, ["dt"]) == dt


def test_translate_predicate_duration() -> None:
    dur = datetime.timedelta(days=1, hours=2)
    assert _pushed_value(pl.col("dur") > dur, ["dur"]) == dur


def test_translate_predicate_binary() -> None:
    assert _pushed_value(pl.col("b") == b"hello", ["b"]) == b"hello"


def test_translate_predicate_decimal() -> None:
    value = Decimal("1.50")
    assert _pushed_value(pl.col("x") > value, ["x"]) == value


def test_translate_predicate_timezone_aware_datetime_declines() -> None:
    """Declines rather than risk a silently-wrong timezone comparison.

    A tz name alone isn't enough to reconstruct a correct offset without a
    tzdata dependency — declines rather than risk a silently-wrong comparison
    (falls back to local filtering, per Design Principle 1).
    """
    dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    assert translate_predicate(pl.col("dt") > dt, ["dt"]) is None


def test_filter_pushdown_end_to_end(catalog: vp.VgiCatalog) -> None:
    """A pushdown-eligible predicate against a real filter-pushdown worker produces correct rows.

    Against the real `filter_echo` fixture (declares filter_pushdown=True),
    a pushdown-eligible predicate still produces exactly correct rows.
    """
    t = catalog.table("data", "filter_echo_table")
    out = t.scan().filter((pl.col("n") > 3) & (pl.col("n") < 8)).collect()
    assert sorted(out["n"].to_list()) == [4, 5, 6, 7]


def test_filter_pushdown_end_to_end_is_in(catalog: vp.VgiCatalog) -> None:
    """An `is_in` predicate against a real filter-pushdown worker produces correct rows AND was actually sent.

    `filter_echo_table` echoes the SQL-like rendering of whatever filters it
    received into its own `pushed_filters` output column — asserting on its
    content (not just the final row set) proves the `is_in` filter actually
    reached the worker, rather than only being locally correct regardless
    (which the final row set alone would be either way, per Design
    Principle 1).
    """
    t = catalog.table("data", "filter_echo_table")
    out = t.scan().filter(pl.col("n").is_in([4, 6, 91])).collect()
    assert sorted(out["n"].to_list()) == [4, 6, 91]
    assert set(out["pushed_filters"].to_list()) == {"n IN (4, 6, 91)"}


def test_local_refilter_survives_a_worker_that_ignores_pushdown(catalog: vp.VgiCatalog, monkeypatch) -> None:
    """Design Principle 1 regression test.

    Monkeypatches the underlying `Client.table_function` to return ALL rows
    unfiltered/unprojected no matter what `projection_ids`/`pushdown_filters`
    it's called with — simulating a worker that declared pushdown support but
    doesn't honor it. `VgiTable.scan()` must still return exactly the right
    rows.
    """
    t = catalog.table("data", "numbers")
    # Force pushdown to actually be attempted (numbers' own scan function may
    # not declare support) by making _function_info_get() report full support.
    fake_info = type("FakeInfo", (), {"projection_pushdown": True, "filter_pushdown": True, "supports_splits": False})()
    monkeypatch.setattr(t, "_function_info_get", lambda: fake_info)

    # Table scans go through the per-thread exchange client
    # (VgiCatalog._exchange_client()), not the shared catalog.client — patch
    # that one, matching the pytest test-thread's own lazily-created instance.
    exchange_client = catalog._exchange_client()
    real_table_function = exchange_client.table_function
    calls = []

    def spying_table_function(*, projection_ids=None, pushdown_filters=None, **kwargs):
        # Record that pushdown was attempted...
        calls.append((projection_ids, pushdown_filters))
        # ...then deliberately call through WITHOUT any pushdown args, so the
        # "worker" ignores whatever was requested and returns everything.
        return real_table_function(projection_ids=None, pushdown_filters=None, **kwargs)

    monkeypatch.setattr(exchange_client, "table_function", spying_table_function)

    # Filter only (no .select() — combining both in one call lets Polars
    # collapse with_columns to None when it's a no-op single-column select,
    # which would make this assertion flaky about *which* pushdown fired).
    out = t.scan().filter(pl.col("value") > 95).collect()

    assert calls, "pushdown was never attempted — test isn't exercising the fake worker"
    assert calls[0][1] is not None, "expected the predicate to actually be translated and pushed"
    assert sorted(out["value"].to_list()) == [96, 97, 98, 99]
