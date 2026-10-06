# Copyright 2026 Query Farm LLC - https://query.farm

"""Filter pushdown translation, and the Design Principle 1 regression test.

Even when the (here, deliberately faked) worker completely ignores the
pushed-down predicate, `VgiTable.scan()` must still return the exactly
correct rows — because `_source.py` always re-applies the original predicate
locally, unconditionally. See CLAUDE.md's "Design Principle 1" section.
"""

from __future__ import annotations

import datetime
import json
from decimal import Decimal

import polars as pl
import pyarrow as pa
from vgi import deserialize_filter_batch
from vgi.client.client import Client
from vgi.filter_v2 import FilterState, PredicateMode, deserialize_snapshot

import vgi_polars as vp
from vgi_polars._filter_translate import translate_predicate


def _schema(**fields: pa.DataType) -> pa.Schema:
    return pa.schema([pa.field(name, data_type) for name, data_type in fields.items()])


def _decode(ipc_bytes: bytes | None) -> tuple[dict, pa.RecordBatch, FilterState]:
    """Decode pushdown bytes the way a worker does: the batch, its JSON document, and the strict v2 parse.

    `deserialize_snapshot` is vgi-python's own consumer — the same strict parser a
    worker runs — so a document it accepts is one a worker accepts structurally.
    (Binding against the output schema needs the embedded DuckDB evaluator, which
    only the worker's environment has; the end-to-end tests below cover that.)
    """
    assert ipc_bytes is not None, "expected the predicate to translate"
    batch = deserialize_filter_batch(ipc_bytes)
    return json.loads(batch.column(0)[0].as_py()), batch, deserialize_snapshot(batch)


def _expressions(document: dict) -> list[dict]:
    return [p["expression"] for p in document["predicates"]]


def test_translate_predicate_flat_and() -> None:
    pred = (pl.col("n") > 3) & (pl.col("n") < 8)
    document, batch, state = _decode(translate_predicate(pred, _schema(n=pa.int64(), s=pa.string())))

    assert document["encoding"] == "vgi.filters.v2"
    assert document["semantics"] == "vgi.duckdb.standard.v1"
    assert document["kind"] == "snapshot"
    # One advisory predicate per conjunct (spec section 8.3: the client keeps its
    # own exact filter, so each conjunct is an independent pruning hint).
    assert [(p["id"], p["mode"], p["source"], p["revision"]) for p in document["predicates"]] == [
        ("query:0", "advisory", "query", 0),
        ("query:1", "advisory", "query", 0),
    ]
    column = {"node": "column_ref", "column_index": 0, "column_name": "n"}
    assert _expressions(document) == [
        {"node": "comparison", "op": "gt", "left": column, "right": {"node": "literal", "value_ref": 0}},
        {"node": "comparison", "op": "lt", "left": column, "right": {"node": "literal", "value_ref": 1}},
    ]
    assert batch.schema.field("filter_spec").nullable is False
    assert batch.schema.metadata[b"vgi_evaluation_context"] == b"vgi.none.v1"
    assert batch.to_pydict()["value_0"] == [3]
    assert batch.to_pydict()["value_1"] == [8]
    assert all(p.mode is PredicateMode.ADVISORY for p in state.predicates)


def test_translate_predicate_flipped_comparison() -> None:
    document, _, _ = _decode(translate_predicate(pl.lit(5) < pl.col("n"), _schema(n=pa.int64())))
    assert _expressions(document)[0]["op"] == "gt"


def test_translate_predicate_is_null_and_is_not_null() -> None:
    pred = pl.col("s").is_null() & pl.col("n").is_not_null()
    document, _, _ = _decode(translate_predicate(pred, _schema(n=pa.int64(), s=pa.string())))
    assert _expressions(document) == [
        {
            "node": "is_null",
            "expression": {"node": "column_ref", "column_index": 1, "column_name": "s"},
            "negated": False,
        },
        {
            "node": "is_null",
            "expression": {"node": "column_ref", "column_index": 0, "column_name": "n"},
            "negated": True,
        },
    ]


def test_translate_predicate_unsupported_returns_none() -> None:
    # `OR` isn't part of the supported grammar (only a top-level AND chain
    # is) — translate_predicate must decline cleanly, not raise.
    assert translate_predicate((pl.col("n") > 3) | (pl.col("n") < 1), _schema(n=pa.int64())) is None


def test_translate_predicate_keeps_translatable_conjuncts_of_a_partial_and() -> None:
    pred = ((pl.col("n") > 3) | (pl.col("n") < 1)) & (pl.col("n") != 7)
    document, _, _ = _decode(translate_predicate(pred, _schema(n=pa.int64())))
    assert [e["op"] for e in _expressions(document)] == ["ne"]


def test_translate_predicate_is_in() -> None:
    pred = pl.col("status").is_in(["active", "pending", "review"])
    document, batch, state = _decode(translate_predicate(pred, _schema(status=pa.string(), n=pa.int64())))
    assert _expressions(document) == [
        {
            "node": "in",
            "expression": {"node": "column_ref", "column_index": 0, "column_name": "status"},
            "set": {"kind": "literal", "value_ref": 0},
            "negated": False,
        }
    ]
    # A literal IN set is ONE Arrow list scalar holding every candidate.
    assert batch.to_pydict()["value_0"] == [["active", "pending", "review"]]
    assert pa.types.is_list(batch.schema.field("value_0").type)
    assert len(state.predicates) == 1


def test_translate_predicate_is_in_nulls_equal_declines() -> None:
    """`nulls_equal=True` changes NULL-matching semantics SQL `IN` can't express.

    Declines rather than risk a worker excluding rows that should have
    matched (falls back to local filtering, per Design Principle 1).
    """
    assert translate_predicate(pl.col("n").is_in([1, 2, 3], nulls_equal=True), _schema(n=pa.int64())) is None


def test_translate_predicate_is_in_composed_with_and() -> None:
    pred = pl.col("a").is_in([1, 2, 3]) & (pl.col("b") > 5)
    document, batch, _ = _decode(translate_predicate(pred, _schema(a=pa.int64(), b=pa.int64())))
    assert [e["node"] for e in _expressions(document)] == ["in", "comparison"]
    assert _expressions(document)[1]["left"]["column_index"] == 1
    assert batch.to_pydict()["value_0"] == [[1, 2, 3]]
    assert batch.to_pydict()["value_1"] == [5]


def test_translate_predicate_names_columns_by_the_bound_schema() -> None:
    """The predicate uses the catalog's declared names; the wire uses the bind output's.

    v2 validates a column reference's name against the bind output schema at
    its index, so `data.numbers` (declared `value`, function emits `n`) must
    send `n`, at the declared column's position.
    """
    document, _, _ = _decode(translate_predicate(pl.col("value") > 95, _schema(n=pa.int64()), ["value"]))
    assert _expressions(document)[0]["left"] == {"node": "column_ref", "column_index": 0, "column_name": "n"}


def test_translate_predicate_declines_unknown_and_misaligned_columns() -> None:
    assert translate_predicate(pl.col("missing") > 1, _schema(n=pa.int64())) is None
    # Declared names that don't line up with the bound schema: nothing is safe to send.
    assert translate_predicate(pl.col("a") > 1, _schema(n=pa.int64()), ["a", "b"]) is None


def test_translate_predicate_declines_type_incompatible_literal() -> None:
    """A comparison the worker could not bind is not pushed (it would fail the scan, not just miss)."""
    assert translate_predicate(pl.col("s") > 5, _schema(s=pa.string())) is None
    assert translate_predicate(pl.col("n") == "x", _schema(n=pa.int64())) is None
    assert translate_predicate(pl.col("s").is_in([1, 2]), _schema(s=pa.string())) is None


def _pushed_value(pred: pl.Expr, schema: pa.Schema) -> pa.Scalar:
    _, batch, _ = _decode(translate_predicate(pred, schema))
    return batch.column("value_0")[0]


def test_translate_predicate_date() -> None:
    d = datetime.date(2024, 1, 1)
    value = _pushed_value(pl.col("d") > d, _schema(d=pa.date32()))
    assert value.type == pa.date32()
    assert value.as_py() == d


def test_translate_predicate_datetime() -> None:
    dt = datetime.datetime(2024, 1, 1, 12, 30, 0)
    assert _pushed_value(pl.col("dt") > dt, _schema(dt=pa.timestamp("us"))).as_py() == dt


def _as_received_by_a_scan(pred: pl.Expr, schema: dict[str, pl.DataType]) -> pl.Expr:
    """`pred` the way a real `io_source` receives it: type-resolved against the scan schema."""
    from polars.io.plugins import register_io_source

    received: list[pl.Expr | None] = []

    def source(with_columns, predicate, n_rows, batch_size):
        received.append(predicate)
        yield pl.DataFrame(schema=schema)

    register_io_source(source, schema=pl.Schema(schema)).filter(pred).collect()
    assert received and received[0] is not None
    return received[0]


def test_translate_predicate_datetime_keeps_nanosecond_precision() -> None:
    """A nanosecond literal is sent as nanoseconds, never truncated.

    Truncation is not a harmless miss: `t < 1.0000005us` truncated to `t <
    1.000000us` is STRICTER than the original, so a worker applying it would
    drop rows the local re-filter can never get back.
    """
    ns = 1_704_067_200_000_000_005
    pred = _as_received_by_a_scan(pl.col("t") < pl.lit(ns, dtype=pl.Datetime("ns")), {"t": pl.Datetime("ns")})
    value = _pushed_value(pred, _schema(t=pa.timestamp("ns")))
    assert value.type == pa.timestamp("ns")
    assert value.cast(pa.int64()).as_py() == ns


def test_translate_predicate_uses_the_resolved_literal_type() -> None:
    """Through a real scan, Polars resolves an int literal to the column's dtype; the payload keeps it."""
    pred = _as_received_by_a_scan(pl.col("i") > 5, {"i": pl.Int32})
    assert _pushed_value(pred, _schema(i=pa.int32())).type == pa.int32()


def test_translate_predicate_duration() -> None:
    dur = datetime.timedelta(days=1, hours=2)
    assert _pushed_value(pl.col("dur") > dur, _schema(dur=pa.duration("us"))).as_py() == dur


def test_translate_predicate_binary() -> None:
    assert _pushed_value(pl.col("b") == b"hello", _schema(b=pa.binary())).as_py() == b"hello"


def test_translate_predicate_decimal() -> None:
    value = Decimal("1.50")
    pushed = _pushed_value(pl.col("x") > value, _schema(x=pa.decimal128(10, 2)))
    assert pa.types.is_decimal(pushed.type)
    assert pushed.as_py() == value


def test_translate_predicate_timezone_aware_datetime_declines() -> None:
    """A time-zone-aware literal is not pushed.

    Comparing it correctly depends on time-zone resolution, which is DuckDB
    session state this client never sends (it uses the `vgi.none.v1`
    evaluation context) — so it falls back to local filtering.
    """
    dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
    assert translate_predicate(pl.col("dt") > dt, _schema(dt=pa.timestamp("us", tz="UTC"))) is None


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

    # Table scans borrow a fresh Client per scan (VgiCatalog._exchange_client()),
    # not the shared catalog.client — so patch the class.
    real_table_function = Client.table_function
    calls = []

    def spying_table_function(self, *, projection_ids=None, pushdown_filters=None, **kwargs):
        # Record that pushdown was attempted...
        calls.append((projection_ids, pushdown_filters))
        # ...then deliberately call through WITHOUT any pushdown args, so the
        # "worker" ignores whatever was requested and returns everything.
        return real_table_function(self, projection_ids=None, pushdown_filters=None, **kwargs)

    monkeypatch.setattr(Client, "table_function", spying_table_function)

    # Filter only (no .select() — combining both in one call lets Polars
    # collapse with_columns to None when it's a no-op single-column select,
    # which would make this assertion flaky about *which* pushdown fired).
    out = t.scan().filter(pl.col("value") > 95).collect()

    assert calls, "pushdown was never attempted — test isn't exercising the fake worker"
    assert calls[0][1] is not None, "expected the predicate to actually be translated and pushed"
    assert sorted(out["value"].to_list()) == [96, 97, 98, 99]


def test_filter_pushdown_uses_bound_column_names_end_to_end(catalog: vp.VgiCatalog, monkeypatch) -> None:
    """`data.numbers` declares column `value`; its scan function binds it as `n`.

    The pushed v2 document must name `n` — the worker validates each column
    reference's name against its bind output and rejects a mismatch, failing
    the whole scan. Spies on (but passes through) the real pushdown so the
    worker really receives and validates the document.
    """
    t = catalog.table("data", "numbers")
    fake_info = type(
        "FakeInfo", (), {"projection_pushdown": False, "filter_pushdown": True, "supports_splits": False}
    )()
    monkeypatch.setattr(t, "_function_info_get", lambda: fake_info)

    real_table_function = Client.table_function
    sent: list[bytes | None] = []

    def spying_table_function(self, *, pushdown_filters=None, **kwargs):
        sent.append(pushdown_filters)
        return real_table_function(self, pushdown_filters=pushdown_filters, **kwargs)

    monkeypatch.setattr(Client, "table_function", spying_table_function)

    out = t.scan().filter(pl.col("value") > 95).collect()

    assert sorted(out["value"].to_list()) == [96, 97, 98, 99]
    document, _, _ = _decode(sent[0])
    assert _expressions(document)[0]["left"] == {"node": "column_ref", "column_index": 0, "column_name": "n"}
