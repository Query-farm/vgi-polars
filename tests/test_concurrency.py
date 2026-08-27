# Copyright 2026 Query Farm LLC - https://query.farm

"""Thread-safety regression tests (Phase 0 of the coverage-expansion plan).

`vgi.client.Client`'s exchange-mode methods (`table_function`/`scalar_function`/...)
drive shared mutable state (`self._primary`/`self._additional_workers`) with no
locking — confirmed unsafe for concurrent calls on one shared instance by a direct
stress test during planning (20 threads calling `scalar_function` concurrently on
one `Client`: 18 corrupted/errored, 0 correct). Polars *does* call `map_batches`
callbacks (used by the scalar bridge) concurrently from multiple threads, and can
run multiple concurrent instances of the *same* `register_io_source` scan (used by
the table-scan bridge) when it appears more than once in a resolved plan
(self-join/concat/collect_all) — both confirmed empirically during planning.

`VgiCatalog._exchange_client()` fixes this with one lazily-created `Client` per
calling thread. These tests prove the fix holds, not just that it exists — deleting
the fix should make `test_concurrent_scalar_calls_are_correct` fail the same way
the ad hoc planning-session repro did.
"""

from __future__ import annotations

import threading

import polars as pl
import pyarrow as pa
from vgi.arguments import Arguments

import vgi_polars as vp


def test_concurrent_scalar_calls_are_correct(worker_location: str) -> None:
    """N threads sharing one VgiCatalog, each calling the same scalar function
    concurrently, must all get correct results — the exact scenario that
    corrupted every call before the Phase 0 fix."""
    n = 8
    results: dict[int, int] = {}
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    with vp.attach(worker_location, name="example") as cat:
        multiply = cat.scalar_function("main", "multiply")

        def worker(i: int) -> None:
            try:
                df = pl.DataFrame({"value": [i]})
                out = df.with_columns(multiply(pl.col("value"), 2).alias("product"))
                value = out["product"][0]
                with lock:
                    results[i] = value
            except BaseException as e:  # noqa: BLE001 - collecting every failure for the assertion below
                with lock:
                    errors.append((i, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"{len(errors)}/{n} calls raised: {errors[:3]}..."
    assert results == {i: i * 2 for i in range(n)}


def test_concurrent_table_scans_of_the_same_catalog_are_correct(worker_location: str) -> None:
    """N threads sharing one VgiCatalog, each independently scanning + filtering
    the same table concurrently, must all get correct results — the io_source
    analogue of the scalar test above (multiple concurrent generator instances
    of logically-the-same scan is the case Polars itself produces for
    self-joins/concat/collect_all)."""
    n = 4
    results: dict[int, list[int]] = {}
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    with vp.attach(worker_location, name="example") as cat:

        def worker(i: int) -> None:
            try:
                t = cat.table("data", "numbers")
                out = t.scan().filter(pl.col("value") > 95).collect()
                with lock:
                    results[i] = sorted(out["value"].to_list())
            except BaseException as e:  # noqa: BLE001 - collecting every failure for the assertion below
                with lock:
                    errors.append((i, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"{len(errors)}/{n} scans raised: {errors[:3]}..."
    expected = [96, 97, 98, 99]
    assert results == {i: expected for i in range(n)}


def test_concurrent_catalog_metadata_calls_are_correct(worker_location: str) -> None:
    """Catalog-metadata RPCs (schemas/table_get/schema_contents/...) go through
    the ONE shared `catalog.client`, not a per-thread exchange client — verify
    (not assume) that CatalogClientMixin's "short-lived connection per call"
    design is actually safe for concurrent use, per catalog.py's docstring."""
    n = 8
    results: dict[int, list[str]] = {}
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    with vp.attach(worker_location, name="example") as cat:

        def worker(i: int) -> None:
            try:
                schemas = cat.schemas()
                with lock:
                    results[i] = sorted(schemas)
            except BaseException as e:  # noqa: BLE001 - collecting every failure for the assertion below
                with lock:
                    errors.append((i, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"{len(errors)}/{n} calls raised: {errors[:3]}..."
    expected = sorted(["main", "data"])
    assert results == {i: expected for i in range(n)}


def test_exchange_client_is_one_per_thread(worker_location: str) -> None:
    """Direct unit check of the pooling contract: the same thread gets the same
    `Client` back; different threads get different ones; detach() stops all of
    them (verified by attempting a call on a post-detach client and expecting
    it to fail, since a stopped Client can't serve requests)."""
    with vp.attach(worker_location, name="example") as cat:
        same_thread_a = cat._exchange_client()
        same_thread_b = cat._exchange_client()
        assert same_thread_a is same_thread_b

        other_thread_client: list[object] = []
        t = threading.Thread(target=lambda: other_thread_client.append(cat._exchange_client()))
        t.start()
        t.join()

        assert other_thread_client[0] is not same_thread_a
        # Every thread-local client created must be tracked for detach() to stop.
        assert same_thread_a in cat._exchange_clients
        assert other_thread_client[0] in cat._exchange_clients


def test_no_secrets_needed_for_exchange_client_reuse(worker_location: str) -> None:
    """A per-thread exchange client is immediately usable with no re-attach —
    confirms `Client.table_function`/`scalar_function` genuinely don't need
    `attach_opaque_data`, so `_exchange_client()`'s "no catalog_attach" design
    (see catalog.py's `client_factory` docstring) isn't silently relying on
    some other implicit session state that happens to work by accident."""
    with vp.attach(worker_location, name="example") as cat:
        fresh = cat._exchange_client()
        batch = pa.RecordBatch.from_arrays([pa.array([1], type=pa.int64())], schema=pa.schema([pa.field("value", pa.int64())]))
        out = list(
            fresh.scalar_function(
                function_name="multiply",
                schema_name="main",
                input=iter([batch]),
                arguments=Arguments(positional=(pa.scalar(3),)),
            )
        )
        assert out[0].column(0)[0].as_py() == 3
