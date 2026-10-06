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

`VgiCatalog._exchange_client()` fixes this by borrowing a **fresh** `Client` for
exactly the duration of one exchange-mode operation, then returning it — no
shared mutable state is ever touched by more than one caller at a time, so this
is thread-safe by construction, not by locking (see catalog.py's module
docstring for the full rationale, including why this is also cheap: it reuses
vgi-python's own subprocess `WorkerPool` rather than caching one `Client` per
thread the way an earlier version of this fix did). These tests prove the fix
holds, not just that it exists — deleting it should make
`test_concurrent_scalar_calls_are_correct` fail the same way the ad hoc
planning-session repro did.
"""

from __future__ import annotations

import threading

import polars as pl
import pyarrow as pa
from vgi.arguments import Arguments

import vgi_polars as vp


def test_concurrent_scalar_calls_are_correct(worker_location: str) -> None:
    """N threads sharing one VgiCatalog, each calling the same scalar function concurrently.

    Must all get correct results — the exact scenario that corrupted every
    call before the Phase 0 fix.
    """
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
    """N threads sharing one VgiCatalog, each independently scanning + filtering the same table concurrently.

    Must all get correct results — the io_source analogue of the scalar
    test above (multiple concurrent generator instances of
    logically-the-same scan is the case Polars itself produces for
    self-joins/concat/collect_all).
    """
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
    """Catalog-metadata RPCs (schemas/table_get/schema_contents/...) go through the ONE shared `catalog.client`.

    Not a per-thread exchange client — verify (not assume) that
    CatalogClientMixin's "short-lived connection per call" design is
    actually safe for concurrent use, per catalog.py's docstring.
    """
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


def test_exchange_client_borrows_fresh_each_time(worker_location: str) -> None:
    """Direct unit check of the borrow-and-return contract (see catalog.py's module docstring).

    Every `with cat._exchange_client() as client:` gets a genuinely fresh
    `Client` — even two sequential calls on the *same* thread, not just
    across different threads (the old per-thread-cached design would have
    returned the same object for the two same-thread borrows below). Each
    is stopped — returned to vgi-python's own `WorkerPool` — the moment its
    `with` block exits; `Client.stop()` clears `_primary`, so checking that
    is a direct signal the borrow was actually released, not just that the
    object went out of scope.
    """
    with vp.attach(worker_location, name="example") as cat:
        with cat._exchange_client() as client_a:
            pass
        with cat._exchange_client() as client_b:
            pass
        assert client_a is not client_b
        assert client_a._primary is None
        assert client_b._primary is None

        other_thread_client: list[object] = []

        def borrow_and_release() -> None:
            with cat._exchange_client() as c:
                other_thread_client.append(c)

        t = threading.Thread(target=borrow_and_release)
        t.start()
        t.join()

        assert other_thread_client[0] is not client_a
        assert other_thread_client[0] is not client_b


def test_no_secrets_needed_for_exchange_client_reuse(worker_location: str) -> None:
    """A borrowed exchange client is immediately usable with no re-attach.

    Confirms `Client.table_function`/`scalar_function` genuinely don't need
    `attach_opaque_data`, so `_exchange_client()`'s "no catalog_attach"
    design (see catalog.py's `client_factory` docstring) isn't silently
    relying on some other implicit session state that happens to work by
    accident.
    """
    with vp.attach(worker_location, name="example") as cat:
        batch = pa.RecordBatch.from_arrays(
            [pa.array([1], type=pa.int64())], schema=pa.schema([pa.field("value", pa.int64())])
        )
        with cat._exchange_client() as fresh:
            out = list(
                fresh.scalar_function(
                    function_name="multiply",
                    schema_path=["main"],
                    input=iter([batch]),
                    arguments=Arguments(positional=(pa.scalar(3),)),
                )
            )
        assert out[0].column(0)[0].as_py() == 3
