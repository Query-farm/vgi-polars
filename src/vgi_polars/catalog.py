# Copyright 2026 Query Farm LLC - https://query.farm

"""`VgiCatalog` — an attached VGI catalog, and the `attach()` entry point.

Wraps `vgi.client.Client` (vgi-python's pure-Python, Arrow-native reference
client — the same wire-protocol implementation the DuckDB extension speaks, just
without DuckDB). See this package's README/CLAUDE.md for the architectural
rationale: vgi-polars is an adapter over that existing client, not a new
protocol implementation.

**Thread safety.** `vgi.client.Client`'s exchange-mode methods (`table_function`,
`scalar_function`, and friends) drive shared mutable state (`self._primary` /
`self._additional_workers`) with no locking — confirmed unsafe for concurrent use
on one shared instance by direct stress test (20 threads calling
`scalar_function` concurrently on one `Client`: 18 corrupted/errored, 0 correct).
This matters because Polars *does* call `LazyFrame.map_batches` callbacks (used
by the scalar-function bridge, `_scalar.py`) concurrently from multiple threads,
and can run multiple concurrent instances of a `register_io_source` scan (used by
`_source.py`) when the same scan appears more than once in a resolved plan
(self-join, `concat`, `collect_all`) — both confirmed empirically. So every
exchange-mode call site uses `VgiCatalog._exchange_client()` (one lazily-created
`Client` per calling thread, all sharing the one `attach_opaque_data` from the
catalog's single `catalog_attach` — mirrors the DuckDB C++ extension's own
solved pattern: one attach, many pooled per-thread connections), never the
catalog's single shared `Client` directly. Catalog-metadata methods (`schemas`,
`table_get`, `schema_contents`, `table_scan_function_get`,
`table_column_statistics`) keep using the shared `Client` — `CatalogClientMixin`
opens a short-lived connection per call rather than reusing `self._primary`, so
they don't have this hazard (see `tests/test_concurrency.py`, which verifies
this rather than assuming it).
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any, Literal, Self

from vgi.catalog.catalog_interface import CatalogAttachResult, SchemaObjectType
from vgi.client.client import Client

from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from collections.abc import Callable

    from vgi.catalog.catalog_interface import AttachOpaqueData

    from vgi_polars._aggregate import AggregateFunction
    from vgi_polars._scalar import ScalarFunctionCall
    from vgi_polars._table_in_out import TableInOutFunction
    from vgi_polars.table import VgiTable

__all__ = ["VgiCatalog", "attach"]

Transport = Literal["subprocess", "http", "tcp"]


class VgiCatalog:
    """An attached VGI catalog. Construct via `attach()`, not directly."""

    def __init__(
        self, *, client: Client, client_factory: Callable[[], Client], name: str, attach_result: CatalogAttachResult
    ) -> None:
        """Wrap an already-attached `client`/`attach_result` pair. Use `attach()`, not this directly."""
        self._client = client
        self._client_factory = client_factory
        self._name = name
        self._attach_result = attach_result
        self._detached = False
        # One `Client` per calling thread for exchange-mode RPCs (table_function/
        # scalar_function/...) — see the module docstring's "Thread safety" section.
        # `_exchange_clients` tracks every one ever created so `detach()` can stop
        # them all; guarded by `_exchange_clients_lock` since threads race to append.
        self._thread_local = threading.local()
        self._exchange_clients_lock = threading.Lock()
        self._exchange_clients: list[Client] = []

    @property
    def name(self) -> str:
        """The attach alias this catalog was attached under."""
        return self._name

    @property
    def attach_opaque_data(self) -> AttachOpaqueData:
        """The opaque attachment id every catalog-scoped RPC threads through."""
        return self._attach_result.attach_opaque_data

    @property
    def default_schema(self) -> str:
        """The catalog's default schema.

        The second place (after a table's own schema) `VgiTable` looks for its
        resolved scan function, mirroring the DuckDB C++ extension's own
        resolution order (`vgi_table_entry.cpp`: *"the worker registers
        function names per schema and may reuse one name across schemas, so
        the bind request has to name the schema we found it in — not just the
        table's"*).
        """
        return self._attach_result.default_schema

    @property
    def catalog_version(self) -> int:
        """The catalog's version at attach time.

        Bumps when schemas, tables, or other catalog objects change. Compare
        against a later `vgi_clear_cache()`-equivalent (there isn't a
        client-side metadata cache in vgi-polars to invalidate, but a
        worker-side version bump is still a useful staleness signal for a
        long-lived catalog handle).
        """
        return self._attach_result.catalog_version

    @property
    def catalog_version_frozen(self) -> bool:
        """Whether the worker asserts its catalog metadata is frozen for this attach.

        Covers schema/table/function metadata never changing for the lifetime
        of this attach (an optimization hint — vgi-polars doesn't currently
        cache catalog metadata client-side, so this is informational only,
        not yet load-bearing here).
        """
        return self._attach_result.catalog_version_frozen

    @property
    def supports_transactions(self) -> bool:
        """Whether the worker supports transactions for this catalog."""
        return self._attach_result.supports_transactions

    @property
    def supports_time_travel(self) -> bool:
        """Whether tables in this catalog support time travel (`AT` clauses).

        Note this is catalog-wide capability advertisement only — vgi-polars
        has no way to actually *perform* a time-travel scan even when this is
        `True`: `Client.table_get`/`Client.table_function` don't accept
        `at_unit`/`at_value` at all (only `Client.table_scan_function_get`
        does), an upstream vgi-python gap. See CLAUDE.md's Scope section.
        """
        return self._attach_result.supports_time_travel

    @property
    def resolved_data_version(self) -> str | None:
        """The concrete data version the worker resolved for this attach.

        `None` if the worker has no opinion, or if `data_version_spec` wasn't
        passed to `attach()`. See `attach()`'s `data_version_spec` parameter.
        """
        return self._attach_result.resolved_data_version

    @property
    def resolved_implementation_version(self) -> str | None:
        """The concrete implementation version the worker resolved for this attach.

        `None` if the worker has no opinion. See `attach()`'s
        `implementation_version` parameter.
        """
        return self._attach_result.resolved_implementation_version

    @property
    def comment(self) -> str | None:
        """An optional comment describing this catalog/database, if the worker set one."""
        return self._attach_result.comment

    @property
    def tags(self) -> dict[str, str]:
        """Optional key-value tags associated with this catalog/database."""
        return dict(self._attach_result.tags or {})

    @property
    def client(self) -> Client:
        """The underlying vgi-python `Client` used for **catalog-metadata** RPCs only.

        Covers `schemas`, `table_get`, `schema_contents`,
        `table_scan_function_get`, `table_column_statistics`. Not part of the
        stable public API. Exchange-mode calls (table/scalar/aggregate/
        table-in-out function invocation) must use `_exchange_client()`
        instead — see the module docstring's "Thread safety" section.
        """
        return self._client

    def _exchange_client(self) -> Client:
        """A `Client` safe for the calling thread's exclusive use, for exchange-mode RPCs.

        Covers `table_function`, `scalar_function`, and — once added —
        `aggregate_function`/`table_in_out_function`/`table_buffering_function`.
        Lazily creates and starts one per thread via `_client_factory`,
        reusing it across that thread's subsequent calls; never shared across
        threads. See the module docstring.
        """
        existing: Client | None = getattr(self._thread_local, "client", None)
        if existing is not None:
            return existing
        new_client = self._client_factory()
        new_client.start()
        self._thread_local.client = new_client
        with self._exchange_clients_lock:
            self._exchange_clients.append(new_client)
        return new_client

    def schemas(self) -> list[str]:
        """List schema names in this catalog."""
        try:
            infos = self._client.schemas(attach_opaque_data=self.attach_opaque_data)
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        return [s.name for s in infos]

    def tables(self, schema_name: str) -> list[str]:
        """List table names in `schema_name`."""
        try:
            infos = self._client.schema_contents(
                attach_opaque_data=self.attach_opaque_data,
                name=schema_name,
                type=SchemaObjectType.TABLE,
            )
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        return [t.name for t in infos]

    def table(
        self, schema_name: str, name: str, *, at_unit: str | None = None, at_value: str | None = None
    ) -> VgiTable:
        """A lazy handle to a catalog table.

        No RPC happens until `.schema`, `.scan()`, or `.read()` is used.

        `at_unit`/`at_value` request a time-travel view (e.g. `at_unit=
        "VERSION", at_value="3"`) — a worker that doesn't support it on this
        table rejects the request at bind, the same as any other unsupported
        bind option. A different AT clause is a different `VgiTable`
        instance (schema/scan-function resolution is cached per instance,
        never shared across AT clauses); call `table()` again to get another
        version, don't mutate one you already have.
        """
        from vgi_polars.table import VgiTable

        return VgiTable(catalog=self, schema_name=schema_name, name=name, at_unit=at_unit, at_value=at_value)

    def scalar_function(self, schema_name: str, name: str) -> ScalarFunctionCall:
        """A callable usable inside `pl.Expr.map_batches` (see `_scalar.py`)."""
        from vgi_polars._scalar import make_scalar_function

        return make_scalar_function(self, schema_name, name)

    def table_in_out_function(self, schema_name: str, name: str) -> TableInOutFunction:
        """Return a callable for a streaming or buffered table-in-out function.

        The returned callable has signature `fn(lf: pl.LazyFrame, *,
        settings=None) -> pl.LazyFrame` (see `_table_in_out.py`).
        """
        from vgi_polars._table_in_out import make_table_in_out_function

        return make_table_in_out_function(self, schema_name, name)

    def aggregate_function(self, schema_name: str, name: str) -> AggregateFunction:
        """Return an eager callable for an aggregate function.

        The returned callable has signature `fn(df: pl.DataFrame, *,
        group_by=(), ...) -> pl.DataFrame` (see `_aggregate.py`).
        """
        from vgi_polars._aggregate import make_aggregate_function

        return make_aggregate_function(self, schema_name, name)

    def detach(self) -> None:
        """Detach from the catalog and close the underlying client(s).

        Closes every per-thread exchange client `_exchange_client()` created.
        Safe to call more than once.
        """
        if self._detached:
            return
        self._detached = True
        try:
            try:
                self._client.catalog_detach(attach_opaque_data=self.attach_opaque_data)
            except VGI_CLIENT_ERRORS as e:
                raise VgiPolarsError(str(e)) from e
        finally:
            self._client.stop()
            with self._exchange_clients_lock:
                exchange_clients, self._exchange_clients = self._exchange_clients, []
            for exchange_client in exchange_clients:
                # Best-effort cleanup: one client's failure to stop must never
                # block stopping the rest.
                with contextlib.suppress(Exception):
                    exchange_client.stop()

    def __enter__(self) -> Self:
        """Support `with attach(...) as catalog:` — returns `self`."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Support `with attach(...) as catalog:` — calls `detach()`."""
        self.detach()


def _detect_transport(location: str) -> Transport:
    """Auto-detect transport from `location`'s scheme.

    Mirrors the DuckDB extension's LOCATION scheme table (`http://`/`https://`
    -> HTTP, `tcp://` -> TCP, anything else -> subprocess/shlex argv).
    """
    if location.startswith(("http://", "https://")):
        return "http"
    if location.startswith("tcp://"):
        return "tcp"
    return "subprocess"


def attach(
    location: str,
    *,
    name: str,
    transport: Transport | None = None,
    options: dict[str, Any] | None = None,
    data_version_spec: str | None = None,
    implementation_version: str | None = None,
    bearer_token: str | None = None,
    worker_limit: int | None = None,
    **client_kwargs: Any,
) -> VgiCatalog:
    """Attach to a VGI catalog and return a `VgiCatalog`.

    Args:
        location: For subprocess transport (the default for anything that
            isn't a recognized URL scheme), the worker command (shlex-split,
            no shell — matches `vgi.client.Client`'s own semantics, e.g.
            `"uv run --project ~/Development/vgi-python vgi-fixture-worker"`).
            For HTTP, `"http://..."`/`"https://..."`. For TCP, `"tcp://host:port"`.
        name: The catalog name to attach to (a worker can serve more than one).
        transport: `"subprocess"`, `"http"`, or `"tcp"`. Defaults to `None`,
            which auto-detects from `location`'s scheme (an `http(s)://` or
            `tcp://` prefix selects that transport; anything else is treated
            as a subprocess command). Pass explicitly to override — e.g. a
            subprocess command that happens to start with `http` for some
            reason, though that shouldn't come up in practice.
        options: Catalog-specific ATTACH options.
        data_version_spec: Semver constraint for the catalog's data version.
        implementation_version: Semver constraint for the worker's implementation.
        bearer_token: Static bearer token, HTTP transport only.
        worker_limit: Max concurrent workers, subprocess transport only.
        **client_kwargs: Passed through to `Client(...)` / `Client.from_http(...)`
            / `Client.from_tcp(...)`.

    Returns:
        The attached `VgiCatalog`.

    """
    resolved_transport = transport if transport is not None else _detect_transport(location)

    def client_factory() -> Client:
        """Build one fresh, unstarted `Client` connected the same way every time.

        Used both for the initial attach-time client and, via
        `VgiCatalog._exchange_client()`, once per thread thereafter (see the
        module docstring's "Thread safety" section). No `catalog_attach` here
        — exchange-mode RPCs (`table_function`/`scalar_function`/...) don't
        take `attach_opaque_data` at all, so a fresh connection is immediately
        usable with no re-attach step.
        """
        if resolved_transport == "subprocess":
            return Client(location, worker_limit=worker_limit, **client_kwargs)
        if resolved_transport == "http":
            return Client.from_http(location, bearer_token=bearer_token, **client_kwargs)
        if resolved_transport == "tcp":
            # Strip the tcp:// prefix if auto-detected or passed explicitly with it.
            host_port = location.removeprefix("tcp://")
            host, _, port = host_port.partition(":")
            if not port:
                raise ValueError(f"tcp transport expects 'tcp://host:port' or 'host:port', got {location!r}")
            return Client.from_tcp(host, int(port), **client_kwargs)
        raise ValueError(f"unknown transport: {resolved_transport!r}")

    client = client_factory()

    def _cleanup() -> None:
        # start() itself may have failed before setting up the primary
        # worker, in which case stop() raises "Client not started" — a
        # secondary failure that would mask the real one. Best-effort only:
        # deliberately swallows anything, since we're already unwinding a
        # real error and a cleanup failure must never shadow it.
        with contextlib.suppress(Exception):
            client.stop()

    try:
        client.start()
        result = client.catalog_attach(
            name=name,
            options=options,
            data_version_spec=data_version_spec,
            implementation_version=implementation_version,
        )
    except (*VGI_CLIENT_ERRORS, OSError) as e:
        # OSError: a bad subprocess-transport command (e.g. a nonexistent
        # executable path) raises this raw from subprocess.Popen — vgi-python
        # doesn't get a chance to wrap it, so vgi-polars does.
        _cleanup()
        raise VgiPolarsError(str(e)) from e
    except BaseException:
        _cleanup()
        raise

    return VgiCatalog(client=client, client_factory=client_factory, name=name, attach_result=result)
