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
(self-join, `concat`, `collect_all`) — both confirmed empirically.

The fix is `VgiCatalog._exchange_client()` — a context manager that **borrows a
fresh `Client` for exactly the duration of one exchange-mode operation**, then
returns it. No shared mutable state is ever touched by more than one caller at a
time, so this is thread-safe by construction, not by locking. This mirrors
`CatalogClientMixin`'s own pattern for catalog-metadata calls (`schemas`,
`table_get`, `schema_contents`, `table_scan_function_get`,
`table_column_statistics` — a short-lived connection per call, never
`self._primary`, see `tests/test_concurrency.py`) rather than inventing a
different one: `_exchange_client()` doesn't build any pooling of its own either,
it just stops bypassing vgi-python's existing subprocess pool.

Concretely, for subprocess transport, `Client()`'s own default `pool=` (a
module-level `WorkerPool` in vgi-python, `max_idle=8, idle_timeout=30.0`) is
what actually makes each borrow cheap — it *borrows an idle worker process*
rather than spawning one, the same pool `CatalogClientMixin` already borrows
from for catalog calls. Bind+init already run fresh on every single exchange
call regardless of whether the `Client` object itself is reused (confirmed by
reading `_initialize_stream_common`, which every exchange method calls
unconditionally) — so caching a `Client` across calls was never amortizing a
handshake, only a subprocess spawn the pool already amortizes for free. For
HTTP transport, `client_factory()` shares one `httpx2.Client` across every
borrowed exchange `Client` (mirroring how `CatalogClientMixin` shares it for
HTTP catalog calls), so per-call connection setup is cheap there too. TCP is
the one transport with no pooling concept in vgi-python at all — a borrowed
TCP exchange client pays a fresh raw handshake per call instead of holding one
connection open per thread, same as today's `attach()`-time-only reuse
elsewhere in this file. Accepted: TCP is already documented as loopback/dev-
only (no auth), not the transport this optimization needs to be free on.

**Catalog snapshot.** Listings (`schemas`, `tables`, `views`, `macros`,
`functions`, `function_info`) and the catalog lookups the bridges do (a table's
`TableInfo`, a function's `FunctionInfo`) are answered from one whole-catalog
snapshot loaded lazily by `Client.load_catalog` — a single `catalog_contents`
RPC when the attach advertises `supports_catalog_contents`, otherwise (or when
that call fails) `catalog_schemas` plus the per-schema
`catalog_schema_contents_*` RPCs. `catalog_contents` is never sent to a worker
that did not advertise it. vgi-polars has no transactions, so every catalog
read is a "transaction start" in the sense of the DuckDB extension's
`docs/catalog_contents.md` ("Caching and revalidation"), and the snapshot is
checked at each read by the same rules:

- **version-frozen** catalog (`catalog_version_frozen`): never rechecked;
- snapshot with an **etag**: one `catalog_contents(if_none_match=etag)`;
  `not_modified` keeps it, a full answer replaces it, a failure drops the etag
  (the next read polls the version instead);
- otherwise **`catalog_version` poll**: unchanged non-zero version keeps it, a
  changed one reloads;
- version **0** with no etag (not frozen): the client cannot tell whether
  anything changed, so the snapshot from the first load serves that read only
  and later reads use the targeted per-schema RPCs — vgi-polars' pre-snapshot
  behaviour — instead of re-downloading the whole catalog each time (the
  extension's "version-0 rule").

So a non-frozen catalog never serves a listing older than the worker's current
state (DDL from anywhere is seen on the next read), and a frozen one costs no
RPCs after the first load. A table or function the snapshot does not contain
still falls through to its per-name RPC, so the worker stays authoritative for
misses. `VgiCatalog.clear_cache()` drops the snapshot. Time-travel (`at_unit`)
lookups never use it. The check-and-load runs under one lock, so concurrent
readers share a single load instead of racing to issue their own.
"""

from __future__ import annotations

import contextlib
import shlex
import threading
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from vgi.catalog.catalog_interface import CatalogAttachResult, SchemaObjectType
from vgi.client.catalog_mixin import CatalogClientError
from vgi.client.client import Client

from vgi_polars.errors import VGI_CLIENT_ERRORS, VgiPolarsError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from vgi.catalog.catalog_interface import AttachOpaqueData, FunctionInfo, SchemaInfo, TableInfo
    from vgi.client.catalog_mixin import CatalogSnapshot

    from vgi_polars._aggregate import AggregateFunction
    from vgi_polars._row_transform import RowTransformFunctionCall
    from vgi_polars._scalar import ScalarFunctionCall
    from vgi_polars._table_in_out import TableInOutFunction
    from vgi_polars.table import VgiTable

__all__ = ["VgiCatalog", "attach"]

Transport = Literal["subprocess", "http", "tcp", "launch"]

#: A `SchemaContentsInfo` attribute, i.e. one kind of schema object.
SchemaObjectKind = Literal[
    "tables",
    "views",
    "scalar_functions",
    "aggregate_functions",
    "table_functions",
    "scalar_macros",
    "table_macros",
    "indexes",
]

#: The per-schema `schema_contents` type for each kind (the uncached path).
_KIND_TYPES: dict[SchemaObjectKind, SchemaObjectType] = {
    "tables": SchemaObjectType.TABLE,
    "views": SchemaObjectType.VIEW,
    "scalar_functions": SchemaObjectType.SCALAR_FUNCTION,
    "aggregate_functions": SchemaObjectType.AGGREGATE_FUNCTION,
    "table_functions": SchemaObjectType.TABLE_FUNCTION,
    "scalar_macros": SchemaObjectType.SCALAR_MACRO,
    "table_macros": SchemaObjectType.TABLE_MACRO,
    "indexes": SchemaObjectType.INDEX,
}


def schema_path(schema_name: str) -> list[str]:
    """The wire schema path for a vgi-polars schema name.

    vgi-polars addresses a schema by one name (`cat.table("main", "t")`), so it
    is a one-component path. Nested (multi-component) schema paths, which the
    v2 protocol allows, are listed by `VgiCatalog.schemas()` as their
    dot-joined components but cannot be addressed through this str-based API.
    """
    return [schema_name]


def _schema_display_name(info: SchemaInfo) -> str:
    """The name `VgiCatalog.schemas()` lists for `info` (see `schema_path`)."""
    return ".".join(info.path)


class VgiCatalog:
    """An attached VGI catalog. Construct via `attach()`, not directly."""

    def __init__(
        self,
        *,
        client: Client,
        client_factory: Callable[[], Client],
        name: str,
        attach_result: CatalogAttachResult,
        persistent_exchange_client: bool = False,
    ) -> None:
        """Wrap an already-attached `client`/`attach_result` pair. Use `attach()`, not this directly.

        `persistent_exchange_client`: `True` only for OAuth-mode HTTP attaches
        (set by `attach()`) — see `_exchange_client()`'s docstring for why
        OAuth can't use the normal per-call borrow-and-return model.
        """
        self._client = client
        self._client_factory = client_factory
        self._name = name
        self._attach_result = attach_result
        self._detached = False
        self._persistent_exchange_client = persistent_exchange_client
        # Catalog snapshot state — see the module docstring's "Catalog snapshot"
        # section. Guarded by `_snapshot_lock`, which is held across the check
        # and any (re)load so concurrent readers share one load.
        self._snapshot_lock = threading.Lock()
        self._snapshot: CatalogSnapshot | None = None
        # The catalog version the held snapshot is known to be current at.
        self._known_version = attach_result.catalog_version
        # Set once the worker turns out to be unversioned (version 0, no etag,
        # not frozen): reads then use the targeted per-schema RPCs.
        self._snapshot_uncacheable = False
        if persistent_exchange_client:
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

        Bumps when schemas, tables, or other catalog objects change. This is
        the attach-time value; the catalog snapshot tracks the current one on
        its own (see the module docstring's "Catalog snapshot" section).
        """
        return self._attach_result.catalog_version

    @property
    def catalog_version_frozen(self) -> bool:
        """Whether the worker asserts its catalog metadata is frozen for this attach.

        Covers schema/table/function metadata never changing for the lifetime
        of this attach. When set, the catalog snapshot is loaded once and never
        revalidated (see the module docstring's "Catalog snapshot" section).
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

        Covers `load_catalog`, `catalog_version`, `schemas`, `table_get`,
        `schema_contents`, `table_scan_function_get`, `table_column_statistics`.
        DDL issued through it is seen by the next listing of a non-frozen
        catalog (each read revalidates); `clear_cache()` forces it. Not part of the
        stable public API. Exchange-mode calls (table/scalar/aggregate/
        table-in-out function invocation) must use `_exchange_client()`
        instead — see the module docstring's "Thread safety" section.
        """
        return self._client

    @contextlib.contextmanager
    def _exchange_client(self) -> Iterator[Client]:
        """A `Client` for the duration of exactly one exchange-mode operation.

        Covers `table_function`, `scalar_function`, `aggregate_function`,
        `table_in_out_function`, `table_buffering_function` — every caller
        wraps its whole operation in `with catalog._exchange_client() as
        client:` (one whole scan for a table-scan generator, one call for a
        scalar/aggregate/table-in-out invocation), never holds the client
        past that `with` block.

        Normally borrows a **fresh** `Client` and returns it when the block
        exits — see the module docstring's "Thread safety" section for why
        this is correct *and* cheap, not just correct. `_persistent_
        exchange_client` (OAuth-mode HTTP attaches only) instead falls back
        to one cached `Client` per calling thread, reused across that
        thread's subsequent calls and only stopped at `detach()`: OAuth's
        own httpx wiring is mutually exclusive with a shared `httpx_client`
        (`Client.__init__`), so borrowing a fresh `Client` per call would
        mean a fresh, uncached `VgiOAuthAuth` — and likely a fresh login
        prompt or at least a fresh token refresh — on every single exchange
        call. This is exactly the caching behavior every exchange call had
        before the per-call model existed; not a regression, just not a
        newly-fixed case.
        """
        if self._persistent_exchange_client:
            yield self._persistent_thread_client()
            return
        client = self._client_factory()
        client.start()
        try:
            yield client
        finally:
            # Best-effort return-to-pool: a failure here must never mask
            # whatever happened (or didn't) inside the `with` block, mirroring
            # attach()'s own `_cleanup()`.
            with contextlib.suppress(Exception):
                client.stop()

    def _persistent_thread_client(self) -> Client:
        """OAuth-mode fallback: one cached `Client` per calling thread. See `_exchange_client()`."""
        existing: Client | None = getattr(self._thread_local, "client", None)
        if existing is not None:
            return existing
        new_client = self._client_factory()
        new_client.start()
        self._thread_local.client = new_client
        with self._exchange_clients_lock:
            self._exchange_clients.append(new_client)
        return new_client

    # ------------------------------------------------------------------
    # Catalog snapshot (see the module docstring's "Catalog snapshot" section)
    # ------------------------------------------------------------------

    def _current_snapshot(self) -> CatalogSnapshot | None:
        """The catalog snapshot, loaded or revalidated as this read requires.

        `None` means "no snapshot for this catalog": it is unversioned (version
        0, no etag, not frozen), and the caller uses the per-schema RPCs.
        Raises whatever `Client.load_catalog`/`catalog_version` raise
        (`VGI_CLIENT_ERRORS`); the held snapshot is left unchanged then.
        """
        if self._snapshot_uncacheable:
            return None
        with self._snapshot_lock:
            if self._snapshot_uncacheable:
                return None
            held = self._snapshot
            if held is None:
                return self._adopt(self._load_catalog(previous=None), polled_version=None)
            if self.catalog_version_frozen:
                return held
            if held.etag is not None:
                # One conditional catalog_contents instead of a version poll;
                # load_catalog keeps `held`'s content on not_modified, replaces
                # it on a full answer, and falls back (dropping the etag) on error.
                return self._adopt(self._load_catalog(previous=held), polled_version=None)
            version = self._client.catalog_version(attach_opaque_data=self.attach_opaque_data)
            if version != 0 and version == self._known_version:
                return held
            # Changed, or unknown (0): poll first, then load, so a DDL racing
            # the load can only make the snapshot newer than `version` — the
            # next poll then reloads once, never serves stale content.
            return self._adopt(self._load_catalog(previous=held), polled_version=version)

    def _load_catalog(self, *, previous: CatalogSnapshot | None) -> CatalogSnapshot:
        """`Client.load_catalog` for this attach, with its errors normalized to `CatalogClientError`.

        vgi-python decodes a `catalog_contents` answer outside the error
        conversion every other catalog call gets, so a malformed object in it
        (e.g. an unknown wire-enum value) escapes as a bare `KeyError`/
        `ValueError` rather than `CatalogClientError`. Normalized here so it
        surfaces as `VgiPolarsError` like the same failure on the per-schema
        path does.
        """
        try:
            return self._client.load_catalog(attach=self._attach_result, previous=previous)
        except (KeyError, ValueError) as e:
            raise CatalogClientError(f"catalog_contents: {e}") from e

    def _adopt(self, snapshot: CatalogSnapshot, *, polled_version: int | None) -> CatalogSnapshot:
        """Record `snapshot` as current (caller holds `_snapshot_lock`); returns it for this read.

        Version adoption follows the extension: a snapshot's own non-zero
        version becomes the known one; a per-schema snapshot (no version of its
        own) is current as of the version polled before loading it. A
        non-frozen catalog with no etag and known version 0 cannot be
        revalidated, so the snapshot serves only this read.
        """
        if snapshot.catalog_version:
            self._known_version = snapshot.catalog_version
        elif polled_version is not None:
            self._known_version = polled_version
        if not self.catalog_version_frozen and snapshot.etag is None and self._known_version == 0:
            self._snapshot = None
            self._snapshot_uncacheable = True
        else:
            self._snapshot = snapshot
        return snapshot

    def clear_cache(self) -> None:
        """Drop the catalog snapshot; the next catalog read loads a fresh one.

        Rarely needed: a non-frozen catalog is revalidated at every read anyway.
        Useful for a version-frozen catalog whose worker was redeployed, or to
        release the snapshot's memory.
        """
        with self._snapshot_lock:
            self._snapshot = None

    def _schema_infos(self) -> list[SchemaInfo]:
        """Every schema's `SchemaInfo`, from the snapshot or `catalog_schemas`."""
        snapshot = self._current_snapshot()
        if snapshot is not None:
            return [entry.schema for entry in snapshot.schemas]
        return list(self._client.schemas(attach_opaque_data=self.attach_opaque_data))

    def _schema_objects(self, schema_name: str, *kinds: SchemaObjectKind) -> list[Any]:
        """The objects of `kinds` (in that order) in `schema_name`, from the snapshot or per-schema RPCs.

        One snapshot check per call, however many kinds. A schema the snapshot
        does not contain goes to the per-schema RPCs, so the worker's own answer
        (an error, or an empty list) is what the caller sees — the same as
        before the snapshot existed.
        """
        path = schema_path(schema_name)
        snapshot = self._current_snapshot()
        if snapshot is not None:
            for entry in snapshot.schemas:
                if list(entry.schema.path) == path:
                    return [obj for kind in kinds for obj in getattr(entry, kind)]
        return [
            obj
            for kind in kinds
            for obj in self._client.schema_contents(
                attach_opaque_data=self.attach_opaque_data, path=path, type=_KIND_TYPES[kind]
            )
        ]

    def _snapshot_lookup(self, schema_name: str, kind: SchemaObjectKind, name: str) -> Any | None:
        """`name` among `kind` in `schema_name`, if the snapshot has it; else `None`.

        Never issues a per-schema listing: with no snapshot (an unversioned
        catalog) or a miss, the caller uses its own per-name RPC instead.
        """
        path = schema_path(schema_name)
        snapshot = self._current_snapshot()
        if snapshot is None:
            return None
        for entry in snapshot.schemas:
            if list(entry.schema.path) == path:
                return next((obj for obj in getattr(entry, kind) if obj.name == name), None)
        return None

    def _table_info(self, schema_name: str, name: str) -> TableInfo | None:
        """The live `TableInfo` for `schema_name.name` from the snapshot, if it lists it."""
        return cast("TableInfo | None", self._snapshot_lookup(schema_name, "tables", name))

    def _function_infos(self, schema_name: str, *kinds: SchemaObjectKind) -> list[FunctionInfo]:
        """The `FunctionInfo`s of the given function kinds in `schema_name`, in that order."""
        return cast("list[FunctionInfo]", self._schema_objects(schema_name, *kinds))

    def schemas(self) -> list[str]:
        """List schema names in this catalog.

        A nested schema (multi-component path) is listed as its dot-joined
        components; see `schema_path`.
        """
        try:
            infos = self._schema_infos()
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        return [_schema_display_name(s) for s in infos]

    def tables(self, schema_name: str) -> list[str]:
        """List table names in `schema_name`."""
        return self._object_names(schema_name, "tables")

    def views(self, schema_name: str) -> list[str]:
        """List view names in `schema_name`."""
        return self._object_names(schema_name, "views")

    def macros(self, schema_name: str) -> list[str]:
        """List macro names in `schema_name` (scalar macros, then table macros)."""
        return self._object_names(schema_name, "scalar_macros", "table_macros")

    def _object_names(self, schema_name: str, *kinds: SchemaObjectKind) -> list[str]:
        try:
            return [obj.name for obj in self._schema_objects(schema_name, *kinds)]
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e

    def _all_function_infos(self, schema_name: str) -> list[FunctionInfo]:
        """Scalar + table + aggregate `FunctionInfo`s in `schema_name`, in that order."""
        return self._function_infos(schema_name, "scalar_functions", "table_functions", "aggregate_functions")

    def functions(self, schema_name: str) -> list[str]:
        """List function names in `schema_name` (scalar, table, and aggregate).

        One entry per overload, like `duckdb_functions()` — an overloaded name
        (e.g. two `format_number` arities) appears more than once.
        """
        try:
            infos = self._all_function_infos(schema_name)
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        return [i.name for i in infos]

    def function_info(self, schema_name: str, name: str) -> FunctionInfo:
        """Metadata for a function: `.comment`, `.description`, `.tags`, `.examples`, and more.

        Tries scalar, then table, then aggregate functions in turn and
        returns the first name match — same resolution `scalar_function()`
        and friends already use (no overload-by-arity disambiguation; an
        overloaded name resolves to whichever overload the worker lists
        first). Raises `VgiPolarsError` if no function in `schema_name` has
        this name under any of the three kinds.
        """
        try:
            infos = self._all_function_infos(schema_name)
        except VGI_CLIENT_ERRORS as e:
            raise VgiPolarsError(str(e)) from e
        info = next((i for i in infos if i.name == name), None)
        if info is None:
            raise VgiPolarsError(f"function not found: {schema_name}.{name}")
        return info

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

    def row_transform_function(self, schema_name: str, name: str) -> RowTransformFunctionCall:
        """Return a callable for a blended row-transform function (`RowTransformFunction`).

        The returned callable has signature `fn(lf: pl.LazyFrame | None =
        None, *args, settings=None, dedup=True, **named_args) -> pl.LazyFrame`
        (see `_row_transform.py`). `lf=None` — a bare literal call — is not
        yet supported.
        """
        from vgi_polars._row_transform import make_row_transform_function

        return make_row_transform_function(self, schema_name, name)

    def detach(self) -> None:
        """Detach from the catalog and close the underlying client(s).

        Normally there are no per-thread exchange clients to close —
        `_exchange_client()` borrows and returns one per operation, never
        holds one open past that (see the module docstring). The OAuth-mode
        fallback (`_persistent_exchange_client`) does hold one per thread
        until now; those get stopped here too. Safe to call more than once.
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
            if self._persistent_exchange_client:
                with self._exchange_clients_lock:
                    exchange_clients, self._exchange_clients = self._exchange_clients, []
                for exchange_client in exchange_clients:
                    # Best-effort cleanup: one client's failure to stop must
                    # never block stopping the rest.
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
    -> HTTP, `tcp://` -> TCP, `launch:` -> the AF_UNIX launcher, anything
    else -> subprocess/shlex argv).
    """
    if location.startswith(("http://", "https://")):
        return "http"
    if location.startswith("tcp://"):
        return "tcp"
    if location.startswith("launch:"):
        return "launch"
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
    oauth: bool = False,
    oauth_refresh_token: str | None = None,
    oauth_flow: Literal["auto", "device_code", "pkce"] = "auto",
    oauth_timeout_seconds: float = 120.0,
    oauth_prompt: Literal["none", "login", "select_account", "consent"] = "none",
    worker_limit: int | None = None,
    **client_kwargs: Any,
) -> VgiCatalog:
    """Attach to a VGI catalog and return a `VgiCatalog`.

    Args:
        location: For subprocess transport (the default for anything that
            isn't a recognized URL scheme), the worker command (shlex-split,
            no shell — matches `vgi.client.Client`'s own semantics, e.g.
            `"uv run --project ~/Development/vgi-python vgi-fixture-worker"`).
            For HTTP, `"http://..."`/`"https://..."`. For TCP,
            `"tcp://host:port"`. For the AF_UNIX launcher, `"launch:<argv>"`
            (same shlex-split argv convention as subprocess, just prefixed —
            e.g. `"launch:uv run --project ~/Development/vgi-python
            vgi-fixture-worker"`); every caller across the machine pointing
            at the same argv shares one warm worker process, coordinated by
            `vgi_rpc.launcher`'s per-command-hash flock (see
            `Client.from_launch`'s docstring). Requires the
            `vgi-python[launch]` extra.
        name: The catalog name to attach to (a worker can serve more than one).
        transport: `"subprocess"`, `"http"`, `"tcp"`, or `"launch"`. Defaults
            to `None`, which auto-detects from `location`'s scheme (an
            `http(s)://`, `tcp://`, or `launch:` prefix selects that
            transport; anything else is treated as a subprocess command).
            Pass explicitly to override — e.g. a subprocess command that
            happens to start with `http` for some reason, though that
            shouldn't come up in practice.
        options: Catalog-specific ATTACH options.
        data_version_spec: Semver constraint for the catalog's data version.
        implementation_version: Semver constraint for the worker's implementation.
        bearer_token: Static bearer token, HTTP transport only. Mutually
            exclusive with `oauth`/`oauth_refresh_token` (enforced by `Client`).
        oauth: HTTP transport only. When `True`, obtain and refresh bearer
            tokens automatically via OAuth (device-code flow today — PKCE is
            not yet implemented) whenever the worker answers 401 with an
            RFC 9728 challenge. Implied by passing `oauth_refresh_token`. The
            first call blocks and prints a "Visit: ... Enter code: ..."
            prompt until login completes; later calls reuse the cached token
            and refresh it silently. See `vgi.client.Client`'s own docs for
            the full mechanism, and `catalog.client.oauth_identity()` to read
            back the signed-in identity once authenticated.
        oauth_refresh_token: HTTP transport only. Pre-obtained refresh token,
            seeded so the first request can silently refresh instead of
            running an interactive login. Implies `oauth=True`.
        oauth_flow: HTTP transport only, OAuth only. `"auto"` (default) picks
            device-code when the server offers it; `"device_code"` forces it;
            `"pkce"` is not yet implemented and raises `NotImplementedError`
            when actually needed.
        oauth_timeout_seconds: HTTP transport only, OAuth only. How long an
            interactive login may take before giving up.
        oauth_prompt: HTTP transport only, OAuth only. Reserved for the PKCE
            flow (not yet implemented); has no effect on device-code logins.
        worker_limit: Max concurrent workers. Subprocess and launch
            transports only.
        **client_kwargs: Passed through to `Client(...)` / `Client.from_http(...)`
            / `Client.from_tcp(...)` / `Client.from_launch(...)` — e.g.
            `idle_timeout`/`state_dir`/`socket_path` for the launch transport.

    Returns:
        The attached `VgiCatalog`.

    """
    resolved_transport = transport if transport is not None else _detect_transport(location)
    use_oauth = oauth or oauth_refresh_token is not None

    # Shared across every borrowed HTTP exchange Client (see client_factory's
    # "http" branch) so a per-call connection doesn't pay a fresh TLS/TCP
    # handshake — mirrors CatalogClientMixin's own httpx2.Client sharing for
    # HTTP catalog calls. A single-element list, not a plain variable, so the
    # closure below can both read AND populate it (`nonlocal` would work too;
    # this reads slightly clearer at the call site). Never touched in OAuth
    # mode — see client_factory's "http" branch for why.
    _shared_httpx_client: list[Any] = [None]

    def client_factory() -> Client:
        """Build one fresh, unstarted `Client` connected the same way every time.

        Used both for the initial attach-time client and, via
        `VgiCatalog._exchange_client()`, once per exchange-mode operation
        thereafter (see the module docstring's "Thread safety" section). No
        `catalog_attach` here — exchange-mode RPCs (`table_function`/
        `scalar_function`/...) don't take `attach_opaque_data` at all, so a
        fresh connection is immediately usable with no re-attach step.
        """
        if resolved_transport == "subprocess":
            return Client(location, worker_limit=worker_limit, **client_kwargs)
        if resolved_transport == "http":
            if use_oauth:
                # httpx_client= is mutually exclusive with oauth/oauth_refresh_token
                # in Client.__init__ (OAuth builds its own VgiOAuthAuth-wired httpx2.Client
                # internally) -- sharing one here would mean either violating that, or
                # duplicating vgi-python's internal VgiOAuthAuth construction ourselves, a
                # correctness risk not worth taking. Every call just gets oauth=True/
                # oauth_refresh_token= again; VgiCatalog's OAuth fallback path (module
                # docstring) is what keeps this from re-authenticating on every call.
                return Client.from_http(
                    location,
                    bearer_token=bearer_token,
                    oauth=oauth,
                    oauth_refresh_token=oauth_refresh_token,
                    oauth_flow=oauth_flow,
                    oauth_timeout_seconds=oauth_timeout_seconds,
                    oauth_prompt=oauth_prompt,
                    **client_kwargs,
                )
            # Caller-supplied httpx_client (via **client_kwargs) always wins and is
            # never overwritten by our shared one -- copy client_kwargs rather than
            # mutating the closed-over dict, since client_factory() runs many times.
            kwargs = dict(client_kwargs)
            caller_httpx_client = kwargs.pop("httpx_client", None)
            c = Client.from_http(
                location,
                bearer_token=bearer_token,
                # use_oauth is False on this branch, so these are always their
                # inert defaults -- forwarded explicitly anyway (not left to
                # Client.from_http's own defaults) so attach()'s observable
                # call shape doesn't depend on which branch was taken.
                oauth=oauth,
                oauth_refresh_token=oauth_refresh_token,
                oauth_flow=oauth_flow,
                oauth_timeout_seconds=oauth_timeout_seconds,
                oauth_prompt=oauth_prompt,
                httpx_client=caller_httpx_client if caller_httpx_client is not None else _shared_httpx_client[0],
                **kwargs,
            )
            if caller_httpx_client is None and _shared_httpx_client[0] is None:
                # Force it to materialize now (rather than lazily on first
                # request) so every later client_factory() call sees it —
                # cheap: constructing an httpx2.Client doesn't itself open a
                # connection. _get_or_create_httpx_client is vgi-python-
                # internal (leading underscore); vgi-polars already couples
                # this tightly to Client's internals elsewhere in this file.
                _shared_httpx_client[0] = c._get_or_create_httpx_client()
            return c
        if resolved_transport == "tcp":
            # Strip the tcp:// prefix if auto-detected or passed explicitly with it.
            host_port = location.removeprefix("tcp://")
            host, _, port = host_port.partition(":")
            if not port:
                raise ValueError(f"tcp transport expects 'tcp://host:port' or 'host:port', got {location!r}")
            return Client.from_tcp(host, int(port), **client_kwargs)
        if resolved_transport == "launch":
            # `launch:<argv>` — same shlex-split convention plain subprocess
            # LOCATIONs use (`Client.__init__` does this itself for
            # transport="subprocess"; `from_launch` takes an already-split
            # argv sequence, so it's done here instead), and the same
            # scheme-strip-then-tokenize shape the DuckDB C++ extension's
            # own `StripLaunchScheme` + `launcher::ParseLaunchArgv` use.
            argv = shlex.split(location.removeprefix("launch:"))
            return Client.from_launch(argv, worker_limit=worker_limit, **client_kwargs)
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

    return VgiCatalog(
        client=client,
        client_factory=client_factory,
        name=name,
        attach_result=result,
        persistent_exchange_client=use_oauth,
    )
