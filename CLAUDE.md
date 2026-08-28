# vgi-polars

A **client SDK** — structurally the same role the VGI DuckDB extension
(`~/Development/vgi`) plays for DuckDB, but for [Polars](https://pola.rs). New client
code, not a fork of the DuckDB extension (mirrors how `vgi-sqlite` frames itself for
SQLite). Sibling repos in this family: `vgi-sqlite` (client, SQLite), `vgi-spark`
(client, Spark — ported from `vgi-trino`'s design), `vgi-python`/`vgi-rust`/`vgi-go`/
`vgi-java`/`vgi-typescript`/`vgi-csharp` (worker SDKs — the opposite role).

## Architecture

vgi-polars is an **adapter**, not a new VGI protocol implementation. It wraps
vgi-python's existing pure-Python, Arrow-native `vgi.client.Client` (+
`CatalogClientMixin`) — the same wire-protocol code the DuckDB extension speaks,
independent of DuckDB — the same way `vgi-spark` depends on `vgi-java` directly rather
than reimplementing VGI's RPC framing (its own README: *"`farm.query.vgi` ... is
depended on directly rather than reimplemented"*).

Package import name is `vgi_polars` (distribution `vgi-polars`), not `vgi.polars` —
`vgi-python` already owns the top-level `vgi` package.

## Design Principle 1 (load-bearing): pushdown is an optimization, never a correctness delegation

Verified empirically (see the plan doc for the exact repro): Polars' `register_io_source`
does **not** re-verify a predicate the source claims to have handled — an `io_source`
that ignores `predicate` gets all rows back unfiltered. This is the opposite of
DuckDB's own posture, which VGI worker implementations are written and tested against
(workers are allowed to apply a pushed filter only approximately because DuckDB
re-verifies). Consequence, applied everywhere in `src/vgi_polars/_source.py`:

> Attempt projection/filter/limit pushdown to reduce data volume. Always apply the
> *original, complete* `with_columns` selection, `predicate`, and `n_rows` truncation
> locally after scanning, regardless of what was pushed or what the worker claims to
> have done.

A partial or entirely-failed pushdown translation is therefore only ever a performance
loss, never a correctness one. Do not weaken this without a very good reason — see
`_source.py`'s module docstring for the full argument.

## Non-obvious facts discovered building this (all live against vgi-fixture-worker)

- **A catalog table's scan function is not necessarily registered in the table's own
  schema.** `data.filter_echo_table` resolves to a function registered only in schema
  `main`. Mirrors the DuckDB C++ extension's own resolution order (found by reading
  `vgi_table_entry.cpp`): try the table's own schema first, then the catalog's
  `default_schema`. `VgiTable._resolve_scan_function` implements this; `_source.py`
  calls `table.scan_function_schema()`, never `table.schema_name`, for the actual RPC.
- **A `ConstParam` scalar argument (e.g. `main.multiply(value, factor)`'s `factor`) uses
  a *separate, densely-numbered* index space from array/batch arguments** — not the
  argument's absolute declared position. `Arguments.positional` must hold only the
  const values in their own relative order (`_scalar.py`'s module docstring has the
  full mechanism, traced through `vgi/scalar_function.py`'s `column_index`/`const_index`
  counters). Getting this wrong raises a confusing `IndexError` deep in the worker
  (empty `Arguments`) — not a hang; see the `uv run` note below for what the apparent
  "hang" during initial development actually was.
- **A `pl.Expr` literal's JSON AST shape depends on whether the expression has been
  type-resolved against a schema yet.** A freshly authored `pl.col("x") > 5` serializes
  its literal as `{"Literal": {"Dyn": {"Int": 5}}}`; the *identical* predicate, once it
  has flowed through a real `LazyFrame`'s query engine (which `register_io_source`'s
  `io_source` callback always receives), carries `{"Literal": {"Scalar": {"Int64": 5}}}`
  instead — same value, different outer key, different inner type-name. `_literal_value`
  in `_filter_translate.py` handles both; a predicate built directly in a test
  (`test_translate_predicate_*`) and the one an actual scan receives are NOT
  interchangeable for testing this code path — the end-to-end tests in
  `test_scan_filter.py` exist because the unit tests alone would have missed this.
- **`col.is_in([...])`'s needle list doesn't serialize as plain JSON, unlike every
  other literal.** It comes across as a complete Arrow IPC stream embedded as a raw
  list of ints (byte values) under `Literal.Scalar.List` — decoding it (`pa.ipc.
  open_stream`) yields a RecordBatch with one unnamed column and *one row per
  candidate value*, not a single row holding a list. `_translate_is_in`/
  `_decode_is_in_values` in `_filter_translate.py` handle the decode; the wire
  format's own `value_ref` column is then built the other way around — a single
  row whose value IS the list of candidates (`pa.ListArray.from_arrays`), matching
  `docs/filter-pushdown.md`'s `_val_0: ["active", "pending", "review"]` example.
  Only the default `nulls_equal=False` case is translated — `nulls_equal=True`
  makes a NULL needle match NULL haystack values, which the plain "col IN (...)"
  wire filter can't express, so that case is left untranslated rather than risk a
  worker that auto-applies filters silently dropping rows that should have matched.
- **The table a catalog declares (`TableInfo.columns`) and what its resolved scan
  function actually names its output columns can differ.** `data.numbers` declares
  column `value`; the function it resolves to emits `n`. `_source.py` renames each
  batch positionally to the declared schema before applying `with_columns`/`predicate`
  (which reference the declared names) — see the comment at its rename site.
- **`vgi.client.Client` is not thread-safe for concurrent calls on one shared
  instance** — see "Thread safety" below. Found via a direct stress test (20 threads,
  one shared `Client`, `scalar_function`): 18/20 corrupted or errored, 0 correct.
  Fixed via a per-thread `Client` pool; `tests/test_concurrency.py` regression-tests
  it. This is *not* the `uv run` slowness described above — that's a separate,
  already-resolved false alarm from initial development; this is a real bug that was
  live in the shipped code until the Phase 0 fix.
- **A schema's bulk catalog listing RPC (`schema_contents(type=SCALAR_FUNCTION)`,
  used to resolve `FunctionInfo` for `cat.scalar_function(...)`) fetches every
  function in that schema in ONE call and deserializes the whole response
  together** — so one function with corrupted/unparseable metadata (e.g. an
  unrecognized wire-enum value) fails metadata resolution for every *other*
  function in the same schema too, not just the broken one. Confirmed live against
  `vgi-fixture-bad-enum-worker` (`double`'s corrupted `null_handling` also breaks
  resolving `multiply`, a perfectly healthy function). Inherited directly from
  vgi-python's catalog-RPC granularity, not a vgi-polars bug — see
  `test_protocol_robustness.py::test_unrecognized_wire_enum_poisons_the_whole_
  schema_listing`.
- **vgi-polars has no "scan an arbitrary function with positional args" entry
  point** — only `cat.table(schema, name)` (which requires a catalog `Table()`
  registration) reaches the scan path. Several useful diagnostic/adversarial
  fixtures (e.g. `generator_exception`) are bare schema functions, not catalog
  tables, and so aren't reachable from vgi-polars' public API at all — tests that
  need one construct a minimal duck-typed stand-in for `VgiTable` and drive
  `_source.make_io_source` directly instead (see `test_errors.py`'s
  `_FakeTableForScanFunction`). Worth a real API addition if this gap starts
  mattering for more than tests.
- **A table-in-out function's declared arguments (`FunctionInfo.arguments`) carry
  per-field wire metadata distinguishing the table-input slot from real
  arguments, and named from positional** — confirmed live against
  `main.accumulate`: `metadata[b"vgi_type"] == b"table"` marks the table-input
  field (never appears in `Arguments`, supplied via the exchange's `input`
  batches instead); `metadata[b"vgi_arg"] == b"named"` marks a named argument;
  anything else is positional, in a *dense, table-slot-excluded* index space
  (mirroring `_scalar.py`'s const/array split — see `_table_in_out.py`'s module
  docstring). None of this is documented anywhere obvious; found by inspecting
  the actual decoded `pa.Schema` field metadata.
- **A table-in-out function's static `FunctionInfo.output_schema` cannot be
  trusted, in two different failure modes** — confirmed live: it's an empty
  placeholder for a function whose output schema mirrors its input (`echo`'s
  `on_bind` sets `output_schema = the bound input schema`, unknowable statically),
  and it's non-empty but *wrong* for `accumulate` (declares `{"x": Int64}`,
  silently omitting the real `_timestamp` column its actual `bind()` response
  adds). `_table_in_out.py`'s `call()` therefore always does one eager,
  zero-row probe-bind exchange up front (via `bind_result_callback`) to learn
  the real, argument-bound output schema before constructing `map_batches`,
  rather than trusting the catalog declaration — the same thing the DuckDB C++
  extension itself does (always binds for real before planning a query).

## Thread safety

`VgiCatalog._exchange_client()` — one lazily-created `Client` per calling thread for
exchange-mode RPCs (`table_function`/`scalar_function`, and any future
`aggregate_function`/`table_in_out_function`/`table_buffering_function`), all sharing
the single `catalog_attach` from the original `attach()` call (exchange RPCs don't
take `attach_opaque_data` at all, confirmed — no re-attach needed per thread).
Mirrors the DuckDB C++ extension's own solved pattern: one attach, many pooled
per-thread connections. Full rationale + empirical evidence in `catalog.py`'s module
docstring. **Why this matters, concretely**: Polars calls `map_batches(streamable=
True)` callbacks concurrently from multiple threads (confirmed: 8 threads, 170
overlapping invocation pairs for the *same* function object reused across
`pl.collect_all`), and can run multiple concurrent instances of the same
`register_io_source` scan when it appears more than once in a resolved plan
(self-join/concat/collect_all sharing an upstream scan). Catalog-metadata methods
(`schemas`/`table_get`/`schema_contents`/`table_scan_function_get`/
`table_column_statistics`) keep using the catalog's one shared `client` —
`CatalogClientMixin` opens a short-lived connection per call rather than reusing
`self._primary`, verified safe under concurrency by
`test_concurrent_catalog_metadata_calls_are_correct`, not just assumed.

**Any new exchange-mode bridge (table-in-out, aggregate, ...) must call
`catalog._exchange_client()`, never `catalog.client`, for its actual RPC — this is
the single most important rule for extending vgi-polars.**

## Scope

**Implemented (Tier 1):** catalog attach/detach, schema/table introspection incl.
per-column statistics (`catalog.py`, `table.py`), lazy/eager table scan with
best-effort projection + filter pushdown incl. Date/Datetime/Duration/Binary/
Decimal literals and `is_in` (`_source.py`, `_filter_translate.py`), `required_filters`
cost-safety enforcement (a table declaring `TableInfo.required_filters` raises
`VgiPolarsError` before scanning if the query's predicate doesn't reference at
least one column from every AND'd OR-group — `_check_required_filters` in
`_source.py`, column-coverage via `pl.Expr.meta.root_names()`, not full
pushdown-translatability; a dotted struct-subfield requirement like `"s.a"` is
conservatively treated as satisfied by any predicate touching top-level `s`),
`row_id`-style virtual/hidden columns (confirmed via the `rff_rowid` fixture —
these need **no special handling**: there's no SQL planner in vgi-polars to hide
a column from `SELECT *` the way DuckDB does, so a `row_id` field is just an
ordinary queryable/filterable/selectable schema column), scalar function calls
via `pl.Expr.map_batches` incl. scoped secrets (`_scalar.py`), streaming +
buffered table-in-out functions via `pl.LazyFrame.map_batches` incl. positional/
named bind-time arguments (`_table_in_out.py` — covers `accumulate`/
`accumulate_read`/`accumulate_clear` too, no dedicated code needed), aggregate
functions via an eager `pl.DataFrame` bridge incl. `ConstParam` args
(`_aggregate.py`), blended (row-transform) table functions — both the
correlated/column shape (`map_batches` + `vgi_rpc.parent_row` provenance
decoding) and the bare literal-call shape (`pl.defer`) — via
`cat.row_transform_function(schema, name)` (`_row_transform.py`, see the
dedicated paragraph below for the design), native scan-function delegation
for `read_parquet` (`_native_scan.py`, see its own paragraph below),
thread-safe exchange-mode calls (see "Thread safety" above),
HTTP bearer auth, protocol-robustness (bad-protocol-version/bad-enum-wire/
mid-stream error) handling, per-chunk scalar input dedup (`dedup=True` default
on every scalar-function callable, the client-side mirror of the C++
extension's `vgi_exchange_input_dedup` — `_dedup_positions` in `_scalar.py`,
gated on `FunctionInfo.stability != VOLATILE`), catalog-versioning
introspection (`VgiCatalog.catalog_version`/`resolved_data_version`/
`resolved_implementation_version`/`supports_time_travel`/
`supports_transactions`/`catalog_version_frozen`/`comment`/`tags` — all
already returned by `Client.catalog_attach` in `CatalogAttachResult` but
previously left unexposed past `default_schema`/`attach_opaque_data`). Both
subprocess and HTTP transports, plus TCP (raw Arrow-IPC framing, loopback —
see "TCP transport" below).

`required_filters` note on error surfacing: `_check_required_filters` raises
`VgiPolarsError` from inside the `io_source` generator, but a Python-source
generator runs inside Polars' own execution engine — through `.collect()` the
exception arrives wrapped as `polars.exceptions.ComputeError` (message text
preserved, not exception-chained). `test_required_filters.py` asserts both: the
raw `VgiPolarsError` by calling `_check_required_filters` directly (unit level,
mirrors `test_errors.py`'s `_FakeTableForScanFunction` pattern), and the wrapped
`ComputeError` through the real `.scan().collect()` path (integration level).

Shared argument-conversion helpers (`is_const_field`, `to_scalar`, `build_
arguments`) live in `_arguments.py` — both `_scalar.py` and `_aggregate.py` use the
identical `vgi_const` wire-metadata convention for bind-time constant arguments,
confirmed identical on `main.multiply` and `main.vgi_percentile`.

The coverage-expansion work below (matching the VGI DuckDB extension's own protocol
surface, feature by feature) is complete: every item landed, or got a documented,
technically-justified reason it can't — tracked here, never silently dropped.

**Splits — implemented, sequential not parallel.** `_source.py`'s
`_iter_splits_sequential` + a `function_info.supports_splits` gate in
`make_io_source`. This needed a real upstream fix first: `Client` had **no way
to reach splits at all** — no `table_function_plan()` wrapper for `on_plan()`,
and `table_function()`/`_do_init` had no `split_tokens` parameter to redeem
one, even though the wire protocol (`TableFunctionPlanRequest`/`PlanResponse`/
`InitRequest.split_tokens`) and `VgiProtocol.table_function_plan` already
supported it — the C++ extension drives this today, the reference Python
client never grew the equivalent. Added `Client.table_function_plan(...)` +
`Client.table_function(split_tokens=..., split_execution_id=...,
split_init_opaque_data=...)` to vgi-python (`vgi/client/client.py`, additive,
no wire/protocol-version change — purely exposing existing RPC methods/wire
fields the Python client hadn't caught up to). **Sequential, not parallel, by
design**, per the earlier research finding this section used to cite as a
reason to skip splits entirely: Polars' `register_io_source` gets scan
parallelism only from independent generator *instances* for repeated plan
occurrences (self-join/concat/collect_all) — there's no mechanism to
cooperatively drive one generator's splits across threads. Redeeming splits
one at a time in order is still worth doing over skipping them: it's the
client-scoped decomposition the worker actually tuned for (bounded per-split
cost, exact-count/statistics metadata, replayability), just without a
parallelism win here. `_iter_splits_sequential` drains `PlanResponse.
next_cursors` pagination via a queue (a plan response can hand back more than
one continuation cursor for parallel enumeration branches; this consumer
visits each in turn). See "CI" below — `VGI_PYTHON_REF` is now pinned to
`v0.29.2`, which includes this fix (and the other three below), so this and
every other RPC-dependent test in this section's coverage runs for real in
CI, not skipped.

**Time travel — implemented.** `VgiCatalog.table(schema, name, at_unit=...,
at_value=...)` returns an immutable per-AT-clause `VgiTable` (own memoized
schema/scan-function/branches state, never shared with a live or
differently-versioned handle to the same table). Needed a vgi-python fix
first: `Client.table_get`/`Client.table_function` didn't accept `at_unit`/
`at_value` at all, even though the wire protocol (`BindRequest.at_unit`/
`.at_value`, `catalog_table_get`) always carried them and
`table_scan_function_get` was the only method that exposed them. Added
`at_unit`/`at_value` params to both (`vgi/client/client.py`,
`vgi/client/catalog_mixin.py`), threaded straight into the existing wire
fields — additive, no protocol bump. Two distinct worker patterns proven
working (`tests/client/test_time_travel_client.py`, vgi-python): a
pre-resolved-bind-argument style (`data.versioned_data`) and a style that
reads `at_unit`/`at_value` directly off the init request
(`data.tt_pushdown_fn`/`tt_pushdown_scan`) — only the latter needed
`table_function`'s own new parameters, proving they're load-bearing, not
redundant with `table_scan_function_get`'s resolution. **Not done:**
`TableInfo.supports_time_travel` for per-table discovery (only the
catalog-wide `VgiCatalog.supports_time_travel` exists) — a genuine
wire-schema addition needing a protocol minor bump + C++ codegen regen in
the sibling `vgi` repo, out of scope for this pass; a worker that doesn't
support time travel on a given table just rejects the request at bind, same
as any other unsupported bind option.

**Multi-branch tables — implemented** (companion-catalog federation still
isn't, see below). `VgiTable.scan()` transparently decomposes a multi-branch
table into `pl.concat` of one scan per branch (`_multi_branch.py`), each
branch's `branch_filter` parsed and applied (`parse_branch_filter` — an
AND-chain of `col OP const` comparisons, the same minimal grammar the C++
extension's own v1.0 binder supports; **correctness-critical, not an
optional pushdown** — an unparseable filter raises rather than risk
duplicate/wrong rows from an unconstrained overlapping branch). Needed a
vgi-python fix first: `Client` had no `table_scan_branches_get` at all —
grepped the whole `vgi/client/` package for "branch", the only hit was an
unrelated code comment. Worse than merely unsupported: scanning a
multi-branch table through the old `table_scan_function_get`-only path
**silently returned only its first branch** — a real, silent correctness
gap this fix closes, not just a missing feature. Added
`Client.table_scan_branches_get(...)` (deserializes the wire's `list[bytes]`
into typed `ScanBranch` objects, mirroring `table_function_plan`'s
`ScanSplit` handling; falls back to wrapping `table_scan_function_get` as a
one-branch result for a worker that predates the RPC — the same fallback
the C++ extension itself uses). `VgiTable.scan()` calls it unconditionally
(one more cheap, memoized, unary catalog RPC — no behavior change for the
common single-branch case). **Scoped to function branches only** — a
catalog-table branch (`source_table` populated: companion-catalog
federation, e.g. a DuckLake arm) or a format branch (`format_name`: a
declarative `read_parquet`/`read_csv`-style reader) raises a clear
`VgiPolarsError` rather than mis-scanning; both need real design work this
pass didn't reach (catalog-table branches have no established "attach
another catalog" concept in Polars at all; format branches need a
`format_name` → reader resolver like the C++ extension's own).

**Native scan-function delegation — implemented for `read_parquet`**
(`_native_scan.py`). `ScanFunctionResult.function_name` can name a reader the
*calling engine* should run itself rather than a VGI-hosted function (its own
docstring: "read_parquet, iceberg_scan, or a custom VGI table function") —
the DuckDB C++ extension resolves this by checking its own function catalog
before ever treating it as an RPC target. Found genuinely blocking, not
theoretical: a real public worker
([vgi-overture-maps](https://github.com/Query-farm/vgi-overture-maps-typescript))
ships no data at all — every table natively delegates to `read_parquet`
against Overture's public S3 GeoParquet — and `Client.table_function
(function_name="read_parquet", ...)` against it raised
`FunctionNotFoundError` (its `FunctionRegistry` is deliberately empty).
`VgiTable.scan()` now resolves this before ever reaching
`register_io_source`: `read_parquet` -> `pl.scan_parquet`, using
`ScanFunctionResult`'s positional/named arguments, bypassing the worker
entirely for the actual data (real Polars-native pushdown, not a hand-rolled
approximation). Two new `scan()` parameters: `storage_options` (a plain
passthrough — cloud credentials/region are genuinely out-of-band on VGI's
wire, the same way DuckDB's own reference worker needs a separate `SET
s3_region=...`) and `acknowledge_required_filters` (required-filters
cost-safety enforcement has no equivalent hook for a bare `LazyFrame`
returned immediately — no callback with the resolved predicate the way
`register_io_source` gets one at collect time — so `scan()` refuses outright
when a natively-delegated table declares `required_filters`, unless the
caller explicitly opts in). Scoped to single-branch tables and `read_parquet`
only; the analogous multi-branch "format branch" case above stays
unimplemented, and other native targets (`read_csv`, `iceberg_scan`, ...) are
a per-function mapping to add as a real need arises, not solved generically.

**Table-function result cache — implemented, minimal slice.** An in-memory,
TTL-only, producer-mode(whole-scan)-only cache (`_result_cache.py`):  when a
worker advertises `vgi.cache.ttl` on its result's first batch, the complete
raw result is cached in memory (keyed on the catalog's `attach_opaque_data`
+ function/schema/args/projection/pushdown-filters/AT-clause) and an
identical repeat scan is served without a worker round-trip — Design
Principle 1 still holds (the cached *raw* batches are re-filtered/
re-selected locally on every serve, cache hit or not). Needed a vgi-python
fix first: `Client._table_function_parallel` discarded `AnnotatedBatch.
custom_metadata` unconditionally (`if output.batch.num_rows > 0:
output_queue.put(output.batch)` — only the bare `pa.RecordBatch` survived),
so `vgi.cache.*` was structurally unreachable through `table_function()`'s
public generator. Added `table_function(batch_metadata_callback=...)`
(mirrors the existing `bind_result_callback` pattern — additive, no
breaking change to the generator's yield type), invoked once per batch with
its `custom_metadata`. **What's deliberately not here** (each is its own
multi-milestone feature on the C++ side, see that repo's CLAUDE.md): no
disk tier (memory only, lost on process exit); no conditional revalidation
(`etag`/`stale_while_revalidate` are parsed and ignored — a stale entry is
evicted, never refreshed); no byte/entry caps (unbounded, TTL is the only
eviction); no `vgi.cache.scope=transaction` support (never cached, not
mis-cached); never-partial commit is enforced by only attempting caching
when `n_rows is None` for the call (a LIMIT-truncated scan never drains its
generator to EOS, so it never has a complete raw result to commit).
**Identity scoping is a reasonable boundary, not an audited one** — folds in
the attach's opaque session token, not a verified auth-principal
fingerprint the way the C++ side's does; don't treat this as a credential
isolation guarantee without re-auditing that gap.

**Version-pin gap — closed.** `vgi-python` released `v0.29.2` (`chore: 0.29.2
— client support for splits, multi-branch, cache metadata, time travel`,
protocol unchanged at 1.4.0 — a client-only release, no wire-schema bump)
carrying all four fixes above, and this repo's CI (`VGI_PYTHON_REF`, see
"CI" below) is pinned to it — every RPC-dependent test across
`test_splits.py`/`test_multi_branch.py`/`test_result_cache.py`/
`test_time_travel.py` runs for real now, none skipped (verified: 130
passed, 0 skipped, against a fresh clone of the published tag). The
`hasattr`/`inspect.signature` capability guards on every call site in
`_source.py`/`table.py` are kept regardless — they cost nothing on a
current install and mean a *future* re-pin to an older tag (or a
downstream consumer of this package pinning vgi-python separately) still
degrades gracefully (a clear `VgiPolarsError` for an explicit time-travel/
split request, never a raw `AttributeError`/`TypeError`, and an ordinary
scan of a single-branch, non-cacheable, non-split table is completely
unaffected either way).

**Deliberately not yet implemented (Tier 2):**
writes (`TableInfo.supports_insert` etc. are read but unused — read-only connector,
matching `vgi-spark`'s stated v1 non-goal); scalar per-value memoization *across*
chunks/queries (the client-side `vgi_exchange_input_dedup` mirror above is implemented
and scoped to *within* one chunk/call — a genuine cross-call cache, mirroring the C++
side's separate `vgi_result_cache_per_value`, needs its own TTL/eviction design and is
lower priority than the correctness/coverage gaps that came before it in this plan);
companion-catalog federation (see "Multi-branch tables" above — no established
Polars-side "attach another catalog" concept, needs its own research pass);
`TableInfo.supports_time_travel` per-table discovery (see "Time travel" above — a
genuine wire-schema addition, needs a protocol bump + C++ codegen regen).

**Implemented: blended (row-transform) table functions / correlated LATERAL
joins** (`cat.row_transform_function(schema, name)`, `src/vgi_polars/
_row_transform.py`). A worker function called directly with caller-supplied
literal or column arguments and no separate table input — the shape the
public `open_meteo` worker (github.com/Query-farm/vgi-open-meteo,
`geocoding`/`forecast_hourly`/etc.) needs, since it has zero plain catalog
tables reachable via `cat.table(schema, name)` and `table_in_out_function`
can't cover it either (it requires an existing `LazyFrame`/`DataFrame` as
input, which a blended call in its literal-args form doesn't have). Two call
shapes, one bridge, dispatched on whether `lf` is given: the correlated/
column shape (`FROM t, f(t.x)`) routes through `LazyFrame.map_batches`,
decoding the worker's `vgi_rpc.parent_row` provenance (via vgi-python's
`Client.table_in_out_function(parent_row_callback=...)`, v0.29.5+) to
correctly re-associate a 1->N/1->0 emit's outer columns; the bare
literal-call shape (`f(None, 'some place')`) routes through `pl.defer`
instead and needs no provenance at all (there's no outer frame to stamp).
See `_row_transform.py`'s module docstring for the full design — gather
safety, the outer-column policy, and the dedup group-and-replicate
composition (two real bugs were caught and fixed against exactly those
during implementation).

**`launch:` LOCATIONs — implemented in vgi-python, not yet wired in here.**
`vgi.client.Client.from_launch(...)` / `transport="launch"` (vgi-python
v0.29.4+) spawns-or-reuses a warm worker over an AF_UNIX socket via
`vgi_rpc.launcher`, mirroring the existing `tcp` transport's shape — this
closed the gap this section used to describe (`Client`'s transport used to
be a closed `Literal["subprocess", "http", "tcp"]` with zero unix-socket
equivalent to `vgi_rpc.launcher`). `attach()`'s `_detect_transport` here
still only recognizes `http(s)://`/`tcp://`/bare-command — extending it for
`launch:`/`unix://` is now a small, well-scoped addition (mirror the
existing three-branch shape in `catalog.py`), not blocked on any upstream
gap.

**`container://`/`github://` LOCATIONs — not possible without a Python
transport layer (a substantially larger, separate effort — confirmed, not
deferred by scope choice).** Implemented **only** in this repo's C++
extension (`vgi_container_runtime.cpp` launching an OCI runtime,
`vgi_github.cpp` downloading+extracting a release archive), with zero
vgi-python-side equivalent to adapt. Supporting either in vgi-polars means
building that transport layer from scratch in Python.

**Not possible without upstream changes:** a `pl.sql(...)` string surface for VGI table
functions — `pl.SQLContext.register` only accepts an already-built frame, no
user-registrable table-valued function exists in Polars SQL. The Python API
(`cat.table(...).scan()`) is the only entry point.

## Build / Test

```bash
cd ~/Development/vgi-polars
uv sync
uv run pytest -v
```

`vgi-python` is sourced from the local sibling checkout (`[tool.uv.sources]` in
`pyproject.toml`, path `../vgi-python`), not the published PyPI release — this repo
tracks vgi-python's client-side surface as both develop together, the same way
`vgi-spark`'s `settings.gradle.kts` composite-builds a sibling `vgi-java` checkout.

Integration tests use `vgi-fixture-worker` against the `example` catalog (schema
`data`, table `numbers`; schema `main`, scalar function `multiply`; among many others —
see `~/Development/vgi-python/vgi/_test_fixtures/worker.py`):

```bash
VGI_TEST_WORKER=~/Development/vgi-python/.venv/bin/vgi-fixture-worker uv run pytest -v
```

Default (no env override) resolves the same path (`{VGI_PYTHON}/.venv/bin/
vgi-fixture-worker`, `VGI_PYTHON` overriding which checkout).

**Tests point at the venv binary directly rather than `uv run --project
~/Development/vgi-python vgi-fixture-worker`** (the convention every other repo in
this family uses for `VGI_TEST_WORKER`) — **not** because of a hang/deadlock bug.

That was the initial (wrong) diagnosis: two early runs of `Client.scalar_function`
against a `uv run --project ...`-wrapped worker command appeared to hang past a 20-60s
timeout, while the same call against the bare venv binary always returned in well under
a second. Investigated properly (2026-08-27) by reproducing under controlled load:
- 8 sequential runs of the exact original repro: all completed normally.
- 4 concurrent runs of the same repro: all completed normally.
- 12 concurrent runs **plus** a real `uv sync --reinstall-package vgi-python` running
  at the same time (deliberately recreating the kind of contention this machine was
  actually under during the original session — many other long-lived `uv run`
  processes from unrelated parallel sessions were already running, per `ps -ef`):
  every run still completed, but took **~22-23 seconds each**, well past the 20s
  timeout the very first diagnosis used, and *nearly* past the original 60s timeout
  had the machine been even a little more loaded.

So the effect is real (contended `uv run --project` invocations get slow — `uv`
resolves/verifies the project environment on every invocation before exec'ing the
target command), but it is **slowness that scales with concurrent `uv` activity on
the machine, not a deadlock**: every single run, across ~29 total attempts at every
concurrency level tried, eventually completed with the correct result. There is no
evidence of a genuine protocol-level or IPC-framing bug in `vgi.client.Client`,
`vgi-rpc`, or the const-arg scalar exchange path specifically — the const-arg call
shape that first hit this was incidental, not causal.

The venv-binary path is kept anyway because it's still strictly better for a test
suite: it skips `uv run`'s per-invocation resolve/verify step entirely (no
environment-contention exposure at all, not just less of it), and it's exactly what
`vgi-python`'s own test suite does (`tests/conftest.py`'s `fixture_worker` fixture
resolves the bare, already-installed `vgi-fixture-worker` console script directly,
never through `uv run`). `vgi-fixtures` is deliberately not a vgi-polars dependency
(the venv path above is `vgi-python`'s own venv, not this repo's) — this is purely a
robustness/speed choice, not a workaround for broken functionality.

HTTP-transport tests spin up `vgi-fixture-http` from the same venv's bin/.

**TCP-transport tests reuse a launcher-style warm worker, not a fresh spawn per
run.** `vgi.client.Client` has no unix-socket transport, so the real
`vgi_rpc.launcher` module (unix-socket-only, hash+flock+probe+spawn, the same
mechanism the DuckDB C++ extension's `launch:`/`unix://` LOCATIONs use) has
nothing to pair with here — see the "Not possible without upstream changes"
entry in Scope. `conftest.py`'s `_launch_tcp_worker` is the TCP analogue,
hand-rolled with stdlib `fcntl.flock` (no new dependency): keyed by a hash of
the worker argv, it probes a recorded port for a live worker and reuses it, or
spawns one **detached** (`start_new_session=True`) with `--tcp 127.0.0.1:0
--idle-timeout 1800` and records the port it announces (`TCP:127.0.0.1:<port>`
on stdout — the same discovery-line contract `vgi_rpc.launcher` uses for
`UNIX:<path>`). "Detached" is the point: the worker outlives the pytest
process that spawned it, so the *next* `pytest`/CI-dry-run invocation reuses
it too (self-terminating after 30 min idle) — module-level `WorkerPool`s in
vgi-python's `Client`/`CatalogClientMixin` already amortize subprocess spawn
*within* one pytest process, but each fresh `pytest` process starts those
pools cold, which is the actual repeated-Python-startup cost this avoids
(measured live: 0.84s for the first TCP-transport test-file run in a session,
0.25s for the immediately following one, one worker process the whole time —
confirmed via `ps aux`).

## HTTP transport

Fully supported, same as subprocess — `vp.attach(location, name=...)` auto-detects
transport from `location`'s scheme (`http://`/`https://` -> HTTP, `tcp://` -> TCP,
anything else -> subprocess argv); see `catalog.py`'s `_detect_transport`. Pass
`transport="http"`/`"tcp"`/`"subprocess"` explicitly to override. `tests/test_http_
transport.py` exercises catalog/scan/scalar over HTTP; `conftest.py`'s
`http_worker_base_url` fixture spins up `vgi-fixture-http` per session.

## TCP transport

Also fully supported (raw Arrow-IPC framing, no auth/encryption — loopback /
trusted networks only, same caveat as vgi-python's own `Client.from_tcp`
docstring). `tests/test_tcp_transport.py` exercises catalog/scan/scalar/
required_filters over TCP — previously only scheme-detection was unit-tested
(`test_catalog.py`), a real live-integration gap now closed. See "Build /
Test" above for how the backing worker is spawned (launcher-style, reused
across test runs).

## CI

`.github/workflows/ci.yml` — `lint` (ruff) + `test` (the full pytest suite, both
transports, with coverage). Checks out `Query-farm/vgi-python` (public on GitHub) as
a **sibling directory** alongside this repo, exactly reproducing the local dev layout
`[tool.uv.sources]` expects (`path = "../vgi-python"`) — the same pattern vgi-python's
own `integration.yml` uses to pull in a second repo, and the same rationale as
`~/Development/vgi`'s `VGI_FIXTURES_REF`: pinned to a release tag (`VGI_PYTHON_REF:
v0.29.2`, workflow_dispatch-overridable), not `main`, so a vgi-python change can't
silently flip this repo's CI red with nothing changed here. Verified end-to-end by
hand before committing: cloned `vgi-python@v0.29.2` fresh into a scratch sibling
layout (not the locally-modified checkout) and ran the full suite against it —
130/130 passed, 0 skipped, both transports, exactly reproducing what the workflow
does.

`v0.29.2` (`chore: 0.29.2 — client support for splits, multi-branch, cache
metadata, time travel`) is the release that closed the "Version-pin gap" Scope
entries above cite — splits (`Client.table_function_plan`/`split_tokens`),
multi-branch tables (`Client.table_scan_branches_get`), the result cache
(`table_function(batch_metadata_callback=...)`), and time travel (`table_get`/
`table_function(at_unit=..., at_value=...)`) all landed in it, client-only, no
protocol bump. Bumped in lockstep here the same day — the same discipline
`~/Development/vgi`'s CLAUDE.md documents for `VGI_FIXTURES_REF`: never point at
`main`, always a released tag, bump only when this repo's own usage of the new
surface is ready to be exercised for real in CI.

## Coverage

`uv run pytest --cov=vgi_polars --cov-report=term-missing` — **90%** line coverage
(887 statements, 93 missed). Per-file: `__init__.py`/`_arguments.py`/`errors.py`
100%; `_multi_branch.py` 95%; `_scalar.py` 95%; `_source.py` 91%; `catalog.py` 90%;
`_result_cache.py`/`_aggregate.py`/`_table_in_out.py` 88-89%; `_filter_translate.py`
85%; `table.py` 83%. The gaps are overwhelmingly defensive/error-handling branches
not yet exercised by a dedicated test (e.g. a few `translate_predicate`
shape-mismatch fallbacks in `_filter_translate.py`) rather than untested core
logic — every file's primary code path is covered by the end-to-end tests.
Uploaded as a `coverage-xml` artifact on each CI run. **130 tests, all running for
real (0 skipped)** against the pinned `v0.29.2`, both
subprocess and HTTP transports, plus a dedicated TCP-transport suite.
