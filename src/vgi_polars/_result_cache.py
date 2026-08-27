# Copyright 2026 Query Farm LLC - https://query.farm

"""Minimal table-function result cache — in-memory, TTL-based,
producer-mode (whole-scan) only.

Mirrors the smallest slice of the DuckDB C++ extension's own much larger
`vgi_result_cache.cpp` (see that repo's CLAUDE.md "Table-Function Result
Cache" section): when a worker advertises `vgi.cache.ttl` on its result's
first batch, the *complete* raw result (before any local re-filter/re-select
— see below) is cached in memory keyed by call identity, and an identical
repeat scan is served from memory without a worker round-trip.

This needed the `batch_metadata_callback` vgi-python fix first (see
CLAUDE.md's "Table-function result cache" section): `Client.
_table_function_parallel` discarded `AnnotatedBatch.custom_metadata`
unconditionally, so `vgi.cache.*` was structurally unreachable through
`table_function()`'s public generator.

**What's deliberately NOT here, matching the "minimal slice" scope** (the
C++ side's disk tier / projection-coverage reuse / conditional revalidation /
per-partition caching / exchange-mode memoization are each their own
multi-milestone feature there): no disk tier (memory only, lost on process
exit); no revalidation (`etag`/`stale_while_revalidate` keys are parsed and
ignored — a stale entry is just evicted, never conditionally refreshed); no
byte/entry caps (unbounded — a pathologically large or numerous cacheable
result grows this process's memory with no eviction beyond TTL expiry); no
`vgi.cache.scope=transaction` support (transaction-scoped entries are never
cached at all, not incorrectly cached catalog-wide — see `_scope_is_supported`).

**Identity scoping.** The cache key folds in the catalog's
`attach_opaque_data` (the per-attach session token `Client.catalog_attach`
returns) alongside function/schema/args/projection/pushdown-filters/AT-clause
— so two different attaches of the same worker (even under different
credentials) never share an entry. This is a *reasonable* boundary, not the
C++ side's audited one (which folds in a verified auth-principal fingerprint,
not just the opaque session token) — do not treat this cache as a security
boundary for credential-scoped data without re-auditing that gap first.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["flush", "get_default_cache"]

_CACHE_TTL_KEY = b"vgi.cache.ttl"
_CACHE_NO_STORE_KEY = b"vgi.cache.no_store"
_CACHE_SCOPE_KEY = b"vgi.cache.scope"
_CACHE_SCOPE_TRANSACTION = b"transaction"


@dataclass
class _CacheEntry:
    batches: list[pa.RecordBatch]
    expires_at: float


class ResultCache:
    """Process-wide, in-memory, thread-safe TTL cache of raw (pre-local-
    filter) table-function result batches. One instance is enough — no
    per-catalog state to isolate beyond what the key already encodes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[Any, ...], _CacheEntry] = {}

    def get(self, key: tuple[Any, ...]) -> list[pa.RecordBatch] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                del self._entries[key]
                return None
            return entry.batches

    def put(self, key: tuple[Any, ...], batches: list[pa.RecordBatch], ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._entries[key] = _CacheEntry(batches=batches, expires_at=time.monotonic() + ttl_seconds)

    def flush(self) -> int:
        """Drop every entry; returns the count dropped."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            return n


def parse_cache_control(metadata: Any) -> float | None:
    """Return the freshness TTL in seconds if `metadata` (a batch's
    `custom_metadata`, or `None`) advertises cacheability — `None` if it
    doesn't, is `no_store`, or is transaction-scoped (unsupported here, see
    module docstring). `vgi.cache.expires` (an absolute RFC3339 timestamp,
    the alternative to `ttl`) is not parsed in this minimal slice — only
    `ttl` opts in."""
    if metadata is None:
        return None
    md = dict(metadata)
    if md.get(_CACHE_NO_STORE_KEY):
        return None
    if md.get(_CACHE_SCOPE_KEY) == _CACHE_SCOPE_TRANSACTION:
        return None
    ttl_bytes = md.get(_CACHE_TTL_KEY)
    if ttl_bytes is None:
        return None
    try:
        return float(ttl_bytes)
    except ValueError:
        return None


_default_cache = ResultCache()


def get_default_cache() -> ResultCache:
    return _default_cache


def flush() -> int:
    """Drop every cached entry; returns the count dropped."""
    return _default_cache.flush()
