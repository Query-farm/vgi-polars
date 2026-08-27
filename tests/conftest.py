# Copyright 2026 Query Farm LLC - https://query.farm

"""Shared fixtures for vgi-polars' test suite.

Drives `vgi-fixture-worker` (vgi-python's reference test/example worker,
catalog `example`) over the subprocess and HTTP transports. Mirrors
`vgi-sqlite/test/integration/conftest.py`'s `VGI_TEST_WORKER`/`VGI_PYTHON`
override convention used across this repo family, with one deliberate
deviation — see `worker_location`'s docstring.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

import vgi_polars as vp


def _vgi_python_venv() -> Path:
    override = os.environ.get("VGI_PYTHON")
    root = Path(override) if override else Path.home() / "Development" / "vgi-python"
    return root / ".venv" / "bin"


@pytest.fixture(scope="session")
def worker_location() -> str:
    """The `example` catalog's worker command.

    Deliberately NOT `uv run --project <vgi-python> vgi-fixture-worker` (every
    other repo in this family's convention) — not because it's broken (a `uv
    run`-wrapped worker was confirmed, under repeated stress testing, to always
    eventually return the correct result; see CLAUDE.md's Build/Test section),
    but because `uv run` re-resolves/verifies the project environment on every
    invocation, which gets slow under concurrent `uv` activity elsewhere on the
    machine (measured ~22-23s per call under load vs. sub-second here). Points
    directly at vgi-python's own venv console script instead, skipping that
    step entirely — exactly what vgi-python's own test suite resolves via a
    bare, PATH-found `vgi-fixture-worker` (see
    `~/Development/vgi-python/tests/conftest.py`).
    """
    override = os.environ.get("VGI_TEST_WORKER")
    if override:
        return override
    return str(_vgi_python_venv() / "vgi-fixture-worker")


@pytest.fixture(scope="session")
def bad_protocol_worker_location() -> str:
    """A worker that enforces an incompatible `protocol_version` — for
    testing that the mismatch surfaces as a clean `VgiPolarsError`, not a
    hang or a raw traceback. Mirrors vgi-python's own
    `tests/conformance/test_protocol_version.py`."""
    return str(_vgi_python_venv() / "vgi-fixture-bad-protocol-worker")


@pytest.fixture(scope="session")
def versioned_worker_location() -> str:
    """A worker (catalog `versioned`) that validates `data_version_spec`/
    `implementation_version` at ATTACH time and returns `resolved_data_version`/
    `resolved_implementation_version` — see `vgi/_test_fixtures/versioned.py`."""
    return str(_vgi_python_venv() / "vgi-fixture-versioned-worker")


@pytest.fixture(scope="session")
def bad_enum_worker_location() -> str:
    """A worker that advertises an unrecognized wire-enum value (`double`'s
    `null_handling`) — for testing that a metadata-parse failure surfaces as
    a clean `VgiPolarsError`, not a hang or a raw traceback."""
    return str(_vgi_python_venv() / "vgi-fixture-bad-enum-worker")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launcher_state_dir() -> Path:
    base = Path(tempfile.gettempdir()) / f"vgi-polars-test-launcher-{os.getuid()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _tcp_alive(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _launch_tcp_worker(argv: list[str], *, idle_timeout: float = 1800.0) -> int:
    """Reuse-or-spawn a single warm `--tcp` worker process, keyed by `argv` —
    the TCP analogue of `vgi_rpc.launcher`'s hash+flock+probe+spawn design
    (that module is unix-socket-only, and `vgi.client.Client` has no
    unix-socket transport to pair it with — only subprocess/http/tcp; see
    CLAUDE.md's transport-gap note). Spawned **detached**
    (`start_new_session=True`) so the worker outlives this pytest process
    and is reused by the *next* `pytest`/CI-dry-run invocation too, not just
    within one run — that cross-invocation reuse is the actual point: the
    module-level `WorkerPool`s in vgi-python's `Client`/`CatalogClientMixin`
    already amortize subprocess spawn *within* one pytest process, but each
    fresh `pytest` process starts those pools cold. The worker
    self-terminates after `idle_timeout` seconds with no connections
    (`--idle-timeout`), so a stale entry self-heals with no explicit
    teardown needed. Returns the bound port.
    """
    state_dir = _launcher_state_dir()
    key = hashlib.sha256(json.dumps(argv, sort_keys=True).encode()).hexdigest()[:16]
    lock_path = state_dir / f"{key}.lock"
    meta_path = state_dir / f"{key}.json"

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                port = meta["port"]
                if _tcp_alive(port):
                    return port
            proc = subprocess.Popen(  # noqa: S603 - argv built from trusted fixture paths, not user input
                [*argv, "--tcp", "127.0.0.1:0", "--idle-timeout", str(idle_timeout)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=(sys.platform != "win32"),
            )
            assert proc.stdout is not None
            line = proc.stdout.readline()
            proc.stdout.close()
            prefix = "TCP:127.0.0.1:"
            if not line.startswith(prefix):
                proc.kill()
                raise RuntimeError(f"expected 'TCP:127.0.0.1:<port>' announce line, got {line!r}")
            port = int(line[len(prefix) :].strip())
            meta_path.write_text(json.dumps({"port": port, "pid": proc.pid}))
            return port
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _spawn_http_worker(extra_env: dict[str, str] | None = None):
    """Spawn `vgi-fixture-http` on a free port, wait for it to accept
    connections, and yield its base URL — shared by the anonymous and
    bearer-auth HTTP fixtures below. Skips (doesn't fail) the calling
    fixture's tests if the binary is missing or never comes up."""
    http_bin = _vgi_python_venv() / "vgi-fixture-http"
    if not http_bin.exists():
        pytest.skip(f"{http_bin} not found — build/sync vgi-python with the 'http' extra")

    env = dict(os.environ)
    env.update(extra_env or {})
    port = _free_port()
    proc = subprocess.Popen(
        [str(http_bin), "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    tail: list[str] = []

    def _drain(stream) -> None:
        for line in stream:
            tail.append(line)
            del tail[:-50]

    threads = [threading.Thread(target=_drain, args=(s,), daemon=True) for s in (proc.stdout, proc.stderr)]
    for t in threads:
        t.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                ready = True
                break
        except OSError:
            time.sleep(0.2)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        pytest.skip(f"vgi-fixture-http didn't start on {base_url} (rc={proc.returncode}). Output:\n{''.join(tail)}")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="session")
def tcp_worker_base_url(worker_location: str) -> str:
    """A `tcp://127.0.0.1:<port>` location for the `example` catalog, backed
    by a launcher-style reused-or-spawned warm worker (see
    `_launch_tcp_worker`) — `vp.attach()` auto-detects the `tcp://` scheme
    with no code changes needed (`catalog.py`'s `_detect_transport`)."""
    port = _launch_tcp_worker([worker_location])
    return f"tcp://127.0.0.1:{port}"


@pytest.fixture(scope="session")
def http_worker_base_url():
    """Session-scoped `vgi-fixture-http` (HTTP-transport counterpart of
    `worker_location`), no bearer auth (anonymous mode)."""
    yield from _spawn_http_worker()


@pytest.fixture(scope="session")
def http_bearer_token() -> str:
    return "vgi-polars-test-token"


@pytest.fixture(scope="session")
def http_bearer_worker_base_url(http_bearer_token: str):
    """Session-scoped `vgi-fixture-http` with bearer auth enforced
    (`VGI_BEARER_TOKENS`) — separate server instance from
    `http_worker_base_url`, which must stay anonymous for the other HTTP
    tests. Mirrors `vgi-sqlite/test/integration/conftest.py`'s
    `http_bearer_token`/`VGI_BEARER_TOKENS` fixture pattern."""
    yield from _spawn_http_worker({"VGI_BEARER_TOKENS": f"{http_bearer_token}=test-principal"})


@pytest.fixture
def catalog(worker_location: str):
    """A fresh `VgiCatalog` attached to the `example` catalog over subprocess."""
    with vp.attach(worker_location, name="example") as cat:
        yield cat


@pytest.fixture
def http_catalog(http_worker_base_url: str):
    """A fresh `VgiCatalog` attached to the `example` catalog over HTTP."""
    with vp.attach(http_worker_base_url, name="example") as cat:  # http:// auto-detected
        yield cat


@pytest.fixture
def tcp_catalog(tcp_worker_base_url: str):
    """A fresh `VgiCatalog` attached to the `example` catalog over TCP."""
    with vp.attach(tcp_worker_base_url, name="example") as cat:  # tcp:// auto-detected
        yield cat


@pytest.fixture
def accumulate_catalog(worker_location: str):
    """A fresh `VgiCatalog` attached to the `accumulate` catalog — a
    *separate* catalog from `example` (`vgi-fixture-worker`'s MetaWorker
    dispatches by attach name), hosting the `accumulate`/`accumulate_read`/
    `accumulate_clear` table-in-out functions."""
    with vp.attach(worker_location, name="accumulate") as cat:
        yield cat
