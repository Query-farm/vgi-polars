# Copyright 2026 Query Farm LLC - https://query.farm

"""Docstring-consistency gate.

Runs pydoclint over the ``src/vgi_polars/`` package as part of the test suite,
mirroring vgi-python's own ``tests/test_docstrings.py`` exactly (same
mechanism, same rationale — see that file). pydoclint complements ruff's
``D`` rules: ruff checks docstring *shape*, while pydoclint verifies that
documented arguments, return values, yields, and dataclass attributes
actually match the code. Configuration lives in ``[tool.pydoclint]`` in
``pyproject.toml`` — this test invokes the same CLI, so there is no
duplicated rule set.

pydoclint runs through ``uvx`` (an isolated, ephemeral environment) rather
than as a project dependency, for the same reason vgi-python keeps it out of
its own project env: it requires ``docstring-parser-fork``, which clobbers
the canonical ``docstring-parser`` (a ``vgi-rpc`` runtime dependency, pulled
in transitively here through ``vgi-python``) in the shared
``docstring_parser`` import namespace when both are installed into the same
environment.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# A real violation line looks like ``    42: DOC101: ...``. If pydoclint exits
# non-zero without emitting any such code, it failed to *run* rather than
# finding violations.
_VIOLATION_RE = re.compile(r"\bDOC\d{3}\b")


def test_pydoclint_clean() -> None:
    """Check that ``src/vgi_polars/`` passes the pydoclint docstring gate."""
    uvx = shutil.which("uvx") or shutil.which("uv")
    if uvx is None:  # pragma: no cover - uv is always present in dev/CI
        pytest.skip("uv/uvx is not available to run pydoclint")

    # `uvx pydoclint ...` == `uv tool run pydoclint ...`; both run pydoclint in
    # an isolated env so docstring-parser-fork never enters the project tree.
    # Pin the ephemeral env to *this* interpreter (``--python sys.executable``)
    # so pydoclint parses with a Python new enough for the repo's syntax.
    base = [uvx] if Path(uvx).name == "uvx" else [uvx, "tool", "run"]
    cmd = [*base, "--python", sys.executable, "pydoclint", "--config", "pyproject.toml", "src/vgi_polars/"]
    result = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if result.returncode == 0:
        return

    if not _VIOLATION_RE.search(output):  # pragma: no cover - env-dependent (e.g. offline uvx)
        pytest.skip(f"pydoclint could not run via uvx:\n{output}")

    pytest.fail(
        "pydoclint found docstring violations:\n\n"
        f"{output}\n"
        "Fix the docstrings, or run `uvx pydoclint --config pyproject.toml "
        "--generate-baseline=True --baseline=.pydoclint-baseline src/vgi_polars/` "
        "to defer them (see [tool.pydoclint] in pyproject.toml)."
    )
