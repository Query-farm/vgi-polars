# Copyright 2026 Query Farm LLC - https://query.farm

"""vgi-polars: a Polars client for VGI.

Lets a `polars.LazyFrame`/`polars.DataFrame` scan a VGI catalog's tables and
call its scalar functions — the same role the VGI DuckDB extension plays for
DuckDB, but for Polars, built as a thin adapter over vgi-python's existing
pure-Python `vgi.client.Client` rather than a new protocol implementation.

    import vgi_polars as vp

    with vp.attach("uv run --project ~/Development/vgi-python vgi-fixture-worker",
                    name="example") as cat:
        t = cat.table("data", "numbers")
        print(t.schema)
        print(t.scan().filter(pl.col("value") > 90).collect())

See README.md and CLAUDE.md for the full API and design principles.
"""

from __future__ import annotations

from vgi_polars.catalog import VgiCatalog, attach
from vgi_polars.errors import VgiPolarsError
from vgi_polars.table import VgiTable

__all__ = ["VgiCatalog", "VgiPolarsError", "VgiTable", "attach"]
