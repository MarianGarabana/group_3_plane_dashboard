"""
explore.py — ad-hoc DB2 exploration helpers.

These are NOT used by the dashboard or the extraction pipeline. They live
here, separate from db.py, so the data-access module only contains what the
running app depends on. Use them interactively while figuring out a schema.

Example:
    python -c "from explore import preview_table; print(preview_table('ATTGRP3', 'FLIGHTS'))"
"""

import polars as pl

from db import _read_sql, make_engine, qualified_table


def preview_table(schema: str, table: str, limit: int = 10, engine=None) -> pl.DataFrame:
    """Return the first `limit` rows of a table as a Polars DataFrame."""
    if engine is None:
        engine = make_engine()
    query = (
        f"SELECT * FROM {qualified_table(schema, table)} "
        f"FETCH FIRST {int(limit)} ROWS ONLY"
    )
    return _read_sql(query, engine)


def count_rows(schema: str, table: str, engine=None) -> int:
    """Return the exact row count of a table as a Python int."""
    if engine is None:
        engine = make_engine()
    query = f"SELECT COUNT(*) AS n_rows FROM {qualified_table(schema, table)}"
    return _read_sql(query, engine).item(0, "n_rows")
