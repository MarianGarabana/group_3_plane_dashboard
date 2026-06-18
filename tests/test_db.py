"""Tests for db.py.

The pure helpers (identifier quoting, column normalisation, the missing-file
guard, credential validation) run with no database. The extraction queries
need a live DB2 connection and are marked `db`, so they skip cleanly in CI
and on any machine without reachable credentials.
"""

import os

import polars as pl
import pytest

import db


# ── pure helpers ───────────────────────────────────────────────────────────────

def test_q_ident_quotes():
    assert db.q_ident("FLIGHTS") == '"FLIGHTS"'


def test_q_ident_escapes_embedded_quote():
    assert db.q_ident('a"b') == '"a""b"'


def test_qualified_table():
    assert db.qualified_table("ATTGRP3", "TICKETS") == '"ATTGRP3"."TICKETS"'


def test_normalize_column_names_strips_and_lowercases():
    df = pl.DataFrame({" COL A ": [1], "Col_B": [2]})
    out = db.normalize_column_names(df)
    assert out.columns == ["col a", "col_b"]


# ── credential validation ──────────────────────────────────────────────────────

def test_make_engine_requires_credentials(monkeypatch):
    monkeypatch.delenv("DB_USERNAME", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    with pytest.raises(RuntimeError) as exc:
        db.make_engine()
    assert "DB_USERNAME" in str(exc.value)


# ── missing-file guard ──────────────────────────────────────────────────────────

def test_read_extract_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    with pytest.raises(db.DataNotGeneratedError):
        db.read_extract("does_not_exist.parquet")


def test_read_extract_reads_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    pl.DataFrame({"a": [1, 2]}).write_parquet(tmp_path / "x.parquet")
    out = db.read_extract("x.parquet")
    assert out["a"].to_list() == [1, 2]


# ── live DB integration (skipped without a reachable DB) ─────────────────────────

@pytest.fixture
def db_engine():
    if not (os.getenv("DB_USERNAME") and os.getenv("DB_PASSWORD")):
        pytest.skip("DB credentials not set; skipping live DB integration tests")
    try:
        engine = db.make_engine()
        if not db.test_connection(engine):
            pytest.skip("DB connection test failed; skipping live DB tests")
    except Exception as exc:  # driver missing or host unreachable
        pytest.skip(f"DB not reachable: {exc}")
    return engine


@pytest.mark.db
def test_extract_revenue_has_year(db_engine):
    df = db.extract_revenue(db_engine)
    assert "yr" in df.columns
    assert df.height > 0


@pytest.mark.db
def test_extract_fuel_has_year(db_engine):
    df = db.extract_fuel(db_engine)
    assert "yr" in df.columns  # the audit fix
    assert "total_fuel_gallons" in df.columns
    assert df.height > 0
