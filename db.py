"""
db.py — DB2 connection, helper functions, and data extraction.

Workflow: run this file directly (python db.py) to pull all three
pre-aggregated datasets from the DB and save them as Parquet files
under data/. The Streamlit app then reads those files — no live DB
connection at dashboard runtime.

Credentials are read from environment variables (a local .env file works
via python-dotenv). See the README for the required keys. There are no
hardcoded credential defaults: the password is a secret and must never
live in source.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import polars as pl
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── connection parameters ────────────────────────────────────────────────────
#
# Host / port / database name carry non-secret defaults for convenience.
# Username and password have NO defaults — they are secrets and are required
# at connect time (validated in make_engine, not at import, so the pure
# helpers below can be imported and unit-tested without any credentials).

DB_HOST = os.getenv("DB_HOST", "52.211.123.34")
DB_PORT = int(os.getenv("DB_PORT", "25010"))
DB_NAME = os.getenv("DB_NAME", "ATTPLANE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
SCHEMA = os.getenv("DB_SCHEMA", "ATTGRP3")

DATA_DIR = Path(__file__).parent / "data"

# Env var names required to open a connection — used for fail-fast messaging.
_REQUIRED_ENV = ("DB_USERNAME", "DB_PASSWORD")


# ── engine factory ───────────────────────────────────────────────────────────

def make_engine():
    """Return a SQLAlchemy engine for DB2 via ibm_db_sa dialect.

    Fails fast with a clear message if the required credential env vars are
    missing, rather than producing a cryptic driver error at query time.
    """
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in a .env file (see README → How to Install)."
        )

    user = quote_plus(os.environ["DB_USERNAME"])
    password = quote_plus(os.environ["DB_PASSWORD"])
    url = f"db2+ibm_db://{user}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    # pool_pre_ping checks the connection is alive before handing it out
    return create_engine(url, pool_pre_ping=True)


# ── identifier helpers ───────────────────────────────────────────────────────

def q_ident(name: str) -> str:
    """Double-quote a DB2 identifier (schema or table name) to handle
    reserved words and mixed case safely."""
    return '"' + name.replace('"', '""') + '"'


def qualified_table(schema: str, table: str) -> str:
    """Return schema.table with both parts safely quoted."""
    return f"{q_ident(schema)}.{q_ident(table)}"


# ── low-level query helper ───────────────────────────────────────────────────

def _read_sql(query: str, engine) -> pl.DataFrame:
    """Execute a SQL query and return a Polars DataFrame."""
    with engine.connect() as conn:
        df = pl.read_database(query=query, connection=conn)
    return normalize_column_names(df)


# ── column normalisation ─────────────────────────────────────────────────────

def normalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Strip whitespace and lowercase all column names so Python code
    never has to deal with UPPER_CASE DB2 identifiers."""
    return df.rename({col: col.strip().lower() for col in df.columns})


def test_connection(engine) -> bool:
    """Ping the DB with a lightweight query. DB2 uses SYSIBM.SYSDUMMY1
    as its equivalent of PostgreSQL's SELECT 1."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1 AS ok FROM SYSIBM.SYSDUMMY1"))
        row = result.fetchone()
    return row is not None and row[0] == 1


# ── pre-aggregated extraction queries ────────────────────────────────────────
#
# TICKETS has 35 million rows — never load it raw.
# All three queries push GROUP BY into the DB so Python only receives
# summary rows.

def extract_revenue(engine) -> pl.DataFrame:
    """Query 1: revenue + taxes grouped by route / year / cabin class.

    Grouping by YEAR(DEPARTURE) and CLASS means Python receives one row
    per (route, year, class) combination — roughly 59 routes × 16 years
    × 3 classes ≈ 3 000 rows.
    """
    query = f"""
        SELECT
            t.ROUTE_CODE,
            r.ORIGIN,
            r.DESTINATION,
            YEAR(t.DEPARTURE)              AS yr,
            t.CLASS,
            COUNT(t.TICKET_ID)             AS ticket_count,
            SUM(t.PRICE)                   AS net_revenue,
            SUM(t.AIRPORT_TAX + t.LOCAL_TAX) AS total_taxes,
            SUM(t.TOTAL_AMOUNT)            AS gross_revenue
        FROM {qualified_table(SCHEMA, "TICKETS")} t
        JOIN {qualified_table(SCHEMA, "ROUTES")}  r
          ON t.ROUTE_CODE = r.ROUTE_CODE
        GROUP BY
            t.ROUTE_CODE,
            r.ORIGIN,
            r.DESTINATION,
            YEAR(t.DEPARTURE),
            t.CLASS
    """
    df = _read_sql(query, engine)
    # DB2 SUM() returns DECIMAL — cast money columns to Float64 so Polars
    # arithmetic in analysis.py works without Decimal-specific handling.
    return df.with_columns(
        pl.col("net_revenue").cast(pl.Float64),
        pl.col("total_taxes").cast(pl.Float64),
        pl.col("gross_revenue").cast(pl.Float64),
    )


def extract_fuel(engine) -> pl.DataFrame:
    """Query 2: fuel consumption grouped by route + aircraft model + year.

    total_fuel_gallons = flights operated × fuel burn rate × flight hours.
    Multiplying by the fuel price in analysis.py gives estimated fuel cost.

    YEAR(f.DEPARTURE) is grouped in so fuel carries the same year dimension
    as revenue. Without it, any year-filtered view would pair filtered
    revenue against all-history fuel — see analysis.py for why that matters.
    The grouping keeps the fuel burn attributes (FUEL_GALLONS_HOUR,
    FLIGHT_MINUTES) in the result so analysis.py can recalculate at any
    fuel price without re-querying the DB.
    """
    # Note: the multiplication COUNT * fuel_rate * hours overflows DB2 INTEGER,
    # so we fetch the raw components and compute total_fuel_gallons in Polars.
    query = f"""
        SELECT
            f.ROUTE_CODE,
            a.MODEL,
            YEAR(f.DEPARTURE)  AS yr,
            r.FLIGHT_MINUTES,
            a.FUEL_GALLONS_HOUR,
            COUNT(*) AS flights_operated
        FROM {qualified_table(SCHEMA, "FLIGHTS")}   f
        JOIN {qualified_table(SCHEMA, "AIRPLANES")} a
          ON f.AIRPLANE = a.AIRCRAFT_REGISTRATION
        JOIN {qualified_table(SCHEMA, "ROUTES")}    r
          ON f.ROUTE_CODE = r.ROUTE_CODE
        GROUP BY
            f.ROUTE_CODE,
            a.MODEL,
            YEAR(f.DEPARTURE),
            r.FLIGHT_MINUTES,
            a.FUEL_GALLONS_HOUR
    """
    df = _read_sql(query, engine)
    # Compute gallons in Polars — avoids DB2 INTEGER overflow
    return df.with_columns(
        (
            pl.col("flights_operated").cast(pl.Float64)
            * pl.col("fuel_gallons_hour").cast(pl.Float64)
            * (pl.col("flight_minutes").cast(pl.Float64) / 60.0)
        ).alias("total_fuel_gallons")
    )


def extract_airports(engine) -> pl.DataFrame:
    """Query 3: full AIRPORTS table (30 rows — safe to load completely).

    Used to enrich routes with continent and city names for filtering
    and chart colouring.
    """
    query = f"SELECT * FROM {qualified_table(SCHEMA, 'AIRPORTS')}"
    return _read_sql(query, engine)


# ── reading generated extracts (no DB needed) ─────────────────────────────────

class DataNotGeneratedError(FileNotFoundError):
    """Raised when a Parquet extract is requested before db.py produced it."""


def read_extract(name: str) -> pl.DataFrame:
    """Read one generated Parquet extract from DATA_DIR.

    Raises DataNotGeneratedError (a FileNotFoundError) if the file is missing,
    so the UI layer can show a friendly 'run python db.py first' message
    instead of an opaque traceback.
    """
    path = DATA_DIR / name
    if not path.exists():
        raise DataNotGeneratedError(
            f"Parquet extract '{name}' not found in {DATA_DIR}. "
            "Run `python db.py` to generate the data files."
        )
    return pl.read_parquet(path)


# ── generation metadata ──────────────────────────────────────────────────────

def write_generation_metadata(row_counts: dict[str, int]) -> None:
    """Record when the Parquet files were generated and how big they are.

    The dashboard reads this to show a "data generated on" date, so a viewer
    can tell how fresh the committed Parquet is. Since the data files are
    committed to the repo (so Streamlit Cloud can read them without a DB),
    this is the only signal that they might be stale.
    """
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": SCHEMA,
        "row_counts": row_counts,
    }
    (DATA_DIR / "_generated.json").write_text(json.dumps(meta, indent=2))


# ── main: pull data and save to Parquet ──────────────────────────────────────

def main():
    print("Connecting to DB2 …")
    engine = make_engine()

    assert test_connection(engine), "DB2 connection test failed — check credentials."
    print("  Connection OK")

    DATA_DIR.mkdir(exist_ok=True)

    print("Extracting revenue data (this may take ~1–2 min for 35M-row TICKETS) …")
    revenue = extract_revenue(engine)
    revenue.write_parquet(DATA_DIR / "revenue.parquet")
    print(f"  revenue.parquet — {revenue.height:,} rows, {revenue.width} cols")
    print(revenue.head(3))

    print("Extracting fuel data …")
    fuel = extract_fuel(engine)
    fuel.write_parquet(DATA_DIR / "fuel.parquet")
    print(f"  fuel.parquet — {fuel.height:,} rows, {fuel.width} cols")

    print("Extracting airports …")
    airports = extract_airports(engine)
    airports.write_parquet(DATA_DIR / "airports.parquet")
    print(f"  airports.parquet — {airports.height:,} rows")

    write_generation_metadata(
        {
            "revenue": revenue.height,
            "fuel": fuel.height,
            "airports": airports.height,
        }
    )
    print("  _generated.json — freshness metadata written")

    print("\nAll Parquet files saved to data/. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
