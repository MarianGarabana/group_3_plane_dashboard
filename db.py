"""
db.py — DB2 connection, helper functions, and data extraction.

Workflow: run this file directly (python db.py) to pull all three
pre-aggregated datasets from the DB and save them as Parquet files
under data/. The Streamlit app then reads those files — no live DB
connection at dashboard runtime.
"""

import os
from pathlib import Path
from urllib.parse import quote_plus

import polars as pl
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── connection parameters ────────────────────────────────────────────────────

DB_HOST = os.getenv("DB_HOST", "52.211.123.34")
DB_PORT = int(os.getenv("DB_PORT", "25010"))
DB_NAME = os.getenv("DB_NAME", "ATTPLANE")
DB_USERNAME = os.getenv("DB_USERNAME", "attgrp3")
DB_PASSWORD = os.getenv("DB_PASSWORD", "bigdata")
SCHEMA = "ATTGRP3"

DATA_DIR = Path(__file__).parent / "data"


# ── engine factory ───────────────────────────────────────────────────────────

def make_engine():
    """Return a SQLAlchemy engine for DB2 via ibm_db_sa dialect."""
    user = quote_plus(DB_USERNAME)
    password = quote_plus(DB_PASSWORD)
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


# ── exploration helpers ──────────────────────────────────────────────────────

def preview_table(schema: str, table: str, limit: int = 10, engine=None) -> pl.DataFrame:
    """Return the first `limit` rows of a table as a Polars DataFrame.
    Useful during exploration — never used by the dashboard itself."""
    if engine is None:
        engine = make_engine()
    query = (
        f"SELECT * FROM {qualified_table(schema, table)} "
        f"FETCH FIRST {int(limit)} ROWS ONLY"
    )
    return _read_sql(query, engine)


def count_rows(schema: str, table: str, engine=None) -> int:
    """Return the exact row count of a table as a Python int.
    Uses .item() so the caller gets a scalar, not a DataFrame."""
    if engine is None:
        engine = make_engine()
    query = f"SELECT COUNT(*) AS n_rows FROM {qualified_table(schema, table)}"
    return _read_sql(query, engine).item(0, "n_rows")


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
    per (route, year, class) combination — roughly 59 routes × 25 years
    × 3 classes ≈ 4 000 rows maximum.
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
    """Query 2: fuel consumption grouped by route + aircraft model.

    total_fuel_gallons = flights operated × fuel burn rate × flight hours.
    Multiplying by $3/gallon in analysis.py gives estimated fuel cost.
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

    print("\nAll Parquet files saved to data/. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
