"""
test_analysis_extended.py
Additional edge-case and regression tests for analysis.py.
Covers gaps not addressed in test_analysis.py:
  - route_label_expr formatting
  - route_profitability sorting and fill_null behaviour
  - tax_drain_by_route sorting and null-safe tax_pct
  - fleet_efficiency gallon-share allocation
  - margin_trend fill_null for years with no fuel data
"""

import polars as pl
import pytest
from analysis import (
    route_label_expr,
    route_profitability,
    tax_drain_by_route,
    fleet_efficiency,
    margin_trend,
)


# ── shared helpers ────────────────────────────────────────────────────────────

def make_revenue(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns([
        pl.col("net_revenue").cast(pl.Float64),
        pl.col("gross_revenue").cast(pl.Float64),
        pl.col("total_taxes").cast(pl.Float64),
        pl.col("ticket_count").cast(pl.Int64),
    ])


def make_fuel(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({
            "route_code": pl.Series([], dtype=pl.Utf8),
            "model": pl.Series([], dtype=pl.Utf8),
            "total_fuel_gallons": pl.Series([], dtype=pl.Float64),
            "flights_operated": pl.Series([], dtype=pl.Int64),
            "yr": pl.Series([], dtype=pl.Int32),
        })
    return pl.DataFrame(rows).with_columns([
        pl.col("total_fuel_gallons").cast(pl.Float64),
        pl.col("flights_operated").cast(pl.Int64),
        pl.col("yr").cast(pl.Int32),
    ])


def make_base_revenue():
    return make_revenue([
        {"route_code": "A", "origin": "NAP", "destination": "LAS",
         "origin_continent": "EU", "net_revenue": 800_000.0,
         "gross_revenue": 850_000.0, "total_taxes": 50_000.0, "ticket_count": 5000, "yr": 2023},
        {"route_code": "B", "origin": "CDG", "destination": "LIL",
         "origin_continent": "EU", "net_revenue": 20_000.0,
         "gross_revenue": 25_000.0, "total_taxes": 5_000.0, "ticket_count": 200, "yr": 2023},
    ])


def make_base_fuel():
    return make_fuel([
        {"route_code": "A", "model": "B747", "total_fuel_gallons": 100_000.0,
         "flights_operated": 50, "yr": 2023},
        {"route_code": "B", "model": "A319", "total_fuel_gallons": 5_000.0,
         "flights_operated": 20, "yr": 2023},
    ])


# ── route_label_expr ──────────────────────────────────────────────────────────

def test_route_label_expr_format():
    """route_label should be 'ORIGIN → DESTINATION'."""
    df = pl.DataFrame({"origin": ["MAD"], "destination": ["LHR"]})
    result = df.with_columns(route_label_expr())
    assert result["route_label"][0] == "MAD → LHR"


def test_route_label_expr_multiple_rows():
    """route_label works correctly across multiple rows."""
    df = pl.DataFrame({
        "origin": ["MAD", "CDG", "NAP"],
        "destination": ["LHR", "LIL", "LAS"],
    })
    result = df.with_columns(route_label_expr())
    assert result["route_label"].to_list() == ["MAD → LHR", "CDG → LIL", "NAP → LAS"]


# ── route_profitability ───────────────────────────────────────────────────────

def test_route_profitability_sorted_descending():
    """Most profitable route should appear first."""
    result = route_profitability(make_base_revenue(), make_base_fuel(), fuel_price=3.0)
    assert result["route_code"][0] == "A"


def test_route_profitability_fill_null_fuel():
    """A route with no fuel data should get fuel_cost=0, not null."""
    result = route_profitability(make_base_revenue(), make_fuel([]), fuel_price=3.0)
    assert result["fuel_cost"].null_count() == 0
    assert (result["fuel_cost"] == 0.0).all()


def test_route_profitability_est_profit_calculation():
    """est_profit should equal net_revenue minus fuel_cost."""
    result = route_profitability(make_base_revenue(), make_base_fuel(), fuel_price=3.0)
    for row in result.iter_rows(named=True):
        expected = row["net_revenue"] - row["fuel_cost"]
        assert abs(row["est_profit"] - expected) < 0.01


# ── tax_drain_by_route ────────────────────────────────────────────────────────

def test_tax_drain_sorted_by_tax_pct_descending():
    """Highest tax burden route should appear first."""
    rev = make_revenue([
        {"route_code": "LHR-MAN", "origin": "LHR", "destination": "MAN",
         "origin_continent": "EU", "net_revenue": 580_000.0,
         "gross_revenue": 1_000_000.0, "total_taxes": 420_000.0,
         "ticket_count": 3000, "yr": 2023},
        {"route_code": "MAD-BCN", "origin": "MAD", "destination": "BCN",
         "origin_continent": "EU", "net_revenue": 90_000.0,
         "gross_revenue": 100_000.0, "total_taxes": 10_000.0,
         "ticket_count": 1000, "yr": 2023},
    ])
    result = tax_drain_by_route(rev)
    assert result["route_code"][0] == "LHR-MAN"


def test_tax_drain_zero_gross_revenue_is_null_not_crash():
    """A route with zero gross_revenue should produce null tax_pct, not an error."""
    rev = make_revenue([
        {"route_code": "X", "origin": "A", "destination": "B",
         "origin_continent": "EU", "net_revenue": 0.0,
         "gross_revenue": 0.0, "total_taxes": 0.0,
         "ticket_count": 0, "yr": 2023},
    ])
    result = tax_drain_by_route(rev)
    assert result["tax_pct"][0] is None


# ── fleet_efficiency ──────────────────────────────────────────────────────────

def test_fleet_efficiency_gallon_share_allocation():
    """When two models share a route, revenue should be split by gallon share."""
    rev = make_revenue([
        {"route_code": "R1", "origin": "A", "destination": "B",
         "origin_continent": "EU", "net_revenue": 100_000.0,
         "gross_revenue": 110_000.0, "total_taxes": 10_000.0,
         "ticket_count": 500, "yr": 2023},
    ])
    fuel = make_fuel([
        {"route_code": "R1", "model": "B747", "total_fuel_gallons": 75_000.0,
         "flights_operated": 30, "yr": 2023},
        {"route_code": "R1", "model": "A320", "total_fuel_gallons": 25_000.0,
         "flights_operated": 10, "yr": 2023},
    ])
    result = fleet_efficiency(rev, fuel, fuel_price=3.0)
    b747 = result.filter(pl.col("model") == "B747")["net_revenue"][0]
    a320 = result.filter(pl.col("model") == "A320")["net_revenue"][0]
    assert abs(b747 - 75_000.0) < 1.0
    assert abs(a320 - 25_000.0) < 1.0


def test_fleet_efficiency_sorted_by_rev_per_gallon():
    """Most efficient aircraft model should appear first."""
    result = fleet_efficiency(make_base_revenue(), make_base_fuel(), fuel_price=3.0)
    rev_per_gal = result["rev_per_gallon"].to_list()
    assert rev_per_gal == sorted(rev_per_gal, reverse=True)


# ── margin_trend ──────────────────────────────────────────────────────────────

def test_margin_trend_missing_fuel_year_fills_zero():
    """A year present in revenue but not in fuel should have est_fuel_cost=0."""
    rev = make_revenue([
        {"route_code": "A", "origin": "X", "destination": "Y",
         "origin_continent": "EU", "net_revenue": 100_000.0,
         "gross_revenue": 110_000.0, "total_taxes": 10_000.0,
         "ticket_count": 500, "yr": 2020},
    ])
    result = margin_trend(rev, make_fuel([]), fuel_price=3.0)
    assert result["est_fuel_cost"][0] == 0.0


def test_margin_trend_sorted_by_year():
    """Results should be in ascending year order."""
    rev = make_revenue([
        {"route_code": "A", "origin": "X", "destination": "Y",
         "origin_continent": "EU", "net_revenue": 50_000.0,
         "gross_revenue": 55_000.0, "total_taxes": 5_000.0,
         "ticket_count": 300, "yr": yr}
        for yr in [2022, 2020, 2021]
    ])
    fuel = make_fuel([
        {"route_code": "A", "model": "B747", "total_fuel_gallons": 10_000.0,
         "flights_operated": 10, "yr": yr}
        for yr in [2020, 2021, 2022]
    ])
    result = margin_trend(rev, fuel, fuel_price=3.0)
    years = result["yr"].to_list()
    assert years == sorted(years)


def test_margin_trend_margin_below_100_pct():
    """Margin should be less than 100% whenever there is any fuel cost."""
    result = margin_trend(make_base_revenue(), make_base_fuel(), fuel_price=3.0)
    assert (result["margin_pct"] < 100).all()