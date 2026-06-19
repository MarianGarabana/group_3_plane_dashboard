"""
test_routes_metrics.py — Additional unit tests for analysis.py.

Focused on route-level and trend behaviour, using only small populated
DataFrames built by hand. No empty-frame construction anywhere, so the suite
is portable across Polars builds.

Covered:
  - route_label_expr on a realistic multi-row frame
  - fuel_cost_by_route aggregation and price scaling
  - route_profitability ordering, fuel fill, and est_profit arithmetic
  - tax_drain_by_route percentage maths and ordering
  - fleet_efficiency revenue conservation and rev/gallon ordering
  - margin_trend per-year fuel cost, ordering, and sub-100% margins
"""

import math

import polars as pl
import pytest

from analysis import (
    DEFAULT_FUEL_PRICE_USD,
    fuel_cost_by_route,
    margin_trend,
    route_label_expr,
    route_profitability,
    tax_drain_by_route,
    fleet_efficiency,
)


# ── local fixtures (always populated) ─────────────────────────────────────────

def _revenue() -> pl.DataFrame:
    """Two routes, two years, with continent already attached."""
    return pl.DataFrame(
        {
            "route_code": ["R1", "R1", "R2"],
            "origin": ["MAD", "MAD", "CDG"],
            "destination": ["LHR", "LHR", "JFK"],
            "origin_continent": ["Europe", "Europe", "Europe"],
            "yr": [2022, 2023, 2022],
            "net_revenue": [600_000.0, 700_000.0, 300_000.0],
            "total_taxes": [60_000.0, 70_000.0, 90_000.0],
            "gross_revenue": [660_000.0, 770_000.0, 390_000.0],
            "ticket_count": [4000, 4500, 1500],
        }
    )


def _fuel() -> pl.DataFrame:
    """R1 flown by two models across two years; R2 by one model."""
    return pl.DataFrame(
        {
            "route_code": ["R1", "R1", "R1", "R2"],
            "model": ["B777", "A350", "B777", "A320"],
            "yr": [2022, 2022, 2023, 2022],
            "flights_operated": [20, 10, 22, 15],
            "total_fuel_gallons": [60_000.0, 40_000.0, 66_000.0, 30_000.0],
        }
    )


# ── route_label_expr ──────────────────────────────────────────────────────────

def test_route_label_expr_builds_arrow_label():
    df = pl.DataFrame({"origin": ["MAD", "CDG"], "destination": ["LHR", "JFK"]})
    out = df.with_columns(route_label_expr())
    assert out["route_label"].to_list() == ["MAD → LHR", "CDG → JFK"]


# ── fuel_cost_by_route ────────────────────────────────────────────────────────

def test_fuel_cost_by_route_sums_gallons_times_price():
    out = fuel_cost_by_route(_fuel(), 3.0).collect().sort("route_code")
    # R1 gallons = 60k + 40k + 66k = 166k → 498k; R2 = 30k → 90k
    assert out["route_code"].to_list() == ["R1", "R2"]
    assert out["fuel_cost"].to_list() == pytest.approx([498_000.0, 90_000.0])


def test_fuel_cost_by_route_scales_linearly_with_price():
    at3 = fuel_cost_by_route(_fuel(), 3.0).collect().sort("route_code")["fuel_cost"]
    at6 = fuel_cost_by_route(_fuel(), 6.0).collect().sort("route_code")["fuel_cost"]
    assert at6.to_list() == pytest.approx([v * 2 for v in at3.to_list()])


# ── route_profitability ───────────────────────────────────────────────────────

def test_route_profitability_orders_by_profit_desc():
    out = route_profitability(_revenue(), _fuel(), fuel_price=3.0)
    # R1: net 1.3M - fuel 498k = 802k; R2: net 300k - fuel 90k = 210k → R1 first
    assert out["route_code"][0] == "R1"


def test_route_profitability_est_profit_is_net_minus_fuel():
    out = route_profitability(_revenue(), _fuel(), fuel_price=3.0)
    for row in out.iter_rows(named=True):
        assert row["est_profit"] == pytest.approx(row["net_revenue"] - row["fuel_cost"])


def test_route_profitability_uses_default_fuel_price():
    explicit = route_profitability(_revenue(), _fuel(), fuel_price=DEFAULT_FUEL_PRICE_USD)
    implicit = route_profitability(_revenue(), _fuel())
    assert implicit["fuel_cost"].to_list() == pytest.approx(explicit["fuel_cost"].to_list())


# ── tax_drain_by_route ────────────────────────────────────────────────────────

def test_tax_drain_pct_matches_manual():
    out = tax_drain_by_route(_revenue())
    by_route = {r["route_code"]: r["tax_pct"] for r in out.to_dicts()}
    # R1: taxes 130k / gross 1.43M; R2: taxes 90k / gross 390k
    assert by_route["R1"] == pytest.approx(130_000 / 1_430_000 * 100)
    assert by_route["R2"] == pytest.approx(90_000 / 390_000 * 100)


def test_tax_drain_orders_by_tax_pct_desc():
    out = tax_drain_by_route(_revenue())
    # R2 (~23%) is more tax-burdened than R1 (~9%), so R2 leads.
    assert out["route_code"][0] == "R2"


# ── fleet_efficiency ──────────────────────────────────────────────────────────

def test_fleet_efficiency_conserves_total_revenue():
    out = fleet_efficiency(_revenue(), _fuel(), fuel_price=3.0)
    # Both routes have fuel rows, so all net revenue (1.6M) is allocated.
    assert out["net_revenue"].sum() == pytest.approx(1_600_000.0)


def test_fleet_efficiency_rev_per_gallon_is_finite_and_sorted():
    out = fleet_efficiency(_revenue(), _fuel(), fuel_price=3.0)
    vals = out["rev_per_gallon"].to_list()
    assert vals == sorted(vals, reverse=True)
    for v in vals:
        assert v is None or math.isfinite(v)


# ── margin_trend ──────────────────────────────────────────────────────────────

def test_margin_trend_per_year_fuel_cost():
    out = margin_trend(_revenue(), _fuel(), fuel_price=3.0).sort("yr")
    by_year = {r["yr"]: r for r in out.to_dicts()}
    # 2022 gallons = 60k + 40k + 30k = 130k → 390k; 2023 = 66k → 198k
    assert by_year[2022]["est_fuel_cost"] == pytest.approx(390_000.0)
    assert by_year[2023]["est_fuel_cost"] == pytest.approx(198_000.0)


def test_margin_trend_sorted_and_below_100():
    out = margin_trend(_revenue(), _fuel(), fuel_price=3.0)
    years = out["yr"].to_list()
    assert years == sorted(years)
    assert (out["margin_pct"] < 100).all()
