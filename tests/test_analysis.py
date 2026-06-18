"""Unit tests for analysis.py.

Expected values are computed by hand from the synthetic fixtures in
conftest.py. The tests marked "regression" guard the two correctness fixes
from the audit: fuel must respond to the year filter, and the margin trend
must not be a flat line.
"""

import math

import polars as pl
import pytest

from analysis import (
    DEFAULT_FUEL_PRICE_USD,
    EMPTY_KPIS,
    apply_filters,
    compute_kpis,
    enrich_revenue_with_airports,
    filter_fuel_by_year,
    fleet_efficiency,
    fuel_cost_by_route,
    margin_trend,
    route_profitability,
    safe_div,
    tax_drain_by_route,
    total_fuel_cost,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def test_safe_div_returns_null_on_zero_or_null_denominator():
    df = pl.DataFrame({"n": [10.0, 10.0, 10.0], "d": [2.0, 0.0, None]})
    out = df.with_columns(safe_div(pl.col("n"), pl.col("d")).alias("r"))
    assert out["r"].to_list() == [5.0, None, None]


def test_total_fuel_cost_scales_with_price(fuel):
    # total gallons = 100 + 100 + 100 + 200 = 500
    assert total_fuel_cost(fuel, 3.0) == pytest.approx(1500.0)
    assert total_fuel_cost(fuel, 4.0) == pytest.approx(2000.0)


def test_fuel_cost_by_route_aggregates_per_route(fuel):
    out = fuel_cost_by_route(fuel, 3.0).collect().sort("route_code")
    # R1 gallons = 100 + 100 + 100 = 300 → 900; R2 = 200 → 600
    assert out["route_code"].to_list() == ["R1", "R2"]
    assert out["fuel_cost"].to_list() == pytest.approx([900.0, 600.0])


def test_filter_fuel_by_year_window(fuel):
    only_2021 = filter_fuel_by_year(fuel, 2021, 2021)
    assert only_2021.height == 1
    assert only_2021["total_fuel_gallons"].sum() == pytest.approx(100.0)


# ── enrichment ─────────────────────────────────────────────────────────────────

def test_enrich_adds_origin_continent(revenue_raw, airports):
    out = enrich_revenue_with_airports(revenue_raw, airports).sort("route_code")
    by_route = {r["route_code"]: r["origin_continent"] for r in out.to_dicts()}
    assert by_route["R1"] == "Europe"
    assert by_route["R2"] == "Asia"


# ── filters ────────────────────────────────────────────────────────────────────

def test_apply_filters_year_window(revenue_enriched):
    out = apply_filters(revenue_enriched, 2020, 2020, [], [], 0)
    assert set(out["yr"].unique().to_list()) == {2020}


def test_apply_filters_cabin_class(revenue_enriched):
    out = apply_filters(revenue_enriched, 2010, 2030, ["B"], [], 0)
    assert set(out["class"].unique().to_list()) == {"B"}
    assert set(out["route_code"].unique().to_list()) == {"R2"}


def test_apply_filters_continent(revenue_enriched):
    out = apply_filters(revenue_enriched, 2010, 2030, [], ["Europe"], 0)
    assert set(out["route_code"].unique().to_list()) == {"R1"}


def test_apply_filters_min_tickets(revenue_enriched):
    # R1 has 200 tickets total, R2 has 50; threshold 100 keeps only R1.
    out = apply_filters(revenue_enriched, 2010, 2030, [], [], 100)
    assert set(out["route_code"].unique().to_list()) == {"R1"}


# ── KPIs ───────────────────────────────────────────────────────────────────────

def test_compute_kpis_full_window(revenue_enriched, fuel):
    kpis = compute_kpis(revenue_enriched, fuel, fuel_price=3.0)
    assert kpis["is_empty"] is False
    assert kpis["total_net_revenue"] == pytest.approx(4200.0)  # 1000+1200+2000
    assert kpis["total_fuel_cost"] == pytest.approx(1500.0)    # 500 gal * 3
    assert kpis["margin_pct"] == pytest.approx((4200 - 1500) / 4200 * 100)
    assert kpis["tax_burden_pct"] == pytest.approx(900 / 5100 * 100)
    # R1 profit = 2200 - 900 = 1300; R2 profit = 2000 - 600 = 1400 → R2 best
    assert kpis["best_route"] == "A2 → B2"
    assert kpis["worst_route"] == "A1 → B1"


def test_compute_kpis_empty_returns_sentinel(revenue_enriched, fuel):
    empty = revenue_enriched.clear()
    kpis = compute_kpis(empty, fuel, fuel_price=3.0)
    assert kpis == EMPTY_KPIS
    assert kpis["is_empty"] is True


def test_compute_kpis_default_fuel_price(revenue_enriched, fuel):
    explicit = compute_kpis(revenue_enriched, fuel, fuel_price=DEFAULT_FUEL_PRICE_USD)
    implicit = compute_kpis(revenue_enriched, fuel)
    assert implicit["total_fuel_cost"] == explicit["total_fuel_cost"]


def test_compute_kpis_fuel_responds_to_year_filter(revenue_enriched, fuel):
    """Regression: fuel cost must change with the year window, not stay frozen
    at all-history scale (the original bug)."""
    rev_2020 = apply_filters(revenue_enriched, 2020, 2020, [], [], 0)
    fuel_2020 = filter_fuel_by_year(fuel, 2020, 2020)
    kpis_2020 = compute_kpis(rev_2020, fuel_2020, fuel_price=3.0)

    kpis_all = compute_kpis(revenue_enriched, fuel, fuel_price=3.0)

    # 2020 fuel = 400 gal * 3 = 1200; all years = 500 gal * 3 = 1500
    assert kpis_2020["total_fuel_cost"] == pytest.approx(1200.0)
    assert kpis_2020["total_fuel_cost"] != kpis_all["total_fuel_cost"]


# ── route profitability ──────────────────────────────────────────────────────

def test_route_profitability_est_profit(revenue_enriched, fuel):
    out = route_profitability(revenue_enriched, fuel, fuel_price=3.0)
    by_route = {r["route_code"]: r for r in out.to_dicts()}
    # R1: net 2200, fuel 900 → 1300; R2: net 2000, fuel 600 → 1400
    assert by_route["R1"]["est_profit"] == pytest.approx(1300.0)
    assert by_route["R2"]["est_profit"] == pytest.approx(1400.0)
    assert by_route["R1"]["route_label"] == "A1 → B1"


# ── tax drain ──────────────────────────────────────────────────────────────────

def test_tax_drain_pct(revenue_enriched):
    out = tax_drain_by_route(revenue_enriched)
    by_route = {r["route_code"]: r["tax_pct"] for r in out.to_dicts()}
    # R1: taxes 400 / gross 2600; R2: taxes 500 / gross 2500
    assert by_route["R1"] == pytest.approx(400 / 2600 * 100)
    assert by_route["R2"] == pytest.approx(20.0)


# ── fleet efficiency ──────────────────────────────────────────────────────────

def test_fleet_efficiency_conserves_revenue(revenue_enriched, fuel):
    """Regression: revenue allocated across models must sum to total route
    revenue. The old code double-counted revenue for multi-model routes and
    would multiply it per model-year once fuel carried a year dimension."""
    out = fleet_efficiency(revenue_enriched, fuel, fuel_price=3.0)
    total_alloc = out["net_revenue"].sum()
    # Both R1 (2200) and R2 (2000) have fuel rows, so all 4200 is allocated.
    assert total_alloc == pytest.approx(4200.0)
    # No model should have an infinite or NaN rev/gallon.
    for v in out["rev_per_gallon"].to_list():
        assert v is None or math.isfinite(v)


def test_fleet_efficiency_respects_continent_filter(revenue_enriched, fuel):
    rev_europe = apply_filters(revenue_enriched, 2010, 2030, [], ["Europe"], 0)
    out = fleet_efficiency(rev_europe, fuel, fuel_price=3.0)
    # Only R1 (Europe) revenue remains → only its models (M1, M2) appear.
    assert set(out["model"].to_list()) <= {"M1", "M2"}
    assert out["net_revenue"].sum() == pytest.approx(2200.0)


# ── margin trend ──────────────────────────────────────────────────────────────

def test_margin_trend_is_not_flat(revenue_enriched, fuel):
    """Regression: the old proportional allocation made margin identical every
    year (the net_revenue terms cancelled). Real per-year fuel must produce
    different margins across years."""
    out = margin_trend(revenue_enriched, fuel, fuel_price=3.0).sort("yr")
    margins = out["margin_pct"].to_list()
    assert len(set(round(m, 4) for m in margins)) > 1, "margin should vary by year"


def test_margin_trend_values(revenue_enriched, fuel):
    out = margin_trend(revenue_enriched, fuel, fuel_price=3.0).sort("yr")
    by_year = {r["yr"]: r for r in out.to_dicts()}
    # 2020: net 3000, fuel 400 gal * 3 = 1200 → margin (3000-1200)/3000 = 60%
    assert by_year[2020]["est_fuel_cost"] == pytest.approx(1200.0)
    assert by_year[2020]["margin_pct"] == pytest.approx(60.0)
    # 2021: net 1200, fuel 100 gal * 3 = 300 → margin (1200-300)/1200 = 75%
    assert by_year[2021]["est_fuel_cost"] == pytest.approx(300.0)
    assert by_year[2021]["margin_pct"] == pytest.approx(75.0)
