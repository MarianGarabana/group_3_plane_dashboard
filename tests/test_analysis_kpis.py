"""
test_analysis_kpis.py — Additional unit tests for analysis.py.

These cover behaviour not exercised by test_analysis.py or
test_analysis_extended.py, focusing on:
  - safe_div column-wise edge cases (negative / mixed denominators)
  - total_fuel_cost and fuel_cost_by_route on empty / single-route frames
  - filter_fuel_by_year boundary inclusivity
  - apply_filters interaction (class + continent together, no-op window)
  - compute_kpis margin/tax arithmetic and fuel-price sensitivity
  - enrich_revenue_with_airports preserving row count and adding dest columns

All expected values are computed by hand from the conftest.py fixtures so the
suite stays fast and needs no database.
"""

import math

import polars as pl
import pytest

from analysis import (
    DEFAULT_FUEL_PRICE_USD,
    apply_filters,
    compute_kpis,
    enrich_revenue_with_airports,
    filter_fuel_by_year,
    fuel_cost_by_route,
    safe_div,
    total_fuel_cost,
)


# ── safe_div edge cases ───────────────────────────────────────────────────────

def test_safe_div_handles_negative_denominator():
    """A negative (non-zero) denominator should still divide normally."""
    df = pl.DataFrame({"n": [10.0, -8.0], "d": [-2.0, 4.0]})
    out = df.with_columns(safe_div(pl.col("n"), pl.col("d")).alias("r"))
    assert out["r"].to_list() == [-5.0, -2.0]


def test_safe_div_zero_numerator_is_zero_not_null():
    """0 / non-zero is a valid 0.0, only a zero denominator yields null."""
    df = pl.DataFrame({"n": [0.0, 5.0], "d": [3.0, 0.0]})
    out = df.with_columns(safe_div(pl.col("n"), pl.col("d")).alias("r"))
    assert out["r"].to_list() == [0.0, None]


# ── total_fuel_cost / fuel_cost_by_route on edge frames ───────────────────────

def test_total_fuel_cost_zero_gallons_is_zero(fuel):
    """If every row has zero gallons, total fuel cost is exactly 0.0.

    Built by zeroing an existing frame's gallons rather than constructing an
    empty DataFrame, which keeps the schema real and the arithmetic exact.
    """
    zeroed = fuel.with_columns(pl.lit(0.0).alias("total_fuel_gallons"))
    assert total_fuel_cost(zeroed, 3.0) == 0.0


def test_total_fuel_cost_empty_after_filter_is_zero(fuel):
    """Summing gallons over a frame filtered down to no rows must be 0.0.

    Filtering a populated frame to empty exercises the empty-aggregate path
    without ever constructing an empty DataFrame directly.
    """
    none_left = fuel.filter(pl.col("route_code") == "DOES_NOT_EXIST")
    assert none_left.height == 0
    assert total_fuel_cost(none_left, 3.0) == 0.0


def test_fuel_cost_by_route_single_route(fuel):
    """Restricting to one route returns exactly one aggregated row."""
    only_r2 = fuel.filter(pl.col("route_code") == "R2")
    out = fuel_cost_by_route(only_r2, 3.0).collect()
    assert out.height == 1
    assert out["route_code"].to_list() == ["R2"]
    assert out["fuel_cost"].to_list() == pytest.approx([600.0])  # 200 gal * 3


# ── filter_fuel_by_year boundary behaviour ────────────────────────────────────

def test_filter_fuel_by_year_is_inclusive(fuel):
    """is_between is inclusive: a 2020–2021 window keeps every fuel row."""
    out = filter_fuel_by_year(fuel, 2020, 2021)
    assert out.height == fuel.height
    assert out["total_fuel_gallons"].sum() == pytest.approx(500.0)


def test_filter_fuel_by_year_excludes_outside_window(fuel):
    """A window with no matching years yields an empty frame, not an error."""
    out = filter_fuel_by_year(fuel, 2030, 2031)
    assert out.height == 0


# ── apply_filters combinations ────────────────────────────────────────────────

def test_apply_filters_class_and_continent_together(revenue_enriched):
    """Business class AND Asia origin both point to R2 — intersection holds."""
    out = apply_filters(revenue_enriched, 2010, 2030, ["B"], ["Asia"], 0)
    assert set(out["route_code"].unique().to_list()) == {"R2"}


def test_apply_filters_wide_window_keeps_all_rows(revenue_enriched):
    """A window spanning all data with no other filters is a no-op."""
    out = apply_filters(revenue_enriched, 2000, 2100, [], [], 0)
    assert out.height == revenue_enriched.height


# ── compute_kpis arithmetic ───────────────────────────────────────────────────

def test_compute_kpis_margin_and_tax_match_manual(revenue_enriched, fuel):
    """Cross-check the headline KPI arithmetic against hand figures."""
    kpis = compute_kpis(revenue_enriched, fuel, fuel_price=3.0)
    # net 4200, fuel 500 gal * 3 = 1500 → margin = (4200-1500)/4200*100
    assert kpis["margin_pct"] == pytest.approx((4200 - 1500) / 4200 * 100)
    # taxes 900, gross 5100 → tax burden = 900/5100*100
    assert kpis["tax_burden_pct"] == pytest.approx(900 / 5100 * 100)
    assert kpis["total_tickets"] == 250


def test_compute_kpis_higher_fuel_price_lowers_margin(revenue_enriched, fuel):
    """Margin must fall as fuel gets more expensive (monotonic sensitivity)."""
    cheap = compute_kpis(revenue_enriched, fuel, fuel_price=2.0)
    dear = compute_kpis(revenue_enriched, fuel, fuel_price=5.0)
    assert dear["total_fuel_cost"] > cheap["total_fuel_cost"]
    assert dear["margin_pct"] < cheap["margin_pct"]


# ── enrich_revenue_with_airports ──────────────────────────────────────────────

def test_enrich_preserves_row_count(revenue_raw, airports):
    """The double left-join must not drop or duplicate revenue rows."""
    out = enrich_revenue_with_airports(revenue_raw, airports)
    assert out.height == revenue_raw.height


def test_enrich_adds_destination_continent(revenue_raw, airports):
    """Enrichment should attach the destination continent, not just origin."""
    out = enrich_revenue_with_airports(revenue_raw, airports).sort("route_code")
    by_route = {r["route_code"]: r["dest_continent"] for r in out.to_dicts()}
    assert by_route["R1"] == "Europe"  # B1 is Europe
    assert by_route["R2"] == "Asia"    # B2 is Asia
