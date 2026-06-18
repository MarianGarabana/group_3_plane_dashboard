"""
analysis.py — Polars transformations that turn the raw Parquet extracts
into dashboard-ready metrics.

All functions accept the three base DataFrames (revenue, fuel, airports)
and return new DataFrames. No DB calls here — pure Polars.

Fuel price is a parameter (defaulting to DEFAULT_FUEL_PRICE_USD) rather than
a fixed constant, so the dashboard can run a sensitivity analysis without
re-querying the DB. The fuel extract keeps the raw burn components, so any
price recalculates instantly.

Both the revenue and fuel extracts carry a `yr` column. Callers are expected
to filter both to the same year window before combining them, so a
year-filtered view never pairs filtered revenue against all-history fuel.
"""

import polars as pl

DEFAULT_FUEL_PRICE_USD = 3.0  # $/gallon — default cost assumption

# Returned by compute_kpis when the filtered frame is empty, so the UI can
# render zeroed cards instead of crashing on an empty aggregate.
EMPTY_KPIS = {
    "total_net_revenue": 0.0,
    "total_fuel_cost": 0.0,
    "margin_pct": 0.0,
    "tax_burden_pct": 0.0,
    "total_tickets": 0,
    "best_route": "—",
    "worst_route": "—",
    "is_empty": True,
}


# ── shared expression / aggregate helpers ────────────────────────────────────

def route_label_expr() -> pl.Expr:
    """`origin → destination` as a Polars expression, named route_label.
    Single definition so every chart labels routes identically."""
    return (pl.col("origin") + " → " + pl.col("destination")).alias("route_label")


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Division that yields null (not inf or NaN) when the denominator is
    zero or null. Keeps a single empty/zero-volume route from poisoning a
    chart with infinities."""
    return (
        pl.when(denominator.fill_null(0) == 0)
        .then(None)
        .otherwise(numerator / denominator)
    )


def fuel_cost_by_route(fuel: pl.DataFrame, fuel_price: float) -> pl.LazyFrame:
    """Estimated fuel cost per route = total gallons × price.
    Returns a LazyFrame so callers can keep their chain lazy."""
    return (
        fuel.lazy()
        .group_by("route_code")
        .agg((pl.col("total_fuel_gallons").sum() * fuel_price).alias("fuel_cost"))
    )


def total_fuel_cost(fuel: pl.DataFrame, fuel_price: float) -> float:
    """Scalar total fuel cost across the whole (already filtered) fuel frame."""
    return fuel["total_fuel_gallons"].sum() * fuel_price


def filter_fuel_by_year(fuel: pl.DataFrame, year_min: int, year_max: int) -> pl.DataFrame:
    """Restrict fuel to a year window, mirroring the revenue year filter.
    This is what keeps fuel-based metrics aligned with the selected years."""
    return fuel.filter(pl.col("yr").is_between(year_min, year_max))


# ── filter helpers ───────────────────────────────────────────────────────────

def apply_filters(
    revenue: pl.DataFrame,
    year_min: int,
    year_max: int,
    cabin_classes: list[str],
    origin_continents: list[str],
    min_tickets: int,
) -> pl.DataFrame:
    """Return a filtered slice of the revenue DataFrame.

    Applying filters here (not inside each chart function) means every
    chart section works from the same filtered base — consistent counts.
    """
    df = (
        revenue.lazy()
        .filter(pl.col("yr").is_between(year_min, year_max))
        .collect()
    )

    if cabin_classes:  # empty list means "all"
        df = df.filter(pl.col("class").is_in(cabin_classes))

    if origin_continents:
        df = df.filter(pl.col("origin_continent").is_in(origin_continents))

    if min_tickets > 0:
        # Remove routes that have fewer total tickets than the threshold
        # across the whole filtered window — avoids noisy single-flight routes.
        route_totals = (
            df.lazy()
            .group_by("route_code")
            .agg(pl.col("ticket_count").sum().alias("total_tickets"))
            .collect()
        )
        keep = route_totals.filter(pl.col("total_tickets") >= min_tickets)["route_code"]
        df = df.filter(pl.col("route_code").is_in(keep.to_list()))

    return df


# ── revenue enrichment ───────────────────────────────────────────────────────

def enrich_revenue_with_airports(
    revenue: pl.DataFrame,
    airports: pl.DataFrame,
) -> pl.DataFrame:
    """Join continent and city info from AIRPORTS onto the revenue rows.

    We join twice — once for ORIGIN, once for DESTINATION — because a
    route has two endpoints and we need the continent for chart colouring.
    """
    origin_slim = airports.select(
        pl.col("iata_code"),
        pl.col("continent").alias("origin_continent"),
        pl.col("city").alias("origin_city"),
    )
    dest_slim = airports.select(
        pl.col("iata_code"),
        pl.col("continent").alias("dest_continent"),
        pl.col("city").alias("dest_city"),
    )

    return (
        revenue.lazy()
        .join(origin_slim.lazy(), left_on="origin", right_on="iata_code", how="left")
        .join(dest_slim.lazy(), left_on="destination", right_on="iata_code", how="left")
        .collect()
    )


# ── KPI aggregates ───────────────────────────────────────────────────────────

def compute_kpis(
    revenue_filtered: pl.DataFrame,
    fuel: pl.DataFrame,
    fuel_price: float = DEFAULT_FUEL_PRICE_USD,
) -> dict:
    """Return a dict of scalar KPIs for the top cards.

    `fuel` is expected to be already filtered to the same year window as
    `revenue_filtered`. Fuel cost is further restricted to the routes present
    in the filtered revenue, so the headline margin responds to the continent
    and volume filters too (fuel has no class dimension, so class filtering
    cannot narrow fuel further — a documented limitation).

    Margin = (net_revenue - fuel_cost) / net_revenue.
    Tax burden = total_taxes / gross_revenue.
    """
    if revenue_filtered.is_empty():
        return dict(EMPTY_KPIS)

    total_net_revenue = revenue_filtered["net_revenue"].sum()
    total_taxes = revenue_filtered["total_taxes"].sum()
    total_gross = revenue_filtered["gross_revenue"].sum()
    total_tickets = revenue_filtered["ticket_count"].sum()

    routes_in_view = revenue_filtered["route_code"].unique().to_list()
    fuel_in_view = fuel.filter(pl.col("route_code").is_in(routes_in_view))
    total_fuel_cost_usd = total_fuel_cost(fuel_in_view, fuel_price)

    margin_pct = (
        (total_net_revenue - total_fuel_cost_usd) / total_net_revenue * 100
        if total_net_revenue > 0
        else 0.0
    )
    tax_burden_pct = (
        total_taxes / total_gross * 100
        if total_gross > 0
        else 0.0
    )

    # Most / least profitable route by (net_revenue - fuel cost), both
    # measured over the same filtered window. Left join so routes with no
    # fuel rows still appear (fuel cost treated as 0).
    route_pnl = (
        revenue_filtered.lazy()
        .group_by("route_code", "origin", "destination")
        .agg(pl.col("net_revenue").sum())
        .join(fuel_cost_by_route(fuel, fuel_price), on="route_code", how="left")
        .with_columns(
            (pl.col("net_revenue") - pl.col("fuel_cost").fill_null(0)).alias("est_profit")
        )
        .collect()
    )

    best = route_pnl.sort("est_profit", descending=True).row(0, named=True)
    worst = route_pnl.sort("est_profit", descending=False).row(0, named=True)

    return {
        "total_net_revenue": total_net_revenue,
        "total_fuel_cost": total_fuel_cost_usd,
        "margin_pct": margin_pct,
        "tax_burden_pct": tax_burden_pct,
        "total_tickets": total_tickets,
        "best_route": f"{best['origin']} → {best['destination']}",
        "worst_route": f"{worst['origin']} → {worst['destination']}",
        "is_empty": False,
    }


# ── Section 1: Route Profitability scatter ───────────────────────────────────

def route_profitability(
    revenue_filtered: pl.DataFrame,
    fuel: pl.DataFrame,
    fuel_price: float = DEFAULT_FUEL_PRICE_USD,
) -> pl.DataFrame:
    """One row per route with net_revenue, fuel_cost, ticket_count, continent.

    Used for the scatter chart: x=fuel_cost, y=net_revenue, size=ticket_count.
    `fuel` should already be filtered to the same year window as the revenue.
    """
    return (
        revenue_filtered.lazy()
        .group_by("route_code", "origin", "destination", "origin_continent")
        .agg(
            pl.col("net_revenue").sum(),
            pl.col("ticket_count").sum(),
        )
        .join(fuel_cost_by_route(fuel, fuel_price), on="route_code", how="left")
        .with_columns(pl.col("fuel_cost").fill_null(0))
        .with_columns(
            (pl.col("net_revenue") - pl.col("fuel_cost")).alias("est_profit"),
            route_label_expr(),
        )
        .sort("est_profit", descending=True)
        .collect()
    )


# ── Section 2: Tax Drain by Route ─────────────────────────────────────────────

def tax_drain_by_route(revenue_filtered: pl.DataFrame) -> pl.DataFrame:
    """Net revenue vs taxes per route, plus tax-as-%-of-gross.

    Sorted by tax share descending so the most tax-burdened routes appear
    first in the bar chart. No fuel involved, so this section is unaffected
    by the fuel grain.
    """
    return (
        revenue_filtered.lazy()
        .group_by("route_code", "origin", "destination")
        .agg(
            pl.col("net_revenue").sum(),
            pl.col("total_taxes").sum(),
            pl.col("gross_revenue").sum(),
            pl.col("ticket_count").sum(),
        )
        .with_columns(
            (safe_div(pl.col("total_taxes"), pl.col("gross_revenue")) * 100)
            .alias("tax_pct"),
            route_label_expr(),
        )
        .sort("tax_pct", descending=True, nulls_last=True)
        .collect()
    )


# ── Section 3: Fleet Cost Efficiency ──────────────────────────────────────────

def fleet_efficiency(
    revenue_filtered: pl.DataFrame,
    fuel: pl.DataFrame,
    fuel_price: float = DEFAULT_FUEL_PRICE_USD,
) -> pl.DataFrame:
    """Revenue per fuel gallon by aircraft model.

    Revenue is per route, not per model, so a route flown by several models
    has its revenue allocated across those models in proportion to each
    model's share of the route's gallons. This avoids counting a route's
    revenue once per model (or, with the year grain, once per model-year).

    `fuel` should already be filtered to the same year window as the revenue.
    The join is inner, so only routes present in the filtered revenue
    contribute — continent and volume filters propagate to the fleet view.
    """
    # Collapse the year dimension: gallons and flights per route-model.
    route_model = (
        fuel.lazy()
        .group_by("route_code", "model")
        .agg(
            pl.col("total_fuel_gallons").sum().alias("gallons"),
            pl.col("flights_operated").sum().alias("flights_operated"),
        )
    )
    route_total_gallons = (
        route_model
        .group_by("route_code")
        .agg(pl.col("gallons").sum().alias("route_gallons"))
    )
    rev_by_route = (
        revenue_filtered.lazy()
        .group_by("route_code")
        .agg(pl.col("net_revenue").sum().alias("route_net_rev"))
    )

    return (
        route_model
        .join(route_total_gallons, on="route_code", how="left")
        .join(rev_by_route, on="route_code", how="inner")
        .with_columns(
            # allocate route revenue to each model by its gallon share
            (pl.col("route_net_rev") * safe_div(pl.col("gallons"), pl.col("route_gallons")))
            .fill_null(0)
            .alias("alloc_rev")
        )
        .group_by("model")
        .agg(
            pl.col("gallons").sum().alias("gallons"),
            pl.col("alloc_rev").sum().alias("net_revenue"),
            pl.col("flights_operated").sum(),
        )
        .with_columns(
            safe_div(pl.col("net_revenue"), pl.col("gallons")).alias("rev_per_gallon"),
            (pl.col("gallons") * fuel_price).alias("fuel_cost"),
        )
        .sort("rev_per_gallon", descending=True, nulls_last=True)
        .collect()
    )


# ── Section 4: Margin Trend (full history) ────────────────────────────────────

def margin_trend(
    revenue: pl.DataFrame,
    fuel: pl.DataFrame,
    fuel_price: float = DEFAULT_FUEL_PRICE_USD,
) -> pl.DataFrame:
    """Annual net revenue, estimated fuel cost, and margin % per year.

    Both revenue and fuel now carry a `yr` column, so fuel cost is summed per
    year and joined directly. This replaces the old proportional allocation,
    which forced an identical margin every year (the net_revenue terms
    cancelled), and lets the margin actually vary year to year.

    Pass the unfiltered revenue and fuel here to show the full timeline.
    """
    annual_rev = (
        revenue.lazy()
        .group_by("yr")
        .agg(
            pl.col("net_revenue").sum(),
            pl.col("ticket_count").sum(),
        )
    )
    annual_fuel = (
        fuel.lazy()
        .group_by("yr")
        .agg((pl.col("total_fuel_gallons").sum() * fuel_price).alias("est_fuel_cost"))
    )

    return (
        annual_rev
        .join(annual_fuel, on="yr", how="left")
        .with_columns(pl.col("est_fuel_cost").fill_null(0))
        .with_columns(
            (
                safe_div(
                    pl.col("net_revenue") - pl.col("est_fuel_cost"),
                    pl.col("net_revenue"),
                )
                * 100
            ).alias("margin_pct")
        )
        .sort("yr")
        .collect()
    )
