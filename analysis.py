"""
analysis.py — Polars transformations that turn the raw Parquet extracts
into dashboard-ready metrics.

All functions accept the three base DataFrames (revenue, fuel, airports)
and return new DataFrames. No DB calls here — pure Polars.

Fuel price assumption: $3 / gallon (documented in README and noted in
every function that uses it so readers know where the number comes from).
"""

import polars as pl

FUEL_PRICE_USD = 3.0  # $/gallon — fixed assumption for cost estimation


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
        df = df.filter(pl.col("route_code").is_in(keep))

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
    airports_slim = airports.select(
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
        .join(airports_slim.lazy(), left_on="origin", right_on="iata_code", how="left")
        .join(dest_slim.lazy(), left_on="destination", right_on="iata_code", how="left")
        .collect()
    )


# ── KPI aggregates ───────────────────────────────────────────────────────────

def compute_kpis(revenue_filtered: pl.DataFrame, fuel: pl.DataFrame) -> dict:
    """Return a dict of scalar KPIs for the top cards.

    Fuel cost = total_fuel_gallons × FUEL_PRICE_USD.
    Margin = (net_revenue - fuel_cost) / net_revenue.
    Tax burden = total_taxes / gross_revenue, averaged per ticket.
    """
    total_net_revenue = revenue_filtered["net_revenue"].sum()
    total_taxes = revenue_filtered["total_taxes"].sum()
    total_gross = revenue_filtered["gross_revenue"].sum()
    total_tickets = revenue_filtered["ticket_count"].sum()

    # Fuel totals come from the unfiltered fuel table (no year/class split
    # in that extract), so we use the full fuel dataset for cost estimation.
    total_fuel_gallons = fuel["total_fuel_gallons"].sum()
    total_fuel_cost = total_fuel_gallons * FUEL_PRICE_USD

    margin_pct = (
        (total_net_revenue - total_fuel_cost) / total_net_revenue * 100
        if total_net_revenue > 0
        else 0.0
    )
    tax_burden_pct = (
        total_taxes / total_gross * 100
        if total_gross > 0
        else 0.0
    )

    # Most / least profitable route by (net_revenue - allocated fuel cost).
    # We can only compute a rough allocation here since fuel isn't split by
    # year/class — divide total fuel proportionally by net_revenue share.
    route_rev = (
        revenue_filtered.lazy()
        .group_by("route_code", "origin", "destination")
        .agg(pl.col("net_revenue").sum())
        .collect()
    )
    route_fuel = (
        fuel.lazy()
        .group_by("route_code")
        .agg((pl.col("total_fuel_gallons").sum() * FUEL_PRICE_USD).alias("fuel_cost"))
        .collect()
    )
    route_pnl = (
        route_rev.lazy()
        .join(route_fuel.lazy(), on="route_code", how="left")
        .with_columns(
            (pl.col("net_revenue") - pl.col("fuel_cost").fill_null(0)).alias("est_profit")
        )
        .collect()
    )

    best = route_pnl.sort("est_profit", descending=True).row(0, named=True)
    worst = route_pnl.sort("est_profit", descending=False).row(0, named=True)

    def route_label(row: dict) -> str:
        orig = row.get("origin") or "?"
        dest = row.get("destination") or "?"
        return f"{orig} → {dest}"

    return {
        "total_net_revenue": total_net_revenue,
        "total_fuel_cost": total_fuel_cost,
        "margin_pct": margin_pct,
        "tax_burden_pct": tax_burden_pct,
        "total_tickets": total_tickets,
        "best_route": route_label(best),
        "worst_route": route_label(worst),
    }


# ── Section 1: Route Profitability scatter ───────────────────────────────────

def route_profitability(
    revenue_filtered: pl.DataFrame,
    fuel: pl.DataFrame,
) -> pl.DataFrame:
    """One row per route with net_revenue, fuel_cost, ticket_count, continent.

    Used for the scatter chart: x=fuel_cost, y=net_revenue, size=ticket_count.
    """
    rev = (
        revenue_filtered.lazy()
        .group_by("route_code", "origin", "destination", "origin_continent")
        .agg(
            pl.col("net_revenue").sum(),
            pl.col("ticket_count").sum(),
        )
        .collect()
    )
    fuel_by_route = (
        fuel.lazy()
        .group_by("route_code")
        .agg((pl.col("total_fuel_gallons").sum() * FUEL_PRICE_USD).alias("fuel_cost"))
        .collect()
    )
    return (
        rev.lazy()
        .join(fuel_by_route.lazy(), on="route_code", how="left")
        .with_columns(pl.col("fuel_cost").fill_null(0))
        .with_columns(
            (pl.col("net_revenue") - pl.col("fuel_cost")).alias("est_profit"),
            (pl.col("origin") + " → " + pl.col("destination")).alias("route_label"),
        )
        .sort("est_profit", descending=True)
        .collect()
    )


# ── Section 2: Tax Drain by Airport ─────────────────────────────────────────

def tax_drain_by_route(revenue_filtered: pl.DataFrame) -> pl.DataFrame:
    """Net revenue vs taxes per route, plus tax-as-%-of-gross.

    Sorted by tax share descending so the most tax-burdened routes appear
    first in the bar chart.
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
            (pl.col("total_taxes") / pl.col("gross_revenue") * 100)
            .alias("tax_pct"),
            (pl.col("origin") + " → " + pl.col("destination")).alias("route_label"),
        )
        .sort("tax_pct", descending=True)
        .collect()
    )


# ── Section 3: Fleet Cost Efficiency ────────────────────────────────────────

def fleet_efficiency(
    revenue_filtered: pl.DataFrame,
    fuel: pl.DataFrame,
) -> pl.DataFrame:
    """Revenue per fuel gallon by aircraft model.

    Revenue here is net_revenue from the filtered window; fuel gallons
    come from the full fuel extract (no year filter on the flight side).
    Joining on route_code links the two aggregates.
    """
    rev_by_route = (
        revenue_filtered.lazy()
        .group_by("route_code")
        .agg(pl.col("net_revenue").sum())
        .collect()
    )
    # Keep model in fuel so we can group by it after the join
    fuel_with_rev = (
        fuel.lazy()
        .join(rev_by_route.lazy(), on="route_code", how="left")
        .with_columns(pl.col("net_revenue").fill_null(0))
        .group_by("model")
        .agg(
            pl.col("total_fuel_gallons").sum().alias("gallons"),
            pl.col("net_revenue").sum().alias("net_revenue"),
            pl.col("flights_operated").sum(),
        )
        .with_columns(
            (pl.col("net_revenue") / pl.col("gallons")).alias("rev_per_gallon"),
            (pl.col("gallons") * FUEL_PRICE_USD).alias("fuel_cost"),
        )
        .sort("rev_per_gallon", descending=True)
        .collect()
    )
    return fuel_with_rev


# ── Section 4: Margin Trend 2000–2024 ───────────────────────────────────────

def margin_trend(
    revenue: pl.DataFrame,
    fuel: pl.DataFrame,
) -> pl.DataFrame:
    """Annual net revenue, estimated fuel cost, and margin % from 2000–2024.

    Fuel doesn't carry a year dimension (flights aren't date-filtered in
    Query 2), so we allocate total fuel cost proportionally across years
    using each year's share of total net revenue. This is an approximation
    documented in the README.
    """
    annual_rev = (
        revenue.lazy()
        .group_by("yr")
        .agg(
            pl.col("net_revenue").sum(),
            pl.col("ticket_count").sum(),
        )
        .sort("yr")
        .collect()
    )

    total_fuel_cost = fuel["total_fuel_gallons"].sum() * FUEL_PRICE_USD
    total_net_rev = annual_rev["net_revenue"].sum()

    return (
        annual_rev.lazy()
        .with_columns(
            # Allocate fleet-wide fuel cost by this year's revenue share
            (pl.col("net_revenue") / total_net_rev * total_fuel_cost)
            .alias("est_fuel_cost"),
        )
        .with_columns(
            (
                (pl.col("net_revenue") - pl.col("est_fuel_cost"))
                / pl.col("net_revenue")
                * 100
            ).alias("margin_pct")
        )
        .collect()
    )
