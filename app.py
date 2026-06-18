"""
app.py — IE Airlines Group 3 Streamlit Dashboard
"The Real P&L of IE Airlines"

Run: streamlit run app.py
Data must be pre-generated first: python db.py
"""

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from db import DataNotGeneratedError, read_extract
from analysis import (
    DEFAULT_FUEL_PRICE_USD,
    apply_filters,
    compute_kpis,
    enrich_revenue_with_airports,
    filter_fuel_by_year,
    fleet_efficiency,
    margin_trend,
    route_profitability,
    tax_drain_by_route,
)

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IE Airlines — Real P&L",
    page_icon="✈",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"


# ── data loading (cached so Streamlit doesn't re-read on every interaction) ──

def _load_parquet(name: str) -> pl.DataFrame:
    """Read one Parquet extract, or stop the app with a clear message if the
    data has not been generated yet."""
    try:
        return read_extract(name)
    except DataNotGeneratedError as exc:
        st.error(f"{exc} Then reload this page.")
        st.stop()


@st.cache_data
def load_revenue() -> pl.DataFrame:
    return _load_parquet("revenue.parquet")


@st.cache_data
def load_fuel() -> pl.DataFrame:
    return _load_parquet("fuel.parquet")


@st.cache_data
def load_airports() -> pl.DataFrame:
    return _load_parquet("airports.parquet")


@st.cache_data
def load_enriched_revenue() -> pl.DataFrame:
    """Revenue joined with airport continent/city. Cached because it depends
    only on the (cached) Parquet files, never on the sidebar filters."""
    return enrich_revenue_with_airports(load_revenue(), load_airports())


@st.cache_data
def compute_margin_trend(fuel_price: float) -> pl.DataFrame:
    """Full-history margin trend. Cached per fuel price because it uses the
    unfiltered revenue and so does not change when the sidebar filters move."""
    return margin_trend(load_enriched_revenue(), load_fuel(), fuel_price)


def data_generated_at() -> str | None:
    """Read the generation timestamp written by db.py, if present."""
    meta_path = DATA_DIR / "_generated.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text()).get("generated_at")
    except (ValueError, OSError):
        return None


# ── load and enrich ──────────────────────────────────────────────────────────

revenue_enriched = load_enriched_revenue()
fuel = load_fuel()

# ── sidebar filters ──────────────────────────────────────────────────────────

st.sidebar.title("Filters")

yr_min = int(revenue_enriched["yr"].min())
yr_max = int(revenue_enriched["yr"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=yr_min,
    max_value=yr_max,
    value=(yr_min, yr_max),
)

class_options = {"All": [], "Business (B)": ["B"], "Premium (P)": ["P"], "Economy (E)": ["E"]}
class_choice = st.sidebar.selectbox("Cabin class", list(class_options.keys()))
cabin_classes = class_options[class_choice]

continent_list = sorted(
    revenue_enriched["origin_continent"].drop_nulls().unique().to_list()
)
origin_continents = st.sidebar.multiselect(
    "Origin continent",
    continent_list,
    default=[],
    placeholder="All continents",
)

min_tickets = st.sidebar.slider(
    "Min ticket volume per route",
    min_value=0,
    max_value=500_000,
    value=0,
    step=10_000,
    help="Remove low-volume routes that can skew the profitability scatter.",
)

fuel_price = st.sidebar.slider(
    "Fuel price ($/gallon)",
    min_value=1.5,
    max_value=5.0,
    value=float(DEFAULT_FUEL_PRICE_USD),
    step=0.25,
    help="Jet-A spot price has ranged roughly $1.5–5/gallon. "
    "All fuel cost and margin figures recalculate live.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Fuel price: **${fuel_price:.2f}/gallon**.\n\n"
    "Taxes treated as pass-through — excluded from net revenue."
)

# ── apply filters ────────────────────────────────────────────────────────────

revenue_filtered = apply_filters(
    revenue_enriched,
    year_min=year_range[0],
    year_max=year_range[1],
    cabin_classes=cabin_classes,
    origin_continents=origin_continents,
    min_tickets=min_tickets,
)
# Fuel carries the same year dimension as revenue; filter it to the same
# window so no chart pairs filtered revenue against all-history fuel.
fuel_filtered = filter_fuel_by_year(fuel, year_range[0], year_range[1])

# ── header ───────────────────────────────────────────────────────────────────

st.title("✈ IE Airlines — The Real P&L")
st.caption(
    "Everyone shows revenue. We show what's left after costs.  "
    "Net revenue = ticket price excluding taxes. "
    f"Fuel cost estimated at ${fuel_price:.2f}/gallon."
)

if revenue_filtered.is_empty():
    st.warning(
        "No data matches the current filters. Widen the year range, clear the "
        "continent filter, or lower the minimum ticket volume."
    )
    st.stop()

# ── KPI cards ────────────────────────────────────────────────────────────────

kpis = compute_kpis(revenue_filtered, fuel_filtered, fuel_price)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Net Revenue", f"${kpis['total_net_revenue'] / 1e9:.2f}B")
c2.metric("Est. Fuel Cost", f"${kpis['total_fuel_cost'] / 1e9:.2f}B")
c3.metric("Est. Margin", f"{kpis['margin_pct']:.1f}%")
c4.metric("Avg Tax Burden", f"{kpis['tax_burden_pct']:.1f}%")
c5.metric("Best Route", kpis["best_route"])
c6.metric("Worst Route", kpis["worst_route"])

st.markdown("---")

# ── Section 1: Route Profitability ───────────────────────────────────────────

st.subheader("1 · Route Profitability")
st.caption(
    "Each bubble is one route. Size = ticket volume. "
    "Routes above the diagonal earn more than they cost in fuel; "
    "below means fuel cost dominates."
)

scatter_df = route_profitability(revenue_filtered, fuel_filtered, fuel_price)

if scatter_df.is_empty():
    st.warning("No data for selected filters.")
else:
    fig1 = px.scatter(
        scatter_df.to_pandas(),
        x="fuel_cost",
        y="net_revenue",
        size="ticket_count",
        color="origin_continent",
        hover_name="route_label",
        hover_data={
            "est_profit": ":,.0f",
            "ticket_count": ":,",
            "fuel_cost": ":,.0f",
            "net_revenue": ":,.0f",
        },
        labels={
            "fuel_cost": "Est. Fuel Cost ($)",
            "net_revenue": "Net Revenue ($)",
            "origin_continent": "Continent",
            "ticket_count": "Tickets sold",
        },
        title="Net Revenue vs. Estimated Fuel Cost by Route",
        size_max=60,
    )
    # Diagonal reference line: break-even (revenue = cost)
    axis_max = max(scatter_df["fuel_cost"].max(), scatter_df["net_revenue"].max()) * 1.05
    fig1.add_trace(
        go.Scatter(
            x=[0, axis_max],
            y=[0, axis_max],
            mode="lines",
            line=dict(color="grey", dash="dash", width=1),
            name="Break-even",
            showlegend=True,
        )
    )
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── Section 2: Tax Drain by Route ────────────────────────────────────────────

st.subheader("2 · Tax Drain by Route")
st.caption(
    "How much of each ticket's gross amount is consumed by taxes (airport + local). "
    "High tax routes may warrant different pricing strategies."
)

tax_df = tax_drain_by_route(revenue_filtered)

col_a, col_b = st.columns(2)

with col_a:
    # Top 20 routes by tax share
    top_tax = tax_df.head(20)
    fig2a = px.bar(
        top_tax.to_pandas(),
        x="tax_pct",
        y="route_label",
        orientation="h",
        title="Tax Burden % by Route (top 20)",
        labels={"tax_pct": "Tax % of Gross Revenue", "route_label": "Route"},
        color="tax_pct",
        color_continuous_scale="Reds",
    )
    fig2a.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2a, use_container_width=True)

with col_b:
    # Stacked: net revenue vs taxes (top 15 by gross revenue)
    top_routes = tax_df.sort("gross_revenue", descending=True).head(15)
    stacked_pd = top_routes.select(
        "route_label", "net_revenue", "total_taxes"
    ).to_pandas()

    fig2b = go.Figure()
    fig2b.add_trace(go.Bar(
        x=stacked_pd["route_label"],
        y=stacked_pd["net_revenue"],
        name="Net Revenue",
        marker_color="#1f77b4",
    ))
    fig2b.add_trace(go.Bar(
        x=stacked_pd["route_label"],
        y=stacked_pd["total_taxes"],
        name="Taxes",
        marker_color="#d62728",
    ))
    fig2b.update_layout(
        barmode="stack",
        title="Net Revenue vs Taxes — Top 15 Routes by Volume",
        xaxis_tickangle=-45,
        height=500,
        yaxis_title="Amount ($)",
    )
    st.plotly_chart(fig2b, use_container_width=True)

st.markdown("---")

# ── Section 3: Fleet Cost Efficiency ─────────────────────────────────────────

st.subheader("3 · Fleet Cost Efficiency")
st.caption(
    "Revenue generated per gallon of fuel consumed, by aircraft model. "
    "Higher is better — models with low rev/gallon are burning fuel without "
    "proportional revenue return."
)

fleet_df = fleet_efficiency(revenue_filtered, fuel_filtered, fuel_price)

if fleet_df.is_empty():
    st.warning("No fleet data available.")
else:
    fig3 = px.bar(
        fleet_df.to_pandas(),
        x="model",
        y="rev_per_gallon",
        color="rev_per_gallon",
        color_continuous_scale="Greens",
        title="Net Revenue per Fuel Gallon by Aircraft Model",
        labels={
            "model": "Aircraft Model",
            "rev_per_gallon": "Revenue / Gallon ($)",
        },
        hover_data={
            "gallons": ":,.0f",
            "fuel_cost": ":,.0f",
            "net_revenue": ":,.0f",
            "flights_operated": ":,",
        },
    )
    fig3.update_layout(height=420, xaxis_tickangle=-30)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Section 4: Margin Trend ──────────────────────────────────────────────────

trend_df = compute_margin_trend(fuel_price)  # full timeline, unaffected by filters
trend_yr_min = int(trend_df["yr"].min())
trend_yr_max = int(trend_df["yr"].max())

st.subheader(f"4 · Margin Trend {trend_yr_min}–{trend_yr_max}")
st.caption(
    "Annual net revenue vs estimated fuel cost, with fuel summed per year. "
    "Shown across the full data history regardless of the sidebar filters."
)

fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=trend_df["yr"].to_list(),
    y=trend_df["net_revenue"].to_list(),
    name="Net Revenue",
    mode="lines+markers",
    line=dict(color="#1f77b4", width=2),
))
fig4.add_trace(go.Scatter(
    x=trend_df["yr"].to_list(),
    y=trend_df["est_fuel_cost"].to_list(),
    name="Est. Fuel Cost",
    mode="lines+markers",
    line=dict(color="#d62728", width=2, dash="dot"),
))
fig4.update_layout(
    title="Net Revenue vs Estimated Fuel Cost — Full History",
    xaxis_title="Year",
    yaxis_title="Amount ($)",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig4, use_container_width=True)

# Secondary: margin % trend
fig4b = px.line(
    trend_df.to_pandas(),
    x="yr",
    y="margin_pct",
    title="Estimated Margin % by Year",
    labels={"yr": "Year", "margin_pct": "Margin (%)"},
    markers=True,
)
fig4b.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
fig4b.update_layout(height=300)
st.plotly_chart(fig4b, use_container_width=True)

st.markdown("---")

# ── Data preview + download ───────────────────────────────────────────────────

with st.expander("Data preview — filtered revenue table"):
    preview = (
        revenue_filtered
        .group_by("route_code", "origin", "destination", "origin_continent")
        .agg(
            pl.col("ticket_count").sum(),
            pl.col("net_revenue").sum(),
            pl.col("total_taxes").sum(),
        )
        .sort("net_revenue", descending=True)
    )
    st.dataframe(preview.to_pandas(), use_container_width=True)
    st.download_button(
        "Download as CSV",
        data=preview.write_csv(),
        file_name="ie_airlines_revenue_filtered.csv",
        mime="text/csv",
    )

# ── footer: data freshness ────────────────────────────────────────────────────

_generated = data_generated_at()
if _generated:
    st.caption(f"Data generated on {_generated} (committed Parquet extracts).")
else:
    st.caption("Data generation date unknown — run `python db.py` to refresh.")
