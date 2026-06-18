"""
app.py — IE Airlines Group 3 Streamlit Dashboard
"The Real P&L of IE Airlines"

Run: streamlit run app.py
Data must be pre-generated first: python db.py
"""

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from analysis import (
    FUEL_PRICE_USD,
    apply_filters,
    compute_kpis,
    enrich_revenue_with_airports,
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

@st.cache_data
def load_revenue() -> pl.DataFrame:
    return pl.read_parquet(DATA_DIR / "revenue.parquet")


@st.cache_data
def load_fuel() -> pl.DataFrame:
    return pl.read_parquet(DATA_DIR / "fuel.parquet")


@st.cache_data
def load_airports() -> pl.DataFrame:
    return pl.read_parquet(DATA_DIR / "airports.parquet")


# ── load and enrich ──────────────────────────────────────────────────────────

revenue_raw = load_revenue()
fuel = load_fuel()
airports = load_airports()

# Join continent / city onto revenue rows once; filters operate on this df
revenue_enriched = enrich_revenue_with_airports(revenue_raw, airports)

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

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Fuel price assumption: **${FUEL_PRICE_USD}/gallon** (fixed).\n\n"
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

# ── header ───────────────────────────────────────────────────────────────────

st.title("✈ IE Airlines — The Real P&L")
st.caption(
    "Everyone shows revenue. We show what's left after costs.  "
    "Net revenue = ticket price excluding taxes. "
    "Fuel cost estimated at $3/gallon."
)

# ── KPI cards ────────────────────────────────────────────────────────────────

kpis = compute_kpis(revenue_filtered, fuel)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Net Revenue", f"${kpis['total_net_revenue'] / 1e9:.2f}B")
c2.metric("Est. Fuel Cost", f"${kpis['total_fuel_cost'] / 1e9:.2f}B")
c3.metric("Est. Margin", f"{kpis['margin_pct']:.1f}%")
c4.metric("Avg Tax Burden", f"{kpis['tax_burden_pct']:.1f}%")
c5.metric("Best Route", kpis["best_route"])
c6.metric("Worst Route", kpis["worst_route"])

st.markdown("---")

# ── Question-based tabs ───────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Q1 · Which routes earn most after fuel?",
    "Q2 · Where does tax eat the most margin?",
    "Q3 · Which aircraft is most fuel-efficient?",
    "Q4 · Is the margin improving over time?",
    "Data",
])

# ── Q1: Route Profitability ───────────────────────────────────────────────────

with tab1:
    st.subheader("Q1 · Which routes earn the most after subtracting fuel costs?")
    st.markdown(
        """
        **Why this matters:** Revenue figures look healthy across all routes, but fuel is the
        largest controllable cost. A route can have high gross revenue and still destroy value
        if it flies fuel-hungry aircraft over long distances with thin load factors.

        **How to read this chart:** Each bubble is a route. The X-axis is estimated fuel cost;
        the Y-axis is net revenue (ticket price minus airport taxes). Bubbles **above the dashed
        line** earn more than they cost in fuel — they are profitable on a fuel basis. Bubbles
        **below** are burning more fuel cost than they generate in net revenue.
        Bubble size = total tickets sold.
        """
    )

    scatter_df = route_profitability(revenue_filtered, fuel)

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
        fig1.update_layout(height=520)
        st.plotly_chart(fig1, use_container_width=True)

        # Key findings
        top3 = scatter_df.sort("est_profit", descending=True).head(3)
        bottom3 = scatter_df.sort("est_profit", descending=False).head(3)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Top 3 routes by estimated profit**")
            for row in top3.iter_rows(named=True):
                st.markdown(
                    f"- **{row['route_label']}** — "
                    f"${row['est_profit']:,.0f} profit "
                    f"({row['ticket_count']:,} tickets)"
                )
        with col_r:
            st.markdown("**Bottom 3 routes by estimated profit**")
            for row in bottom3.iter_rows(named=True):
                st.markdown(
                    f"- **{row['route_label']}** — "
                    f"${row['est_profit']:,.0f} profit "
                    f"({row['ticket_count']:,} tickets)"
                )

# ── Q2: Tax Drain ─────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Q2 · Where does airport tax eat the most margin?")
    st.markdown(
        """
        **Why this matters:** Airport taxes (departure, arrival, local) are passed through to the
        passenger but erode gross revenue. Routes with a heavy tax burden have less pricing
        headroom — raising fares risks demand destruction while taxes already absorb ~16% on
        average. Knowing *which* routes carry the highest tax share lets the commercial team
        apply targeted ancillary pricing or route renegotiations.

        **How to read these charts:** The left bar ranks routes by tax as a percentage of gross
        revenue. The right chart stacks net revenue and taxes for the top-volume routes so you
        can see absolute magnitudes alongside percentages.
        """
    )

    tax_df = tax_drain_by_route(revenue_filtered)

    col_a, col_b = st.columns(2)

    with col_a:
        top_tax = tax_df.head(20)
        fig2a = px.bar(
            top_tax.to_pandas(),
            x="tax_pct",
            y="route_label",
            orientation="h",
            title="Tax Burden % by Route (top 20 most taxed)",
            labels={"tax_pct": "Tax % of Gross Revenue", "route_label": "Route"},
            color="tax_pct",
            color_continuous_scale="Reds",
        )
        fig2a.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2a, use_container_width=True)

    with col_b:
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
            height=520,
            yaxis_title="Amount ($)",
        )
        st.plotly_chart(fig2b, use_container_width=True)

    # Key finding callout
    worst_tax_row = tax_df.row(0, named=True)
    st.info(
        f"**Highest tax burden:** {worst_tax_row['route_label']} — "
        f"{worst_tax_row['tax_pct']:.1f}% of gross revenue goes to taxes "
        f"(${worst_tax_row['total_taxes']:,.0f} total across {worst_tax_row['ticket_count']:,} tickets)."
    )

# ── Q3: Fleet Efficiency ──────────────────────────────────────────────────────

with tab3:
    st.subheader("Q3 · Which aircraft model generates the most revenue per gallon of fuel?")
    st.markdown(
        """
        **Why this matters:** Fuel efficiency is the primary lever for reducing operating costs
        without cutting capacity. Revenue per gallon captures both the aircraft's physical
        efficiency and the revenue quality of the routes it typically flies. A model with low
        rev/gallon is either physically inefficient or assigned to low-yield routes — either way,
        it is a candidate for redeployment or phase-out.

        **How to read this chart:** Bars represent aircraft models ranked by net revenue earned
        per gallon of fuel burned. Hover for absolute fuel consumption and total flights operated.
        """
    )

    fleet_df = fleet_efficiency(revenue_filtered, fuel)

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
        fig3.update_layout(height=450, xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

        best_model = fleet_df.row(0, named=True)
        worst_model = fleet_df.row(-1, named=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.success(
                f"**Most efficient:** {best_model['model']} — "
                f"${best_model['rev_per_gallon']:.2f} revenue/gallon "
                f"({best_model['flights_operated']:,} flights)"
            )
        with col_r:
            st.error(
                f"**Least efficient:** {worst_model['model']} — "
                f"${worst_model['rev_per_gallon']:.2f} revenue/gallon "
                f"({worst_model['flights_operated']:,} flights)"
            )

# ── Q4: Margin Trend ──────────────────────────────────────────────────────────

with tab4:
    st.subheader("Q4 · Is the overall margin improving over time?")
    st.markdown(
        """
        **Why this matters:** A single margin snapshot can be misleading — what matters for
        management is the trajectory. Is the gap between revenue and fuel cost widening
        (efficiency gains, better pricing) or narrowing (rising costs, yield erosion)?

        **Methodology note:** The fuel dataset does not carry a year dimension, so fleet-wide
        fuel cost is allocated to each year proportionally by that year's share of total net
        revenue. This means the margin % trend reflects revenue mix changes, not actual
        year-by-year fuel price variation. A real improvement would use historical jet-A prices
        per year.

        **How to read these charts:** The top chart shows absolute net revenue vs estimated fuel
        cost by year. The bottom chart shows the resulting margin percentage, with a red
        break-even line at 0%.
        """
    )

    trend_df = margin_trend(revenue_enriched, fuel)  # unfiltered revenue for full timeline

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

    fig4b = px.line(
        trend_df.to_pandas(),
        x="yr",
        y="margin_pct",
        title="Estimated Margin % by Year",
        labels={"yr": "Year", "margin_pct": "Margin (%)"},
        markers=True,
    )
    fig4b.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig4b.update_layout(height=320)
    st.plotly_chart(fig4b, use_container_width=True)

    # Year with highest / lowest margin
    best_yr = trend_df.sort("margin_pct", descending=True).row(0, named=True)
    worst_yr = trend_df.sort("margin_pct", descending=False).row(0, named=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.success(
            f"**Best year:** {best_yr['yr']} — {best_yr['margin_pct']:.1f}% margin"
        )
    with col_r:
        st.warning(
            f"**Weakest year:** {worst_yr['yr']} — {worst_yr['margin_pct']:.1f}% margin"
        )

# ── Data tab ──────────────────────────────────────────────────────────────────

with tab5:
    st.subheader("Filtered revenue table")
    st.caption("Aggregated to route level for the current sidebar filter selection.")

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
