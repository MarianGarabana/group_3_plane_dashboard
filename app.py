"""
app.py — IE Airlines Group 3 Streamlit Dashboard
"The Real P&L of IE Airlines"

Run: streamlit run app.py
Data must be pre-generated first: python db.py

Navigation: session-state driven single-file multi-page app.
  st.session_state["page"] = "home" | "q1" | "q2" | "q3" | "q4"
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

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IE Airlines — Real P&L",
    page_icon="✈",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# ── question catalogue (single source of truth for cards + routing) ───────────

QUESTIONS = {
    "q1": {
        "emoji": "📍",
        "title": "Route Profitability",
        "question": "Which routes earn the most after subtracting fuel costs?",
        "teaser": (
            "Revenue looks healthy everywhere — but fuel is the largest controllable "
            "cost. See which routes are truly profitable and which burn more than they earn."
        ),
    },
    "q2": {
        "emoji": "🏛️",
        "title": "Tax Drain",
        "question": "Where does airport tax eat the most margin?",
        "teaser": (
            "Airport taxes absorb ~16 % of gross revenue on average, but the burden "
            "is not spread evenly. Identify the routes with least pricing headroom."
        ),
    },
    "q3": {
        "emoji": "⚙️",
        "title": "Fleet Efficiency",
        "question": "Which aircraft model generates the most revenue per gallon?",
        "teaser": (
            "Revenue per gallon combines physical efficiency with route yield. "
            "Low-scoring models are candidates for redeployment or phase-out."
        ),
    },
    "q4": {
        "emoji": "📈",
        "title": "Margin Trend",
        "question": "Is the overall margin improving over time?",
        "teaser": (
            "A single margin snapshot can be misleading. Track the revenue-vs-fuel gap "
            "year by year to see whether the business is getting stronger or weaker."
        ),
    },
}

# ── data loading (cached) ─────────────────────────────────────────────────────

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


revenue_enriched = load_enriched_revenue()
fuel = load_fuel()

# ── routing helpers ───────────────────────────────────────────────────────────

def go_to(page: str) -> None:
    st.session_state["page"] = page

if "page" not in st.session_state:
    st.session_state["page"] = "home"

current_page = st.session_state["page"]

# ── shared sidebar (only shown on analysis pages) ─────────────────────────────

def render_sidebar() -> tuple:
    """Render sidebar filters and return (revenue_filtered, fuel_filtered, fuel_price).

    Fuel carries the same year dimension as revenue, so it is filtered to the
    same year window here — no chart should pair filtered revenue against
    all-history fuel.
    """
    icon_col, btn_col = st.sidebar.columns([1, 3])
    icon_col.markdown(
        '<p style="font-size:2.2rem; text-align:center; margin:0">🏠</p>',
        unsafe_allow_html=True,
    )
    btn_col.button(
        "← Home",
        on_click=go_to,
        args=("home",),
        use_container_width=True,
        type="primary",
    )
    st.sidebar.markdown("---")
    st.sidebar.title("Filters")

    yr_min = int(revenue_enriched["yr"].min())
    yr_max = int(revenue_enriched["yr"].max())
    year_range = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    class_options = {
        "All": [], "Business (B)": ["B"], "Premium (P)": ["P"], "Economy (E)": ["E"],
    }
    cabin_classes = class_options[st.sidebar.selectbox("Cabin class", list(class_options))]

    continent_list = sorted(
        revenue_enriched["origin_continent"].drop_nulls().unique().to_list()
    )
    origin_continents = st.sidebar.multiselect(
        "Origin continent", continent_list, default=[], placeholder="All continents",
    )

    min_tickets = st.sidebar.slider(
        "Min ticket volume per route", 0, 500_000, 0, step=10_000,
        help="Remove low-volume routes that can skew the profitability scatter.",
    )

    fuel_price = st.sidebar.slider(
        "Fuel price ($/gallon)", 1.5, 5.0, float(DEFAULT_FUEL_PRICE_USD), step=0.25,
        help="Jet-A spot price has ranged roughly $1.5–5/gallon. "
        "All fuel cost and margin figures recalculate live.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Fuel price: **${fuel_price:.2f}/gallon**.\n\n"
        "Taxes treated as pass-through — excluded from net revenue."
    )

    revenue_filtered = apply_filters(
        revenue_enriched,
        year_min=year_range[0],
        year_max=year_range[1],
        cabin_classes=cabin_classes,
        origin_continents=origin_continents,
        min_tickets=min_tickets,
    )
    fuel_filtered = filter_fuel_by_year(fuel, year_range[0], year_range[1])
    return revenue_filtered, fuel_filtered, fuel_price


def render_kpi_strip(kpis: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Net Revenue",   f"${kpis['total_net_revenue'] / 1e9:.2f}B")
    c2.metric("Est. Fuel Cost", f"${kpis['total_fuel_cost']  / 1e9:.2f}B")
    c3.metric("Est. Margin",   f"{kpis['margin_pct']:.1f}%")
    c4.metric("Avg Tax Burden", f"{kpis['tax_burden_pct']:.1f}%")
    c5.metric("Best Route",    kpis["best_route"])
    c6.metric("Worst Route",   kpis["worst_route"])


# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def render_home() -> None:
    st.title("✈ IE Airlines — The Real P&L")
    st.markdown(
        "### Everyone shows revenue. We show what's left after costs.\n"
        "Select a question below to explore the analysis."
    )
    st.markdown("---")

    # Overview at the default fuel price; each analysis page has a live slider.
    kpis = compute_kpis(revenue_enriched, fuel, DEFAULT_FUEL_PRICE_USD)
    render_kpi_strip(kpis)
    st.markdown("---")

    # 2 × 2 grid of question cards
    row1 = st.columns(2)
    row2 = st.columns(2)
    card_slots = [row1[0], row1[1], row2[0], row2[1]]

    for slot, (key, q) in zip(card_slots, QUESTIONS.items()):
        with slot:
            with st.container(border=True):
                st.markdown(f"## {q['emoji']}")
                st.markdown(f"**{q['title']}**")
                st.markdown(f"*{q['question']}*")
                st.caption(q["teaser"])
                st.button(
                    "Explore →",
                    key=f"btn_{key}",
                    on_click=go_to,
                    args=(key,),
                    use_container_width=True,
                    type="primary",
                )

    st.markdown("---")
    st.caption(
        f"Data: IE Airlines internal extract · Fuel assumption: ${DEFAULT_FUEL_PRICE_USD}/gallon · "
        "Taxes treated as pass-through."
    )
    generated = data_generated_at()
    if generated:
        st.caption(f"Data generated on {generated} (committed Parquet extracts).")


# ═══════════════════════════════════════════════════════════════════════════════
# Q1 — ROUTE PROFITABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def render_q1(revenue_filtered: pl.DataFrame, fuel_filtered: pl.DataFrame, fuel_price: float) -> None:
    st.title(f"{QUESTIONS['q1']['emoji']} {QUESTIONS['q1']['title']}")
    st.subheader(QUESTIONS["q1"]["question"])
    st.markdown("---")

    st.markdown(
        """
        **Why this matters:** A route can have high gross revenue and still destroy value
        if it flies fuel-hungry aircraft over long distances with thin load factors.

        **How to read this chart:** Each bubble is a route. X-axis = estimated fuel cost;
        Y-axis = net revenue. Bubbles **above the dashed line** earn more than they cost in
        fuel. Bubble size = total tickets sold.
        """
    )

    scatter_df = route_profitability(revenue_filtered, fuel_filtered, fuel_price)

    if scatter_df.is_empty():
        st.warning("No data for selected filters.")
        return

    fig = px.scatter(
        scatter_df.to_pandas(),
        x="fuel_cost", y="net_revenue",
        size="ticket_count", color="origin_continent",
        hover_name="route_label",
        hover_data={
            "est_profit": ":,.0f", "ticket_count": ":,",
            "fuel_cost": ":,.0f", "net_revenue": ":,.0f",
        },
        labels={
            "fuel_cost": "Est. Fuel Cost ($)", "net_revenue": "Net Revenue ($)",
            "origin_continent": "Continent", "ticket_count": "Tickets sold",
        },
        title="Net Revenue vs. Estimated Fuel Cost by Route",
        size_max=60,
    )
    axis_max = max(scatter_df["fuel_cost"].max(), scatter_df["net_revenue"].max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, axis_max], y=[0, axis_max],
        mode="lines", line=dict(color="grey", dash="dash", width=1),
        name="Break-even", showlegend=True,
    ))
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    top3    = scatter_df.sort("est_profit", descending=True).head(3)
    bottom3 = scatter_df.sort("est_profit", descending=False).head(3)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Top 3 routes by estimated profit**")
        for row in top3.iter_rows(named=True):
            st.markdown(
                f"- **{row['route_label']}** — "
                f"${row['est_profit']:,.0f} profit ({row['ticket_count']:,} tickets)"
            )
    with col_r:
        st.markdown("**Bottom 3 routes by estimated profit**")
        for row in bottom3.iter_rows(named=True):
            st.markdown(
                f"- **{row['route_label']}** — "
                f"${row['est_profit']:,.0f} profit ({row['ticket_count']:,} tickets)"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 — TAX DRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def render_q2(revenue_filtered: pl.DataFrame) -> None:
    st.title(f"{QUESTIONS['q2']['emoji']} {QUESTIONS['q2']['title']}")
    st.subheader(QUESTIONS["q2"]["question"])
    st.markdown("---")

    st.markdown(
        """
        **Why this matters:** Airport taxes erode gross revenue and reduce pricing headroom.
        Routes with a heavy tax burden are harder to reprice without demand destruction.

        **How to read these charts:** Left = routes ranked by tax as % of gross revenue.
        Right = absolute net revenue vs taxes stacked for the highest-volume routes.
        """
    )

    tax_df = tax_drain_by_route(revenue_filtered)

    if tax_df.is_empty():
        st.warning("No data for selected filters.")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        fig_a = px.bar(
            tax_df.head(20).to_pandas(),
            x="tax_pct", y="route_label", orientation="h",
            title="Tax Burden % by Route (top 20 most taxed)",
            labels={"tax_pct": "Tax % of Gross Revenue", "route_label": "Route"},
            color="tax_pct", color_continuous_scale="Reds",
        )
        fig_a.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_a, use_container_width=True)

    with col_b:
        top_vol = tax_df.sort("gross_revenue", descending=True).head(15)
        pd_stacked = top_vol.select("route_label", "net_revenue", "total_taxes").to_pandas()
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            x=pd_stacked["route_label"], y=pd_stacked["net_revenue"],
            name="Net Revenue", marker_color="#1f77b4",
        ))
        fig_b.add_trace(go.Bar(
            x=pd_stacked["route_label"], y=pd_stacked["total_taxes"],
            name="Taxes", marker_color="#d62728",
        ))
        fig_b.update_layout(
            barmode="stack",
            title="Net Revenue vs Taxes — Top 15 Routes by Volume",
            xaxis_tickangle=-45, height=520, yaxis_title="Amount ($)",
        )
        st.plotly_chart(fig_b, use_container_width=True)

    worst = tax_df.row(0, named=True)
    st.info(
        f"**Highest tax burden:** {worst['route_label']} — "
        f"{worst['tax_pct']:.1f}% of gross revenue goes to taxes "
        f"(${worst['total_taxes']:,.0f} across {worst['ticket_count']:,} tickets)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — FLEET EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════════

def render_q3(revenue_filtered: pl.DataFrame, fuel_filtered: pl.DataFrame, fuel_price: float) -> None:
    st.title(f"{QUESTIONS['q3']['emoji']} {QUESTIONS['q3']['title']}")
    st.subheader(QUESTIONS["q3"]["question"])
    st.markdown("---")

    st.markdown(
        """
        **Why this matters:** Revenue per gallon captures both physical efficiency and
        route yield quality. A model with low rev/gallon is a candidate for redeployment
        or phase-out — even if it looks busy in the schedule.

        **How to read this chart:** Bars ranked by net revenue per gallon of fuel.
        Hover for absolute fuel consumption and total flights operated.
        """
    )

    fleet_df = fleet_efficiency(revenue_filtered, fuel_filtered, fuel_price)

    if fleet_df.is_empty():
        st.warning("No fleet data available.")
        return

    fig = px.bar(
        fleet_df.to_pandas(),
        x="model", y="rev_per_gallon",
        color="rev_per_gallon", color_continuous_scale="Greens",
        title="Net Revenue per Fuel Gallon by Aircraft Model",
        labels={"model": "Aircraft Model", "rev_per_gallon": "Revenue / Gallon ($)"},
        hover_data={"gallons": ":,.0f", "fuel_cost": ":,.0f",
                    "net_revenue": ":,.0f", "flights_operated": ":,"},
    )
    fig.update_layout(height=450, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    best  = fleet_df.row(0,  named=True)
    worst = fleet_df.row(-1, named=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.success(
            f"**Most efficient:** {best['model']} — "
            f"${best['rev_per_gallon']:.2f}/gallon ({best['flights_operated']:,} flights)"
        )
    with col_r:
        st.error(
            f"**Least efficient:** {worst['model']} — "
            f"${worst['rev_per_gallon']:.2f}/gallon ({worst['flights_operated']:,} flights)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Q4 — MARGIN TREND
# ═══════════════════════════════════════════════════════════════════════════════

def render_q4(fuel_price: float) -> None:
    st.title(f"{QUESTIONS['q4']['emoji']} {QUESTIONS['q4']['title']}")
    st.subheader(QUESTIONS["q4"]["question"])
    st.markdown("---")

    trend_df = compute_margin_trend(fuel_price)  # always full history
    yr_lo, yr_hi = int(trend_df["yr"].min()), int(trend_df["yr"].max())

    st.markdown(
        f"""
        **Why this matters:** A single margin snapshot is misleading — what matters is
        the trajectory. Is the gap between revenue and fuel cost widening or narrowing?

        **Methodology note:** Fuel is grouped by year at the source (`YEAR(DEPARTURE)`),
        so each year's fuel cost is the real per-year burn — not a revenue-proportional
        estimate. The margin can therefore genuinely vary year to year. Shown across the
        full history ({yr_lo}–{yr_hi}) regardless of the sidebar filters.

        **How to read these charts:** Top = absolute net revenue vs estimated fuel cost by year.
        Bottom = resulting margin %, with a red break-even line at 0%.
        """
    )

    fig_abs = go.Figure()
    fig_abs.add_trace(go.Scatter(
        x=trend_df["yr"].to_list(), y=trend_df["net_revenue"].to_list(),
        name="Net Revenue", mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
    ))
    fig_abs.add_trace(go.Scatter(
        x=trend_df["yr"].to_list(), y=trend_df["est_fuel_cost"].to_list(),
        name="Est. Fuel Cost", mode="lines+markers",
        line=dict(color="#d62728", width=2, dash="dot"),
    ))
    fig_abs.update_layout(
        title="Net Revenue vs Estimated Fuel Cost — Full History",
        xaxis_title="Year", yaxis_title="Amount ($)", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_abs, use_container_width=True)

    fig_pct = px.line(
        trend_df.to_pandas(), x="yr", y="margin_pct",
        title="Estimated Margin % by Year",
        labels={"yr": "Year", "margin_pct": "Margin (%)"},
        markers=True,
    )
    fig_pct.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig_pct.update_layout(height=320)
    st.plotly_chart(fig_pct, use_container_width=True)

    best_yr  = trend_df.sort("margin_pct", descending=True).row(0,  named=True)
    worst_yr = trend_df.sort("margin_pct", descending=False).row(0, named=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.success(f"**Best year:** {best_yr['yr']} — {best_yr['margin_pct']:.1f}% margin")
    with col_r:
        st.warning(f"**Weakest year:** {worst_yr['yr']} — {worst_yr['margin_pct']:.1f}% margin")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

if current_page == "home":
    render_home()
else:
    revenue_filtered, fuel_filtered, fuel_price = render_sidebar()

    if current_page == "q1":
        render_q1(revenue_filtered, fuel_filtered, fuel_price)
    elif current_page == "q2":
        render_q2(revenue_filtered)
    elif current_page == "q3":
        render_q3(revenue_filtered, fuel_filtered, fuel_price)
    elif current_page == "q4":
        render_q4(fuel_price)
    else:
        go_to("home")
