# IE Airlines Group 3 — The Real P&L Dashboard

> "Everyone shows revenue. We show what's left after costs."

## Group Members

*(add names here)*

## Schema

`ATTGRP3` on DB2 database `ATTPLANE` (host `52.211.123.34:25010`).

## How to Install

```bash
pip install -r requirements.txt
```

> On macOS, `ibm_db` compiles a native extension. If the install hangs,
> make sure Xcode command-line tools are installed (`xcode-select --install`).

## How to Run

### Step 1 — Pull data from DB2 (run once, takes ~2 min for 35M-row TICKETS)

```bash
python db.py
```

This saves three Parquet files to `data/`:
- `revenue.parquet` — revenue + taxes aggregated by route / year / class
- `fuel.parquet` — fuel consumption aggregated by route / aircraft model
- `airports.parquet` — full airport reference table

### Step 2 — Launch the dashboard

```bash
streamlit run app.py
```

## Business Questions Answered

1. **Which routes are actually profitable after fuel costs?**  
   Section 1 scatter plot shows net revenue vs estimated fuel cost per route.
   Routes above the break-even diagonal earn more than their fuel burn.

2. **Which airports impose the heaviest tax burden, and by how much?**  
   Section 2 shows tax as % of gross revenue per route, and stacks net revenue
   vs taxes to show how much of each ticket's value is consumed by taxes.

3. **Which aircraft models deliver the best revenue per gallon of fuel?**  
   Section 3 ranks aircraft models by net revenue / fuel gallon — a proxy for
   fleet cost efficiency that goes beyond simple utilisation counts.

4. **How has the airline's margin evolved from 2010 to 2026?**  
   Section 4 plots the annual net revenue vs estimated fuel cost trend and
   derives a year-by-year estimated margin percentage.

## Key Findings

*(Fill in after running the dashboard against real data.)*

1. …
2. …
3. …
4. …
5. …

## Assumptions & Limitations

| Assumption | Rationale |
|---|---|
| Fuel price = **$3 / gallon** (fixed) | Jet-A spot price midpoint; real prices vary ~$1.5–5/gallon over the 2000–2024 period |
| `TICKETS.PRICE` = net revenue | Column definition excludes taxes; verified against `TOTAL_AMOUNT = PRICE + AIRPORT_TAX + LOCAL_TAX` |
| Taxes are pass-through | AIRPORT_TAX and LOCAL_TAX are collected on behalf of airports/governments, not retained by the airline |
| Annual fuel allocation | `extract_fuel` (Query 2) has no year dimension on the FLIGHTS side, so yearly fuel cost is estimated by allocating total fleet fuel cost proportionally to each year's share of net revenue |
| No crew / overhead costs | Maintenance, crew salaries, and airport handling are not modelled — "margin" is specifically revenue minus fuel |

## File Structure

```
group_3_plane_dashboard/
├── app.py          — Streamlit dashboard UI
├── db.py           — DB2 connection, helpers, data extraction
├── analysis.py     — Polars transformations for P&L metrics
├── data/
│   ├── revenue.parquet
│   ├── fuel.parquet
│   └── airports.parquet
├── .env            — DB credentials (not committed to git)
├── requirements.txt
└── README.md
```
