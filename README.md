# IE Airlines Group 3 — The Real P&L Dashboard

> "Everyone shows revenue. We show what's left after costs."

## Group Members

#Nour Romdhane
#Philine de Graaf
#Marian Garabana
#Emily (Em) Echeverria
#Nuria Díaz Jiménez

## Schema

`ATTGRP3` on DB2 database `ATTPLANE` (host `52.211.123.34:25010`).

## How to Install

```bash
pip install -r requirements.txt
```

> On macOS, `ibm_db` compiles a native extension. If the install hangs,
> make sure Xcode command-line tools are installed (`xcode-select --install`).

### Database credentials

`db.py` reads connection settings from environment variables (a local `.env`
file is loaded automatically). The credentials are required and have no
hardcoded defaults. Create a `.env` file in the project root:

```dotenv
DB_USERNAME=your_user        # required
DB_PASSWORD=your_password    # required
# Optional — sensible defaults are used if omitted:
DB_HOST=52.211.123.34
DB_PORT=25010
DB_NAME=ATTPLANE
DB_SCHEMA=ATTGRP3
```

`.env` is gitignored and must never be committed.

## How to Run

### Step 1 — Pull data from DB2 (run once, takes ~2 min for 35M-row TICKETS)

```bash
python db.py
```

This saves three Parquet files plus a freshness marker to `data/`:

- `revenue.parquet` — revenue + taxes aggregated by route / year / class
- `fuel.parquet` — fuel consumption aggregated by route / aircraft model / **year**
- `airports.parquet` — full airport reference table
- `_generated.json` — generation timestamp and row counts (shown in the dashboard footer)

> **The Parquet files are committed to the repo** so Streamlit Cloud can run
> without a live DB connection. After any change to `db.py` (especially the
> extraction queries), re-run `python db.py` **and commit the regenerated
> `data/` files**, or the deployed dashboard will show stale numbers. The
> footer date tells you how fresh the committed data is.

### Step 2 — Launch the dashboard

```bash
streamlit run app.py
```

The sidebar includes a **fuel price slider** ($1.5–5/gallon) so you can test
how sensitive route profitability and margins are to fuel cost, without
re-querying the database.

## Testing

```bash
pytest          # unit tests (no database needed)
```

The analysis math runs on synthetic frames, so the suite is fast and needs no
DB. Tests tagged `db` hit the live database and skip automatically unless
credentials are set; run them explicitly with `pytest -m db`. CI
(`.github/workflows/tests.yml`) runs the unit tests on every push and PR.

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
   Section 4 plots annual net revenue vs estimated fuel cost (fuel summed per
   year) and derives a year-by-year estimated margin percentage.

## Key Findings

_Figures below are from the committed extracts (2010–2026, ~248.6M tickets) at
the default $3/gallon fuel price. They recalculate live as you move the
sidebar fuel slider._

1. **Fuel is a small slice of the P&L.** Across the full history, net revenue
   is about **$233.6B** and estimated fuel cost about **$11.6B** — roughly
   **5%** of net revenue. So the fuel-only margin sits near **95%** (this
   excludes crew, maintenance, and overhead — see Limitations).
2. **The most profitable routes are long-haul leisure corridors.** NAP↔LAS
   leads at roughly **$815M / $805M** net-of-fuel, with HND→FCO close behind
   (~**$801M**). The least profitable are short domestic hops such as CDG↔LIL
   (~**$15M**).
3. **UK domestic routes carry the heaviest tax burden.** LHR↔MAN tops the list
   at about **42.8%** of gross revenue lost to airport + local taxes, far above
   the ~16% system average.
4. **Widebodies return more revenue per gallon than regional jets.** The Boeing
   B747-400 (~**$72/gal**) and Airbus A330-300 variants (~$67–71/gal) lead;
   the Bombardier CRJ-200 (~**$28/gal**) and Airbus A319 (~$41/gal) trail.
5. **The fuel-only margin is remarkably stable.** Year-to-year margins stay
   within **95.0%–95.1%** from 2010 to 2025 (2026 reads slightly higher as a
   partial year), because fuel cost tracks revenue closely across the period.

## Assumptions & Limitations

| Assumption                           | Rationale                                                                                                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fuel price (default **$3 / gallon**, adjustable) | Jet-A spot price midpoint; real prices vary ~$1.5–5/gallon over the 2010–2026 period. The dashboard exposes a sidebar slider so any price recalculates live. |
| `TICKETS.PRICE` = net revenue        | Column definition excludes taxes; verified against `TOTAL_AMOUNT = PRICE + AIRPORT_TAX + LOCAL_TAX`                                                                                         |
| Taxes are pass-through               | AIRPORT_TAX and LOCAL_TAX are collected on behalf of airports/governments, not retained by the airline                                                                                      |
| Fuel by year via `YEAR(DEPARTURE)`   | `extract_fuel` groups flights by year, so yearly fuel cost is the real per-year burn — not a revenue-proportional estimate. Fuel has no cabin-class dimension, so class filters narrow revenue but not fuel. |
| No crew / overhead costs             | Maintenance, crew salaries, and airport handling are not modelled — "margin" is specifically revenue minus fuel                                                                             |

## File Structure

```
group_3_plane_dashboard/
├── app.py          — Streamlit dashboard UI
├── db.py           — DB2 connection, helpers, data extraction, extract reading
├── analysis.py     — Polars transformations for P&L metrics
├── explore.py      — ad-hoc DB exploration helpers (not used by the app)
├── tests/          — pytest suite (analysis math, db helpers, regression guards)
├── data/
│   ├── revenue.parquet
│   ├── fuel.parquet
│   ├── airports.parquet
│   └── _generated.json   — generation timestamp + row counts
├── .github/workflows/tests.yml — CI: runs pytest on push / PR
├── .env            — DB credentials (not committed to git)
├── requirements.txt
└── README.md
```
