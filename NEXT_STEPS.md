# Next Steps — Group 3 Dashboard

## Immediate (do today)

- [ ] Open `http://localhost:8501` — test all sidebar filters, check for visual bugs
- [ ] Write 5 key findings in `README.md` — use real numbers from the scatter and tax drain charts

## Before Submission

- [ ] Be ready to answer the professor's expected questions:
  - *Why these metrics?* → fuel + tax is the P&L angle other groups skip
  - *Which tables joined and why?* → TICKETS→ROUTES, FLIGHTS→AIRPLANES→ROUTES
  - *What assumptions?* → \$3/gallon fixed, taxes pass-through, fuel allocated proportionally by year
- [ ] Create the submission ZIP (excludes `.env` and `data/` — grader runs `python db.py` themselves):
  ```bash
  cd ~ && zip -r group_3_plane_dashboard.zip group_3_plane_dashboard/ \
    --exclude "*/data/*" \
    --exclude "*/.env" \
    --exclude "*/__pycache__/*"
  ```

## 8-Minute Demo Structure

| Time | Content |
|------|---------|
| 1 min | Business problem: "We show margin, not just revenue — fuel cost is the biggest controllable cost" |
| 2 min | Code walkthrough: `db.py` — pre-aggregation (never loads 35M rows raw), `q_ident` helper, Decimal→Float64 cast fix |
| 5 min | Live dashboard: year slider, cabin filter, scatter (best/worst routes), tax drain bar |
| 2 min | Limitations: fuel price fixed at \$3 — real improvement uses historical jet-A prices; margin % trend flat due to proportional allocation |

## Exam Prep — Highest Risk Areas

**Polars (25% of grade)** — know these cold:

- Why `.lazy()→.collect()`? Query optimizer defers and combines operations before execution
- Every `group_by().agg()` call in `analysis.py` — be able to explain what each aggregation produces and why
- Why `fill_null(0)` after joins? Routes with no fuel data would produce null profit

**Database (15% of grade)**:

- Why pre-aggregate in SQL? TICKETS has 35M rows — loading raw into Python would OOM or time out
- What does `q_ident()` do? Double-quotes identifiers to handle reserved words and mixed case safely
- The fuel overflow bug: `COUNT(*) * INTEGER * INTEGER` overflows DB2 INT range → fixed by computing in Polars with Float64

**Business insight (25% of grade)**:

- Most profitable route: check KPI card → `NAP → LAS`
- Least profitable route: `CDG → LIL`
- Overall estimated margin: ~94.7% (fuel is a small % of revenue at \$3/gallon)
- Average tax burden: ~16.3% of gross revenue
