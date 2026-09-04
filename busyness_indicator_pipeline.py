# %% [markdown]
# # GB Rail Busyness Indicator — Production Pipeline
#
# **Purpose**: produces the railway busyness indicator agreed with the stakeholder:
#
# 1. Passenger journey counts by travel date (day / week / month)
# 2. The same counts broken down by destination region (ITL1)
# 3. Average journey length (passenger miles), including a provisional
#    commute-like / leisure-like split by distance
# 4. A time-of-day / day-of-week pattern (added on request — see section 17 for
#    an important scope caveat: this covers same-day-issued travel only)
#
# **Audience**: this is intended to become a published ONS statistic (part of a
# weekly/monthly "faster indicators" release, general public audience) — not an
# internal one-off analysis. That has real consequences for how this file is written:
# every methodology choice is a named, documented, easily-changed parameter rather
# than a hardcoded assumption, because the definition of "commute vs. leisure" in
# particular is explicitly provisional and expected to be revised.
#
# **How to run**: top to bottom, either as a plain Python script
# (`python busyness_indicator_pipeline.py`) or as a notebook — the `# %%` markers
# make this file directly openable as a notebook in VS Code, or convertible with
# `jupytext --to notebook busyness_indicator_pipeline.py`.
#
# **What it produces**:
# - Nine CSV files in `OUTPUT_DIR`: raw daily/weekly/monthly and indexed weekly/
#   monthly figures (fully unmasked, for exploration inside this environment
#   only), plus four **disclosure-safe** exports (weekly/monthly, raw and
#   indexed) with any cell built from 10 or fewer underlying rows masked —
#   see section 16. **The disclosure-safe files are the only ones that should
#   ever be considered for export out of this environment.**
# - One self-contained interactive HTML file (section 19) for exploring the
#   unmasked figures inside the environment, plus a full set of static
#   matplotlib charts (sections 9-15, 18) covering the same ground.
# - A printed cost/performance/sanity-check summary at the end of the run.
#
# **Companion reading**: `busyness-indicator-documentation.md` covers the full
# methodology, every assumption, and every open question this pipeline currently
# depends on. `analysis-notes.md` and `data-dictionary.md` cover how we got here.
#
# **Change log** (update this as the pipeline evolves — see "designed to change"
# below):
# - v1 — initial build, three indicators, provisional commute threshold.
# - v2 — fixed a Decimal/float arithmetic error (NUMERIC columns cast to
#   FLOAT64 at the SQL boundary); fixed the unweighted average being dropped
#   (and mathematically wrong if it hadn't been) at weekly/monthly grain by
#   summing ticket_miles in SQL rather than averaging it; corrected the
#   disclosure threshold from "< 10" to "<= 10" throughout; added real
#   primary-suppression masking (section 16) rather than flagging only; added
#   the time-of-day/day-of-week pattern (sections 17-18); added an
#   interactive HTML export (section 19).

# %%
# ============================================================================
# 0. IMPORTS
# ============================================================================
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

# %% [markdown]
# ## 1. Configuration
#
# **Everything that might reasonably need to change lives here, and nowhere else.**
# The stakeholder was explicit that the commute/leisure distance threshold is a
# starting proposal, not a final answer, and that new requests are expected — the
# whole point of collecting every tunable value in one place at the top of the file
# is that changing the analysis later means editing a number here, not hunting
# through query logic further down.

# %%
# ============================================================================
# 1. CONFIGURATION
# ============================================================================

VIEW_FQN = "ons-ids-data-rfasta-prod.views.prices_lennon-staged_research"

# --- Commute vs. leisure classification -------------------------------------
# PROVISIONAL. The stakeholder asked us to propose a starting definition while
# they work on their own, on the understanding this must stay trivially
# adjustable. A journey with ticket_miles <= this threshold is treated as
# "commute-like"; anything longer is "leisure-like". 30 miles was chosen as a
# round, defensible starting point (roughly the outer edge of a typical UK
# metro-area commute) — NOT derived from a rigorous review of commuting
# literature, and should be revisited once the stakeholder's own work lands.
COMMUTE_DISTANCE_THRESHOLD_MILES: float = 30.0

# --- Date range ---------------------------------------------------------------
# None = process the full available history. The stakeholder confirmed "as far
# back as the data goes" — resolved at runtime from the data itself (see the
# sanity-check section below) rather than hardcoded, since the true earliest
# `collection_date` isn't something this file should need to know in advance.
DATE_RANGE_START: date | None = None
DATE_RANGE_END: date | None = None

# --- Indexing -----------------------------------------------------------------
# Base period for the indexed output (index = 100 in this period). None = use
# the first complete period in the processed range. Format: 'YYYY-MM' for
# monthly, ISO year-week (e.g. '2019-W27') for weekly — set by the indexing
# function itself if left as None.
INDEX_BASE_PERIOD: str | None = None

# --- Row-level filtering -------------------------------------------------------
# Fare product groups excluded from the passenger-journey count. This mirrors
# ONS's own published cleaning methodology for this exact dataset (APCP-T(22)02,
# Annex B): "N/A" covers non-journey products (car parking, seat reservations);
# "Other tickets" covers obscure non-consumer products. Neither represents a
# real passenger journey, so both are excluded from every indicator here.
EXCLUDED_FARE_PRODUCT_GROUPS = ["N/A", "Other Tickets"]

# --- Disclosure control ---------------------------------------------------------
# A cell (one travel_date/region/period combination) built from this many
# underlying rows OR FEWER is treated as disclosive. "Up to and including 10"
# per the confirmed rule, hence <= throughout this pipeline, not <.
#
# Two different things happen with this threshold, for two different outputs:
#   1. Every internal dataframe (daily_df, weekly_df, monthly_df, and their
#      indexed versions) gets a `low_count_flag` column marking which cells
#      are affected -- nothing in these is masked, because these are for
#      exploration *inside* the secure environment, where seeing the true
#      figures is fine and useful.
#   2. The SEPARATE "disclosure-safe" exports (section 16) have every measure
#      in a flagged cell actually replaced with a suppression marker before
#      being written out -- these, not the internal dataframes, are the
#      candidates for actually leaving the environment.
#
# This is PRIMARY suppression only (masking the small cells themselves). It
# does not implement SECONDARY/complementary suppression (additionally
# masking other, larger cells that would let a masked value be
# back-calculated from a published row/column total) -- that is a more
# involved statistical disclosure control step, out of scope for what was
# specifically requested here, and worth flagging explicitly as a known
# limitation rather than silently not doing it.
LOW_COUNT_FLAG_THRESHOLD: int = 10
DISCLOSURE_SUPPRESSION_MARKER = "c"  # ONS convention: "c" = suppressed for confidentiality

# --- Cost safety ----------------------------------------------------------------
# Refuse to actually run any single query whose dry-run estimate exceeds this
# many GB. This is a deliberate circuit breaker: if a future edit to this
# pipeline accidentally produces a much more expensive query (e.g. an
# unintended cross join, or a filter that stops pruning), this stops it before
# it runs, rather than after the bill arrives.
MAX_QUERY_GB: float = 100.0

# --- Cost estimation ------------------------------------------------------------
# BigQuery on-demand list price, used only to produce an ESTIMATED cost in the
# sanity-check output. This has never been confirmed against ONS IDS's actual
# billing model (it may be a shared slot reservation rather than per-byte
# billing — see analysis-notes.md, Phase 0) — treat every £/$ figure this
# pipeline prints as an estimate assuming on-demand pricing, not a real bill.
BIGQUERY_ON_DEMAND_USD_PER_TB: float = 6.25

# --- Output -----------------------------------------------------------------------
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 2. Cost, performance, and correctness tracking
#
# Every BigQuery call this pipeline makes goes through `run_query()` below,
# never a bare `client.query()`. That single choke point is what makes it
# possible to report, at the end of a run: how much data every step touched,
# how long it took, what it likely cost, and to refuse to run anything that
# blows past `MAX_QUERY_GB` without a human noticing.
#
# The pattern — dry-run first, then run for real — is the same discipline used
# throughout the exploratory phase of this project (see Principle 1 in
# `analysis-notes.md`), just wrapped into a reusable function instead of typed
# out by hand each time.

# %%
# ============================================================================
# 2. COST & PERFORMANCE TRACKING
# ============================================================================


@dataclass
class QueryRun:
    """A record of one query execution, kept for the end-of-run summary."""

    name: str
    rows_returned: int
    bytes_billed: int
    elapsed_seconds: float
    estimated_cost_usd: float


@dataclass
class CostTracker:
    """Accumulates every QueryRun made during this pipeline execution."""

    runs: list[QueryRun] = field(default_factory=list)

    def record(self, run: QueryRun) -> None:
        self.runs.append(run)

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "query": r.name,
                    "rows_returned": r.rows_returned,
                    "gb_billed": round(r.bytes_billed / 1024**3, 3),
                    "seconds": round(r.elapsed_seconds, 1),
                    "est_cost_usd": round(r.estimated_cost_usd, 4),
                }
                for r in self.runs
            ]
        )

    def print_summary(self) -> None:
        df = self.summary_df()
        total_gb = df["gb_billed"].sum() if not df.empty else 0.0
        total_cost = df["est_cost_usd"].sum() if not df.empty else 0.0
        total_seconds = df["seconds"].sum() if not df.empty else 0.0
        print("\n" + "=" * 72)
        print("PIPELINE COST & PERFORMANCE SUMMARY")
        print("=" * 72)
        if not df.empty:
            print(df.to_string(index=False))
        print("-" * 72)
        print(
            f"Total: {total_gb:,.3f} GB billed | "
            f"~${total_cost:,.4f} estimated (on-demand pricing, unconfirmed "
            f"billing model — see config notes) | "
            f"{total_seconds:,.1f}s query time"
        )
        print("=" * 72 + "\n")


def bytes_to_estimated_usd(num_bytes: int) -> float:
    tb = num_bytes / 1024**4
    return tb * BIGQUERY_ON_DEMAND_USD_PER_TB


def run_query(
    client: bigquery.Client,
    tracker: CostTracker,
    sql: str,
    name: str,
    query_parameters: list | None = None,
) -> pd.DataFrame:
    """
    Dry-run a query first, refuse to proceed if it's implausibly expensive,
    then run it for real, time it, and record everything in `tracker`.

    This is the single choke point every BigQuery call in this pipeline goes
    through — see the section 2 docstring above for why.
    """
    base_config = bigquery.QueryJobConfig(query_parameters=query_parameters or [])

    dry_run_config = bigquery.QueryJobConfig(
        query_parameters=query_parameters or [],
        dry_run=True,
        use_query_cache=False,
    )
    dry_run_job = client.query(sql, job_config=dry_run_config)
    estimated_gb = dry_run_job.total_bytes_processed / 1024**3
    print(f"[{name}] dry run: ~{estimated_gb:,.2f} GB estimated")

    if estimated_gb > MAX_QUERY_GB:
        raise RuntimeError(
            f"[{name}] refusing to run: estimated {estimated_gb:,.2f} GB exceeds "
            f"MAX_QUERY_GB={MAX_QUERY_GB} GB. If this is genuinely expected, raise "
            f"MAX_QUERY_GB in the config section deliberately — don't silently "
            f"bypass this."
        )

    start = time.time()
    job = client.query(sql, job_config=base_config)
    df = job.to_dataframe()
    elapsed = time.time() - start

    bytes_billed = job.total_bytes_billed or 0
    tracker.record(
        QueryRun(
            name=name,
            rows_returned=len(df),
            bytes_billed=bytes_billed,
            elapsed_seconds=elapsed,
            estimated_cost_usd=bytes_to_estimated_usd(bytes_billed),
        )
    )
    print(
        f"[{name}] done: {len(df):,} rows, "
        f"{bytes_billed / 1024**3:,.2f} GB billed, {elapsed:,.1f}s"
    )
    return df


client = bigquery.Client()
tracker = CostTracker()

# %% [markdown]
# ## 3. Sanity checks — establish what we're actually working with
#
# Before computing anything, confirm the basic shape of the data this run will
# process: how many rows, what date range, and how much of the raw table the
# exclusion filter removes. Stakeholders reviewing this pipeline want to see
# these numbers up front, not discover them by reading query results — and
# it's also the cheapest possible check that nothing has silently changed
# about the underlying data since this pipeline was last run (recall: this
# table is live and growing — see `analysis-notes.md`).

# %%
# ============================================================================
# 3. SANITY CHECKS
# ============================================================================

sanity_sql = f"""
SELECT
  COUNT(*) AS total_rows,
  MIN(DATE(collection_date)) AS earliest_travel_date,
  MAX(DATE(collection_date)) AS latest_travel_date,
  COUNTIF(pro_fpg_description IN UNNEST(@excluded_fpg)) AS rows_excluded_by_fpg_filter,
  -- Confirms the assumption this pipeline relies on: that passenger_journeys
  -- carries the same +/- sign as number_of_tickets for refund rows, so that
  -- SUM(passenger_journeys) nets refunds out the same way SUM(number_of_tickets)
  -- does. This was a reasonable but unverified assumption when this pipeline
  -- was first written — checked here on every run rather than assumed silently.
  COUNTIF(number_of_tickets = -1 AND passenger_journeys > 0) AS refund_sign_mismatch_count
FROM `{VIEW_FQN}`
"""

sanity_df = run_query(
    client,
    tracker,
    sanity_sql,
    name="sanity_checks",
    query_parameters=[
        bigquery.ArrayQueryParameter(
            "excluded_fpg", "STRING", EXCLUDED_FARE_PRODUCT_GROUPS
        ),
    ],
)
print(sanity_df.to_string(index=False))

_row = sanity_df.iloc[0]
if _row["refund_sign_mismatch_count"] > 0:
    print(
        f"\n⚠ WARNING: {_row['refund_sign_mismatch_count']:,} refund rows have a "
        f"POSITIVE passenger_journeys value. The assumption that SUM(passenger_journeys) "
        f"nets out refunds the same way SUM(number_of_tickets) does may not hold for "
        f"these rows — this pipeline's journey counts could be a small overcount. "
        f"Worth investigating before trusting the headline figures if this number is "
        f"large relative to total_rows."
    )
else:
    print("\n✓ Refund sign convention confirmed: passenger_journeys nets out correctly.")

# Resolve the actual date range to process, now that we know the data's real bounds.
_effective_start = DATE_RANGE_START or _row["earliest_travel_date"]
_effective_end = DATE_RANGE_END or _row["latest_travel_date"]
print(f"\nProcessing travel_date range: {_effective_start} to {_effective_end}")
print(
    f"Fare-product filter will exclude {_row['rows_excluded_by_fpg_filter']:,} of "
    f"{_row['total_rows']:,} rows "
    f"({100 * _row['rows_excluded_by_fpg_filter'] / _row['total_rows']:.1f}%)."
)

# %% [markdown]
# ## 4. Pull the base data — one query, daily × region grain
#
# Everything downstream (day/week/month rollups, raw and indexed views) is
# derived locally from a single BigQuery pull at the finest grain requested
# (daily, by destination region). This mirrors a discipline established
# throughout the exploratory phase of this project: minimize the number of
# full-table passes, not the number of things you learn from each one. Pulling
# three separate day/week/month queries would cost roughly three times as much
# for the same information, since week/month are just sums over day.
#
# `destination_region_code`/`destination_region_name` being NULL (16.5% of rows,
# per prior profiling) is bucketed here as an explicit "Unknown region" category
# — never silently dropped.

# %%
# ============================================================================
# 4. BASE QUERY — daily x region grain
# ============================================================================

base_sql = f"""
SELECT
  DATE(collection_date) AS travel_date,
  COALESCE(destination_region_code, 'UNKNOWN') AS destination_region_code,
  COALESCE(destination_region_name, 'Unknown region') AS destination_region_name,

  -- Primary journey-count measure. passenger_journeys is defined (confirmed
  -- against ONS's own published data dictionary) as the actual number of
  -- passenger journeys represented by the row -- not the number of tickets --
  -- so this, not number_of_tickets, is the right field for "count of
  -- passenger journeys". Refunds (number_of_tickets = -1) net out
  -- automatically via the sign convention checked in the sanity-check section.
  --
  -- Cast to FLOAT64: passenger_journeys is NUMERIC (BigQuery's exact-decimal
  -- type), which the Python client returns as decimal.Decimal -- and Decimal
  -- deliberately refuses to mix with float in arithmetic. Casting here, at
  -- the SQL boundary, avoids that TypeError recurring in every downstream
  -- calculation (weekly/monthly rollups, indexing) that divides by this
  -- column later in the pipeline.
  CAST(SUM(passenger_journeys) AS FLOAT64) AS net_passenger_journeys,

  -- Secondary measure, kept alongside for cross-validation against
  -- net_passenger_journeys -- the two should move together; if they diverge
  -- sharply in a future run, that's worth investigating before trusting either.
  SUM(number_of_tickets) AS net_tickets,

  -- Numerator for the passenger-weighted average journey length.
  SUM(ticket_miles * passenger_journeys) AS total_passenger_miles,

  -- Numerator for the unweighted average journey length -- summed here, not
  -- averaged, specifically so this can be correctly re-summed when rolling
  -- daily figures up to weekly/monthly. An AVG() can't be correctly
  -- re-averaged across days without knowing how many rows each day
  -- represents; a SUM can be re-summed at any grain and divided by the
  -- summed row_count (see roll_up() below). Reported alongside the weighted
  -- figure, both clearly labeled, per the agreed methodology (see
  -- documentation).
  SUM(ticket_miles) AS total_ticket_miles_unweighted,

  -- Provisional commute/leisure split by distance -- see
  -- COMMUTE_DISTANCE_THRESHOLD_MILES in the config section. Cast to FLOAT64
  -- for the same reason as net_passenger_journeys above.
  CAST(SUM(IF(ticket_miles <= @commute_threshold, passenger_journeys, 0)) AS FLOAT64) AS commute_like_journeys,
  CAST(SUM(IF(ticket_miles > @commute_threshold, passenger_journeys, 0)) AS FLOAT64) AS leisure_like_journeys,

  -- Raw row count per cell -- used below to flag (not mask) low-count cells
  -- for downstream statistical disclosure control review.
  COUNT(*) AS row_count

FROM `{VIEW_FQN}`
WHERE pro_fpg_description NOT IN UNNEST(@excluded_fpg)
  AND DATE(collection_date) BETWEEN @start_date AND @end_date
GROUP BY travel_date, destination_region_code, destination_region_name
ORDER BY travel_date, destination_region_code
"""

base_df = run_query(
    client,
    tracker,
    base_sql,
    name="base_daily_by_region",
    query_parameters=[
        bigquery.ScalarQueryParameter(
            "commute_threshold", "FLOAT64", COMMUTE_DISTANCE_THRESHOLD_MILES
        ),
        bigquery.ArrayQueryParameter(
            "excluded_fpg", "STRING", EXCLUDED_FARE_PRODUCT_GROUPS
        ),
        bigquery.ScalarQueryParameter("start_date", "DATE", _effective_start),
        bigquery.ScalarQueryParameter("end_date", "DATE", _effective_end),
    ],
)

# Weighted average computed locally rather than in SQL, since it's a simple
# division of two already-summed columns -- no reason to pay for BigQuery
# compute to do arithmetic pandas can do for free on data already in memory.
base_df["avg_journey_miles_weighted"] = (
    base_df["total_passenger_miles"] / base_df["net_passenger_journeys"]
)
base_df["avg_journey_miles_unweighted"] = (
    base_df["total_ticket_miles_unweighted"] / base_df["row_count"]
)
base_df["low_count_flag"] = base_df["row_count"] <= LOW_COUNT_FLAG_THRESHOLD

print(f"\nBase table: {len(base_df):,} rows (travel_date x region)")
print(
    f"{base_df['low_count_flag'].sum():,} of these are flagged as low-count "
    f"(row_count <= {LOW_COUNT_FLAG_THRESHOLD}). This dataframe itself is NOT "
    f"masked -- it's for exploration inside the secure environment. The "
    f"disclosure-safe, actually-masked exports are built separately in "
    f"section 16."
)

# %% [markdown]
# ## 5. Roll up to week and month, all done locally
#
# ISO weeks (Monday-start) are used throughout, matching standard UK/European
# statistical convention rather than a Sunday-start week.

# %%
# ============================================================================
# 5. WEEKLY / MONTHLY ROLLUPS
# ============================================================================

_measure_cols = [
    "net_passenger_journeys",
    "net_tickets",
    "total_passenger_miles",
    "total_ticket_miles_unweighted",
    "commute_like_journeys",
    "leisure_like_journeys",
    "row_count",
]


def roll_up(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby([period_col, "destination_region_code", "destination_region_name"])[
            _measure_cols
        ]
        .sum()
        .reset_index()
    )
    grouped["avg_journey_miles_weighted"] = (
        grouped["total_passenger_miles"] / grouped["net_passenger_journeys"]
    )
    # Correctly re-derived, not a naive average-of-averages -- see the
    # total_ticket_miles_unweighted comment in section 4 for why that matters.
    grouped["avg_journey_miles_unweighted"] = (
        grouped["total_ticket_miles_unweighted"] / grouped["row_count"]
    )
    grouped["low_count_flag"] = grouped["row_count"] <= LOW_COUNT_FLAG_THRESHOLD
    return grouped


daily_df = base_df.copy()

weekly_df = base_df.copy()
weekly_df["iso_week"] = pd.to_datetime(weekly_df["travel_date"]).dt.strftime("%G-W%V")
weekly_df = roll_up(weekly_df, "iso_week")

monthly_df = base_df.copy()
monthly_df["month"] = pd.to_datetime(monthly_df["travel_date"]).dt.strftime("%Y-%m")
monthly_df = roll_up(monthly_df, "month")

print(f"Daily rows: {len(daily_df):,} | Weekly rows: {len(weekly_df):,} | Monthly rows: {len(monthly_df):,}")

# %% [markdown]
# ## 6. Index the weekly and monthly series (base period = 100)
#
# Follows the same convention ONS's own Prices Division paper on this exact
# dataset uses for its own rail fares index (e.g. "Jan 2019 = 100") — each
# region's series is indexed independently, against its own value in the base
# period, so regional comparisons are about relative *change*, not absolute
# journey volume.

# %%
# ============================================================================
# 6. INDEXING
# ============================================================================


def index_series(df: pd.DataFrame, period_col: str, base_period: str | None) -> pd.DataFrame:
    df = df.copy()
    resolved_base = base_period or sorted(df[period_col].unique())[0]

    base_values = (
        df[df[period_col] == resolved_base]
        .set_index("destination_region_code")["net_passenger_journeys"]
    )

    def _index(row):
        base = base_values.get(row["destination_region_code"])
        if not base:
            return None
        return 100.0 * row["net_passenger_journeys"] / base

    df["index_base_period"] = resolved_base
    df["journeys_index"] = df.apply(_index, axis=1)
    return df


weekly_indexed_df = index_series(weekly_df, "iso_week", INDEX_BASE_PERIOD)
monthly_indexed_df = index_series(monthly_df, "month", INDEX_BASE_PERIOD)

print(f"Weekly index base period: {weekly_indexed_df['index_base_period'].iloc[0]}")
print(f"Monthly index base period: {monthly_indexed_df['index_base_period'].iloc[0]}")

# %% [markdown]
# ## 7. Export
#
# Five files: raw figures at all three granularities, plus indexed figures at
# the two granularities that are actually meaningful to index (a daily index
# would be dominated by day-of-week noise rather than showing a genuine trend).

# %%
# ============================================================================
# 7. EXPORT
# ============================================================================

_exports = {
    "busyness_indicator_raw_daily.csv": daily_df,
    "busyness_indicator_raw_weekly.csv": weekly_df,
    "busyness_indicator_raw_monthly.csv": monthly_df,
    "busyness_indicator_indexed_weekly.csv": weekly_indexed_df,
    "busyness_indicator_indexed_monthly.csv": monthly_indexed_df,
}

for filename, df in _exports.items():
    out_path = OUTPUT_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df):,} rows)")

# %% [markdown]
# ## 8. Final summary
#
# Everything a reviewer would want to see at a glance: how much data this run
# touched, how long it took, what it likely cost, and how many rows didn't
# make it into the output because of the fare-product filter or the
# unknown-region bucket.

# %%
# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================

print(f"\nRun completed: {datetime.now().isoformat(timespec='seconds')}")
print(f"Travel date range processed: {_effective_start} to {_effective_end}")
print(f"Commute/leisure threshold used: {COMMUTE_DISTANCE_THRESHOLD_MILES} miles (provisional)")
tracker.print_summary()

# %% [markdown]
# ## 9. Exploring the output visually
#
# Everything below reads from `daily_df`/`weekly_df`/`monthly_df`/`*_indexed_df`,
# which already exist in memory from sections 4-6 -- **none of this costs anything
# further in BigQuery terms.** No files are written to disk here either: since this
# environment doesn't allow exporting data out directly, every chart is rendered
# inline in the notebook, which is a normal, safe way to explore data in a
# disclosure-controlled environment (viewing a chart on screen is not the same as
# extracting a file from it).
#
# Uses plain `matplotlib` throughout rather than a fancier plotting library (e.g.
# seaborn) — this environment's exact set of installed packages isn't something to
# assume, and matplotlib is the one plotting library virtually guaranteed to already
# be present anywhere pandas is.

# %%
# ============================================================================
# 9. VISUALIZATION SETUP
# ============================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# %% [markdown]
# ## 10. Indicator 1 — overall journey trend, monthly
#
# Monthly, not daily or weekly, for the headline trend chart -- daily data is noisy
# enough (day-of-week effects alone dwarf most genuine trend movements) that a daily
# line would visually bury the actual signal.

# %%
# ============================================================================
# 10. OVERALL JOURNEY TREND (MONTHLY)
# ============================================================================

overall_monthly = (
    monthly_df.groupby("month")["net_passenger_journeys"].sum().reset_index()
)

fig, ax = plt.subplots()
ax.plot(overall_monthly["month"], overall_monthly["net_passenger_journeys"], marker="o", markersize=3)
ax.set_title("Net passenger journeys by month (all regions combined)")
ax.set_xlabel("Month")
ax.set_ylabel("Net passenger journeys")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
# Monthly labels get unreadable over a multi-year range -- show roughly one per year.
ax.set_xticks(overall_monthly["month"][::12])
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Indicator 2 — regional comparison
#
# Two views: total volume over the whole period (which regions are simply larger),
# and a time series limited to the largest few regions plus an "Other" bucket for
# everything else (showing all 14 region categories as separate lines would be
# unreadable clutter).

# %%
# ============================================================================
# 11. REGIONAL COMPARISON
# ============================================================================

regional_totals = (
    monthly_df.groupby("destination_region_name")["net_passenger_journeys"]
    .sum()
    .sort_values(ascending=True)  # ascending so the horizontal bar chart reads top-to-bottom as largest-to-smallest
)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#c0392b" if name == "Unknown region" else "#2c7fb8" for name in regional_totals.index]
ax.barh(regional_totals.index, regional_totals.values, color=colors)
ax.set_title("Total net passenger journeys by destination region (whole period)")
ax.set_xlabel("Net passenger journeys")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.show()
print(
    "Red bar = the 'Unknown region' bucket (rows with no destination_region assigned) "
    "-- shown deliberately alongside the rest, not hidden."
)

# Top 6 regions by volume, individually; everything else collapsed into "Other".
_top_n = 6
top_region_names = regional_totals.sort_values(ascending=False).head(_top_n).index.tolist()

_regional_ts = monthly_df.copy()
_regional_ts["region_group"] = _regional_ts["destination_region_name"].where(
    _regional_ts["destination_region_name"].isin(top_region_names), "Other regions (combined)"
)
regional_pivot = (
    _regional_ts.groupby(["month", "region_group"])["net_passenger_journeys"]
    .sum()
    .unstack("region_group")
)

fig, ax = plt.subplots()
regional_pivot.plot(ax=ax)
ax.set_title(f"Net passenger journeys by month, top {_top_n} regions + Other")
ax.set_xlabel("Month")
ax.set_ylabel("Net passenger journeys")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Indicator 3a — average journey length, weighted vs. unweighted
#
# Both lines plotted together specifically so the gap between them is visible -- if
# they move apart over time, that itself is informative (e.g. group bookings
# becoming more or less common relative to solo travel).

# %%
# ============================================================================
# 12. AVERAGE JOURNEY LENGTH -- WEIGHTED VS UNWEIGHTED
# ============================================================================

_avg_length_monthly = monthly_df.groupby("month").apply(
    lambda g: pd.Series(
        {
            "avg_journey_miles_weighted": g["total_passenger_miles"].sum() / g["net_passenger_journeys"].sum(),
            "avg_journey_miles_unweighted": g["total_ticket_miles_unweighted"].sum() / g["row_count"].sum(),
        }
    )
).reset_index()

fig, ax = plt.subplots()
ax.plot(_avg_length_monthly["month"], _avg_length_monthly["avg_journey_miles_weighted"], label="Weighted (per passenger journey)", marker="o", markersize=3)
ax.plot(_avg_length_monthly["month"], _avg_length_monthly["avg_journey_miles_unweighted"], label="Unweighted (per ticket product)", marker="o", markersize=3)
ax.set_title("Average journey length by month -- weighted vs. unweighted")
ax.set_xlabel("Month")
ax.set_ylabel("Average miles")
ax.set_xticks(_avg_length_monthly["month"][::12])
ax.tick_params(axis="x", rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Indicator 3b — commute-like vs. leisure-like journey mix
#
# A 100%-stacked view (share of journeys, not raw counts) is the clearer chart for
# this specific question -- "has the *mix* of journey types changed" is a different
# question from "has total volume changed", and a raw stacked-count chart conflates
# the two. Plotted separately, raw counts, right after for anyone who wants the
# absolute-volume view too.

# %%
# ============================================================================
# 13. COMMUTE-LIKE VS. LEISURE-LIKE JOURNEY MIX
# ============================================================================

_split_monthly = (
    monthly_df.groupby("month")[["commute_like_journeys", "leisure_like_journeys"]]
    .sum()
    .reset_index()
)
_split_monthly["total"] = _split_monthly["commute_like_journeys"] + _split_monthly["leisure_like_journeys"]
_split_monthly["commute_share"] = _split_monthly["commute_like_journeys"] / _split_monthly["total"]
_split_monthly["leisure_share"] = _split_monthly["leisure_like_journeys"] / _split_monthly["total"]

fig, ax = plt.subplots()
ax.stackplot(
    _split_monthly["month"],
    _split_monthly["commute_share"],
    _split_monthly["leisure_share"],
    labels=[
        f"Commute-like (<= {COMMUTE_DISTANCE_THRESHOLD_MILES:.0f} miles)",
        f"Leisure-like (> {COMMUTE_DISTANCE_THRESHOLD_MILES:.0f} miles)",
    ],
)
ax.set_title("Share of journeys by distance-based type, per month (provisional threshold)")
ax.set_xlabel("Month")
ax.set_ylabel("Share of journeys")
ax.set_ylim(0, 1)
ax.set_xticks(_split_monthly["month"][::12])
ax.tick_params(axis="x", rotation=45)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.stackplot(
    _split_monthly["month"],
    _split_monthly["commute_like_journeys"],
    _split_monthly["leisure_like_journeys"],
    labels=["Commute-like", "Leisure-like"],
)
ax.set_title("Journey counts by distance-based type, per month (raw volumes)")
ax.set_xlabel("Month")
ax.set_ylabel("Net passenger journeys")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_xticks(_split_monthly["month"][::12])
ax.tick_params(axis="x", rotation=45)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Indexed regional comparison (base period = 100)
#
# The same style as ONS's own published rail fares index chart for this dataset
# (their Figure 3: regional lines, all starting near 100 at the base period) --
# reusing a chart convention a reader of the eventual publication will likely
# already recognize, rather than inventing a new one.

# %%
# ============================================================================
# 14. INDEXED REGIONAL COMPARISON
# ============================================================================

indexed_pivot = monthly_indexed_df.pivot_table(
    index="month", columns="destination_region_name", values="journeys_index"
)

fig, ax = plt.subplots(figsize=(11, 6))
indexed_pivot.plot(ax=ax)
ax.axhline(100, color="black", linewidth=0.8, linestyle="--")
ax.set_title(
    f"Journey volume index by region, {monthly_indexed_df['index_base_period'].iloc[0]} = 100"
)
ax.set_xlabel("Month")
ax.set_ylabel("Index")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 15. Data-quality check — is the "Unknown region" share stable or growing?
#
# A quick visual sanity check, not one of the three requested indicators: if the
# 16.5% unknown-region rate is concentrated in a particular period rather than
# spread evenly, that's worth knowing before reporting a flat "16.5%" figure as if
# it applied uniformly across the whole time range.

# %%
# ============================================================================
# 15. DATA QUALITY -- UNKNOWN REGION SHARE OVER TIME
# ============================================================================

_unknown_share = monthly_df.copy()
_unknown_share["is_unknown"] = _unknown_share["destination_region_name"] == "Unknown region"
_unknown_monthly = (
    _unknown_share.groupby("month")
    .apply(lambda g: g.loc[g["is_unknown"], "net_passenger_journeys"].sum() / g["net_passenger_journeys"].sum())
    .reset_index(name="unknown_share")
)

fig, ax = plt.subplots()
ax.plot(_unknown_monthly["month"], _unknown_monthly["unknown_share"], marker="o", markersize=3, color="#c0392b")
ax.set_title('Share of journeys with "Unknown region", by month')
ax.set_xlabel("Month")
ax.set_ylabel("Share of journeys")
ax.set_ylim(0, max(0.3, _unknown_monthly["unknown_share"].max() * 1.2))
ax.set_xticks(_unknown_monthly["month"][::12])
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 16. Disclosure-safe exports — the actual masked output
#
# Everything above (`daily_df`, `weekly_df`, `monthly_df`, and the indexed
# versions) stays fully unmasked — deliberately, since those exist for
# exploration *inside* this secure environment, where seeing the true figures
# is the whole point. This section builds SEPARATE, masked copies. **These,
# not the dataframes above, are the actual candidates for requesting export
# out of the environment.**
#
# Masking rule (confirmed): any cell built from `LOW_COUNT_FLAG_THRESHOLD`
# (10) or FEWER underlying rows has every measure replaced with
# `DISCLOSURE_SUPPRESSION_MARKER` ("c") — `<=`, matching "up to and including
# 10" exactly, not `<`. This is **primary suppression only** — see the
# config section for what that does and doesn't cover (no protection against
# back-calculating a suppressed cell from a published total).

# %%
# ============================================================================
# 16. DISCLOSURE-SAFE EXPORTS
# ============================================================================

_RAW_MASK_COLUMNS = [
    "net_passenger_journeys",
    "net_tickets",
    "total_passenger_miles",
    "total_ticket_miles_unweighted",
    "commute_like_journeys",
    "leisure_like_journeys",
    "avg_journey_miles_weighted",
    "avg_journey_miles_unweighted",
    "row_count",
]

_INDEXED_MASK_COLUMNS = _RAW_MASK_COLUMNS + ["journeys_index"]


def mask_for_disclosure(df: pd.DataFrame, measure_cols: list[str]) -> pd.DataFrame:
    """
    Returns a NEW dataframe (never mutates df) where every column in
    measure_cols is replaced with DISCLOSURE_SUPPRESSION_MARKER for any row
    where low_count_flag is True. Adds a disclosure_status column stating
    plainly which rows were suppressed and why, so a reader never has to
    guess whether "c" means suppressed data or a literal entry in the data.
    """
    masked = df.copy()
    masked["disclosure_status"] = masked["low_count_flag"].map(
        {True: f"Suppressed (underlying rows <= {LOW_COUNT_FLAG_THRESHOLD})", False: "OK"}
    )
    for col in measure_cols:
        masked[col] = masked[col].where(~masked["low_count_flag"], DISCLOSURE_SUPPRESSION_MARKER)
    return masked


weekly_disclosure_safe_df = mask_for_disclosure(weekly_df, _RAW_MASK_COLUMNS)
monthly_disclosure_safe_df = mask_for_disclosure(monthly_df, _RAW_MASK_COLUMNS)
weekly_indexed_disclosure_safe_df = mask_for_disclosure(weekly_indexed_df, _INDEXED_MASK_COLUMNS)
monthly_indexed_disclosure_safe_df = mask_for_disclosure(monthly_indexed_df, _INDEXED_MASK_COLUMNS)

_disclosure_exports = {
    "busyness_indicator_DISCLOSURE_SAFE_weekly.csv": weekly_disclosure_safe_df,
    "busyness_indicator_DISCLOSURE_SAFE_monthly.csv": monthly_disclosure_safe_df,
    "busyness_indicator_DISCLOSURE_SAFE_indexed_weekly.csv": weekly_indexed_disclosure_safe_df,
    "busyness_indicator_DISCLOSURE_SAFE_indexed_monthly.csv": monthly_indexed_disclosure_safe_df,
}

for filename, df in _disclosure_exports.items():
    out_path = OUTPUT_DIR / filename
    df.to_csv(out_path, index=False)
    n_suppressed = (df["disclosure_status"] != "OK").sum()
    print(f"Wrote {out_path} ({len(df):,} rows, {n_suppressed:,} suppressed cells)")

print(
    "\nDaily-grain figures are NOT exported in disclosure-safe form -- at daily x "
    "region grain, small cells are the norm rather than the exception, so a daily "
    "disclosure-safe export would likely be mostly suppressed and of limited use. "
    "Daily data stays internal-only, for exploration inside this environment."
)

# %% [markdown]
# ## 17. Time-of-day pattern
#
# A genuinely new dimension, not part of the original three indicators — added
# because a time-of-day view is exactly what a stakeholder looking at "railway
# busyness" would expect to see, and it comes with a real limitation worth
# stating plainly before showing it.
#
# **The honest caveat, upfront**: no field in this data directly records what
# time of day someone traveled. The closest available signal is
# `issuing_datetime`'s time-of-day component — but that's when a ticket was
# ISSUED, not when it was USED, and for advance-purchased tickets (bought
# days or weeks ahead) those two times can be completely unrelated.
#
# This section restricts to the subset where issue time genuinely is a good
# proxy for travel time: rows where `DATE(issuing_datetime)` equals the
# travel date itself (`collection_date`) — tickets issued the same day they
# were used. Earlier profiling in this project (see `analysis-notes.md`)
# already established that this same-day-issued group behaves very
# differently by fulfilment method (near-100% same-day for EMV/PAYG/Oyster/
# Self Print; well under half for e-Ticket/M-ticket) — restricting to it is
# what makes the resulting hour-of-day pattern trustworthy, at the direct
# cost of excluding advance-purchased journeys entirely. **This describes
# "same-day travel," not "all travel"** — a genuine scope narrowing, not a
# footnote.

# %%
# ============================================================================
# 17. TIME-OF-DAY PATTERN
# ============================================================================

time_of_day_sql = f"""
SELECT
  EXTRACT(DAYOFWEEK FROM DATE(collection_date)) AS day_of_week_num,  -- 1=Sunday .. 7=Saturday (BigQuery convention)
  EXTRACT(HOUR FROM issuing_datetime) AS hour_of_day,
  CAST(SUM(passenger_journeys) AS FLOAT64) AS net_passenger_journeys,
  COUNT(*) AS row_count
FROM `{VIEW_FQN}`
WHERE pro_fpg_description NOT IN UNNEST(@excluded_fpg)
  AND DATE(collection_date) BETWEEN @start_date AND @end_date
  -- Restrict to same-day-issued tickets -- see the markdown cell above for why.
  AND DATE(issuing_datetime) = DATE(collection_date)
GROUP BY day_of_week_num, hour_of_day
ORDER BY day_of_week_num, hour_of_day
"""

time_of_day_df = run_query(
    client,
    tracker,
    time_of_day_sql,
    name="time_of_day_pattern",
    query_parameters=[
        bigquery.ArrayQueryParameter("excluded_fpg", "STRING", EXCLUDED_FARE_PRODUCT_GROUPS),
        bigquery.ScalarQueryParameter("start_date", "DATE", _effective_start),
        bigquery.ScalarQueryParameter("end_date", "DATE", _effective_end),
    ],
)

_day_names = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
time_of_day_df["day_of_week"] = time_of_day_df["day_of_week_num"].map(_day_names)
time_of_day_df["low_count_flag"] = time_of_day_df["row_count"] <= LOW_COUNT_FLAG_THRESHOLD

print(f"Time-of-day table: {len(time_of_day_df):,} rows (day-of-week x hour-of-day)")
print(
    f"Covers {time_of_day_df['net_passenger_journeys'].sum():,.0f} same-day-issued "
    f"passenger journeys -- a subset of the full total, not all rail travel."
)

# %% [markdown]
# ## 18. Time-of-day heatmap
#
# Day-of-week on the vertical axis, hour-of-day on the horizontal — the
# classic "when do people travel" view. Ordered Monday-first rather than
# BigQuery's native Sunday-first numbering, since that reads more naturally
# for a UK audience.

# %%
# ============================================================================
# 18. TIME-OF-DAY HEATMAP
# ============================================================================

_day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

heatmap_data = time_of_day_df.pivot_table(
    index="day_of_week", columns="hour_of_day", values="net_passenger_journeys", fill_value=0
).reindex(_day_order)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns)
ax.set_xlabel("Hour of day (24h)")
ax.set_title(
    "Same-day-issued passenger journeys by day of week and hour\n"
    "(excludes advance-purchased journeys -- see section 17 caveat)"
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Net passenger journeys")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 19. Interactive, self-contained exploration file
#
# A single HTML file a stakeholder can open by double-clicking — no server, no
# notebook, no code — with hoverable, zoomable charts they can genuinely
# "play with": the monthly trend, the regional breakdown, the commute/leisure
# mix, and the indexed regional comparison, all in one file.
#
# Uses `plotly`, which isn't guaranteed to be installed in every environment
# the way matplotlib is — this section detects that and prints a clear
# message rather than crashing the notebook if it's missing, so everything
# else still runs either way.
#
# **Disclosure note**: this file is built from the same UNMASKED dataframes
# as the rest of section 9 onward — it's an internal exploration tool, not
# the disclosure-safe output. If an HTML version of the disclosure-safe
# figures is ever needed, build it from the `*_disclosure_safe_df`
# dataframes in section 16 instead, following the same pattern below.

# %%
# ============================================================================
# 19. INTERACTIVE HTML EXPORT
# ============================================================================

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    print(
        "plotly is not installed in this environment -- skipping the interactive "
        "HTML export. Everything else in this notebook (CSVs, matplotlib charts) "
        "is unaffected."
    )

if _PLOTLY_AVAILABLE:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Net passenger journeys by month",
            "Top regions by month",
            "Commute-like vs. leisure-like share",
            "Journey volume index by region (base = 100)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=overall_monthly["month"], y=overall_monthly["net_passenger_journeys"], mode="lines+markers", name="All regions"),
        row=1, col=1,
    )

    for region in regional_pivot.columns:
        fig.add_trace(
            go.Scatter(x=regional_pivot.index, y=regional_pivot[region], mode="lines", name=region, legendgroup="regions"),
            row=1, col=2,
        )

    fig.add_trace(
        go.Scatter(x=_split_monthly["month"], y=_split_monthly["commute_share"], mode="lines", name="Commute-like share", stackgroup="mix"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=_split_monthly["month"], y=_split_monthly["leisure_share"], mode="lines", name="Leisure-like share", stackgroup="mix"),
        row=2, col=1,
    )

    for region in indexed_pivot.columns:
        fig.add_trace(
            go.Scatter(x=indexed_pivot.index, y=indexed_pivot[region], mode="lines", name=region, legendgroup="index"),
            row=2, col=2,
        )

    fig.update_layout(
        height=900,
        width=1400,
        title_text=(
            "GB Rail Busyness Indicator -- Interactive Exploration "
            "(internal use, not disclosure-safe -- see section 16 for the masked exports)"
        ),
        hovermode="x unified",
    )

    _html_path = OUTPUT_DIR / "busyness_indicator_interactive_exploration.html"
    fig.write_html(str(_html_path))
    print(f"Wrote {_html_path} -- open directly in a browser, no server needed.")
    print(
        "Every chart supports hover-to-see-values, zoom (click-drag), pan, and "
        "toggling individual series on/off by clicking their legend entry."
    )
