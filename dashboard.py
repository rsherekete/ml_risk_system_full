"""Interactive, read-only dashboard for daily A/B-book model research."""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.routing_model import fit_routing_model, score_routing_quality
from trading_data.bigquery_data_client import (
    BQ_DATASET_FOR_DATABASE,
    DEMO_DATABASES,
    BigQueryDataClient,
    cached_trade_records,
    is_demo_database,
)

#: Walk-forward settings for the rich model. 20 days is the minimum history
#: before anything is scored out of sample; refitting every 5 days rather than
#: daily costs nothing measurable in AUC and cuts fitting work ~5x.
MIN_TRAIN_DAYS = 20
REFIT_EVERY_DAYS = 5

from trading_data.research import (
    Economics,
    Policy,
    account_group_lookup,
    add_provenance,
    attach_symbol_specs,
    classifier_performance,
    clients_from_yaml,
    client_intelligence,
    coupled_scenario,
    current_day_account_performance,
    daily_account_features,
    firm_risk_timeseries,
    fit_oos_expected_value,
    fit_oos_predictions,
    is_realised_trade,
    realised_trade_pnl,
    model_scorecard,
    non_client_logins,
    platform_for_database,
    predict_live_expected_value,
    recent_account_features,
    recommendation,
    risk_appetite_from_weights,
    rolling_symbol_exposure,
    supervised_dataset,
    symbol_mapping_table,
)
from trading_data.risk import (
    daily_account_exposure,
    daily_symbol_drivers,
    portfolio_var,
    symbol_var,
)
from trading_data.book_assignment import (
    assign_books,
    daily_intelligence_components,
    expanding_account_pnl_volatility,
    expanding_client_profile,
    firm_daily_pnl,
    firm_daily_pnl_fixed_book,
    real_efficient_frontier,
)

# --- visual system -----------------------------------------------------------
# A validated categorical/status palette (see the dataviz skill's
# references/palette.md), dark-surface steps -- `.streamlit/config.toml` sets
# a dark theme by default, so these are the palette's dark-mode hexes, applied
# consistently across every chart in this dashboard rather than each tab
# picking its own colors.
COLOR_BLUE = "#3987e5"      # primary series / A-book
COLOR_ORANGE = "#d95926"    # secondary series / B-book
COLOR_AQUA = "#199e70"      # tertiary series
COLOR_VIOLET = "#9085e9"    # quaternary series
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
COLOR_SERIOUS = "#ec835a"
COLOR_CRITICAL = "#d03b3b"
COLOR_MUTED = "#898781"

st.set_page_config(page_title="A/B-book research", layout="wide", page_icon="ðŸ“Š")
st.markdown(
    """
    <style>
    :root { --accent: #3987e5; --accent-glow: rgba(57, 135, 229, 0.30); }
    [data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
        font-family: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        text-shadow: 0 0 14px var(--accent-glow);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
        font-size: 0.76rem; opacity: 0.75;
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }
    h1 { letter-spacing: 0.02em; }
    h2, h3, h4 { border-bottom: 1px solid var(--accent-glow); padding-bottom: 0.35rem; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
    div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("ðŸ“Š Daily A/B-book research")
st.caption("Read-only decision support. Proxy economics require venue hedge and cost data before operational use.")


def _kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """A row of bordered, card-style KPI tiles instead of bare `st.metric` calls."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col.container(border=True):
            st.metric(label, value, delta=delta)


def _bar_chart(data: pd.DataFrame, x: str, y: str, color: str = COLOR_BLUE, y_title: str | None = None, sort: str = "-y") -> alt.Chart:
    return alt.Chart(data).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{x}:N", sort=sort, title=None),
        y=alt.Y(f"{y}:Q", title=y_title or y),
        tooltip=[x, alt.Tooltip(y, format=",.0f")],
    )


def _line_chart(data: pd.DataFrame, x: str, y_cols: list[str], colors: list[str], y_title: str | None = None) -> alt.Chart:
    long = data.melt(id_vars=[x], value_vars=y_cols, var_name="series", value_name="value")
    return alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X(f"{x}:T", title=None),
        y=alt.Y("value:Q", title=y_title or None),
        color=alt.Color("series:N", scale=alt.Scale(domain=y_cols, range=colors[:len(y_cols)]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[x, "series", alt.Tooltip("value:Q", format=",.0f")],
    )


with st.sidebar:
    st.subheader("Data source")
    data_source = st.radio(
        "Read trade records from",
        ["Auto (MySQL, fall back to BigQuery)", "MySQL only", "BigQuery only"],
        index=0,
        help=(
            "MySQL is the live on-prem source of truth. BigQuery reads the "
            "`zfx-dwh-prod` warehouse mirror instead -- use it when the MySQL "
            "proxy is unreachable. Auto tries MySQL per database and silently "
            "falls back to BigQuery for any that fail, so a partial outage "
            "still yields a complete book."
        ),
    )
    use_bq_cache = st.checkbox(
        "Cache BigQuery locally", value=True,
        help=(
            "Pull the full window once, then re-query only the last few days "
            "and merge with the local Parquet cache. Cuts BigQuery cost to "
            "roughly a cent a day after the first load."
        ),
    )
    bq_overlap_days = st.slider(
        "BigQuery refresh overlap (days)", 1, 7, 3,
        help=(
            "How many recent days to re-pull each load. traderecord is a "
            "change log -- a trade opened before the last pull can gain new "
            "rows after it, so a pure tail-append would miss them. 2-3 days "
            "is the safe default."
        ),
    )

    account_universe_choice = st.radio(
        "Account universe",
        ["Live accounts only (recommended)", "Live + demo accounts"],
        index=0,
        help=(
            "Demo accounts trade simulated money. The firm never hedges demo flow with an "
            "LP and never takes the other side of it, so counting demo results as firm P&L "
            "invents revenue that does not exist, overstates VaR and notional exposure, and "
            "trains the loss-probability / expected-value models on traders who behave "
            "nothing like real clients. Include them only to exercise the pipeline, never "
            f"to read real economics. Demo servers: {', '.join(sorted(DEMO_DATABASES))}."
        ),
    )
    include_demo = account_universe_choice == "Live + demo accounts"

    st.divider()
    st.subheader("Risk-return preference")
    st.caption(
        "One coupled control, not two independent ones: moving either slider "
        "moves the other (they always sum to 100), and together they select "
        "one point on the empirical profit/drawdown frontier -- see the "
        "**Efficient frontier** tab for the curve this point sits on, and the "
        "**Book assignment** tab for the actual firm P&L it implies."
    )
    if "profit_weight" not in st.session_state:
        st.session_state["profit_weight"] = 70
        st.session_state["drawdown_weight"] = 30

    def _sync_from_profit() -> None:
        st.session_state["drawdown_weight"] = 100 - st.session_state["profit_weight"]

    def _sync_from_drawdown() -> None:
        st.session_state["profit_weight"] = 100 - st.session_state["drawdown_weight"]

    st.slider("Profit maximisation", 0, 100, key="profit_weight", on_change=_sync_from_profit)
    st.slider("Drawdown minimisation", 0, 100, key="drawdown_weight", on_change=_sync_from_drawdown)
    profit_weight = st.session_state["profit_weight"]
    drawdown_weight = st.session_state["drawdown_weight"]

    var_confidence = st.selectbox("VaR confidence", [0.90, 0.95, 0.975, 0.99], index=1, format_func=lambda v: f"{v:.1%}")
    top_n_accounts = st.slider("Top N accounts / symbols shown", 5, 50, 15)

    with st.form("scenario_controls"):
        lookback_days = st.slider("History (days)", 30, 365, 90)
        analysis_day = st.date_input("Analysis day", value=pd.Timestamp.now(tz="UTC").date())
        activity_window_days = st.slider("Account activity window (days)", 1, 30, 7)
        min_observations = st.slider("Minimum observations", 1, 100, 20)
        confidence_floor = st.slider("Confidence floor", 0.0, 1.0, 0.0, 0.05)
        regions = st.multiselect("Regions", ["global", "dubai"], default=["global", "dubai"])
        markup_bps = st.number_input("A-book markup (bps)", min_value=0.0, value=2.0, step=0.1)
        lp_cost_bps = st.number_input("LP cost (bps)", min_value=0.0, value=0.5, step=0.1)
        slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=0.5, step=0.1)
        st.form_submit_button("Apply scenario")

    st.divider()
    compare_enabled = st.checkbox("Compare to another day")
    # `key=` gives this widget a stable identity across reruns -- without it,
    # `value=` (derived from `analysis_day`) is hashed into the widget's ID,
    # so changing the analysis day would silently create a "new" widget and
    # discard whatever comparison date the user had already picked.
    compare_day = st.date_input(
        "Compare day", value=pd.Timestamp(analysis_day) - pd.Timedelta(days=7), key="compare_day",
    ) if compare_enabled else None


def _fetch_from_bigquery(database: str, start: pd.Timestamp, end: pd.Timestamp, use_cache: bool, overlap_days: int) -> pd.DataFrame:
    """One database's trade records from the BigQuery warehouse mirror."""
    bq_client = BigQueryDataClient(database)
    if use_cache:
        return cached_trade_records(bq_client, start.to_pydatetime(), end.to_pydatetime(), overlap_days=overlap_days)
    return bq_client.get_trade_records(start.to_pydatetime(), end.to_pydatetime())


@st.cache_data(ttl=300)
def load_records(
    start: pd.Timestamp, end: pd.Timestamp, source: str = "Auto (MySQL, fall back to BigQuery)",
    use_cache: bool = True, overlap_days: int = 3, include_demo: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch every database's trade records and drop known internal test accounts.

    "Accounts" here means *any* order activity in the window (opens, modifies,
    closes -- including a position still open with no realised outcome yet),
    not just closed/realised trades; a back-office system that only counts
    closed trades will show meaningfully fewer accounts for the same period.
    That gap is expected, not a sign of a data-pulling bug -- see the "All
    known accounts" caption. What genuinely doesn't belong in this count is
    internal test accounts, which is what `excluded` reports.

    `source` picks MySQL (the live source of truth), BigQuery (the
    `zfx-dwh-prod` warehouse mirror), or Auto -- which tries MySQL per
    database and falls back to BigQuery only for the ones that fail, so a
    partial MySQL outage still produces a complete book rather than a silently
    short one. The returned `provenance` frame records which source each
    database actually came from: never guess whether a number is live or
    mirrored, read it there.
    """
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    excluded: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []

    mysql_clients: dict[str, Any] = {}
    if source != "BigQuery only":
        try:
            mysql_clients = clients_from_yaml()
        except Exception as exc:
            failures.append({"database": "(config)", "error": f"could not build MySQL clients: {type(exc).__name__}: {exc}"})

    databases = sorted(set(mysql_clients) | set(BQ_DATASET_FOR_DATABASE))
    if not include_demo:
        databases = [database for database in databases if not is_demo_database(database)]
    for database in databases:
        frame: pd.DataFrame | None = None
        used = None
        mysql_error = None

        if source != "BigQuery only" and database in mysql_clients:
            try:
                frame = add_provenance(
                    mysql_clients[database].get_trade_records(start.to_pydatetime(), end.to_pydatetime()), database,
                )
                used = "mysql"
            except Exception as exc:
                mysql_error = f"{type(exc).__name__}: {exc}"

        if frame is None and source != "MySQL only":
            try:
                frame = add_provenance(_fetch_from_bigquery(database, start, end, use_cache, overlap_days), database)
                used = "bigquery"
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                failures.append({
                    "database": database,
                    "error": f"MySQL: {mysql_error} | BigQuery: {detail}" if mysql_error else f"BigQuery: {detail}",
                })
                continue

        if frame is None:
            failures.append({"database": database, "error": mysql_error or "no configured source"})
            continue

        # Test-account exclusion needs a MySQL client's group tables; skip it
        # (rather than fail) when this database came from BigQuery.
        if used == "mysql" and database in mysql_clients:
            try:
                group_lookup = account_group_lookup(
                    mysql_clients[database], platform_for_database(database), frame.get("login", pd.Series(dtype="float64")),
                )
                flagged = non_client_logins(group_lookup)
                if flagged:
                    is_flagged = frame["login"].astype("Int64").isin(flagged)
                    if is_flagged.any():
                        excluded.append(frame.loc[is_flagged, ["database", "login", "account_key"]].drop_duplicates())
                        frame = frame.loc[~is_flagged].copy()
            except Exception:
                pass  # group lookup is best-effort; never lose real trade rows over it

        provenance.append({
            "database": database, "source": used, "rows": len(frame),
            "accounts": int(frame["account_key"].nunique()) if not frame.empty else 0,
        })
        frames.append(frame)

    records = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    excluded_frame = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame(columns=["database", "login", "account_key"])
    return records, pd.DataFrame(failures), excluded_frame, pd.DataFrame(provenance)


@st.cache_data(ttl=300)
def load_symbol_specs(
    source: str = "Auto (MySQL, fall back to BigQuery)", include_demo: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Per-database symbol contract metadata, with the same MySQL/BigQuery fallback as `load_records`.

    This matters as much as the trade records themselves: without a server's
    real `contract_size`, `attach_symbol_specs` falls back to an asset-class
    guess, and every notional, exposure, and VaR figure derived from it
    quietly loses accuracy. A MySQL outage must not silently downgrade those.
    """
    specs: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []

    mysql_clients: dict[str, Any] = {}
    if source != "BigQuery only":
        try:
            mysql_clients = clients_from_yaml()
        except Exception as exc:
            failures.append({"database": "(config)", "error": f"could not build MySQL clients: {type(exc).__name__}: {exc}"})

    spec_databases = sorted(set(mysql_clients) | set(BQ_DATASET_FOR_DATABASE))
    if not include_demo:
        spec_databases = [database for database in spec_databases if not is_demo_database(database)]
    for database in spec_databases:
        mysql_error = None
        if source != "BigQuery only" and database in mysql_clients:
            try:
                specs[database] = mysql_clients[database].symbol_specs()
                continue
            except Exception as exc:
                mysql_error = f"{type(exc).__name__}: {exc}"
        if source != "MySQL only":
            try:
                specs[database] = BigQueryDataClient(database).symbol_specs()
                continue
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                failures.append({
                    "database": database,
                    "error": f"MySQL: {mysql_error} | BigQuery: {detail}" if mysql_error else f"BigQuery: {detail}",
                })
                continue
        if mysql_error:
            failures.append({"database": database, "error": mysql_error})
    return specs, pd.DataFrame(failures)


@st.cache_data(ttl=300)
def attach_server_specs(records: pd.DataFrame, specs_by_database: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for database, specs in specs_by_database.items():
        source = records.loc[records["database"] == database]
        if source.empty:
            continue
        frames.append(attach_symbol_specs(source, specs))
    return pd.concat(frames, ignore_index=True) if frames else records.copy()


@st.cache_data(ttl=300)
def prepare_features(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = daily_account_features(records)
    return features, supervised_dataset(features)


@st.cache_data(ttl=900)
def cached_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    """The expensive walk-forward fit -- cached separately from scenario scoring."""
    return fit_oos_predictions(dataset)


@st.cache_data(ttl=900)
def cached_routing_model(records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], str, dict[str, Any]]:
    """The production two-stage routing model, plus its own quality scorecard.

    Returns ``(predictions, diagnostics, reason, quality)`` rather than the
    result object itself so Streamlit can hash and cache it.
    """
    result = fit_routing_model(records)
    quality = score_routing_quality(result.predictions) if result.is_usable() else {"available": False}
    return result.predictions, result.diagnostics, result.reason, quality


@st.cache_data(ttl=900)
def cached_rich_predictions(records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Walk-forward P(client wins next ACTIVE day) from the rich behavioural feature set.

    Measured on 90 days of real BigQuery data, this scores ROC AUC ~0.762,
    against ~0.650 for the original 13-feature/calendar-label pipeline. Three
    changes account for the gap, in order of contribution:

    * ~174 behavioural features (holding times, martingale and revenge
      patterns, payoff ratio, streaks, client drawdown, peer-relative ranks,
      lag sequence, lifetime record) rather than 13 same-day aggregates;
    * labels keyed to the account's next **active** trading day with no gap
      horizon, which both matches how a routing decision actually persists and
      recovers the infrequent accounts a 7-day tolerance silently deleted;
    * gradient boosting rather than logistic regression.

    Returns the scored frame plus a diagnostics dict, so the caller can show
    what was actually fitted instead of implying more coverage than exists.
    Falls back to an empty frame (never a fabricated score) if the model
    cannot be fitted -- `assign_books` already treats "no prediction" as a
    defined case.
    """
    diagnostics: dict[str, Any] = {"available": False, "reason": "", "rows": 0, "accounts": 0, "features": 0}
    try:
        frame = build_active_day_frame(records, max_gap_days=None)
    except Exception as exc:
        diagnostics["reason"] = f"feature build failed: {type(exc).__name__}: {exc}"
        return pd.DataFrame(), diagnostics
    if frame.empty:
        diagnostics["reason"] = "no labelled account-days in this window"
        return pd.DataFrame(), diagnostics

    columns = feature_columns(frame)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
    days = sorted(frame["decision_day"].unique())
    if len(days) <= MIN_TRAIN_DAYS:
        diagnostics["reason"] = f"only {len(days)} distinct days; need more than {MIN_TRAIN_DAYS} to score any out of sample"
        return pd.DataFrame(), diagnostics

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=150, random_state=0)

    predictions = pd.Series(np.nan, index=frame.index, dtype="float64")
    fitted = False
    for offset in range(MIN_TRAIN_DAYS, len(days)):
        test_mask = frame["decision_day"] == days[offset]
        if not test_mask.any():
            continue
        # Refit on a cadence rather than daily: a model fitted from data
        # strictly before day D and reused for D..D+4 never sees the future,
        # and this is what production desks actually do.
        if (offset - MIN_TRAIN_DAYS) % REFIT_EVERY_DAYS == 0 or not fitted:
            train_mask = frame["decision_day"].isin(days[:offset])
            target = frame.loc[train_mask, "target_client_wins"]
            if target.notna().sum() > 100 and target.astype(bool).nunique() >= 2:
                model.fit(frame.loc[train_mask, columns], target.astype(bool))
                fitted = True
        if fitted:
            predictions.loc[test_mask] = model.predict_proba(frame.loc[test_mask, columns])[:, 1]

    scored = frame.loc[predictions.notna(), ["account_key", "database", "platform", "decision_day"]].copy()
    if scored.empty:
        diagnostics["reason"] = "model could not be fitted on this window"
        return pd.DataFrame(), diagnostics
    # `assign_books` reads the loss probability; the model predicts wins.
    scored["model_probability_loss"] = 1.0 - predictions.loc[predictions.notna()].to_numpy()
    scored["target_loss"] = ~frame.loc[predictions.notna(), "target_client_wins"].astype(bool)
    scored["target_profit"] = frame.loc[predictions.notna(), "target_profit"].to_numpy()
    diagnostics.update({
        "available": True, "rows": int(len(scored)),
        "accounts": int(scored["account_key"].nunique()),
        "features": len(columns),
        "days_scored": int(scored["decision_day"].nunique()),
        "model": type(model).__name__,
    })
    return scored, diagnostics


@st.cache_data(ttl=900)
def cached_expected_value(dataset: pd.DataFrame) -> pd.DataFrame:
    """The expensive walk-forward *dollar* regression backing real book routing.

    Distinct from `cached_predictions` (a loss-probability classifier, kept
    only for the OOS validation tab's diagnostic): this predicts continuous
    expected $P&L, which is what `assign_books`'s risk-budget ranking needs
    to compare accounts in dollar terms, not just direction.
    """
    return fit_oos_expected_value(dataset)


@st.cache_data(ttl=900)
def cached_live_expected_value(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return predict_live_expected_value(dataset, features)


@st.cache_data(ttl=900)
def cached_pnl_volatility(components: pd.DataFrame) -> pd.DataFrame:
    return expanding_account_pnl_volatility(components)


@st.cache_data(ttl=900)
def cached_scorecard(predictions: pd.DataFrame, expected_value: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Full OOS model quality -- confusion matrix, AUC, calibration, regression fit."""
    return model_scorecard(predictions, expected_value, threshold)


@st.cache_data(ttl=900)
def cached_real_frontier(
    records: pd.DataFrame, predictions: pd.DataFrame, profile: pd.DataFrame, volatility: pd.DataFrame, economics: Economics,
) -> pd.DataFrame:
    """The actual dollar profit/drawdown frontier -- sweeps `assign_books` +
    `firm_daily_pnl` (real client P/L, real markup economics), not a
    classifier-only proxy. Cached on the underlying data, not the sliders --
    the sliders only pick a point that already exists on this curve.
    """
    return real_efficient_frontier(records, predictions, profile, volatility, economics)


@st.cache_data(ttl=300)
def cached_risk(records: pd.DataFrame) -> pd.DataFrame:
    return firm_risk_timeseries(records)


@st.cache_data(ttl=300)
def cached_intelligence(records: pd.DataFrame) -> pd.DataFrame:
    return client_intelligence(records)


@st.cache_data(ttl=300)
def cached_symbol_mapping(records: pd.DataFrame, specs_by_database: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return symbol_mapping_table(records, specs_by_database)


@st.cache_data(ttl=300)
def cached_symbol_var(records: pd.DataFrame, as_of: pd.Timestamp, confidence: float) -> pd.DataFrame:
    return symbol_var(records, as_of, confidence=confidence)


@st.cache_data(ttl=300)
def cached_portfolio_var(records: pd.DataFrame, as_of: pd.Timestamp, confidence: float) -> pd.DataFrame:
    return portfolio_var(records, as_of, confidence=confidence)


@st.cache_data(ttl=900)
def cached_intelligence_components(records: pd.DataFrame) -> pd.DataFrame:
    """Per-account, per-day building blocks for the point-in-time trader profile."""
    return daily_intelligence_components(records)


@st.cache_data(ttl=900)
def cached_expanding_profile(components: pd.DataFrame) -> pd.DataFrame:
    return expanding_client_profile(components)


decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
records, failures, excluded_test_accounts, source_provenance = load_records(
    decision_day - pd.Timedelta(days=lookback_days), decision_day + pd.Timedelta(days=1),
    source=data_source, use_cache=use_bq_cache, overlap_days=bq_overlap_days, include_demo=include_demo,
)
if include_demo:
    st.error(
        "**Demo accounts are included.** Firm P&L, VaR, exposure, and the ML training "
        "population all now contain simulated-money accounts the firm never actually hedges "
        "or books against -- these figures overstate real economics and should not be used "
        "for routing or reporting decisions. Switch **Account universe** back to live-only "
        "in the sidebar to read real numbers."
    )
if not failures.empty:
    st.warning(f"{len(failures)} database(s) unavailable; recommendations are incomplete.")
    st.dataframe(failures, width="stretch")
if not records.empty:
    records_gb = records.memory_usage(deep=True).sum() / 1e9
    if records_gb > 6.0:
        st.error(
            f"**Loaded {len(records):,} rows using {records_gb:.1f} GB of memory.** Feature "
            "building and model fitting need several GB more on top of this, so a window this "
            "wide risks exhausting memory mid-render. BigQuery's warehouse keeps far more "
            "granular history than the MySQL servers do (which are pruned), so the same "
            "lookback costs dramatically more here. **Reduce History (days) in the sidebar** "
            "-- 30 days is usually ample for the walk-forward model, which only needs enough "
            "distinct days to train."
        )
    elif records_gb > 3.0:
        st.warning(
            f"Loaded {len(records):,} rows using {records_gb:.1f} GB. Still workable, but "
            "narrowing **History (days)** will make everything noticeably faster."
        )

if not source_provenance.empty:
    from_bq = source_provenance.loc[source_provenance["source"] == "bigquery"]
    from_mysql = source_provenance.loc[source_provenance["source"] == "mysql"]
    if not from_bq.empty and not from_mysql.empty:
        st.warning(
            f"Mixed sources: {len(from_mysql)} database(s) live from MySQL, "
            f"{len(from_bq)} from the BigQuery mirror ({', '.join(from_bq['database'])}). "
            "Warehouse data can lag the live servers -- check the per-database breakdown below."
        )
    elif not from_bq.empty:
        st.info(
            f"All {len(from_bq)} database(s) served from the BigQuery warehouse mirror, not live MySQL. "
            "Figures may lag the live servers by the warehouse's own load cadence."
        )
    with st.expander(f"Data sources ({len(source_provenance)} database(s), {int(source_provenance['accounts'].sum()):,} accounts)"):
        st.dataframe(source_provenance, width="stretch")
if records.empty:
    st.error("No records were returned for the selected period.")
    st.stop()

records = records.loc[records["region"].isin(regions)]
specs_by_database, spec_failures = load_symbol_specs(source=data_source, include_demo=include_demo)
records = attach_server_specs(records, specs_by_database)
if not spec_failures.empty:
    st.warning(f"{len(spec_failures)} symbol metadata source(s) unavailable; asset-class fallback contract sizes used there.")
    st.dataframe(spec_failures, width="stretch")

economics = Economics(markup_bps=markup_bps, lp_cost_bps=lp_cost_bps, slippage_bps=slippage_bps)
features, dataset = prepare_features(records)

predictions = cached_predictions(dataset)
scenario = coupled_scenario(profit_weight, drawdown_weight)

# Live recommendations use a recent activity window. The supervised dataset
# drops the last day by design because it has no future label.
latest_day = features["day"].max()
requested_day = pd.Timestamp(analysis_day).floor("D")
display_day = min(requested_day, latest_day)
recent = recent_account_features(features, display_day, activity_window_days)
latest = recommendation(recent, Policy(min_observations=int(min_observations)))
latest = latest.loc[latest["confidence"] >= confidence_floor]

observed_day = pd.to_datetime(records["timestamp"]).dt.floor("D").max()
performance_day = min(display_day, observed_day)
account_universe = current_day_account_performance(records, performance_day, economics)

# --- book assignment: the flagship computation, shared by the sidebar intro
# banner and the Book assignment tab -------------------------------------------
components = cached_intelligence_components(records)
profile = cached_expanding_profile(components)
volatility = cached_pnl_volatility(components)

expected_value_predictions = cached_expected_value(dataset)
live_expected_value = cached_live_expected_value(dataset, features)
expected_value_with_live = (
    pd.concat([expected_value_predictions, live_expected_value], ignore_index=True)
    if not live_expected_value.empty else expected_value_predictions
)
# Prefer the rich behavioural classifier (ROC AUC ~0.76 measured on real data)
# over the expected-value regression (ranking measured as noise). `assign_books`
# picks whichever signal is present; merging the probability in here is what
# makes it available.
routing_predictions, routing_diagnostics, routing_reason, routing_quality = cached_routing_model(records)
routing_available = not routing_predictions.empty
routing_input = expected_value_with_live
if routing_available:
    routing_input = expected_value_with_live.merge(
        routing_predictions[[
            "account_key", "decision_day", "model_probability_loss",
            "probability_win", "expected_win_size_usd", "abook_priority_usd", "abook_priority_rank",
        ]],
        on=["account_key", "decision_day"], how="left",
    )

assignment = assign_books(routing_input, profile, volatility, profit_weight, drawdown_weight, economics=economics)
daily_pnl, book_detail, pnl_metrics = firm_daily_pnl(records, assignment, economics)

classifier_stats = classifier_performance(predictions, profit_weight, drawdown_weight)
real_frontier = cached_real_frontier(records, expected_value_with_live, profile, volatility, economics)

if routing_available:
    auc_text = f"{routing_quality['roc_auc']:.3f}" if routing_quality.get("roc_auc") == routing_quality.get("roc_auc") else "n/a"
    capture10 = routing_quality.get("cost_capture", {}).get("top_10pct")
    capture_text = (
        f" Its top-ranked 10% of accounts holds **{capture10:.0%}** of all the money the firm "
        f"loses to winning clients ({routing_quality.get('capture_lift_top10', float('nan')):.1f}x "
        "better than hedging at random)." if capture10 is not None else ""
    )
    st.success(
        f"**Routing on the two-stage behavioural model** -- {routing_diagnostics['features']} features over "
        f"{routing_diagnostics['rows']:,} out-of-sample account-days "
        f"({routing_diagnostics['accounts']:,} accounts, {routing_diagnostics['days_scored']} days). "
        f"Walk-forward **ROC AUC {auc_text}** for predicting whether a client wins on their next active "
        f"trading day.{capture_text} Accounts it cannot score default to B_BOOK rather than being dropped."
    )
else:
    st.warning(
        f"**Routing on the fallback expected-value regression** -- the behavioural model was not "
        f"available: {routing_reason or 'unknown'}. That regression's ranking was measured as noise "
        "(Spearman ~0), so treat today's ordering as low-confidence and widen **History (days)** if you can."
    )

risk_appetite = risk_appetite_from_weights(profit_weight, drawdown_weight)
st.info(
    f"Risk-budget policy: risk appetite {risk_appetite:.0%} of eligible B-book risk admitted today. "
    "Book routing ranks every account **once** by risk-adjusted expected value and only ever extends "
    "how far down that fixed ranking the B-book reaches as this rises -- moving the slider toward "
    "profit-seeking can only ever ADD accounts to the B-book, never reshuffle who's on which book, which "
    "is what makes expected firm P&L and the risk budget consumed move together, monotonically. "
    "(The classifier-calibration diagnostic below still uses the older coupled preference: effective "
    f"profit {scenario.effective_profit_weight:.0%} / drawdown {scenario.effective_drawdown_weight:.0%}.) "
    "See **Book assignment** and **Efficient frontier**."
)


def render_daily_drivers(day: pd.Timestamp, label: str) -> None:
    """One day's P/L drivers: real client P/L, firm proxy P/L, and who/what drove it."""
    st.subheader(f"{label}: {day.date()}")
    symbol_drivers = daily_symbol_drivers(records, day, economics)
    if symbol_drivers.empty:
        st.info("No activity on this day.")
        return
    totals = symbol_drivers[[
        "notional_usd_proxy", "client_realised_profit", "expected_net_markup_usd",
        "bbook_proxy_pnl_usd", "abook_residual_proxy_usd", "combined_firm_value_proxy_usd",
    ]].sum()
    _kpi_row([
        ("Notional proxy", f"${totals['notional_usd_proxy']:,.0f}", None),
        ("Client realised P/L", f"${totals['client_realised_profit']:,.0f}", None),
        ("Net markup (A-book)", f"${totals['expected_net_markup_usd']:,.2f}", None),
        ("B-book proxy P/L", f"${totals['bbook_proxy_pnl_usd']:,.0f}", None),
        ("Combined firm value proxy", f"${totals['combined_firm_value_proxy_usd']:,.2f}", None),
    ])
    st.caption(
        "Client realised P/L is actual closed-trade profit/loss, not a scenario. B-book proxy "
        "assumes the firm took the other side of every trade (``-client P/L``); A-book residual "
        "assumes a 10% unhedged residual on top of markup revenue. Combined firm value proxy adds "
        "net markup and the A-book residual -- it does not include the B-book alternative, since a "
        "symbol is carried on one book or the other, not both. For the *actual* routing decision "
        "per account (not per symbol) and its real firm P&L, see **Book assignment**."
    )

    driver_cols = st.columns(2)
    symbols_sorted = symbol_drivers.sort_values("combined_firm_value_proxy_usd", ascending=False)
    driver_cols[0].altair_chart(_bar_chart(symbols_sorted, "canonical_symbol", "combined_firm_value_proxy_usd", COLOR_BLUE, "Combined firm value proxy ($)").properties(height=280), use_container_width=True)
    driver_cols[1].altair_chart(_bar_chart(symbols_sorted, "canonical_symbol", "client_realised_profit", COLOR_ORANGE, "Client realised P/L ($)").properties(height=280), use_container_width=True)
    st.dataframe(symbol_drivers, width="stretch")

    st.markdown("**Biggest notional exposure accounts**")
    account_exposure = daily_account_exposure(records, day, top_n=top_n_accounts)
    if account_exposure.empty:
        st.info("No account activity on this day.")
        return
    exposure_cols = st.columns(2)
    gross_ranked = account_exposure.loc[account_exposure["rank_type"] == "gross"].sort_values("gross_notional_usd", ascending=False)
    net_ranked = account_exposure.loc[account_exposure["rank_type"] == "net"].sort_values("abs_net_notional_usd", ascending=False)
    exposure_cols[0].caption("By gross notional -- total flow, both directions")
    exposure_cols[0].altair_chart(_bar_chart(gross_ranked, "account_key", "gross_notional_usd", COLOR_BLUE).properties(height=260), use_container_width=True)
    exposure_cols[1].caption("By absolute net notional -- unhedged directional risk")
    exposure_cols[1].altair_chart(_bar_chart(net_ranked, "account_key", "abs_net_notional_usd", COLOR_ORANGE).properties(height=260), use_container_width=True)
    st.dataframe(account_exposure, width="stretch")


(
    assignment_tab, priority_tab, today_tab, mapping_tab, var_tab, frontier_tab,
    review_tab, intelligence_tab, oos_tab, risk_tab, account_tab,
) = st.tabs([
    "Book assignment", "A-book priority", "Daily drivers", "Symbol mapping", "VaR & exposure",
    "Efficient frontier", "Recommendations", "Client intelligence", "Model scorecard", "Firm risk",
    "Account detail",
])

with assignment_tab:
    st.subheader("Who should be A-booked vs B-booked, and what it's worth")
    st.caption(
        "**A_BOOK** = hedged with an LP: no market risk, only markup revenue net of "
        "LP/slippage cost. **B_BOOK** = unhedged: the firm takes the other side, so its "
        "P&L is the negative of the client's own realised P&L. **B_BOOK is the default**: "
        "absent evidence an account is expected to make money, the firm takes the other side "
        "of it. Every non-override account is ranked **once** by risk-adjusted expected dollar "
        "value (`excess_value_usd - risk_aversion * sigma_usd^2`) -- a fixed order that never "
        "changes with the slider. The rule, in priority order: (1) a demonstrated persistent edge "
        "or arbitrage pattern -> always A_BOOK; (2) otherwise, walk down that fixed ranking, "
        "admitting accounts into B_BOOK while the cumulative risk consumed stays within the "
        "sidebar's risk-appetite budget -- an eligible account the budget can't reach is hedged "
        "(A_BOOK) instead; (3) an account the model expects to be *more* valuable hedged than "
        "taken-the-other-side-of -> A_BOOK regardless of the dial; (4) no trustworthy prediction "
        "yet -> B_BOOK by default, never silently A_BOOK. Because the ranking is fixed and the "
        "budget only grows with the dial, the B-book set only ever grows as it moves toward "
        "profit-seeking -- expected firm P&L and the risk budget consumed move together, "
        "monotonically (a real guarantee, not empirical -- see `assign_books`'s docstring for the "
        "proof and its honest scope: realised max drawdown, a path-dependent statistic, is **not** "
        "covered by it, for any per-account routing rule). Every account with any activity in the "
        "lookback gets a book on every active day, including its very first one (defaulted to "
        "B_BOOK when there's no prior decision yet) -- A-book accounts plus B-book accounts always "
        "equals total active accounts."
    )

    _kpi_row([
        ("Total firm P&L (life-to-date)", f"${pnl_metrics['total_firm_pnl_usd']:,.0f}", None),
        ("Max daily drawdown", f"${pnl_metrics['max_daily_drawdown_usd']:,.0f}", None),
        ("Current drawdown", f"${pnl_metrics['current_drawdown_usd']:,.0f}", None),
        ("Win-day rate", f"{pnl_metrics['win_day_rate']:.0%}", None),
        ("Annualised Sharpe-like", f"{pnl_metrics['sharpe_like_annualised']:.2f}", None),
    ])

    today_row = daily_pnl.loc[daily_pnl["decision_day"] == performance_day]
    today_firm_pnl = float(today_row["firm_pnl_usd"].iloc[0]) if not today_row.empty else 0.0
    today_a_pnl = float(today_row["a_book_pnl_usd"].iloc[0]) if not today_row.empty else 0.0
    today_b_pnl = float(today_row["b_book_pnl_usd"].iloc[0]) if not today_row.empty else 0.0
    today_a_accounts = int(today_row["a_book_accounts"].iloc[0]) if not today_row.empty else 0
    today_b_accounts = int(today_row["b_book_accounts"].iloc[0]) if not today_row.empty else 0
    _kpi_row([
        (f"Firm P&L: {performance_day.date()}", f"${today_firm_pnl:,.0f}", None),
        ("A-book P&L today", f"${today_a_pnl:,.2f}", f"{today_a_accounts} accounts"),
        ("B-book P&L today", f"${today_b_pnl:,.0f}", f"{today_b_accounts} accounts"),
    ])

    if daily_pnl.empty:
        st.info("No book-assignable activity in the selected lookback.")
    else:
        st.markdown("#### Firm equity curve -- ML routing vs. always-A-book vs. always-B-book")
        st.caption(
            "Three ways the same period could have played out: the ML system's actual routing "
            "decisions each day, versus two constant-policy baselines -- hedging every account "
            "always (Always A-book) or taking the other side of every account always (Always "
            "B-book) -- computed from the exact same trades and economics. If ML-routed doesn't "
            "clearly beat both baselines, the model isn't adding value over the simplest policy."
        )
        always_a_book = firm_daily_pnl_fixed_book(records, "A_BOOK", economics)
        always_b_book = firm_daily_pnl_fixed_book(records, "B_BOOK", economics)
        equity_comparison = pd.concat([
            daily_pnl[["decision_day", "equity_usd", "firm_pnl_usd"]].assign(scenario="ML-routed (actual)"),
            always_a_book[["decision_day", "equity_usd", "firm_pnl_usd"]].assign(scenario="Always A-book"),
            always_b_book[["decision_day", "equity_usd", "firm_pnl_usd"]].assign(scenario="Always B-book"),
        ], ignore_index=True)
        scenario_domain = ["ML-routed (actual)", "Always A-book", "Always B-book"]
        scenario_range = [COLOR_AQUA, COLOR_BLUE, COLOR_ORANGE]
        equity_chart = alt.Chart(equity_comparison).mark_line(strokeWidth=2).encode(
            x=alt.X("decision_day:T", title=None),
            y=alt.Y("equity_usd:Q", title="Cumulative firm P&L ($)"),
            color=alt.Color("scenario:N", scale=alt.Scale(domain=scenario_domain, range=scenario_range), legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("decision_day:T", title="Day"), "scenario:N", alt.Tooltip("equity_usd:Q", title="Equity", format=",.0f"), alt.Tooltip("firm_pnl_usd:Q", title="Day P&L", format=",.0f")],
        )
        st.altair_chart(equity_chart.properties(height=320).interactive(), use_container_width=True)

        st.markdown("#### Daily firm P&L, by book")
        stacked = daily_pnl.melt(id_vars=["decision_day"], value_vars=["a_book_pnl_usd", "b_book_pnl_usd"], var_name="book", value_name="pnl_usd")
        stacked["book"] = stacked["book"].map({"a_book_pnl_usd": "A_BOOK", "b_book_pnl_usd": "B_BOOK"})
        book_bar = alt.Chart(stacked).mark_bar().encode(
            x=alt.X("decision_day:T", title=None),
            y=alt.Y("pnl_usd:Q", title="Firm P&L ($)"),
            color=alt.Color("book:N", scale=alt.Scale(domain=["A_BOOK", "B_BOOK"], range=[COLOR_BLUE, COLOR_ORANGE]), legend=alt.Legend(title=None, orient="top")),
            tooltip=["decision_day:T", "book:N", alt.Tooltip("pnl_usd:Q", format=",.0f")],
        )
        st.altair_chart(book_bar.properties(height=220), use_container_width=True)

    today_detail = book_detail.loc[book_detail["decision_day"] == performance_day].copy()
    export_columns = [
        "account_key", "book", "firm_pnl_usd", "notional_usd_proxy", "client_realised_profit",
        "excess_value_usd", "sigma_usd", "quality_score", "risk_budget_admitted",
        "expanding_edge_flag", "expanding_toxicity_flag", "expanding_arbitrage_flag", "low_confidence", "reason_codes",
    ]
    export_columns = [c for c in export_columns if c in today_detail.columns]

    st.divider()
    st.markdown(f"#### A-book / B-book account lists -- {performance_day.date()}")
    st.caption(
        "The full, unfiltered account list for each book on the selected day, sorted by |firm P&L "
        "impact| descending -- the most profit/drawdown-affecting accounts first -- ready to hand to "
        "dealing/ops for same-day use. Downloads as two separate CSVs, one per book."
    )
    if today_detail.empty:
        st.info("No account activity assignable to a book on this day.")
    else:
        today_detail_ranked = today_detail.assign(abs_impact_usd=today_detail["firm_pnl_usd"].abs())
        a_book_export = today_detail_ranked.loc[today_detail_ranked["book"] == "A_BOOK"].sort_values("abs_impact_usd", ascending=False)
        b_book_export = today_detail_ranked.loc[today_detail_ranked["book"] == "B_BOOK"].sort_values("abs_impact_usd", ascending=False)

        export_cols = st.columns(2)
        with export_cols[0]:
            st.markdown(f"**A-book accounts ({len(a_book_export):,})** -- hedged, markup-only economics")
            st.dataframe(a_book_export[export_columns], width="stretch", height=360)
            st.download_button(
                " Download A-book accounts (CSV)",
                a_book_export[export_columns].to_csv(index=False).encode("utf-8"),
                file_name=f"a_book_accounts_{performance_day.date()}.csv",
                mime="text/csv",
                key="download_a_book",
            )
        with export_cols[1]:
            st.markdown(f"**B-book accounts ({len(b_book_export):,})** -- unhedged, firm profits from client loss")
            st.dataframe(b_book_export[export_columns], width="stretch", height=360)
            st.download_button(
                " Download B-book accounts (CSV)",
                b_book_export[export_columns].to_csv(index=False).encode("utf-8"),
                file_name=f"b_book_accounts_{performance_day.date()}.csv",
                mime="text/csv",
                key="download_b_book",
            )

    st.divider()
    st.markdown(f"#### Explore & filter -- ranked by expected impact ({performance_day.date()})")
    st.caption("Sorted by |firm P&L impact| descending: the accounts at the top are where today's routing decision matters most.")

    if today_detail.empty:
        st.info("No account activity assignable to a book on this day.")
    else:
        filter_cols = st.columns(4)
        book_filter = filter_cols[0].multiselect("Book", ["A_BOOK", "B_BOOK"], default=["A_BOOK", "B_BOOK"])
        flag_filter = filter_cols[1].multiselect("Categorisation", ["Persistent edge", "Toxicity", "Arbitrage pattern", "Low confidence"], default=[])
        min_impact = filter_cols[2].number_input("Min |impact| ($)", min_value=0.0, value=0.0, step=100.0)
        search = filter_cols[3].text_input("Search account")

        filtered = today_detail.loc[today_detail["book"].isin(book_filter)]
        if flag_filter:
            # Union (OR), not successive narrowing (AND): a normal multiselect
            # tag filter means "any of these", so selecting both "Persistent
            # edge" and "Toxicity" should show accounts flagged with EITHER,
            # not only accounts flagged with both simultaneously.
            flag_columns = {
                "Persistent edge": "expanding_edge_flag",
                "Toxicity": "expanding_toxicity_flag",
                "Arbitrage pattern": "expanding_arbitrage_flag",
                "Low confidence": "low_confidence",
            }
            matches = pd.Series(False, index=filtered.index)
            for label in flag_filter:
                matches |= filtered[flag_columns[label]].fillna(False)
            filtered = filtered.loc[matches]
        filtered = filtered.loc[filtered["firm_pnl_usd"].abs() >= min_impact]
        if search:
            filtered = filtered.loc[filtered["account_key"].astype(str).str.contains(search, case=False, na=False, regex=False)]

        filtered = filtered.assign(abs_impact_usd=filtered["firm_pnl_usd"].abs()).sort_values("abs_impact_usd", ascending=False)
        display_columns = export_columns
        st.caption(f"{len(filtered):,} of {len(today_detail):,} accounts shown.")
        st.dataframe(filtered[display_columns], width="stretch", height=420)
        st.download_button(
            " Download filtered assignment (CSV)",
            filtered[display_columns].to_csv(index=False).encode("utf-8"),
            file_name=f"book_assignment_{performance_day.date()}.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown(f"#### Top client profitability contributors -- {performance_day.date()}")
        st.caption(
            "Real, realised client P/L (not a scenario): the clients whose own trading made the most money "
            "and the clients whose own trading lost the most, today. A big winner is exactly the account "
            "`expanding_edge_flag`/the persistent-edge screen is designed to catch and route to A-book; a big "
            "loser is the flow a B-book policy is designed to profit from."
        )
        contributors = today_detail.loc[today_detail["client_realised_profit"] != 0].copy()
        if contributors.empty:
            st.info("No closed, realised client P/L on this day.")
        else:
            top_winners = contributors.sort_values("client_realised_profit", ascending=False).head(10)
            top_losers = contributors.sort_values("client_realised_profit", ascending=True).head(10)
            contrib_cols = st.columns(2)
            contrib_cols[0].caption("Top winning clients (biggest realised gains)")
            contrib_cols[0].altair_chart(
                _bar_chart(top_winners, "account_key", "client_realised_profit", COLOR_GOOD, "Client realised P/L ($)").properties(height=280),
                use_container_width=True,
            )
            contrib_cols[1].caption("Top losing clients (biggest realised losses)")
            contrib_cols[1].altair_chart(
                _bar_chart(top_losers, "account_key", "client_realised_profit", COLOR_CRITICAL, "Client realised P/L ($)", sort="y").properties(height=280),
                use_container_width=True,
            )

with today_tab:
    render_daily_drivers(performance_day, "Daily drivers")
    if compare_enabled and compare_day is not None:
        st.divider()
        render_daily_drivers(pd.Timestamp(compare_day).floor("D"), "Comparison day")

    st.divider()
    st.subheader("All known accounts")
    any_activity_accounts = int(records["account_key"].nunique())
    closed_trade_accounts = int(records.loc[is_realised_trade(records), "account_key"].nunique())
    ratio = (any_activity_accounts / closed_trade_accounts) if closed_trade_accounts else float("nan")
    _kpi_row([
        ("Accounts: any activity", f"{any_activity_accounts:,}", "opens, modifies, or closes -- this dashboard's definition"),
        ("Accounts: closed/realised trade", f"{closed_trade_accounts:,}", "what a back-office system usually counts"),
        ("Ratio", f"{ratio:.1f}x" if closed_trade_accounts else "n/a", "why the two numbers won't match"),
        ("Test accounts excluded", f"{len(excluded_test_accounts):,}", "already removed from both counts above"),
    ])
    st.caption(
        "Accounts come from the selected lookback. Inactive accounts remain visible with zero "
        "current-day activity. 'Account' here means any order activity in the window -- opens, "
        "modifies, and closes, including a position still open with no realised outcome yet -- "
        "not just closed/realised trades (see the KPIs above). A system that counts only closed "
        "trades for the same period will show meaningfully fewer accounts by design; that gap is "
        "expected, not a data-pulling error. Known internal test-group accounts are excluded "
        "(best-effort, by account group) before either count above is taken."
    )
    if not excluded_test_accounts.empty:
        with st.expander(f"Excluded test accounts ({len(excluded_test_accounts)})"):
            st.dataframe(excluded_test_accounts, width="stretch")
    st.dataframe(account_universe, width="stretch")

    st.subheader("Canonical-symbol rolling exposure")
    st.caption("Gross = total flow; net = signed directional exposure after netting buys against sells within the canonical group.")
    rolling_exposure = rolling_symbol_exposure(records, performance_day)
    st.dataframe(rolling_exposure.sort_values(["window_days", "gross_exposure_usd"], ascending=[True, False]), width="stretch")

with mapping_tab:
    st.subheader("Raw-to-canonical symbol mapping")
    st.caption(
        "Every raw symbol actually traded on this venue, the canonical bucket it aggregates "
        "into everywhere else in this dashboard, and where its contract size came from. Add an "
        "unmapped or misclassified raw symbol to `symbol_map.yaml` (see `symbol_map.example.yaml`) "
        "rather than relying on the automatic suffix stripping to guess it."
    )
    mapping = cached_symbol_mapping(records, specs_by_database)
    _kpi_row([
        ("Raw symbols", f"{len(mapping):,}", None),
        ("Canonical symbols", f"{mapping['canonical_symbol'].nunique():,}", None),
        ("Fallback contract size", f"{(mapping['contract_size_source'] == 'asset_class_fallback').sum():,}", None),
        ("Heuristic/alias mapped", f"{(mapping['mapping_source'] != 'server_exact').sum():,}", None),
    ])
    only_fallback = st.checkbox("Show only fallback / heuristic rows")
    display_mapping = mapping
    if only_fallback:
        display_mapping = mapping.loc[(mapping["contract_size_source"] == "asset_class_fallback") | (mapping["mapping_source"] != "server_exact")]
    st.dataframe(display_mapping, width="stretch")
    asset_class_totals = mapping.groupby("asset_class", observed=True, as_index=False)["observations"].sum()
    st.altair_chart(_bar_chart(asset_class_totals, "asset_class", "observations", COLOR_AQUA).properties(height=260), use_container_width=True)

with var_tab:
    st.subheader(f"Symbol VaR as of {performance_day.date()} ({var_confidence:.1%} confidence)")
    st.caption(
        "Parametric VaR = z * EWMA daily volatility * sqrt(horizon) * current USD notional "
        "(square-root-of-time scaling of a trade-implied volatility estimate). Historical VaR is "
        "the empirical quantile of the symbol's own trailing N-day returns applied to the same "
        "exposure -- the tail matching the book's actual direction (loss/left tail for a net long, "
        "gain/right tail for a net short) -- and is NaN rather than a guess where there isn't enough "
        "joint history yet. Both are undiversified, single-symbol figures."
    )
    var_table = cached_symbol_var(records, performance_day, var_confidence)
    if var_table.empty:
        st.info("No exposure on this day to compute VaR against.")
    else:
        window_choice = st.select_slider("Horizon (days)", options=sorted(var_table["window_days"].unique()), value=1)
        window_slice = var_table.loc[var_table["window_days"] == window_choice].sort_values("parametric_gross_var_usd", ascending=False)
        var_cols = st.columns(2)
        var_cols[0].caption("Parametric gross VaR by symbol")
        var_cols[0].altair_chart(_bar_chart(window_slice, "canonical_symbol", "parametric_gross_var_usd", COLOR_BLUE).properties(height=280), use_container_width=True)
        var_cols[1].caption("Historical gross VaR by symbol (blank = insufficient history)")
        var_cols[1].altair_chart(_bar_chart(window_slice.assign(historical_gross_var_usd=window_slice["historical_gross_var_usd"].fillna(0)), "canonical_symbol", "historical_gross_var_usd", COLOR_ORANGE).properties(height=280), use_container_width=True)
        st.dataframe(window_slice, width="stretch")

    st.subheader("Portfolio VaR (correlation-adjusted)")
    st.caption(
        "Undiversified VaR sums every symbol's parametric VaR (the fully-correlated worst case). "
        "Diversified VaR applies the variance-covariance identity "
        "VaR_p = sqrt(sum_i sum_j VaR_i * VaR_j * rho_ij) using the pairwise correlation of "
        "trade-implied returns. An unmeasured pair falls back to the correlation that maximizes "
        "its contribution to portfolio variance (no free diversification credit for a relationship "
        "that hasn't actually been measured), so the gap between the two is a conservative estimate "
        "of the current book's diversification benefit, not an optimistic one."
    )
    portfolio_table = cached_portfolio_var(records, performance_day, var_confidence)
    st.dataframe(portfolio_table, width="stretch")

with frontier_tab:
    st.subheader("Profit / drawdown efficient frontier -- real firm economics")
    st.caption(
        "Each point re-runs the *actual* routing simulation (`assign_books` + `firm_daily_pnl`) at a "
        "different risk-appetite setting (drawdown_weight = 100 - profit_weight, exactly as the two "
        "sidebar sliders are linked) -- real client P/L for B-book, real markup economics for A-book, "
        "not a classifier-only proxy. `total_firm_pnl_usd` is **guaranteed** monotonically non-decreasing "
        "left-to-right along `profit_weight` (the risk-budget walk only ever admits more accounts as "
        "the dial rises, never reshuffles who's on which book -- see `assign_books`'s docstring for the "
        "proof). `max_daily_drawdown_usd` is **not** covered by that guarantee -- it's a realised, "
        "path-dependent statistic (a minimum over cumulative sums with a different slope per day), which "
        "is why filled (Pareto-optimal) points matter: no other swept scenario gets both a higher total "
        "P&L and a shallower drawdown, but the curve can still bend either way on the drawdown axis. The "
        "diamond is the sidebar's current selection, from the exact same **Book assignment** computation: "
        "move either slider and it slides along this same curve, never off it independently."
    )
    if real_frontier.empty or real_frontier["days"].max() == 0:
        st.info("Not enough routable history yet to trace a frontier -- widen the lookback window.")
    else:
        current_point = pd.DataFrame([{
            "max_daily_drawdown_usd": pnl_metrics["max_daily_drawdown_usd"],
            "total_firm_pnl_usd": pnl_metrics["total_firm_pnl_usd"],
        }])
        base = alt.Chart(real_frontier).mark_line(color=COLOR_MUTED, strokeDash=[4, 2]).encode(
            x=alt.X("max_daily_drawdown_usd:Q", title="Max daily drawdown ($)"),
            y=alt.Y("total_firm_pnl_usd:Q", title="Total firm P&L ($)"),
        )
        points = alt.Chart(real_frontier).mark_circle(size=70).encode(
            x="max_daily_drawdown_usd:Q",
            y="total_firm_pnl_usd:Q",
            color=alt.Color("is_pareto_optimal:N", scale=alt.Scale(domain=[True, False], range=[COLOR_AQUA, "#c9c9c9"]), legend=alt.Legend(title="Pareto-optimal")),
            tooltip=["profit_weight:Q", "drawdown_weight:Q", "total_firm_pnl_usd:Q", "max_daily_drawdown_usd:Q", "win_day_rate:Q"],
        )
        current = alt.Chart(current_point).mark_point(size=260, shape="diamond", color=COLOR_CRITICAL, filled=True).encode(
            x="max_daily_drawdown_usd:Q", y="total_firm_pnl_usd:Q",
        )
        st.altair_chart((base + points + current).interactive(), use_container_width=True)
        st.dataframe(real_frontier, width="stretch")

with review_tab:
    st.subheader("Daily recommendation queue (single-day heuristic baseline)")
    st.caption(
        "A simpler, explainable A/B/review heuristic on a single day's risk score -- kept as a "
        "baseline distinct from the model- and categorisation-driven **Book assignment** tab, which "
        "is the more rigorous, P&L-validated routing decision."
    )
    st.dataframe(latest.sort_values(["risk_score", "max_position_lots"], ascending=False), width="stretch")

with intelligence_tab:
    st.subheader("Behavior, toxicity, arbitrage, and edge screens")
    st.caption(
        "Whole-history screening hypotheses from four independent detectors -- "
        "`detect_edge_clients`, `detect_toxicity`, `detect_arbitrage`, `detect_fraud_review` -- "
        "composed by `client_intelligence`. Not fraud findings or trading orders; flags require "
        "human review and independent evidence. The point-in-time versions of the edge/toxicity/"
        "arbitrage screens (built day-by-day, without seeing the future) are what actually drive "
        "the **Book assignment** tab."
    )
    intelligence = cached_intelligence(records)
    _kpi_row([
        ("Accounts screened", f"{len(intelligence):,}", None),
        ("Arbitrage screens", f"{intelligence['arbitrage_screen'].sum():,}", None),
        ("Edge screens", f"{intelligence['edge_screen'].sum():,}", None),
        ("Toxicity screens", f"{(intelligence['toxicity_score'] >= 0.5).sum():,}", None),
        ("Fraud reviews", f"{intelligence['fraud_review_screen'].sum():,}", None),
    ])
    st.dataframe(intelligence, width="stretch")

with oos_tab:
    st.subheader("Model scorecard -- how good is the ML system, really?")
    st.caption(
        "How well the loss-probability model actually predicts tomorrow, independent of any dollar "
        "estimate -- for the real dollar walk-forward simulation of the daily A/B routing itself, see "
        "**Book assignment** (current day) and **Efficient frontier** (swept across every "
        "profit/drawdown preference). **Precision** = of the account-days this threshold would route "
        "B_BOOK (predicted loss), how many actually lost -- a low precision means the B-book routing "
        "at this threshold is unreliable. **Recall** = of the account-days that actually lost, how "
        "many did the model catch -- a low recall means the firm is missing genuinely bad days. "
        "**Base rate** is how often an OOS day was actually a loss, for context: a model that always "
        "predicts \"loss\" trivially gets recall=1 but precision equal to the base rate, so read the "
        "two together, never precision alone."
    )
    scorecard = cached_scorecard(predictions, expected_value_predictions, float(classifier_stats["threshold"]))

    if scorecard.get("empty"):
        st.warning(
            "No out-of-sample predictions yet. The walk-forward model needs `min_train_days` "
            "(20 by default) of distinct trading days before it scores its first day, so a "
            "lookback shorter than that produces no OOS rows at all -- widen **History (days)**."
        )
    else:
        coverage = scorecard["coverage"]
        st.markdown("#### How much evidence is behind these numbers")
        st.caption(
            "Read this first -- every metric below is only as trustworthy as the data behind it. "
            "The model is **walk-forward**: for each day it is refit on every day strictly before "
            "it and then scores that day once, unseen. There is no single fixed train/test split; "
            "the first `min_train_days` (20) days are consumed as the initial training base and "
            "produce no OOS rows, and every day after that is both scored out-of-sample and then "
            "folded into the training set for the next day."
        )
        _kpi_row([
            ("Days in window", f"{coverage['total_days_in_window']:,}", "from the History slider"),
            ("Train-only days", f"{coverage['train_only_days']:,}", "consumed before first prediction"),
            ("Days scored OOS", f"{coverage['oos_days']:,}", "each scored unseen"),
            ("OOS account-days", f"{coverage['oos_rows']:,}", f"{coverage['accounts']:,} accounts"),
            ("Actual loss rate", f"{coverage['base_rate']:.1%}", "the base rate to beat"),
        ])
        st.caption(f"OOS period: {pd.Timestamp(coverage['first_oos_day']).date()} to {pd.Timestamp(coverage['last_oos_day']).date()}.")

        confusion = scorecard["confusion"]
        discrimination = scorecard["discrimination"]

        st.markdown("#### Is the model actually any good?")
        auc = discrimination["roc_auc"]
        verdict = (
            "no better than a coin flip -- do not route on it" if not (auc == auc) or auc < 0.55 else
            "weak but real signal" if auc < 0.65 else
            "moderate, usable signal" if auc < 0.75 else
            "strong signal"
        )
        lift = confusion["precision"] / coverage["base_rate"] if coverage["base_rate"] else float("nan")
        st.info(
            f"**ROC AUC {auc:.3f}** -- {verdict}. AUC is the probability the model ranks a randomly "
            f"chosen losing account-day above a randomly chosen winning one: 0.5 is random, 1.0 is "
            f"perfect. At the current threshold it flags {confusion['predicted_positive_rate']:.1%} "
            f"of account-days and is right {confusion['precision']:.1%} of the time, against a "
            f"{coverage['base_rate']:.1%} base rate -- a **lift of {lift:.2f}x** over guessing."
        )
        _kpi_row([
            ("ROC AUC", f"{auc:.3f}", "0.5 = random"),
            ("Average precision", f"{discrimination['average_precision']:.3f}", "vs base rate"),
            ("Balanced accuracy", f"{confusion['balanced_accuracy']:.1%}", "class-imbalance safe"),
            ("Matthews corr.", f"{confusion['matthews_corrcoef']:.3f}", "-1 to +1, 0 = random"),
            ("Calibration error", f"{confusion.get('calibration_error', float('nan')):.3f}", "lower is better"),
        ])

        st.markdown("#### Confusion matrix")
        st.caption(
            f"At the current threshold of {confusion['threshold']:.2f}. **Positive = the model "
            "predicts this account loses tomorrow** (the B-book case: the firm profits by taking "
            "the other side). A false positive means the firm B-booked an account that actually "
            "won -- a direct loss. A false negative means it hedged away an account that would "
            "have lost -- forgone profit."
        )
        matrix = pd.DataFrame({
            "": ["**Actually lost**", "**Actually won**"],
            "Predicted loss (B-book)": [
                f"{confusion['true_positive']:,}  (true positive)",
                f"{confusion['false_positive']:,}  (false positive -- costly)",
            ],
            "Predicted win (A-book)": [
                f"{confusion['false_negative']:,}  (false negative -- forgone)",
                f"{confusion['true_negative']:,}  (true negative)",
            ],
        })
        st.table(matrix.set_index(""))
        _kpi_row([
            ("Precision", f"{confusion['precision']:.1%}", "of B-book calls, right"),
            ("Recall", f"{confusion['recall']:.1%}", "of real losses, caught"),
            ("Specificity", f"{confusion['specificity']:.1%}", "of winners, correctly hedged"),
            ("F1", f"{confusion['f1']:.3f}", None),
            ("Accuracy", f"{confusion['accuracy']:.1%}", "misleading if imbalanced"),
        ])

        chart_columns = st.columns(2)
        calibration = scorecard["calibration"]
        if not calibration.empty:
            chart_columns[0].markdown("**Calibration** -- predicted vs actual")
            chart_columns[0].caption("On the diagonal = predicted probabilities are literally true. Above = under-confident, below = over-confident.")
            diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                color=COLOR_MUTED, strokeDash=[4, 2],
            ).encode(x=alt.X("x:Q", title="Mean predicted probability"), y=alt.Y("y:Q", title="Actual loss rate"))
            points = alt.Chart(calibration).mark_point(size=90, filled=True, color=COLOR_BLUE).encode(
                x="mean_predicted:Q", y="actual_rate:Q",
                size=alt.Size("observations:Q", legend=None),
                tooltip=["mean_predicted:Q", "actual_rate:Q", "observations:Q"],
            )
            chart_columns[0].altair_chart((diagonal + points).properties(height=300), use_container_width=True)

        by_threshold = scorecard["by_threshold"]
        chart_columns[1].markdown("**Precision / recall vs threshold**")
        chart_columns[1].caption("The trade-off curve. Dashed line is the sidebar's current operating point.")
        sweep_long = by_threshold.melt(id_vars=["threshold"], value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value")
        sweep_chart = alt.Chart(sweep_long).mark_line(strokeWidth=2).encode(
            x=alt.X("threshold:Q", title="Loss-probability threshold"),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall", "f1"], range=[COLOR_BLUE, COLOR_ORANGE, COLOR_AQUA]), legend=alt.Legend(title=None, orient="top")),
        )
        rule = alt.Chart(pd.DataFrame([{"t": confusion["threshold"]}])).mark_rule(color=COLOR_MUTED, strokeDash=[4, 2]).encode(x="t:Q")
        chart_columns[1].altair_chart((sweep_chart + rule).properties(height=300), use_container_width=True)

        regression = scorecard.get("regression")
        if regression and regression.get("observations"):
            st.markdown("#### Expected-P&L regression (what actually drives routing)")
            st.caption(
                "The risk-budget router ranks accounts by predicted **dollar** value, not by the "
                "loss probability above -- so this is the model that matters most for routing. "
                "**Sign accuracy** and **Spearman** matter more than R^2 here: the router only needs "
                "the direction and the ordering right, not the exact magnitude. A near-zero or "
                "negative R^2 with good sign accuracy is still a usable router; poor sign accuracy "
                "is not, whatever R^2 says."
            )
            _kpi_row([
                ("Sign accuracy", f"{regression['sign_accuracy']:.1%}", "direction correct"),
                ("Spearman", f"{regression['spearman']:.3f}", "ranking quality"),
                ("R-squared", f"{regression['r2']:.3f}", "magnitude fit"),
                ("MAE", f"${regression['mae_usd']:,.0f}", "avg error"),
                ("RMSE", f"${regression['rmse_usd']:,.0f}", "outlier-sensitive"),
            ])
            st.caption(
                f"{regression['observations']:,} OOS account-days. Mean predicted "
                f"${regression['mean_predicted_usd']:,.2f} vs mean actual ${regression['mean_actual_usd']:,.2f} -- "
                "a large gap between these two means the model is biased, not just noisy."
            )

        with st.expander("Full threshold sweep table"):
            st.dataframe(by_threshold, width="stretch")

    st.divider()
    st.markdown("#### Precision / recall across the full profit/drawdown sweep")
    st.caption("Where the sidebar's current threshold (dashed line) sits relative to every other choice on the same coupled preference.")
    sweep_rows = [classifier_performance(predictions, p, 100.0 - p) for p in range(0, 101, 5)]
    sweep = pd.DataFrame(sweep_rows)
    if sweep["observations"].max() > 0:
        sweep_long = sweep.melt(id_vars=["profit_weight"], value_vars=["precision", "recall"], var_name="metric", value_name="value")
        sweep_chart = alt.Chart(sweep_long).mark_line(strokeWidth=2).encode(
            x=alt.X("profit_weight:Q", title="Profit-maximisation weight"),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall"], range=[COLOR_BLUE, COLOR_ORANGE]), legend=alt.Legend(title=None, orient="top")),
        )
        threshold_rule = alt.Chart(pd.DataFrame([{"profit_weight": profit_weight}])).mark_rule(color=COLOR_MUTED, strokeDash=[4, 2]).encode(x="profit_weight:Q")
        st.altair_chart((sweep_chart + threshold_rule).properties(height=280), use_container_width=True)
        st.dataframe(sweep, width="stretch")
    else:
        st.info("Not enough OOS history yet to sweep classifier quality -- widen the lookback window.")

with risk_tab:
    st.subheader("Firm risk through time")
    risk = cached_risk(records)
    firm_daily = risk.groupby("day", as_index=False).agg(gross_notional_usd=("gross_notional_usd", "sum"), abs_net_notional_usd=("abs_net_notional_usd", "sum"), accounts=("accounts", "sum"))
    st.altair_chart(_line_chart(firm_daily, "day", ["gross_notional_usd", "abs_net_notional_usd"], [COLOR_BLUE, COLOR_ORANGE], "Notional ($)").properties(height=300), use_container_width=True)
    st.dataframe(risk.sort_values(["day", "gross_lots"], ascending=False).head(100), width="stretch")

with priority_tab:
    st.subheader("A-book priority -- who to hedge first, and why")
    st.caption(
        "Hedging capacity is finite, so the question is not only *which* accounts to A-book but in "
        "**what order**. Priority is the expected dollar cost of leaving an account on the B-book: "
        "**P(client wins next active day) x E[size of that win]**. Two models, deliberately: the "
        "classifier owns direction (measurable, ROC AUC ~0.76) and the magnitude model owns size "
        "*given* a win -- a single model asked to rank raw P&L scored ~0, because it had to explain "
        "sign and size at once. Work down this list until the risk budget is spent."
    )

    if not routing_available:
        st.warning(f"The routing model is unavailable, so no priority ranking exists: {routing_reason or 'unknown'}")
    else:
        latest_scored_day = routing_predictions["decision_day"].max()
        day_choices = sorted(routing_predictions["decision_day"].unique(), reverse=True)[:30]
        chosen_day = st.selectbox(
            "Decision day", day_choices, index=0,
            format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        )
        day_frame = routing_predictions.loc[routing_predictions["decision_day"] == chosen_day].copy()
        day_frame = day_frame.sort_values("abook_priority_usd", ascending=False).reset_index(drop=True)
        day_frame["rank"] = day_frame.index + 1
        day_frame["cumulative_cost_share"] = (
            day_frame["expected_win_size_usd"].fillna(0).mul(day_frame["probability_win"]).cumsum()
            / max(day_frame["abook_priority_usd"].sum(), 1e-9)
        )

        quality_auc = routing_quality.get("roc_auc", float("nan"))
        capture = routing_quality.get("cost_capture", {})
        _kpi_row([
            ("Accounts scored", f"{len(day_frame):,}", pd.Timestamp(chosen_day).strftime("%Y-%m-%d")),
            ("Model ROC AUC", f"{quality_auc:.3f}" if quality_auc == quality_auc else "n/a", "walk-forward, out of sample"),
            ("Top 10% captures", f"{capture.get('top_10pct', float('nan')):.0%}", "of all firm cost to winners"),
            ("Lift vs random", f"{routing_quality.get('capture_lift_top10', float('nan')):.1f}x", "1.0 = no better than chance"),
            ("Predicted win rate", f"{day_frame['probability_win'].mean():.1%}", f"base {routing_quality.get('win_base_rate', float('nan')):.1%}"),
        ])

        st.markdown("#### How much do you want to hedge?")
        budget_share = st.slider(
            "Hedge the top N% of accounts by priority", 1, 50, 10,
            help=(
                "The concentration curve below shows what this buys. If the top 10% holds far more "
                "than 10% of the cost, selective hedging is working; if it tracks the diagonal, the "
                "ranking is not adding value and a flat policy would do as well."
            ),
        )
        cutoff_count = max(1, int(len(day_frame) * budget_share / 100))
        selected = day_frame.head(cutoff_count)
        st.info(
            f"Hedging the top **{budget_share}%** means A-booking **{cutoff_count:,}** of "
            f"{len(day_frame):,} accounts, carrying **${selected['abook_priority_usd'].sum():,.0f}** of "
            f"expected B-book cost out of **${day_frame['abook_priority_usd'].sum():,.0f}** total "
            f"(**{selected['abook_priority_usd'].sum() / max(day_frame['abook_priority_usd'].sum(), 1e-9):.0%}**)."
        )

        curve = pd.DataFrame({
            "share_of_accounts": (day_frame.index + 1) / len(day_frame),
            "share_of_cost": day_frame["abook_priority_usd"].cumsum() / max(day_frame["abook_priority_usd"].sum(), 1e-9),
        })
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            color=COLOR_MUTED, strokeDash=[4, 2],
        ).encode(x=alt.X("x:Q", title="Share of accounts hedged (ranked by priority)"),
                 y=alt.Y("y:Q", title="Share of expected B-book cost removed"))
        curve_line = alt.Chart(curve).mark_line(color=COLOR_AQUA, strokeWidth=2).encode(
            x="share_of_accounts:Q", y="share_of_cost:Q",
            tooltip=[alt.Tooltip("share_of_accounts:Q", format=".1%"), alt.Tooltip("share_of_cost:Q", format=".1%")],
        )
        marker = alt.Chart(pd.DataFrame([{"x": budget_share / 100}])).mark_rule(
            color=COLOR_CRITICAL, strokeDash=[3, 3],
        ).encode(x="x:Q")
        st.altair_chart((diagonal + curve_line + marker).properties(height=300), use_container_width=True)
        st.caption(
            "The dashed diagonal is hedging at random. The further the curve bows above it, the more "
            "the cost is concentrated in the accounts the model ranks highest."
        )

        st.markdown("#### The A-book list, in priority order")
        display_columns = [c for c in [
            "rank", "account_key", "database", "abook_priority_usd", "probability_win",
            "expected_win_size_usd", "abook_priority_rank", "days_until_next_active",
        ] if c in selected.columns]
        st.dataframe(selected[display_columns], width="stretch", height=420)
        st.download_button(
            "Download A-book priority list (CSV)",
            selected[display_columns].to_csv(index=False).encode("utf-8"),
            file_name=f"abook_priority_{pd.Timestamp(chosen_day):%Y%m%d}.csv",
            mime="text/csv", key="download_priority",
        )
        st.caption(
            "`abook_priority_usd` = P(win) x E[win size] -- the expected dollars saved by hedging this "
            "account rather than taking the other side of it. `days_until_next_active` is how long this "
            "decision must stand before the account trades again."
        )


with account_tab:
    st.subheader("Account detail -- everything the models know about one client")
    st.caption(
        "Pick an account to see why it is routed the way it is: its behavioural "
        "classification, the model's own track record *on this specific account* (not just "
        "firm-wide), its full trading history, and the day-by-day routing decisions with the "
        "reason attached to each. Use this before overriding a routing decision -- the "
        "per-account accuracy below tells you whether the model has earned trust on this "
        "client, which the firm-wide scorecard cannot."
    )

    all_accounts = sorted(records["account_key"].dropna().unique())
    search = st.text_input("Filter accounts (substring of login or database)", value="")
    shortlist = [a for a in all_accounts if search.lower() in a.lower()] if search else all_accounts
    st.caption(f"{len(shortlist):,} of {len(all_accounts):,} accounts match.")

    if not shortlist:
        st.info("No account matches that filter.")
    else:
        selected = st.selectbox("Account", shortlist[:2000], index=0)
        account_records = records.loc[records["account_key"] == selected].copy()
        account_records["day"] = pd.to_datetime(account_records["timestamp"]).dt.floor("D")
        # `realised_trade_pnl` prefers net_profit (price P/L + commission + swap
        # + taxes) and falls back to raw profit where a source has no net column.
        account_records["realised_pnl"] = realised_trade_pnl(account_records)
        realised = account_records.loc[is_realised_trade(account_records)]
        account_assignment = assignment.loc[assignment["account_key"] == selected].copy()
        account_profile = profile.loc[profile["account_key"] == selected].sort_values("decision_day")
        account_detail = book_detail.loc[book_detail["account_key"] == selected].copy()

        total_pnl = float(realised["realised_pnl"].sum()) if not realised.empty else 0.0
        win_rate = float((realised["realised_pnl"] > 0).mean()) if not realised.empty else float("nan")
        latest_profile = account_profile.iloc[-1] if not account_profile.empty else None
        latest_book = account_assignment.sort_values("decision_day").iloc[-1] if not account_assignment.empty else None

        _kpi_row([
            ("Current book", str(latest_book["book"]) if latest_book is not None else "n/a",
             str(latest_book["reason_codes"]) if latest_book is not None else None),
            ("Client realised P/L", f"${total_pnl:,.0f}", "negative = firm profits if B-booked"),
            ("Closed trades", f"{len(realised):,}", f"{account_records['day'].nunique()} active days"),
            ("Win rate", f"{win_rate:.1%}" if win_rate == win_rate else "n/a", None),
            ("Firm P&L from this account", f"${account_detail['firm_pnl_usd'].sum():,.0f}" if not account_detail.empty else "n/a", None),
        ])

        st.markdown("#### Behavioural classification")
        if latest_profile is None:
            st.info("No point-in-time profile yet for this account.")
        else:
            flags = [
                ("Persistent edge", bool(latest_profile.get("expanding_edge_flag", False)),
                 "Consistently profitable over many closes -- always hedged (A-book), never taken the other side of."),
                ("Arbitrage pattern", bool(latest_profile.get("expanding_arbitrage_flag", False)),
                 "Rapid-fire and reversal-heavy: latency/price-feed exploitation shape. Always hedged."),
                ("Toxicity", bool(latest_profile.get("expanding_toxicity_flag", False)),
                 "High loss rate or heavy position concentration -- profitable to B-book, but concentrated risk."),
            ]
            for name, active, meaning in flags:
                st.markdown(f"{'ðŸ”´' if active else 'âšª'} **{name}** â€” {'FLAGGED' if active else 'not flagged'}. {meaning}")
            _kpi_row([
                ("Win rate (expanding)", f"{float(latest_profile.get('expanding_win_rate', float('nan'))):.1%}", None),
                ("Loss rate (expanding)", f"{float(latest_profile.get('expanding_loss_rate', float('nan'))):.1%}", None),
                ("Profit per notional", f"{float(latest_profile.get('expanding_profit_per_notional', float('nan'))):.6f}", None),
                ("Rapid-fire share", f"{float(latest_profile.get('expanding_rapid_share', float('nan'))):.1%}", "<=5s between events"),
                ("Reversal rate", f"{float(latest_profile.get('expanding_reversal_rate', float('nan'))):.1%}", None),
            ])

        st.markdown("#### Has the model actually been right about *this* account?")
        account_scored = expected_value_with_live.loc[
            (expected_value_with_live["account_key"] == selected)
            & expected_value_with_live["model_expected_client_profit"].notna()
            & expected_value_with_live["target_profit"].notna()
        ]
        if len(account_scored) < 3:
            st.warning(
                f"Only {len(account_scored)} scored day(s) for this account -- too few to judge the "
                "model's accuracy here. Treat its routing as low-confidence and lean on the "
                "behavioural flags above instead."
            )
        else:
            predicted = account_scored["model_expected_client_profit"].astype(float)
            actual = account_scored["target_profit"].astype(float)
            sign_accuracy = float((np.sign(predicted) == np.sign(actual)).mean())
            _kpi_row([
                ("Days scored OOS", f"{len(account_scored):,}", None),
                ("Direction correct", f"{sign_accuracy:.1%}", "50% = coin flip"),
                ("Mean predicted", f"${predicted.mean():,.0f}", None),
                ("Mean actual", f"${actual.mean():,.0f}", None),
                ("Mean abs error", f"${(predicted - actual).abs().mean():,.0f}", None),
            ])
            comparison = account_scored[["decision_day", "model_expected_client_profit", "target_profit"]].copy()
            comparison["decision_day"] = pd.to_datetime(comparison["decision_day"])
            st.altair_chart(
                _line_chart(comparison, "decision_day", ["model_expected_client_profit", "target_profit"],
                            [COLOR_BLUE, COLOR_ORANGE], "$ per day").properties(height=260),
                use_container_width=True,
            )

        st.markdown("#### Daily P/L and routing history")
        if not realised.empty:
            daily_pnl_account = realised.groupby("day", as_index=False).agg(
                realised_pnl=("realised_pnl", "sum"), trades=("realised_pnl", "size"),
            )
            daily_pnl_account["cumulative_pnl"] = daily_pnl_account["realised_pnl"].cumsum()
            st.altair_chart(
                _line_chart(daily_pnl_account, "day", ["cumulative_pnl"], [COLOR_AQUA], "Cumulative client P/L ($)").properties(height=240),
                use_container_width=True,
            )
        if not account_detail.empty:
            st.dataframe(
                account_detail[[c for c in ["decision_day", "book", "reason_codes", "firm_pnl_usd", "client_realised_profit",
                                            "excess_value_usd", "sigma_usd", "quality_score", "low_confidence"]
                                if c in account_detail.columns]].sort_values("decision_day", ascending=False),
                width="stretch",
            )

        with st.expander(f"Raw trade records ({len(account_records):,} rows)"):
            st.dataframe(account_records.sort_values("timestamp", ascending=False).head(500), width="stretch")

