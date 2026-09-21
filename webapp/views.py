"""Query helpers that turn a cached score artefact into what a screen needs.

Everything here reads the artefact and slices it. No function in this module
fits a model, which is what makes day / account / symbol filters instant and
keeps a control change from silently altering the numbers.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

#: Routing taxonomy. A dealer looking at a hedge list needs to know WHY an
#: account is on it -- a persistent-edge client and a one-off large position
#: both warrant hedging, but for different reasons and with different follow-up.
CATEGORY_STYLE = {
    "PERSISTENT_EDGE":  ("tag-edge", "Consistently profitable over a long record."),
    "LATENCY_ARBITRAGE": ("tag-arb", "Fills anticipate the move: speed advantage, not skill."),
    "TOXIC_FLOW":       ("tag-toxic", "Systematically adverse for the firm after entry."),
    "HIGH_MAGNITUDE":   ("tag-magnitude", "Position large enough that one day moves the book."),
    "SCALPER":          ("tag-neutral", "Very short holds -- execution-sensitive flow."),
    "MOMENTUM_WINNER":  ("tag-edge", "Recent form well ahead of the account's own history."),
    "MODEL_SIGNAL":     ("tag-neutral", "No single pattern dominates; the model expects a win."),
}


def classify(row: pd.Series) -> tuple[str, str]:
    """Assign a routing reason, most specific first.

    Thresholds are set to be defensible rather than generous: an earlier version
    tagged 137k account-days as latency arbitrage by firing on any positive
    reading, which is an accusation a dealing desk cannot act on.
    """
    closes = row.get("life_closes") or 0
    win_rate = row.get("life_win_rate")
    profit_factor = row.get("life_profit_factor")
    if closes >= 20 and pd.notna(profit_factor) and profit_factor > 1.1 and (row.get("pnl_life") or 0) >= 0:
        return "PERSISTENT_EDGE", (
            f"Profit factor {profit_factor:.2f} over {int(closes):,} closed trades"
            + (f", win rate {win_rate:.0%}." if pd.notna(win_rate) else "."))
    scalp = row.get("scalp_rate")
    if pd.notna(scalp) and scalp >= 0.5 and closes >= 50:
        return "SCALPER", f"{scalp:.0%} of trades held under five minutes across {int(closes):,} trades."
    if pd.notna(row.get("martingale_rate")) and row.get("martingale_rate") >= 0.2:
        return "TOXIC_FLOW", f"Doubling-down pattern on {row['martingale_rate']:.0%} of trades."
    notional = row.get("gross_notional")
    if pd.notna(notional) and notional >= (row.get("_notional_p90") or np.inf):
        return "HIGH_MAGNITUDE", f"Gross notional ${notional:,.0f} -- top decile for the day."
    if pd.notna(win_rate) and win_rate >= 0.6 and closes >= 20:
        return "MOMENTUM_WINNER", f"Win rate {win_rate:.0%} over {int(closes):,} trades."
    return "MODEL_SIGNAL", f"Model probability {row.get('score', float('nan')):.0%} against a ~52% base rate."


def available_days(frame: pd.DataFrame) -> list[str]:
    """Distinct trading days, newest first.

    Takes `.unique()` BEFORE converting to date. The previous order built 19.5M
    Python `date` objects and then discarded all but ~90 of them, costing ~13s
    per render for a list of ninety strings.
    """
    unique_days = pd.Series(frame["day"].unique()).sort_values(ascending=False)
    return unique_days.dt.strftime("%Y-%m-%d").tolist()


#: Row boundaries per day, keyed by frame identity. Comparing a 19.5M-row date
#: column on every request cost seconds; once the frame is sorted by day, a
#: slice is two binary searches.
_DAY_INDEX: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}


def _day_index(frame: pd.DataFrame):
    """(sorted unique days, start offsets) for a day-sorted frame, or None.

    Returns None when the frame is not sorted by day -- the caller then falls
    back to a mask. Sorting here would copy the whole frame, which is exactly
    the cost being avoided.
    """
    key = id(frame)
    cached = _DAY_INDEX.get(key)
    if cached is not None and cached[0] == len(frame):
        return cached[1], cached[2]

    days = pd.to_datetime(frame["day"]).dt.normalize().to_numpy()
    if len(days) > 1 and not (days[:-1] <= days[1:]).all():
        return None
    unique = np.unique(days)
    starts = np.searchsorted(days, unique)
    if len(_DAY_INDEX) > 8:
        _DAY_INDEX.clear()
    _DAY_INDEX[key] = (len(frame), unique, starts)
    return unique, starts


def day_slice(frame: pd.DataFrame, day: str | None) -> pd.DataFrame:
    if not day:
        return frame
    wanted = pd.Timestamp(day).normalize()

    index = _day_index(frame)
    if index is not None:
        unique, starts = index
        position = np.searchsorted(unique, np.datetime64(wanted))
        if position >= len(unique) or unique[position] != np.datetime64(wanted):
            return frame.iloc[0:0]
        start = starts[position]
        end = starts[position + 1] if position + 1 < len(starts) else len(frame)
        return frame.iloc[start:end]

    return frame.loc[pd.to_datetime(frame["day"]).dt.normalize() == wanted]


def eligible_population(frame: pd.DataFrame, day: str | None,
                        max_stale_days: int = 30) -> pd.DataFrame:
    """Every account the desk could route on `day`, carried forward.

    A routing decision has to be made BEFORE the session, when nobody knows who
    will trade. Scoring only accounts that turn out to be active that day is
    both circular (it uses tomorrow's activity to pick today's candidates) and
    far too narrow -- it was showing ~6k of ~29k accounts.

    So the population is every account with a scored observation on or before
    the decision day, reduced to its MOST RECENT one. An account that last
    traded a week ago is still on the book and can trade tomorrow; its latest
    known behaviour is the best available estimate of what it will do.

    `max_stale_days` drops accounts dormant long enough that their features no
    longer describe anything current. `days_stale` is carried through so a
    dealer can see how fresh each row's evidence is.
    """
    working = frame.copy()
    working["day"] = pd.to_datetime(working["day"])
    if day:
        cutoff = pd.Timestamp(day).normalize()
        working = working.loc[working["day"] <= cutoff]
    else:
        cutoff = working["day"].max()
    if working.empty:
        return working

    latest = (working.sort_values("day")
                     .groupby("account_key", observed=True, as_index=False)
                     .tail(1)
                     .copy())
    latest["days_stale"] = (cutoff - latest["day"]).dt.days
    latest = latest.loc[latest["days_stale"] <= max_stale_days]
    return latest.reset_index(drop=True)


def abook_manifest(frame: pd.DataFrame, day: str | None, hedge_fraction: float,
                   probability_threshold: float = 0.0, limit: int = 500,
                   max_stale_days: int = 30) -> pd.DataFrame:
    """The accounts to A-book, ordered by impact on profitability.

    Ranked by expected dollar impact rather than raw probability: a confident
    call on a tiny account is worth nothing to the desk, and a dealer working
    down a list should meet the consequential names first.
    """
    population = eligible_population(frame, day, max_stale_days)
    if population.empty:
        return population
    scores = population["score"].to_numpy(dtype="float64")
    if probability_threshold > 0:
        selected = scores >= probability_threshold
    else:
        selected = scores >= np.quantile(scores, 1 - max(1e-6, hedge_fraction))
    chosen = population.loc[selected].copy()
    if chosen.empty:
        return chosen

    # Expected impact = confidence above the base rate, scaled by how much this
    # account typically puts at risk.
    # Exposure scale for ranking. `gross_notional` is unavailable on the
    # exposure-day frame (it has no trade sizes), so fall back in order of
    # fidelity: the account's own recent P&L magnitude, then live position
    # count, then a flat weight. Ranking by raw confidence alone would put a
    # certain call on a tiny account above a likely one on a large one, which is
    # the opposite of what a dealer working down the list needs.
    exposure = None
    for candidate in ("gross_notional", "pnl_20d", "live_positions"):
        if candidate in chosen.columns:
            values = pd.to_numeric(chosen[candidate], errors="coerce").abs()
            if values.notna().any() and values.sum() > 0:
                exposure = values
                break
    if exposure is None:
        exposure = pd.Series(1.0, index=chosen.index)
    chosen["expected_impact"] = (chosen["score"] - 0.5).clip(lower=0) * exposure.fillna(0).abs()
    chosen["_notional_p90"] = exposure.quantile(0.9) if exposure.notna().any() else np.inf
    categories = chosen.apply(classify, axis=1, result_type="expand")
    chosen["category"] = categories[0]
    chosen["reason"] = categories[1]
    chosen["css"] = chosen["category"].map(lambda c: CATEGORY_STYLE.get(c, ("tag-neutral", ""))[0])
    return chosen.sort_values("expected_impact", ascending=False).head(limit)


def realised_day_outcome(frame: pd.DataFrame, day: str | None,
                         hedged_accounts: set[str]) -> dict:
    """What following the manifest actually earned on `day`.

    The manifest is a decision; this is the scoreboard. Compares the firm's P&L
    from B-booking everything against the P&L from hedging exactly the accounts
    on the list, using that day's REALISED client outcomes. Only accounts that
    actually traded contribute -- a hedge on an account that stayed flat costs
    and earns nothing.
    """
    actual = day_slice(frame, day)
    if actual.empty:
        return {}
    pnl = actual["pnl"].to_numpy(dtype="float64")
    was_hedged = actual["account_key"].isin(hedged_accounts).to_numpy()
    flat_firm = float(-pnl.sum())
    model_firm = float(-pnl[~was_hedged].sum())
    return {
        "active_accounts": int(actual["account_key"].nunique()),
        "hedged_active": int(actual.loc[was_hedged, "account_key"].nunique()),
        "flat_firm_pnl": flat_firm,
        "model_firm_pnl": model_firm,
        "delta": model_firm - flat_firm,
        # What the hedge cost or saved, split so a dealer can see both sides:
        # money not paid out to winners, against profit given up on losers.
        "avoided_losses": float(-pnl[was_hedged & (pnl > 0)].sum()),
        "forfeited_profit": float(-pnl[was_hedged & (pnl < 0)].sum()),
    }


def trade_signals(frame: pd.DataFrame, day: str | None, hedge_fraction: float,
                  probability_threshold: float = 0.0, limit: int = 500) -> pd.DataFrame:
    """Individual trades to copy/hedge for a day, largest expected impact first."""
    today = day_slice(frame, day).copy()
    if today.empty:
        return today
    scores = today["score"].to_numpy(dtype="float64")
    if probability_threshold > 0:
        selected = scores >= probability_threshold
    else:
        selected = scores >= np.quantile(scores, 1 - max(1e-6, hedge_fraction))
    today = today.loc[selected].copy()
    if today.empty:
        return today
    today["expected_impact"] = (today["score"] - 0.5).clip(lower=0) * today["notional"].abs()
    return today.sort_values("expected_impact", ascending=False).head(limit)


#: Path-aware exit report, keyed by the study file's mtime.
_EXIT_REPORT_CACHE: dict[str, tuple] = {}


def exit_policy_report(frame: pd.DataFrame, hedge_fraction: float,
                       probability_threshold: float = 0.0) -> dict:
    """Where copied trades should close, measured on the trades we would copy.

    Two things this deliberately does NOT do.

    It does not evaluate policies across all client flow. The desk chooses an
    exit only for trades it has taken on, and the full population is dominated
    by trades the model would never have copied -- so a policy that looked good
    there would be answering a different question.

    It does not pretend a truncated loss is a real backtest. The warehouse
    stores entry and exit, not the path between, so a stop is applied to the
    REALISED outcome: a trade that dipped through the stop and recovered is not
    counted as stopped. That flatters stops, and is stated on the page rather
    than buried here.
    """
    from webapp import exit_policy

    # A path-aware study (minute-bar MAE/MFE per copied trade) supersedes the
    # realised-only evaluation whenever one has been built: it is the only
    # version whose stop and target rows mean anything. `exit_path_study`
    # writes it; the page picks it up with no further wiring.
    from webapp import model_service
    path_file = model_service.SCRATCH / "quant_mae_mfe.parquet"
    if path_file.exists():
        # Comparing every policy over ~1M path-aware trades costs ~40s and the
        # answer changes only when the study file does -- cache on its mtime.
        stamp = path_file.stat().st_mtime
        cached = _EXIT_REPORT_CACHE.get("report")
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            study = pd.read_parquet(path_file)
            if len(study) and {"mae", "mfe", "pnl", "day"} <= set(study.columns):
                table = exit_policy.compare(study)
                baseline = table.loc[table["policy"] == "mirror"].iloc[0]
                days = pd.to_datetime(study["day"]).dt.normalize()
                daily = (pd.to_numeric(study["pnl"], errors="coerce")
                         .fillna(0.0).groupby(days).sum())
                report = {
                    "available": True,
                    "rule": "path-aware study: top 10% by score, minute-bar excursions",
                    "trades": int(len(study)),
                    "share_of_flow": float("nan"),
                    "has_hold": "hold_hours" in study.columns,
                    "path_aware": True,
                    "days": int(len(daily)),
                    "losing_days": int((daily < 0).sum()),
                    "baseline": baseline.to_dict(),
                    "rows": table.to_dict("records"),
                    "dominating": int(table["dominates"].sum()),
                }
                _EXIT_REPORT_CACHE["report"] = (stamp, report)
                return report
        except Exception:
            pass  # fall through to the realised-only evaluation

    if frame is None or frame.empty or "score" not in frame.columns:
        return {"available": False}

    scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(scores)
    if not finite.any():
        return {"available": False}

    if probability_threshold > 0:
        selected = finite & (scores >= probability_threshold)
        rule = f"probability >= {probability_threshold:.2f}"
    else:
        cutoff = float(np.quantile(scores[finite], 1 - max(1e-6, hedge_fraction)))
        selected = finite & (scores >= cutoff)
        rule = f"top {hedge_fraction:.1%} by score (cutoff {cutoff:.3f})"

    copied = frame.loc[selected].copy()
    if copied.empty:
        return {"available": False, "rule": rule}

    # Holding period, where the frame carries both ends of the trade. Without it
    # the time-based policy is simply not offered rather than guessed at.
    if "open_time" in copied.columns and "close_time" in copied.columns:
        hold = (pd.to_datetime(copied["close_time"], errors="coerce")
                - pd.to_datetime(copied["open_time"], errors="coerce"))
        copied["hold_hours"] = hold.dt.total_seconds() / 3600.0

    table = exit_policy.compare(copied)
    baseline = table.loc[table["policy"] == "mirror"].iloc[0]
    path_aware = bool(table["path_aware"].any())

    days = pd.to_datetime(copied["day"]).dt.normalize()
    daily = pd.to_numeric(copied["pnl"], errors="coerce").fillna(0.0).groupby(days).sum()
    return {
        "available": True,
        "rule": rule,
        "trades": int(len(copied)),
        "share_of_flow": float(len(copied) / max(1, len(frame))),
        "has_hold": "hold_hours" in copied.columns,
        "path_aware": path_aware,
        # Drawdown over 71 days with one losing day is not a drawdown estimate.
        # The window is reported so nobody reads a flattering Calmar as durable.
        "days": int(len(daily)),
        "losing_days": int((daily < 0).sum()),
        "baseline": baseline.to_dict(),
        "rows": table.to_dict("records"),
        "dominating": int(table["dominates"].sum()),
    }


def account_history(frame: pd.DataFrame, account_key: str) -> pd.DataFrame:
    subset = frame.loc[frame["account_key"] == account_key].copy()
    if subset.empty:
        # Brand-new account (or the tail past the scores-frame edge): synthesise
        # the daily curve straight from live closed trades so the detail page
        # is never blank just because the model frame hasn't been refreshed yet.
        return _live_account_history(account_key)
    subset = subset.sort_values("day")
    subset["cum_pnl"] = subset["pnl"].cumsum()
    return subset


def _live_account_history(account_key: str) -> pd.DataFrame:
    """Daily P&L curve built from live closed trades (net_profit summed per day).
    `score` is left NaN -- these trades post-date the scored frame, so the panel
    honestly shows 'live, not yet scored' rather than a fabricated probability."""
    trades = account_trades(account_key, limit=5000)
    if trades.empty or "net_profit" not in trades.columns:
        return pd.DataFrame(columns=["account_key", "day", "pnl", "cum_pnl", "score"])
    trades = trades.dropna(subset=["open_time"]).copy()
    trades["day"] = pd.to_datetime(trades["open_time"]).dt.normalize()
    daily = (trades.groupby("day", as_index=False)["net_profit"].sum()
             .rename(columns={"net_profit": "pnl"}).sort_values("day"))
    daily["account_key"] = account_key
    daily["cum_pnl"] = daily["pnl"].cumsum()
    daily["score"] = float("nan")
    return daily


#: Post-trade markout horizons, in ascending order. Every one is on by default:
#: the shape ACROSS horizons is the signal -- a move that appears at 1m and
#: decays is latency, one that builds to 1d is genuine information.
MARKOUT_HORIZONS = ("1m", "5m", "30m", "1h", "4h", "1d", "3d")

#: Each horizon in MINUTES. The chart plots against real elapsed time on a log
#: scale rather than equal-spaced categories: 1m to 3d spans four orders of
#: magnitude, and spacing them evenly makes a fast decay and a slow build look
#: identical, which is exactly the distinction the curve exists to show.
HORIZON_MINUTES = {"1m": 1, "5m": 5, "30m": 30, "1h": 60,
                   "4h": 240, "1d": 1440, "3d": 4320}
#: Pre-trade run-up sits five minutes BEFORE entry.
PRE_TRADE_MINUTES = -5

#: Pre-trade context. Only a 5-minute run-up was computed, so "pre-trade" here
#: is one point rather than a curve -- stated plainly rather than padded out
#: with horizons that do not exist.
PRE_TRADE_COLUMN = "runup_5m"

_MARKOUT_CACHE: dict[str, pd.DataFrame] = {}
#: Serialises the cache FILL, not the reads. The file is ~80 MB on OneDrive and
#: takes ~37 s to load; without this every concurrent first-hit request started
#: its own copy, so three analysts opening Account Detail at once each waited
#: for three simultaneous 80 MB reads instead of one (observed 17 Sep 2026: a
#: single page took 3.4 minutes). The double check inside means the waiters
#: return the frame the winner loaded rather than reloading it.
_MARKOUT_LOCK = threading.Lock()


def load_markouts(path) -> pd.DataFrame:
    """Per account-day markouts, cached in-process (the file is ~460k rows)."""
    key = str(path)
    if key in _MARKOUT_CACHE:
        return _MARKOUT_CACHE[key]
    with _MARKOUT_LOCK:
        if key not in _MARKOUT_CACHE:
            frame = pd.read_parquet(path)
            frame["day"] = pd.to_datetime(frame["day"])
            for column in ([f"markout_{h}" for h in MARKOUT_HORIZONS]
                           + [PRE_TRADE_COLUMN]):
                if column in frame:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
            _MARKOUT_CACHE[key] = frame
    return _MARKOUT_CACHE[key]


def account_markouts(markouts: pd.DataFrame, account_key: str,
                     horizons: tuple[str, ...] = MARKOUT_HORIZONS) -> dict:
    """Average markout per selected horizon for one account.

    Only the selected horizons are averaged -- a deselected horizon is excluded
    entirely rather than being folded into a blended number, so the curve always
    means exactly what the chips say.
    """
    subset = markouts.loc[markouts["account_key"] == account_key]
    if subset.empty:
        return {}
    result = {
        "trades": int(subset["context_trades"].fillna(0).sum()),
        "days": int(len(subset)),
        "pre": float(subset[PRE_TRADE_COLUMN].mean()) if PRE_TRADE_COLUMN in subset else None,
        "post": {},
        "spread": float(subset["avg_relative_spread"].mean())
        if "avg_relative_spread" in subset else None,
    }
    # Always emitted in ascending time order, regardless of the order the chips
    # were clicked in -- the querystring carries no ordering guarantee and an
    # out-of-sequence curve is unreadable.
    for horizon in sorted(horizons, key=lambda h: HORIZON_MINUTES.get(h, 0)):
        column = f"markout_{horizon}"
        if column in subset and subset[column].notna().any():
            result["post"][horizon] = {
                "value": float(subset[column].mean()),
                "minutes": HORIZON_MINUTES.get(horizon, 0),
            }
    for column in ("markout_persistence", "anticipation_5m", "markout_positive_share"):
        if column in subset and subset[column].notna().any():
            result[column] = float(subset[column].mean())
    return result


_LIVE_MK_CACHE: dict = {}


def live_account_markouts(account_key: str, symbol: str | None = None) -> dict:
    """Live markouts from the tick tape, cached — INCLUDING failures.

    Caching only successes looks prudent and is the opposite here. The tick
    store is DuckDB and allows a single writer, so in the beta instance this
    call cannot ever succeed while the scanning instance holds the file; the
    old code therefore re-ran ~5.5 s of doomed work on EVERY account page load,
    which is most of why Account Detail took minutes to open (17 Sep 2026).
    A failure is cached for a tenth as long as a success: long enough to stop
    the per-request bleed, short enough that the page recovers on its own once
    the store frees up.
    """
    import time as _time
    key = (account_key, symbol)
    cached = _LIVE_MK_CACHE.get(key)
    if cached:
        # 300 s on failure, not 60: a retry costs 5-12 s of work that cannot
        # succeed while another instance holds the store, and at 60 s one
        # unlucky user per minute wore that cost. Still self-healing -- the
        # page recovers within five minutes of the store actually freeing up.
        ttl = 300 if cached[1].get("_err") else 600
        if _time.time() - cached[0] < ttl:
            return cached[1]
    result = _live_account_markouts_inner(account_key, symbol)
    _LIVE_MK_CACHE[key] = (_time.time(), result)
    return result


def _live_account_markouts_inner(account_key: str,
                                 symbol: str | None = None) -> dict:
    """Markout profile computed ON DEMAND from the live tick tape for accounts
    the precomputed parquet has never seen (brand-new / single-day clients).
    Same shape as account_markouts(); horizons limited to what 7-day quote
    retention supports."""
    # SUB-MINUTE horizons included: the latency question lives at seconds.
    horizons = {"5s": 5 / 60, "15s": 0.25, "30s": 0.5,
                "1m": 1, "5m": 5, "30m": 30, "1h": 60}
    trades = account_trades(account_key, symbol or None, limit=2000)
    if trades is None or trades.empty:
        return {"_err": "no trades at all"}
    from datetime import datetime, timedelta
    recent = trades.loc[pd.to_datetime(trades["open_time"])
                        >= datetime.utcnow() - timedelta(days=7)].copy()
    recent = recent.loc[pd.to_numeric(recent["open_price"],
                                      errors="coerce") > 0]
    if recent.empty:
        return {"_err": f"no priced trades in 7d ({len(trades)} total)"}
    try:
        from webapp.trade_feed import _canonical
        from webapp.kafka_service import shared_cursor as _store
        probe = pd.DataFrame({
            "rid": np.arange(len(recent), dtype=np.int64),
            "canonical": [(_canonical(s) or s)
                          for s in recent["symbol"].astype(str)],
            "open_time": pd.to_datetime(recent["open_time"])
                .astype("datetime64[us]").to_numpy(),
            "open_price": pd.to_numeric(recent["open_price"],
                                        errors="coerce").astype(float).to_numpy(),
            "direction": np.where(recent["cmd"].astype(str).str.lower()
                                  .str.startswith("b"), 1.0, -1.0),
        })
        # PLAIN-COLUMN ASOF targets: an expression on the inequality side of
        # an ASOF join matches unreliably; precomputed columns always do.
        for name, minutes in horizons.items():
            probe[f"t_{name}"] = probe["open_time"] \
                + np.timedelta64(int(round(minutes * 60)), "s")
        post = {}
        matched = 0
        with _store() as cx:
            cx.execute("CREATE OR REPLACE TEMP TABLE _am AS SELECT * FROM probe")
            for name, minutes in horizons.items():
                # COALESCE: many store rows carry only bid/ask, mid NULL.
                # Freshness guard: a match older than the entry itself is a
                # coverage-gap artifact, not a markout.
                joined = cx.execute(f"""
                    SELECT t.rid,
                           COALESCE(q.mid, (q.bid + q.ask) / 2) AS px,
                           q.event_time AS qt
                    FROM _am t ASOF LEFT JOIN quotes q
                      ON q.canonical = t.canonical
                     AND q.event_time <= t.t_{name}
                """).df().set_index("rid")
                qt = pd.to_datetime(joined["qt"]).reindex(probe["rid"])
                px = pd.to_numeric(joined["px"], errors="coerce") \
                    .reindex(probe["rid"])
                opens = pd.Series(pd.to_datetime(probe["open_time"]).values,
                                  index=probe["rid"])
                fresh = qt >= (opens - pd.Timedelta(seconds=5))
                got = px.where(fresh).to_numpy()
                matched = max(matched, int(np.isfinite(got).sum()))
                mk = (probe["direction"].to_numpy()
                      * (got - probe["open_price"].to_numpy())
                      / probe["open_price"].to_numpy())
                if np.isfinite(mk).any():
                    post[name] = {"value": float(np.nanmean(mk)),
                                  "minutes": minutes}
        if not post:
            return {"_err": f"no tick coverage for "
                            f"{sorted(set(probe['canonical']))[:4]} "
                            f"({len(recent)} trades, {matched} quote matches)"}
        return {"trades": int(len(recent)),
                "days": int(pd.to_datetime(recent["open_time"])
                            .dt.date.nunique()),
                "pre": None, "post": post, "spread": None, "live": True}
    except Exception as error:
        return {"_err": f"{type(error).__name__}: {error}"}


def add_canonical_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `canonical_symbol`, collapsing broker decorations.

    Brokers publish the same instrument under many tickers -- gold appears as
    XAUUSD, XAUUSDe, XAUUSDmin, XAUUSD247, XAUUSDs and XAUUSDx across the six
    servers, and XAUUSDe alone carries 958,787 trades. Aggregating risk by raw
    ticker therefore splits a single exposure into six, understating every
    per-symbol number.

    Mapped on the DISTINCT tickers only (a few hundred) and broadcast by code;
    running the resolver per row would cost minutes on 19.5M rows.
    """
    from trading_data.research import canonical_symbol

    if "symbol" not in frame.columns:
        return frame
    codes, uniques = pd.factorize(frame["symbol"], sort=False)
    mapped = pd.Index([canonical_symbol(s) for s in uniques]).to_numpy()
    frame = frame.copy()
    frame["canonical_symbol"] = mapped[codes]
    return frame


_TRADE_CACHE: dict[str, pd.DataFrame] = {}


def _load_server_trades(database: str) -> pd.DataFrame:
    """Raw trades for one server, cached in-process.

    Loaded lazily and per server so opening one account never pulls the whole
    90-day, 8-server tape into memory.
    """
    from webapp.model_service import SCRATCH
    if database not in _TRADE_CACHE:
        columns = ["account_key", "symbol", "cmd", "volume_lots", "open_time", "close_time",
                   "open_price", "close_price", "sl", "tp", "net_profit", "state", "reason"]
        frame = pd.read_parquet(SCRATCH / "bq_90d_records.parquet", columns=columns,
                                filters=[("database", "==", database)])
        frame["account_key"] = frame["account_key"].astype(str)
        _TRADE_CACHE[database] = frame
    return _TRADE_CACHE[database]


_ACCT_LIVE_CACHE: dict = {}          # account -> (fetched_at, frame)
_ACCT_LIVE_TTL = 120.0               # seconds; brief, so the drawer feels live


def _live_account_trades(account_key: str) -> pd.DataFrame:
    """This account's trades since the BQ snapshot edge, from production MySQL.
    Cached briefly. This is what makes a brand-new account (or the last two
    days of any account) actually appear -- the snapshot ends ~Aug 27."""
    import time as _time
    hit = _ACCT_LIVE_CACHE.get(account_key)
    if hit and _time.time() - hit[0] < _ACCT_LIVE_TTL:
        return hit[1]
    frame = pd.DataFrame()
    try:
        from webapp import vantage
        # overlap the snapshot by a couple of days so nothing falls in a gap
        since = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=9)
        recent = vantage._recent_closed_for(account_key, since)
        if recent is not None and len(recent):
            frame = recent
        failed = recent is None
    except Exception:
        frame, failed = pd.DataFrame(), True
    if len(_ACCT_LIVE_CACHE) > 500:
        _ACCT_LIVE_CACHE.clear()
    # A failed MySQL read is NOT cached: caching it left a latency client's
    # page with no trades table for two minutes (mt4_live04:6875852).
    if not failed:
        _ACCT_LIVE_CACHE[account_key] = (_time.time(), frame)
    return frame


def account_trades(account_key: str, symbol: str | None = None, limit: int = 500,
                   pinned_orders=None) -> pd.DataFrame:
    """The account's filled trades, newest first, capped at `limit` -- plus any
    `pinned_orders` (e.g. latency-flagged orders) older than the cap, so every
    flagged order stays clickable."""
    database = account_key.split(":", 1)[0]
    try:
        frame = _load_server_trades(database)
        snapshot = frame.loc[frame["account_key"] == account_key].copy()
    except Exception:
        snapshot = pd.DataFrame()
    live = _live_account_trades(account_key)
    parts = [p for p in (snapshot, live) if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    subset = pd.concat(parts, ignore_index=True)
    subset["open_time"] = pd.to_datetime(subset["open_time"], errors="coerce")
    # DEDUPE ON THE ORDER ID, not on trade attributes. The snapshot and the live
    # window overlap, so the same trade can arrive twice and one of them has to
    # go -- but an account that splits a position across several tickets opens
    # them in the SAME SECOND, on the same symbol, at the same size, so an
    # attribute key treats those distinct trades as one. mt4_live04:6874451 lost
    # 5 of its 28 trades that way (found 2026-09-18), including a 5-lot XAUUSD
    # pair worth $3,370 and $3,345 -- and order splitting is precisely the
    # behaviour these pages exist to surface. Two of the five were identical on
    # every column except the ticket, so no attribute key can separate them.
    # The snapshot carries no order id (ids are attached later, by
    # _attach_orders), so those rows still fall back to the attribute key.
    attr_key = [c for c in ("account_key", "symbol", "open_time", "volume_lots")
                if c in subset.columns]
    if "order" in subset.columns:
        ident = subset["order"].map(
            lambda v: "" if v is None or (isinstance(v, float) and v != v)
            else str(v).strip())
        has_id = ident.ne("") & ~ident.str.lower().isin(("nan", "none", "<na>"))
        keyed = subset.loc[has_id].drop_duplicates(subset=["order"], keep="first")
        rest = subset.loc[~has_id]
        if len(rest) and attr_key:
            rest = rest.drop_duplicates(subset=attr_key, keep="first")
            # An un-identified row that repeats a trade we already have with an
            # id is the same trade seen through the other source -- drop it,
            # otherwise the overlap the dedupe exists for would come back.
            seen = set(map(tuple, keyed[attr_key].itertuples(index=False, name=None)))
            rest = rest.loc[[tuple(r) not in seen
                             for r in rest[attr_key].itertuples(index=False, name=None)]]
        subset = pd.concat([keyed, rest], ignore_index=True) if len(rest) else keyed
    elif attr_key:
        subset = subset.drop_duplicates(subset=attr_key, keep="first")
    if symbol:
        subset = subset.loc[subset["symbol"].astype(str) == symbol]
    # FILLED TRADES ONLY. The BigQuery snapshot also carries balance/credit
    # records (no symbol, price 0) and cancelled or expired pending orders
    # (buy_limit / sell_limit ... with P&L 0) -- on mt4_live02:2948283 these
    # were 24 of 500 rows, shown as trades with no order id.
    if "cmd" in subset.columns:
        subset = subset.loc[subset["cmd"].astype(str).str.lower().isin(["buy", "sell"])]
    subset = subset.loc[subset["symbol"].notna()
                        & (pd.to_numeric(subset["open_price"], errors="coerce") > 0)]
    if subset.empty:
        return subset
    subset["net_profit"] = pd.to_numeric(subset["net_profit"], errors="coerce")
    # Duration and INSTRUMENT-UNIT notional (lots x contract size -- units of
    # the thing traded, not USD): both belong to the trade row itself.
    close = pd.to_datetime(subset.get("close_time"), errors="coerce")
    subset["duration_s"] = (close - subset["open_time"]).dt.total_seconds()
    subset["notional_units"] = (
        pd.to_numeric(subset["volume_lots"], errors="coerce")
        * subset["symbol"].astype(str).map(_contract_units))
    subset = subset.sort_values("open_time", ascending=False)
    pinned = {str(o) for o in (pinned_orders or []) if str(o)}
    if not pinned:
        return _attach_orders(subset.head(limit), account_key)
    subset = _attach_orders(subset, account_key)
    keep = subset.head(limit)
    extra = subset.iloc[limit:].loc[lambda f: f["order"].isin(pinned)]
    parts = [keep, extra]
    # Flagged orders the table sources do not reach (a very active account's
    # live window can end before its flagged trades): read them straight from
    # the warehouse by order id so every flagged order stays clickable.
    absent = pinned - set(keep["order"]) - set(extra["order"])
    if absent:
        fetched = _warehouse_orders(account_key, absent, subset.columns)
        if len(fetched):
            parts.append(fetched)
    return pd.concat(parts, ignore_index=True).sort_values("open_time", ascending=False)


def account_pnl_impact(account_key: str, latency_orders=None, toxic_orders=None,
                       window_days: float = 7.0) -> dict:
    """Realised P&L on record and the flagged-flow impact for one account.

    Realised P&L sums every closed trade the warehouse holds for the account
    (trading profit + commission + swap; cent accounts already deflated to
    USD). The impact block uses the engines' own window (closed in the last
    `window_days`): flagged trades over all trades, and the P&L the flagged
    trades made."""
    from webapp import data_store
    try:
        database, login = account_key.split(":", 1)
        login = int(login)
    except Exception:
        return {}
    cols = ["order", "open_time", "close_time", "net_profit", "commission", "storage"]
    parts = []
    # A PARTITION THAT WILL NOT READ MUST BE REPORTED, NOT SKIPPED. This used to
    # `continue` silently, so a corrupt month was indistinguishable from a month
    # with no trades: the panel rendered "Realised -- / 0 closed trades" beside a
    # table listing 33 of them, and nothing anywhere said why. Found 2026-09-18
    # on mt4_live02, whose 2026-09.parquet had been truncated since 17 Sep.
    unreadable = []
    for path in sorted((data_store.WAREHOUSE / database).glob("*.parquet")):
        try:
            f = pd.read_parquet(path, columns=cols, filters=[("login", "==", login)])
        except Exception as error:
            unreadable.append({"file": path.name,
                               "error": f"{type(error).__name__}: {error}"})
            continue
        if len(f):
            parts.append(f)
    if not parts:
        # Still distinguish "nothing to show" from "could not read it".
        return {"unreadable": unreadable} if unreadable else {}
    t = pd.concat(parts, ignore_index=True)
    t["order"] = pd.to_numeric(t["order"], errors="coerce")
    t = t.drop_duplicates("order") if t["order"].notna().all() else t
    for c in ("net_profit", "commission", "storage"):
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0.0)
    t["net"] = t["net_profit"] + t["commission"] + t["storage"]
    t["close_time"] = pd.to_datetime(t["close_time"], errors="coerce")
    t["order_s"] = t["order"].map(lambda v: str(int(v)) if pd.notna(v) else "")
    out = {"realized": {
        "trades": int(len(t)),
        "trading_pnl": round(float(t["net_profit"].sum()), 2),
        "commission": round(float(t["commission"].sum()), 2),
        "swap": round(float(t["storage"].sum()), 2),
        "net_pnl": round(float(t["net"].sum()), 2),
        "since": str(t["close_time"].min())[:10], "until": str(t["close_time"].max())[:16]}}
    if unreadable:
        # Partial data: some months read, others did not. Say so -- these totals
        # are real but INCOMPLETE, which is worse than obviously empty.
        out["unreadable"] = unreadable

    # Daily realised equity path, from the SAME closed trades as the figures
    # above and keyed on CLOSE time -- that is when the money is realised.
    # Account Detail used to draw this curve from the model scores frame, which
    # only covers the scored window and excludes commission and swap, so its
    # level disagreed with the realised total sitting right beside it. Built
    # here rather than in a second function because the warehouse read above is
    # the expensive part and it has already happened.
    daily = (t.dropna(subset=["close_time"])
              .assign(day=lambda d: d["close_time"].dt.normalize())
              .groupby("day", as_index=False)["net"].sum()
              .sort_values("day"))
    out["equity"] = {
        "days": [d.strftime("%Y-%m-%d") for d in daily["day"]],
        "cum": [round(float(v), 2) for v in daily["net"].cumsum()],
    }

    start = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=window_days)
    w = t.loc[t["close_time"] >= start]
    n = int(len(w))
    out["window"] = {"days": window_days, "trades": n,
                     "net_pnl": round(float(w["net"].sum()), 2),
                     "trading_pnl": round(float(w["net_profit"].sum()), 2)}

    def impact(orders) -> dict:
        wanted = {str(o) for o in (orders or [])}
        hit = w.loc[w["order_s"].isin(wanted)]
        gross_win = float(w["net_profit"].clip(lower=0).sum())
        return {"flagged": int(len(hit)), "total": n,
                "share": round(len(hit) / n, 4) if n else None,
                "flagged_pnl": round(float(hit["net_profit"].sum()), 2),
                "flagged_net_pnl": round(float(hit["net"].sum()), 2),
                "share_of_profit": (round(float(hit["net_profit"].clip(lower=0).sum()) / gross_win, 4)
                                    if gross_win > 0 else None),
                # Flagged orders outside the window (still listed by the scan).
                "outside_window": int(len(wanted) - len(hit))}

    out["latency"] = impact(latency_orders)
    out["toxic"] = impact(toxic_orders)
    # Toxic trades in the window by s5.3 signature ("<sig>|<horizons>" kinds).
    if isinstance(toxic_orders, dict):
        in_window = set(w["order_s"])
        sigs = [str(k).split("|")[0] for o, k in toxic_orders.items()
                if str(o) in in_window and "|" in str(k)]
        out["toxic"]["by_signature"] = {s: sigs.count(s) for s in ("sharp_fast", "persistent", "other")} \
            if sigs else {}
    # Trades both engines flag: a fast advantage that fades is also materially
    # adverse at its early horizons, so the two sets overlap by design.
    both = ({str(o) for o in (latency_orders or [])} & {str(o) for o in (toxic_orders or [])}) \
        & set(w["order_s"])
    out["both"] = {"flagged": len(both),
                   "flagged_pnl": round(float(w.loc[w["order_s"].isin(both), "net_profit"].sum()), 2)}
    return out


def _warehouse_orders(account_key: str, orders: set, columns) -> pd.DataFrame:
    """Trades for specific order ids from the warehouse (last 3 close months),
    shaped like account_trades rows."""
    try:
        from webapp import data_store
        database, login = account_key.split(":", 1)
        wanted = {int(o) for o in orders if str(o).isdigit()}
        now = pd.Timestamp.utcnow().tz_localize(None)
        frames = []
        for period in pd.period_range((now - pd.Timedelta(days=70)).to_period("M"), now.to_period("M"), freq="M"):
            path = data_store.WAREHOUSE / database / f"{period}.parquet"
            if path.exists():
                f = pd.read_parquet(path, filters=[("login", "==", int(login))])
                frames.append(f.loc[pd.to_numeric(f["order"], errors="coerce").isin(wanted)])
        if not frames:
            return pd.DataFrame(columns=columns)
        out = pd.concat(frames, ignore_index=True).drop_duplicates("order")
        out["account_key"] = account_key
        out["order"] = out["order"].map(lambda v: str(int(v)))
        out["open_time"] = pd.to_datetime(out["open_time"])
        out["close_time"] = pd.to_datetime(out["close_time"])
        out["duration_s"] = (out["close_time"] - out["open_time"]).dt.total_seconds()
        out["notional_units"] = (pd.to_numeric(out["volume_lots"], errors="coerce")
                                 * out["symbol"].astype(str).map(_contract_units))
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        return out[list(columns)]
    except Exception:
        return pd.DataFrame(columns=columns)


def _attach_orders(subset: pd.DataFrame, account_key: str) -> pd.DataFrame:
    """Order ids from the warehouse (MT4 ticket; MT5 exit deal) for trades
    whose source carries none. The BigQuery snapshot has no order id, and the
    order is what a client and the evidence report refer to.

    Matched on symbol + open second + open price. NOT on lots: for cent
    accounts the warehouse stores lots 100x smaller than the snapshot (0.0001
    vs 0.01), and some accounts switch scale mid-period -- a lots key left
    mt4_live02:2402711 with 0 of 500 ids (15 Sep 2026). Lots only break ties
    between same-second, same-price trades, and each order is used once."""
    subset = subset.copy()

    def clean(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        text = str(v).strip()
        if text.lower() in ("", "nan", "none", "<na>"):
            return ""
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    if "order" in subset.columns:
        subset["order"] = subset["order"].map(clean).astype(object)
    else:
        subset["order"] = pd.Series("", index=subset.index, dtype=object)
    missing = subset["order"] == ""
    if not missing.any():
        return subset
    try:
        from webapp import data_store
        database, login = account_key.split(":", 1)
        opens = pd.to_datetime(subset["open_time"], errors="coerce").dropna()
        closes = pd.to_datetime(subset.get("close_time"), errors="coerce").dropna()
        if not len(opens):
            return subset
        # The warehouse files a trade under its CLOSE month: a position still
        # open in the snapshot may have closed months later, so read to now.
        lo, hi = opens.min(), pd.Timestamp.utcnow().tz_localize(None)
        parts = []
        for period in pd.period_range(lo.to_period("M"), hi.to_period("M"), freq="M"):
            path = data_store.WAREHOUSE / database / f"{period}.parquet"
            if path.exists():
                parts.append(pd.read_parquet(path, columns=["order", "login", "symbol", "open_time", "open_price", "volume_lots"],
                                             filters=[("login", "==", int(login))]))
        if not parts:
            return subset
        wh = pd.concat(parts, ignore_index=True)
        key = lambda f: (f["symbol"].astype(str) + "|" + pd.to_datetime(f["open_time"]).dt.floor("s").astype(str)
                         + "|" + pd.to_numeric(f["open_price"], errors="coerce").round(5).astype(str))
        candidates: dict = {}
        for k, order, lots in zip(key(wh), wh["order"], pd.to_numeric(wh["volume_lots"], errors="coerce")):
            candidates.setdefault(k, []).append((clean(order), lots))
        used = set(subset.loc[~missing, "order"])
        picked = []
        for k, lots in zip(key(subset.loc[missing]), pd.to_numeric(subset.loc[missing, "volume_lots"], errors="coerce")):
            options = [c for c in candidates.get(k, []) if c[0] and c[0] not in used]
            if not options:
                picked.append("")
                continue
            # Same lots first, then the cent-account 100x scale, then any.
            def rank(c):
                if lots != lots or c[1] != c[1] or not c[1]:
                    return 2
                ratio = lots / c[1]
                return 0 if abs(ratio - 1) < 1e-6 else 1 if abs(ratio - 100) < 1e-4 else 2
            choice = min(options, key=rank)[0]
            used.add(choice)
            picked.append(choice)
        subset.loc[missing, "order"] = picked
    except Exception:
        pass
    subset["order"] = subset["order"].map(clean).astype(object)
    return subset


def _contract_units(symbol: str) -> float:
    """Contract size in INSTRUMENT units per lot: FX 100k base units, gold
    100 oz, silver 5,000 oz, crypto 1 coin, index/other 1 contract."""
    root = "".join(c for c in str(symbol).upper() if c.isalnum())
    if root.startswith("XAU"):
        return 100.0
    if root.startswith("XAG"):
        return 5000.0
    try:
        from webapp.trade_features import symbol_class
        klass = symbol_class(symbol)
    except Exception:
        klass = ""
    if klass == "crypto":
        return 1.0
    if len(root) == 6 and root.isalpha():
        return 100_000.0
    return 1.0


def fmt_duration(seconds) -> str:
    """Human duration for the trades table: 42s / 7m 05s / 3h 12m / 2d 4h."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(s) or s < 0:
        return "--"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


def account_symbols(account_key: str) -> list[str]:
    # via account_trades so brand-new accounts (live MySQL only) still list.
    frame = account_trades(account_key, limit=5000)
    if frame.empty or "symbol" not in frame.columns:
        return []
    return sorted(frame["symbol"].dropna().astype(str).unique().tolist())


def coverage_note(frame: pd.DataFrame) -> dict:
    """What population a model actually covers.

    The Trading and Quant baselines differ ($117.4M vs $71.1M) almost entirely
    because they cover different servers, not because either is wrong. Showing
    the coverage on each screen makes the two comparable instead of leaving the
    discrepancy to be discovered.
    """
    if frame is None or frame.empty:
        return {}
    if "server" in frame.columns:
        # Written at training time -- nothing to derive.
        server = frame["server"]
    else:
        # Fallback for artefacts trained before `server` was stored. Splits only
        # the ~23k DISTINCT keys and maps back by code: running
        # `.str.split(":")` across all 19.5M rows -- twice -- took 103s and was
        # the single largest cost in rendering the Quant overview.
        codes, uniques = pd.factorize(frame["account_key"], sort=False)
        server_of_unique = pd.Index(uniques).str.split(":").str[0].to_numpy()
        server = pd.Series(server_of_unique[codes], index=frame.index)

    servers = server.value_counts().to_dict()
    by_server = frame.groupby(server)["pnl"].sum().mul(-1).round(0).to_dict()
    days = frame["day"]
    return {
        "servers": sorted(servers),
        "rows_by_server": servers,
        "firm_pnl_by_server": by_server,
        "first_day": str(days.min().date()),
        "last_day": str(days.max().date()),
        "accounts": int(frame["account_key"].nunique()),
        # MT5 stores deals, not round-trip trades: every closed row has a null
        # open_time, so a trade-level router (which must decide at entry) cannot
        # use it until entry and exit deals are paired by order id.
        "mt5_excluded": not any(s.startswith("mt5") for s in servers),
    }


def executive_summary(frame: pd.DataFrame, meta: dict | None,
                      hedge_fraction: float, view: str = "trading") -> dict:
    """One page for someone who will not open nine tabs.

    Deliberately answers four questions and stops: what did the book make, what
    is the model worth, what is the biggest risk, and what needs attention. A
    summary that tries to show everything is another dashboard.
    """
    if frame is None or frame.empty:
        return {}

    from webapp import model_service

    # Both aggregations are cached per artefact -- on the 19.5M-row Quant frame
    # they are seconds each, and they change only on retrain.
    daily = model_service.daily_series(view, frame)
    recent = daily.tail(30)
    prior = daily.iloc[-60:-30] if len(daily) >= 60 else daily.head(0)

    curve = daily.cumsum()
    drawdown = float((curve - curve.cummax()).min())
    metrics = (meta or {}).get("metrics", {})
    flat = metrics.get("flat_bbook", {})
    policy = metrics.get("by_fraction", {}).get(f"{hedge_fraction:.2f}", {})

    concentration = model_service.account_totals(view, frame)
    positive = concentration[concentration > 0]
    top_share = float(positive.head(10).sum() / positive.sum()) if len(positive) else 0.0

    return {
        "total_firm_pnl": float(daily.sum()),
        "days": int(len(daily)),
        "last_30": float(recent.sum()),
        "prior_30": float(prior.sum()) if len(prior) else None,
        "trend": (float(recent.sum() - prior.sum()) if len(prior) else None),
        "best_day": float(daily.max()),
        "worst_day": float(daily.min()),
        "losing_days": int((daily < 0).sum()),
        "max_drawdown": drawdown,
        "accounts": int(frame["account_key"].nunique()),
        # The model's headline: what the routing policy is worth against
        # B-booking everything, on both axes.
        "model_uplift": (float(policy.get("total_pnl_usd", 0) - flat.get("total_pnl_usd", 0))
                         if policy and flat else None),
        "drawdown_reduction": (float(policy.get("max_drawdown_usd", 0)
                                     - flat.get("max_drawdown_usd", 0))
                               if policy and flat else None),
        "roc_auc": metrics.get("roc_auc"),
        "top10_account_share": top_share,
        "top_accounts": [{"account_key": str(k), "firm_pnl": float(v)}
                         for k, v in concentration.head(5).items()],
    }


_FINDINGS_CACHE: dict[str, tuple[int, pd.DataFrame]] = {}


def ceo_dashboard(trading: pd.DataFrame | None, quant: pd.DataFrame | None,
                  trading_meta: dict | None, quant_meta: dict | None,
                  hedge_fraction: float = 0.05) -> dict:
    """Everything a CEO needs on one page, and nothing that needs a second.

    Answers five questions in order of what gets asked first:
      1. What did the business make, and is it improving?
      2. What would the models have added, on both books?
      3. Where is the risk concentrated, and how bad can a day get?
      4. What is happening right now?
      5. What needs a decision?

    Both books are shown side by side because they are genuinely different
    businesses -- client routing and trade-level copying -- and the natural
    executive question is which is worth more.
    """
    from webapp import model_service

    report: dict = {"books": [], "generated": pd.Timestamp.utcnow().isoformat()}

    for label, frame, meta, view in (("Trading", trading, trading_meta, "trading"),
                                     ("Quant", quant, quant_meta, "quant")):
        if frame is None or frame.empty:
            continue
        daily = model_service.daily_series(view, frame)
        curve = daily.cumsum()
        metrics = (meta or {}).get("metrics", {})
        flat = metrics.get("flat_bbook", {})
        policy = metrics.get("by_fraction", {}).get(f"{hedge_fraction:.2f}", {})

        recent, prior = daily.tail(30), daily.iloc[-60:-30]
        month_over_month = (float(recent.sum() - prior.sum()) if len(prior) else None)

        report["books"].append({
            "name": label,
            "view": view,
            "days": int(len(daily)),
            "first_day": str(daily.index.min().date()),
            "last_day": str(daily.index.max().date()),
            "accounts": int(frame["account_key"].nunique()),
            "rows": int(len(frame)),
            # Realised, flat-book: what the business actually made unhedged.
            "flat_pnl": float(daily.sum()),
            "flat_drawdown": float((curve - curve.cummax()).min()),
            "last_30": float(recent.sum()),
            "prior_30": float(prior.sum()) if len(prior) else None,
            "momentum": month_over_month,
            "best_day": float(daily.max()),
            "worst_day": float(daily.min()),
            "losing_day_share": float((daily < 0).mean()),
            "daily_mean": float(daily.mean()),
            "daily_vol": float(daily.std()),
            # What the model would have added, on both axes.
            "model_pnl": float(policy.get("total_pnl_usd")) if policy else None,
            "model_uplift": (float(policy.get("total_pnl_usd", 0) - flat.get("total_pnl_usd", 0))
                             if policy and flat else None),
            "model_drawdown": float(policy.get("max_drawdown_usd")) if policy else None,
            "drawdown_change": (float(policy.get("max_drawdown_usd", 0)
                                      - flat.get("max_drawdown_usd", 0))
                                if policy and flat else None),
            "sharpe_flat": float(flat.get("sharpe", 0)) if flat else None,
            "sharpe_model": float(policy.get("sharpe", 0)) if policy else None,
            "roc_auc": metrics.get("roc_auc"),
            "curve": [[str(d.date()), round(float(v))] for d, v in curve.items()],
        })

    # Concentration on the client book: how fragile is the earnings base.
    if trading is not None and not trading.empty:
        totals = model_service.account_totals("trading", trading)
        positive = totals[totals > 0]
        report["concentration"] = {
            "contributors": int(len(positive)),
            "top10_share": float(positive.head(10).sum() / positive.sum()) if len(positive) else 0,
            "top100_share": float(positive.head(100).sum() / positive.sum()) if len(positive) else 0,
            "top_accounts": [{"account_key": str(k), "firm_pnl": float(v)}
                             for k, v in totals.head(8).items()],
            "worst_accounts": [{"account_key": str(k), "firm_pnl": float(v)}
                               for k, v in totals.tail(5)[::-1].items()],
        }
    return report


def surveillance_findings(frame: pd.DataFrame, limit: int = 300) -> pd.DataFrame:
    """Run the surveillance detections over the current population.

    Cached against the frame's size and last day: the detections scan 460k
    markout rows and iterate per account, which cost 17s per page load for a
    result that only changes when the underlying data does.
    """
    from webapp import surveillance
    from webapp.model_service import SCRATCH

    if frame is None or frame.empty:
        return pd.DataFrame()

    key = f"{len(frame)}:{pd.to_datetime(frame['day']).max()}"
    cached = _FINDINGS_CACHE.get("findings")
    if cached is not None and cached[0] == key:
        return cached[1].head(limit)

    aggregates = ["trades", "martingale_rate", "scalp_rate", "gross_notional"]
    available = [c for c in aggregates if c in frame.columns]
    latest = (frame.sort_values("day").groupby("account_key", observed=True)
                   .tail(1).reset_index(drop=True))
    if "trades" not in latest.columns:
        latest["trades"] = latest.get("life_closes", 0)

    markouts = None
    try:
        markouts = load_markouts(SCRATCH / "markout_all_servers.parquet")
        markouts = (markouts.sort_values("day").groupby("account_key", observed=True)
                            .tail(1).reset_index(drop=True))
    except Exception:
        markouts = None

    findings = surveillance.detect(latest[["account_key"] + available + ["trades"]]
                                   .drop_duplicates("account_key"), markouts)
    _FINDINGS_CACHE["findings"] = (key, findings)
    return findings.head(limit) if not findings.empty else findings


def reconcile_baselines(trading: pd.DataFrame | None,
                        quant: pd.DataFrame | None) -> dict:
    """Explain why the two flat-B-book baselines differ.

    They measure different quantities and cannot be identical:

    * Trading sums each account-day's NEXT ACTIVE DAY P&L, so every account's
      final observation is dropped (no successor exists).
    * Quant sums every closed trade but excludes trades OPENED before the
      extract window, which are survivor-biased by construction.
    * The walk-forward warm-up lands on a different number of days for
      account-days than for trades.

    Measured, the residual is ~6.7% and proportional across all six servers,
    which is what a systematic unit difference looks like -- as opposed to a
    server-specific gap, which would indicate missing data.

    Restricting both to the same (account, day) pairs gives the like-for-like
    figure, which is the honest comparison.
    """
    if trading is None or quant is None or trading.empty or quant.empty:
        return {}

    left = trading.assign(day=pd.to_datetime(trading["day"]).dt.normalize())
    right = quant.assign(day=pd.to_datetime(quant["day"]).dt.normalize())
    keys = set(map(tuple, left[["account_key", "day"]].to_numpy())) & \
        set(map(tuple, right[["account_key", "day"]].to_numpy()))

    def restrict(frame):
        mask = [tuple(row) in keys for row in frame[["account_key", "day"]].to_numpy()]
        return frame.loc[mask]

    shared_left, shared_right = restrict(left), restrict(right)
    return {
        "trading_total": float(-left["pnl"].sum()),
        "quant_total": float(-right["pnl"].sum()),
        "gap": float(-left["pnl"].sum() + right["pnl"].sum()),
        "shared_pairs": len(keys),
        "trading_shared": float(-shared_left["pnl"].sum()),
        "quant_shared": float(-shared_right["pnl"].sum()),
        "trading_days": int(left["day"].nunique()),
        "quant_days": int(right["day"].nunique()),
        "trading_accounts": int(left["account_key"].nunique()),
        "quant_accounts": int(right["account_key"].nunique()),
    }


def summary_stats(frame: pd.DataFrame, day: str | None = None) -> dict:
    subset = day_slice(frame, day)
    if subset.empty:
        return {}
    pnl = subset["pnl"].to_numpy(dtype="float64")
    return {
        "accounts": int(subset["account_key"].nunique()),
        "rows": int(len(subset)),
        "client_pnl": float(pnl.sum()),
        "firm_pnl": float(-pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "mean_score": float(subset["score"].mean()),
    }
