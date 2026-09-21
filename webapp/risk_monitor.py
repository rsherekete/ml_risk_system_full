"""Risk monitoring: what the book is exposed to, historically and live.

The routing screens answer "what should we do". This answers "what are we
carrying, and what could it cost" -- which is a different question and the one a
risk desk actually opens first.

Three layers, deliberately kept apart because they have different provenance and
different trust levels:

* **Realised** -- what a chosen day actually cost, from the warehouse. Settled
  fact.
* **Concentration** -- how much of the outcome depends on a handful of accounts
  or symbols. A book that makes its money from three clients is a different risk
  from one that makes it from three thousand, at identical P&L.
* **Live** -- what the event stream shows right now. Only meaningful while the
  consumer is genuinely connected, so it is never blended with the historical
  figures; a stale live number is more dangerous than an absent one.

VaR here is HISTORICAL, not parametric. Client P&L is violently heavy-tailed --
the top 1% of winning account-days carry ~40% of all winner dollars -- so a
normal approximation would understate the tail badly. An empirical quantile of
the actual daily series makes no distributional assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Confidence levels for historical VaR / expected shortfall.
VAR_LEVELS = (0.95, 0.99)


def daily_firm_series(frame: pd.DataFrame) -> pd.Series:
    """Firm P&L per day: the negative of client P&L on unhedged flow."""
    working = frame.copy()
    working["day"] = pd.to_datetime(working["day"]).dt.normalize()
    return (-working.groupby("day")["pnl"].sum()).sort_index()


def value_at_risk(series: pd.Series, levels: tuple[float, ...] = VAR_LEVELS) -> dict:
    """Historical VaR and expected shortfall on the daily firm P&L series.

    Empirical quantiles rather than a normal fit: the loss distribution here has
    a fat left tail, and a parametric model would report a comfortable number
    precisely on the days that matter.
    """
    if series.empty:
        return {}
    values = series.to_numpy(dtype="float64")
    result = {
        "days": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "worst_day": float(values.min()),
        "best_day": float(values.max()),
        "losing_days": int((values < 0).sum()),
        "losing_day_share": float((values < 0).mean()),
    }
    for level in levels:
        threshold = float(np.quantile(values, 1 - level))
        tail = values[values <= threshold]
        result[f"var_{int(level * 100)}"] = threshold
        # Expected shortfall: the average loss GIVEN the threshold is breached.
        # VaR alone says how bad a bad day is; this says how bad the bad days are.
        result[f"es_{int(level * 100)}"] = float(tail.mean()) if tail.size else threshold
    return result


def concentration(frame: pd.DataFrame, day: str | None = None,
                  column: str = "account_key", top: int = 10) -> dict:
    """How much of the day's outcome rests on a few names.

    Reported because two books with identical P&L can carry very different risk:
    one that earns from three clients is exposed to those three changing
    behaviour, and that fragility is invisible in a P&L total.
    """
    working = frame.copy()
    working["day"] = pd.to_datetime(working["day"]).dt.normalize()
    if day:
        working = working.loc[working["day"] == pd.Timestamp(day).normalize()]
    if working.empty or column not in working:
        return {}

    firm = (-working.groupby(column)["pnl"].sum()).sort_values(ascending=False)
    positive = firm[firm > 0]
    total = float(positive.sum())
    leaders = firm.head(top)
    # Herfindahl over the profitable names: 1.0 means one name is everything.
    shares = (positive / total) if total > 0 else positive * 0
    return {
        "total_firm_pnl": float(firm.sum()),
        "contributors": int(len(firm)),
        "top_names": [{"name": str(k), "firm_pnl": float(v),
                       "share": float(v / total) if total > 0 else 0.0}
                      for k, v in leaders.items()],
        "top1_share": float(shares.iloc[0]) if len(shares) else 0.0,
        "top10_share": float(shares.head(10).sum()) if len(shares) else 0.0,
        "herfindahl": float((shares ** 2).sum()) if len(shares) else 0.0,
    }


def day_risk_report(frame: pd.DataFrame, day: str | None,
                    hedge_fraction: float = 0.05, view: str = "trading") -> dict:
    """Everything the risk desk needs for one trading day.

    Slices to the target day FIRST and only then does per-row work. The earlier
    version copied the whole frame and re-derived the day column before
    filtering, which on the 19.5M-row Quant artefact cost ~17s per render for
    figures that concern a single day.
    """
    from webapp import model_service

    # Cached per artefact: the daily aggregation over 19.5M rows is seconds,
    # and every risk render needs the same series.
    series = model_service.daily_series(view, frame)
    days = pd.to_datetime(frame["day"]).dt.normalize()

    target = pd.Timestamp(day).normalize() if day else series.index.max()
    today = frame.loc[(days == target).to_numpy()]
    if today.empty:
        return {"day": str(target.date()) if target is not None else None, "empty": True}

    pnl = today["pnl"].to_numpy(dtype="float64")
    scores = today["score"].to_numpy(dtype="float64")
    hedged = scores >= np.quantile(scores, 1 - max(1e-6, hedge_fraction)) if scores.size else scores

    # Where this day sits in the historical distribution -- a $2M loss is
    # unremarkable on one book and a tail event on another, and the percentile
    # is what makes that legible without knowing the book.
    firm_today = float(-pnl.sum())
    percentile = float((series.to_numpy() <= firm_today).mean()) if len(series) else None

    return {
        "day": str(target.date()),
        "empty": False,
        "firm_pnl": firm_today,
        "firm_pnl_hedged": float(-pnl[~hedged].sum()) if hedged.size else firm_today,
        "client_pnl": float(pnl.sum()),
        "percentile": percentile,
        "accounts": int(today["account_key"].nunique()),
        "rows": int(len(today)),
        "win_rate": float((pnl > 0).mean()),
        "worst_account_loss": float(-pnl.max()) if pnl.size else 0.0,
        "largest_single_payout": float(pnl.max()) if pnl.size else 0.0,
        "var": value_at_risk(series),
        # Concentration is computed from the single-day slice, not the full
        # frame -- it only ever describes that day.
        "by_account": concentration(today, None, "account_key"),
        "by_symbol": concentration(today, None, "symbol") if "symbol" in today else {},
    }


def exposure_timeline(frame: pd.DataFrame, window: int = 30,
                      view: str = "trading") -> pd.DataFrame:
    """Rolling risk over time: P&L, volatility and drawdown together.

    Shown as one series rather than three panels because the question is
    whether a profitable stretch was calm or violent, and that is only visible
    when the three are read against each other.

    Uses the cached daily series. `daily_firm_series` copies the whole frame to
    normalise its date column, which on 19.5M rows was the single largest cost
    in rendering this page.
    """
    from webapp import model_service

    series = model_service.daily_series(view, frame)
    table = pd.DataFrame({"day": series.index, "firm_pnl": series.to_numpy()})
    table["cumulative"] = table["firm_pnl"].cumsum()
    table["drawdown"] = table["cumulative"] - table["cumulative"].cummax()
    table["rolling_vol"] = table["firm_pnl"].rolling(window, min_periods=5).std()
    table["rolling_mean"] = table["firm_pnl"].rolling(window, min_periods=5).mean()
    # Rolling VaR: the empirical 5th percentile of the trailing window, so the
    # risk line adapts to the regime rather than assuming one for the sample.
    table["rolling_var95"] = table["firm_pnl"].rolling(window, min_periods=10).quantile(0.05)
    return table
