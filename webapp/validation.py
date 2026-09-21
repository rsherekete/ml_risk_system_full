"""Evidence that the numbers on the rest of the site can be believed.

Every other screen reports what the model earned. This one asks whether that
report is true, because three things can each make the headline wrong without
making it look wrong:

1. THE P&L COULD BE INVENTED. Exposure-day expansion repeats a trade once per
   day it was open. If realised P&L were booked on more than the closing day the
   totals would inflate silently and every screen would agree with every other
   screen, because they all read the same artefact. So the artefact's P&L is
   reconciled against the warehouse's own ``sum(net_profit)`` -- an independent
   path to the same quantity. This is the only check here that reads the
   warehouse, it costs a full scan of ~140M trades, and it is therefore cached.

2. THE MODEL COULD BE SCORED ON THE WRONG TARGET. The model predicts forward
   self-relative P&L, not same-day P&L. Scoring it against a contemporaneous
   label produces a number below 0.5 that looks like an inverted model and is
   really a measurement error. The label used here is the stored target.

3. THE RESULT COULD BE A REGIME ARTEFACT. A single AUC over two years is an
   average, and an average hides a model that worked in 2024 and stopped. The
   desk would be running this today, so the economics are recomputed on recent
   windows alone and the honest answer -- which windows the model actually
   dominates in, and which it does not -- is printed rather than summarised.

Nothing here retrains. These are the stored walk-forward scores, each produced
by a model fitted only on days strictly before the day it scores.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import timezone

import numpy as np
import pandas as pd

from . import model_service

#: Windows the report is computed over, newest first. "full" is whatever the
#: artefact holds; the shorter ones answer "does it still work now?".
WINDOWS = (("Last 90 days", 90), ("Last 180 days", 180),
           ("Last 365 days", 365), ("Full history", 100_000))

#: Hedge fractions shown in the dominance table -- the same ladder the training
#: summary reports, so the two can be compared line for line.
FRACTIONS = (0.02, 0.05, 0.10, 0.20)

#: A target beyond this many sigma is a division by a near-zero volatility, not
#: a real move. They are counted and excluded rather than silently averaged in.
SIGMA_SANE_LIMIT = 100.0

_RECON_LOCK = threading.Lock()
_recon_thread: threading.Thread | None = None


def _recon_path(view: str):
    return model_service.ARTIFACTS / f"{view}_reconciliation.json"


def cached_reconciliation(view: str) -> dict | None:
    """The stored warehouse reconciliation, if one matches the live artefact."""
    path = _recon_path(view)
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    meta = model_service.artifact_meta(view) or {}
    # Tie the result to the artefact it was computed against. A reconciliation
    # of a previous model is worse than none: it would show a match that no
    # longer refers to the numbers on screen.
    if stored.get("trained_at") != meta.get("trained_at"):
        stored["stale"] = True
    return stored


def reconciliation_running() -> bool:
    return _recon_thread is not None and _recon_thread.is_alive()


def start_reconciliation(view: str) -> bool:
    """Kick off the warehouse scan in the background. One at a time."""
    global _recon_thread
    with _RECON_LOCK:
        if reconciliation_running():
            return False
        _recon_thread = threading.Thread(
            target=_reconcile, args=(view,), daemon=True,
            name=f"reconcile-{view}")
        _recon_thread.start()
        return True


def _reconcile(view: str) -> None:
    from . import data_store

    started = time.time()
    try:
        frame = model_service.load_scores(view)
        if frame is None or frame.empty:
            return
        day = pd.to_datetime(frame["day"])
        lo, hi = day.min(), day.max()
        artefact_pnl = float(frame["pnl"].sum())

        # The warehouse must be summed over the artefact's OWN date range. Using
        # the configured history length instead compares 730 days of warehouse
        # against 670 days of artefact and reports the 60-day difference as a
        # 3.9% discrepancy in the model -- which is a difference in the question,
        # not in the answer.
        start = lo.to_pydatetime().replace(tzinfo=timezone.utc)
        end = (hi + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc)

        warehouse_pnl, trades = 0.0, 0
        servers = sorted(p.name for p in data_store.WAREHOUSE.iterdir() if p.is_dir())
        for server in servers:
            chunk = data_store.read_history(
                databases=(server,), start=start, end=end,
                columns=["database", "net_profit", "close_time"])
            if chunk.empty:
                continue
            # P&L is realised on the closing day, so only closed trades count.
            closed = chunk.loc[chunk["close_time"].notna() & chunk["net_profit"].notna()]
            warehouse_pnl += float(closed["net_profit"].sum())
            trades += len(closed)
            del chunk, closed

        gap = artefact_pnl - warehouse_pnl
        meta = model_service.artifact_meta(view) or {}
        payload = {
            "artefact_pnl": artefact_pnl,
            "warehouse_pnl": warehouse_pnl,
            "gap": gap,
            "relative_gap": abs(gap) / max(abs(warehouse_pnl), 1.0),
            "trades": trades,
            "start": str(lo.date()),
            "end": str(hi.date()),
            "seconds": time.time() - started,
            "trained_at": meta.get("trained_at"),
            "computed_at": time.time(),
        }
        _recon_path(view).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except Exception as error:  # a failed check must not take the page down
        _recon_path(view).write_text(json.dumps({
            "error": f"{type(error).__name__}: {error}",
            "computed_at": time.time()}, indent=1), encoding="utf-8")


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC without a hard sklearn dependency at import time."""
    if len(labels) < 1000 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    return float(roc_auc_score(labels, scores))


#: Reports keyed by the artefact they describe. Building one costs 14s for the
#: Trading view and over two minutes for Quant's 19.5M trade rows, and the
#: answer cannot change until the model is retrained -- so it is computed once
#: per artefact rather than once per page load.
_REPORT_CACHE: dict[str, tuple[object, dict]] = {}


_BUILDERS: dict[str, threading.Thread] = {}


def _report_path(view: str):
    return model_service.ARTIFACTS / f"{view}_validation_report.json"


def report(view: str) -> dict:
    """The full validation picture -- served instantly, built in the background.

    Building costs seconds for Trading and over two minutes for Quant's 18M
    rows, and a page that computes it inline simply times out. So the report
    persists to disk keyed by the artifact it describes; a request that finds
    no current report kicks off ONE background build and says so, and the page
    refresh finds it done.
    """
    meta = model_service.artifact_meta(view) or {}
    stamp = meta.get("trained_at")

    def finish(payload: dict) -> dict:
        payload = dict(payload)
        payload["reconciliation"] = cached_reconciliation(view)
        payload["reconciling"] = reconciliation_running()
        payload["coverage"] = _coverage()
        return payload

    cached = _REPORT_CACHE.get(view)
    if cached is not None and cached[0] == stamp and stamp is not None:
        return finish(cached[1])

    path = _report_path(view)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("_stamp") == stamp:
                _REPORT_CACHE[view] = (stamp, stored)
                return finish(stored)
        except (ValueError, OSError):
            pass

    builder = _BUILDERS.get(view)
    if builder is None or not builder.is_alive():
        def build():
            result = _build_report(view)
            if result.get("available") and stamp is not None:
                result["_stamp"] = stamp
                _REPORT_CACHE[view] = (stamp, result)
                try:
                    path.write_text(json.dumps(result, default=float),
                                    encoding="utf-8")
                except OSError:
                    pass
        builder = threading.Thread(target=build, daemon=True,
                                   name=f"validation-{view}")
        _BUILDERS[view] = builder
        builder.start()
    return {"available": False, "building": True}


def _build_report(view: str) -> dict:
    frame = model_service.load_scores(view)
    if frame is None or frame.empty:
        return {"available": False}

    config = model_service.load_config(view)
    columns = [c for c in ("account_key", "day", "pnl", "score", "sigma")
               if c in frame.columns]
    frame = frame[columns].copy()
    frame["day"] = pd.to_datetime(frame["day"])
    latest = frame["day"].max()

    score = frame["score"].to_numpy(dtype="float64")
    threshold = getattr(config, "sigma_threshold", 0.5)

    # The two views are fitted on genuinely different targets, and scoring one
    # against the other's would be meaningless.
    #
    # Trading routes ACCOUNT-DAYS on `sigma` -- forward P&L over the horizon
    # divided by the account's own volatility -- so the positive class is a
    # client gain beyond half a sigma. Quant routes INDIVIDUAL TRADES, where the
    # decision is simply whether to copy this trade, so the outcome of the trade
    # itself is the target. That is not circular: the score was produced by a
    # model fitted only on earlier days, and the P&L is what happened next.
    if "sigma" in frame.columns:
        sigma = frame["sigma"].to_numpy(dtype="float64")
        finite = np.isfinite(sigma) & np.isfinite(score)
        # A target beyond the sane limit is a division by a near-zero
        # volatility, not a real move.
        sane = finite & (np.abs(sigma) <= SIGMA_SANE_LIMIT)
        labels = sigma > threshold
        target_note = (f"forward {getattr(config, 'horizon_active_days', 5)} exposure days, "
                       f"beyond {threshold}σ of the account's own volatility")
    else:
        sigma = None
        pnl = frame["pnl"].to_numpy(dtype="float64")
        sane = np.isfinite(pnl) & np.isfinite(score)
        # Positive class = the client profited on the trade, which is the trade
        # the desk would have wanted to copy.
        labels = pnl > 0
        target_note = "the trade's own realised P&L -- did the client profit on it"

    # Each window's frame is prepared once and reused across every hedge
    # fraction. Preparing inside each curve instead meant copying, sorting and
    # factorising a 4.3M-row frame twenty-four times, which took the page from
    # seconds to minutes.
    #
    # The Quant view routes individual TRADES, not account-days, and has its own
    # builder. Reusing the account-level one here would report a number that no
    # screen in that view agrees with, so the reuse applies only where the
    # builder actually matches.
    trade_level = view == model_service.VIEW_QUANT
    prepared: dict[str, dict] = {}
    window_frames: dict[str, pd.DataFrame] = {}

    def curve_for(title: str, fraction: float, threshold: float = 0.0):
        if trade_level:
            return model_service.trade_equity_curves(
                window_frames[title], fraction, threshold)
        return model_service._curves_from(prepared[title], fraction, threshold)

    windows = []
    for title, days in WINDOWS:
        mask = frame["day"] > latest - pd.Timedelta(days=days)
        keep = mask.to_numpy() & sane
        window = frame.loc[mask]
        if window.empty:
            continue
        window_frames[title] = window
        if not trade_level:
            prepared[title] = model_service._prepare_curve_inputs(window)
        curve = curve_for(title, config.hedge_fraction, config.probability_threshold)
        windows.append({
            "title": title,
            "days": int(window["day"].nunique()),
            "rows": int(len(window)),
            "scored": int(keep.sum()),
            "base_rate": float(labels[keep].mean()) if keep.any() else float("nan"),
            "auc": _auc(labels[keep], score[keep]),
            "flat": float(curve["flat"].sum()),
            "model": float(curve["model"].sum()),
            "flat_dd": float(curve["flat_dd"].min()),
            "model_dd": float(curve["model_dd"].min()),
        })

    # Which hedge fractions genuinely dominate -- more profit AND less
    # drawdown -- in each window. Where none do, the honest answer is that the
    # model buys stability with profit, and the table says so.
    dominance = []
    for title, _days in WINDOWS:
        if title not in window_frames:
            continue
        base = curve_for(title, 1e-6, 0.0)
        flat_profit, flat_dd = float(base["flat"].sum()), float(base["flat_dd"].min())
        ladder = []
        for fraction in FRACTIONS:
            curve = curve_for(title, fraction, 0.0)
            profit, drawdown = float(curve["model"].sum()), float(curve["model_dd"].min())
            daily = curve["model"]
            ladder.append({
                "fraction": fraction,
                "profit": profit,
                "profit_delta": profit - flat_profit,
                "drawdown": drawdown,
                "drawdown_delta": drawdown - flat_dd,
                "sharpe": float(daily.mean() / max(daily.std(), 1e-9) * np.sqrt(252)),
                "dominates": bool(profit > flat_profit and drawdown > flat_dd),
            })
        dominance.append({
            "title": title,
            "flat_profit": flat_profit,
            "flat_drawdown": flat_dd,
            "ladder": ladder,
            "winners": [f"{r['fraction']:.0%}" for r in ladder if r["dominates"]],
        })

    if sigma is not None:
        finite_sigma = sigma[np.isfinite(sigma)]
        target = {
            "has_sigma": True,
            "rows": int(len(frame)),
            "non_finite": int((~np.isfinite(sigma)).sum()),
            "extreme": int((np.abs(finite_sigma) > SIGMA_SANE_LIMIT).sum()),
            "p01": float(np.percentile(finite_sigma, 0.1)) if finite_sigma.size else float("nan"),
            "p99": float(np.percentile(finite_sigma, 99.9)) if finite_sigma.size else float("nan"),
            "median": float(np.median(finite_sigma)) if finite_sigma.size else float("nan"),
        }
    else:
        target = {"has_sigma": False, "rows": int(len(frame)),
                  "non_finite": int((~sane).sum())}

    return {
        "available": True,
        "windows": windows,
        "dominance": dominance,
        "threshold": threshold,
        "horizon": getattr(config, "horizon_active_days", 5),
        "hedge_fraction": config.hedge_fraction,
        "threshold_live": config.probability_threshold,
        "target_note": target_note,
        "target": target,
        "policies": _policy_sweep(prepared, window_frames, trade_level, curve_for),
        "coverage": _coverage(),
        "reconciliation": cached_reconciliation(view),
        "reconciling": reconciliation_running(),
    }


#: Policies compared on the Validation screen. Quotas hedge a fixed share of the
#: routable population each day; thresholds hedge only what the model is
#: confident about and let the daily count float.
POLICIES = (
    ("quota 0.5%", {"fraction": 0.005}),
    ("quota 1%", {"fraction": 0.01}),
    ("quota 2%", {"fraction": 0.02}),
    ("quota 5%", {"fraction": 0.05}),
    ("prob >= 0.70", {"threshold": 0.70}),
    ("prob >= 0.80", {"threshold": 0.80}),
    ("prob >= 0.90", {"threshold": 0.90}),
)


def _policy_sweep(prepared, window_frames, trade_level, curve_for) -> list[dict]:
    """How each routing policy would have done, per window.

    This exists because the default quota was costing real money and nothing on
    the site would have shown it. Hedging the top 5% of scores lost $43.8M
    against a flat B-book over two years, while hedging the top 1% GAINED $6M
    and still cut drawdown -- the same model, the same scores, a different
    selection rule.
    #
    The reason is that almost every client cohort loses to the broker, so every
    hedge costs expected profit. Hedging is insurance, not alpha: a wide quota
    spends premium on marginal cases, and only genuinely confident calls pay for
    themselves. Ranking by expected DOLLARS rather than probability was tried
    and is far worse -- the largest accounts are the firm's biggest profit
    source, so weighting by size hedges exactly the wrong ones.
    """
    results = []
    for title in window_frames:
        base = curve_for(title, 1e-6, 0.0)
        flat_profit = float(base["flat"].sum())
        flat_dd = float(base["flat_dd"].min())
        rows = []
        for name, kwargs in POLICIES:
            try:
                curve = curve_for(title, kwargs.get("fraction", 1e-6),
                                  kwargs.get("threshold", 0.0))
            except Exception:
                continue
            profit, drawdown = float(curve["model"].sum()), float(curve["model_dd"].min())
            hedged = curve["hedged_accounts"].mean() if "hedged_accounts" in curve else float("nan")
            rows.append({
                "name": name,
                "profit": profit,
                "profit_delta": profit - flat_profit,
                "drawdown": drawdown,
                "drawdown_delta": drawdown - flat_dd,
                "hedged_per_day": float(hedged),
                "dominates": bool(profit > flat_profit and drawdown > flat_dd),
            })
        results.append({"title": title, "flat_profit": flat_profit,
                        "flat_drawdown": flat_dd, "rows": rows})
    return results


def _coverage() -> dict:
    """Which of the six live servers the numbers on this site actually include."""
    try:
        from . import data_store
        return data_store.coverage()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}",
                "servers": [], "missing": [], "expected": 0, "present": 0}
