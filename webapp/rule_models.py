"""Generic per-category rule engine + ML models.

THE ARCHITECTURE (as specified):
- Every category is a COMPLEX RULE over the per-client feature table (the
  174-column account-day corpus + trade-shape + event features). Active
  rules feed the tabs; `use_ml` rules each get a generated tab and a
  dedicated model.
- TARGET, uniformly: the account satisfies the rule over the NEXT n days
  (the rule evaluated on forward-window behaviour) AND its equity rises in
  that window. Inputs are the prior 90 days of behaviour.
- The definition's hash is stored with the model: change the definition and
  the next scan auto-retrains in the background.
- Scans persist to disk and load instantly; refresh recomputes.

Latency Arbitrage keeps its specialised tape-label engine (latency_arb).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

OBS_DAYS = 90             # "use at least 90 days of history"
FORWARD_DAYS = 5
EVENT_TOL_MIN = 10.0

_CACHE: dict = {}
_TRAIN_LOCKS: dict = {}


def _paths(key: str):
    return (ART / f"rule_ml_{key}.txt", ART / f"rule_ml_{key}_meta.json",
            ART / f"rule_scan_{key}.json")


def rule_meta(key: str) -> dict:
    try:
        return json.loads(_paths(key)[1].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _definition(key: str) -> str:
    from webapp import af_registry
    for c in af_registry.load().get("categories", []):
        if c["key"] == key:
            return c.get("definition", "")
    return ""


def _def_hash(key: str) -> str:
    return hashlib.md5(_definition(key).encode()).hexdigest()[:12]


# ------------------------------------------------- fixed event anchors
VOL_CAL_PATH = ART / "vol_event_calendar.json"


def _market_anchors(days: int = 95) -> list:
    """Market-derived event calendar: minutes where several major symbols
    spiked together (z-scored M1 range) -- objective, computable for ANY
    historical window, covering what the young feed calendar cannot.
    Cached daily."""
    try:
        cached = json.loads(VOL_CAL_PATH.read_text(encoding="utf-8"))
        if time.time() - cached.get("at", 0) < 86400:
            return [datetime.fromisoformat(t) for t in cached["anchors"]]
    except Exception:
        pass
    anchors: list = []
    try:
        from datetime import timezone as _tz
        import MetaTrader5 as mt5
        from webapp.vantage import _mt5
        m = _mt5()
        end = datetime.now(_tz.utc)
        start = end - timedelta(days=days)
        per_symbol = {}
        for sym in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "NAS100", "US30"):
            rows = []
            lo = start
            while lo < end:
                hi = min(lo + timedelta(days=7), end)
                r = m.copy_rates_range(sym, mt5.TIMEFRAME_M1, lo, hi)
                if r is not None and len(r):
                    rows.append(pd.DataFrame(r)[["time", "high", "low"]])
                lo = hi
            if not rows:
                continue
            bars = pd.concat(rows).drop_duplicates("time")
            rng = (bars["high"] - bars["low"]).to_numpy()
            mu, sd = np.nanmean(rng), np.nanstd(rng) + 1e-12
            z = (rng - mu) / sd
            hot = bars["time"].to_numpy()[z >= 6]
            per_symbol[sym] = set((hot // 60).astype(np.int64))
        counts: dict = {}
        for s in per_symbol.values():
            for minute in s:
                counts[minute] = counts.get(minute, 0) + 1
        # server clock UTC+3 -> UTC
        anchors = sorted(datetime.utcfromtimestamp(minute * 60)
                         - timedelta(hours=3)
                         for minute, n in counts.items() if n >= 2)
        VOL_CAL_PATH.write_text(json.dumps(
            {"at": time.time(),
             "anchors": [a.isoformat() for a in anchors]}), encoding="utf-8")
    except Exception:
        pass
    return anchors


def combined_anchors() -> np.ndarray:
    """The FIXED calendar: economic feed history (+schedule backfill) merged
    with market-derived synchronized-volatility minutes."""
    from webapp import econ_calendar
    ec = econ_calendar.event_times_utc("medium", historical=True)
    mk = _market_anchors()
    allts = sorted({pd.Timestamp(t).to_pydatetime().replace(tzinfo=None)
                    for t in list(ec) + list(mk)})
    return np.array(allts, dtype="datetime64[s]")


def _event_stats(trades: pd.DataFrame, anchors: np.ndarray) -> pd.DataFrame:
    t = trades["open_time"].to_numpy(dtype="datetime64[s]")
    near = np.zeros(len(t), dtype=bool)
    if len(anchors):
        idx = np.searchsorted(anchors, t)
        tol = np.timedelta64(int(EVENT_TOL_MIN * 60), "s")
        for shift in (0, 1):
            j = np.clip(idx - shift, 0, len(anchors) - 1)
            near |= np.abs(t - anchors[j]) <= tol
    g = trades.assign(event_trade=near).groupby("account_key", observed=True)
    return pd.DataFrame({
        "event_share": g["event_trade"].mean(),
        "event_trades": g["event_trade"].sum(),
        "event_pnl": g.apply(lambda x: float(
            x.loc[x["event_trade"], "net_profit"].sum())),
    })


def _window_trades(start: datetime, end: datetime) -> pd.DataFrame:
    from webapp import data_store
    frame = data_store.read_history(start=start, end=end)
    if frame is None or not len(frame):
        return pd.DataFrame()
    frame = frame.copy()
    frame["account_key"] = (frame["database"].astype(str) + ":"
                            + frame["login"].astype(str))
    frame["open_time"] = pd.to_datetime(frame["open_time"])
    frame["hold_seconds"] = (
        pd.to_datetime(frame["close_time"]) - frame["open_time"]
    ).dt.total_seconds()
    return frame


def feature_table(trades: pd.DataFrame, anchors: np.ndarray,
                  corpus: pd.DataFrame | None = None) -> pd.DataFrame:
    """The per-client feature table every rule speaks over: trade-shape +
    event features + the 174-column account-day corpus row. `corpus`
    overrides the latest-row corpus with an AS-OF-day snapshot (used by the
    walk-forward backfill so historical tables carry no future corpus)."""
    from webapp.latency_arb import _account_features
    feats = _account_features(trades)
    feats = feats.join(_event_stats(trades, anchors), how="left")
    for c in ("event_share", "event_trades", "event_pnl"):
        feats[c] = feats[c].fillna(0.0)
    try:
        # rsuffix: the corpus carries a few names the trade-shape set also
        # uses (win_rate, profit_factor); the corpus copy gets _corpus.
        feats = feats.join(corpus if corpus is not None else _ad_latest(),
                           how="left", rsuffix="_corpus")
        _JOIN_ERR[0] = ""
    except Exception as error:
        _JOIN_ERR[0] = f"{type(error).__name__}: {error}"
    # DERIVED registry fields -- every default rule must speak over columns
    # that actually exist here.
    try:
        # notional_z: peer z-score of position scale (corpus max notional).
        base_col = next((c for c in ("max_notional", "gross_notional")
                         if c in feats.columns), None)
        if base_col is not None:
            v = np.log1p(pd.to_numeric(feats[base_col],
                                       errors="coerce").clip(lower=0))
            feats["notional_z"] = ((v - v.mean())
                                   / (v.std() + 1e-9)).fillna(0.0)
    except Exception:
        pass
    try:
        # mk_short / mk_trades: tape markouts from the latency scan pass.
        from webapp import latency_arb
        mk = latency_arb.markout_fields()
        if len(mk):
            feats = feats.join(mk, how="left")
        for col, default in (("mk_short", 0.0), ("mk_trades", 0)):
            if col in feats.columns:
                feats[col] = pd.to_numeric(
                    feats[col], errors="coerce").fillna(default)
            else:
                feats[col] = default
    except Exception:
        pass
    try:
        # bonus_extraction: withdrawn / deposited money over the window.
        from webapp import cashflow_store
        flows = cashflow_store.read_cashflows(
            start=trades["open_time"].min(), end=trades["open_time"].max())
        if len(flows):
            amt = pd.to_numeric(flows["amount"], errors="coerce").fillna(0.0)
            g = pd.DataFrame({
                "dep": amt.clip(lower=0), "wd": (-amt).clip(lower=0),
                "account_key": flows["account_key"].astype(str),
            }).groupby("account_key").sum()
            ratio = (g["wd"] / g["dep"].clip(lower=1e-9)).clip(0, 1)
            ratio = ratio.where(g["dep"] > 0, 0.0)
            feats["bonus_extraction"] = ratio.reindex(feats.index).fillna(0.0)
        else:
            feats["bonus_extraction"] = 0.0
    except Exception:
        feats["bonus_extraction"] = 0.0
    return feats


_JOIN_ERR = [""]
_AD_CACHE: dict = {}


def _ad_latest() -> pd.DataFrame:
    """Latest account-day corpus row per account (the 174 features), cached
    30 minutes -- the sort+groupby over 600k rows is not per-request work."""
    now = time.time()
    if _AD_CACHE.get("at", 0) > now - 1800:
        return _AD_CACHE["df"]
    from webapp import antifraud
    frame = antifraud._frame()
    latest = (frame.sort_values("decision_day")
              .groupby("account_key", observed=True).last())
    ad_cols = [c for c in latest.columns
               if c != "decision_day"
               and pd.api.types.is_numeric_dtype(latest[c])]
    out = latest[ad_cols]
    _AD_CACHE.update(at=now, df=out)
    return out


def _rule_mask(definition: str, table: pd.DataFrame) -> pd.Series:
    from webapp.antifraud import safe_expr_mask
    try:
        return safe_expr_mask(definition, table).fillna(False).astype(bool)
    except Exception:
        return pd.Series(False, index=table.index)


_TABLE_CACHE: dict = {}


OBS_DISK = ART / "obs_table.parquet"
OBS_META = ART / "obs_table_meta.json"


def _obs_from_disk(ttl: float) -> bool:
    """Load the persisted daily feature table into the memory cache if the
    disk copy is fresh -- restarts must never cost the 7-minute rebuild."""
    try:
        meta = json.loads(OBS_META.read_text(encoding="utf-8"))
        if float(meta.get("at", 0)) <= time.time() - ttl:
            return False
        table = pd.read_parquet(OBS_DISK)
        _TABLE_CACHE.update(
            at=float(meta["at"]), table=table,
            active=set(meta.get("active") or []),
            active_day=meta.get("active_day"),
            last_ts=meta.get("last_ts") or {})
        return True
    except Exception:
        return False


def _obs_to_disk() -> None:
    try:
        _TABLE_CACHE["table"].to_parquet(OBS_DISK)
        OBS_META.write_text(json.dumps({
            "at": _TABLE_CACHE.get("at"),
            "active": sorted(_TABLE_CACHE.get("active") or []),
            "active_day": _TABLE_CACHE.get("active_day"),
            "last_ts": _TABLE_CACHE.get("last_ts") or {}}),
            encoding="utf-8")
    except Exception:
        pass


_OBS_BUILD_LOCK = threading.Lock()


def _obs_table_cached(ttl: float = 900.0):
    """(feature table over the last OBS_DAYS, set of accounts active TODAY)
    -- shared by rule counts, scans and profiling so ONE build serves
    everything. Memory first, then the DISK copy (survives restarts, valid
    2h), then a rebuild -- SERIALISED: concurrent callers wait for the one
    build instead of doubling the memory bill."""
    now = time.time()
    if _TABLE_CACHE.get("at", 0) > now - ttl:
        return _TABLE_CACHE["table"], _TABLE_CACHE["active"]
    if _obs_from_disk(ttl=7200.0):
        return _TABLE_CACHE["table"], _TABLE_CACHE["active"]
    with _OBS_BUILD_LOCK:
        # another thread may have finished the build while we waited
        if _TABLE_CACHE.get("at", 0) > time.time() - max(ttl, 900.0):
            return _TABLE_CACHE["table"], _TABLE_CACHE["active"]
        return _obs_build()


def _obs_build():
    now = time.time()
    end = datetime.utcnow()
    trades = _window_trades(end - timedelta(days=OBS_DAYS), end)
    if not len(trades):
        return pd.DataFrame(), set()
    table = feature_table(trades, combined_anchors())
    # warehouse open_time is SERVER clock (UTC+3); closed-trade sync can lag,
    # so fall back to the most recent day actually present in the data.
    days = trades["open_time"].dt.strftime("%Y-%m-%d")
    active_day = (end + timedelta(hours=3)).strftime("%Y-%m-%d")
    active = set(trades.loc[days == active_day, "account_key"].astype(str))
    if not active and len(days):
        active_day = str(days.max())
        active = set(trades.loc[days == active_day,
                                "account_key"].astype(str))
    last_ts = trades.groupby("account_key", observed=True)["open_time"] \
        .max().astype(str).to_dict()
    _TABLE_CACHE.update(at=now, table=table, active=active,
                        active_day=active_day, last_ts=last_ts)
    _obs_to_disk()
    return table, active


_COUNTS_CACHE: dict = {}
_BUILD_THREAD = threading.Lock()


def registry_counts() -> dict:
    """Per category, over ALL observed accounts: HINDSIGHT hits (the rule
    definition satisfied now), the same among accounts active on the latest
    trading day, and -- for use_ml categories -- ML-now hits (classification
    model over threshold) and EW-5d hits (early-warning model over
    threshold).

    NEVER blocks: a cold feature table returns {"building": true}
    immediately and builds in a background thread; the UI polls."""
    from webapp import af_registry
    now = time.time()
    cached = _COUNTS_CACHE.get("result")
    if cached is not None and _COUNTS_CACHE.get("at", 0) > now - 300:
        return cached
    if _TABLE_CACHE.get("at", 0) <= now - 900:
        if not _BUILD_THREAD.locked():
            def _build():
                with _BUILD_THREAD:
                    try:
                        _obs_table_cached()
                    except Exception:
                        pass
            threading.Thread(target=_build, daemon=True).start()
        return {"building": True,
                "note": "feature table building in the background"}
    table, active = _obs_table_cached()
    if not len(table):
        return {"error": "no data"}
    n_all = int(len(table))
    act_idx = table.index.astype(str).isin(active)
    n_act = int(act_idx.sum())
    out = {"n_accounts": n_all, "n_active_today": n_act, "categories": {},
           "active_day": _TABLE_CACHE.get("active_day"),
           "join_error": _JOIN_ERR[0] or None}
    for c in af_registry.load().get("categories", []):
        key = c["key"]
        mask = _rule_mask(c.get("definition", ""), table)
        hits = int(mask.sum())
        hits_act = int((mask & act_idx).sum())
        entry = {
            "hits": hits, "pct": round(100 * hits / max(n_all, 1), 2),
            "hits_active": hits_act,
            "pct_active": round(100 * hits_act / max(n_act, 1), 2),
            "use_ml": bool(c.get("use_ml")),
            "ml_hits": None, "ml_pct": None,
            "ew_hits": None, "ew_pct": None}
        if c.get("use_ml") and c.get("active"):
            meta = rule_meta(key)
            mp = _paths(key)[0]
            now_meta = meta.get("now") or {}
            ew_meta = meta.get("ew") or (meta if meta.get("auc") else {})
            if now_meta.get("features"):
                ml = _predict(mp.with_name(mp.stem + "_now.txt"), table,
                              now_meta["features"])
                n_ml = int((ml >= float(now_meta.get("threshold", 0.5))).sum())
                entry["ml_hits"] = n_ml
                entry["ml_pct"] = round(100 * n_ml / max(n_all, 1), 2)
            if ew_meta.get("features"):
                ew = _predict(mp, table, ew_meta["features"])
                n_ew = int((ew >= float(ew_meta.get("threshold", 0.5))).sum())
                entry["ew_hits"] = n_ew
                entry["ew_pct"] = round(100 * n_ew / max(n_all, 1), 2)
            # NOTE deliberately no auto-train here: models retrain ONLY on a
            # definition change (_maybe_retrain during scans) or a manual
            # Retrain click -- missing artifacts just show as "—".
        out["categories"][key] = entry
    _COUNTS_CACHE.update(at=now, result=out)
    return out


# ---------------------------------------------- walk-forward verification
#: The NOW model's claim is testable one day later: "when this account is
#: next active, it will satisfy the rule." Every night we snapshot the
#: prediction for EVERY account; the next trading day we take the accounts
#: that were ACTUALLY ACTIVE and check whether the rule truly holds --
#: realised precision/recall on the live universe, not a CV estimate.
PRED_DB = ART / "rule_predictions.sqlite"


def _pred_cx():
    import sqlite3
    cx = sqlite3.connect(PRED_DB)
    cx.execute("CREATE TABLE IF NOT EXISTS preds ("
               "as_of_day TEXT, rule TEXT, account TEXT, "
               "p_now REAL, p_ew REAL, flag_now INT, flag_ew INT, "
               "PRIMARY KEY (as_of_day, rule, account))")
    return cx


def snapshot_predictions() -> dict:
    """Score EVERY account with every use_ml rule's saved models and persist
    the flags, stamped with the feature table's trading day."""
    from webapp import af_registry
    table, _ = _obs_table_cached()
    if not len(table):
        return {"error": "no feature table"}
    day = str(_TABLE_CACHE.get("active_day") or
              datetime.utcnow().strftime("%Y-%m-%d"))
    out = {"as_of_day": day, "rules": {}}
    with _pred_cx() as cx:
        for c in af_registry.load().get("categories", []):
            if not (c.get("active") and c.get("use_ml")):
                continue
            key = c["key"]
            meta = rule_meta(key)
            now_meta, ew_meta = meta.get("now") or {}, meta.get("ew") or {}
            if not now_meta.get("features"):
                continue
            mp = _paths(key)[0]
            p_now = _predict(mp.with_name(mp.stem + "_now.txt"), table,
                             now_meta["features"])
            p_ew = _predict(mp, table, ew_meta.get("features"))
            t_now = float(now_meta.get("threshold", 0.5) or 0.5)
            t_ew = float(ew_meta.get("threshold", 0.5) or 0.5)
            cx.executemany(
                "INSERT OR REPLACE INTO preds VALUES (?,?,?,?,?,?,?)",
                [(day, key, str(a), float(p_now.get(a, 0)),
                  float(p_ew.get(a, 0)),
                  int(p_now.get(a, 0) >= t_now),
                  int(p_ew.get(a, 0) >= t_ew)) for a in table.index])
            out["rules"][key] = {"scored": int(len(table)),
                                 "pred_flag_now": int((p_now >= t_now).sum())}
    return out


def verify_predictions() -> dict:
    """Realised walk-forward check: the most recent snapshot from an EARLIER
    trading day, verified against today's active accounts and their actual
    rule state. Plain-language metrics per rule."""
    from webapp import af_registry
    table, active = _obs_table_cached()
    if not len(table) or not active:
        return {"error": "no feature table"}
    today = str(_TABLE_CACHE.get("active_day") or "")
    out = {"verified_against_day": today, "rules": {}}
    with _pred_cx() as cx:
        for c in af_registry.load().get("categories", []):
            if not (c.get("active") and c.get("use_ml")):
                continue
            key = c["key"]
            row = cx.execute(
                "SELECT MAX(as_of_day) FROM preds WHERE rule = ? "
                "AND as_of_day < ?", (key, today)).fetchone()
            snap_day = row[0] if row else None
            if not snap_day:
                out["rules"][key] = {
                    "status": "accumulating",
                    "explain": "first snapshot taken today; the first "
                               "verification lands on the next trading day"}
                continue
            preds = dict(cx.execute(
                "SELECT account, flag_now FROM preds WHERE rule = ? "
                "AND as_of_day = ?", (key, snap_day)).fetchall())
            truth = _rule_mask(c.get("definition", ""), table)
            cohort = [a for a in table.index if str(a) in active]
            tp = fp = fn = tn = 0
            for a in cohort:
                p = int(preds.get(str(a), 0))
                t = bool(truth.get(a))
                tp += p and t; fp += p and not t
                fn += (not p) and t; tn += (not p) and not t
            n = max(tp + fp + fn + tn, 1)
            out["rules"][key] = {
                "status": "verified", "scope": "ML model over ALL active "
                "accounts (distinct from the tape-scan flagged table)",
                "snapshot_day": snap_day,
                "active_accounts": n,
                "predicted_flag_active": tp + fp,
                "actually_flagged": tp + fn,
                "true_positives": tp, "false_positives": fp,
                "false_negatives": fn,
                "precision": round(tp / max(tp + fp, 1), 3),
                "recall": round(tp / max(tp + fn, 1), 3),
                "accuracy": round((tp + tn) / n, 3),
                "base_rate": round((tp + fn) / n, 4),
                "explain": (f"ML model over all {n} accounts active on "
                            f"{today}: predicted {tp + fp} would flag, "
                            f"{tp + fn} actually did; {tp} correct → caught "
                            f"{tp}/{tp + fn} of real flags (recall) and "
                            f"{tp}/{tp + fp} of its picks were right "
                            f"(precision); {fp} false alarms, {fn} missed."),
            }
    try:
        (ART / "rule_verification.json").write_text(
            json.dumps(out), encoding="utf-8")
    except Exception:
        pass
    return out


def rule_hits_stats(key: str) -> dict | None:
    """Non-blocking 'N rule hits (90d obs table) / M accounts observed' for
    ANY rule -- the stat card every tab carries. None while the shared
    table is still building (never blocks a request)."""
    now = time.time()
    if _TABLE_CACHE.get("at", 0) <= now - 900 \
            and not _obs_from_disk(ttl=7200.0):
        return None
    from webapp import af_registry
    table, _ = _obs_table_cached()
    if not len(table):
        return None
    for c in af_registry.load().get("categories", []):
        if c["key"] == key:
            mask = _rule_mask(c.get("definition", ""), table)
            return {"rule_hits": int(mask.sum()),
                    "n_accounts": int(len(table))}
    return None


def latest_verification() -> dict:
    try:
        return json.loads((ART / "rule_verification.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return {}


def profiling_scan(as_of: str | None = None,
                   start: str | None = None) -> dict:
    """Behavioural Profiling = the latency-format table UNIONED across every
    ACTIVE registry rule (an inactive rule contributes NOTHING). One row per
    (account, profile). ML column = fits-the-category-now model, EW 5D =
    5-day early warning, uniformly across profiles; latency rows carry their
    full evidence set, generic rules show '—' where a field doesn't apply."""
    from webapp import af_registry, latency_arb
    end_dt, start_dt = latency_arb._range_window(as_of, start)
    cats = [c for c in af_registry.load().get("categories", [])
            if c.get("active")]
    rows, building = [], []
    summaries: dict = {}
    for c in cats:
        key, label = c["key"], c.get("label") or c["key"]
        if key == "latency_arbitrage":
            r = latency_arb.scan(as_of=as_of, start=start)
            if r.get("building"):
                building.append(label)
                continue
            for row in r.get("rows", []):
                rows.append(dict(row, profile=label, rule=key))
            summaries[label] = {
                "shown": len(r.get("rows", [])),
                "flagged_trades": r.get("n_flagged_trades"),
                "note": "identical to the Latency Arbitrage tab "
                        "(same scan, same range, same escalation rule)"}
        elif c.get("use_ml"):
            r = rule_scan(key, as_of=as_of, start=start)
            if r.get("building"):
                building.append(label)
                continue
            for row in r.get("rows", []):
                # unify semantics: ML column = NOW classifier, EW = 5d model
                rows.append(dict(row, profile=label, rule=key,
                                 ml=row.get("ml_now"), ew=row.get("ml")))
            summaries[label] = {
                "shown": len(r.get("rows", [])),
                "rule_hits": r.get("rule_hits"),
                "n_accounts": r.get("n_accounts"),
                "note": "identical to this rule's own tab (same scan, "
                        "same range); rule_hits is the unranged hindsight "
                        "total the Rules & Coverage chip shows"}
        else:
            now = time.time()
            # try memory THEN the disk copy before declaring "building"
            if _TABLE_CACHE.get("at", 0) <= now - 900 \
                    and not _obs_from_disk(ttl=7200.0):
                building.append(label)
                if not _BUILD_THREAD.locked():
                    def _build():
                        with _BUILD_THREAD:
                            try:
                                _obs_table_cached()
                            except Exception:
                                pass
                    threading.Thread(target=_build, daemon=True).start()
                continue
            table, _ = _obs_table_cached()
            if not len(table):
                continue
            mask = _rule_mask(c.get("definition", ""), table)
            hits = table.loc[mask]
            if "n_trades" in hits:
                hits = hits.sort_values("n_trades", ascending=False)
            last_map = _TABLE_CACHE.get("last_ts") or {}
            # range-filter FIRST, cap AFTER -- capping first undercounted
            # the accounts active in the selected range.
            ts_ser = pd.to_datetime(pd.Series(
                [last_map.get(str(a), "") for a in hits.index],
                index=hits.index), errors="coerce")
            in_range = hits.loc[ts_ser.isna() | (ts_ser >= start_dt)]
            summaries[label] = {
                "shown": int(min(len(in_range), 400)),
                "rule_hits": int(mask.sum()),
                "n_accounts": int(len(table)),
                "in_range": int(len(in_range)),
                "note": "hindsight rule over the shared feature table; "
                        "rule_hits matches the Rules & Coverage chip"}
            for a, hrow in in_range.head(400).iterrows():
                ts = str(last_map.get(str(a), ""))
                rows.append({
                    "account": str(a), "profile": label, "rule": key,
                    "rule_hit": True, "risk": None, "ml": None, "ew": None,
                    "obs_trades": int(hrow.get("n_trades") or 0),
                    "med_hold_s": round(float(hrow.get("med_hold_s") or 0), 1)
                    if pd.notna(hrow.get("med_hold_s")) else None,
                    "event_share": round(float(hrow.get("event_share") or 0), 3),
                    "event_trades": int(hrow.get("event_trades") or 0),
                    "event_pnl": round(float(hrow.get("event_pnl") or 0), 2),
                    "last_ts": ts, "verdict": "monitor",
                })
    rows.sort(key=lambda r: -(r.get("risk") or 0))
    # NO global cap: per-rule lists match each rule tab EXACTLY, so
    # filtering to one class mirrors that tab row for row.
    return {"rows": rows, "n_rows": len(rows),
            "profiles": [c.get("label") or c["key"] for c in cats],
            "summaries": summaries,
            "building": building or None,
            "range_start": str(start_dt)[:16], "range_end": str(end_dt)[:16],
            "generated_at": str(datetime.utcnow())}


def account_rule_status(account: str) -> list[dict]:
    """For the account page: every registry rule this client currently
    satisfies, with its most recent trigger time (the client's latest trade
    inside the satisfying window, stamped by the scan build)."""
    from webapp import af_registry
    table, _ = _obs_table_cached()
    if account not in table.index:
        return []
    row = table.loc[[account]]
    out = []
    built = datetime.utcfromtimestamp(
        _TABLE_CACHE.get("at", time.time())).strftime("%Y-%m-%d %H:%M UTC")
    for c in af_registry.load().get("categories", []):
        if not c.get("active"):
            continue
        try:
            hit = bool(_rule_mask(c.get("definition", ""), row).iloc[0])
        except Exception:
            hit = False
        if hit:
            out.append({"key": c["key"], "label": c["label"],
                        "use_ml": bool(c.get("use_ml")),
                        "last_trigger": built,
                        "definition": c.get("definition", "")})
    return out


def _dataset(key: str, as_of: datetime | None = None):
    """Obs features (90d) + forward flag (rule on the NEXT n days' behaviour)
    + forward equity direction. `as_of` moves the whole frame back in time."""
    now = as_of or datetime.utcnow()
    split = now - timedelta(days=FORWARD_DAYS)
    start = split - timedelta(days=OBS_DAYS)
    all_trades = _window_trades(start, now)
    if not len(all_trades):
        return None
    obs = all_trades[all_trades["open_time"] < split]
    fwd = all_trades[all_trades["open_time"] >= split]
    if not len(obs):
        return None
    anchors = combined_anchors()
    obs_table = feature_table(obs, anchors)
    fwd_table = feature_table(fwd, anchors) if len(fwd) else pd.DataFrame()
    definition = _definition(key)
    fwd_flag = _rule_mask(definition, fwd_table) if len(fwd_table) \
        else pd.Series(dtype=bool)
    fwd_pnl = fwd.groupby("account_key", observed=True)["net_profit"].sum() \
        if len(fwd) else pd.Series(dtype=float)
    obs_flag = _rule_mask(definition, obs_table)
    # most recent activity per account over the FULL frame (incl. forward
    # days): the tab's range filter selects on this.
    last_ts = all_trades.groupby(
        "account_key", observed=True)["open_time"].max()
    return {"obs_table": obs_table, "obs_flag": obs_flag,
            "fwd_flag": fwd_flag, "fwd_pnl": fwd_pnl,
            "last_ts": last_ts, "definition": definition}


def _fit_scored(X: pd.DataFrame, y: pd.Series,
                groups: pd.Series | None = None) -> tuple:
    """OOF-evaluated LightGBM classifier + tuned threshold. Returns
    (fitted_final_model, metrics_dict) or (None, {'error':...}).
    `groups` (account ids) forces account-grouped folds: the sequential
    panel holds many rows per account, and ungrouped CV would leak."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (roc_auc_score, precision_score,
                                 recall_score, average_precision_score)
    if int(y.sum()) < 10:
        return None, {"error": f"only {int(y.sum())} positive labels"}
    Xv, yv = X.fillna(0).to_numpy(), y.to_numpy()
    if groups is not None:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            folds = StratifiedGroupKFold(5, shuffle=True, random_state=0) \
                .split(Xv, yv, groups.to_numpy())
        except Exception:
            from sklearn.model_selection import GroupKFold
            folds = GroupKFold(5).split(Xv, yv, groups.to_numpy())
    else:
        folds = StratifiedKFold(5, shuffle=True, random_state=0).split(Xv, yv)
    oof = np.zeros(len(yv))
    for tr, te in folds:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=63, class_weight="balanced",
                               random_state=0, verbosity=-1)
        m.fit(Xv[tr], yv[tr])
        oof[te] = m.predict_proba(Xv[te])[:, 1]
    auc = roc_auc_score(yv, oof)
    pr_auc = average_precision_score(yv, oof)
    base_rate = float(yv.mean())
    best = (0.5, 0.0, 0.0, 0.0)
    for thr in np.linspace(0.05, 0.95, 91):
        p = precision_score(yv, oof >= thr, zero_division=0)
        r = recall_score(yv, oof >= thr, zero_division=0)
        if min(p, r) > best[3]:
            best = (float(thr), p, r, min(p, r))
    final = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=63, class_weight="balanced",
                               random_state=0, verbosity=-1)
    final.fit(Xv, yv)
    return final, {
        "auc": round(float(auc), 3), "precision": round(best[1], 3),
        "recall": round(best[2], 3), "threshold": round(best[0], 3),
        "pr_auc": round(float(pr_auc), 3), "base_rate": round(base_rate, 4),
        "precision_lift": round(best[1] / max(base_rate, 1e-9), 1),
        "positives": int(y.sum()), "labelable": int(len(y)),
        "features": list(X.columns)}


#: Sequential active-day panel: evaluation boundaries every this many days.
#: 7 = weekly (9 feature-table builds per training run); lower it toward 1
#: for true per-day granularity at proportionally higher training cost.
PANEL_STEP_DAYS = 7
PANEL_SPLITS = 8


def _panel_tables(cutoff: datetime | None = None):
    """The rule-INDEPENDENT part of the panel: boundary dates, the trades
    frame, and one feature table per boundary. Built ONCE and shared across
    every rule's training -- the tables don't depend on the rule, only the
    labels do (this was a 3x waste when each rule rebuilt them)."""
    now = (cutoff or datetime.utcnow()).replace(hour=0, minute=0, second=0,
                                                microsecond=0)
    last = now - timedelta(days=FORWARD_DAYS)
    splits = [last - timedelta(days=PANEL_STEP_DAYS * j)
              for j in range(PANEL_SPLITS, -1, -1)]
    all_trades = _window_trades(splits[0] - timedelta(days=OBS_DAYS), now)
    if not len(all_trades):
        return None
    anchors = combined_anchors()
    ot = all_trades["open_time"]
    tables = {}
    for s in splits:
        obs = all_trades[(ot > s - timedelta(days=OBS_DAYS)) & (ot <= s)]
        if len(obs):
            tables[s] = feature_table(obs, anchors)
    return {"now": now, "splits": splits, "trades": all_trades,
            "tables": tables}


def train_all_rules(cutoff: datetime | None = None) -> dict:
    """Nightly-efficient training: build the shared panel ONCE, then fit
    every active use_ml rule's NOW+EW pair off it."""
    from webapp import af_registry
    shared = _panel_tables(cutoff)
    out = {}
    for c in af_registry.load().get("categories", []):
        if c.get("active") and c.get("use_ml"):
            try:
                out[c["key"]] = _train_generic(c["key"], shared=shared)
            except Exception as error:
                out[c["key"]] = {"error": f"{type(error).__name__}: {error}"}
    return out


def _panel_dataset(key: str, cutoff: datetime | None = None,
                   shared: dict | None = None):
    """SEQUENTIAL framework (per the Sep-9 spec): an account's active days
    form a sequence; each sample uses features known at the PREVIOUS
    evaluation boundary and asks whether the account -- being active in
    between -- triggers the rule at the NEXT evaluated boundary.
      NOW target: rule holds at the next evaluated state.
      EW  target: NOW  AND  equity rises over the FORWARD_DAYS after that
                  state (realised trade P&L: deposits/withdrawals excluded
                  by construction).
    Boundaries are PANEL_STEP_DAYS apart (weekly by default) -- the same
    structure as per-active-day, at tractable training cost.

    TRAINING CUTOFF: midnight of the CURRENT day. Nothing from today ever
    enters training, so tonight's snapshot predictions come from a model
    that has never seen the day they will be verified against -- the
    realised walk-forward cards are valid by construction."""
    shared = shared or _panel_tables(cutoff)
    if shared is None:
        return None
    now = shared["now"]
    splits = shared["splits"]
    all_trades = shared["trades"]
    tables = shared["tables"]
    definition = _definition(key)
    ot = all_trades["open_time"]
    X_parts, y_now_parts, y_ew_parts, group_parts = [], [], [], []
    fwd_parts = []
    for a, b in zip(splits[:-1], splits[1:]):
        ta, tb = tables.get(a), tables.get(b)
        if ta is None or tb is None:
            continue
        # "next active": the account traded between the two boundaries.
        active = set(all_trades.loc[(ot > a) & (ot <= b),
                                    "account_key"].astype(str))
        idx = ta.index[ta.index.astype(str).isin(active)]
        if not len(idx):
            continue
        flag_b = _rule_mask(definition, tb).reindex(idx).fillna(False)
        fwd = all_trades.loc[(ot > b)
                             & (ot <= b + timedelta(days=FORWARD_DAYS))]
        fwd_pnl = fwd.groupby("account_key", observed=True)["net_profit"] \
            .sum().reindex(idx).fillna(0.0)
        X_parts.append(ta.loc[idx])
        y_now_parts.append(flag_b.astype(int))
        y_ew_parts.append((flag_b & (fwd_pnl > 0)).astype(int))
        fwd_parts.append(fwd_pnl)
        group_parts.append(pd.Series(idx.astype(str), index=idx))
    if not X_parts:
        return None
    # positional index throughout: the same account appears in several
    # pairs, and duplicate labels would corrupt any label-based alignment.
    return {"X": pd.concat(X_parts, ignore_index=True),
            "y_now": pd.concat(y_now_parts).reset_index(drop=True),
            "y_ew": pd.concat(y_ew_parts).reset_index(drop=True),
            "fwd_pnl": pd.concat(fwd_parts).reset_index(drop=True),
            "groups": pd.concat(group_parts).reset_index(drop=True),
            "definition": definition, "pairs": len(X_parts),
            "train_cutoff": str(now)[:10]}


def _now_features(obs_table: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Feature set for the CLASSIFICATION (rule-now) model: every column
    EXCEPT the fields the definition itself references -- otherwise the
    model just reads the rule back and reports a fake 1.0."""
    import re as _re
    used = set(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", definition or "")) \
        - {"and", "or", "not", "abs", "True", "False"}
    return obs_table[[c for c in obs_table.columns if c not in used]]


def train_rule(key: str) -> dict:
    """TWO models per rule, one artifact pair each:
    - EW  (early warning): does the rule hold over the NEXT 5 days with the
      client's equity rising?  (the existing forward target)
    - NOW (classification): does the account satisfy the rule TODAY, judged
      WITHOUT the definition's own fields (generalisation, not echo).
    Latency additionally trains its bespoke tape-proved detector; the
    generic EW model feeds the latency tab's EW 5D column."""
    if key == "latency_arbitrage":
        from webapp import latency_arb
        try:
            tape = latency_arb.train_model()
        except Exception as error:
            # tick store busy (another process holds it): the existing tape
            # artifact stands; the generic pair below must still train.
            tape = {"error": f"{type(error).__name__}: {error}"}
        gen = _train_generic(key)
        if isinstance(gen, dict) and not gen.get("error"):
            gen["tape"] = tape
            try:
                _paths(key)[1].write_text(json.dumps(gen), encoding="utf-8")
            except Exception:
                pass
            return gen
        out = dict(tape if isinstance(tape, dict) else {})
        out["generic_error"] = (gen or {}).get("error")
        return out
    return _train_generic(key)


def _train_generic(key: str, shared: dict | None = None) -> dict:
    data = _panel_dataset(key, shared=shared)
    if data is None:
        return {"error": "no trades in window"}
    X = data["X"]
    keep = (X["n_trades"] >= 10) if "n_trades" in X \
        else pd.Series(True, index=X.index)
    X = X.loc[keep].astype(float)
    y_now = data["y_now"].loc[keep]
    y_ew = data["y_ew"].loc[keep]
    fwd_pnl = data["fwd_pnl"].loc[keep]
    groups = data["groups"].loc[keep]
    mp, meta_p, _ = _paths(key)

    def _undersample(Xs, ys, gs, max_ratio=10):
        """Cap negatives at max_ratio x positives (random, seeded) --
        LightGBM-appropriate imbalance handling; class weights and
        threshold tuning do the rest. (SMOTE deliberately not used:
        synthetic interpolation in a 250-dim account space manufactures
        accounts that cannot exist.)"""
        pos = ys[ys == 1].index
        neg = ys[ys == 0].index
        if len(neg) <= max_ratio * max(len(pos), 1):
            return Xs, ys, gs
        rng = np.random.default_rng(0)
        keep_neg = pd.Index(rng.choice(neg.to_numpy(),
                                       max_ratio * len(pos), replace=False))
        idx = pos.append(keep_neg)
        return Xs.loc[idx], ys.loc[idx], gs.loc[idx]

    # EW = the ACTION-RELEVANT CONDITIONAL: among samples whose NOW label
    # fired (rule triggers at the next active boundary), is the account
    # profitable over the following 5 days? Exact-zero forward P&L rows are
    # EXCLUDED (the account did not trade forward: absence of evidence).
    ew_mask = (y_now == 1) & (fwd_pnl != 0)
    y_ew_cond = (fwd_pnl.loc[ew_mask] > 0).astype(int)
    Xe, ye, ge = _undersample(X.loc[ew_mask], y_ew_cond,
                              groups.loc[ew_mask])
    ew_model, ew = _fit_scored(Xe, ye, ge)
    if ew_model is not None:
        ew["target"] = ("conditional: profitable over next 5d GIVEN the "
                        "rule fires at next active day; zero-forward-P&L "
                        "rows excluded")
        ew_model.booster_.save_model(str(mp))

    X_now = _now_features(X, data["definition"])
    Xn, yn, gn = _undersample(X_now, y_now, groups)
    now_model, now = _fit_scored(Xn, yn, gn)
    if now_model is not None:
        now_model.booster_.save_model(str(mp.with_name(mp.stem + "_now.txt")))

    if ew_model is None and now_model is None:
        return {"error": f"EW: {ew.get('error')} · NOW: {now.get('error')}"}
    # top level keeps the EW metrics (existing consumers); both models sit
    # side by side under 'ew' / 'now'.
    meta = dict(ew if ew_model is not None else {})
    meta.update({
        "ew": ew, "now": now,
        "train_cutoff": data.get("train_cutoff"),
        "framework": "sequential active-day panel",
        "panel_step_days": PANEL_STEP_DAYS, "panel_pairs": data["pairs"],
        "samples": int(len(X)),
        "obs_days": OBS_DAYS, "forward_days": FORWARD_DAYS,
        "def_hash": _def_hash(key), "definition": data["definition"],
        "features": list(data["X"].columns),
        "join_error": _JOIN_ERR[0] or None,
        "trained_at": str(datetime.utcnow())})
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    _CACHE.pop(key, None)
    return meta


def _maybe_retrain(key: str) -> None:
    """Definition changed since the model was trained -> retrain, once, in
    the background."""
    meta = rule_meta(key)
    if meta and meta.get("def_hash") == _def_hash(key):
        return
    lock = _TRAIN_LOCKS.setdefault(key, threading.Lock())
    if lock.locked():
        return

    def _run():
        with lock:
            train_rule(key)
    threading.Thread(target=_run, daemon=True).start()


def _predict(booster_path, obs_table: pd.DataFrame,
             features: list | None) -> pd.Series:
    out = pd.Series(0.0, index=obs_table.index)
    if not features or not booster_path.exists():
        return out
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(booster_path))
        X = obs_table.reindex(columns=features).fillna(0).astype(float)
        out = pd.Series(booster.predict(X.to_numpy()),
                        index=obs_table.index).clip(0, 1)
    except Exception:
        pass
    return out


def rule_scan(key: str, refresh: bool = False, as_of: str | None = None,
              start: str | None = None) -> dict:
    """Latency-format scan for any registry rule: both models scored per
    account (EW 5d + rule-now classification), table filtered to accounts
    active in [start, as_of] -- defaulting to the last 24 LIVE hours."""
    if key == "latency_arbitrage":
        from webapp import latency_arb
        return latency_arb.scan(refresh=refresh, as_of=as_of, start=start)
    from webapp.latency_arb import _range_window, _range_filter
    end_dt, start_dt = _range_window(as_of, start)
    historical = bool(as_of) \
        and str(as_of) != datetime.utcnow().strftime("%Y-%m-%d")
    _, meta_p, scan_p = _paths(key)
    if not refresh and not historical:
        cached = _CACHE.get(key)
        if cached:
            _maybe_retrain(key)
            return _range_filter(cached, end_dt, start_dt)
        try:
            disk = json.loads(scan_p.read_text(encoding="utf-8"))
            disk["from_disk"] = True
            _CACHE[key] = disk
            _maybe_retrain(key)
            return _range_filter(disk, end_dt, start_dt)
        except Exception:
            pass
    if historical:
        hkey = f"{key}:{as_of}"
        cached = _CACHE.get(hkey)
        if cached and not refresh:
            return _range_filter(cached, end_dt, start_dt)
    # NEVER build on the request thread: cold cache or refresh kicks ONE
    # background build; the tab polls the building flag.
    lock = _SCAN_LOCKS.setdefault(key, threading.Lock())
    if lock.locked():
        return {"building": True, "rows": [],
                "note": f"{key} scan building in the background"}

    def _build_scan():
        with lock:
            try:
                _rule_scan_build(key, end_dt if historical else None,
                                 as_of if historical else None)
            except Exception as error:
                _CACHE[key] = {"rows": [], "error":
                               f"{type(error).__name__}: {error}"}
    threading.Thread(target=_build_scan, daemon=True).start()
    return {"building": True, "rows": [],
            "note": f"{key} scan started in the background"}


_SCAN_LOCKS: dict = {}


def _rule_scan_build(key: str, as_of_dt=None, as_of: str | None = None):
    _, meta_p, scan_p = _paths(key)
    if as_of_dt is not None:
        # historical as-of: honest rebuild at that date.
        data = _dataset(key, as_of=as_of_dt)
        if data is None:
            _CACHE[key] = {"error": "no trades", "rows": []}
            return
        obs_table, obs_flag = data["obs_table"], data["obs_flag"]
        last_map = data["last_ts"].astype(str).to_dict()
        fwd_pnl = data["fwd_pnl"]
    else:
        # DEFAULT (live) path: pretrained models over the SHARED cached
        # feature table -- seconds, not a per-rule trade-window rebuild.
        obs_table, _ = _obs_table_cached()
        if not len(obs_table):
            _CACHE[key] = {"error": "feature table unavailable", "rows": []}
            return
        obs_flag = _rule_mask(_definition(key), obs_table)
        last_map = _TABLE_CACHE.get("last_ts") or {}
        fwd_pnl = None
    meta = rule_meta(key)
    mp = _paths(key)[0]
    ew_meta = meta.get("ew") or meta
    now_meta = meta.get("now") or {}
    ml = _predict(mp, obs_table, ew_meta.get("features") or meta.get("features"))
    ml_now = _predict(mp.with_name(mp.stem + "_now.txt"), obs_table,
                      now_meta.get("features"))
    thr = float(ew_meta.get("threshold", 0.5) or 0.5)
    thr_now = float(now_meta.get("threshold", 0.5) or 0.5)
    keep_idx = obs_table.index[(obs_flag) | (ml >= thr) | (ml_now >= thr_now)]
    keep = obs_table.loc[keep_idx]
    rows = []
    for a in keep_idx:
        mlv, mnv = float(ml.get(a, 0.0)), float(ml_now.get(a, 0.0))
        risk = round(100 * (0.4 * float(bool(obs_flag.get(a)))
                            + 0.3 * mlv + 0.3 * mnv), 1)
        rows.append({
            "account": str(a), "risk": risk,
            "ml": round(mlv, 3), "ml_now": round(mnv, 3),
            "rule_hit": bool(obs_flag.get(a)),
            "last_ts": str(last_map.get(str(a), "")),
            "event_share": round(float(keep.at[a, "event_share"]), 3)
            if "event_share" in keep else None,
            "event_trades": int(keep.at[a, "event_trades"])
            if "event_trades" in keep else None,
            "event_pnl": round(float(keep.at[a, "event_pnl"]), 2)
            if "event_pnl" in keep else None,
            "obs_trades": int(keep.at[a, "n_trades"])
            if "n_trades" in keep else None,
            "avg_daily_cost": round(float(keep.at[a, "avg_daily_pnl"]), 2)
            if "avg_daily_pnl" in keep else None,
            "fwd_pnl": round(float(fwd_pnl.get(a, 0.0)), 2)
            if fwd_pnl is not None else None,
            "verdict": ("escalate" if risk >= 75 else
                        "restrict" if risk >= 55 else "monitor"),
        })
    rows.sort(key=lambda r: -r["risk"])
    out = {"rows": rows[:400], "meta": meta,
           "generated_at": str(datetime.utcnow()),
           "n_accounts": int(len(obs_table)),
           "rule_hits": int(obs_flag.sum()),
           "verification": (latest_verification().get("rules") or {}
                            ).get(key)}
    if as_of is not None:
        out["as_of"] = str(as_of)
        _CACHE[f"{key}:{as_of}"] = out
        return
    _CACHE[key] = out
    try:
        scan_p.write_text(json.dumps(out, default=str), encoding="utf-8")
    except Exception:
        pass
