"""P1 Engine B -- Toxic Flow: scan, persistence, audit and automation feed.

Section 12 of the specification fixes the architecture: "The Markout Engine
should be implemented once and consumed by all three P1 engines so calculations
remain consistent." So this engine does NOT take its own pass over the tick
tape. The latency scan (Engine A) already builds the seven-horizon markout
frame for every trade in the window; at the end of that pass it hands the frame
to `build()` here, which scores Engine B and persists the result. One tape
pass, two engines, identical markouts -- and Engine B costs seconds rather than
the ~13 minutes a second pass would take against a single-writer tick store.

Consequences worth knowing:
  * Engine B is exactly as fresh as the latency scan (currently every 20 min).
  * Changing a rule that alters what counts as a toxic TRADE (the s5.2 markout
    minimums, the holding-time filter) needs a new tape pass; the tab says so
    and offers a rescan. Everything downstream of that is already stored.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from webapp import toxic_spec, toxic_tags

ROOT = Path(__file__).resolve().parent
RULES_PATH = ROOT / "toxic_rules.json"
SCAN_CACHE = ROOT / "artifacts" / "toxic_scan.json"
FEATURES_PATH = ROOT / "artifacts" / "toxic_features.parquet"
TOXIC_ORDERS_PATH = ROOT / "artifacts" / "toxic_orders.parquet"
MARKOUT_SAMPLE_PATH = ROOT / "artifacts" / "markout_sample.parquet"
DB = ROOT.parent / "app.db"

#: Rules are the spec's s5.1/s5.2 parameters plus the engine's own display and
#: governance switches. The markout horizons themselves are NOT here: they are
#: the shared s2 framework, owned by the latency rules file, because both
#: engines must measure the same seven moments (s12).
DEFAULT_RULES: dict = {
    #: s13 "Automated actions are governed separately from classification."
    #: Shadow mode logs every state change and applies nothing.
    "shadow_mode": True,
    #: Rows below this score are summarised, not listed (s10: under 50 is
    #: passive monitoring). 25 keeps the Monitor tier visible.
    "table_min_score": 25.0,
    "max_rows": 500,
}
DEFAULT_RULES.update(json.loads(json.dumps(toxic_spec.TOXIC_DEFAULT_RULES)))

_CACHE: dict = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------
def load_rules() -> dict:
    merged = dict(DEFAULT_RULES)
    try:
        disk = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        for key, value in (disk or {}).items():
            if key in DEFAULT_RULES and isinstance(DEFAULT_RULES[key], dict) \
                    and isinstance(value, dict):
                nested = dict(DEFAULT_RULES[key])
                nested.update(value)
                merged[key] = nested
            elif key in DEFAULT_RULES:
                merged[key] = value
    except Exception:
        return dict(DEFAULT_RULES)
    return merged


def save_rules(rules: dict) -> dict:
    """Merge over the EFFECTIVE rules, key by key. The tab posts only what it
    edits, and a partial post must never reset a stored threshold."""
    merged = load_rules()
    for key, value in (rules or {}).items():
        if key not in DEFAULT_RULES:
            continue
        if isinstance(DEFAULT_RULES[key], dict) and isinstance(value, dict):
            nested = dict(merged.get(key) or {})
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    RULES_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    with _LOCK:
        _CACHE.clear()
    return merged


def _rules_key(rules: dict) -> str:
    """Identity of the rules that MATTER to a stored scan. Display-only keys
    are excluded so editing them does not mark the scan stale."""
    ignore = {"table_min_score", "max_rows", "shadow_mode"}
    return json.dumps({k: v for k, v in rules.items() if k not in ignore},
                      sort_keys=True, default=str)


# --------------------------------------------------------------------------
# audit + history (same idioms as latency_arb: lazy CREATE TABLE, tolerant ALTER)
# --------------------------------------------------------------------------
def _ensure_audit() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS toxic_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, account TEXT,
        state TEXT, score REAL, confidence REAL, action TEXT, mode TEXT,
        operator TEXT, status TEXT, note TEXT, tags TEXT)""")
    conn.commit()
    return conn


def _audit(account: str, state: str, score: float, confidence: float, action: str,
           mode: str, operator: str, status: str, note: str = "",
           tags: list | None = None) -> None:
    try:
        conn = _ensure_audit()
        conn.execute(
            "INSERT INTO toxic_audit (ts, account, state, score, confidence, action,"
            " mode, operator, status, note, tags) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), str(account), state, float(score), float(confidence),
             action, mode, operator, status, note, json.dumps(tags or [])))
        conn.commit()
        conn.close()
    except Exception:
        pass


def audit_tail(limit: int = 200) -> list[dict]:
    try:
        conn = _ensure_audit()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM toxic_audit ORDER BY id DESC LIMIT ?",
                            (int(limit),)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def decide(account: str, decision: str, operator: str = "risk", note: str = "") -> dict:
    """Analyst decision on one account. s10 requires a human between a critical
    classification and any control, so confirm/dismiss both demand a note."""
    if decision not in ("confirm", "dismiss", "clear"):
        return {"error": "unknown decision"}
    if decision in ("confirm", "dismiss") and not (note or "").strip():
        return {"error": "a note is required"}
    current = account_toxic_current(account) or {}
    _audit(account, str(current.get("state") or ""), float(current.get("score") or 0),
           float(current.get("confidence") or 0), str(current.get("action") or ""),
           "shadow" if load_rules().get("shadow_mode", True) else "live",
           operator, decision, note, current.get("tags") or [])
    return {"ok": True, "account": account, "decision": decision}


def _ensure_history() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS toxic_history (
        day TEXT, account TEXT, score REAL, confidence REAL, severity REAL,
        adverse_usd REAL, toxic_trades REAL, state TEXT, tier TEXT,
        PRIMARY KEY (day, account))""")
    conn.commit()
    return conn


def record_history(rows: list[dict], day: str | None = None) -> None:
    """One row per (day, account). The day is replaced wholesale so a rescan
    corrects the day instead of appending a second, conflicting version."""
    if not rows:
        return
    day = day or str(datetime.utcnow().date())
    try:
        conn = _ensure_history()
        conn.execute("DELETE FROM toxic_history WHERE day = ?", (day,))
        conn.executemany(
            "INSERT OR REPLACE INTO toxic_history VALUES (?,?,?,?,?,?,?,?,?)",
            [(day, str(r["account"]), float(r.get("score") or 0),
              float(r.get("confidence") or 0), float(r.get("severity") or 0),
              float(r.get("adverse_usd") or 0), float(r.get("toxic_trades") or 0),
              str(r.get("state") or ""), str(r.get("risk_tier") or "")) for r in rows])
        conn.commit()
        conn.close()
    except Exception:
        pass


def history_trend(days: int = 30) -> list[dict]:
    try:
        conn = _ensure_history()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT day, COUNT(*) AS accounts,
                      SUM(CASE WHEN tier IN ('critical','high') THEN 1 ELSE 0 END) AS actionable,
                      SUM(adverse_usd) AS adverse_usd, AVG(score) AS avg_score
               FROM toxic_history GROUP BY day ORDER BY day DESC LIMIT ?""",
            (int(days),)).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]
    except Exception:
        return []


def account_toxic_history(account: str, limit: int = 60) -> list[dict]:
    try:
        conn = _ensure_history()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM toxic_history WHERE account = ? ORDER BY day DESC LIMIT ?",
            (str(account), int(limit))).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]
    except Exception:
        return []


# --------------------------------------------------------------------------
# the build hook, called by the latency scan with the shared markout frame
# --------------------------------------------------------------------------
def _event_share(trades: pd.DataFrame) -> pd.Series | None:
    """Share of each account's entries within the macro-calendar tolerance.
    Vectorised over the whole frame -- one searchsorted, not one call per
    account -- because this runs over every scanned trade."""
    try:
        from webapp import econ_calendar
        anchors = econ_calendar.event_times_utc("high")
        if not anchors:
            return None
        stamps = pd.to_datetime(trades["open_time"]).to_numpy("datetime64[s]")
        marks = np.array(sorted(anchors), dtype="datetime64[s]")
        idx = np.searchsorted(marks, stamps)
        before = marks[np.clip(idx - 1, 0, len(marks) - 1)]
        after = marks[np.clip(idx, 0, len(marks) - 1)]
        gap = np.minimum(np.abs((stamps - before).astype("int64")),
                         np.abs((after - stamps).astype("int64")))
        near = pd.Series(gap <= 600, index=trades.index)  # +/- 10 minutes
        return near.groupby(trades["account_key"], observed=True).mean()
    except Exception:
        return None


def build(trades: pd.DataFrame, latency_rules: dict, horizons: list, early: list,
          fallback: list, late: float, mcol, hlabel,
          replicated: dict | None = None) -> dict:
    """Score Engine B from the shared markout frame and persist the result.

    Called at the end of the latency scan (s12: one Markout Engine). Never
    raises into the caller -- Engine A's scan must not fail because Engine B
    did. Returns a small summary for the latency scan's own envelope.
    """
    started = time.time()
    rules = load_rules()
    # One fade definition for both engines (toxic_spec per-trade signatures).
    spec_rules = dict(rules, min_decay_pct=float(latency_rules.get("min_decay_pct", 50.0)))
    try:
        metrics, summary = toxic_spec.account_metrics(
            trades, spec_rules, horizons, early, fallback, late, mcol, hlabel,
            replicated=replicated, event_share=_event_share(trades))
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    toxic_orders = summary.pop("_toxic_orders", None)
    if metrics is None or not len(metrics):
        return {"error": "no tick-covered trades to score"}

    try:
        metrics.to_parquet(FEATURES_PATH)
    except Exception:
        pass
    try:
        _save_toxic_orders(toxic_orders)
    except Exception:
        pass
    try:
        _save_markout_sample(trades, horizons, mcol)
    except Exception:
        pass

    result = _assemble(metrics, summary, rules, latency_rules)
    result["build_seconds"] = round(time.time() - started, 1)
    try:
        SCAN_CACHE.write_text(json.dumps(result, default=str), encoding="utf-8")
    except Exception:
        pass
    with _LOCK:
        _CACHE["scan"] = result
        _CACHE["rules_key"] = _rules_key(rules)
    try:
        record_history(result["rows"])
    except Exception:
        pass
    _write_audit(result, rules)
    return {"accounts_scored": summary.get("accounts_scored"),
            "rows": len(result["rows"]), "states": summary.get("states"),
            "seconds": result["build_seconds"]}


#: Numeric columns copied onto each row (s8 Common Event Record + s3 metrics).
_ROW_FIELDS = (
    "trades", "toxic_trades", "toxic_early", "toxic_late", "sharp_deals_24h",
    "toxic_trade_rate", "toxic_rate_excess", "adverse_usd", "avg_adverse_usd",
    "markout_usd", "realized_pnl", "gross_profit", "profit_factor",
    "toxic_pnl_concentration", "latency_events", "latency_share",
    "peak_markout_bps", "early_peak_bps", "persistence_ratio", "decay_rate_pct",
    "curve_slope_bps_per_decade", "auc_bps", "auc_excess_bps",
    "consistency_days", "consistency_symbols", "markout_consistency",
    "n_symbols", "top_symbol_share", "buy_toxic_share", "direction_skew",
    "clustered_share", "condition_lift_bps", "wide_spread_share",
    "event_share", "replicated_share", "ref_checked", "ref_agree_share",
    "ref_mo_bps", "profit_spread_ratio", "med_hold_s", "active_days",
    "severity", "stability", "conf_base", "n_profiles",
    "c_seven_horizon_markout", "c_toxic_trade_rate", "c_economic_impact",
    "c_repetition_persistence", "c_profit_concentration",
    "c_reference_corroboration", "c_lp_execution_evidence")


def _num(value, digits: int = 4):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _assemble(metrics: pd.DataFrame, summary: dict, rules: dict,
              latency_rules: dict) -> dict:
    """Metrics frame -> the envelope the tab and the automation feed consume."""
    frame = metrics.sort_values("toxic_score", ascending=False)
    floor = float(rules.get("table_min_score", 25.0))
    # EVERY account above passive monitoring is listed: the rows are also the
    # automation feed, and an account missing from them is logged as cleared.
    # A plain head(max_rows) by score cut 199 accounts still under review
    # (16 Sep 2026). max_rows only bounds the passive accounts that fill in.
    active = frame.loc[frame["state"] != "passive_monitoring"]
    passive = frame.loc[(frame["state"] == "passive_monitoring")
                        & (frame["toxic_score"] >= floor)]
    room = max(int(rules.get("max_rows", 500)) - len(active), 0)
    listed = pd.concat([active, passive.head(room)]).sort_values(
        "toxic_score", ascending=False)
    mode = "shadow" if rules.get("shadow_mode", True) else "live"

    curve_cols = [c for c in frame.columns if c.startswith("mo_") and c.endswith("_bps")]
    hit_cols = [c for c in frame.columns if c.startswith("hit_")]
    rows = []
    for account, r in listed.iterrows():
        row = {
            "account": str(account),
            "score": _num(r["toxic_score"], 1),
            "confidence": _num(r["toxic_confidence"], 1),
            "state": str(r["state"]), "state_label": str(r["state_label"]),
            "risk_tier": str(r["risk_tier"]),
            "curve_profile": str(r["curve_profile"]),
            "markout_signature": str(r["markout_signature"]),
            "profiles": str(r["profiles"] or ""),
            "gates_failed": str(r["gates_failed"] or ""),
            "capped": bool(r["capped"]),
            #: True when the advantage only appears at the long horizons, so
            #: there is no early peak for a persistence ratio to be taken from.
            "late_emerging": bool(r.get("late_emerging", False)),
            "top_symbol": str(r.get("top_symbol") or ""),
            "curve": {c[3:-4]: _num(r[c], 3) for c in curve_cols},
            "hit": {c[4:]: _num(r[c], 3) for c in hit_cols},
        }
        for field in _ROW_FIELDS:
            if field in frame.columns:
                row[field] = _num(r[field])
        row["tags"] = toxic_tags.account_tags(row)
        row["action"] = next((t for t in row["tags"] if t.startswith("TF_ACTION_")), "")
        rows.append(row)

    scored = len(frame)
    dist = frame["toxic_score"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "rows": rows,
        "mode": mode,
        "generated_at": str(datetime.utcnow()),
        "accounts_scored": int(scored),
        "accounts_listed": int(len(rows)),
        "passive_not_listed": int(len(passive) - min(room, len(passive))),
        "n_trades": int(frame["trades"].sum()),
        "n_toxic_trades": int(frame["toxic_trades"].sum()),
        "adverse_usd_total": _num(frame["adverse_usd"].clip(lower=0).sum(), 2),
        "score_distribution": {k: _num(v, 1) for k, v in dist.items()
                               if k in ("25%", "50%", "75%", "90%", "99%", "max")},
        "tiers": {k: int(v) for k, v in frame["risk_tier"].value_counts().items()},
        "states": {k: int(v) for k, v in frame["state"].value_counts().items()},
        "curve_profiles": {k: int(v) for k, v in frame["curve_profile"].value_counts().items()},
        "signatures": {k: int(v) for k, v in frame["markout_signature"].value_counts().items()},
        "spec": summary,
        "rules_key": _rules_key(rules),
        "horizons_from": "latency_rules.json (shared s2 framework)",
        "window_days": latency_rules.get("window_days"),
    }


def _write_audit(result: dict, rules: dict) -> None:
    """Log a row whenever an account's state or tag set changes, so the
    automation layer has a change feed rather than a snapshot (s8)."""
    try:
        previous: dict = {}
        for entry in reversed(audit_tail(5000)):
            if entry.get("status") == "auto":
                previous[str(entry.get("account"))] = entry
        mode = result.get("mode", "shadow")
        seen = set()
        for row in result["rows"]:
            account = row["account"]
            seen.add(account)
            was = previous.get(account)
            tags = row.get("tags") or []
            if was and was.get("state") == row["state"]:
                try:
                    if json.loads(was.get("tags") or "[]") == tags:
                        continue
                except Exception:
                    pass
            _audit(account, row["state"], row["score"] or 0, row["confidence"] or 0,
                   row.get("action") or "", mode, "system", "auto", "", tags)
        # Accounts that dropped out: log a clear so treatment is removed. Every
        # non-passive account is listed, so an absent account is passive or
        # unscored; one last logged as passive already carries no treatment.
        for account, entry in previous.items():
            if account not in seen and entry.get("state") not in (
                    "", "cleared", "passive_monitoring"):
                _audit(account, "cleared", 0, 0, "TF_ACTION_NONE", mode,
                       "system", "auto", "no longer scored", ["TF_ACTION_NONE"])
    except Exception:
        pass


# --------------------------------------------------------------------------
# reading the scan
# --------------------------------------------------------------------------
def scan(refresh: bool = False) -> dict:
    """The current Engine B result.

    Never computes: the scan is produced by `build()` inside the latency pass
    (s12). `refresh=True` asks the shared Markout Engine for a new pass, which
    rebuilds both engines; this returns the current result with `building` set
    so the tab can poll.
    """
    rules = load_rules()
    key = _rules_key(rules)
    with _LOCK:
        cached = _CACHE.get("scan")
    if cached is None:
        try:
            cached = json.loads(SCAN_CACHE.read_text(encoding="utf-8"))
            with _LOCK:
                _CACHE["scan"] = cached
        except Exception:
            cached = None

    if refresh:
        started = _request_rescan()
        out = dict(cached or _empty())
        out["building"] = True
        out["note"] = ("Rescanning the shared markout engine -- Engine A and B "
                       "rebuild together" if started else
                       "A markout scan is already running; this refreshes when it finishes")
        return out

    if cached is None:
        out = _empty()
        out["note"] = ("No Engine B scan yet. It is produced by the shared markout "
                       "pass -- press Rescan, or wait for the next automatic one.")
        return out
    out = dict(cached)
    if out.get("rules_key") and out["rules_key"] != key:
        out["rules_changed"] = True
        out["note"] = ("Rules changed since this scan. Thresholds that define a toxic "
                       "trade need a new markout pass before they take effect.")
    out.pop("rules_key", None)
    try:
        from webapp import latency_arb
        out["autoscan"] = latency_arb.autoscan_state()
    except Exception:
        pass
    return out


def _empty() -> dict:
    return {"rows": [], "mode": "shadow", "generated_at": None, "accounts_scored": 0,
            "accounts_listed": 0, "n_trades": 0, "n_toxic_trades": 0,
            "adverse_usd_total": 0, "score_distribution": {}, "tiers": {},
            "states": {}, "curve_profiles": {}, "signatures": {}, "spec": {}}


def _request_rescan() -> bool:
    """Ask Engine A's scan to run; it rebuilds Engine B on the way out."""
    try:
        from webapp import latency_arb
        if latency_arb._SCAN_BUILD.locked():
            return False
        threading.Thread(target=lambda: latency_arb.scan(refresh=True),
                         daemon=True).start()
        return True
    except Exception:
        return False


def latest_full_scan() -> dict:
    """The unfiltered result for automation. Prefers memory, then disk."""
    with _LOCK:
        cached = _CACHE.get("scan")
    if cached and cached.get("generated_at") and not cached.get("error"):
        return cached
    try:
        return json.loads(SCAN_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return _empty()


def account_toxic_current(account: str) -> dict | None:
    """This account's current Engine B row, for the account page."""
    result = latest_full_scan()
    for row in result.get("rows") or []:
        if str(row.get("account")) == str(account):
            return row
    return None


def tags_feed(catalog_only: bool = False) -> dict:
    """The automation feed: the catalogue, and every account's current tags."""
    out = {"catalog": toxic_tags.catalog(),
           "categories": ["STATE", "TIER", "CAP", "ACTION", "PROFILE", "CURVE",
                          "SIGNATURE", "EVIDENCE", "DATA", "REVIEW"]}
    if catalog_only:
        return out
    result = latest_full_scan()
    out["generated_at"] = result.get("generated_at")
    out["mode"] = result.get("mode")
    out["accounts"] = [{"account": r["account"], "state": r["state"],
                        "score": r["score"], "confidence": r["confidence"],
                        "action": r.get("action"), "tags": r.get("tags") or []}
                       for r in (result.get("rows") or [])]
    return out


_ORDERS_CACHE: dict = {}


def _save_toxic_orders(frame: pd.DataFrame | None) -> None:
    """Persist each account's materially adverse orders (s5.2) as
    "<signature>|<horizons>", e.g. "sharp_fast|100ms 500ms 1s": the s5.3 curve
    signature (toxic_spec) and the horizons at which the trade cleared its
    minimum. A separate file, so the tab's scan JSON stays small.

    Horizons alone are not a signature: a trade adverse only at 1 s and 5 s
    may have fully reversed by 60 s, so "late" is not "persistent"
    (#1927708789, 16 Sep 2026)."""
    if frame is None:
        return
    orders = pd.to_numeric(frame["order"], errors="coerce")
    hit_cols = [c for c in frame.columns if c.startswith("_hit_")]
    labels = [c[len("_hit_"):] for c in hit_cols]
    hits = frame[hit_cols].fillna(False).astype(bool).to_numpy()
    named = [" ".join(lab for lab, on in zip(labels, row) if on) for row in hits] if hit_cols \
        else [""] * len(frame)
    sig = frame["_sig"].astype(str).tolist() if "_sig" in frame.columns else ["other"] * len(frame)
    kind = [f"{s}|{n}" for s, n in zip(sig, named)]
    out = pd.DataFrame({
        "account": frame["account_key"].astype(str),
        "order": orders,
        "open_time": pd.to_datetime(frame["open_time"]),
        "kind": kind,
    }).dropna(subset=["order"])
    # EVERY adverse order is kept: no per-account cap (user decision, 16 Sep 2026).
    out["order"] = out["order"].astype("int64").astype(str)
    out = out.sort_values("open_time", ascending=False)
    out.to_parquet(TOXIC_ORDERS_PATH, index=False)
    _ORDERS_CACHE.clear()


def _save_markout_sample(trades: pd.DataFrame, horizons: list, mcol,
                         share: float = 0.15) -> None:
    """A random sample of the book's tick-covered trades with their seven
    markouts, for calibrating the per-trade toxic rule offline (the full
    per-trade frame only exists inside the scan). Seeded by the day, so a
    day's rescans draw the same trades."""
    cols = [mcol(h) for h in horizons]
    t = trades.loc[trades[cols[-1]].notna()]
    seed = int(pd.Timestamp.utcnow().strftime("%Y%m%d"))
    t = t.sample(frac=share, random_state=seed) if len(t) > 20_000 else t
    keep = [c for c in ("account_key", "order", "canonical", "direction", "open_time", "hold_seconds",
                        "net_profit", "volume_lots", "open_price", "flagged", "directional",
                        "early_peak_bps", "ms_fill", "entry_spread_rel", "spread_ratio") if c in t.columns]
    t[keep + cols].to_parquet(MARKOUT_SAMPLE_PATH, index=False)


def account_toxic_orders(account: str) -> dict[str, str]:
    """order id -> adverse kind for one account, from the latest live build."""
    try:
        stamp = TOXIC_ORDERS_PATH.stat().st_mtime
    except OSError:
        return {}
    if _ORDERS_CACHE.get("stamp") != stamp:
        frame = pd.read_parquet(TOXIC_ORDERS_PATH, columns=["account", "order", "kind"])
        _ORDERS_CACHE.clear()
        _ORDERS_CACHE["stamp"] = stamp
        _ORDERS_CACHE["by_account"] = {
            a: dict(zip(g["order"], g["kind"])) for a, g in frame.groupby("account", sort=False)}
    return dict(_ORDERS_CACHE["by_account"].get(str(account), {}))


_FEATURES_CACHE: dict = {}


def account_toxic_metrics(account: str) -> dict:
    """Engine B's scored metrics for ANY scored account (listed or not), from
    the features file of the latest live build."""
    try:
        stamp = FEATURES_PATH.stat().st_mtime
        if _FEATURES_CACHE.get("stamp") != stamp:
            _FEATURES_CACHE.clear()
            _FEATURES_CACHE["frame"] = pd.read_parquet(FEATURES_PATH)
            _FEATURES_CACHE["stamp"] = stamp
        frame = _FEATURES_CACHE["frame"]
        if str(account) not in frame.index:
            return {}
        row = frame.loc[str(account)]
    except Exception:
        return {}
    keys = ("trades", "toxic_trades", "toxic_trade_rate", "adverse_usd", "avg_adverse_usd",
            "markout_usd", "realized_pnl", "toxic_pnl_concentration", "sharp_deals_24h",
            "toxic_score", "toxic_confidence")
    out = {k: _num(row.get(k)) for k in keys if k in row.index}
    for k in ("state", "state_label"):
        if k in row.index:
            out[k] = str(row[k])
    out["generated_at"] = str(datetime.utcfromtimestamp(_FEATURES_CACHE["stamp"]))[:16]
    return out


def account_top_symbols(account: str, limit: int = 5) -> list[dict]:
    """Per-symbol toxicity for one account, read from the persisted features."""
    try:
        frame = pd.read_parquet(FEATURES_PATH)
        row = frame.loc[str(account)]
    except Exception:
        return []
    return [{"symbol": str(row.get("top_symbol") or ""),
             "share": _num(row.get("top_symbol_share"), 3)}][:limit]
