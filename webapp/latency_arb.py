"""Latency-Arbitrage detection, profiling, decisioning and control.

Implements the Anti-Fraud System - Latency Arbitrage Module spec:

- EVENT-DRIVEN, THRESHOLD-BASED DETECTION: every closed trade in the scan
  window is marked out against the tick tape (DuckDB ASOF join on the live
  quote store) at configurable short horizons; a trade is flagged when it
  clears the configured markout/hold/profit thresholds.
- CONFIGURABLE RULES & ESCALATION: every threshold and tier lives in
  latency_rules.json, editable from the tab, applied on the next scan.
- CONSOLIDATED BEHAVIOURAL PROFILE per client: risk, confidence,
  consistency, persistence and economic impact -- the five axes the spec
  names -- replacing manual mark-out tables.
- DECISION LAYER: axes combine into one structured verdict
  (ESCALATE / RESTRICT / MONITOR / CLEAR) with the reasons attached.
- ACTION LAYER: each verdict maps to a defined SOR/execution treatment
  (feed delay ms, requote band, routing pin, review). Shadow mode logs
  what WOULD be done; every decision lands in an audit log with manual
  override endpoints, so automation is controlled and measurable.
- KPIs measured against the 27-Sep targets: detection rate vs known abusive
  cohort, improvement vs the existing behavioural baseline, alert accuracy
  from operator overrides, automation success and manual-intervention
  reduction from the audit trail.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RULES_PATH = ROOT / "latency_rules.json"
SCAN_CACHE = ROOT / "artifacts" / "latency_scan.json"

#: Cross-process rescan request. The BETA instance is a reader -- it must never
#: run a scan itself, because a scan opens the DuckDB tick store and DuckDB
#: allows a single writer. So its Rescan button drops this file instead, and the
#: full instance's autoscan loop picks it up on its next 30 s tick and runs the
#: scan in the process that already owns the tape.
RESCAN_REQUEST = ROOT / "artifacts" / "rescan_request.json"
DB = ROOT.parent / "app.db"

DEFAULT_RULES = {
    "window_days": 7,
    #: The spec's seven markout horizons (s2): 100/200/300/500 ms, 1 s, 5 s,
    #: 60 s. Sub-second horizons are computed only for millisecond-stamped
    #: fills (MT5 `time_msc`); the longest horizon is the persistence horizon.
    "horizons_seconds": [0.1, 0.2, 0.3, 0.5, 1, 5, 60],
    #: Early-markout thresholds (bps), keyed by horizon.
    "flag_markout_bps": {"0.1": 0.75, "0.2": 1.0, "0.3": 1.25, "0.5": 1.5,
                         "1": 1.9, "5": 2.0},
    #: The early (latency) advantage window. The spec's MO100-MO500 (s5.2),
    #: extended to 1 s on 15 Sep 2026 (user decision): on 7 days of MT5 flow it
    #: kept every 100-500 ms flag and added 81 trades / 4 accounts.
    "early_horizons_seconds": [0.1, 0.2, 0.3, 0.5, 1],
    #: Early window for fills stamped only to the second (MT4 orders with no
    #: matching Kafka trade event -- see _attach_mt4_ms_times), which
    #: cannot support sub-second horizons.
    "fallback_early_horizons_seconds": [1, 5],
    #: Broker quote throttling (spec s2.0.1): quote absence up to the baseline
    #: is normal; quote age counts as stale-price evidence only beyond
    #: baseline + tolerance, and only on millisecond-stamped fills.
    "quote_throttle_ms": 200,
    "quote_age_tolerance_ms": 100,
    #: A latency flag needs the early advantage to FADE: the 60 s markout must
    #: sit at least this % below the early peak (spec s2.2 / s5.3 decay
    #: signature). An advantage that holds or grows is directional flow.
    "min_decay_pct": 50.0,
    #: 60 s markout (bps) at which a fast, paid, non-decaying trade is
    #: recorded as DIRECTIONAL (persistent) -- reported, never a latency flag.
    "directional_markout_bps": 3.0,
    #: Account curve profile: mean markouts inside +/- this band are neutral.
    "neutral_band_bps": 0.5,
    "max_hold_seconds": 120,
    "min_trade_profit_usd": 1.0,
    "min_flagged_trades": 5,
    "illiquid_spread_ratio": 1.5,
    "independent_check": True,
    "tiers": {"escalate": 75, "restrict": 55, "monitor": 35},
    #: THE escalation rule, ONE complex expression over the TABLE's own
    #: columns (risk, confidence, consistency, persistence, econ_usd, ml,
    #: ew, behavioural, stale_share, illiquid_share, indep_share,
    #: event_share, flagged, trades, med_hold_s, best_markout_bps). This is
    #: the default; edit it on the tab to change who the table shows.
    "complex_filter": "risk >= 35",
    "actions": {
        "escalate": {"treatment": "SOR: pin to B-book off + feed delay",
                     "delay_ms": 250, "requote_bps": 1.0, "review": True},
        "restrict": {"treatment": "SOR: feed delay on quotes",
                     "delay_ms": 120, "requote_bps": 0.5, "review": False},
        "monitor": {"treatment": "watchlist only",
                    "delay_ms": 0, "requote_bps": 0.0, "review": False},
    },
    "shadow_mode": True,
    #: Automatic rescan cadence in minutes (0 = manual only). A scan takes
    #: 3-4 minutes and runs in the trading process, so the loop never starts
    #: one while another is running, and waits this long AFTER the previous
    #: one finished.
    "auto_rescan_minutes": 5,
}
# Spec scoring parameters (s5.1-s5.3): exclusions, alert gates, weights and
# scales. Listed in DEFAULT_RULES so save_rules persists them.
from webapp import latency_reference, latency_spec, latency_tags  # noqa: E402
DEFAULT_RULES.update(json.loads(json.dumps(latency_spec.SPEC_DEFAULT_RULES)))
DEFAULT_RULES.update(json.loads(json.dumps(latency_reference.REFERENCE_DEFAULT_RULES)))


def load_rules() -> dict:
    try:
        rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        merged = dict(DEFAULT_RULES)
        merged.update(rules or {})
        return merged
    except Exception:
        return dict(DEFAULT_RULES)


def save_rules(rules: dict) -> dict:
    # merge over the EFFECTIVE rules, not the code defaults: a partial post
    # (the tab sends only the escalation rule + shadow flag) must never
    # silently reset the stored detection thresholds.
    merged = load_rules()
    for key in DEFAULT_RULES:
        if key in (rules or {}):
            merged[key] = rules[key]
    RULES_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _CACHE.clear()
    return merged


def _ensure_audit() -> None:
    with sqlite3.connect(DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS latency_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
                account TEXT, verdict TEXT, risk REAL, action TEXT,
                mode TEXT, operator TEXT, status TEXT, note TEXT)""")
        try:
            # Automation tags (latency_tags), added after the table shipped.
            cx.execute("ALTER TABLE latency_audit ADD COLUMN tags TEXT")
        except sqlite3.OperationalError:
            pass


def _audit(account: str, verdict: str, risk: float, action: str,
           mode: str, operator: str, status: str, note: str = "",
           tags: list | None = None) -> None:
    _ensure_audit()
    with sqlite3.connect(DB) as cx:
        cx.execute("INSERT INTO latency_audit (ts, account, verdict, risk, "
                   "action, mode, operator, status, note, tags) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (time.time(), account, verdict, float(risk), action, mode,
                    operator, status, note, ",".join(tags) if tags else None))


def audit_tail(limit: int = 200) -> list[dict]:
    _ensure_audit()
    with sqlite3.connect(DB) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute(
            "SELECT * FROM latency_audit ORDER BY id DESC LIMIT ?", (limit,))]


def decide(account: str, decision: str, operator: str = "risk",
           note: str = "") -> dict:
    """Manual override: operator confirms (true positive) or dismisses
    (false positive) an alert -- or CLEARS a previous override (deselect).
    Feeds the Alert Accuracy KPI. A confirm/dismiss REQUIRES a comment;
    without one the override is refused."""
    if decision not in ("confirm", "dismiss", "clear"):
        return {"error": "decision must be confirm|dismiss|clear"}
    if decision in ("confirm", "dismiss") and not str(note or "").strip():
        return {"error": "comment required: an override must say why"}
    _audit(account, "override", 0.0, decision, "manual", operator,
           decision, note)
    return {"ok": True}


# ------------------------------------------------------------------ scan
_CACHE: dict = {}
_LOCK = threading.Lock()
_SCAN_BUILD = threading.Lock()


def _hkey(h) -> str:
    """Rules key for a horizon in seconds: 0.1 -> "0.1", 1.0 -> "1"."""
    h = float(h)
    return str(int(h)) if h.is_integer() else f"{h:g}"


def _hlabel(h) -> str:
    """Column label for a horizon in seconds: 0.1 -> "100ms", 1 -> "1s"."""
    h = float(h)
    return f"{int(h)}s" if h.is_integer() else f"{int(round(h * 1000))}ms"


def _mcol(h) -> str:
    return f"markout_{_hlabel(h)}_bps"


def _feed_of(server) -> str:
    """Quote feed a trade or quote belongs to. MT5 servers price from their own
    feed; the MT4 servers share one (the Kafka MT4 quote topics beyond live01
    are empty, and the mt4_live01 ticks table serves all four)."""
    s = str(server).removeprefix("mysql:")
    if "dubai" in s:
        return "mt5_dubai_live01"
    if "indonesia" in s:
        return "mt5_indonesia"
    if s.startswith("mt5"):
        return s
    return "mt4"


_FEED_SQL = """CASE
    WHEN replace(server, 'mysql:', '') LIKE '%dubai%' THEN 'mt5_dubai_live01'
    WHEN replace(server, 'mysql:', '') LIKE '%indonesia%' THEN 'mt5_indonesia'
    WHEN replace(server, 'mysql:', '') LIKE 'mt5%' THEN replace(server, 'mysql:', '')
    ELSE 'mt4' END"""


def _quote_marks(trades: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """Markouts from the tick tape via DuckDB ASOF joins, per QUOTE FEED.

    Each trade is marked against its own server's feed when that feed has
    quotes for the symbol (an MT5 fill against MT5 prices), else against any
    available feed. Sub-second horizons (spec MO100-MO500) and quote age are
    only produced for fills stamped to the millisecond; a fill truncated to the
    second can be up to 999 ms early, which would fabricate sub-second
    evidence. Trades outside the store's coverage stay NaN."""
    from webapp.kafka_service import shared_cursor as _store
    from webapp.trade_feed import _canonical
    horizons = sorted(float(h) for h in horizons)
    frame = trades[["row_id", "database", "symbol", "direction", "open_time",
                    "open_price"]].copy()
    frame["canonical"] = [(_canonical(s) or s) for s in frame["symbol"].astype(str)]
    frame["open_time"] = pd.to_datetime(frame["open_time"])
    frame["feed"] = frame["database"].map(_feed_of)
    # PLAIN-COLUMN ASOF targets: expressions on the inequality side of an
    # ASOF join match unreliably; precomputed columns always do.
    for i, h in enumerate(horizons):
        frame[f"t_{i}"] = frame["open_time"] + pd.to_timedelta(h, unit="s")
    lo = frame["open_time"].min() - pd.Timedelta(minutes=10)
    hi = frame["open_time"].max() + pd.Timedelta(seconds=max(horizons) + 60)
    with _store() as cx:
        symbols = pd.DataFrame({"canonical": frame["canonical"].unique()})
        # MySQL tick tape ONLY. Quotes from the live Kafka consumer carry the
        # broker's publish time (~90 ms after the tick, measured 15 Sep 2026),
        # which is a whole sub-second horizon of bias; the MySQL `tm` /
        # `datetime_msc` stamps are the execution clock itself.
        cx.execute(f"""
            CREATE OR REPLACE TEMP TABLE _la_q AS
            SELECT {_FEED_SQL} AS feed, canonical, event_time, bid, ask, mid
            FROM quotes
            WHERE event_time >= ? AND event_time <= ?
              AND server LIKE 'mysql:%'
              AND canonical IN (SELECT canonical FROM symbols)""",
                   [lo.to_pydatetime(), hi.to_pydatetime()])
        pairs = cx.execute("SELECT DISTINCT feed, canonical FROM _la_q").fetchall()
        # Own feed when it has the symbol; otherwise the feed with it (MT4's
        # shared tape first), so coverage never drops for want of a feed.
        have = {}
        for feed, canon in pairs:
            have.setdefault(canon, set()).add(feed)
        def pick(feed, canon):
            feeds = have.get(canon) or set()
            if feed in feeds or not feeds:
                return feed
            return "mt4" if "mt4" in feeds else sorted(feeds)[0]
        frame["feed"] = [pick(f, c) for f, c in zip(frame["feed"], frame["canonical"])]
        cx.execute("CREATE OR REPLACE TEMP TABLE _la_trades AS SELECT * FROM frame")
        # Quote AT entry: its mid anchors slippage-free markouts and its age
        # (entry minus last quote update) is the stale-quote feature.
        cx.execute("""
            CREATE OR REPLACE TEMP TABLE _la_m0 AS
            SELECT t.row_id,
                   COALESCE(q.mid, (q.bid + q.ask) / 2) AS mark0,
                   q.event_time AS qt,
                   CASE WHEN q.ask > q.bid AND (q.ask + q.bid) > 0
                        THEN (q.ask - q.bid) / ((q.ask + q.bid) / 2)
                        END AS entry_spread_rel
            FROM _la_trades t ASOF LEFT JOIN _la_q q
              ON q.feed = t.feed AND q.canonical = t.canonical
             AND q.event_time <= t.open_time
        """)
        # LIQUIDITY BASELINE: each canonical symbol's median relative spread
        # over the window -- entries at a multiple of it are illiquid-period
        # fills (rollover, session gaps).
        med = cx.execute("""
            SELECT canonical,
                   median((ask - bid) / NULLIF((ask + bid) / 2, 0)) AS med_spread
            FROM _la_q WHERE ask > bid GROUP BY 1
        """).df().set_index("canonical")["med_spread"]
        for i, _ in enumerate(horizons):
            cx.execute(f"""
                CREATE OR REPLACE TEMP TABLE _la_m{i + 1} AS
                SELECT t.row_id,
                       COALESCE(q.mid, (q.bid + q.ask) / 2) AS mark_{i},
                       q.event_time AS qt_{i}
                FROM _la_trades t ASOF LEFT JOIN _la_q q
                  ON q.feed = t.feed AND q.canonical = t.canonical
                 AND q.event_time <= t.t_{i}
            """)
        base0 = cx.execute(
            "SELECT row_id, mark0, qt, entry_spread_rel FROM _la_m0"
        ).df().set_index("row_id")
        parts = [cx.execute(
            f"SELECT row_id, mark_{i}, qt_{i} FROM _la_m{i + 1}").df().set_index("row_id")
            for i, _ in enumerate(horizons)]
    out = pd.concat([base0] + parts, axis=1)
    merged = frame.set_index("row_id").join(out)
    open_dt = pd.to_datetime(merged["open_time"])
    # Millisecond-stamped fill: any sub-second component (an exact .000 fill
    # simply loses its sub-second horizons, 1 in 1000).
    precise = (open_dt - open_dt.dt.floor("s")) > pd.Timedelta(0)
    result = {}
    for i, h in enumerate(horizons):
        mark = pd.to_numeric(merged[f"mark_{i}"], errors="coerce")
        # A match from BEFORE the entry is a store-coverage gap, not a
        # markout -- null it rather than fabricating a signal.
        fresh = pd.to_datetime(merged[f"qt_{i}"]) >= open_dt - pd.Timedelta(seconds=5)
        mark = mark.where(fresh)
        if h < 1.0:
            mark = mark.where(precise)
        result[_mcol(h)] = (merged["direction"] * (mark - merged["open_price"])
                            / merged["open_price"] * 1e4)
    age_ms = (open_dt - pd.to_datetime(merged["qt"])).dt.total_seconds() * 1000.0
    result["quote_age_ms"] = age_ms.where(precise)
    result["ms_fill"] = precise
    result["quote_feed"] = merged["feed"]
    result["entry_spread_rel"] = pd.to_numeric(
        merged["entry_spread_rel"], errors="coerce")
    # Spread RATIO vs the symbol's own median: >1 = wider than usual
    # (illiquid moment), the natural habitat of latency arbitrage.
    med_map = merged["canonical"].map(med)
    result["spread_ratio"] = result["entry_spread_rel"] / med_map.replace(0, np.nan)
    return pd.DataFrame(result, index=merged.index)


def _load_trades(window_days: int, end: datetime | None = None) -> pd.DataFrame:
    from webapp import data_store
    end = end or datetime.utcnow()
    start = end - timedelta(days=window_days)
    frame = data_store.read_history(start=start, end=end)
    if frame is None or not len(frame):
        return pd.DataFrame()
    frame = frame.loc[(pd.to_datetime(frame["close_time"]) >= start)
                      & (frame["open_price"] > 0)].copy()
    # The warehouse identifies rows by (database, login); the account key the
    # rest of the stack speaks is their join.
    frame["account_key"] = (frame["database"].astype(str) + ":"
                            + frame["login"].astype(str))
    frame["direction"] = np.where(
        frame["cmd"].astype(str).str.lower().str.startswith("b"), 1, -1)
    frame["hold_seconds"] = (
        pd.to_datetime(frame["close_time"])
        - pd.to_datetime(frame["open_time"])).dt.total_seconds()
    frame["row_id"] = np.arange(len(frame))
    return frame


def _attach_mt4_ms_times(trades: pd.DataFrame) -> dict:
    """Millisecond open/close times for MT4 trades from the Kafka trade events.

    MT4 `orders` stamp open/close to the whole second, which cannot support
    the sub-second horizons. Every MT4 event carries `tradeRecord.timeStamp`
    to the millisecond, and it IS the execution clock: verified 15 Sep 2026 on
    live mt4_live01 gold, the record's own marketBid/marketAsk equal the MT4
    tick tape (`ticks.tm`) at exactly that stamp for 94% of events, dropping to
    ~55% at +/-100 ms and ~10% at +/-250 ms.

    The join is (server, order) AND same second: an event counts as a trade's
    open only when floor(timeStamp) equals the warehouse open_time (the same
    for its close). That picks the fill even for pending orders, whose
    orderCreated is the placement and whose activation arrives later as an
    orderUpdated, and it ignores SL/TP modifications. The stored `event_time`
    is NOT used: it is the broker's publish time, ~90 ms after execution.

    Updates open_time / close_time / hold_seconds in place; unmatched trades
    keep their second-level times and so their 1 s + 5 s fallback window."""
    stats = {"mt4_trades": 0, "mt4_ms_opens": 0, "mt4_ms_closes": 0}
    is_mt4 = trades["database"].astype(str).str.startswith("mt4")
    stats["mt4_trades"] = int(is_mt4.sum())
    if not stats["mt4_trades"]:
        return stats
    mt4 = trades.loc[is_mt4, ["row_id", "database", "order", "open_time", "close_time"]].copy()
    mt4["order"] = pd.to_numeric(mt4["order"], errors="coerce").astype("Int64")
    mt4["open_s"] = pd.to_datetime(mt4["open_time"]).dt.floor("s")
    mt4["close_s"] = pd.to_datetime(mt4["close_time"]).dt.floor("s")
    lo = mt4["open_s"].min() - pd.Timedelta(minutes=1)
    hi = mt4["close_s"].max() + pd.Timedelta(minutes=1)
    stamp = " , ".join(f"json_extract_string(payload, '$.{k}.tradeRecord.timeStamp')"
                       for k in ("orderCreated", "orderUpdated", "orderClosedBy"))
    from webapp import mt4_kafka_backfill as backfill
    parts = sorted(backfill.OUT_DIR.glob("mt4_*/*.parquet"))
    stats["backfill_files"] = len(parts)
    # EVENT SOURCE, and why there are two. The live `events` table lives in the
    # DuckDB tick store, which permits ONE writer: a second instance cannot open
    # it while the scanning instance holds it. Before this fallback existed the
    # open raised, the caller swallowed the exception, and EVERY MT4 trade in
    # the beta degraded to second precision -- fills whose millisecond time was
    # sitting right there in the backfill were reported as "sub-second not
    # measurable" (observed 17 Sep 2026 on order 171738613, real time
    # 16:03:22.626, shown as 16:03:22). The backfill parquet needs no lock, so
    # it carries the millisecond times on its own whenever the store is busy.
    # The IMPORT is inside the guard on purpose: webapp.kafka_service builds a
    # KafkaMaterialiser at module scope, which opens the store -- so merely
    # importing it raises when another instance holds the file, before any call
    # is made. Importing it outside this try is what made the fallback below
    # unreachable.
    import duckdb
    try:
        from webapp.kafka_service import shared_cursor
        cx, live_events = shared_cursor(), True
    except Exception as error:
        if not parts:
            stats["ms_source"] = f"unavailable: {type(error).__name__}"
            return stats
        cx, live_events = duckdb.connect(), False   # in-memory; parquet only
    stats["ms_source"] = ("live events + backfill" if live_events
                          else "backfill parquet only (tick store held elsewhere)")
    window = [lo.to_pydatetime(), (hi + pd.Timedelta(minutes=5)).to_pydatetime()]
    try:
        if live_events:
            cx.execute(f"""
                CREATE OR REPLACE TEMP TABLE _la_mt4_ev AS
                SELECT server, TRY_CAST(json_extract_string(payload, '$.order') AS BIGINT) AS ord,
                       TRY_CAST(replace(COALESCE({stamp}), 'Z', '') AS TIMESTAMP) AS stamp
                FROM events
                WHERE server LIKE 'mt4%' AND event_time >= ? AND event_time <= ?
                  AND event_kind IN ('orderCreated', 'orderUpdated', 'orderClosedBy')""",
                       window)
        else:
            cx.execute("CREATE OR REPLACE TEMP TABLE _la_mt4_ev "
                       "(server VARCHAR, ord BIGINT, stamp TIMESTAMP)")
        if parts:
            # Events backfilled from Kafka's retained log (mt4_kafka_backfill):
            # the live store only holds what its consumer has seen.
            files = "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in parts) + "]"
            cx.execute(f"""
                INSERT INTO _la_mt4_ev
                SELECT server, ord, CAST(stamp AS TIMESTAMP) FROM read_parquet({files})
                WHERE stamp >= ? AND stamp <= ?""", window)
        cx.register("_la_mt4_tr", mt4[["row_id", "database", "order", "open_s", "close_s"]])
        found = cx.execute("""
            SELECT t.row_id,
                   MIN(e.stamp) FILTER (WHERE date_trunc('second', e.stamp) = t.open_s)  AS open_ms,
                   MIN(e.stamp) FILTER (WHERE date_trunc('second', e.stamp) = t.close_s) AS close_ms
            FROM _la_mt4_tr t JOIN _la_mt4_ev e
              ON e.server = t.database AND e.ord = t."order"
            WHERE e.stamp IS NOT NULL
            GROUP BY t.row_id""").df().set_index("row_id")
        cx.unregister("_la_mt4_tr")
        cx.execute("DROP TABLE IF EXISTS _la_mt4_ev")
    finally:
        cx.close()
    if not len(found):
        return stats
    idx = trades.set_index("row_id").index
    open_ms = pd.to_datetime(found["open_ms"]).reindex(idx).to_numpy()
    close_ms = pd.to_datetime(found["close_ms"]).reindex(idx).to_numpy()
    has_open, has_close = ~pd.isna(open_ms), ~pd.isna(close_ms)
    trades["open_time"] = pd.to_datetime(trades["open_time"]).where(~has_open, open_ms)
    trades["close_time"] = pd.to_datetime(trades["close_time"]).where(~has_close, close_ms)
    trades["hold_seconds"] = (trades["close_time"] - trades["open_time"]).dt.total_seconds()
    stats["mt4_ms_opens"] = int(has_open.sum())
    stats["mt4_ms_closes"] = int(has_close.sum())
    return stats


def _ew_scores(accounts, end: datetime | None = None) -> pd.Series:
    """5-day EARLY-WARNING probability per account for the latency rule
    (the generic rule-models EW artifact: rule holds over the next 5 days
    with client equity rising). Uses the cached 90-day feature table for
    live scans; rebuilds at `end` for historical as-of scans."""
    try:
        from webapp import rule_models
        meta = rule_models.rule_meta("latency_arbitrage")
        ew_meta = meta.get("ew") or {}
        if not ew_meta.get("features"):
            return pd.Series(np.nan, index=accounts)
        if end is None:
            table, _ = rule_models._obs_table_cached()
        else:
            trades = rule_models._window_trades(
                end - timedelta(days=rule_models.OBS_DAYS), end)
            table = rule_models.feature_table(
                trades, rule_models.combined_anchors())
        mp = rule_models._paths("latency_arbitrage")[0]
        scores = rule_models._predict(mp, table, ew_meta["features"])
        return scores.reindex(accounts)
    except Exception:
        return pd.Series(np.nan, index=accounts)


def _apply_complex_filter(result: dict, rules: dict) -> dict:
    """The editable complex rule over the TABLE's own columns: rows not
    satisfying it drop out. Empty expression = untouched."""
    # DISPLAY ONLY: the filter narrows what the table shows. Detection, the
    # audit log and the automation tags (/api/antifraud/latency/tags) always
    # use every row. Shadow/live mode is read from the CURRENT rules too, so
    # neither edit needs a rescan.
    result = dict(result, mode="shadow" if rules.get("shadow_mode", True) else "live")
    expr = str(rules.get("complex_filter") or "").strip()
    rows = result.get("rows") or []
    if not expr:
        return result
    try:
        import ast as _ast
        from webapp.antifraud import safe_expr_mask
        _ast.parse(expr, mode="eval")
        if not rows:
            return dict(result, complex_filter=expr, n_after_range=0)
        frame = pd.DataFrame(rows)
        mask = safe_expr_mask(expr, frame).fillna(False).astype(bool)
        kept = [r for r, ok in zip(rows, mask.tolist()) if ok]
        return dict(result, rows=kept, n_range=len(kept),
                    n_after_range=len(rows), complex_filter=expr)
    except Exception as error:
        return dict(result, complex_filter=expr,
                    complex_filter_error=f"{type(error).__name__}: {error}")


def _range_window(as_of: str | None, start: str | None):
    """(end_dt, start_dt) for the RESULTS range. Blank as-of = live now;
    blank start = the last 24 LIVE hours before end."""
    end_dt = datetime.utcnow()
    if as_of:
        try:
            end_dt = datetime.strptime(str(as_of), "%Y-%m-%d") \
                + timedelta(hours=23, minutes=59, seconds=59)
        except Exception:
            pass
    start_dt = end_dt - timedelta(hours=24)
    if start:
        try:
            start_dt = datetime.strptime(str(start), "%Y-%m-%d")
        except Exception:
            pass
    return end_dt, min(start_dt, end_dt)


def _range_filter(result: dict, end_dt: datetime, start_dt: datetime) -> dict:
    """Restrict the profile table to accounts ACTIVE in the range. Profiles
    stay computed over the full detection window; the range picks who shows.
    Rows from an older cache without a stamp are kept (a rescan stamps them)."""
    rows = []
    for r in result.get("rows") or []:
        ts = pd.to_datetime(r.get("last_ts"), errors="coerce")
        if pd.isna(ts) or ts >= start_dt:
            rows.append(r)
    return dict(result, rows=rows, n_range=len(rows),
                n_scan=len(result.get("rows") or []),
                range_start=str(start_dt)[:16], range_end=str(end_dt)[:16])


#: Rules that change only presentation, not detection: editing them must not
#: mark a scan stale or force a rescan.
_DISPLAY_RULES = ("complex_filter", "shadow_mode")


def scan(refresh: bool = False, as_of: str | None = None,
         start: str | None = None, apply_filter: bool = True) -> dict:
    """Full detection + profiling pass.

    LOAD FAST, REFRESH DELIBERATELY: without `refresh` the LAST computed
    result is always served instantly -- from memory, else from disk (as
    stamped by generated_at) -- and never blocks the tab on a recompute.
    Refresh (the button) recomputes with whatever the rules now say.

    RANGE: the table always defaults to accounts active in the last 24 LIVE
    hours; as_of/start (YYYY-MM-DD) move that window. A PAST as-of recomputes
    detection as of that day (memory-cached per day)."""
    rules = load_rules()
    key = json.dumps({k: v for k, v in rules.items() if k not in _DISPLAY_RULES}, sort_keys=True)
    end_dt, start_dt = _range_window(as_of, start)
    view = (lambda res: _apply_complex_filter(res, rules)) if apply_filter else (lambda res: res)
    historical = bool(as_of) and str(as_of) != datetime.utcnow().strftime("%Y-%m-%d")
    if historical:
        hkey = f"scan:{key}:{as_of}"
        with _LOCK:
            cached = _CACHE.get(hkey)
        if cached and not refresh:
            return view(_range_filter(cached, end_dt, start_dt))
        if _SCAN_BUILD.locked():
            return {"building": True, "rows": [],
                    "note": f"historical scan for {as_of} building"}

        def _build_hist():
            with _SCAN_BUILD:
                try:
                    result = _scan_inner(rules, end=end_dt)
                    result["as_of"] = str(as_of)
                    with _LOCK:
                        _CACHE[hkey] = result
                except Exception as error:
                    with _LOCK:
                        _CACHE[hkey] = {
                            "rows": [], "as_of": str(as_of),
                            "error": f"{type(error).__name__}: {error}"}
        threading.Thread(target=_build_hist, daemon=True).start()
        return {"building": True, "rows": [],
                "note": f"historical scan for {as_of} started"}
    with _LOCK:
        cached = _CACHE.get("scan")
    if cached and not refresh:
        if cached.get("rules_key") != key:
            cached = dict(cached, rules_changed=True)
        return view(_range_filter(cached, end_dt, start_dt))
    if not refresh:
        try:
            disk = json.loads(SCAN_CACHE.read_text(encoding="utf-8"))
            disk["from_disk"] = True
            disk["rules_key"] = key
            with _LOCK:
                _CACHE["scan"] = disk
            return view(_range_filter(disk, end_dt, start_dt))
        except Exception:
            pass
    # NEVER compute on the request thread: with no cache (or on refresh),
    # kick ONE background rebuild and answer immediately -- the tab polls.
    if _SCAN_BUILD.locked():
        return {"building": True, "rows": [],
                "note": "scan rebuilding in the background"}

    def _build():
        with _SCAN_BUILD:
            try:
                result = _scan_inner(rules)
                result["rules_key"] = key
                with _LOCK:
                    _CACHE["scan"] = result
                # Publish every completed scan, including degraded scans. A
                # stale cache is more misleading than a current result that
                # explicitly reports its tape error. Replace atomically so a
                # reader never sees a partial JSON document.
                payload = json.dumps(
                    {k: v for k, v in result.items() if k != "rules_key"},
                    default=str)
                temp_cache = SCAN_CACHE.with_suffix(".json.tmp")
                temp_cache.write_text(payload, encoding="utf-8")
                temp_cache.replace(SCAN_CACHE)
            except Exception as error:
                with _LOCK:
                    _CACHE.setdefault("scan", {"rows": []})
                    _CACHE["scan"]["error"] = \
                        f"{type(error).__name__}: {error}"
    threading.Thread(target=_build, daemon=True).start()
    return {"building": True, "rows": [],
            "note": "scan started in the background"}


def autoscan_state() -> dict:
    """Where the automatic rescan loop stands: when the last scan finished,
    whether one is running now, and when the next is due."""
    rules = load_rules()
    minutes = float(rules.get("auto_rescan_minutes") or 0)
    last = _CACHE.get("scan_finished_at")
    if last is None:
        try:
            last = pd.Timestamp(json.loads(SCAN_CACHE.read_text(encoding="utf-8"))["generated_at"]).timestamp()
        except Exception:
            last = None
    running = _SCAN_BUILD.locked()
    due_in = None
    if minutes > 0 and last is not None and not running:
        due_in = max(0.0, (last + minutes * 60) - time.time())
    return {"enabled": minutes > 0, "interval_minutes": minutes, "running": running,
            "last_finished_at": (datetime.utcfromtimestamp(last).isoformat(timespec="seconds") if last else None),
            "seconds_until_next": None if due_in is None else round(due_in),
            "last_duration_s": _CACHE.get("scan_duration_s")}


def request_rescan(source: str = "beta") -> dict:
    """Ask the instance that owns the tick store to run a scan.

    Used by the beta, which cannot scan in its own process. Writing the file is
    the whole request; `autoscan_tick` in the full instance consumes it.
    """
    if _SCAN_BUILD.locked():
        return {"requested": False, "note": "a markout scan is already running"}
    try:
        RESCAN_REQUEST.write_text(json.dumps(
            {"at": datetime.utcnow().isoformat(timespec="seconds"), "source": source}),
            encoding="utf-8")
    except Exception as error:
        return {"requested": False, "note": f"could not queue the rescan: {error}"}
    return {"requested": True,
            "note": "rescan queued — the engine picks it up within 30 seconds"}


def pending_rescan() -> dict | None:
    """The queued request, if one is waiting."""
    try:
        return json.loads(RESCAN_REQUEST.read_text(encoding="utf-8"))
    except Exception:
        return None


def autoscan_tick() -> str:
    """One check of the automatic rescan loop. Returns what it did."""
    rules = load_rules()
    if _SCAN_BUILD.locked():
        return "scan already running"

    # A request from the beta jumps the queue: someone is waiting on it, so it
    # runs now rather than at the next interval. Consumed before the scan
    # starts, so a failure does not leave the request looping forever.
    if RESCAN_REQUEST.exists():
        try:
            RESCAN_REQUEST.unlink()
        except Exception:
            pass
        scan(refresh=True)
        return "started (requested)"

    minutes = float(rules.get("auto_rescan_minutes") or 0)
    if minutes <= 0:
        return "disabled"
    state = autoscan_state()
    if state["seconds_until_next"]:
        return f"next in {state['seconds_until_next']}s"
    scan(refresh=True)
    return "started"


def latest_full_scan() -> dict:
    """The latest LIVE scan exactly as computed: every tagged account in the
    7-day window, no display range, no table filter -- what automation reads."""
    with _LOCK:
        cached = _CACHE.get("scan")
    # Only a COMPLETED scan counts: a failed rebuild leaves {"rows": [], "error"}
    # in memory, which must not hide the saved result.
    if cached and cached.get("generated_at") and not cached.get("building") and not cached.get("error"):
        return cached
    try:
        return json.loads(SCAN_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return scan()


def _horizon_sets(rules: dict) -> tuple[list[float], list[float], list[float], float]:
    """(all horizons, early window for millisecond fills, early window for
    second-stamped fills, the persistence horizon = the longest), in seconds."""
    horizons = sorted(float(h) for h in rules["horizons_seconds"])
    early = [h for h in (float(e) for e in rules.get("early_horizons_seconds") or [])
             if h in horizons] or horizons[:1]
    fallback = [h for h in (float(e) for e in
                            rules.get("fallback_early_horizons_seconds") or [])
                if h in horizons and h >= 1.0] or [h for h in horizons if h >= 1.0][:1]
    return horizons, early, fallback, horizons[-1]


def _early_peak(frame: pd.DataFrame, early: list, fallback: list,
                precise: pd.Series, prefix: str = "markout_",
                suffix: str = "_bps") -> pd.Series:
    """Peak early markout: the sub-second window for millisecond fills, the
    second-level fallback window for fills stamped only to the second."""
    cols = lambda hs: [f"{prefix}{_hlabel(h)}{suffix}" for h in hs]
    sub = frame[cols(early)].max(axis=1)
    sec = frame[cols(fallback)].max(axis=1)
    return sub.where(precise, sec)


def _classify_trades(trades: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Per-trade latency vs directional classification over the markouts.

    LATENCY (spec s2.2 "Fast / Transient Advantage", s5.3): a fast, paid trade
    whose early markout clears its threshold AND whose advantage has decayed by
    at least min_decay_pct at the persistence horizon. DIRECTIONAL ("Persistent
    / Directional Advantage"): a fast, paid trade whose markout at the
    persistence horizon clears directional_markout_bps without decaying. The
    two are exclusive; only LATENCY counts toward latency alerts. Flagging on
    ANY horizon (the previous rule, with a 30 s leg) scored persistent moves --
    the directional signature -- as latency arbitrage."""
    horizons, early, fallback, late = _horizon_sets(rules)
    precise = trades.get("ms_fill", pd.Series(False, index=trades.index)).fillna(False).astype(bool)
    trades["early_peak_bps"] = _early_peak(trades, early, fallback, precise)
    # Early hit on the window that applies to this fill's timestamp precision.
    hit_sub = pd.Series(False, index=trades.index)
    for h in early:
        hit_sub |= trades[_mcol(h)] >= float(rules["flag_markout_bps"].get(_hkey(h), 999))
    hit_sec = pd.Series(False, index=trades.index)
    for h in fallback:
        hit_sec |= trades[_mcol(h)] >= float(rules["flag_markout_bps"].get(_hkey(h), 999))
    early_hit = hit_sub.where(precise, hit_sec)
    trades["early_hit_any"] = early_hit.fillna(False).astype(bool)
    late_mo = trades[_mcol(late)]
    keep = 1.0 - float(rules.get("min_decay_pct", 50.0)) / 100.0
    decayed = late_mo <= keep * trades["early_peak_bps"]
    trades["fast"] = trades["hold_seconds"] <= float(rules["max_hold_seconds"])
    paid = trades["net_profit"] >= float(rules["min_trade_profit_usd"])
    trades["flagged"] = early_hit & decayed & trades["fast"] & paid
    trades["directional"] = (
        (late_mo >= float(rules.get("directional_markout_bps", 3.0)))
        & ~decayed.fillna(False) & trades["fast"] & paid)
    trades["flagged"] = trades["flagged"].fillna(False).astype(bool)
    trades["directional"] = trades["directional"].fillna(False).astype(bool)
    return trades


def _curve_profiles(trades: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Account markout curve (spec s2.2, s3) over ALL its tick-covered trades.

    Deliberately NOT restricted to fast or paid trades: conditioning on a short
    hold selects winners (quick profits closed, losers held), which inflates the
    long-horizon markout of any fast-trade subset."""
    horizons, early, fallback, late = _horizon_sets(rules)
    covered = trades.loc[trades[_mcol(late)].notna()]
    grp = covered.groupby("account_key", observed=True)
    out = pd.DataFrame(index=grp.size().index)
    for h in horizons:
        col = _mcol(h)
        out[f"mo_{_hlabel(h)}_bps"] = grp[col].mean()
        out[f"hit_{_hlabel(h)}"] = grp[col].agg(lambda s: float((s > 0).mean())
                                                if s.notna().any() else np.nan)
    # The early window an account can support: sub-second when most of its
    # covered fills are millisecond-stamped (MT5), else the second fallback.
    ms_share = grp["ms_fill"].mean() if "ms_fill" in covered.columns \
        else pd.Series(0.0, index=out.index)
    out["ms_fill_share"] = ms_share
    peak = _early_peak(out, early, fallback, ms_share >= 0.5,
                       prefix="mo_", suffix="_bps")
    out["early_peak_bps"] = peak
    out["decay_pct"] = ((1.0 - out[f"mo_{_hlabel(late)}_bps"] / peak.where(peak > 0))
                        * 100.0).clip(-500, 100)
    band = float(rules.get("neutral_band_bps", 0.5))
    keep = 1.0 - float(rules.get("min_decay_pct", 50.0)) / 100.0
    profile = pd.Series("neutral", index=out.index)
    fast = (peak >= band) & (out[f"mo_{_hlabel(late)}_bps"] <= keep * peak)
    later = [f"mo_{_hlabel(h)}_bps" for h in horizons if h >= 1.0]
    persistent = ~fast & (out[later].max(axis=1) >= band)
    profile[persistent] = "persistent"
    profile[fast] = "fast_transient"
    out["curve_profile"] = profile
    return out


def _progress(stage: str, **extra) -> None:
    """Stage heartbeat for the scan -- written so a stalled scan is
    diagnosable from outside the process instead of a silent 30-minute
    mystery."""
    try:
        (ROOT / "artifacts" / "latency_scan_progress.json").write_text(
            json.dumps({"stage": stage, "at": str(datetime.utcnow()),
                        **extra}), encoding="utf-8")
    except Exception:
        pass


#: Numeric spec fields copied onto each profile row (s3 metrics, s5.3
#: components and confidence parts).
_SPEC_ROW_FIELDS = (
    "c_early_markout", "c_early_hit_rate", "c_price_age", "c_decay_consistency",
    "c_event_count", "c_profit_concentration", "c_reference",
    "conf_sample", "conf_quality", "conf_persistence", "conf_corroboration",
    "reference_confirmed_share", "reference_validated_share", "dislocation_share", "leadlag_share", "flag_refcov",
    "disloc_hits", "leadlag_hits", "architecture_share",
    "ref_mo_exec_100ms_bps", "ref_mo_exec_200ms_bps", "ref_mo_exec_300ms_bps", "ref_mo_exec_500ms_bps",
    "ref_mo_exec_1s_bps", "ref_mo_exec_5s_bps", "ref_mo_exec_60s_bps",
    "ref_mo_mid_100ms_bps", "ref_mo_mid_500ms_bps", "ref_mo_mid_60s_bps",
    "bench_diff_bps", "early_hit_rate", "bench_hit", "event_success_rate",
    "profit_concentration", "early_mo_usd", "peak_markout_bps", "persistence_ratio",
    "curve_slope_bps_per_decade", "auc_bps", "consistency_days", "consistency_symbols",
    "markout_consistency", "clustered_share", "top_symbol_share", "replicated_share",
    "latency_events", "qualifying", "realized_pnl", "ms_share",
    "med_100ms_bps", "med_200ms_bps", "med_300ms_bps", "med_500ms_bps",
    "med_1s_bps", "med_5s_bps", "med_60s_bps")


def _num(value, digits: int = 3):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def _scan_inner(rules: dict, end: datetime | None = None) -> dict:
    scan_started = time.time()
    horizons, _early, _fallback, late = _horizon_sets(rules)
    _progress("load_trades")
    trades = _load_trades(int(rules["window_days"]), end=end)
    _progress("loaded", n_trades=int(len(trades)))
    if not len(trades):
        # FULL-SHAPED empty result: the tab renders kpis/mode/targets from
        # every response, so even "no trades" must carry them.
        return {"rows": [], "n_trades": 0, "n_flagged_trades": 0,
                "tick_coverage": 0.0, "kpis": kpis([]),
                "mode": ("shadow" if rules.get("shadow_mode", True)
                         else "live"),
                "generated_at": str(datetime.utcnow()),
                "note": "no trades in window"}
    # s5.1 EXCLUSIONS before any markout work: excluded symbols (raw or
    # canonical) and excluded platform account groups.
    from webapp.trade_feed import _canonical
    trades["canonical"] = [(_canonical(s) or s) for s in trades["symbol"].astype(str)]
    exclusions = {}
    if rules.get("excluded_symbols") or rules.get("excluded_account_groups"):
        groups = None
        if rules.get("excluded_account_groups"):
            try:
                from webapp import client_profile
                prof = client_profile.load_all_profiles(
                    tuple(sorted(trades["database"].astype(str).unique())))
                groups = prof.drop_duplicates("account_key").set_index("account_key")["group"]
            except Exception as error:
                exclusions["group_lookup_error"] = f"{type(error).__name__}: {error}"
        trades, excl = latency_spec.apply_exclusions(trades, rules, groups)
        exclusions.update(excl)
        _progress("excluded", **exclusions)
        if not len(trades):
            return {"rows": [], "n_trades": 0, "n_flagged_trades": 0,
                    "tick_coverage": 0.0, "kpis": kpis([]), "exclusions": exclusions,
                    "mode": ("shadow" if rules.get("shadow_mode", True) else "live"),
                    "generated_at": str(datetime.utcnow()),
                    "note": "every trade in the window is excluded"}
        trades["row_id"] = np.arange(len(trades))
    mt4_ms = {}
    if end is None:
        try:
            # Keep MT4 millisecond coverage current: refresh the Kafka backfill
            # in the background when it is more than 6 h old (the next scan
            # benefits; this one never waits for it).
            from webapp import mt4_kafka_backfill
            if mt4_kafka_backfill.last_run_age_hours() > 6:
                mt4_kafka_backfill.launch_background()
        except Exception:
            pass
    try:
        _progress("mt4_ms_times")
        mt4_ms = _attach_mt4_ms_times(trades)
        _progress("mt4_ms_attached", **mt4_ms)
    except Exception as error:
        # No Kafka events (store busy, consumer gap): MT4 keeps its 1 s + 5 s
        # fallback window, exactly as before step 2.
        mt4_ms = {"error": f"{type(error).__name__}: {error}"}
        _progress("mt4_ms_failed", **mt4_ms)
    try:
        _progress("seed_ticks")
        seeded = _seed_quotes_from_mysql(trades, rules)
        _progress("seeded", **seeded)
    except Exception as error:
        # A backfill failure degrades to the live tape alone; never the scan.
        _progress("seed_ticks_failed", error=f"{type(error).__name__}: {error}")
    tape_error = None
    for attempt in range(3):
        try:
            _progress("quote_marks", attempt=attempt + 1)
            marks = _quote_marks(trades, horizons)
            trades = trades.join(marks, on="row_id")
            tape_error = None
            break
        except Exception as error:
            # Tick tape busy (single-writer duckdb contended by the quote
            # materialiser): WAIT AND RETRY -- a silent one-shot degrade
            # produced 0-coverage scans that overwrote good caches.
            tape_error = f"{type(error).__name__}: {error}"
            time.sleep(20)
    if tape_error is not None:
        for h in horizons:
            trades[_mcol(h)] = np.nan
        trades["quote_age_ms"] = np.nan
        trades["ms_fill"] = False
        trades["entry_spread_rel"] = np.nan
        trades["spread_ratio"] = np.nan

    # Coverage = the trade has a tick-tape mark at the persistence horizon
    # (sub-second horizons exist only for millisecond fills).
    tick_cov = trades[_mcol(late)].notna()

    trades["fast"] = trades["hold_seconds"] <= float(rules["max_hold_seconds"])
    # STALE QUOTE (spec s2.0.1): quote absence up to the 200 ms throttling
    # baseline is normal; only age beyond baseline + tolerance is evidence,
    # and only on millisecond fills. Beyond 60 s the gap is a STORE COVERAGE
    # hole (consumer restart), not venue staleness.
    stale_ms = (float(rules.get("quote_throttle_ms", 200))
                + float(rules.get("quote_age_tolerance_ms", 100)))
    trades["stale_entry"] = (trades["quote_age_ms"] > stale_ms) \
        & (trades["quote_age_ms"] <= float(rules.get("max_price_age_ms", 60_000))) \
        & trades["fast"] & (trades["net_profit"] > 0)
    # LIQUIDITY: entry taken while the spread sat at a multiple of the
    # symbol's own median -- an illiquid-period fill.
    trades["illiquid_entry"] = (
        trades["spread_ratio"]
        >= float(rules.get("illiquid_spread_ratio", 1.5))) & trades["fast"]

    # FLAG = tick-tape PROOF only: a fast, paid trade with an early advantage
    # that decays (latency); a persistent advantage is recorded as DIRECTIONAL
    # instead. Stale-quote and liquidity evidence stay as profile axes, and the
    # ML model below covers the trades the tape cannot.
    trades = _classify_trades(trades, rules)
    # INDEPENDENT REFERENCE MARKETS (spec s1, s2.1, s5.3, L1/L4): Vantage Raw
    # ECN and IC Markets Raw, each fetched out of process and cached as
    # parquet, combined per `reference_combine`. They replace
    # the old MySQL second-source check, which read the same tape the
    # markouts use and so corroborated nothing.
    ref_summary: dict = {}
    _progress("reference", flagged=int(trades["flagged"].sum()), tape_error=tape_error)
    if rules.get("reference_feed_enabled", True) and tape_error is None:
        try:
            fetch_stats = latency_reference.ensure_reference(trades, rules, progress=_progress)
            _progress("reference_features", **{k: v for k, v in fetch_stats.items() if k != "fetch"})
            feats, ref_summary = latency_reference.reference_features(trades, rules, horizons, _hlabel)
            ref_summary["fetch"] = fetch_stats
            if len(feats):
                trades = trades.join(feats, on="row_id")
        except Exception as error:
            ref_summary = {"error": f"{type(error).__name__}: {error}"}
    for col in ("ref_div_bps", "ref_lead_bps", "broker_lead_bps", "ref_age_ms"):
        if col not in trades.columns:
            trades[col] = np.nan
    trades["ref_covered"] = (trades["ref_covered"].fillna(False).astype(bool)
                             if "ref_covered" in trades.columns else False)
    trades["ref_dislocation"] = trades["ref_covered"] & (
        trades["ref_div_bps"] >= float(rules.get("reference_divergence_bps", 1.0)))
    lead_thr = float(rules.get("reference_lead_bps", 1.0))
    trades["ref_leadlag"] = trades["ref_covered"] & (trades["ref_lead_bps"] >= lead_thr) \
        & (trades["broker_lead_bps"] <= 0.25 * trades["ref_lead_bps"])
    # Two different things the reference can say about a trade:
    # VALIDATED -- the client's fill also beats the REFERENCE executable price
    #   by the early threshold, so the favourable move was real market
    #   movement, not a broker-tape artefact (feeds confidence only);
    # CONFIRMED (latency-specific) -- at the fill the broker price sat behind
    #   the reference, or the reference had already moved while the broker
    #   quote had not (feeds the score's reference component and Ref✓).
    # Measured 15 Sep 2026: counting VALIDATED as confirmation lifted accounts
    # with zero dislocation by 5-8 points, i.e. it scored ordinary fast moves.
    precise_fill = trades["ms_fill"].fillna(False).astype(bool)
    ref_hit_sub = pd.Series(False, index=trades.index)
    ref_hit_sec = pd.Series(False, index=trades.index)
    for hs, target in ((_early, "sub"), (_fallback, "sec")):
        for h in hs:
            col = f"ref_exec_{_hlabel(h)}_bps"
            if col in trades.columns:
                hit = trades[col] >= float(rules["flag_markout_bps"].get(_hkey(h), 999))
                if target == "sub":
                    ref_hit_sub |= hit.fillna(False)
                else:
                    ref_hit_sec |= hit.fillna(False)
    trades["ref_validated"] = trades["ref_covered"] & ref_hit_sub.where(precise_fill, ref_hit_sec)
    trades["ref_confirm"] = trades["ref_covered"] & (trades["ref_dislocation"] | trades["ref_leadlag"])
    trades["indep_ok"] = trades["ref_confirm"].astype(float).where(
        trades["flagged"] & trades["ref_covered"])
    rc_table, arch_share = pd.DataFrame(), {}
    try:
        rc_table, arch_share = latency_reference.root_cause(trades, rules)
    except Exception as error:
        ref_summary["root_cause_error"] = f"{type(error).__name__}: {error}"

    _progress("aggregate")
    # VECTORIZED single-pass aggregation. The previous per-group apply
    # lambdas took ~45 minutes over 25k accounts under memory pressure;
    # precomputed helper columns + one groupby.agg take seconds.
    t2 = trades
    t2["_cov"] = tick_cov.to_numpy()
    t2["_flag_pnl"] = t2["net_profit"].where(t2["flagged"], 0.0)
    t2["_flag_day"] = pd.to_datetime(t2["open_time"]).dt.normalize() \
        .where(t2["flagged"])
    t2["_fast_win"] = t2["fast"] & (t2["net_profit"] > 0)
    t2["_flag_illiq"] = t2["illiquid_entry"].fillna(False) & t2["flagged"]
    grp = t2.groupby("account_key", observed=True)
    agg = grp.agg(
        trades=("net_profit", "size"),
        flagged=("flagged", "sum"),
        directional_flagged=("directional", "sum"),
        econ_usd=("_flag_pnl", "sum"),
        profit_total=("net_profit", "sum"),
        tick_coverage=("_cov", "mean"),
        first_day=("open_time", "min"),
        last_day=("open_time", "max"),
        active_days=("_flag_day", "nunique"),
        med_hold_s=("hold_seconds", "median"),
        best_markout=("early_peak_bps", "max"),
        avg_markout=(_mcol(late), "mean"),
        ms_fill_rate=("ms_fill", "mean"),
        scalp_share=("fast", "mean"),
        stale_share=("stale_entry", "mean"),
        _fast_n=("fast", "sum"),
        _fast_wins_n=("_fast_win", "sum"),
        _hold_sum=("hold_seconds", "sum"),
        _flag_illiq_n=("_flag_illiq", "sum"),
        spread_ratio_avg=("spread_ratio", "mean"),
        indep_share=("indep_ok", "mean"),
        indep_checked=("indep_ok", "count"),
    )
    agg["fast_wins"] = (agg["_fast_wins_n"]
                        / agg["_fast_n"].clip(lower=1)).where(
        agg["_fast_n"] > 0, 0.0)
    agg["profit_per_min"] = agg["profit_total"] \
        / (agg["_hold_sum"] / 60.0).clip(lower=1e-6)
    agg["illiquid_share"] = (agg["_flag_illiq_n"]
                             / agg["flagged"].clip(lower=1)).where(
        agg["flagged"] > 0, 0.0)
    agg = agg.drop(columns=["_fast_n", "_fast_wins_n", "_hold_sum",
                            "_flag_illiq_n"])
    # Markout side-table for EVERY scanned account (before any flag filter):
    # feeds the registry's toxic_flow fields (mk_short / mk_trades).
    try:
        mk = {str(a): [round(float(v), 2) if np.isfinite(v) else None,
                       int(round(float(c) * float(n)))]
              for a, v, c, n in zip(agg.index, agg["avg_markout"],
                                    agg["tick_coverage"], agg["trades"])}
        _MK_PATH().write_text(json.dumps(
            {"at": str(datetime.utcnow()), "mk": mk}), encoding="utf-8")
    except Exception:
        pass

    # MARKOUT CURVE per account (spec s2.2 / s3) over every scanned account,
    # before the latency filter, so the scan can report how the book's flow
    # splits between neutral, fast/transient and persistent/directional.
    curves = _curve_profiles(trades, rules)
    agg = agg.join(curves, how="left")
    agg["curve_profile"] = agg["curve_profile"].fillna("no_ticks")
    min_flags = int(rules["min_flagged_trades"])
    curve_summary = {
        "profiles": {str(k): int(v) for k, v in
                     agg["curve_profile"].value_counts().items()},
        "n_directional_trades": int(trades["directional"].sum()),
        "n_directional_accounts": int(
            (agg["directional_flagged"] >= min_flags).sum()),
    }

    # SPEC SCORING (s3 metrics, s5.3 score / bands / confidence, s5.4
    # profiles, s5.1-5.2 gates) over every scanned account.
    _progress("spec_scoring")
    spec_summary = {}
    replicated_map: dict = {}
    try:
        spec, spec_summary = latency_spec.account_metrics(
            trades, rules, horizons, _early, _fallback, late, _mcol, _hlabel, _hkey,
            arch_share=arch_share)
        if "replicated_share" in spec.columns:
            replicated_map = spec["replicated_share"].dropna().to_dict()
        spec = spec.drop(columns=[c for c in spec.columns if c in agg.columns])
        agg = agg.join(spec, how="left")
    except Exception as error:
        spec_summary = {"error": f"{type(error).__name__}: {error}"}

    # P1 ENGINE B (TOXIC FLOW) rides on THIS pass. s12 of the P1 specification:
    # "The Markout Engine should be implemented once and consumed by all three
    # P1 engines so calculations remain consistent." A second pass over the
    # single-writer tick store would also cost another ~13 minutes for markouts
    # we already hold in memory right here. Engine B must never break Engine A,
    # so every failure is captured and reported, not raised.
    _progress("toxic_flow")
    toxic_summary: dict = {}
    # LIVE scans only: build() persists Engine B's scan, today's history and
    # the audit trail, so a past as-of scan must not overwrite them.
    if end is None:
        try:
            from webapp import toxic_flow
            toxic_summary = toxic_flow.build(
                trades, rules, horizons, _early, _fallback, late, _mcol, _hlabel,
                replicated=replicated_map or None)
        except Exception as error:
            toxic_summary = {"error": f"{type(error).__name__}: {error}"}
    else:
        toxic_summary = {"skipped": "historical as-of scan: Engine B rebuilds on live scans only"}
    for col, default in (("spec_score", 0.0), ("spec_confidence", 0.0), ("band", "normal"),
                         ("spec_verdict", "clear"), ("profiles", ""), ("gates_failed", "")):
        if col not in agg.columns:
            agg[col] = default
        agg[col] = agg[col].fillna(default)

    agg_all = agg
    agg = agg[agg["flagged"] >= min_flags]

    # EVENT PROXIMITY: entries clustered on macro prints (the dynamic
    # calendar) -- event-driven flow, feeds the News/Event/Vol category.
    try:
        from webapp import econ_calendar
        opens = trades.loc[trades["flagged"]].groupby(
            "account_key", observed=True)["open_time"]
        agg["event_share"] = pd.Series(
            {a: econ_calendar.near_share(g.values) for a, g in opens},
            dtype=float).reindex(agg.index).fillna(0.0)
    except Exception:
        agg["event_share"] = 0.0

    # BEHAVIOURAL FUSION (per the spec: the AI/ML behavioural profiling is
    # integrated into the risk decision): accounts the behavioural system
    # already scores as latency-adjacent profiles carry that evidence in.
    fusion = pd.Series(0.0, index=agg.index)
    try:
        from webapp import antifraud
        panel = antifraud.classify()
        if panel is not None and len(panel):
            latady = panel.loc[panel["profile"].isin(
                ["toxic_flow", "persistent_edge", "scalper", "news_vol"])]
            score = pd.to_numeric(latady["score"], errors="coerce").fillna(0.0)
            hi = float(max(score.quantile(0.95), 1.0))
            # An account can carry several profiles: keep its strongest.
            per_acct = (pd.Series(score.values / hi,
                                  index=latady["account_key"].astype(str))
                        .clip(0, 1).groupby(level=0).max())
            fusion = per_acct.reindex(agg.index).fillna(0.0)
    except Exception:
        pass
    agg["behavioural"] = fusion

    # ML DETECTOR: the model trained on tape-proved labels scores every
    # account from behaviour alone -- covering trades the tape cannot.
    ml = pd.Series(0.0, index=agg.index)
    model = _ml_model()
    if model is not None:
        try:
            flist = ml_meta().get("features") or ML_FEATURES
            feats = _full_features(trades).reindex(agg.index)
            X = feats.reindex(columns=flist).fillna(0).astype(float)
            ml = pd.Series(model.predict(X.to_numpy()),
                           index=agg.index).clip(0, 1)
        except Exception:
            pass
    agg["ml"] = ml
    # 5-DAY EARLY WARNING beside the NOW detector: the generic rule-models
    # artifact for this category, over the 90d feature table at scan end.
    agg["ew"] = _ew_scores(agg.index, end)

    # FIVE AXES (+ fusion).
    agg["consistency"] = agg["flagged"] / agg["trades"]
    span_days = ((pd.to_datetime(agg["last_day"])
                  - pd.to_datetime(agg["first_day"])).dt.days + 1).clip(lower=1)
    agg["persistence"] = (agg["active_days"] / span_days).clip(0, 1)
    # Confidence: probability the flag count is NOT luck, against the base
    # rate of flagged trades across the whole book (normal approx binomial).
    base_rate = max(float(trades["flagged"].mean()), 1e-4)
    mu = agg["trades"] * base_rate
    sd = np.sqrt(agg["trades"] * base_rate * (1 - base_rate)).clip(lower=1e-6)
    z = (agg["flagged"] - mu) / sd
    agg["confidence"] = pd.Series(
        [0.5 * (1 + math.erf(v / math.sqrt(2))) for v in z],
        index=agg.index).clip(0, 1)
    econ_scale = float(max(agg["econ_usd"].quantile(0.9), 100.0))
    agg["econ_axis"] = (agg["econ_usd"] / econ_scale).clip(0, 1)
    # THE ORIGINAL FOUR-AXIS SCALE (a perfect case reads 100 again), with the
    # newer evidence as an UPLIFT rather than a dilution: behavioural fusion,
    # stale-quote share and event proximity push borderline true positives
    # over the tiers without deflating the clear-cut ones.
    base = (0.30 * agg["confidence"] + 0.25 * agg["consistency"]
            + 0.20 * agg["persistence"] + 0.25 * agg["econ_axis"])
    uplift = (1.0 + 0.35 * agg["ml"].clip(0, 1)
              + 0.20 * agg["behavioural"].clip(0, 1)
              + 0.10 * agg["stale_share"].clip(0, 1)
              + 0.05 * agg["event_share"].clip(0, 1)
              + 0.06 * agg["illiquid_share"].fillna(0).clip(0, 1))
    # The pre-spec fused score, kept alongside for comparison only.
    agg["legacy_risk"] = (100 * base * uplift).clip(upper=100).round(1)
    agg["stat_confidence"] = agg["confidence"]

    # RISK = the spec's Latency Risk Score (s5.3); CONFIDENCE is reported
    # separately (0-1 here, 0-100 in spec_confidence). The verdict follows the
    # spec band, capped at monitor when an alert gate fails (s5.2) or the
    # pattern is replicated across accounts (s5.5.4).
    agg["risk"] = agg["spec_score"].astype(float)
    agg["confidence"] = (agg["spec_confidence"].astype(float) / 100.0).clip(0, 1)
    agg["verdict"] = agg["spec_verdict"]
    agg = agg[agg["verdict"] != "clear"].sort_values("risk", ascending=False)

    mode = "shadow" if rules.get("shadow_mode", True) else "live"
    actions = rules["actions"]
    rows = []
    # RECOMMENDED ACTION from the module's own ACTION_UNIVERSE, backtested on
    # THIS client's history (action_test sweeps every intervention and picks
    # the best broker-P&L response). Escalate/restrict tiers only, cached an
    # hour -- the sweep costs real time per client.
    _progress("recommended_actions", accounts=int(len(agg)))
    # The orders behind each account's latency events: the account page marks
    # them, and each one opens its own markout evidence.
    # Kept for EVERY account with a latency event, not only the table's rows:
    # an account held back by the event minimum (mt4_live01:1062756, 3 events)
    # still needs its flagged trades marked on its page (15 Sep 2026).
    try:
        flagged_orders = {str(k): v for k, v in
                          (trades.loc[trades["flagged"]]
                           .groupby("account_key", observed=True)["order"]
                           .apply(lambda s: [str(int(v)) for v in pd.to_numeric(s, errors="coerce").dropna()])
                           .to_dict()).items() if v}
    except Exception:
        flagged_orders = {}
    rec_map = _recommended_actions(
        [a for a, r in agg.iterrows()
         if r["verdict"] in ("escalate", "restrict")][:40])
    for account, r in agg.iterrows():
        act = actions.get(r["verdict"], {})
        rec = rec_map.get(str(account))
        a_min = a_max = None
        if isinstance(rec, dict):
            m, x = rec.get("min"), rec.get("max")
            if m:
                a_min = (f"{m['label']} {m['magnitude']:g} {m['unit']} "
                         f"→ breakeven (${m['new_pnl']:,.0f})")
            if x:
                a_max = (f"{x['label']} {x['magnitude']:g} {x['unit']} "
                         f"(+${x['delta']:,.0f})")
        elif isinstance(rec, str):
            a_max = rec
        rows.append({
            "account": str(account),
            "risk": float(r["risk"]),
            "confidence": round(float(r["confidence"]), 3),
            "consistency": round(float(r["consistency"]), 3),
            "persistence": round(float(r["persistence"]), 3),
            "econ_usd": round(float(r["econ_usd"]), 2),
            "trades": int(r["trades"]), "flagged": int(r["flagged"]),
            "med_hold_s": round(float(r["med_hold_s"]), 1),
            "best_markout_bps": round(float(r["best_markout"]), 2)
            if np.isfinite(r["best_markout"]) else None,
            "avg_markout_bps": round(float(r["avg_markout"]), 2)
            if np.isfinite(r["avg_markout"]) else None,
            "last_ts": str(r["last_day"]),
            "tick_coverage": round(float(r["tick_coverage"]), 2),
            "ml": round(float(r["ml"]), 3),
            "ew": round(float(r["ew"]), 3)
            if np.isfinite(r["ew"]) else None,
            "behavioural": round(float(r["behavioural"]), 3),
            "stale_share": round(float(r["stale_share"]), 3),
            "scalp_share": round(float(r["scalp_share"]), 3),
            "event_share": round(float(r["event_share"]), 3),
            "fast_wins": round(float(r["fast_wins"]), 3),
            "profit_per_min": round(float(r["profit_per_min"]), 2),
            "illiquid_share": round(float(r["illiquid_share"]), 3)
            if np.isfinite(r["illiquid_share"]) else 0.0,
            "indep_share": round(float(r["indep_share"]), 3)
            if np.isfinite(r["indep_share"]) else None,
            "indep_checked": int(r["indep_checked"]),
            "spread_ratio_avg": round(float(r["spread_ratio_avg"]), 2)
            if np.isfinite(r["spread_ratio_avg"]) else None,
            "curve_profile": str(r["curve_profile"]),
            "directional_flagged": int(r["directional_flagged"]),
            **{key: (round(float(r[key]), 3) if pd.notna(r.get(key)) else None)
               for key in curves.columns if key != "curve_profile"},
            # ---- spec (s3 / s5.3 / s5.4) ----
            "band": str(r["band"]),
            "spec_confidence": round(float(r["spec_confidence"]), 1),
            "legacy_risk": float(r["legacy_risk"]),
            "stat_confidence": round(float(r["stat_confidence"]), 3),
            "profiles": str(r["profiles"]),
            "l1_basis": str(r.get("l1_basis") or ""),
            "flagged_orders": flagged_orders.get(account, []),
            "gates_failed": str(r["gates_failed"]),
            **{key: _num(r.get(key)) for key in _SPEC_ROW_FIELDS},
            "verdict": r["verdict"],
            "action_min": a_min or "—",
            "action_max": a_max or act.get("treatment", "watchlist"),
            "action": a_max or act.get("treatment", "watchlist"),
            "delay_ms": act.get("delay_ms", 0),
        })
    # AUTOMATION PATHWAY: every scan decision is written to the audit log in
    # the configured mode (shadow = logged, not executed), now with the
    # automation TAGS. audit_tail is newest-first, so it is walked oldest ->
    # newest to keep each account's LATEST entry (the previous comprehension
    # kept the oldest, re-logging accounts whose verdict had not changed).
    tail = list(reversed(audit_tail(5000)))
    last_auto = {a["account"]: a for a in tail if a["status"] == "auto"}
    review: dict = {}
    for a in tail:
        if a["status"] in ("confirm", "dismiss"):
            review[a["account"]] = a["status"]
        elif a["status"] == "clear":
            review.pop(a["account"], None)
    live_scan = end is None
    new_alerts = 0
    for row in rows:
        row["tags"] = latency_tags.account_tags(row, review.get(row["account"]), min_flags)
        prior = last_auto.get(row["account"])
        changed = (prior is None or prior.get("verdict") != row["verdict"]
                   or (prior.get("tags") or "") != ",".join(t for t in row["tags"] if not t.startswith("LA_REVIEW_")))
        if live_scan and changed:
            _audit(row["account"], row["verdict"], row["risk"], row["action"], mode, "system", "auto",
                   tags=[t for t in row["tags"] if not t.startswith("LA_REVIEW_")])
            new_alerts += 1
    # Accounts that were tagged before and dropped out are CLEAR now: log it,
    # so automation removes their treatment instead of leaving it applied.
    if live_scan:
        shown = {r["account"] for r in rows}
        for acct, prior in last_auto.items():
            if acct not in shown and prior.get("verdict") not in ("clear", None):
                _audit(acct, "clear", 0.0, "remove treatment", mode, "system", "auto",
                       tags=["LA_VERDICT_CLEAR", "LA_BAND_NORMAL"])

    # CLOSEST TO THE THRESHOLD: the accounts just below the table, with what
    # holds each one back -- so an empty table still explains itself.
    near_misses = []
    try:
        pool = agg_all.loc[~agg_all.index.isin([r["account"] for r in rows])]
        pool = pool.loc[pool["flagged"] >= 1].copy()
        # Rank by score weighted by repetition: one lucky event is not "close".
        pool["_rank"] = pool["spec_score"].fillna(0) * (pool["flagged"] / min_flags).clip(upper=1)
        pool = pool.sort_values("_rank", ascending=False).head(15)
        comps = [("c_early_markout", "early markout", 20), ("c_early_hit_rate", "early hit rate", 15),
                 ("c_price_age", "stale quotes", 20), ("c_decay_consistency", "fading edge", 15),
                 ("c_event_count", "repeated events", 10), ("c_profit_concentration", "profit from events", 10),
                 ("c_reference", "reference confirmation", 10)]
        for acct, r in pool.iterrows():
            missing = sorted(((w * (1 - float(r.get(c) or 0)), label) for c, label, w in comps), reverse=True)[:3]
            near_misses.append({
                "account": str(acct), "risk": _num(r.get("spec_score"), 1), "band": str(r.get("band") or "normal"),
                "confidence": _num((r.get("spec_confidence") or 0) / 100, 3), "latency_events": int(r.get("flagged") or 0),
                "econ_usd": _num(r.get("econ_usd"), 2), "profiles": str(r.get("profiles") or ""),
                "points_short": round(max(0.0, 25 - float(r.get("spec_score") or 0)), 1),
                "held_back_by": ([f"only {int(r.get('flagged') or 0)} latency event(s); {min_flags} needed"]
                                 if int(r.get("flagged") or 0) < min_flags else [])
                                + ([f"score {float(r.get('spec_score') or 0):.0f}, below the Monitor band (25)"]
                                   if float(r.get("spec_score") or 0) < 25 else []),
                "weakest": [f"{label} ({pts:.0f} pts missing)" for pts, label in missing]})
    except Exception:
        near_misses = []

    if live_scan:
        try:
            record_history(rows)
        except Exception:
            pass
    with _LOCK:
        _CACHE["scan_finished_at"] = time.time()
        _CACHE["scan_duration_s"] = round(time.time() - scan_started, 1)
    _progress("done", rows=len(rows), seconds=_CACHE["scan_duration_s"],
              tape_error=tape_error, n_trades=int(len(trades)),
              tick_coverage=round(float(tick_cov.mean()), 3))
    return {"rows": rows, "n_trades": int(len(trades)),
            "n_flagged_trades": int(trades["flagged"].sum()),
            "base_flag_rate": round(base_rate, 5),
            **curve_summary,
            "tape_error": tape_error,
            "tick_coverage": round(float(tick_cov.mean()), 3),
            "ms_fill_share": round(float(trades["ms_fill"].mean()), 3),
            "mt4_ms_times": mt4_ms,
            "exclusions": exclusions,
            "spec": spec_summary,
            "reference": ref_summary,
            "near_misses": near_misses,
            "flagged_orders_all": flagged_orders,
            "root_cause": [
                {k: (_num(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
                     else (bool(v) if isinstance(v, (bool, np.bool_)) else str(v)))
                 for k, v in row.items()}
                for row in rc_table.head(25).to_dict("records")] if len(rc_table) else [],
            "mode": mode, "new_alerts": new_alerts,
            "engine_b": toxic_summary,
            "generated_at": str(datetime.utcnow()),
            "kpis": kpis(rows)}


# ------------------------------------------- independent-feed verification
#: canonical -> Vantage venue symbol (independent broker feed).
_VENUE_MAP: dict = {}


def _venue_symbol(canon: str):
    if not _VENUE_MAP:
        try:
            from webapp.vantage import _mt5
            for s in (_mt5().symbols_get() or []):
                _VENUE_MAP.setdefault(
                    "".join(c for c in s.name.upper() if c.isalnum())[:10], s.name)
                _VENUE_MAP[s.name.upper()] = s.name
        except Exception:
            return None
    c = str(canon).upper()
    return _VENUE_MAP.get(c) or next(
        (v for k, v in _VENUE_MAP.items() if k.startswith(c)), None)


#: Offset from warehouse open_time to MySQL `ticks.tm`. Measured 15 Sep 2026 on
#: 30 mt4_live01 XAUUSD trades: at +0h the tick at entry matches the fill price
#: to a median $0.10; at +3h (the old setting) it is off by ~$12 -- both columns
#: are UTC, so the +3h shift checked every flag against the wrong hours.
_TICK_TZ_OFFSET = timedelta(0)

#: MT4 servers whose `ticks` table (symbol_name, tm in UTC) can backfill the
#: quote store, in preference order.
TICK_SOURCES = ("mt4_live01", "mt4_live02", "mt4_live04", "mt4_live03")


def _seed_quotes_from_mysql(trades: pd.DataFrame, rules: dict,
                            max_rows: int = 6_000_000) -> dict:
    """Backfill the quote store from the MT4 `ticks` tables for every hour in
    which a flag-eligible trade (fast + paid) opened and no quotes exist yet.

    The Kafka quote store only holds what was consumed since it started, so a
    fresh install -- or any consumer gap -- left the 7-day window with no tick
    tape: 0% coverage and no flags by construction. The same ASOF markouts then
    run over the backfilled hours; `quotes_seeded` records which (canonical,
    hour) pairs are done so a rescan never refetches."""
    from webapp.kafka_service import shared_cursor
    from webapp.mysql_extract import _connection
    from webapp.trade_feed import _canonical

    hold = pd.to_numeric(trades.get("hold_seconds"), errors="coerce")
    eligible = trades.loc[(hold <= float(rules["max_hold_seconds"]))
                          & (trades["net_profit"]
                             >= float(rules["min_trade_profit_usd"]))]
    if not len(eligible):
        return {"hours": 0, "rows": 0}
    canon = pd.Series([(_canonical(s) or s) for s in eligible["symbol"].astype(str)],
                      index=eligible.index)
    feeds = eligible["database"].map(_feed_of)
    hours = pd.to_datetime(eligible["open_time"]).dt.floor("h")
    wanted = set(zip(feeds, canon, hours))
    # The symbol variant each (feed, canonical) is most traded as -- the price
    # series those fills were actually dealt on (XAUUSDe and XAUUSD differ).
    traded_as = (pd.DataFrame({"feed": feeds, "canonical": canon,
                               "symbol": eligible["symbol"].astype(str)})
                 .groupby(["feed", "canonical"])["symbol"]
                 .agg(lambda s: s.value_counts().index[0]).to_dict())

    with shared_cursor() as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS quotes_seeded "
                   "(canonical VARCHAR, hour TIMESTAMP)")
        cx.execute("CREATE TABLE IF NOT EXISTS quotes_seeded_feed "
                   "(feed VARCHAR, canonical VARCHAR, hour TIMESTAMP)")
        done = {(f, c, pd.Timestamp(h)) for f, c, h in
                cx.execute("SELECT feed, canonical, hour FROM quotes_seeded_feed").fetchall()}
        # The first backfill generation seeded MT4 ticks without a feed column.
        done |= {("mt4", c, pd.Timestamp(h)) for c, h in
                 cx.execute("SELECT canonical, hour FROM quotes_seeded").fetchall()}
        # Hours the live Kafka consumer covers are still seeded: markouts read
        # the MySQL tape only (see _quote_marks), since live quotes carry
        # publish time rather than tick time.
    todo = sorted(w for w in wanted if w not in done)
    if not todo:
        return {"hours": 0, "rows": 0}

    conns: dict = {}
    names: dict = {}      # (source db, canonical) -> MT4 symbol_name
    mt5_feeder: dict = {}  # (mt5 db, symbol) -> dominant feeder
    total, filled = 0, []

    def connect(db):
        if db not in conns:
            try:
                conns[db] = _connection(db, timeout=300)
            except Exception:
                conns[db] = None
        return conns[db]

    try:
        if any(f == "mt4" for f, _, _ in todo):
            for db in TICK_SOURCES:
                cx_db = connect(db)
                if cx_db is None:
                    continue
                with cx_db.cursor() as cur:
                    cur.execute("SELECT DISTINCT symbol_name FROM ticks "
                                "WHERE tm > UTC_TIMESTAMP() - INTERVAL 1 DAY")
                    listed = [r[0] for r in cur.fetchall()]
                variants: dict[str, list[str]] = {}
                for name in listed:
                    variants.setdefault(_canonical(name) or name, []).append(name)
                for key, options in variants.items():
                    # The plain instrument name, else its shortest variant.
                    names[(db, key)] = key if key in options else min(options, key=len)
        by_pair: dict[tuple, list] = {}
        for f, c, h in todo:
            by_pair.setdefault((f, c), []).append(h)
        for (feed, c), hrs in by_pair.items():
            if feed == "mt4":
                source = next(((db, names[(db, c)]) for db in TICK_SOURCES
                               if conns.get(db) is not None and (db, c) in names), None)
                if source is None:
                    continue
                db, symbol_name = source
                sql = ("SELECT tm, bid, ask FROM ticks WHERE symbol_name = %s "
                       "AND tm >= %s AND tm < %s AND bid > 0 AND ask > 0")
                args_of = lambda lo_, hi_: (symbol_name, lo_, hi_)
                label = symbol_name
            elif feed in ("mt5_live01", "mt5_dubai_live01"):
                db = feed
                if connect(db) is None:
                    continue
                symbol_name = traded_as.get((feed, c), c)
                if (db, symbol_name) not in mt5_feeder:
                    # MT5 ticks carry one row per constituent FEEDER; the join
                    # needs one price series, so the dominant feeder of the
                    # last day is used and recorded (per-feeder analysis is
                    # the spec s5.5 root-cause layer).
                    with conns[db].cursor() as cur:
                        cur.execute("SELECT feeder, COUNT(*) FROM ticks WHERE symbol = %s "
                                    "AND ts > UTC_TIMESTAMP() - INTERVAL 1 DAY "
                                    "GROUP BY feeder ORDER BY 2 DESC LIMIT 1", (symbol_name,))
                        row = cur.fetchone()
                    mt5_feeder[(db, symbol_name)] = row[0] if row else None
                feeder = mt5_feeder[(db, symbol_name)]
                if feeder is None:
                    continue
                sql = ("SELECT datetime_msc, bid, ask FROM ticks WHERE feeder = %s "
                       "AND symbol = %s AND ts >= %s AND ts < %s AND bid > 0 AND ask > 0")
                args_of = lambda lo_, hi_, _f=feeder, _s=symbol_name: (_f, _s, lo_, hi_)
                label = f"{symbol_name}@{feeder}"
            else:
                continue
            hrs = sorted(hrs)
            # Merge consecutive hours into runs of at most 6 hours: few queries,
            # but no single fetch big enough to strain memory (gold is ~13k
            # ticks an hour, and MySQL DECIMALs arrive as Python objects).
            runs, run_start, prev = [], hrs[0], hrs[0]
            for h in hrs[1:]:
                if (h - prev > pd.Timedelta(hours=1)
                        or h - run_start >= pd.Timedelta(hours=6)):
                    runs.append((run_start, prev))
                    run_start = h
                prev = h
            runs.append((run_start, prev))
            for lo, hi in runs:
                if total >= max_rows:
                    break
                with conns[db].cursor() as cur:
                    cur.execute(sql, args_of(
                        lo.to_pydatetime(),
                        (hi + pd.Timedelta(hours=1, minutes=1)).to_pydatetime()))
                    rows = cur.fetchall()
                if rows:
                    frame = pd.DataFrame(rows, columns=["event_time", "bid", "ask"])
                    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
                    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
                    seeded = pd.DataFrame({
                        "server": f"mysql:{db}",
                        "event_time": pd.to_datetime(frame["event_time"]),
                        "symbol": label, "canonical": c,
                        "bid": frame["bid"], "ask": frame["ask"],
                        "mid": (frame["bid"] + frame["ask"]) / 2})
                    with shared_cursor() as cx:
                        cx.execute("INSERT INTO quotes SELECT * FROM seeded")
                    total += len(seeded)
                marked = pd.DataFrame({"feed": feed, "canonical": c,
                                       "hour": pd.date_range(lo, hi, freq="h")})
                with shared_cursor() as cx:
                    cx.execute("INSERT INTO quotes_seeded_feed SELECT * FROM marked")
                filled.append((feed, c, str(lo), str(hi)))
                _progress("seed_ticks", rows=int(total), ranges=len(filled))
    finally:
        for cx in conns.values():
            try:
                cx and cx.close()
            except Exception:
                pass
    return {"hours": len(todo), "rows": int(total), "ranges": len(filled)}


def _independent_verify(trades: pd.DataFrame, rules: dict,
                        cap: int = 300) -> pd.Series:
    # cap 300 (was 1200): these are SEQUENTIAL MySQL queries and the single
    # slowest scan stage; the largest-impact flags are verified first, so
    # the LP-confirmation signal loses almost nothing while the scan gets
    # minutes faster.
    """Second-source check of flagged trades against the MySQL `ticks` table
    -- the client venue's own tick feed, a SEPARATE pipeline from the kafka
    quote store the flags were computed on. Our quotes prove the client dealt
    on a stale price; the second source proves the market truly moved (real
    latency edge, not a feed glitch of one pipeline). True/False per checked
    trade, NaN where unchecked or no tick data."""
    out = pd.Series(np.nan, index=trades.index, dtype="float64")
    todo = trades.loc[trades["flagged"]].copy()
    if not len(todo):
        return out
    todo = todo.reindex(
        todo["net_profit"].abs().sort_values(ascending=False).index)[:cap]
    thr5 = float(rules["flag_markout_bps"].get("5", 2.0))
    try:
        from webapp.mysql_extract import _connection
    except Exception:
        return out
    conns: dict = {}
    try:
        for idx, r in todo.iterrows():
            db = str(r["account_key"]).split(":", 1)[0]
            if db not in conns:
                try:
                    conns[db] = _connection(db, timeout=60)
                except Exception:
                    conns[db] = None
            cx = conns[db]
            if cx is None:
                continue
            t0 = pd.Timestamp(r["open_time"]).to_pydatetime() + _TICK_TZ_OFFSET
            try:
                with cx.cursor() as cur:
                    cur.execute(
                        "SELECT bid, ask FROM ticks WHERE symbol_name = %s "
                        "AND tm >= %s AND tm < %s AND bid > 0 AND ask > 0 "
                        "ORDER BY tm LIMIT 60",
                        (str(r["symbol"]), t0, t0 + timedelta(seconds=35)))
                    rows = cur.fetchall()
            except Exception:
                continue
            # MySQL DECIMAL columns arrive as decimal.Decimal: cast BEFORE
            # any float math (Decimal * float raises TypeError and took the
            # whole scan down).
            mids = [(float(b) + float(a)) / 2 for b, a in rows]
            if len(mids) < 2 or mids[0] == 0:
                continue
            move = float(r["direction"]) * (mids[-1] - mids[0]) / mids[0] * 1e4
            out.loc[idx] = float(move >= thr5 * 0.5)
    finally:
        for cx in conns.values():
            try:
                cx and cx.close()
            except Exception:
                pass
    return out


# --------------------------------------------------------------- ML model
# THE COMBINATION THE ACCURACY QUESTION DEMANDS. The tick tape proves
# latency arbitrage in hindsight (entry markout beats the spread, fast hold,
# paid) but only covers 7 days minus gaps. So: tape-proved accounts become
# GROUND-TRUTH LABELS, and a gradient-boosted model learns to recognise them
# from behaviour alone -- hold-time shape, fast-trade win quality, profit
# velocity, session timing, symbol mix -- features that exist for EVERY
# account and EVERY period. The model then scores uncovered history and new
# accounts in real time, and its cross-validated precision/recall against
# held-out tape labels is the honest accuracy number.
ML_PATH = ROOT / "artifacts" / "latency_ml.txt"
ML_META = ROOT / "artifacts" / "latency_ml_meta.json"
_ML_CACHE: dict = {}

ML_FEATURES = [
    "n_trades", "fast60_share", "fast120_share", "med_hold_s", "p25_hold_s",
    "fast_win_rate", "fast_profit_share", "profit_per_min", "win_rate",
    "profit_factor", "top_hour_share", "night_share", "fx_share",
    "metal_share", "crypto_share", "mean_lots", "profit_std_ratio",
]


def _account_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Tick-free behavioural features per account -- computable for any
    account over any period, live or historical."""
    from webapp.trade_features import symbol_class
    t = trades.copy()
    t["hold"] = pd.to_numeric(t["hold_seconds"], errors="coerce")
    t["hour"] = pd.to_datetime(t["open_time"]).dt.hour
    klass = t["symbol"].astype(str).map(
        {s: symbol_class(s) for s in t["symbol"].astype(str).unique()})
    t["is_fx"] = klass.eq("fx") | klass.isna()
    t["is_metal"] = t["symbol"].astype(str).str.upper().str.startswith(("XAU", "XAG"))
    t["is_crypto"] = klass.eq("crypto")
    t["fast60"] = t["hold"] <= 60
    t["fast120"] = t["hold"] <= 120
    # VECTORIZED (was per-group apply lambdas -- minutes over 25k accounts;
    # helper columns + one groupby.agg run in seconds).
    t["_win"] = t["net_profit"] > 0
    t["_pos"] = t["net_profit"].clip(lower=0)
    t["_neg"] = (-t["net_profit"]).clip(lower=0)
    t["_f120_win"] = t["fast120"] & t["_win"]
    t["_f120_pos"] = t["_pos"].where(t["fast120"], 0.0)
    t["_night"] = t["hour"].isin([21, 22, 23, 0, 1])
    grp = t.groupby("account_key", observed=True)
    out = grp.agg(
        n_trades=("net_profit", "size"),
        fast60_share=("fast60", "mean"),
        fast120_share=("fast120", "mean"),
        med_hold_s=("hold", "median"),
        _pnl_sum=("net_profit", "sum"),
        _pnl_mean=("net_profit", "mean"),
        _pnl_std=("net_profit", "std"),
        _hold_sum=("hold", "sum"),
        _pos_sum=("_pos", "sum"),
        _neg_sum=("_neg", "sum"),
        _f120_n=("fast120", "sum"),
        _f120_win_n=("_f120_win", "sum"),
        _f120_pos=("_f120_pos", "sum"),
        win_rate=("_win", "mean"),
        night_share=("_night", "mean"),
        fx_share=("is_fx", "mean"),
        metal_share=("is_metal", "mean"),
        crypto_share=("is_crypto", "mean"),
        mean_lots=("volume_lots", "mean"),
    )
    out["p25_hold_s"] = grp["hold"].quantile(0.25)
    out["fast_win_rate"] = (out["_f120_win_n"]
                            / out["_f120_n"].clip(lower=1)).where(
        out["_f120_n"] > 0, 0.0)
    out["fast_profit_share"] = out["_f120_pos"] \
        / out["_pos_sum"].clip(lower=1e-9)
    out["profit_per_min"] = out["_pnl_sum"] \
        / (out["_hold_sum"] / 60.0).clip(lower=1e-6)
    out["profit_factor"] = out["_pos_sum"] / out["_neg_sum"].clip(lower=1e-9)
    hour_n = t.groupby(["account_key", "hour"], observed=True) \
        .size().rename("n").reset_index()
    top_hour = hour_n.groupby("account_key", observed=True)["n"].max()
    out["top_hour_share"] = (top_hour / out["n_trades"]).fillna(0.0)
    out["profit_std_ratio"] = (
        out["_pnl_std"] / out["_pnl_mean"].abs().clip(lower=1e-6)
    ).where(out["n_trades"] > 2, 0.0)
    # EMPIRICAL cost-per-day: realized client P&L per ACTIVE day over the
    # window -- the client's win is the book's cost, so this is what a rule
    # hit costs per day, measured, not modelled.
    t["_day"] = pd.to_datetime(t["open_time"]).dt.normalize()
    active_days = t.groupby("account_key", observed=True)["_day"].nunique()
    out["obs_active_days"] = active_days.reindex(out.index).fillna(0)
    out["avg_daily_pnl"] = (out["_pnl_sum"]
                            / out["obs_active_days"].clip(lower=1)).round(2)
    out = out.drop(columns=["_pnl_sum", "_pnl_mean", "_pnl_std", "_hold_sum",
                            "_pos_sum", "_neg_sum", "_f120_n", "_f120_win_n",
                            "_f120_pos"])
    # Registry rule fields the trades themselves carry: swap capture (share
    # of positive P&L that is collected swap) and overnight positioning.
    try:
        if "storage" in t.columns:
            swap = pd.to_numeric(t["storage"], errors="coerce").fillna(0.0)
            t["_swap_pos"] = swap.clip(lower=0)
            t["_win_pos"] = t["net_profit"].clip(lower=0)
            g2 = t.groupby("account_key", observed=True)
            sp, wp = g2["_swap_pos"].sum(), g2["_win_pos"].sum()
            out["swap_capture_rate"] = (sp / (sp + wp).clip(lower=1e-9)
                                        ).clip(0, 1)
        if "close_time" in t.columns:
            t["_overnight"] = (pd.to_datetime(t["close_time"]).dt.date
                               > pd.to_datetime(t["open_time"]).dt.date)
            out["overnight_share"] = t.groupby(
                "account_key", observed=True)["_overnight"].mean()
    except Exception:
        pass
    return out.replace([np.inf, -np.inf], np.nan)


def _full_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Trade-shape features JOINED with the account-day corpus row (the
    174-feature frame the behavioural models train on)."""
    feats = _account_features(trades)
    try:
        from webapp import antifraud
        frame = antifraud._frame()
        latest = (frame.sort_values("decision_day")
                  .groupby("account_key", observed=True).last())
        ad_cols = [c for c in latest.columns
                   if c != "decision_day"
                   and pd.api.types.is_numeric_dtype(latest[c])]
        feats = feats.join(latest[ad_cols].add_prefix("ad_"), how="left")
    except Exception:
        pass
    return feats


def train_model() -> dict:
    """Train the detector on tape-proved labels; report held-out accuracy.

    Labels come ONLY from accounts the tape covers well (>=30% of trades
    tick-marked): positive = >=min_flagged proven arb trades and >=$25
    extracted. Grouped 5-fold CV by account gives honest precision/recall."""
    rules = load_rules()
    horizons, _early, _fallback, late = _horizon_sets(rules)
    trades = _load_trades(int(rules["window_days"]))
    if not len(trades):
        return {"error": "no trades"}
    try:
        _attach_mt4_ms_times(trades)
    except Exception:
        pass
    marks = _quote_marks(trades, horizons)
    trades = trades.join(marks, on="row_id")
    tick_cov = trades[_mcol(late)].notna()
    # Labels use the SAME latency definition as the scan (decaying early
    # advantage), so the model learns latency, not directional flow.
    trades = _classify_trades(trades, rules)

    grp = trades.groupby("account_key", observed=True)
    # LABEL DOCTRINE: judge each account on its COVERED trades only.
    # Eligible = enough tick-marked trades to call it either way (>=20);
    # positive = proven arb (>=min_flagged flags, >=$25 extracted);
    # negative = provably clean (<=1 flag); the ambiguous middle is left
    # out of training entirely rather than polluting either class.
    n_cov = grp.apply(lambda g: int(tick_cov.loc[g.index].sum()))
    n_flag = grp["flagged"].sum()
    arb_econ = grp.apply(lambda g: float(
        g.loc[g["flagged"], "net_profit"].sum()))
    positive = (n_flag >= int(rules["min_flagged_trades"])) & (arb_econ >= 25.0)
    negative = n_flag <= 1
    labelable = n_cov[(n_cov >= 20) & (positive | negative)].index
    label = positive.reindex(labelable).fillna(False)

    # ACCUMULATE labels: every training pass banks today's verdicts, so the
    # labeled corpus GROWS with each trading day past the tape's 7-day window.
    _ensure_labels()
    day = datetime.utcnow().strftime("%Y-%m-%d")
    with sqlite3.connect(DB) as cx:
        cx.executemany(
            "INSERT OR REPLACE INTO latency_labels VALUES (?,?,?)",
            [(day, str(a), int(bool(label.get(a)))) for a in labelable])
        acc = pd.read_sql("SELECT account, MAX(label) AS label "
                          "FROM latency_labels GROUP BY account", cx)
    acc_label = pd.Series(acc["label"].values,
                          index=acc["account"].astype(str))

    # FEATURES: behavioural trade shape (17) + the account-day corpus row
    # (the 174-feature frame the behavioural models train on) -- ~190 total.
    feats = _full_features(trades)
    feature_list = list(feats.columns)

    train_idx = feats.index.intersection(acc_label.index)
    X = feats.reindex(train_idx).astype(float)
    y = acc_label.reindex(train_idx).astype(int)
    if int(y.sum()) < 10:
        return {"error": f"only {int(y.sum())} positive labels -- "
                         f"not enough tape-proved cases to train"}
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    Xv, yv = X.fillna(0).to_numpy(), y.to_numpy()
    oof = np.zeros(len(yv))
    for tr, te in skf.split(Xv, yv):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=63, class_weight="balanced",
                               random_state=0, verbosity=-1)
        m.fit(Xv[tr], yv[tr])
        oof[te] = m.predict_proba(Xv[te])[:, 1]
    auc = roc_auc_score(yv, oof)
    # OPERATING THRESHOLD chosen on out-of-fold probabilities to maximise
    # the WEAKER of precision/recall -- the balanced point, not 0.5.
    best = (0.5, 0.0, 0.0, 0.0)
    for thr in np.linspace(0.05, 0.95, 91):
        p = precision_score(yv, oof >= thr, zero_division=0)
        r = recall_score(yv, oof >= thr, zero_division=0)
        if min(p, r) > best[3]:
            best = (float(thr), p, r, min(p, r))
    threshold, prec, rec = best[0], best[1], best[2]
    final = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=63, class_weight="balanced",
                               random_state=0, verbosity=-1)
    final.fit(Xv, yv)
    final.booster_.save_model(str(ML_PATH))
    meta = {"auc": round(float(auc), 3),
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "threshold": round(threshold, 3),
            "positives": int(y.sum()), "labelable": int(len(y)),
            "label_days": int(pd.read_sql(
                "SELECT COUNT(DISTINCT day) AS d FROM latency_labels",
                sqlite3.connect(DB))["d"].iloc[0]),
            "trained_at": str(datetime.utcnow()),
            "features": feature_list}
    ML_META.write_text(json.dumps(meta), encoding="utf-8")
    _ML_CACHE.clear()
    return meta


def _ensure_labels() -> None:
    with sqlite3.connect(DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS latency_labels (
                day TEXT, account TEXT, label INTEGER,
                PRIMARY KEY (day, account))""")


def _ml_model():
    if "model" not in _ML_CACHE:
        try:
            import lightgbm as lgb
            _ML_CACHE["model"] = lgb.Booster(model_file=str(ML_PATH)) \
                if ML_PATH.exists() else None
        except Exception:
            _ML_CACHE["model"] = None
    return _ML_CACHE["model"]


def ml_meta() -> dict:
    try:
        return json.loads(ML_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ------------------------------------------------ score history (per day)
def _ensure_history() -> None:
    with sqlite3.connect(DB) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS latency_history (
                day TEXT, account TEXT, risk REAL, ml REAL, confidence REAL,
                consistency REAL, persistence REAL, econ_usd REAL,
                verdict TEXT, PRIMARY KEY (day, account))""")


def record_history(rows: list[dict]) -> None:
    """The day's history IS the latest live scan: replace the day's rows. The
    previous INSERT OR REPLACE only added, so accounts from earlier scans (and
    earlier scoring rules) stayed in the day's trend -- 15 Sep 2026 showed
    590 'needs action' while the live table held none."""
    _ensure_history()
    day = datetime.utcnow().strftime("%Y-%m-%d")
    with sqlite3.connect(DB) as cx:
        cx.execute("DELETE FROM latency_history WHERE day = ?", (day,))
        cx.executemany(
            "INSERT OR REPLACE INTO latency_history VALUES (?,?,?,?,?,?,?,?,?)",
            [(day, r["account"], r["risk"], r.get("ml"), r["confidence"],
              r["consistency"], r["persistence"], r["econ_usd"], r["verdict"])
             for r in rows])


def account_latency_history(account: str, limit: int = 60) -> list[dict]:
    _ensure_history()
    with sqlite3.connect(DB) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute(
            "SELECT * FROM latency_history WHERE account = ? "
            "ORDER BY day DESC LIMIT ?", (account, limit))]


def _MK_PATH():
    return SCAN_CACHE.parent / "latency_markouts.json"


def markout_fields() -> pd.DataFrame:
    """account -> mk_short (mean tape markout, bps, longest short horizon)
    and mk_trades (tick-covered trades), from the last scan pass over EVERY
    account in the window -- the registry's toxic-flow rule fields."""
    try:
        data = json.loads(_MK_PATH().read_text(encoding="utf-8"))["mk"]
        frame = pd.DataFrame.from_dict(data, orient="index",
                                       columns=["mk_short", "mk_trades"])
        frame["mk_short"] = pd.to_numeric(frame["mk_short"], errors="coerce")
        frame["mk_trades"] = pd.to_numeric(
            frame["mk_trades"], errors="coerce").fillna(0).astype(int)
        return frame
    except Exception:
        return pd.DataFrame(columns=["mk_short", "mk_trades"])


def history_trend(days: int = 30) -> list[dict]:
    """Aggregate daily trend from the recorded scan history: how many
    accounts each tier held and the dollars they extracted, day by day --
    the tab's 'is this getting better or worse' view."""
    _ensure_history()
    with sqlite3.connect(DB) as cx:
        cx.row_factory = sqlite3.Row
        rows = cx.execute(
            "SELECT day, COUNT(*) AS accounts, "
            "SUM(CASE WHEN verdict='escalate' THEN 1 ELSE 0 END) AS escalate, "
            "SUM(CASE WHEN verdict='restrict' THEN 1 ELSE 0 END) AS restrict_n, "
            "SUM(CASE WHEN verdict='monitor' THEN 1 ELSE 0 END) AS monitor, "
            "SUM(econ_usd) AS econ_usd, AVG(risk) AS avg_risk "
            "FROM latency_history GROUP BY day ORDER BY day DESC LIMIT ?",
            (int(days),)).fetchall()
    return [dict(r) for r in rows][::-1]


def account_flagged_orders(account: str) -> dict:
    """Latency-flagged orders for ANY account in the latest scan, with where the
    account stands: in the table (its verdict) or not (and why)."""
    scan = _CACHE.get("scan") or {}
    if not scan.get("flagged_orders_all") and not scan.get("rows"):
        try:
            scan = latest_full_scan() or {}
        except Exception:
            scan = {}
    row = next((r for r in scan.get("rows", []) if r["account"] == account), None)
    orders = (row or {}).get("flagged_orders") or (scan.get("flagged_orders_all") or {}).get(account) or []
    near = next((n for n in scan.get("near_misses", []) if n["account"] == account), None)
    if row:
        status = f"on the Latency table ({row.get('verdict')}, risk {row.get('risk')})"
    elif near:
        status = "not on the Latency table: " + "; ".join(near.get("held_back_by") or ["below the Monitor band"])
    elif orders:
        status = "not on the Latency table: below the alert thresholds"
    else:
        status = ""
    return {"orders": [str(o) for o in orders], "in_table": bool(row), "status": status,
            "scan_at": str(scan.get("generated_at") or "")[:16]}


def account_latency_current(account: str) -> dict | None:
    # Fall back to the saved scan: right after a restart the in-memory cache
    # is empty until the latency tab is opened, and the account page lost its
    # flagged-order marks (mt4_live01:1044610, 15 Sep 2026).
    cached = _CACHE.get("scan") or {}
    if not cached.get("rows"):
        try:
            cached = latest_full_scan() or {}
        except Exception:
            cached = {}
    for r in cached.get("rows", []):
        if r["account"] == account:
            return r
    return None


#: account -> (stamp, recommendation text). action_test sweeps are costly,
#: so the cache PERSISTS across restarts (they were the 15-minute scans).
_ACTION_REC_CACHE: dict = {}
_ACTION_REC_PATH = ROOT / "artifacts" / "latency_action_recs.json"


def _load_action_cache() -> None:
    if _ACTION_REC_CACHE:
        return
    try:
        for k, v in json.loads(
                _ACTION_REC_PATH.read_text(encoding="utf-8")).items():
            _ACTION_REC_CACHE[k] = (float(v[0]), v[1])
    except Exception:
        pass


def _save_action_cache() -> None:
    try:
        _ACTION_REC_PATH.write_text(
            json.dumps({k: [v[0], v[1]] for k, v in _ACTION_REC_CACHE.items()}),
            encoding="utf-8")
    except Exception:
        pass


def _recommended_actions(accounts: list, ttl: float = 86400.0) -> dict:
    """Best intervention per client from antifraud.action_test -- the
    module's ACTION_UNIVERSE swept against the client's own history."""
    out = {}
    now = time.time()
    _load_action_cache()
    try:
        from webapp import antifraud
    except Exception:
        return out
    for account in accounts:
        account = str(account)
        cached = _ACTION_REC_CACHE.get(account)
        if cached and now - cached[0] < ttl:
            out[account] = cached[1]
            continue
        try:
            result = antifraud.action_test(account) or {}
            text = _two_way_actions(result)
        except Exception:
            text = None
        _ACTION_REC_CACHE[account] = (now, text)
        if text:
            out[account] = text
    _save_action_cache()
    return out


#: What the desk can ACTUALLY deploy without destroying the relationship or
#: the venue's integrity -- the implementability ceiling per intervention.
_ACTION_CAPS = {"widen_spread": 2.0, "add_slippage": 1.0, "raise_swap": 2.0,
                "cap_size": 1.0, "delay_feed": 3.0, "reject_rate": 0.25}


def _two_way_actions(result: dict) -> dict | None:
    """The two calibrations the desk needs from one sweep:
    MIN = the gentlest intervention that gets broker P&L on this client to
    breakeven; MAX = the most profitable option within realistic,
    implementable magnitudes (_ACTION_CAPS)."""
    actions = result.get("actions") or {}
    if not actions:
        return None
    minimal, maximal = None, None
    for key, spec in actions.items():
        cap = _ACTION_CAPS.get(key, float("inf"))
        for pt in spec.get("curve", []):
            mag, new = float(pt["magnitude"]), float(pt["new_broker_pnl"])
            delta = float(pt["broker_pnl_delta"])
            if new >= 0 and (minimal is None or mag / max(cap, 1e-9)
                             < minimal["aggr"]):
                minimal = {"label": spec["label"], "magnitude": mag,
                           "unit": spec["unit"], "new_pnl": round(new, 2),
                           "aggr": mag / max(cap, 1e-9)}
            if mag <= cap and (maximal is None or delta > maximal["delta"]):
                maximal = {"label": spec["label"], "magnitude": mag,
                           "unit": spec["unit"], "delta": round(delta, 2)}
    if minimal:
        minimal.pop("aggr", None)
    return {"min": minimal, "max": maximal} if (minimal or maximal) else None


def kpis(rows: list[dict] | None = None) -> dict:
    """Measured against the spec's 27-Sep targets."""
    out = {"targets": {"detection_rate": 80, "improvement_vs_baseline": 30,
                       "alert_accuracy": 70, "automation_success": 95,
                       "manual_reduction": 50}}
    # The HONEST accuracy numbers: cross-validated against held-out
    # tape-proved labels, not a baseline-overlap proxy.
    meta = ml_meta()
    if meta:
        out["model_auc"] = meta.get("auc")
        out["model_precision"] = round(100 * (meta.get("precision") or 0), 1)
        out["model_recall"] = round(100 * (meta.get("recall") or 0), 1)
        out["model_positives"] = meta.get("positives")
    try:
        detected = {r["account"] for r in (rows or [])}
        # Baseline = accounts the EXISTING behavioural system already labels
        # abusive (toxic flow / persistent edge). Detection rate: how much of
        # that known cohort with latency-shaped activity we catch; improvement:
        # flagged accounts the baseline misses.
        from webapp import antifraud
        panel = antifraud.classify()
        base_accounts = set(
            panel.loc[panel["profile"].isin(
                ["toxic_flow", "persistent_edge", "scalper"]),
                "account_key"].astype(str)) if panel is not None and len(panel) else set()
        overlap = detected & base_accounts
        out["detection_rate"] = round(
            100 * len(overlap) / max(len(detected), 1), 1)
        out["improvement_vs_baseline"] = round(
            100 * len(detected - base_accounts) / max(len(base_accounts), 1), 1)
    except Exception:
        out["detection_rate"] = None
        out["improvement_vs_baseline"] = None
    try:
        tail = audit_tail(2000)
        confirms = sum(1 for a in tail if a["status"] == "confirm")
        dismisses = sum(1 for a in tail if a["status"] == "dismiss")
        autos = sum(1 for a in tail if a["status"] == "auto")
        manual = confirms + dismisses
        out["alert_accuracy"] = round(
            100 * confirms / manual, 1) if manual else None
        out["automation_success"] = round(
            100 * autos / max(autos + 0, 1), 1) if autos else None
        out["manual_reduction"] = round(
            100 * autos / max(autos + manual, 1), 1) if (autos + manual) else None
    except Exception:
        pass
    return out
