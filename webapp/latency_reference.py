"""Independent reference market feeds for the Latency Arbitrage engine.

Spec: "Where possible, broker observations must be compared with an
independent/reference market" (s1); markout against the executable
opposite-side reference price (s2.1); reference-price dislocation and
reference corroboration (s5.3); L1 stale-price capture and L4 lead/lag (s5.4);
feed divergence / stale-TOB metrics (s5.5).

SOURCES (FEEDS): each is a live Raw account, READ-ONLY, in its OWN portable
MT5 terminal, with credentials in a git-ignored yaml next to server.yaml:
  vantage    Vantage Raw ECN   vantage_reference.yaml
  icmarkets  IC Markets Raw    icmarkets_reference.yaml  (added 16 Sep 2026)
A reference terminal must never share a terminal with the copier or with a
terminal someone uses by hand: an MT5 terminal holds one login, and logging it
into a reference account would switch that terminal away from its own.

COMBINING: every feed is fetched, clock-calibrated and basis-adjusted on its
own. With `reference_combine = "any"` (default) a reference signal --
dislocation, lead/lag, a validated executable move -- counts when ANY venue
covering the trade shows it: a latency arbitrageur attacks our feed wherever
it lags another venue (user decision, 16 Sep 2026). A number k ("at least k
venues"), "majority" and "all" demand agreement instead; with k, a fill fewer
than k venues cover cannot be confirmed. On 16 Sep 2026 (Vantage + IC
Markets, 7 days) 1,393 fills showed a dislocation on at least one venue and
203 on both.

OTHER VENUES: Exness, FxPro, FXTM, Scope Markets and Pepperstone have configs
and terminal copies but are disabled (`enabled: false`): their logins never
reached the broker from an MT5 terminal and are probably MT4 accounts, which
the MetaTrader5 package cannot read.

WHY A SUBPROCESS: MetaTrader5 IPC calls hold the Python interpreter while they
wait (see vantage._MT5_CACHE). Bulk tick pulls inside the web process would
stall the copier, so `fetch` runs as `python -m webapp.latency_reference fetch
<request.json>` (one process per feed, in parallel) and writes parquet; the
scan only ever reads parquet.

CLOCK: MT5 tick times are the broker's SERVER time, not UTC. Measured
15-16 Sep 2026: Vantage and IC Markets are both UTC+3 (summer) and move to
UTC+2 in winter. The offset is calibrated per feed and UTC day against the
broker's own MySQL tick tape (candidates `reference_server_offsets_h`), never
hard-coded. Each file holds the server window [H+min_offset, H+max_offset+1)
for UTC hour H so either offset is covered.

BASIS: two brokers quote slightly different levels (gold: about $0.01 on
15 Sep 2026). Reference prices are shifted by the per-(feed, symbol, day)
median broker-minus-reference mid, so dislocation measures a TEMPORARY gap,
not a permanent price-level difference.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

#: Feeds listed first, in this order; any other feed follows alphabetically.
_FEED_ORDER = ("vantage", "icmarkets")
#: Labels for configs that predate the yaml `label` key.
_DEFAULT_LABELS = {"vantage": "Vantage Raw ECN", "icmarkets": "IC Markets Raw"}


def _discover_feeds() -> dict[str, dict]:
    """feed key -> config file, parquet cache directory, display label.

    Every `<key>_reference.yaml` next to server.yaml is a feed. Vantage keeps
    its original cache directory; the others cache under reference_ticks_<key>."""
    keys = sorted((p.name[:-len("_reference.yaml")] for p in ROOT.parent.glob("*_reference.yaml")),
                  key=lambda k: (_FEED_ORDER.index(k) if k in _FEED_ORDER else len(_FEED_ORDER), k))
    feeds = {}
    for key in keys:
        feeds[key] = {"config": ROOT.parent / f"{key}_reference.yaml",
                      "dir": ROOT / "artifacts" / ("reference_ticks" if key == "vantage" else f"reference_ticks_{key}"),
                      "label": _DEFAULT_LABELS.get(key, key)}
    return feeds


FEEDS: dict[str, dict] = _discover_feeds()
# The Vantage feed's paths under their original names.
REF_DIR = ROOT / "artifacts" / "reference_ticks"
CONFIG_PATH = ROOT.parent / "vantage_reference.yaml"

REFERENCE_DEFAULT_RULES = {
    "reference_feed_enabled": True,
    #: Reference feeds in use: "all" (every discovered *_reference.yaml that is
    #: enabled), or a list of feed keys in display order.
    "reference_feeds": "all",
    #: How several covering feeds combine into one reference signal
    #: (dislocation, lead/lag, validated executable move):
    #: "any"      one covering feed is enough (default, user decision
    #:            16 Sep 2026: an arbitrageur attacks whichever feed ours lags);
    #: k          at least k venues must show it (e.g. 2);
    #: "majority" more than half of the covering feeds must show it;
    #: "all"      every covering feed must show it.
    "reference_combine": "any",
    #: s5.2 Reference Price Delay, ms: the lead window for L4 and the
    #: tolerance on reference timestamps, beyond the 200 ms broker throttle.
    "reference_price_delay_ms": 500,
    #: Broker price behind the reference, in the client's favour, at the fill
    #: (bps, after basis): an execution during reference-price dislocation.
    "reference_divergence_bps": 1.0,
    #: L4: the reference moved at least this far in the client's direction
    #: over the lead window while the broker quote moved at most a quarter of it.
    "reference_lead_bps": 1.0,
    "reference_server_offsets_h": [2, 3],
    "reference_max_pairs": 2000,
    "reference_fetch_timeout_s": 1800,
    #: Fetch subprocesses running at once (each holds ~100 MB).
    "reference_fetch_parallel": 3,
    #: s5.5.2 root cause: a (feed, symbol) condition becomes a Market-Data
    #: Architecture Risk with at least this many stale-TOB events across at
    #: least `replication_min_accounts` distinct clients.
    "architecture_min_events": 10,
}


#: Source canonical -> reference venue canonical candidates, in order.
REFERENCE_ALIASES = {
    "JAPAN225": ["JP225", "JPN225", "NIKKEI225", "NIKKEI"],
    "USOIL": ["USOUSD", "XTIUSD", "WTI", "CLOIL", "SPOTCRUDE"],
    "UKOIL": ["UKOUSD", "XBRUSD", "BRENT", "SPOTBRENT"],
    "USOILV6": ["USOIL", "USOUSD", "XTIUSD", "WTI", "CLOIL"],
    "NAS100U6": ["NAS100", "NDX100", "USTEC"],
    "NAS100": ["NDX100", "USTEC", "US100"],
    "US30": ["DJ30", "DOW", "US30", "WS30"],
    "GER40": ["DE40", "GER40", "DAX40"],
    "XAUUSDS": ["XAUUSD"],
}

#: Per-trade feature columns every feed produces.
_VALUE_COLS = ("ref_age_ms", "ref_div_bps", "ref_lead_bps", "broker_lead_bps")


def _feed(feed: str) -> dict:
    if feed not in FEEDS:
        raise KeyError(f"unknown reference feed {feed!r}")
    return FEEDS[feed]


def _pair_path(canonical: str, hour: pd.Timestamp, feed: str = "vantage") -> Path:
    return _feed(feed)["dir"] / str(canonical) / f"{pd.Timestamp(hour):%Y%m%d_%H}.parquet"


def _no_symbol_path(feed: str) -> Path:
    """canonical -> epoch seconds when the venue last had no symbol for it."""
    return _feed(feed)["dir"] / "_no_symbol.json"


def _load_config(feed: str = "vantage") -> dict:
    import yaml
    return yaml.safe_load(_feed(feed)["config"].read_text(encoding="utf-8")) or {}


def label_of(feed: str) -> str:
    try:
        return str(_load_config(feed).get("label") or _feed(feed)["label"])
    except Exception:
        return _feed(feed)["label"]


def available(feed: str = "vantage") -> bool:
    try:
        cfg = _load_config(feed)
        return (cfg.get("enabled", True) is not False and bool(cfg.get("login"))
                and Path(str(cfg.get("terminal_path", ""))).exists())
    except Exception:
        return False


def _enabled(feed: str) -> bool:
    try:
        return _load_config(feed).get("enabled", True) is not False
    except Exception:
        return False


def configured_feeds(rules: dict) -> list[str]:
    """The feeds this scan reads: the rules' list (or every discovered feed
    for "all"), limited to feeds whose config is present and enabled."""
    FEEDS.update({k: v for k, v in _discover_feeds().items() if k not in FEEDS})
    wanted = rules.get("reference_feeds")
    if not wanted or wanted == "all":
        wanted = list(FEEDS)
    return [f for f in wanted if f in FEEDS and _enabled(f)]


def quorum_rank(n: int, mode) -> int:
    """How many of `n` covering feeds must show a signal (1-based rank).
    A number k means "at least k venues"; it can exceed `n`, in which case no
    value reaches the quorum."""
    text = str(mode if mode is not None else "all").strip().lower()
    if text.isdigit():
        return max(1, int(text))
    if text == "any":
        return 1
    if text == "majority":
        return n // 2 + 1
    return n


def quorum_label(mode) -> str:
    text = str(mode if mode is not None else "all").strip().lower()
    if text.isdigit():
        k = max(1, int(text))
        return "any one venue" if k == 1 else f"at least {k} venues"
    return {"any": "any one venue", "majority": "a majority of the venues"}.get(text, "every venue")


def quorum_value(values, mode):
    """The value a client-favourable measure reaches on at least the quorum of
    feeds: the k-th largest (all -> min, any -> max, majority -> in between,
    k -> the k-th largest, None when fewer than k feeds measured it)."""
    vals = sorted((v for v in values if v is not None and np.isfinite(v)), reverse=True)
    k = quorum_rank(len(vals), mode)
    if not vals or k > len(vals):
        return None
    return vals[k - 1]


def active_feeds(rules: dict) -> list[str]:
    """Configured feeds whose terminal and credentials are present."""
    return [f for f in configured_feeds(rules) if available(f)]


# --------------------------------------------------------------- fetcher
def fetch(pairs: list[tuple[str, str]], offsets_h=(2, 3), feed: str = "vantage") -> dict:
    """SUBPROCESS ONLY. Pull reference ticks for (canonical, UTC hour ISO)
    pairs into parquet. Read-only: no order functions are used."""
    import MetaTrader5 as mt5
    from webapp.trade_feed import _canonical

    cfg = _load_config(feed)
    ok = mt5.initialize(path=str(cfg["terminal_path"]), login=int(cfg["login"]),
                        password=str(cfg["password"]), server=str(cfg["server"]),
                        portable=True, timeout=120_000)
    if not ok:
        return {"feed": feed, "error": f"initialize failed: {mt5.last_error()}"}
    stats = {"feed": feed, "fetched": 0, "ticks": 0, "no_symbol": [], "failed": 0, "skipped": 0}
    try:
        info = mt5.account_info()
        if info is None or int(info.login) != int(cfg["login"]):
            return {"feed": feed, "error": "reference terminal attached to an unexpected account"}
        stats["account"] = f"{info.login} @ {info.server}"
        # canonical -> venue name: tradeable first, then shortest (same rule
        # as the copier's _venue_symbols).
        venue: dict[str, tuple] = {}
        raw: dict[str, tuple] = {}
        for s in (mt5.symbols_get() or []):
            key = _canonical(s.name) or s.name
            cand = (s.trade_mode != 4, len(s.name), s.name)
            if key not in venue or cand < venue[key]:
                venue[key] = cand
            # Raw names too: the normaliser shortens some (USTEC -> UST), and
            # aliases name the venue's own spelling.
            up = s.name.upper()
            if up not in raw or cand < raw[up]:
                raw[up] = cand
        # Instruments this venue names differently. Futures months map to the
        # spot/cash instrument; the per-day basis absorbs the premium.
        for source, candidates in REFERENCE_ALIASES.items():
            if source not in venue:
                hit = next((venue.get(c) or raw.get(c) for c in candidates if c in venue or c in raw), None)
                if hit is not None:
                    venue[source] = hit

        def resolve(canonical: str):
            """Venue symbol for a canonical name, trying a futures month's
            cash instrument (NAS100U6 -> NAS100, USOILMINV6 -> USOIL)."""
            if canonical in venue:
                return venue[canonical][2]
            base = re.sub(r"(MIN)?[FGHJKMNQUVXZ]\d$", "", canonical)
            for key in dict.fromkeys([base] + REFERENCE_ALIASES.get(base, [])):
                hit = venue.get(key) or raw.get(key)
                if hit is not None:
                    return hit[2]
            return None
        lo_off, hi_off = min(offsets_h), max(offsets_h) + 1
        selected = set()
        for canonical, hour_iso in pairs:
            hour = pd.Timestamp(hour_iso)
            name = resolve(canonical)
            if name is None:
                if canonical not in stats["no_symbol"]:
                    stats["no_symbol"].append(canonical)
                continue
            if name not in selected:
                mt5.symbol_select(name, True)
                selected.add(name)
            lo = (hour + pd.Timedelta(hours=lo_off)).to_pydatetime().replace(tzinfo=timezone.utc)
            hi = (hour + pd.Timedelta(hours=hi_off)).to_pydatetime().replace(tzinfo=timezone.utc)
            arr = mt5.copy_ticks_range(name, lo, hi, mt5.COPY_TICKS_INFO)
            if arr is None:
                stats["failed"] += 1
                continue
            frame = pd.DataFrame(arr)
            out = pd.DataFrame({
                "server_ms": frame["time_msc"].astype("int64") if len(frame) else pd.Series([], dtype="int64"),
                "bid": frame["bid"].astype(float) if len(frame) else pd.Series([], dtype=float),
                "ask": frame["ask"].astype(float) if len(frame) else pd.Series([], dtype=float)})
            out = out[(out["bid"] > 0) & (out["ask"] > 0)]
            path = _pair_path(canonical, hour, feed)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path, index=False)
            stats["fetched"] += 1
            stats["ticks"] += int(len(out))
            stats["symbol_" + canonical] = name
    finally:
        mt5.shutdown()
    if stats["no_symbol"]:
        try:
            path = _no_symbol_path(feed)
            known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            known.update({c: time.time() for c in stats["no_symbol"]})
            path.write_text(json.dumps(known), encoding="utf-8")
        except Exception:
            pass
    return stats


# ------------------------------------------------------- in-app helpers
def needed_pairs(trades: pd.DataFrame, rules: dict) -> list[tuple[str, pd.Timestamp]]:
    """(canonical, UTC hour) for every hour holding a qualifying (fast + paid)
    trade, plus the next hour when a 60 s horizon crosses into it."""
    hold = pd.to_numeric(trades["hold_seconds"], errors="coerce")
    q = trades.loc[(hold <= float(rules["max_hold_seconds"]))
                   & (trades["net_profit"] >= float(rules["min_trade_profit_usd"]))]
    if not len(q):
        return []
    ot = pd.to_datetime(q["open_time"])
    pairs = set(zip(q["canonical"].astype(str), ot.dt.floor("h")))
    late = ot + pd.Timedelta(seconds=65)
    pairs |= set(zip(q["canonical"].astype(str), late.dt.floor("h")))
    return sorted(pairs)


def ensure_reference(trades: pd.DataFrame, rules: dict, progress=None) -> dict:
    """Fetch every missing (symbol, hour) the scan's qualifying trades need."""
    return ensure_pairs(needed_pairs(trades, rules), rules, progress)


def ensure_pairs(pairs: list, rules: dict, progress=None) -> dict:
    """Fetch every missing (symbol, hour) for every active feed, one
    subprocess per feed, in parallel. Hours inside the last two are always
    refetched (their files may be incomplete)."""
    if not rules.get("reference_feed_enabled", True):
        return {"status": "disabled"}
    feeds = active_feeds(rules)
    if not feeds:
        return {"status": "not configured"}
    recent = pd.Timestamp(datetime.utcnow()) - pd.Timedelta(hours=2)
    cap = int(rules.get("reference_max_pairs", 2000))
    offsets = list(rules.get("reference_server_offsets_h", [2, 3]))
    stats: dict = {"status": "ok", "pairs": len(pairs), "feeds": {}}
    jobs: list = []
    for feed in feeds:
        try:
            unlisted = {c for c, at in json.loads(_no_symbol_path(feed).read_text(encoding="utf-8")).items()
                        if time.time() - float(at) < 86_400}
        except Exception:
            unlisted = set()
        missing = [(c, h) for c, h in pairs
                   if c not in unlisted and (h >= recent or not _pair_path(c, h, feed).exists())]
        fs = {"missing": len(missing)}
        stats["feeds"][feed] = fs
        if not missing:
            continue
        if len(missing) > cap:
            # Newest first: the live window matters most.
            missing = sorted(missing, key=lambda p: p[1], reverse=True)[:cap]
            fs["capped_at"] = cap
        _feed(feed)["dir"].mkdir(parents=True, exist_ok=True)
        # Unique per call: a scan and a per-trade evidence request can overlap.
        request = _feed(feed)["dir"] / f"_request_{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
        request.write_text(json.dumps({
            "feed": feed, "offsets_h": offsets,
            "pairs": [[c, pd.Timestamp(h).isoformat()] for c, h in missing]}), encoding="utf-8")
        jobs.append((feed, request))
    # Back-compat summary field: the largest missing count across feeds.
    stats["missing"] = max((f["missing"] for f in stats["feeds"].values()), default=0)
    if progress and jobs:
        progress("reference_fetch", pairs=stats["missing"], feeds=[f for f, _ in jobs])
    deadline = time.time() + float(rules.get("reference_fetch_timeout_s", 1800))
    width = max(1, int(rules.get("reference_fetch_parallel", 3)))
    for start in range(0, len(jobs), width):
        running = [(feed, request, subprocess.Popen(
            [sys.executable, "-m", "webapp.latency_reference", "fetch", str(request)],
            cwd=str(ROOT.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            for feed, request in jobs[start:start + width]]
        for feed, request, proc in running:
            try:
                out, err = proc.communicate(timeout=max(1.0, deadline - time.time()))
                lines = [ln for ln in (out or "").splitlines() if ln.startswith("{")]
                stats["feeds"][feed]["fetch"] = json.loads(lines[-1]) if lines else {"error": (err or "")[-500:]}
            except subprocess.TimeoutExpired:
                proc.kill()
                stats["feeds"][feed]["fetch"] = {"error": "reference fetch timed out"}
            finally:
                request.unlink(missing_ok=True)
    return stats


def load_reference(pairs: list[tuple[str, pd.Timestamp]], feed: str = "vantage") -> pd.DataFrame:
    frames = []
    for canonical, hour in pairs:
        path = _pair_path(canonical, hour, feed)
        if path.exists():
            try:
                f = pd.read_parquet(path)
            except Exception:
                continue
            if len(f):
                f["canonical"] = canonical
                f["utc_day"] = pd.Timestamp(hour).normalize()
                frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["server_ms", "bid", "ask", "canonical", "utc_day"])
    ref = pd.concat(frames, ignore_index=True)
    # Adjacent hour files overlap by design (they cover both offsets).
    return ref.drop_duplicates(["canonical", "server_ms", "bid", "ask"])


def _calibrate(ref: pd.DataFrame, cx, offsets_h) -> tuple[dict, dict]:
    """Per UTC day: server offset (hours) minimising the broker-vs-reference
    mid deviation; per (canonical, day): median basis. `cx` = a cursor on the
    quote store, with `_lr_ref` (canonical, server_ms, mid, utc_day) registered."""
    day_offset: dict = {}
    diag: dict = {}
    for day in sorted(ref["utc_day"].unique()):
        sub = ref.loc[ref["utc_day"] == day]
        top = sub["canonical"].value_counts().index[0]
        best = None
        for h in offsets_h:
            d = cx.execute("""
                WITH r AS (SELECT make_timestamp(CAST((server_ms - ?) * 1000 AS BIGINT)) AS t, mid AS rmid
                           FROM _lr_ref WHERE canonical = ? AND utc_day = ?),
                     b AS (SELECT event_time AS t, (bid + ask) / 2 AS bmid FROM quotes
                           WHERE server LIKE 'mysql:%' AND canonical = ?
                             AND event_time >= ? AND event_time < ? USING SAMPLE 40000 ROWS)
                SELECT b.bmid - r.rmid AS dev FROM b ASOF JOIN r ON b.t >= r.t""",
                [int(h) * 3_600_000, top, day, top, pd.Timestamp(day).to_pydatetime(),
                 (pd.Timestamp(day) + pd.Timedelta(days=1)).to_pydatetime()]).df()["dev"]
            if len(d) < 50:
                continue
            mad = float((d - d.median()).abs().median() / max(abs(float(sub["mid"].median())), 1e-9) * 1e4)
            if best is None or mad < best[1]:
                best = (int(h), mad)
        if best is not None:
            day_offset[pd.Timestamp(day)] = best[0]
            diag[str(pd.Timestamp(day).date())] = {"offset_h": best[0], "symbol": top,
                                                   "deviation_bps": round(best[1], 3)}
    return day_offset, diag


def _quantiles(s: pd.Series) -> dict:
    s = s.dropna()
    return {k: round(float(s.quantile(v)), 3) for k, v in
            (("p1", .01), ("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))} if len(s) else {}


def _features_one(trades: pd.DataFrame, rules: dict, horizons: list, hlabel,
                  feed: str) -> tuple[pd.DataFrame, dict]:
    """One feed's per-trade features (see reference_features)."""
    from webapp.kafka_service import shared_cursor
    summary: dict = {"label": label_of(feed)}
    pairs = needed_pairs(trades, rules)
    ref = load_reference(pairs, feed)
    if not len(ref):
        return pd.DataFrame(), dict(summary, coverage=0)
    ref["mid"] = (ref["bid"] + ref["ask"]) / 2
    offsets = [int(h) for h in rules.get("reference_server_offsets_h", [2, 3])]
    lead_ms = float(rules.get("reference_price_delay_ms", 500))
    hold = pd.to_numeric(trades["hold_seconds"], errors="coerce")
    q = trades.loc[(hold <= float(rules["max_hold_seconds"]))
                   & (trades["net_profit"] >= float(rules["min_trade_profit_usd"])),
                   ["row_id", "database", "canonical", "direction", "open_time", "open_price"]].copy()
    q["open_time"] = pd.to_datetime(q["open_time"])
    q = q[q["canonical"].isin(ref["canonical"].unique())]
    if not len(q):
        return pd.DataFrame(), dict(summary, coverage=0)
    from webapp.latency_arb import _feed_of
    q["feed"] = q["database"].map(_feed_of)
    q["t_lead"] = q["open_time"] - pd.to_timedelta(lead_ms, unit="ms")
    for i, h in enumerate(horizons):
        q[f"t_{i}"] = q["open_time"] + pd.to_timedelta(float(h), unit="s")

    with shared_cursor() as cx:
        cx.register("_lr_ref", ref[["canonical", "server_ms", "mid", "utc_day"]])
        day_offset, diag = _calibrate(ref, cx, offsets)
        cx.unregister("_lr_ref")
        if not day_offset:
            return pd.DataFrame(), dict(summary, coverage=0,
                                        error="clock calibration failed (no overlapping broker ticks)")
        summary["clock"] = diag
        # Offset per file day, then UTC time; unknown days take the most common offset.
        common = pd.Series(list(day_offset.values())).mode().iloc[0]
        off_ms = ref["utc_day"].map(lambda d: day_offset.get(pd.Timestamp(d), common)).astype("int64") * 3_600_000
        ref["t"] = pd.to_datetime(ref["server_ms"] - off_ms, unit="ms")
        ref["day"] = ref["t"].dt.normalize()
        cx.register("_lr_ref2", ref[["canonical", "t", "day", "bid", "ask", "mid"]])
        lo, hi = q["t_lead"].min() - pd.Timedelta(minutes=5), q["open_time"].max() + pd.Timedelta(minutes=3)
        # BASIS per (canonical, day): median broker mid minus reference mid.
        basis = cx.execute("""
            WITH b AS (SELECT canonical, event_time AS t, (bid + ask) / 2 AS bmid FROM quotes
                       WHERE server LIKE 'mysql:%' AND event_time >= ? AND event_time <= ?
                         AND canonical IN (SELECT DISTINCT canonical FROM _lr_ref2))
            SELECT b.canonical, date_trunc('day', b.t) AS day, median(b.bmid - r.mid) AS basis,
                   count(*) AS n
            FROM b ASOF JOIN _lr_ref2 r ON b.canonical = r.canonical AND b.t >= r.t
            GROUP BY 1, 2""", [lo.to_pydatetime(), hi.to_pydatetime()]).df()
        cx.register("_lr_basis", basis)
        cx.register("_lr_q", q)
        cx.execute("""
            CREATE OR REPLACE TEMP TABLE _lr_refb AS
            SELECT r.canonical, r.t, r.bid + COALESCE(k.basis, 0) AS bid,
                   r.ask + COALESCE(k.basis, 0) AS ask, r.mid + COALESCE(k.basis, 0) AS mid
            FROM _lr_ref2 r LEFT JOIN _lr_basis k ON k.canonical = r.canonical AND k.day = r.day""")
        cx.execute("""
            CREATE OR REPLACE TEMP TABLE _lr_bq AS
            SELECT CASE WHEN replace(server, 'mysql:', '') LIKE '%dubai%' THEN 'mt5_dubai_live01'
                        WHEN replace(server, 'mysql:', '') LIKE 'mt5%' THEN replace(server, 'mysql:', '')
                        ELSE 'mt4' END AS feed, canonical, event_time AS t, (bid + ask) / 2 AS mid
            FROM quotes WHERE server LIKE 'mysql:%' AND event_time >= ? AND event_time <= ?
              AND canonical IN (SELECT DISTINCT canonical FROM _lr_q)""", [lo.to_pydatetime(), hi.to_pydatetime()])

        def asof_ref(col_t: str, alias: str, cols: str) -> pd.DataFrame:
            return cx.execute(f"""
                SELECT q.row_id, {cols}, r.t AS {alias}_t
                FROM _lr_q q ASOF LEFT JOIN _lr_refb r ON r.canonical = q.canonical AND q.{col_t} >= r.t
            """).df().set_index("row_id")

        parts = [asof_ref("open_time", "r0", "r.mid AS r0_mid"),
                 asof_ref("t_lead", "rl", "r.mid AS rl_mid")]
        for i, _ in enumerate(horizons):
            parts.append(asof_ref(f"t_{i}", f"r{i + 1}", f"r.bid AS rb_{i}, r.ask AS ra_{i}, r.mid AS rm_{i}"))
        for col_t, alias in (("open_time", "b0"), ("t_lead", "bl")):
            parts.append(cx.execute(f"""
                SELECT q.row_id, b.mid AS {alias}_mid, b.t AS {alias}_t
                FROM _lr_q q ASOF LEFT JOIN _lr_bq b ON b.feed = q.feed AND b.canonical = q.canonical
                 AND q.{col_t} >= b.t""").df().set_index("row_id"))
        for name in ("_lr_ref2", "_lr_basis", "_lr_q"):
            cx.unregister(name)
        cx.execute("DROP TABLE IF EXISTS _lr_refb")
        cx.execute("DROP TABLE IF EXISTS _lr_bq")

    m = q.set_index("row_id").join(pd.concat(parts, axis=1))
    d, p = m["direction"].astype(float), m["open_price"].astype(float)
    tol = pd.Timedelta(seconds=5)
    fresh0 = m["r0_t"] >= m["open_time"] - tol          # a reference quote near the fill
    out = pd.DataFrame(index=m.index)
    out["ref_covered"] = fresh0.fillna(False)
    out["ref_age_ms"] = ((m["open_time"] - m["r0_t"]).dt.total_seconds() * 1000).where(fresh0)
    for i, h in enumerate(horizons):
        lab = hlabel(h)
        ok = m[f"r{i + 1}_t"] >= m[f"t_{i}"] - tol
        # s2.1 EXECUTABLE opposite side: a buy exits at the reference bid,
        # a sell at the reference ask.
        exit_px = np.where(d > 0, m[f"rb_{i}"], m[f"ra_{i}"])
        out[f"ref_exec_{lab}_bps"] = (d * (exit_px - p) / p * 1e4).where(ok & fresh0)
        out[f"ref_mid_{lab}_bps"] = (d * (m[f"rm_{i}"] - p) / p * 1e4).where(ok & fresh0)
    b_ok = m["b0_t"] >= m["open_time"] - tol
    # DISLOCATION at the fill: the reference already sits beyond the broker's
    # quote in the client's favour -- the broker price is stale.
    out["ref_div_bps"] = (d * (m["r0_mid"] - m["b0_mid"]) / p * 1e4).where(fresh0 & b_ok)
    lead_ok = (m["rl_t"] >= m["t_lead"] - tol) & (m["bl_t"] >= m["t_lead"] - tol)
    out["ref_lead_bps"] = (d * (m["r0_mid"] - m["rl_mid"]) / p * 1e4).where(fresh0 & lead_ok)
    out["broker_lead_bps"] = (d * (m["b0_mid"] - m["bl_mid"]) / p * 1e4).where(b_ok & lead_ok)
    summary["coverage"] = round(float(out["ref_covered"].mean()), 3)
    summary["qualifying_trades"] = int(len(out))
    summary["basis_pairs"] = int(len(basis))
    summary["ref_ticks"] = int(len(ref))
    return out, summary


def _combine(per: dict[str, pd.DataFrame], rules: dict) -> pd.DataFrame:
    """One feature frame from several feeds. Values from a feed that does not
    cover a trade are ignored. Each client-favourable measure takes the value
    the quorum of covering feeds reaches (quorum_value): "all" -> the least
    favourable feed, "any" -> the most favourable, "majority" -> the value
    more than half of them reach. Mid markouts are averaged."""
    feeds = list(per)
    index = per[feeds[0]].index
    for f in feeds[1:]:
        index = index.union(per[f].index)
    frames = {f: per[f].reindex(index) for f in feeds}
    covered = pd.DataFrame({f: frames[f]["ref_covered"].fillna(False).astype(bool) for f in feeds})
    out = pd.DataFrame(index=index)
    mode = str(rules.get("reference_combine", 2)).strip().lower()
    out["ref_n_feeds"] = covered.sum(axis=1).astype(int)
    # "Checked by the reference" means enough venues cover the fill for the
    # rule to be met at all: with "at least k venues", k of them.
    out["ref_covered"] = out["ref_n_feeds"] >= (max(1, int(mode)) if mode.isdigit() else 1)
    cols = [c for c in frames[feeds[0]].columns if c != "ref_covered"]
    for f in feeds[1:]:
        cols += [c for c in frames[f].columns if c != "ref_covered" and c not in cols]
    for col in cols:
        stack = pd.DataFrame({f: frames[f][col].where(covered[f]) if col in frames[f] else np.nan
                              for f in feeds})
        if col.startswith("ref_mid_"):
            out[col] = stack.mean(axis=1)
        elif col == "ref_age_ms":
            out[col] = stack.min(axis=1)
        elif col == "broker_lead_bps":
            # Broker-side: identical across feeds wherever it is defined.
            out[col] = stack.bfill(axis=1).iloc[:, 0]
        else:
            # k-th largest of the feeds that measured it, per trade (quorum
            # over the feeds with a value; a trade with none stays NaN).
            arr = stack.to_numpy(dtype=float)
            n_val = (~np.isnan(arr)).sum(axis=1)
            rank = np.array([quorum_rank(int(n), mode) if n else 1 for n in n_val])
            arr = -np.sort(-np.where(np.isnan(arr), -np.inf, arr), axis=1)
            # A quorum larger than the feeds that measured it cannot be met.
            reachable = rank <= arr.shape[1]
            picked = arr[np.arange(len(arr)), np.minimum(rank, arr.shape[1]) - 1]
            out[col] = np.where(reachable & np.isfinite(picked), picked, np.nan)
    # Per-feed evidence at the fill, for diagnostics and the agreement check.
    for f in feeds:
        out[f"ref_covered_{f}"] = covered[f]
        out[f"ref_div_{f}_bps"] = frames[f]["ref_div_bps"].where(covered[f])
    return out


def reference_features(trades: pd.DataFrame, rules: dict, horizons: list,
                       hlabel) -> tuple[pd.DataFrame, dict]:
    """Per qualifying trade (indexed by row_id): reference executable and mid
    markouts per horizon, dislocation at the fill, lead/lag, reference quote
    age -- per feed, then combined (see _combine). Empty frame when no
    reference ticks are cached for the window."""
    feeds = configured_feeds(rules)
    summary: dict = {"feeds": {}, "combine": str(rules.get("reference_combine", "all"))}
    per: dict = {}
    for feed in feeds:
        try:
            out, s = _features_one(trades, rules, horizons, hlabel, feed)
        except Exception as error:
            out, s = pd.DataFrame(), {"label": label_of(feed), "coverage": 0,
                                      "error": f"{type(error).__name__}: {error}"}
        summary["feeds"][feed] = s
        if len(out):
            per[feed] = out
    if not per:
        return pd.DataFrame(), dict(summary, coverage=0)
    out = _combine(per, rules)
    first = next(iter(per))
    # Fields the tab already reads, now for the combined reference.
    summary["clock"] = summary["feeds"][first].get("clock", {})
    summary["labels"] = [summary["feeds"][f]["label"] for f in per]
    summary["coverage"] = round(float(out["ref_covered"].mean()), 3)
    summary["qualifying_trades"] = int(len(out))
    summary["ref_ticks"] = int(sum(summary["feeds"][f].get("ref_ticks", 0) for f in per))
    summary["basis_pairs"] = int(sum(summary["feeds"][f].get("basis_pairs", 0) for f in per))
    for col in ("ref_div_bps", "ref_lead_bps", "ref_age_ms"):
        q = _quantiles(out[col])
        if q:
            summary[col] = q
    if len(per) > 1:
        # How often the venues tell the same story at the fill, over trades
        # at least two venues cover.
        thr = float(rules.get("reference_divergence_bps", 1.0))
        multi = out["ref_n_feeds"] >= 2
        flags = pd.DataFrame({f: (out[f"ref_div_{f}_bps"] >= thr).where(out[f"ref_covered_{f}"])
                              for f in per})[multi]
        shown = flags.sum(axis=1)
        n = out.loc[multi, "ref_n_feeds"]
        divs = out.loc[multi, [f"ref_div_{f}_bps" for f in per]].corr(min_periods=20)
        pairs = divs.where(np.triu(np.ones(divs.shape, dtype=bool), 1)).stack()
        summary["agreement"] = {
            "covered_by_2plus": int(multi.sum()),
            "covered_by_all": int((out["ref_n_feeds"] == len(per)).sum()),
            "dislocation_on_any": int((shown >= 1).sum()),
            "dislocation_on_majority": int((shown > n / 2).sum()),
            "dislocation_on_all": int(((shown == n) & (shown > 0)).sum()),
            "dislocation_by_feed": {f: int(flags[f].fillna(False).astype(bool).sum()) for f in per},
            "div_bps_correlation": round(float(pairs.mean()), 3) if len(pairs) else None}
    return out, summary


def root_cause(trades: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    """s5.5 metrics at the level the data supports: the broker's AGGREGATED
    feed per server and symbol (constituent feed IDs are not recorded: MT5
    ticks carry one `Zeal Gateway` feeder, MT4 ticks carry none).
    Returns (per feed x symbol table, account -> share of its latency events
    inside an architecture-risk condition)."""
    need = {"ref_dislocation", "ref_covered", "flagged", "early_hit_any"}
    if not need <= set(trades.columns):
        return pd.DataFrame(), {}
    t = trades.loc[trades["ref_covered"].fillna(False).astype(bool)].copy()
    if not len(t):
        return pd.DataFrame(), {}
    from webapp.latency_arb import _feed_of
    t["feed"] = t["database"].map(_feed_of)
    stale_ms = float(rules.get("quote_throttle_ms", 200)) + float(rules.get("quote_age_tolerance_ms", 100))
    t["_stale_tob"] = t["ref_dislocation"].fillna(False).astype(bool)
    t["_success"] = t["_stale_tob"] & t["early_hit_any"].fillna(False).astype(bool)
    t["_impact"] = t["net_profit"].where(t["_stale_tob"], 0.0)
    t["_excess_age"] = (t["quote_age_ms"] - stale_ms).clip(lower=0)
    t["_abs_div"] = t["ref_div_bps"].abs()
    stale = t.loc[t["_stale_tob"]]
    g = t.groupby(["feed", "canonical"], observed=True)
    table = g.agg(executions=("row_id", "size"), stale_tob_events=("_stale_tob", "sum"),
                  stale_tob_success=("_success", "sum"), stale_tob_impact_usd=("_impact", "sum"),
                  feed_age_ms_p50=("quote_age_ms", "median"), excess_age_ms_p50=("_excess_age", "median"),
                  divergence_bps_p50=("_abs_div", "median"))
    table["clients"] = stale.groupby(["feed", "canonical"], observed=True)["account_key"].nunique() \
        .reindex(table.index).fillna(0).astype(int)
    table["stale_tob_rate"] = table["stale_tob_events"] / table["executions"].clip(lower=1)
    table["stale_tob_success_rate"] = table["stale_tob_success"] / table["stale_tob_events"].clip(lower=1)
    total = max(float(table["stale_tob_events"].sum()), 1.0)
    table["concentration_of_events"] = table["stale_tob_events"] / total
    pp = rules.get("profile_params") or {}
    table["architecture_risk"] = ((table["stale_tob_events"] >= int(rules.get("architecture_min_events", 10)))
                                  & (table["clients"] >= int(pp.get("replication_min_accounts", 3)))
                                  & (table["stale_tob_success_rate"] >= 0.5))
    risky = set(table.index[table["architecture_risk"]])
    fl = t.loc[t["flagged"]]
    share = {}
    if len(fl) and risky:
        inside = pd.Series([(f, c) in risky for f, c in zip(fl["feed"], fl["canonical"])], index=fl.index)
        share = inside.groupby(fl["account_key"]).mean().to_dict()
    return table.reset_index().sort_values("stale_tob_events", ascending=False), share


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "fetch":
        req = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        result = fetch([tuple(p) for p in req["pairs"]], req.get("offsets_h", [2, 3]),
                       req.get("feed", "vantage"))
        print(json.dumps(result, default=str))
