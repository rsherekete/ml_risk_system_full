#!/usr/bin/env python3
"""
Event Impact — standalone replication script
=============================================
Version 1.0 — 13 September 2026. Validated against the risk-intelligence
application on the NFP event of 4 September 2026 (XAUUSD, 13:29–13:31 London);
see README.md for the reconciliation.

Reproduces the Event Impact analysis end to end from the MT4/MT5 MySQL servers,
without the risk-intelligence application or its local data store:

  1. pulls the trades and cash movements needed for one event window straight
     from MySQL with the same queries the application's extractor uses;
  2. applies the same normalisation (lot scaling, cent-account deflation,
     canonical symbol mapping, server:login account keys);
  3. runs the same analysis logic (impacted clients, every action in the
     window, window P&L, stop-out signature, behaviour baseline and horizons,
     funding, tenure, MLTV, impact class, reaction anomaly, behaviour change,
     segmentation);
  4. writes the same Excel workbook (Summary, All results, one sheet per
     segment) and a JSON summary of the totals.

Usage
-----
    python event_impact_standalone.py --servers servers.yaml \
        --start "2026-09-04 13:29:00" --end "2026-09-04 13:31:00" --symbols XAUUSD \
        --labels event_impact_labels.csv --as-of "2026-09-10 18:07:30" \
        --out event_impact_XAUUSD_2026-09-04.xlsx

servers.yaml (credentials are yours; the file is not distributed):
    defaults: {port: 3306, user: ..., password: ...}
    servers:
      mt4_live01: {host: ...}
      mt4_live02: {host: ...}
      mt4_live03: {host: ...}
      mt4_live04: {host: ...}
      mt5_live01: {host: ...}

Inputs and conventions
----------------------
* --start / --end are Europe/London wall time (to the second); the servers'
  timestamps are treated as UTC, and the analysis converts the window to UTC.
* --labels is an optional CSV (account_key, profile) with the Anti-fraud
  behavioural label per account, exported from the application. It is context
  for segmentation only (the "toxic" test for abuse candidates); without it,
  labels are blank and that test is simply never true.
* --as-of pins the "current time" of the cash-movement data (used for tenure
  and for the withdrawal-coverage flag). Pass the application's
  cashflow_current_to value to reproduce its figures exactly; omit it to use
  the newest movement returned by the servers.
* The application's store also includes mt5_dubai_live01, which has no MySQL
  instance; this script covers the five MySQL servers only.

Requirements: pymysql, pandas, numpy, scikit-learn, xlsxwriter, pyyaml.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# pandas warns on every query that a raw DBAPI connection is "not tested";
# pymysql works correctly with read_sql, and the noise would only obscure the
# progress log.
warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")

# ------------------------------------------------------------------ constants
HORIZONS = [("immediate_4h", 4 / 24), ("d1", 1.0), ("d2", 2.0), ("d5", 5.0)]
BASELINE_DAYS = 5
POST_DAYS = 5
CASH_RETENTION_DAYS = 730          # the application's cash-movement store depth
STOPOUT_BURST_SECONDS = 120
ANOMALY_MIN_ACCOUNTS = 20
ROW_CAP = 800

SEG_LABELS = {"abuse_candidate": "Abuse candidates",
              "retain_high_value": "Retain — high value",
              "compensate_review": "Compensate / review",
              "monitor": "Monitor", "minimal": "Minimal / not impacted"}
SEG_ORDER = {"abuse_candidate": 0, "retain_high_value": 1,
             "compensate_review": 2, "monitor": 3, "minimal": 4}
TOXIC_LABELS = ("toxic_flow", "persistent_edge", "news_vol", "bonus_arb", "swap_arb")

#: MT4 balance-operation types that are NOT external funding.
INTERNAL_TYPES = ("Internal Transfer", "Credit", "Bonus", "Correction")
MT5_BALANCE_ACTION = 2

# ------------------------------------------------------------------ SQL (identical to the application's extractor)
MT4_TRADES_SQL = """
SELECT `order`, login, symbol_name AS symbol, cmd, volume, open_ts, close_ts,
       open_price, close_price, sl, tp, profit, commission, storage, state, reason
FROM orders
WHERE close_ts >= %s AND close_ts <= %s
  AND cmd IN (0, 1) AND open_ts > 0 AND open_price > 0
"""
MT5_TRADES_SQL = """
SELECT x.deal AS `order`, x.login, x.symbol, e.action AS cmd,
       x.volume AS volume, e.time AS open_ts, x.time AS close_ts,
       e.price AS open_price, x.price AS close_price,
       x.profit, x.commission, x.storage
FROM deals x
JOIN deals e ON e.position_id = x.position_id AND e.entry = 0 AND e.time <= x.time
WHERE x.entry = 1 AND x.time >= %s AND x.time <= %s
  AND x.volume > 0 AND e.price > 0
"""
MT4_CASH_SQL = "SELECT login, tm, profit, comment, type FROM balance_ops WHERE tm >= %s AND login IN ({ph})"
MT5_CASH_SQL = ("SELECT login, time AS tm, profit, comment, 'deal' AS type FROM deals "
                "WHERE action = %s AND time >= %s AND login IN ({ph})")
CENT_SQL = ("SELECT DISTINCT a.login FROM `accounts` a "
            "JOIN (SELECT DISTINCT `group` g FROM `groups` WHERE currency = 'CNT') x ON a.`group` = x.g")
CENT_FALLBACK_SQL = ("SELECT login FROM `accounts` WHERE UPPER(`group`) LIKE '%CNT%' "
                     "OR UPPER(`group`) LIKE '%CENT%'")

# ------------------------------------------------------------------ helpers
_CANON_SUFFIX = re.compile(r"(MIN|MICRO|247|PRO|ECN|RAW|[EXZCM])+$")
_CANON_ALIAS = {"SP500": "SPX500", "US500": "SPX500", "DJ30": "US30",
                "USTEC": "NAS100", "DE40": "GER40", "UKOUSD": "UKOIL"}


def canonical(symbol: str) -> str:
    """Canonical instrument name: drop a broker dot-suffix, non-alphanumerics,
    the size/ECN suffix, then apply cross-vendor aliases (XAUUSDe, XAUUSDmin,
    XAUUSD247 -> XAUUSD)."""
    base = str(symbol).split(".")[0]
    raw = re.sub(r"[^A-Z0-9]", "", base.upper())
    canon = _CANON_SUFFIX.sub("", raw) or raw
    return _CANON_ALIAS.get(canon, canon)


def ldn_to_utc(text: str) -> datetime:
    """'YYYY-MM-DD HH:MM[:SS]' as Europe/London wall time -> naive UTC."""
    text = str(text).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            local = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"unrecognised time: {text!r}")
    return local.replace(tzinfo=ZoneInfo("Europe/London")).astimezone(timezone.utc).replace(tzinfo=None)


def connect(servers_cfg: dict, name: str):
    import pymysql
    d = servers_cfg["defaults"]
    host = servers_cfg["servers"][name]["host"]
    return pymysql.connect(host=host, port=int(d.get("port", 3306)), user=d["user"],
                           password=d["password"], database=name, connect_timeout=30,
                           read_timeout=900, charset="utf8mb4")


def cent_logins(cx) -> set[int]:
    """Cent-tier logins: the account's group joined to `groups` where currency = 'CNT'
    (authoritative; group NAMES carry no reliable marker on MT4)."""
    with cx.cursor() as cur:
        try:
            cur.execute(CENT_SQL)
            return {int(r[0]) for r in cur.fetchall()}
        except Exception:
            cur.execute(CENT_FALLBACK_SQL)
            return {int(r[0]) for r in cur.fetchall()}


def deflate_cent(frame: pd.DataFrame, cents: set[int], money_cols, lot_cols=()) -> None:
    """Divide money/lots by 100 on cent-account rows, in place."""
    if frame.empty or not cents:
        return
    mask = frame["login"].astype("int64").isin(cents)
    for c in money_cols:
        if c in frame.columns:
            frame.loc[mask, c] = pd.to_numeric(frame.loc[mask, c], errors="coerce") / 100.0
    for c in lot_cols:
        if c in frame.columns:
            frame.loc[mask, c] = pd.to_numeric(frame.loc[mask, c], errors="coerce") / 100.0


# ------------------------------------------------------------------ extraction
def pull_trades(servers_cfg: dict, start: datetime, end: datetime, log) -> pd.DataFrame:
    """Closed trades from every server with close_time in [start, end], in the
    application's canonical schema, cent-deflated."""
    parts = []
    for name in servers_cfg["servers"]:
        is_mt5 = name.startswith("mt5")
        cx = connect(servers_cfg, name)
        try:
            cents = cent_logins(cx)
            if is_mt5:
                frame = pd.read_sql(MT5_TRADES_SQL, cx, params=(start, end))
            else:
                frame = pd.read_sql(MT4_TRADES_SQL, cx,
                                    params=(int(start.replace(tzinfo=timezone.utc).timestamp()),
                                            int(end.replace(tzinfo=timezone.utc).timestamp())))
        finally:
            cx.close()
        if frame.empty:
            log(f"{name}: 0 trades"); continue
        frame = frame.rename(columns={"profit": "net_profit"})
        frame["database"] = name
        if is_mt5:
            frame["volume_lots"] = pd.to_numeric(frame["volume"], errors="coerce") / 10000.0
            frame["open_time"] = pd.to_datetime(frame["open_ts"], errors="coerce")
            frame["close_time"] = pd.to_datetime(frame["close_ts"], errors="coerce")
            for c in ("sl", "tp", "reason"):
                frame[c] = pd.NA
        else:
            frame["volume_lots"] = pd.to_numeric(frame["volume"], errors="coerce") / 100.0
            frame["open_time"] = pd.to_datetime(frame["open_ts"], unit="s", errors="coerce")
            frame["close_time"] = pd.to_datetime(frame["close_ts"], unit="s", errors="coerce")
        frame["cmd"] = frame["cmd"].map({0: "buy", 1: "sell"})
        deflate_cent(frame, cents, money_cols=("net_profit", "commission", "storage"),
                     lot_cols=("volume_lots",))
        # the store de-duplicates on (server, order); a direct pull cannot contain
        # duplicates, but the same rule is applied for parity
        frame = frame.drop_duplicates(subset=["database", "order"], keep="last")
        log(f"{name}: {len(frame):,} trades ({len(cents):,} cent logins)")
        parts.append(frame[["database", "order", "login", "symbol", "cmd", "volume_lots",
                            "open_time", "close_time", "open_price", "close_price",
                            "net_profit", "commission", "storage"]])
    if not parts:
        return pd.DataFrame()
    trades = pd.concat(parts, ignore_index=True)
    trades["account_key"] = trades["database"].astype(str) + ":" + trades["login"].astype(str)
    return trades


def classify_kind(frame: pd.DataFrame) -> pd.Series:
    """deposit / withdrawal / transfer_in / transfer_out / credit / adjustment."""
    kind = pd.Series("adjustment", index=frame.index, dtype="object")
    amount = pd.to_numeric(frame["profit"], errors="coerce")
    label = frame["type"].astype("string").fillna("")
    comment = frame["comment"].astype("string").fillna("").str.upper()
    is_internal = label.isin(INTERNAL_TYPES) | comment.str.startswith(("DT-", "WT-"))
    kind = kind.mask(amount > 0, "deposit")
    kind = kind.mask(amount < 0, "withdrawal")
    kind = kind.mask(is_internal & (amount > 0), "transfer_in")
    kind = kind.mask(is_internal & (amount < 0), "transfer_out")
    kind = kind.mask(label.str.contains("Credit|Bonus", case=False, na=False), "credit")
    return kind


def pull_cashflows(servers_cfg: dict, impacted: pd.Series, since: datetime, log) -> pd.DataFrame:
    """Cash movements since `since` for the impacted logins, per server."""
    parts = []
    by_server: dict[str, list[int]] = {}
    for key in impacted:
        server, login = str(key).split(":", 1)
        by_server.setdefault(server, []).append(int(login))
    for name, logins in by_server.items():
        if name not in servers_cfg["servers"]:
            log(f"{name}: not a MySQL server, cash movements skipped"); continue
        is_mt5 = name.startswith("mt5")
        cx = connect(servers_cfg, name)
        try:
            cents = cent_logins(cx)
            frames = []
            for i in range(0, len(logins), 1000):          # chunk the IN list
                chunk = logins[i:i + 1000]
                ph = ", ".join(["%s"] * len(chunk))
                if is_mt5:
                    f = pd.read_sql(MT5_CASH_SQL.format(ph=ph), cx,
                                    params=(MT5_BALANCE_ACTION, since, *chunk))
                else:
                    f = pd.read_sql(MT4_CASH_SQL.format(ph=ph), cx, params=(since, *chunk))
                if not f.empty:
                    frames.append(f)
        finally:
            cx.close()
        if not frames:
            log(f"{name}: 0 cash movements"); continue
        frame = pd.concat(frames, ignore_index=True)
        frame["database"] = name
        frame["account_key"] = name + ":" + frame["login"].astype("int64").astype(str)
        frame["when"] = pd.to_datetime(frame["tm"], errors="coerce")
        frame["amount"] = pd.to_numeric(frame["profit"], errors="coerce")
        frame["kind"] = classify_kind(frame)
        deflate_cent(frame, cents, money_cols=("amount",))
        frame = frame.dropna(subset=["when", "amount"])
        log(f"{name}: {len(frame):,} cash movements for {len(logins):,} impacted logins")
        parts.append(frame[["account_key", "when", "amount", "kind"]])
    if not parts:
        return pd.DataFrame(columns=["account_key", "when", "amount", "kind"])
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ analysis (mirrors event_impact._analyze_inner)
def analyze(servers_cfg: dict, start: str, end: str, symbols: str, loss_limit: float,
            profit_limit: float, labels: dict, as_of: datetime | None, log,
            cash_since: datetime | None = None) -> dict:
    t0, t1 = pd.Timestamp(ldn_to_utc(start)), pd.Timestamp(ldn_to_utc(end))
    if t1 <= t0:
        raise ValueError("end must be after start")
    baseline_from = t0 - timedelta(days=BASELINE_DAYS)
    post_to = t1 + timedelta(days=POST_DAYS)
    log(f"window {t0} -> {t1} UTC | trades {baseline_from} -> {post_to}")

    frame = pull_trades(servers_cfg, baseline_from.to_pydatetime(), post_to.to_pydatetime(), log)
    if frame.empty:
        return {"rows": [], "note": "no trades in range"}

    want = {s.strip().upper() for s in symbols.split(",") if s.strip()}
    if want:
        frame["canon"] = [canonical(s).upper() for s in frame["symbol"].astype(str)]
        sym_mask = frame["canon"].isin(want)
    else:
        sym_mask = pd.Series(True, index=frame.index)

    # IMPACTED: a position on the symbols lived through any part of the window.
    hit = sym_mask & (frame["open_time"] <= t1) & (frame["close_time"] >= t0)
    impacted_accounts = frame.loc[hit, "account_key"].astype(str).unique()
    if not len(impacted_accounts):
        return {"rows": [], "note": "no impacted clients found"}
    sub = frame[frame["account_key"].isin(impacted_accounts)].copy()

    # EVERY ACTION touching the window on the affected symbols.
    opened_in = sym_mask & (frame["open_time"] >= t0) & (frame["open_time"] <= t1)
    closed_in = sym_mask & (frame["close_time"] >= t0) & (frame["close_time"] <= t1)
    held_through = sym_mask & (frame["open_time"] < t0) & (frame["close_time"] > t1)
    any_action = opened_in | closed_in | held_through

    def _count(mask):
        return frame.loc[mask].groupby("account_key").size()
    opens_cnt, closes_cnt, held_cnt, action_cnt = (_count(opened_in), _count(closed_in),
                                                   _count(held_through), _count(any_action))
    window_lots = frame.loc[opened_in | closed_in].groupby("account_key")["volume_lots"].sum()

    win = frame.loc[closed_in].copy()                # realised P&L attaches to closes
    g = win.groupby("account_key")
    window_pnl = g["net_profit"].sum()
    window_closes = g.size()

    def _burst(gr):                                    # >= 2 losing closes within any 120s
        losses = gr.loc[gr["net_profit"] < 0].sort_values("close_time")
        if len(losses) < 2:
            return False
        return bool((losses["close_time"].diff().dt.total_seconds() <= STOPOUT_BURST_SECONDS).any())
    stopped = g.apply(_burst) if len(win) else pd.Series(dtype=bool)

    # BEHAVIOUR BASELINE: the 5 days before the event (all symbols).
    pre = sub[(sub["open_time"] >= baseline_from) & (sub["open_time"] < t0)]
    pg = pre.groupby("account_key")
    base_days = max((t0 - baseline_from).days, 1)
    base_trades_pd = pg.size() / base_days
    base_volume_pd = pg["volume_lots"].sum() / base_days
    deltas = {}
    for name, days in HORIZONS:
        seg = sub[(sub["open_time"] > t1) & (sub["open_time"] <= t1 + timedelta(days=days))]
        sg = seg.groupby("account_key")
        rate = sg.size() / max(days, 1e-9)
        vol = sg["volume_lots"].sum() / max(days, 1e-9)
        deltas[f"act_{name}"] = (rate / base_trades_pd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        deltas[f"vol_{name}"] = (vol / base_volume_pd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # FUNDING: impacted accounts only; deposits and withdrawals only.
    accounts = pd.Index(sorted(impacted_accounts.astype(str)), name="account_key")
    since = cash_since or ((as_of or datetime.now(timezone.utc).replace(tzinfo=None))
                           - timedelta(days=CASH_RETENTION_DAYS))
    flows = pull_cashflows(servers_cfg, pd.Series(accounts), since, log)
    withdrawals, deposits = {}, {}
    life_dep = life_wd = pd.Series(dtype=float)
    first_flow = pd.Series(dtype="datetime64[ns]")
    cash_max = None
    if len(flows):
        flows["ts"] = pd.to_datetime(flows["when"])
        cash_max = pd.Timestamp(as_of) if as_of is not None else flows["ts"].max()
        flows = flows[flows["ts"] <= cash_max]
        kind = flows["kind"].astype(str)
        amt = pd.to_numeric(flows["amount"], errors="coerce").fillna(0.0)
        is_dep = (kind == "deposit") | ((kind == "") & (amt > 0))
        is_wd = (kind == "withdrawal") | ((kind == "") & (amt < 0))
        dep_f = flows[is_dep].assign(v=amt[is_dep].abs())
        wd_f = flows[is_wd].assign(v=amt[is_wd].abs())
        for name, days in HORIZONS:
            wm = (wd_f["ts"] > t1) & (wd_f["ts"] <= t1 + timedelta(days=days))
            dm = (dep_f["ts"] > t1) & (dep_f["ts"] <= t1 + timedelta(days=days))
            withdrawals[name] = wd_f.loc[wm].groupby("account_key")["v"].sum()
            deposits[name] = dep_f.loc[dm].groupby("account_key")["v"].sum()
        life_dep = dep_f.groupby("account_key")["v"].sum()
        life_wd = wd_f.groupby("account_key")["v"].sum()
        first_flow = flows.groupby("account_key")["ts"].min()

    table = pd.DataFrame(index=accounts)
    table["window_pnl"] = window_pnl.reindex(accounts).fillna(0.0)
    table["window_closes"] = window_closes.reindex(accounts).fillna(0).astype(int)
    table["opens_in_window"] = opens_cnt.reindex(accounts).fillna(0).astype(int)
    table["closes_in_window"] = closes_cnt.reindex(accounts).fillna(0).astype(int)
    table["held_through"] = held_cnt.reindex(accounts).fillna(0).astype(int)
    table["actions_in_window"] = action_cnt.reindex(accounts).fillna(0).astype(int)
    table["lots_in_window"] = window_lots.reindex(accounts).fillna(0.0).round(2)
    table["stopped_out"] = stopped.reindex(accounts).fillna(False).astype(bool)
    table["base_trades_pd"] = base_trades_pd.reindex(accounts).fillna(0.0)
    for col, series in deltas.items():
        table[col] = series.reindex(accounts)
    for name, _ in HORIZONS:
        wd = withdrawals.get(name, pd.Series(dtype=float)).reindex(accounts).fillna(0.0)
        dep = deposits.get(name, pd.Series(dtype=float)).reindex(accounts).fillna(0.0)
        table[f"wd_{name}"] = wd
        table[f"dep_{name}"] = dep
        table[f"net_dep_{name}"] = (dep - wd).round(2)
    table["life_deposits"] = life_dep.reindex(accounts).fillna(0.0).round(0)
    table["life_withdrawals"] = life_wd.reindex(accounts).fillna(0.0).round(0)
    table["net_deposits"] = (table["life_deposits"] - table["life_withdrawals"]).round(0)
    now_ref = cash_max if cash_max is not None else pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    ff = first_flow.reindex(accounts)
    tenure_m = ((now_ref - ff).dt.total_seconds() / (30.44 * 86400)).clip(lower=1.0)
    table["tenure_months"] = tenure_m.round(1)
    table["mltv"] = (table["net_deposits"] / tenure_m).round(0)

    def impact_class(row):
        if row["stopped_out"]:
            return "stopped_out"
        if row["window_pnl"] <= -abs(loss_limit):
            return "significant_loss"
        if row["window_pnl"] >= abs(profit_limit):
            return "significant_profit"
        return "minor"
    table["impact"] = table.apply(impact_class, axis=1)

    # REACTION ANOMALY: IsolationForest over the behaviour deltas, scaled 0-1 within this run.
    feat_cols = [c for c in table.columns if c.startswith(("act_", "vol_", "wd_"))]
    feats = table[feat_cols].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    table["anomaly"] = 0.0
    if len(feats) >= ANOMALY_MIN_ACCOUNTS:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=200, random_state=0, contamination="auto")
        table["anomaly"] = -iso.fit(feats).score_samples(feats)
        lo, hi = table["anomaly"].min(), table["anomaly"].max()
        if hi > lo:
            table["anomaly"] = (table["anomaly"] - lo) / (hi - lo)

    table["abuse_label"] = [labels.get(a, "") for a in accounts]
    table["value_volume"] = pg["volume_lots"].sum().reindex(accounts).fillna(0.0)
    mltv_hi = float(table["mltv"].quantile(0.8)) if len(table) else 0.0
    v_hi = float(table["value_volume"].quantile(0.8)) if len(table) else 0.0
    table["high_value"] = (table["value_volume"] >= v_hi) | (table["mltv"] >= max(mltv_hi, 1.0))

    def _changed(row):
        reasons = []
        if pd.notna(row.get("act_d5")) and row["act_d5"] < 0.5:
            reasons.append("trading down >50%")
        elif pd.notna(row.get("act_d5")) and row["act_d5"] > 2.0:
            reasons.append("trading up >2x")
        if row.get("wd_d5", 0) > 0:
            reasons.append(f"withdrew ${row['wd_d5']:,.0f} in 5d")
        if row.get("stopped_out"):
            reasons.append("stopped out")
        if row.get("dep_d5", 0) > 0:
            reasons.append(f"deposited ${row['dep_d5']:,.0f} in 5d")
        return "; ".join(reasons)
    table["behaviour_change"] = table.apply(_changed, axis=1)
    table["behaviour_changed"] = table["behaviour_change"].astype(bool)

    def segment(row):
        toxic = row["abuse_label"] in TOXIC_LABELS
        won = row["impact"] == "significant_profit"
        lost = row["impact"] in ("stopped_out", "significant_loss")
        withdrew = row.get("wd_d5", 0) > 0
        if won and (toxic or row["anomaly"] > 0.8 or withdrew):
            return "abuse_candidate"
        if lost:
            if row["high_value"] or (withdrew and row["mltv"] > 0):
                return "retain_high_value"
            return "compensate_review"
        if row["behaviour_changed"] and (withdrew or row["anomaly"] > 0.8 or row["act_d5"] < 0.5):
            return "monitor"
        return "minimal"
    table["segment"] = table.apply(segment, axis=1)
    table = table.sort_values(["segment", "mltv"],
                              key=lambda s: s.map(SEG_ORDER) if s.name == "segment" else -s)

    def _clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else round(float(v), 3)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v
    rows = [{"account": a, **{c: _clean(r[c]) for c in table.columns}} for a, r in table.iterrows()]
    seg_counts = table["segment"].value_counts().to_dict()
    return {
        "rows": rows[:ROW_CAP],
        "n_impacted": int(len(table)),
        "window": {"start": str(t0), "end": str(t1), "symbols": sorted(want) if want else "all"},
        "totals": {
            "window_pnl": round(float(table["window_pnl"].sum()), 2),
            "opens_in_window": int(table["opens_in_window"].sum()),
            "closes_in_window": int(table["closes_in_window"].sum()),
            "held_through": int(table["held_through"].sum()),
            "actions_in_window": int(table["actions_in_window"].sum()),
            "lots_in_window": round(float(table["lots_in_window"].sum()), 2),
            "stopped_out": int(table["stopped_out"].sum()),
            "significant_loss": int((table["impact"] == "significant_loss").sum()),
            "significant_profit": int((table["impact"] == "significant_profit").sum()),
            "withdrawn_5d": round(float(table["wd_d5"].sum()), 2),
            "deposited_5d": round(float(table["dep_d5"].sum()), 2),
            "net_deposit_5d": round(float(table["net_dep_d5"].sum()), 2),
            "net_deposits_total": round(float(table["net_deposits"].sum()), 0),
            "median_mltv": round(float(table["mltv"].median()), 0) if len(table) else 0,
            "behaviour_changed": int(table["behaviour_changed"].sum()),
        },
        "segments": {k: int(v) for k, v in seg_counts.items()},
        "cashflow_current_to": str(cash_max) if cash_max is not None else None,
        "withdrawals_covered": bool(cash_max is not None and cash_max >= (t1 + timedelta(days=POST_DAYS))),
        "notes": ["Login frequency and complaints have no data source; trading-activity frequency is the engagement proxy.",
                  f"Stop-out = >=2 losing closes within any {STOPOUT_BURST_SECONDS}s burst inside the window."],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ------------------------------------------------------------------ Excel (mirrors event_impact.build_excel)
_XL_COLS = [
    ("account", "Account (server:login)"), ("segment", "Segment"), ("impact", "Impact"),
    ("window_pnl", "Window P&L ($)"), ("actions_in_window", "Actions in window"),
    ("opens_in_window", "Opens in window"), ("closes_in_window", "Closes in window"),
    ("held_through", "Held through"), ("lots_in_window", "Lots traded in window"),
    ("stopped_out", "Stopped out"), ("behaviour_change", "Behaviour change (post-event)"),
    ("abuse_label", "Behavioural label"), ("anomaly", "Reaction anomaly (0-1)"),
    ("base_trades_pd", "Baseline trades/day"),
    ("act_immediate_4h", "Activity x (4h)"), ("act_d1", "Activity x (1d)"),
    ("act_d2", "Activity x (2d)"), ("act_d5", "Activity x (5d)"), ("vol_d5", "Volume x (5d)"),
    ("dep_d5", "Deposited 5d ($)"), ("wd_d5", "Withdrawn 5d ($)"), ("net_dep_d5", "Net deposit 5d ($)"),
    ("mltv", "MLTV — net deposits / month ($)"), ("tenure_months", "Tenure (months)"),
    ("net_deposits", "Net deposits lifetime ($)"), ("life_deposits", "Lifetime deposits ($)"),
    ("life_withdrawals", "Lifetime withdrawals ($)"), ("value_volume", "Value proxy (baseline vol)"),
]


def build_excel(data: dict, path: str) -> None:
    import xlsxwriter
    rows = data.get("rows", [])
    wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    ink, f = "#0d1220", {}
    f["title"] = wb.add_format({"bold": True, "font_size": 16, "font_color": ink})
    f["sub"] = wb.add_format({"font_size": 10, "font_color": "#55637d"})
    f["hdr"] = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#1b2438", "border": 1,
                              "align": "center", "valign": "vcenter", "text_wrap": True})
    f["txt"] = wb.add_format({"border": 1})
    f["num"] = wb.add_format({"border": 1, "num_format": "#,##0.00"})
    f["int"] = wb.add_format({"border": 1, "num_format": "#,##0"})
    f["usd"] = wb.add_format({"border": 1, "num_format": "$#,##0;[Red]-$#,##0"})
    f["kpi"] = wb.add_format({"bold": True, "font_size": 20, "font_color": ink})
    f["kpi_l"] = wb.add_format({"font_size": 9, "font_color": "#55637d"})
    f["pos"] = wb.add_format({"border": 1, "num_format": "$#,##0", "bg_color": "#e8fff5"})
    f["neg"] = wb.add_format({"border": 1, "num_format": "[Red]-$#,##0", "bg_color": "#fff0f2"})

    def fmt_for(col):
        if col in ("window_pnl", "wd_d5", "dep_d5", "net_dep_d5", "mltv", "net_deposits",
                   "life_deposits", "life_withdrawals"):
            return f["usd"]
        if col in ("opens_in_window", "closes_in_window", "held_through", "actions_in_window", "window_closes"):
            return f["int"]
        if col in ("account", "segment", "impact", "abuse_label", "stopped_out", "behaviour_change"):
            return f["txt"]
        return f["num"]

    def write_table(ws, rs, startrow=0):
        for c, (_, header) in enumerate(_XL_COLS):
            ws.write(startrow, c, header, f["hdr"])
        ws.set_row(startrow, 30)
        for i, r in enumerate(rs):
            for c, (key, _) in enumerate(_XL_COLS):
                v = r.get(key); fm = fmt_for(key)
                if key == "window_pnl" and isinstance(v, (int, float)):
                    fm = f["pos"] if v >= 0 else f["neg"]
                if key == "stopped_out":
                    v = "YES" if v else ""
                ws.write(startrow + 1 + i, c, v if v is not None else "", fm)
        ws.freeze_panes(startrow + 1, 1)
        widths = [24, 20, 17, 15, 13, 13, 13, 12, 14, 11, 30, 16, 14, 14, 12, 12, 12, 12, 12, 14, 14, 14, 22, 15, 18, 18, 18, 16]
        for c, w in enumerate(widths[:len(_XL_COLS)]):
            ws.set_column(c, c, w)
        ws.autofilter(startrow, 0, startrow + len(rs), len(_XL_COLS) - 1)

    win, tot, segs = data.get("window", {}), data.get("totals", {}), data.get("segments", {})
    ws = wb.add_worksheet("Summary")
    ws.set_column(0, 0, 3); ws.set_column(1, 6, 18); ws.hide_gridlines(2)
    ws.write(1, 1, "Event Impact Report", f["title"])
    syms = win.get("symbols"); syms = ", ".join(syms) if isinstance(syms, list) else syms
    ws.write(2, 1, f"{syms}  ·  {win.get('start', '')} → {win.get('end', '')} (UTC)", f["sub"])
    ws.write(3, 1, f"Generated {data.get('generated_at', '')}", f["sub"])
    kpis = [("Impacted clients", data.get("n_impacted", 0), f["int"]),
            ("Window P&L (client)", tot.get("window_pnl", 0), None),
            ("Actions in window", tot.get("actions_in_window", 0), f["int"]),
            ("Stopped out", tot.get("stopped_out", 0), f["int"]),
            ("Deposited 5d", tot.get("deposited_5d", 0), None),
            ("Withdrawn 5d", tot.get("withdrawn_5d", 0), None),
            ("Net deposit 5d", tot.get("net_deposit_5d", 0), None),
            ("Behaviour changed", tot.get("behaviour_changed", 0), f["int"]),
            ("Sig. losses", tot.get("significant_loss", 0), f["int"]),
            ("Sig. profits", tot.get("significant_profit", 0), f["int"]),
            ("Median MLTV ($/mo)", tot.get("median_mltv", 0), None),
            ("Net deposits (lifetime)", tot.get("net_deposits_total", 0), None)]
    r0 = 5
    for i, (label, val, fm) in enumerate(kpis):
        col, row = 1 + (i % 4), r0 + (i // 4) * 3
        ws.write(row, col, val, fm or f["kpi"]); ws.write(row + 1, col, label, f["kpi_l"])
    sr = r0 + 11
    ws.write(sr, 1, "Segment", f["hdr"]); ws.write(sr, 2, "Clients", f["hdr"]); ws.set_column(1, 1, 22)
    for j, (k, lab) in enumerate(SEG_LABELS.items()):
        ws.write(sr + 1 + j, 1, lab, f["txt"]); ws.write(sr + 1 + j, 2, segs.get(k, 0), f["int"])
    write_table(wb.add_worksheet("All results"), rows)
    for k, lab in SEG_LABELS.items():
        rs = [r for r in rows if r.get("segment") == k]
        if rs:
            write_table(wb.add_worksheet(lab[:28].replace("/", "-")), rs)
    wb.close()


# ------------------------------------------------------------------ main
def main():
    import yaml
    ap = argparse.ArgumentParser(description="Event Impact — standalone replication")
    ap.add_argument("--servers", required=True, help="YAML with defaults{port,user,password} and servers{name:{host}}")
    ap.add_argument("--start", required=True, help="window start, Europe/London wall time 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--end", required=True, help="window end, Europe/London wall time")
    ap.add_argument("--symbols", default="", help="canonical symbols, comma-separated (empty = all)")
    ap.add_argument("--loss", type=float, default=500.0, help="significant-loss threshold, $")
    ap.add_argument("--profit", type=float, default=500.0, help="significant-profit threshold, $")
    ap.add_argument("--labels", default="", help="CSV account_key,profile (behavioural labels)")
    ap.add_argument("--as-of", default="", help="cash-movement 'current to' timestamp (UTC) to pin tenure/coverage")
    ap.add_argument("--cash-since", default="", help="earliest cash movement to include (UTC); default = as-of minus 730 days")
    ap.add_argument("--out", default="", help="Excel output path (default: event_impact_<symbols>_<date>.xlsx)")
    ap.add_argument("--json", default="", help="JSON summary output path")
    args = ap.parse_args()

    servers_cfg = yaml.safe_load(open(args.servers, encoding="utf-8"))
    labels = {}
    if args.labels:
        lab = pd.read_csv(args.labels, dtype=str).fillna("")
        labels = dict(zip(lab["account_key"], lab["profile"]))
    as_of = pd.Timestamp(args.as_of).to_pydatetime() if args.as_of else None
    cash_since = pd.Timestamp(args.cash_since).to_pydatetime() if args.cash_since else None
    log = lambda m: print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

    data = analyze(servers_cfg, args.start, args.end, args.symbols, args.loss, args.profit, labels, as_of, log,
                   cash_since=cash_since)
    if not data.get("rows"):
        print("no result:", data.get("note")); return
    win = data["window"]; sy = win.get("symbols"); sy = "_".join(sy) if isinstance(sy, list) else "all"
    out = args.out or f"event_impact_{sy}_{str(win.get('start', ''))[:10]}.xlsx"
    build_excel(data, out)
    summary = {k: v for k, v in data.items() if k != "rows"}
    summary["rows_in_workbook"] = len(data["rows"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
    log(f"impacted {data['n_impacted']:,} | window P&L ${data['totals']['window_pnl']:,.0f} | "
        f"segments {data['segments']} | workbook -> {out}")


if __name__ == "__main__":
    main()
