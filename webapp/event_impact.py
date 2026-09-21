"""Event Impact analyzer: who a market event hurt, helped, and what to do.

For a gapping / volatility / pricing-error window the module answers, per
client:

- WAS THIS CLIENT IMPACTED: any position open through or closed inside the
  event window on the affected symbols.
- MONETARY IMPACT: realized P&L of trades closed inside the window, with a
  stop-out heuristic (burst of loss-closes inside the window) and
  significant-loss / significant-profit classification.
- BEHAVIOUR CHANGE at horizons (immediate ~4h, 1d, 2d, 5d after): trading
  activity and volume vs the client's own pre-event baseline, and withdrawal
  amounts from the cashflow store. (Complaints and login frequency have no
  data source in this stack; trading-activity frequency stands in as the
  engagement proxy and the UI says so.)
- ML SEGMENTATION: an IsolationForest anomaly score over the behaviour-delta
  features, combined with the AntiFraud behavioural labels, buckets everyone
  into: abuse_candidate / retain_high_value / compensate_review / monitor /
  minimal -- the filter the business asked for.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_CACHE: dict = {}
_LOCK = threading.Lock()

from pathlib import Path
LAST_PATH = Path(__file__).resolve().parent / "artifacts" / "event_impact_last.json"


def last() -> dict:
    """The most recent analysis, served instantly on tab open."""
    import json
    with _LOCK:
        if _CACHE.get("data"):
            return _CACHE["data"]
    try:
        disk = json.loads(LAST_PATH.read_text(encoding="utf-8"))
        disk["from_disk"] = True
        return disk
    except Exception:
        return {"rows": [], "note": "no analysis run yet"}

HORIZONS = [("immediate_4h", 4 / 24), ("d1", 1.0), ("d2", 2.0), ("d5", 5.0)]


def last_nfp() -> dict:
    """The predefined default: the most recent Non-Farm Payrolls session
    (first Friday of the month). NFP is 8:30 AM ET, which is 13:30 LONDON
    wall time year-round -- exactly the tab's input convention -- and the
    analyzer converts that to ~12:30 UTC. A ±1-minute XAUUSD window as the
    user specified (13:29:00 -> 13:31:00 London)."""
    # 'today' is the calendar day; NFP only happens on a past first Friday.
    today = datetime.utcnow().date()

    def first_friday(y, m):
        d = datetime(y, m, 1)
        return (d + timedelta(days=(4 - d.weekday()) % 7)).date()  # Fri=4

    ff = first_friday(today.year, today.month)
    if ff >= today:                          # this month's NFP hasn't happened
        pm = datetime(today.year, today.month, 1).date() - timedelta(days=1)
        ff = first_friday(pm.year, pm.month)
    d = ff.strftime("%Y-%m-%d")
    return {
        "start": f"{d} 13:29:00", "end": f"{d} 13:31:00",   # London wall time
        "symbols": "XAUUSD",
        "label": f"Last NFP — {ff:%a %d %b} 13:30 London (12:30 UTC release)",
    }


#: Named windows behind the tab's one-click buttons (London wall time).
PRESETS = {
    "nfp_2026_09_04": {"start": "2026-09-04 13:29:00", "end": "2026-09-04 13:31:00", "symbols": "XAUUSD",
                       "label": "NFP — Fri 04 Sep 2026 13:30 London (12:30 UTC release), XAUUSD ±1 min"},
}


def preset(name: str) -> dict:
    """A named window: the fixed 4 September NFP, or the calendar's most
    recent NFP (first Friday of the month, 13:30 London)."""
    if name == "last_nfp":
        return last_nfp()
    p = PRESETS.get(name)
    return dict(p, name=name) if p else {"error": f"unknown preset {name}"}


def detect_events(days: int = 7, top: int = 12) -> list[dict]:
    """Auto-detected candidate events from the tick tape: largest 1-minute
    absolute moves per canonical symbol inside quote retention."""
    try:
        from webapp.kafka_service import shared_cursor as _store
        with _store() as cx:
            frame = cx.execute("""
                WITH q AS (
                    SELECT canonical, event_time,
                           COALESCE(mid, (bid + ask) / 2) AS px
                    FROM quotes
                    WHERE event_time > now() - INTERVAL 7 DAY),
                m AS (
                    SELECT canonical,
                           date_trunc('minute', event_time) AS minute,
                           max(px) AS hi, min(px) AS lo, avg(px) AS mid
                    FROM q WHERE px > 0
                    GROUP BY 1, 2)
                SELECT canonical, minute,
                       (hi - lo) / mid * 1e4 AS range_bps
                FROM m WHERE mid > 0
                ORDER BY range_bps DESC LIMIT 400
            """).df()
        if not len(frame):
            return []
        frame["minute"] = pd.to_datetime(frame["minute"])
        frame = (frame.sort_values("range_bps", ascending=False)
                 .groupby("canonical", observed=True).head(2)
                 .sort_values("range_bps", ascending=False).head(top))
        from zoneinfo import ZoneInfo
        from datetime import timezone as _tz
        ldn = ZoneInfo("Europe/London")
        def _l(ts):
            return (ts.tz_localize(_tz.utc).astimezone(ldn)
                    .strftime("%Y-%m-%d %H:%M:%S"))
        return [{"symbol": r.canonical,
                 "start": _l(r.minute - pd.Timedelta(minutes=2)),
                 "end": _l(r.minute + pd.Timedelta(minutes=5)),
                 "range_bps": round(float(r.range_bps), 1)}
                for r in frame.itertuples(index=False)]
    except Exception:
        return []


def _disk_data_for(key: str, start: str, end: str, symbols: str) -> dict | None:
    """The last analysis saved to disk, if it IS this window: matched by the
    cache key it carries, or (files written before keys were stored) by the
    UTC window and symbols. Survives a server restart, which empties the
    in-memory cache -- without this the Excel link recomputed for minutes
    and the browser gave up on the download."""
    import json
    try:
        disk = json.loads(LAST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if disk.get("cache_key") == key:
        return disk
    if disk.get("cache_key"):
        return None
    try:
        from webapp.econ_calendar import ldn_to_utc
        win = disk.get("window") or {}
        want = sorted({s.strip().upper() for s in symbols.split(",") if s.strip()}) or "all"
        got = win.get("symbols")
        got = sorted(s.upper() for s in got) if isinstance(got, list) else (got or "all")
        if (str(pd.Timestamp(ldn_to_utc(start))) == str(win.get("start"))
                and str(pd.Timestamp(ldn_to_utc(end))) == str(win.get("end")) and want == got):
            return disk
    except Exception:
        return None
    return None


def analyze(start: str, end: str, symbols: str = "",
            loss_limit: float = 500.0, profit_limit: float = 500.0,
            refresh: bool = False, cached_only: bool = False) -> dict:
    """`cached_only`: never compute -- serve the memory or disk copy of this
    window, else {"error": "not cached"} (the Excel download uses it, so a
    click can never hang for the minutes a fresh analysis takes)."""
    key = f"{start}|{end}|{symbols}|{loss_limit}|{profit_limit}"
    with _LOCK:
        if not refresh and _CACHE.get("key") == key:
            return _CACHE["data"]
    if not refresh:
        disk = _disk_data_for(key, start, end, symbols)
        if disk is not None:
            with _LOCK:
                _CACHE["key"], _CACHE["data"] = key, disk
            return disk
    if cached_only:
        return {"error": "not cached", "rows": []}
    try:
        data = _analyze_inner(start, end, symbols, loss_limit, profit_limit)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "rows": []}
    data["cache_key"] = key
    with _LOCK:
        _CACHE["key"], _CACHE["data"] = key, data
    try:
        import json
        LAST_PATH.write_text(json.dumps(data, default=str), encoding="utf-8")
    except Exception:
        pass
    return data


def _canon(sym: str) -> str:
    from webapp.trade_feed import _canonical
    return _canonical(str(sym)) or str(sym)


SEG_LABELS = {"abuse_candidate": "Abuse candidates",
              "retain_high_value": "Retain — high value",
              "compensate_review": "Compensate / review",
              "monitor": "Monitor", "minimal": "Minimal / not impacted"}

#: Column order + friendly headers for the Excel export.
_XL_COLS = [
    ("primary_account", "Primary account (server:login)"),
    ("account", "Client account / sub-account (server:login)"),
    ("is_primary", "Is primary"),
    ("subaccounts", "Sub-accounts (linked, n)"),
    ("accounts_in_window", "Linked accounts active in window"),
    ("linked_accounts", "Linked accounts"),
    ("link_source", "Link source"),
    ("segment", "Segment"), ("impact", "Impact"),
    ("client_class", "Client class"),
    ("abuse_reasons", "Abuse flags — why"),
    ("window_pnl", "Window P&L ($)"),
    ("net_revenue", "Net revenue lifetime — after rebates ($)"),
    ("gross_revenue", "Gross revenue lifetime — before rebates ($)"),
    ("notional_usd", "USD notional traded, lifetime ($)"),
    ("notional_usd_monthly", "USD notional traded / month ($)"),
    ("class_notional", "Class by monthly USD notional (percentile)"),
    ("life_rebates", "Rebates lifetime ($)"),
    ("rebates_monthly", "Rebates / month ($)"),
    ("rebate_source", "Rebate source"),
    ("client_notional_usd", "Client USD notional lifetime, all linked accounts ($)"),
    ("client_net_deposits", "Client net deposits, all linked accounts ($)"),
    ("trades_lifetime", "Trades lifetime"),
    ("lots_lifetime", "Lots lifetime"),
    ("actions_in_window", "Actions in window"),
    ("opens_in_window", "Opens in window"),
    ("closes_in_window", "Closes in window"),
    ("held_through", "Held through"),
    ("lots_in_window", "Lots traded in window"),
    ("stopped_out", "Stopped out"),
    ("behaviour_change", "Behaviour change (post-event)"),
    ("abuse_label", "Behavioural label"),
    ("anomaly", "Reaction anomaly (0-1)"),
    ("base_trades_pd", "Baseline trades/day"),
    ("act_immediate_4h", "Activity x (4h)"), ("act_d1", "Activity x (1d)"),
    ("act_d2", "Activity x (2d)"), ("act_d5", "Activity x (5d)"),
    ("vol_d5", "Volume x (5d)"),
    ("dep_d5", "Deposited 5d ($)"), ("wd_d5", "Withdrawn 5d ($)"),
    ("net_dep_d5", "Net deposit 5d ($)"),
    ("mltv", "MLTV — net deposits / month ($)"),
    ("tenure_months", "Tenure (months)"),
    ("net_deposits", "Net deposits lifetime, rebates excluded ($)"),
    ("life_deposits", "Lifetime deposits ($)"),
    ("life_withdrawals", "Lifetime withdrawals ($)"),
    ("value_volume", "Value proxy (baseline vol)"),
    # --- classification components (the business table) ---
    ("class_mltv", "Class by MLTV"),
    ("class_revenue", "Class by monthly net revenue"),
    ("monthly_revenue", "Monthly net revenue — (net dep − equity) / active month ($)"),
    ("combined_ratio", "Combined ratio (revenue / MLTV)"),
    ("equity", "Equity now ($)"),
    ("equity_source", "Equity source"),
    ("balance", "Balance now ($)"),
    ("active_life_months", "Client active life (months)"),
    # --- abuse test components ---
    ("desk_flag", "Desk flag (platform comment)"),
    ("abusedb_flag", "Desk abuse register (AbuseDb)"),
    ("abusedb_saved_usd", "AbuseDb amount saved ($)"),
    ("account_leverage", "Account leverage (1:x)"),
    ("effective_leverage_t0", "Effective leverage at window start, net (x)"),
    ("notional_at_t0", "Net open notional at window start ($)"),
    ("gross_leverage_t0", "Gross leverage at window start (x)"),
    ("gross_notional_at_t0", "Gross open notional at window start ($)"),
    ("positions_at_t0", "Positions open at window start"),
    ("balance_at_t0_est", "Balance at window start, est. ($)"),
    ("hedged_at_t0", "Hedged into window"),
    ("hedge_unwind", "Hedge unwind in window"),
    ("hedge_detail", "Hedge detail"),
    ("net_lots_t0", "Net lots at window start (event symbols)"),
    ("net_lots_after", "Net lots carried out of window"),
    ("close_by_closes", "Close-by closes in window"),
    ("close_by_lots_in_window", "Close-by lots netted in window"),
    ("leverage_after", "Leverage after window (x)"),
    ("fills_in_window", "Fills in window"),
    ("favourable_fills", "Fills better than market"),
    ("max_favourable_bps", "Max advantage (bps)"),
    ("advantage_usd", "Execution advantage ($)"),
    ("stop_gap_fills", "Stops filled beyond trigger"),
    ("execution_detail", "Execution detail"),
    ("group", "Platform group"),
]


def build_excel(data: dict) -> bytes:
    """A polished multi-tab workbook: Summary, All results, one tab per
    segment. Returns the .xlsx bytes."""
    import io
    import xlsxwriter
    rows = data.get("rows", [])
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True,
                                   "nan_inf_to_errors": True})
    # ---- formats ----
    ink = "#0d1220"; accent = "#22d3ee"
    f_title = wb.add_format({"bold": True, "font_size": 16, "font_color": ink})
    f_sub = wb.add_format({"font_size": 10, "font_color": "#55637d"})
    f_hdr = wb.add_format({"bold": True, "font_color": "white",
                           "bg_color": "#1b2438", "border": 1,
                           "align": "center", "valign": "vcenter",
                           "text_wrap": True})
    f_txt = wb.add_format({"border": 1})
    f_num = wb.add_format({"border": 1, "num_format": "#,##0.00"})
    f_int = wb.add_format({"border": 1, "num_format": "#,##0"})
    f_usd = wb.add_format({"border": 1, "num_format": "$#,##0;[Red]-$#,##0"})
    f_kpi = wb.add_format({"bold": True, "font_size": 20, "font_color": ink})
    f_kpi_l = wb.add_format({"font_size": 9, "font_color": "#55637d"})
    f_pos = wb.add_format({"border": 1, "num_format": "$#,##0",
                           "bg_color": "#e8fff5"})
    f_neg = wb.add_format({"border": 1, "num_format": "[Red]-$#,##0",
                           "bg_color": "#fff0f2"})

    USD_COLS = ("window_pnl", "wd_d5", "dep_d5", "net_dep_d5", "mltv", "net_deposits", "life_deposits",
                "life_withdrawals", "monthly_revenue", "equity", "balance", "abusedb_saved_usd",
                "notional_at_t0", "gross_notional_at_t0", "balance_at_t0_est", "advantage_usd",
                "net_revenue", "gross_revenue", "notional_usd", "notional_usd_monthly", "life_rebates",
                "rebates_monthly", "client_notional_usd", "client_net_deposits")
    INT_COLS = ("opens_in_window", "closes_in_window", "held_through", "actions_in_window", "window_closes",
                "fills_in_window", "favourable_fills", "stop_gap_fills", "account_leverage", "positions_at_t0",
                "subaccounts", "accounts_in_window", "trades_lifetime")
    TXT_COLS = ("account", "primary_account", "is_primary", "segment", "impact", "abuse_label", "stopped_out",
                "behaviour_change", "client_class", "abuse_reasons", "class_mltv", "class_revenue", "class_notional",
                "equity_source", "desk_flag", "abusedb_flag", "hedged_at_t0", "hedge_unwind", "hedge_detail",
                "execution_detail", "group", "linked_accounts", "link_source", "rebate_source")
    BOOL_COLS = ("stopped_out", "hedged_at_t0", "hedge_unwind", "is_primary")
    WIDE = {"account": 26, "primary_account": 24, "segment": 20, "impact": 17, "client_class": 12, "abuse_reasons": 60,
            "behaviour_change": 30, "hedge_detail": 50, "execution_detail": 60, "group": 26,
            "desk_flag": 22, "abusedb_flag": 40, "linked_accounts": 44, "link_source": 18, "rebate_source": 30}

    def fmt_for(col):
        if col in USD_COLS:
            return f_usd
        if col in INT_COLS:
            return f_int
        if col in TXT_COLS:
            return f_txt
        return f_num

    def write_table(ws, rs, startrow=0, cols=None):
        cols = cols or _XL_COLS
        for c, (_, header) in enumerate(cols):
            ws.write(startrow, c, header, f_hdr)
        ws.set_row(startrow, 30)
        for i, r in enumerate(rs):
            for c, (key, _) in enumerate(cols):
                v = r.get(key)
                f = fmt_for(key)
                if key == "window_pnl" and isinstance(v, (int, float)):
                    f = f_pos if v >= 0 else f_neg
                if key in BOOL_COLS:
                    v = "YES" if v else ""
                ws.write(startrow + 1 + i, c, v if v is not None else "", f)
        ws.freeze_panes(startrow + 1, 1)
        for c, (key, _) in enumerate(cols):
            ws.set_column(c, c, WIDE.get(key, 15))
        ws.autofilter(startrow, 0, startrow + len(rs), len(cols) - 1)

    #: The two new tabs lead with what the reader needs to see first.
    _ABUSE_FIRST = ["primary_account", "account", "is_primary", "subaccounts", "accounts_in_window", "client_class",
                    "abuse_reasons", "window_pnl", "net_revenue", "notional_usd_monthly", "class_notional",
                    "life_rebates", "rebate_source", "desk_flag", "abusedb_flag",
                    "abusedb_saved_usd", "effective_leverage_t0", "hedge_unwind", "hedge_detail",
                    "favourable_fills", "fills_in_window", "max_favourable_bps", "advantage_usd",
                    "stop_gap_fills", "execution_detail", "account_leverage", "notional_at_t0",
                    "balance_at_t0_est", "net_lots_after", "leverage_after"]
    _VIP_FIRST = ["primary_account", "account", "is_primary", "subaccounts", "accounts_in_window", "client_class",
                  "window_pnl", "impact", "stopped_out", "class_mltv", "class_revenue", "class_notional", "mltv",
                  "monthly_revenue", "notional_usd_monthly", "net_revenue", "gross_revenue", "combined_ratio",
                  "net_deposits", "life_rebates", "rebates_monthly", "rebate_source", "notional_usd", "equity",
                  "active_life_months", "wd_d5", "dep_d5", "behaviour_change", "actions_in_window",
                  "lots_in_window", "abuse_label", "linked_accounts", "client_notional_usd", "client_net_deposits"]
    _BY_KEY = dict(_XL_COLS)

    def ordered(first):
        head = [(k, _BY_KEY[k]) for k in first if k in _BY_KEY]
        return head + [(k, h) for k, h in _XL_COLS if k not in first]

    win = data.get("window", {})
    tot = data.get("totals", {})
    segs = data.get("segments", {})

    # ---- Summary ----
    ws = wb.add_worksheet("Summary")
    ws.set_column(0, 0, 3); ws.set_column(1, 6, 18)
    ws.hide_gridlines(2)
    ws.write(1, 1, "Event Impact Report", f_title)
    syms = win.get("symbols"); syms = ", ".join(syms) if isinstance(syms, list) else syms
    ws.write(2, 1, f"{syms}  ·  {win.get('start','')} → {win.get('end','')} (UTC)", f_sub)
    ws.write(3, 1, f"Generated {data.get('generated_at','')}", f_sub)
    kpis = [("Impacted clients", data.get("n_impacted", 0), f_int),
            ("Window P&L (client)", tot.get("window_pnl", 0), None),
            ("Actions in window", tot.get("actions_in_window", 0), f_int),
            ("Stopped out", tot.get("stopped_out", 0), f_int),
            ("Deposited 5d", tot.get("deposited_5d", 0), None),
            ("Withdrawn 5d", tot.get("withdrawn_5d", 0), None),
            ("Net deposit 5d", tot.get("net_deposit_5d", 0), None),
            ("Behaviour changed", tot.get("behaviour_changed", 0), f_int),
            ("Sig. losses", tot.get("significant_loss", 0), f_int),
            ("Sig. profits", tot.get("significant_profit", 0), f_int),
            ("Median MLTV ($/mo)", tot.get("median_mltv", 0), None),
            ("Net deposits (lifetime)", tot.get("net_deposits_total", 0), None)]
    r0 = 5
    for i, (label, val, f) in enumerate(kpis):
        col = 1 + (i % 4) * 1
        row = r0 + (i // 4) * 3
        ws.write(row, col, val, f or f_kpi)
        ws.write(row + 1, col, label, f_kpi_l)
    # segment breakdown table (below the 3 KPI rows)
    sr = r0 + 11
    ws.write(sr, 1, "Segment", f_hdr); ws.write(sr, 2, "Clients", f_hdr)
    ws.set_column(1, 1, 22)
    for j, (k, lab) in enumerate(SEG_LABELS.items()):
        ws.write(sr + 1 + j, 1, lab, f_txt)
        ws.write(sr + 1 + j, 2, segs.get(k, 0), f_int)
    # abuse tests and client classes, side by side
    ac = data.get("abuse_counts", {}) or {}
    ws.write(sr, 4, "Abuse test", f_hdr); ws.write(sr, 5, "Clients", f_hdr)
    for j, (k, lab) in enumerate([("desk_flag", "0 · desk flag (platform)"), ("abusedb", "0 · desk register (AbuseDb)"),
                                  ("high_leverage", "1a · high effective leverage"), ("hedge_unwind", "1b · hedge unwind"),
                                  ("favourable_execution", "2 · favourable execution"), ("abusive", "Any abuse flag"),
                                  ("vip_impacted", "VIP/High/Mid hurt, no flag")]):
        ws.write(sr + 1 + j, 4, lab, f_txt); ws.write(sr + 1 + j, 5, ac.get(k, 0), f_int)
    ws.set_column(4, 4, 30)
    classes = data.get("classes", {}) or {}
    ws.write(sr, 7, "Client class", f_hdr); ws.write(sr, 8, "Clients", f_hdr)
    for j, k in enumerate(("VIP", "High", "Mid", "Low", "Micro")):
        ws.write(sr + 1 + j, 7, k, f_txt); ws.write(sr + 1 + j, 8, classes.get(k, 0), f_int)
    # definitions: what each flag and class means
    dr = sr + 10
    ws.write(dr, 1, "Definitions", f_hdr)
    f_wrap = wb.add_format({"text_wrap": True, "valign": "top", "font_size": 9})
    for j, line in enumerate(data.get("abuse_definitions", []) or []):
        ws.merge_range(dr + 1 + j, 1, dr + 1 + j, 8, line, f_wrap)
        ws.set_row(dr + 1 + j, 42)
    for j, line in enumerate(data.get("abuse_notes", []) or []):
        ws.write(dr + 2 + len(data.get("abuse_definitions", []) or []) + j, 1, f"Data note: {line}", f_sub)

    # ---- the two tabs the review starts from ----
    write_table(wb.add_worksheet("Abusive clients"), data.get("abusive_rows", []) or [], cols=ordered(_ABUSE_FIRST))
    write_table(wb.add_worksheet("VIP impacted (non-abusive)"), data.get("vip_rows", []) or [], cols=ordered(_VIP_FIRST))

    # ---- All results ----
    write_table(wb.add_worksheet("All results"), rows)

    # ---- one tab per segment (logical order) ----
    for k, lab in SEG_LABELS.items():
        rs = [r for r in rows if r.get("segment") == k]
        if not rs:
            continue
        name = lab[:28].replace("/", "-")
        write_table(wb.add_worksheet(name), rs)

    wb.close()
    return buf.getvalue()


#: How the per-account table collapses to one row per client.
_SUM_COLS = ("window_pnl", "window_closes", "opens_in_window", "closes_in_window", "held_through",
             "actions_in_window", "close_by_closes", "lots_in_window", "base_trades_pd", "value_volume",
             "abusedb_saved_usd", "notional_at_t0", "gross_notional_at_t0", "balance_at_t0_est",
             "positions_at_t0", "fills_in_window", "favourable_fills", "stop_gap_fills", "advantage_usd",
             "net_lots_t0", "net_lots_after", "long_lots_t0", "short_lots_t0", "long_lots_after",
             "short_lots_after", "close_by_lots_in_window")
_ANY_COLS = ("stopped_out", "hedged_at_t0", "hedge_unwind", "abusive")
_MAX_COLS = ("effective_leverage_t0", "gross_leverage_t0", "leverage_after", "max_favourable_bps",
             "account_leverage")
_TEXT_COLS = ("desk_flag", "abusedb_flag", "hedge_detail", "execution_detail", "abuse_reasons", "group",
              "status", "abuse_label")
_PLAIN_TEXT = ("group", "status", "abuse_label")


def _aggregate_clients(table: pd.DataFrame, links: pd.DataFrame, life: pd.DataFrame) -> pd.DataFrame:
    """One row per CLIENT from the per-account table.

    Window figures are summed over the client's impacted accounts (flags
    OR'd, leverages at their worst, activity ratios weighted by baseline
    activity, text joined with the login in front when more than one account
    contributed). Lifetime figures come from `life` over EVERY linked account,
    impacted or not. The row is keyed by the PRIMARY account: the impacted
    account with the largest absolute window P&L."""
    client_of = links["client_key"].reindex(table.index)
    client_of = client_of.where(client_of.notna(), pd.Series(table.index, index=table.index))
    member_lists = links["members"].reindex(table.index)
    horizon_cols = [c for c in table.columns if c.startswith(("wd_", "dep_", "net_dep_"))]
    ratio_cols = [c for c in table.columns if c.startswith(("act_", "vol_"))]
    has_life = {c for c in life.columns}
    rows = []
    for ck, idx in table.groupby(client_of, sort=False).groups.items():
        part = table.loc[idx]
        primary = part["window_pnl"].abs().sort_values(ascending=False, kind="mergesort").index[0]
        mem = set(idx)
        for lst in member_lists.loc[idx]:
            if isinstance(lst, (list, tuple, set)):
                mem |= set(lst)
        mem = sorted(mem)
        multi = len(idx) > 1
        row = {"account": primary, "client_key": ck, "subaccounts": len(mem) - 1,
               "accounts_in_window": int(len(idx)),
               "linked_accounts": "; ".join(m for m in mem if m != primary),
               "link_source": next((s for s in links["link_source"].reindex(idx).fillna("").astype(str) if s), "")}
        for c in _SUM_COLS + tuple(horizon_cols):
            if c in part.columns:
                row[c] = float(pd.to_numeric(part[c], errors="coerce").fillna(0.0).sum())
        for c in _ANY_COLS:
            if c in part.columns:
                row[c] = bool(part[c].fillna(False).astype(bool).any())
        for c in _MAX_COLS:
            if c in part.columns:
                row[c] = float(pd.to_numeric(part[c], errors="coerce").max())
        w = pd.to_numeric(part["base_trades_pd"], errors="coerce").fillna(0.0) if "base_trades_pd" in part.columns \
            else pd.Series(1.0, index=part.index)
        for c in ratio_cols:
            v = pd.to_numeric(part[c], errors="coerce")
            ok = v.notna()
            if ok.any():
                ww = w[ok]
                row[c] = float((v[ok] * ww).sum() / ww.sum()) if float(ww.sum()) > 0 else float(v[ok].mean())
            else:
                row[c] = np.nan
        for c in _TEXT_COLS:
            if c in part.columns:
                vals: list[str] = []
                for a, v in part[c].fillna("").astype(str).items():
                    v = v.strip()
                    if not v:
                        continue
                    v = f"{str(a).rpartition(':')[2]}: {v}" if (multi and c not in _PLAIN_TEXT) else v
                    if v not in vals:
                        vals.append(v)
                row[c] = "; ".join(vals)
        lf = life.reindex(mem)
        for c in ("life_deposits", "life_withdrawals", "life_rebates", "notional_usd", "lots_lifetime", "trades_lifetime"):
            row[c] = float(pd.to_numeric(lf[c], errors="coerce").fillna(0.0).sum()) if c in has_life else 0.0
        eq = pd.to_numeric(lf["equity"], errors="coerce") if "equity" in has_life else pd.Series(np.nan, index=lf.index)
        bal = pd.to_numeric(lf["balance"], errors="coerce") if "balance" in has_life else pd.Series(np.nan, index=lf.index)
        known = eq.notna() | bal.notna()
        row["equity"] = float(eq.where(eq.notna(), bal).fillna(0.0).sum())
        row["balance"] = float(bal.fillna(0.0).sum())
        row["equity_source"] = ("live" if bool(eq.notna().all()) else "balance" if bool(known.all())
                                else "partial" if bool(known.any()) else "n/a")
        for c in ("first_flow", "first_trade"):
            row[c] = pd.to_datetime(lf[c]).min() if c in has_life else pd.NaT
        for c in ("last_flow", "last_trade"):
            row[c] = pd.to_datetime(lf[c]).max() if c in has_life else pd.NaT
        rows.append(row)
    out = pd.DataFrame(rows).set_index("account")
    out.index.name = "account_key"
    return out


def _attach_client_columns(table: pd.DataFrame, links: pd.DataFrame, life: pd.DataFrame) -> pd.DataFrame:
    """Rows stay PER ACCOUNT. Each impacted account gets its own lifetime
    figures from `life` plus the client context: client_key, the PRIMARY
    account (the desk report's primary when it is the link source, else the
    client's oldest account -- earliest first trade / first funding, ties to
    the lowest login), whether this row is the primary, the number of other
    linked accounts, how many linked accounts traded the window, the list of
    the others, the link source, and the client's lifetime totals over every
    linked account for context."""
    out = table.copy()
    idx = out.index
    client_of = links["client_key"].reindex(idx)
    client_of = client_of.where(client_of.notna(), pd.Series(idx, index=idx))
    member_lists = links["members"].reindex(idx)
    report_primary = links["report_primary"].reindex(idx).fillna("") if "report_primary" in links.columns \
        else pd.Series("", index=idx)
    link_source = links["link_source"].reindex(idx).fillna("") if "link_source" in links.columns \
        else pd.Series("", index=idx)
    # per-account lifetime figures (this account only)
    for col in ("life_deposits", "life_withdrawals", "life_rebates", "notional_usd", "lots_lifetime", "trades_lifetime"):
        out[col] = pd.to_numeric(life[col], errors="coerce").reindex(idx).fillna(0.0) if col in life.columns else 0.0
    for col in ("first_flow", "last_flow", "first_trade", "last_trade"):
        out[col] = pd.to_datetime(life[col]).reindex(idx) if col in life.columns else pd.NaT
    out["rebate_source"] = life["rebate_source"].reindex(idx).fillna("") if "rebate_source" in life.columns else ""
    start_of = pd.concat([pd.to_datetime(life["first_trade"]) if "first_trade" in life.columns else pd.Series(pd.NaT, index=life.index),
                          pd.to_datetime(life["first_flow"]) if "first_flow" in life.columns else pd.Series(pd.NaT, index=life.index)],
                         axis=1).min(axis=1)
    notional_all = pd.to_numeric(life["notional_usd"], errors="coerce").fillna(0.0) if "notional_usd" in life.columns \
        else pd.Series(0.0, index=life.index)
    netdep_all = ((pd.to_numeric(life["life_deposits"], errors="coerce").fillna(0.0)
                   - pd.to_numeric(life["life_withdrawals"], errors="coerce").fillna(0.0))
                  if "life_deposits" in life.columns else pd.Series(0.0, index=life.index))

    def login_of(key: str) -> int:
        tail = str(key).rpartition(":")[2]
        return int(tail) if tail.isdigit() else 0

    primary_col, is_primary, subaccounts, in_window, linked, src, ckey, c_not, c_dep = ([] for _ in range(9))
    cache: dict[str, tuple] = {}
    impacted = set(idx)
    for a in idx:
        ck = client_of[a]
        if ck not in cache:
            mem = {a}
            for lst in member_lists[client_of[client_of == ck].index]:
                if isinstance(lst, (list, tuple, set)):
                    mem |= set(lst)
            mem = sorted(mem)
            rp = next((p for p in report_primary[client_of[client_of == ck].index] if p), "")
            if rp and rp in mem:
                prim = rp
            else:
                prim = min(mem, key=lambda m: (pd.Timestamp(start_of.get(m, pd.NaT)) if pd.notna(start_of.get(m, pd.NaT))
                                                else pd.Timestamp.max, login_of(m)))
            cache[ck] = (mem, prim, len([m for m in mem if m in impacted]),
                         float(notional_all.reindex(mem).fillna(0.0).sum()),
                         float(netdep_all.reindex(mem).fillna(0.0).sum()))
        mem, prim, n_in, c_notional, c_netdep = cache[ck]
        primary_col.append(prim); is_primary.append(a == prim)
        subaccounts.append(len(mem) - 1); in_window.append(n_in)
        linked.append("; ".join(m for m in mem if m != a))
        src.append(link_source[a] if len(mem) > 1 else "")
        ckey.append(ck); c_not.append(c_notional); c_dep.append(c_netdep)
    out["client_key"] = ckey
    out["primary_account"] = primary_col
    out["is_primary"] = is_primary
    out["subaccounts"] = subaccounts
    out["accounts_in_window"] = in_window
    out["linked_accounts"] = linked
    out["link_source"] = src
    out["client_notional_usd"] = c_not
    out["client_net_deposits"] = c_dep
    return out


def _analyze_inner(start: str, end: str, symbols: str,
                   loss_limit: float, profit_limit: float,
                   row_cap: int | None = None) -> dict:
    from webapp import data_store
    # Inputs are Europe/London wall time (second granularity); the warehouse
    # speaks naive UTC.
    from webapp.econ_calendar import ldn_to_utc
    t0 = pd.Timestamp(ldn_to_utc(start))
    t1 = pd.Timestamp(ldn_to_utc(end))
    if t1 <= t0:
        raise ValueError("end must be after start")
    baseline_from = t0 - timedelta(days=5)
    post_to = t1 + timedelta(days=5)
    frame = data_store.read_history(start=baseline_from.to_pydatetime(),
                                    end=post_to.to_pydatetime())
    if frame is None or not len(frame):
        return {"rows": [], "note": "no trades in range"}
    frame = frame.copy()
    frame["account_key"] = (frame["database"].astype(str) + ":"
                            + frame["login"].astype(str))
    frame["open_time"] = pd.to_datetime(frame["open_time"])
    frame["close_time"] = pd.to_datetime(frame["close_time"])

    want = {s.strip().upper() for s in symbols.split(",") if s.strip()}
    if want:
        frame["canon"] = [(_canon(s)).upper() for s in frame["symbol"].astype(str)]
        sym_mask = frame["canon"].isin(want)
    else:
        sym_mask = pd.Series(True, index=frame.index)

    # IMPACTED: position lived through any part of the window on the symbols.
    hit = sym_mask & (frame["open_time"] <= t1) & (frame["close_time"] >= t0)
    impacted_accounts = frame.loc[hit, "account_key"].astype(str).unique()
    if not len(impacted_accounts):
        return {"rows": [], "note": "no impacted clients found"}

    # LINKED ACCOUNTS: every login of one client (primary + sub-accounts, on
    # any server) is pulled in, so the lifetime figures, the value class and
    # the window P&L are the CLIENT's, and a client whose two accounts both
    # traded appears once. The window tests still run per account first and
    # are collapsed afterwards (_aggregate_clients).
    from webapp import event_abuse
    link_notes: list[str] = []
    imp_keys = list(impacted_accounts.astype(str))
    try:
        links, link_notes = event_abuse.linked_accounts(imp_keys)
    except Exception as error:
        links = pd.DataFrame({"client_key": imp_keys, "members": [[a] for a in imp_keys],
                              "link_source": ""}, index=pd.Index(imp_keys, name="account_key"))
        link_notes = [f"account linking unavailable: {type(error).__name__}: {error}"]
    universe = sorted({m for mem in links["members"] for m in mem} | set(imp_keys))
    uni = set(universe)

    # CLOSE-BY legs inside the window (a long netted against a short at the
    # counterpart's price): real P&L, but not a market fill and no change in
    # net exposure. Marked once here; the stop-out burst, the hedge-unwind
    # test and the execution check all leave them out.
    cb_notes: list[str] = []
    try:
        from webapp import event_abuse as _ea
        close_by, cb_notes = _ea.close_by_orders(t0.to_pydatetime(), t1.to_pydatetime(),
                                                 list(impacted_accounts.astype(str)))
    except Exception as error:
        close_by, cb_notes = set(), [f"close-by lookup unavailable: {type(error).__name__}: {error}"]
    if close_by:
        orders_int = pd.to_numeric(frame["order"], errors="coerce").fillna(-1).astype("int64")
        frame["close_by"] = [(d, o) in close_by for d, o in zip(frame["database"].astype(str), orders_int)]
    else:
        frame["close_by"] = False

    sub = frame[frame["account_key"].astype(str).isin(impacted_accounts)].copy()
    sub["account_key"] = sub["account_key"].astype(str)

    # EVERY ACTION touching the window on the affected symbols -- so nothing
    # is missed: OPENED inside, CLOSED inside, or HELD THROUGH (straddling).
    opened_in = sym_mask & (frame["open_time"] >= t0) & (frame["open_time"] <= t1)
    closed_in = sym_mask & (frame["close_time"] >= t0) & (frame["close_time"] <= t1)
    held_through = sym_mask & (frame["open_time"] < t0) & (frame["close_time"] > t1)
    any_action = opened_in | closed_in | held_through
    in_window = closed_in                      # realized P&L attaches to closes

    def _count(mask):
        f = frame.loc[mask].copy()
        f["account_key"] = f["account_key"].astype(str)
        return f.groupby("account_key", observed=True).size()
    opens_cnt = _count(opened_in)
    closes_cnt = _count(closed_in)
    held_cnt = _count(held_through)
    action_cnt = _count(any_action)
    close_by_cnt = _count(closed_in & frame["close_by"].astype(bool))
    # gross lots transacted inside the window (opens + closes)
    act = frame.loc[opened_in | closed_in].copy()
    act["account_key"] = act["account_key"].astype(str)
    window_lots = act.groupby("account_key", observed=True)["volume_lots"].sum()

    win = frame.loc[in_window].copy()
    win["account_key"] = win["account_key"].astype(str)
    g = win.groupby("account_key", observed=True)
    window_pnl = g["net_profit"].sum()
    window_closes = g.size()
    # STOP-OUT HEURISTIC: >=2 losing closes within the window landing inside
    # any 120-second burst -- the signature of a margin cascade.
    def _burst(gr):
        losses = gr.loc[(gr["net_profit"] < 0) & ~gr["close_by"].astype(bool)].sort_values("close_time")
        if len(losses) < 2:
            return False
        gaps = losses["close_time"].diff().dt.total_seconds()
        return bool((gaps <= 120).any())
    stopped = g.apply(_burst)

    # BEHAVIOUR BASELINE: activity per day in the 5 days before the event.
    pre = sub[(sub["open_time"] >= baseline_from) & (sub["open_time"] < t0)]
    pg = pre.groupby("account_key", observed=True)
    base_days = max((t0 - baseline_from).days, 1)
    base_trades_pd = (pg.size() / base_days)
    base_volume_pd = (pg["volume_lots"].sum() / base_days)

    deltas = {}
    for name, days in HORIZONS:
        seg = sub[(sub["open_time"] > t1)
                  & (sub["open_time"] <= t1 + timedelta(days=days))]
        sg = seg.groupby("account_key", observed=True)
        rate = sg.size() / max(days, 1e-9)
        vol = sg["volume_lots"].sum() / max(days, 1e-9)
        deltas[f"act_{name}"] = (rate / base_trades_pd.replace(0, np.nan)) \
            .replace([np.inf, -np.inf], np.nan)
        deltas[f"vol_{name}"] = (vol / base_volume_pd.replace(0, np.nan)) \
            .replace([np.inf, -np.inf], np.nan)

    # FUNDING: deposits, withdrawals and NET per horizon AFTER the event --
    # only for the IMPACTED accounts -- plus each client's LIFETIME funding
    # for the LTV proxy. A stale store would silently read 0, so we track
    # its coverage and flag it.
    withdrawals, deposits = {}, {}
    life_dep = life_wd = life_reb = pd.Series(dtype=float)
    cash_max = None
    imp = set(impacted_accounts.astype(str))
    try:
        from webapp import cashflow_store
        flows = cashflow_store.read_cashflows()
        flows["ts"] = pd.to_datetime(flows["when"])
        flows["account_key"] = flows["account_key"].astype(str)
        cash_max = flows["ts"].max()
        flows = flows[flows["account_key"].isin(uni)]        # the clients' every account
        kind = flows.get("kind", pd.Series("", index=flows.index)).astype(str)
        amt = pd.to_numeric(flows["amount"], errors="coerce").fillna(0.0)
        # REBATES credited in-account (partner payouts 'PPF-', IB payments,
        # anything labelled rebate / cashback) are the firm's money, not the
        # client's capital: taken OUT of deposits and reported on their own.
        is_reb = event_abuse.rebate_mask(flows["comment"] if "comment" in flows.columns
                                         else pd.Series("", index=flows.index), amt)
        # real external money: deposits IN, withdrawals OUT (exclude internal
        # transfers, bonus credits and rebates, none of them the client's own capital).
        is_dep = ((kind == "deposit") | ((kind == "") & (amt > 0))) & ~is_reb
        is_wd = (kind == "withdrawal") | ((kind == "") & (amt < 0))
        dep_f = flows[is_dep].assign(v=amt[is_dep].abs())
        wd_f = flows[is_wd].assign(v=amt[is_wd].abs())
        reb_f = flows[is_reb].assign(v=amt[is_reb].abs())
        life_reb = reb_f.groupby("account_key")["v"].sum()
        for name, days in HORIZONS:
            wm = (wd_f["ts"] > t1) & (wd_f["ts"] <= t1 + timedelta(days=days))
            dm = (dep_f["ts"] > t1) & (dep_f["ts"] <= t1 + timedelta(days=days))
            withdrawals[name] = wd_f.loc[wm].groupby("account_key")["v"].sum()
            deposits[name] = dep_f.loc[dm].groupby("account_key")["v"].sum()
        # LIFETIME (all history in the store) for the LTV proxy, plus each
        # client's funding inception (first movement) for tenure.
        life_dep = dep_f.groupby("account_key")["v"].sum()
        life_wd = wd_f.groupby("account_key")["v"].sum()
        first_flow = flows.groupby("account_key")["ts"].min()
        last_flow = flows.groupby("account_key")["ts"].max()
        # net funding AFTER the event start, for the balance-at-t0 estimate
        net_flow_after_t0 = (dep_f.loc[dep_f["ts"] > t0].groupby("account_key")["v"].sum()
                             .subtract(wd_f.loc[wd_f["ts"] > t0].groupby("account_key")["v"].sum(), fill_value=0.0))
    except Exception:
        withdrawals, deposits = {}, {}
        first_flow = pd.Series(dtype="datetime64[ns]")
        last_flow = pd.Series(dtype="datetime64[ns]")
        net_flow_after_t0 = pd.Series(dtype=float)

    accounts = pd.Index(sorted(impacted_accounts.astype(str)), name="account_key")
    table = pd.DataFrame(index=accounts)
    table["window_pnl"] = window_pnl.reindex(accounts).fillna(0.0)
    table["window_closes"] = window_closes.reindex(accounts).fillna(0).astype(int)
    # ALL actions in the window, broken out so nothing is missed.
    table["opens_in_window"] = opens_cnt.reindex(accounts).fillna(0).astype(int)
    table["closes_in_window"] = closes_cnt.reindex(accounts).fillna(0).astype(int)
    table["held_through"] = held_cnt.reindex(accounts).fillna(0).astype(int)
    table["actions_in_window"] = action_cnt.reindex(accounts).fillna(0).astype(int)
    table["close_by_closes"] = close_by_cnt.reindex(accounts).fillna(0).astype(int)
    table["lots_in_window"] = window_lots.reindex(accounts).fillna(0.0).round(2)
    table["stopped_out"] = stopped.reindex(accounts).fillna(False)
    table["base_trades_pd"] = base_trades_pd.reindex(accounts).fillna(0.0)
    for col, series in deltas.items():
        table[col] = series.reindex(accounts)
    for name, _ in HORIZONS:
        wd = withdrawals.get(name, pd.Series(dtype=float)) \
            .reindex(accounts).fillna(0.0)
        dep = deposits.get(name, pd.Series(dtype=float)) \
            .reindex(accounts).fillna(0.0)
        table[f"wd_{name}"] = wd
        table[f"dep_{name}"] = dep
        table[f"net_dep_{name}"] = (dep - wd).round(2)     # deposits - withdrawals
    # LIFETIME funding + LTV proxy. LTV = net lifetime deposits (the money
    # the client has actually left with the firm) which for a warehoused
    # book approximates realised client value; a big positive net-deposit
    # client is high-LTV, a net-withdrawer is capital leaving.
    table["life_deposits"] = life_dep.reindex(accounts).fillna(0.0).round(0)
    table["life_withdrawals"] = life_wd.reindex(accounts).fillna(0.0).round(0)
    table["life_rebates"] = life_reb.reindex(accounts).fillna(0.0).round(0)
    table["net_deposits"] = (table["life_deposits"]
                             - table["life_withdrawals"]).round(0)
    # (tenure, MLTV and revenue are computed after the collapse to clients,
    #  over every linked account -- see below)

    # ================= ABUSE TESTS (event_abuse) + CLASSIFICATION =================
    from webapp import event_abuse
    abuse_notes: list[str] = list(cb_notes)
    acct_list = list(accounts)
    try:
        # the whole universe: equity / balance of every linked account too
        state, n_ = event_abuse.account_state(universe, since=t0.to_pydatetime()); abuse_notes += n_
    except Exception as error:
        state = pd.DataFrame(); abuse_notes.append(f"account state unavailable: {type(error).__name__}: {error}")
    for col, default in (("group", ""), ("status", ""), ("desk_flag", ""), ("account_leverage", np.nan),
                         ("balance", np.nan), ("credit", np.nan), ("floating", np.nan), ("equity", np.nan)):
        table[col] = state[col].reindex(accounts) if col in getattr(state, "columns", []) else default
        if default == "":
            table[col] = table[col].fillna("")
    try:                                   # the desk's own register (CSV export) -- complement / fallback
        adb = event_abuse.abusedb_flags(acct_list)
        table["abusedb_flag"] = adb["abusedb_flag"].reindex(accounts).fillna("") if len(adb) else ""
        table["abusedb_saved_usd"] = (adb["abusedb_saved_usd"].reindex(accounts).fillna(0.0).round(0)
                                      if len(adb) else 0.0)
    except Exception as error:
        table["abusedb_flag"] = ""; table["abusedb_saved_usd"] = 0.0
        abuse_notes.append(f"AbuseDb register unavailable: {type(error).__name__}: {error}")
    try:
        hl, n_ = event_abuse.hedge_leverage_check(
            frame, t0, t1, sym_mask, acct_list, table["balance"], net_flow_after_t0,
            realised_after=(state["realised_after_t0"] if "realised_after_t0" in getattr(state, "columns", []) else None))
        abuse_notes += n_
        for col in hl.columns:
            table[col] = hl[col].reindex(accounts)
    except Exception as error:
        abuse_notes.append(f"leverage / hedge check failed: {type(error).__name__}: {error}")
        for col in ("notional_at_t0", "balance_at_t0_est", "effective_leverage_t0", "leverage_after",
                    "net_lots_after", "long_lots_t0", "short_lots_t0", "long_lots_after", "short_lots_after"):
            table[col] = np.nan
        table["positions_at_t0"] = 0; table["hedged_at_t0"] = False; table["hedge_unwind"] = False; table["hedge_detail"] = ""
    try:
        ex, n_ = event_abuse.execution_check(t0.to_pydatetime(), t1.to_pydatetime(), _canon, want, acct_list)
        abuse_notes += n_
        for col in ("fills_in_window", "favourable_fills", "stop_gap_fills"):
            table[col] = ex[col].reindex(accounts).fillna(0).astype(int) if col in ex.columns else 0
        for col in ("max_favourable_bps", "advantage_usd"):
            table[col] = ex[col].reindex(accounts).fillna(0.0).round(2) if col in ex.columns else 0.0
        table["execution_detail"] = (ex["execution_detail"].reindex(accounts).fillna("")
                                     if "execution_detail" in ex.columns else "")
    except Exception as error:
        abuse_notes.append(f"execution check failed: {type(error).__name__}: {error}")
        for col in ("fills_in_window", "favourable_fills", "stop_gap_fills"):
            table[col] = 0
        table["max_favourable_bps"] = 0.0; table["advantage_usd"] = 0.0; table["execution_detail"] = ""
    table["hedge_unwind"] = table["hedge_unwind"].fillna(False).astype(bool)
    table["abuse_reasons"] = ["; ".join(event_abuse.reasons(r)) for _, r in table.iterrows()]
    table["abusive"] = table["abuse_reasons"].str.len() > 0

    # Behavioural label and the volume value proxy, per ACCOUNT (collapsed
    # to the client next).
    abuse_labels = {}
    try:
        from webapp import antifraud
        panel = antifraud.classify()
        if panel is not None and len(panel):
            abuse_labels = dict(zip(panel["account_key"].astype(str),
                                    panel["profile"].astype(str)))
    except Exception:
        pass
    table["abuse_label"] = [abuse_labels.get(a, "") for a in accounts]
    try:
        table["value_volume"] = pg["volume_lots"].sum().reindex(accounts).fillna(0.0)
    except Exception:
        table["value_volume"] = 0.0

    # ================= COLLAPSE TO CLIENTS (primary + linked accounts) =================
    # Lifetime figures of EVERY account of the client (impacted or not):
    # funding, rebates, equity, USD notional traded, first / last activity.
    try:
        lt, n_ = event_abuse.lifetime_trading(universe); abuse_notes += n_
    except Exception as error:
        lt = pd.DataFrame(); abuse_notes.append(f"lifetime trading unavailable: {type(error).__name__}: {error}")
    life = pd.DataFrame(index=pd.Index(universe, name="account_key"))
    life["life_deposits"] = life_dep.reindex(universe).fillna(0.0)
    life["life_withdrawals"] = life_wd.reindex(universe).fillna(0.0)
    life["life_rebates"] = life_reb.reindex(universe).fillna(0.0)
    life["first_flow"] = pd.to_datetime(first_flow.reindex(universe))
    life["last_flow"] = pd.to_datetime(last_flow.reindex(universe))
    for col in ("equity", "balance"):
        life[col] = state[col].reindex(universe) if col in getattr(state, "columns", []) else np.nan
    for col in ("notional_usd", "lots_lifetime", "trades_lifetime"):
        life[col] = lt[col].reindex(universe) if col in getattr(lt, "columns", []) else 0.0
    for col in ("first_trade", "last_trade"):
        life[col] = pd.to_datetime(lt[col].reindex(universe)) if col in getattr(lt, "columns", []) else pd.NaT
    # REBATES: the per-trade rebate_payout -- the warehouse first
    # (data_marts.closed_trades), a Metabase export on disk next, else the
    # in-account credits; the source is a column on every row.
    try:
        mb, n_ = event_abuse.rebate_payouts_any(universe); abuse_notes += n_
    except Exception as error:
        mb = pd.DataFrame(); abuse_notes.append(f"rebate export unavailable: {type(error).__name__}: {error}")
    life["rebate_source"] = np.where(life["life_rebates"] > 0, "in-account credits (PPF / IB payouts)", "none recorded")
    if len(mb):
        has = mb.index.intersection(life.index)
        life.loc[has, "life_rebates"] = pd.to_numeric(mb.loc[has, "rebates_usd"], errors="coerce").fillna(0.0)
        life.loc[has, "rebate_source"] = mb.loc[has, "rebate_source"]
    n_accounts = int(len(table))
    # rows stay PER ACCOUNT; the client context (primary, sub-accounts) is attached
    table = _attach_client_columns(table, links, life)
    abuse_notes += link_notes
    n_clients = int(table["client_key"].nunique())

    # CLASSIFICATION (the business table), per ACCOUNT: MLTV = net deposits
    # (rebates excluded) per ACTIVE month; net revenue = net deposits - equity
    # (what the firm kept after the rebates it paid); gross revenue adds the
    # rebates back; monthly figures divide by the active life; a third
    # measure, USD notional traded per month, is tiered by percentiles fitted
    # to the affected accounts; the highest tier of the three wins; combined
    # ratio = revenue / MLTV. Active life = the earlier of first funding and
    # first trade -> the last activity (trade or funding).
    now_ref = cash_max if cash_max is not None else pd.Timestamp(datetime.utcnow())
    ff = pd.concat([pd.to_datetime(table["first_flow"]), pd.to_datetime(table["first_trade"])], axis=1).min(axis=1)
    tenure_m = ((now_ref - ff).dt.total_seconds() / (30.44 * 86400)).clip(lower=1.0)
    table["tenure_months"] = tenure_m.round(1)
    table["net_deposits"] = (table["life_deposits"] - table["life_withdrawals"]).round(0)
    table["mltv_tenure"] = (table["net_deposits"] / tenure_m).round(0)
    last_seen = pd.concat([pd.to_datetime(table["last_trade"]), pd.to_datetime(table["last_flow"])], axis=1).max(axis=1)
    life_m = ((last_seen - ff).dt.total_seconds() / (30.44 * 86400)).clip(lower=1.0)
    life_m = life_m.fillna(tenure_m).fillna(1.0)
    table["active_life_months"] = life_m.round(1)
    equity_used = pd.to_numeric(table["equity"], errors="coerce")
    equity_used = equity_used.where(equity_used.notna(), pd.to_numeric(table["balance"], errors="coerce")).fillna(0.0)
    table["equity_source"] = np.where(pd.to_numeric(table["equity"], errors="coerce").notna(), "live",
                                      np.where(pd.to_numeric(table["balance"], errors="coerce").notna(), "balance", "n/a"))
    table["mltv"] = (table["net_deposits"] / life_m).round(0)
    table["net_revenue"] = (table["net_deposits"] - equity_used).round(0)
    table["gross_revenue"] = (table["net_revenue"] + table["life_rebates"]).round(0)
    table["monthly_revenue"] = (table["net_revenue"] / life_m).round(0)
    table["rebates_monthly"] = (table["life_rebates"] / life_m).round(0)
    table["notional_usd_monthly"] = (table["notional_usd"] / life_m).round(0)
    notional_cuts = event_abuse.notional_tiers(table["notional_usd_monthly"])
    cls = event_abuse.classify(table["mltv"], table["monthly_revenue"], table["notional_usd_monthly"], notional_cuts)
    for col in cls.columns:
        table[col] = cls[col]

    def impact_class(row):
        if row["stopped_out"]:
            return "stopped_out"
        if row["window_pnl"] <= -abs(loss_limit):
            return "significant_loss"
        if row["window_pnl"] >= abs(profit_limit):
            return "significant_profit"
        return "minor"
    table["impact"] = table.apply(impact_class, axis=1)

    # ML: anomaly score over behaviour deltas (activity collapse/spike,
    # withdrawal burst) -- ranks who reacted hardest.
    feat_cols = [c for c in table.columns
                 if c.startswith(("act_", "vol_", "wd_"))]
    feats = table[feat_cols].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    table["anomaly"] = 0.0
    try:
        from sklearn.ensemble import IsolationForest
        if len(feats) >= 20:
            iso = IsolationForest(n_estimators=200, random_state=0,
                                  contamination="auto")
            table["anomaly"] = -iso.fit(feats).score_samples(feats)
            lo, hi = table["anomaly"].min(), table["anomaly"].max()
            if hi > lo:
                table["anomaly"] = (table["anomaly"] - lo) / (hi - lo)
    except Exception:
        pass

    # HIGH VALUE blends trading-volume value with MLTV (net deposits per
    # month) -- a strong monthly funder is worth retaining even on modest
    # volume or short tenure.
    mltv_hi = float(table["mltv"].quantile(0.8)) if len(table) else 0.0
    v_hi = float(table["value_volume"].quantile(0.8)) if len(table) else 0.0
    table["high_value"] = ((table["value_volume"] >= v_hi)
                           | (table["mltv"] >= max(mltv_hi, 1.0)))

    # BEHAVIOUR CHANGE after the event: traded materially LESS (5d activity
    # under half baseline), pulled money (5d withdrawal), or was stopped out
    # -- the churn-risk signature.
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

    # The old behavioural abuse signal (won on the event and pulled the
    # money, or a toxic label) stays visible as evidence but no longer makes
    # the abuse tab on its own: that tab is the three tests above.
    toxic_labels = ("toxic_flow", "persistent_edge", "news_vol", "bonus_arb", "swap_arb")
    table["won_and_withdrew"] = ((table["impact"] == "significant_profit")
                                 & ((table["wd_d5"] > 0) | (table["anomaly"] > 0.8)
                                    | table["abuse_label"].isin(toxic_labels)))
    valuable = table["client_class"].isin(("Mid", "High", "VIP"))

    def segment(row):
        if row["abusive"]:
            return "abuse_candidate"
        # LOST MONEY on the event. A stop-out by itself is not a reason to
        # compensate: it must also be a material loss.
        lost = row["window_pnl"] <= -abs(loss_limit)
        withdrew = row.get("wd_d5", 0) > 0
        if lost:
            if row["client_class"] in ("Mid", "High", "VIP") or row["high_value"] or (withdrew and row["mltv"] > 0):
                return "retain_high_value"
            return "compensate_review"
        if row["won_and_withdrew"]:
            return "monitor"
        if row["behaviour_changed"] and (withdrew or row["anomaly"] > 0.8
                                         or (pd.notna(row.get("act_d5")) and row["act_d5"] < 0.5)):
            return "monitor"
        return "minimal"
    table["segment"] = table.apply(segment, axis=1)
    # VIP / high / mid clients, NOT abusive, hurt by the event: the priority
    # list, biggest loss first.
    table["vip_impacted"] = valuable & ~table["abusive"] & (table["window_pnl"] < 0)

    order = {"abuse_candidate": 0, "retain_high_value": 1,
             "compensate_review": 2, "monitor": 3, "minimal": 4}
    # within each segment, rank by MLTV desc (the most valuable clients per
    # month float to the top of every segment -- who to act on first), and
    # keep a client's linked accounts TOGETHER: the group takes its best
    # segment and MLTV, the primary account leads, sub-accounts follow.
    grp = table.groupby("client_key")
    table["_seg_rank"] = grp["segment"].transform(lambda s: min(order.get(x, 9) for x in s))
    table["_grp_mltv"] = grp["mltv"].transform("max")
    table["_row_rank"] = table["segment"].map(order).fillna(9)
    table = table.sort_values(["_seg_rank", "_grp_mltv", "client_key", "is_primary", "_row_rank", "mltv"],
                              ascending=[True, False, True, False, True, False], kind="mergesort")
    table = table.drop(columns=["_seg_rank", "_grp_mltv", "_row_rank"])

    def _clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else round(float(v), 3)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    for col in ("first_flow", "last_flow", "first_trade", "last_trade"):
        if col in table.columns:
            table[col] = pd.to_datetime(table[col]).dt.strftime("%Y-%m-%d").fillna("")
    rows = []
    for account, r in table.iterrows():
        rows.append({"account": account,
                     **{c: _clean(r[c]) for c in table.columns}})

    seg_counts = table["segment"].value_counts().to_dict()
    abusive_rows = [r for r in rows if r.get("abusive")]
    abusive_rows.sort(key=lambda r: (-(r.get("advantage_usd") or 0) - (r.get("abusedb_saved_usd") or 0),
                                     r.get("window_pnl") or 0))
    vip_rows = sorted([r for r in rows if r.get("vip_impacted")], key=lambda r: r.get("window_pnl") or 0)
    definitions = [
        "ABUSE 0 — desk flag: the account is already marked by the dealing desk. MT5: the account "
        "comment ('Toxic', 'Toxic 2/3/4', 'Rebate Abuser'). MT4 carries no such mark in MySQL; the desk's "
        "AbuseDb register (CSV export, matched by login) covers both platforms.",
        f"ABUSE 1a — effective leverage into the event: NET exposure per instrument (|long - short| notional, "
        f"summed) of every position open at the window start, over the balance at that moment (balance now "
        f"minus P&L realised since minus net funding since), of at least {event_abuse.HIGH_LEVERAGE:.0f}x on at "
        f"least ${event_abuse.MIN_NOTIONAL:,.0f} of net notional. Gross notional and gross leverage are "
        f"listed alongside; a hedged book shows high gross and low net until it is unwound (test 1b).",
        f"ABUSE 1b — hedge unwind: long and short open on the event instrument going into the window, at "
        f"least half of one side closed inside it while at least half of the other side is kept, and the "
        f"side that is left carries at least {event_abuse.HEDGE_LEVERAGE:.0f}x leverage -- a hedge (little "
        f"margin) turned into a leveraged directional position at the news price.",
        f"ABUSE 2 — favourable execution: a fill inside the window better than the market at that moment by "
        f"at least {event_abuse.FAVOURABLE_BPS:.0f} bps. MT5: against the deal's own market bid/ask. MT4: "
        f"against the best price on the tick tape in the second around the fill. Stop orders filled beyond "
        f"their trigger in the client's favour are listed as such.",
        "CLASSIFICATION — MLTV = net deposits / active months; monthly revenue = (net deposits - equity) / "
        "active months; tiers VIP 40,000 / 32,000, High 10,000 / 6,000, Mid 1,000 / 400, Low 500 / 100 "
        "(the higher tier of the two measures); combined ratio = revenue / MLTV. Active life = first funding "
        "movement to last activity. Equity = balance + credit + floating P&L from the platform now.",
        "VIP IMPACTED tab — Mid, High and VIP clients with no abuse flag whose window P&L is negative, "
        "biggest loss first. A stop-out inside the window is not by itself a reason to compensate.",
        "CLOSE-BY — a long netted against a short at the counterpart's price ('close hedge by #' on MT4, the "
        "OUT_BY deal on MT5). The P&L is real (the locked spread) but neither leg is a market fill and net "
        "exposure does not change, so close-by legs are counted (Close-by closes / lots) and excluded from "
        "the stop-out burst, the hedge-unwind test and the execution comparison.",
        "LINKED ACCOUNTS — one row per affected ACCOUNT (sub-account), every figure that account's own. A "
        "client's accounts are placed together: 'Primary account' names the client's primary (the desk "
        "report's primary when that is the link source, else the client's oldest account), 'Is primary' marks "
        "that row, 'Sub-accounts' = number of other linked accounts, 'Linked accounts active in window' = how "
        "many of them traded the event, 'Client ... all linked accounts' columns give the client's totals for "
        "context. Link source, in order of preference: the data warehouse's primary_trading_account_number "
        "(data_marts.trading_accounts -- the client register: every trading account of the same primary, on "
        "any server), then a primary/sub-account export on disk (docs/account_links*.csv|xlsx), and only for "
        "accounts neither covers the platform's identity fields (same name + country on a server; same e-mail "
        "on MT5), which are labelled as such.",
        "REBATES & REVENUE — rebates are the firm's money, not the client's capital, and are excluded from "
        "deposits: Net deposits = external deposits - withdrawals. 'Rebate source' says where the figure came "
        "from: the warehouse's per-trade rebate_payout (data_marts.closed_trades, last 730 days -- the figure "
        "Metabase shows), else a Metabase export on disk, else in-account credits (partner payouts 'PPF-', IB "
        "payments), which understate the true rebates -- the trading servers hold no rebate field. Net revenue "
        "= net deposits - equity now (after the rebates paid); Gross revenue = net revenue + rebates. Monthly "
        "figures divide by the account's active life.",
        "CLASS BY MONTHLY USD NOTIONAL — the third classification measure: USD notional traded per active "
        "month, tiered by percentiles fitted to the affected accounts (VIP = top 2.5%, High = top 10%, Mid = "
        "top 40%, Low = top 70%, Micro = the rest; no notional = Micro)"
        + (": cut-offs " + ", ".join(f"{k} ≥ ${v:,.0f}" for k, v in (notional_cuts or {}).items()) if notional_cuts else "")
        + ". The client class is the highest tier of MLTV, monthly net revenue and monthly notional.",
        "USD NOTIONAL — every trade entry over the platform's whole history (one side of each round trip): "
        "lots x contract size x price, FX in the base currency converted at an approximate USD rate; "
        "cent accounts deflated. Monthly = lifetime / active life months.",
    ]
    return {
        "rows": rows[:row_cap] if row_cap else rows,
        "abusive_rows": abusive_rows,
        "vip_rows": vip_rows,
        "abuse_definitions": definitions,
        "abuse_counts": {
            "desk_flag": int((table["desk_flag"].astype(str).str.len() > 0).sum()),
            "abusedb": int((table["abusedb_flag"].astype(str).str.len() > 0).sum()),
            "high_leverage": int(((table["effective_leverage_t0"] >= event_abuse.HIGH_LEVERAGE)
                                  & (table["notional_at_t0"] >= event_abuse.MIN_NOTIONAL)).sum()),
            "hedge_unwind": int(table["hedge_unwind"].sum()),
            "favourable_execution": int((((table["favourable_fills"] > 0)
                                          & (table["advantage_usd"] >= event_abuse.MIN_ADVANTAGE_USD))
                                         | (table["stop_gap_fills"] > 0)).sum()),
            "abusive": int(table["abusive"].sum()),
            "vip_impacted": int(table["vip_impacted"].sum()),
        },
        "classes": table["client_class"].value_counts().to_dict(),
        "abuse_notes": abuse_notes,
        "n_impacted": int(len(table)),          # impacted ACCOUNTS (one row each)
        "n_accounts": int(n_accounts),
        "n_clients": int(n_clients),            # distinct clients behind those accounts
        "n_linked": int((table["subaccounts"] > 0).sum()),
        "notional_cuts": {k: round(v, 0) for k, v in (notional_cuts or {}).items()},
        "window": {"start": str(t0), "end": str(t1),
                   "symbols": sorted(want) if want else "all"},
        "totals": {
            "notional_usd_total": round(float(table["notional_usd"].sum()), 0),
            "rebates_total": round(float(table["life_rebates"].sum()), 0),
            "net_revenue_total": round(float(table["net_revenue"].sum()), 0),
            "gross_revenue_total": round(float(table["gross_revenue"].sum()), 0),
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
        "withdrawals_covered": bool(
            cash_max is not None
            and cash_max >= (t1 + timedelta(days=5))),
        "notes": ["Login frequency and complaints have no data source in this "
                  "stack; trading-activity frequency is the engagement proxy.",
                  "Stop-out = >=2 losing closes within any 120s burst inside "
                  "the window."]
                 + ([f"WITHDRAWAL DATA INCOMPLETE: the cashflow store is only "
                     f"current to {str(cash_max)[:10]}, before this event's "
                     f"5-day withdrawal window ends — withdrawal figures "
                     f"understate reality until it is backfilled."]
                    if (cash_max is not None
                        and cash_max < t1 + timedelta(days=5)) else [])
                 + abuse_notes,
        "generated_at": str(datetime.utcnow()),
    }
