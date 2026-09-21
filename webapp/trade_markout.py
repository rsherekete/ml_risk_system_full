"""Per-trade markout evidence: one order, measured exactly as the latency engine measures it.

Built for evidence that can be shown to a client: the trade as booked
(millisecond times where the platform records them), the broker's own tick
tape and the independent reference market (Vantage Raw ECN) at the fill and at
the spec's seven horizons, and the classification the engine applies.

Sources, in the order they are tried:
- trade: the warehouse (order id, prices, P&L); MT4 millisecond open/close
  from the Kafka trade events; else the values the page passed in.
- broker ticks: the trade server's own MySQL `ticks` table (MT5: the dominant
  feeder in the window, `datetime_msc`; MT4: the shared MT4 tape, `tm`).
- reference ticks: latency_reference's parquet cache, fetched on demand.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

WINDOW_BEFORE = timedelta(seconds=65)   # quote at the fill + the lead window
WINDOW_AFTER = timedelta(seconds=70)    # the 60 s horizon + margin
_CACHE: dict = {}


def _find_trade(account_key: str, order=None, open_time=None, symbol=None) -> dict | None:
    """The trade row from the warehouse, by order id or by (open second, symbol)."""
    from webapp import data_store
    database, login = account_key.split(":", 1)
    base = pd.Timestamp(open_time) if open_time else pd.Timestamp(datetime.utcnow())
    months = pd.period_range((base - pd.Timedelta(days=2)).to_period("M"),
                             min(base + pd.Timedelta(days=62), pd.Timestamp(datetime.utcnow())).to_period("M"), freq="M")
    cols = ["database", "order", "login", "symbol", "cmd", "volume_lots", "open_time", "close_time",
            "open_price", "close_price", "net_profit", "commission", "storage"]
    for period in (months if open_time else reversed(months)):
        path = data_store.WAREHOUSE / database / f"{period}.parquet"
        if not path.exists():
            continue
        try:
            f = pd.read_parquet(path, columns=cols, filters=[("login", "==", int(login))])
        except Exception:
            continue
        if order is not None:
            try:
                wanted = int(float(order))
            except (TypeError, ValueError):
                return None
            hit = f.loc[pd.to_numeric(f["order"], errors="coerce") == wanted]
        else:
            ot = pd.to_datetime(f["open_time"]).dt.floor("s")
            hit = f.loc[(ot == base.floor("s")) & ((f["symbol"].astype(str) == str(symbol)) if symbol else True)]
        if len(hit):
            return hit.iloc[0].to_dict()
    return None


def _broker_ticks(database: str, symbol: str, canonical: str, lo: datetime, hi: datetime) -> tuple[pd.DataFrame, str]:
    # Prefer the same materialised MySQL tick store used by the scan. The
    # direct MySQL fallback below can disagree on symbol spelling, feed choice,
    # or timestamp filtering even when the scan has coverage for this fill.
    try:
        from webapp.kafka_service import shared_cursor
        from webapp.latency_arb import _FEED_SQL, _feed_of
        with shared_cursor() as cx:
            cx.execute(f"""
                SELECT {_FEED_SQL} AS feed, canonical, event_time, bid, ask
                FROM quotes
                WHERE event_time >= ? AND event_time <= ?
                  AND server LIKE 'mysql:%' AND canonical = ?
            """, [lo, hi, canonical])
            rows = cx.fetchall()
        if rows:
            preferred = _feed_of(database)
            feeds = {row[0] for row in rows}
            chosen = preferred if preferred in feeds else ("mt4" if "mt4" in feeds else sorted(feeds)[0])
            selected = [(t, bid, ask) for feed, _canon, t, bid, ask in rows if feed == chosen]
            f = pd.DataFrame(selected, columns=["t", "bid", "ask"])
            f["t"] = pd.to_datetime(f["t"]).astype("datetime64[ns]")
            f[["bid", "ask"]] = f[["bid", "ask"]].astype(float)
            return f.sort_values("t"), f"shared scan quote store, {canonical} @ {chosen}"
    except Exception:
        # The evidence endpoint remains useful if the shared store is busy;
        # fall through to the direct MySQL lookup.
        pass

    from webapp.mysql_extract import _connection
    from webapp.latency_arb import TICK_SOURCES
    if database.startswith("mt5"):
        names = list(dict.fromkeys([symbol, canonical]))
        placeholders = ", ".join(["%s"] * len(names))
        with _connection(database, timeout=60) as cx, cx.cursor() as cur:
            cur.execute(f"SELECT feeder, COUNT(*) FROM ticks WHERE symbol IN ({placeholders}) "
                        "AND ts >= %s AND ts < %s GROUP BY feeder ORDER BY 2 DESC LIMIT 1",
                        (*names, lo, hi))
            row = cur.fetchone()
            if not row:
                return pd.DataFrame(), f"{database} ticks: no {symbol} / {canonical} quotes in the window"
            cur.execute(f"SELECT datetime_msc, bid, ask FROM ticks WHERE feeder = %s AND symbol IN ({placeholders}) "
                        "AND ts >= %s AND ts < %s AND bid > 0 AND ask > 0 ORDER BY datetime_msc",
                        (row[0], *names, lo, hi))
            rows = cur.fetchall()
        label = f"{database} MySQL ticks, {symbol} / {canonical} @ {row[0]}"
    else:
        rows, label = [], "MT4 MySQL ticks: no quotes in the window"
        for db in TICK_SOURCES:
            try:
                with _connection(db, timeout=60) as cx, cx.cursor() as cur:
                    for name in dict.fromkeys([symbol, canonical]):
                        cur.execute("SELECT tm, bid, ask FROM ticks WHERE symbol_name = %s AND tm >= %s AND tm < %s "
                                    "AND bid > 0 AND ask > 0 ORDER BY tm", (name, lo, hi))
                        rows = cur.fetchall()
                        if rows:
                            label = f"{db} MySQL ticks (shared MT4 tape), {name}"
                            break
            except Exception:
                continue
            if rows:
                break
    f = pd.DataFrame(rows, columns=["t", "bid", "ask"])
    if len(f):
        f["t"] = pd.to_datetime(f["t"]).astype("datetime64[ns]")
        f[["bid", "ask"]] = f[["bid", "ask"]].astype(float)
    return f, label


def _reference_feeds(canonical: str, lo: pd.Timestamp, hi: pd.Timestamp, broker: pd.DataFrame,
                     rules: dict) -> list[tuple[str, pd.DataFrame, dict]]:
    """(feed, aligned ticks, meta) for every configured reference feed."""
    from webapp import latency_reference as lr
    pairs = sorted({(canonical, lo.floor("h")), (canonical, hi.floor("h"))})
    fetch = lr.ensure_pairs(pairs, rules)
    out = []
    for feed in lr.configured_feeds(rules):
        try:
            ticks, meta = _reference_ticks(feed, pairs, lo, hi, broker, rules)
        except Exception as error:
            ticks, meta = pd.DataFrame(), {"note": f"reference unavailable: {type(error).__name__}: {error}"}
        meta.update(feed=feed, label=lr.label_of(feed), ticks=int(len(ticks)),
                    fetch=((fetch.get("feeds") or {}).get(feed) or {}).get("fetch", {}).get("error")
                    or fetch.get("status"))
        out.append((feed, ticks, meta))
    return out


def _reference_ticks(feed: str, pairs: list, lo: pd.Timestamp, hi: pd.Timestamp, broker: pd.DataFrame,
                     rules: dict) -> tuple[pd.DataFrame, dict]:
    from webapp import latency_reference as lr
    ref = lr.load_reference(pairs, feed)
    meta = {}
    if not len(ref) or not len(broker):
        meta["note"] = "no reference ticks for this window" if not len(ref) else "no broker ticks to align the reference clock"
        return pd.DataFrame(), meta
    ref["mid"] = (ref["bid"] + ref["ask"]) / 2
    b = broker.assign(mid=(broker["bid"] + broker["ask"]) / 2, t=broker["t"].astype("datetime64[ns]"))[["t", "mid"]].sort_values("t")
    best = None
    for h in rules.get("reference_server_offsets_h", [2, 3]):
        r = pd.DataFrame({"t": pd.to_datetime(ref["server_ms"] - int(h) * 3_600_000, unit="ms").astype("datetime64[ns]"),
                          "rmid": ref["mid"]}).sort_values("t")
        m = pd.merge_asof(b, r, on="t", direction="backward").dropna()
        if len(m) < 5:
            continue
        dev = m["mid"] - m["rmid"]
        mad = float((dev - dev.median()).abs().median())
        if best is None or mad < best[1]:
            best = (int(h), mad, float(dev.median()))
    if best is None:
        meta["note"] = "reference clock could not be aligned in this window"
        return pd.DataFrame(), meta
    h, mad, basis = best
    out = pd.DataFrame({"t": pd.to_datetime(ref["server_ms"] - h * 3_600_000, unit="ms").astype("datetime64[ns]"),
                        "bid": ref["bid"] + basis, "ask": ref["ask"] + basis}).sort_values("t")
    out = out[(out["t"] >= lo) & (out["t"] <= hi)]
    meta.update({"server_offset_h": h, "basis": round(basis, 6), "alignment_mad": round(mad, 6),
                 "source": f"{lr.label_of(feed)} (independent reference)"})
    return out, meta


def _at(ticks: pd.DataFrame, when: pd.Timestamp):
    """The last quote at or before `when` (bid, ask, quote time)."""
    if not len(ticks):
        return None
    i = ticks["t"].searchsorted(when, side="right") - 1
    if i < 0:
        return None
    row = ticks.iloc[i]
    return float(row["bid"]), float(row["ask"]), row["t"]


def trade_markout(account_key: str, order=None, open_time=None, symbol=None, fallback: dict | None = None) -> dict:
    if order is not None and str(order).strip().lower() in ("", "nan", "none", "null", "<na>"):
        order = None
    key = (account_key, str(order), str(open_time), str(symbol))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    from webapp import latency_arb as la
    from webapp.trade_feed import _canonical
    rules = la.load_rules()
    notes = []
    trade = _find_trade(account_key, order, open_time, symbol)
    source = "warehouse"
    if trade is None:
        if not fallback or not fallback.get("open_time"):
            return {"error": "trade not found in the warehouse"}
        trade = dict(fallback, database=account_key.split(":", 1)[0], order=order)
        source = "trades table (not yet in the warehouse)"
        notes.append("Trade values are from the page's trade row; the warehouse has not received it yet.")
    database = str(trade["database"])
    frame = pd.DataFrame([{
        "database": database, "order": trade.get("order"), "account_key": account_key,
        "symbol": str(trade["symbol"]), "cmd": str(trade.get("cmd", "")),
        "open_time": pd.Timestamp(trade["open_time"]), "close_time": pd.Timestamp(trade["close_time"]) if trade.get("close_time") else pd.NaT,
        "open_price": float(trade["open_price"]), "close_price": float(trade.get("close_price") or np.nan),
        "net_profit": float(trade.get("net_profit") or 0.0), "volume_lots": float(trade.get("volume_lots") or 0.0)}])
    frame["direction"] = np.where(frame["cmd"].str.lower().str.startswith("b"), 1, -1)
    frame["hold_seconds"] = (frame["close_time"] - frame["open_time"]).dt.total_seconds()
    frame["row_id"] = 0
    frame["canonical"] = [_canonical(s) or s for s in frame["symbol"]]
    if database.startswith("mt4"):
        try:
            got = la._attach_mt4_ms_times(frame)
            if not got.get("mt4_ms_opens"):
                # NOT a data failure, and measured as such on 17 Sep 2026: of
                # 2,632 second-stamped MT4 fills, every one HAD Kafka events for
                # its order -- but the nearest sat a median 9 minutes from the
                # fill, because these are pending (limit/stop) orders whose
                # events stamp the PLACEMENT, not the activation. Only 2 of the
                # 2,632 would match on a wider join, so widening it buys nothing
                # and risks stamping a placement time onto a fill -- inventing
                # sub-second precision, which is the one thing this engine must
                # never do. The 1 s / 5 s fallback is the correct answer.
                notes.append(
                    "MT4 stamps this fill to the second: no event carries a "
                    "millisecond time at the fill instant. This is normal for "
                    "pending (limit/stop) orders, whose events stamp the order's "
                    "placement rather than its activation. Sub-second horizons "
                    "are therefore not measurable; the trade is still scored on "
                    "the 1 s and 5 s horizons.")
        except Exception as error:
            # Say plainly that this is an ACCESS fault, not a property of the
            # trade: the old wording read as "this fill has no millisecond
            # time", which sent people looking at the client instead of at the
            # tick store's lock.
            notes.append(f"MT4 millisecond times unavailable ({type(error).__name__}); "
                         "sub-second horizons are not shown. This is a data-access "
                         "fault, not a property of the trade -- the fill may well "
                         "have a millisecond time.")
    t0 = pd.Timestamp(frame.at[0, "open_time"])
    precise = (t0 - t0.floor("s")) > pd.Timedelta(0)
    d = int(frame.at[0, "direction"])
    p0 = float(frame.at[0, "open_price"])
    lots = float(frame.at[0, "volume_lots"])
    canonical = str(frame.at[0, "canonical"])
    lo, hi = t0 - WINDOW_BEFORE, t0 + WINDOW_AFTER

    try:
        broker, broker_src = _broker_ticks(database, str(frame.at[0, "symbol"]), canonical, lo.to_pydatetime(), hi.to_pydatetime())
    except Exception as error:
        broker, broker_src = pd.DataFrame(), f"broker ticks unavailable: {type(error).__name__}"
    refs: list = []
    try:
        if rules.get("reference_feed_enabled", True):
            refs = _reference_feeds(canonical, lo, hi, broker, rules)
    except Exception as error:
        notes.append(f"Reference feeds unavailable: {type(error).__name__}: {error}")
    live = [(feed, ticks) for feed, ticks, _ in refs if len(ticks)]
    # Same rule as the scan (latency_reference._combine): with "all", the least
    # client-favourable covering feed decides; with "any", the most favourable.
    from webapp import latency_reference as lr
    mode = str(rules.get("reference_combine", 2)).strip().lower()
    if not mode.isdigit() and mode not in ("all", "majority", "any"):
        mode = "all"
    ref_metas = [m for _, _, m in refs]

    # USD value of a 1.0 price move on this trade: from its own realised P&L
    # when the price moved enough, else the instrument's contract size.
    cp = float(frame.at[0, "close_price"])
    move = d * (cp - p0) if np.isfinite(cp) else 0.0
    if lots > 0 and abs(move) / p0 > 5e-4:
        usd_per_unit = abs(float(frame.at[0, "net_profit"]) / move)
        usd_basis = "from this trade's realised P&L"
    else:
        from webapp.views import _contract_units
        usd_per_unit = _contract_units(str(frame.at[0, "symbol"])) * lots
        usd_basis = "contract size (approximate; assumes a USD-quoted instrument)"

    # The spec's seven horizons decide the classification. For the evidence
    # table and chart only (user-specified, 15 Sep 2026): fill, 100/200/300/
    # 500 ms, 1 s, then 5, 20, 30, 40, 50 and 60 s.
    spec_h = {float(h) for h in rules["horizons_seconds"]}
    horizons = sorted({0.0, *spec_h, 20.0, 30.0, 40.0, 50.0})
    thr = rules["flag_markout_bps"]
    rows = []
    for h in horizons:
        when = t0 + pd.to_timedelta(h, unit="s")
        label = "fill" if h == 0 else la._hlabel(h)
        sub = 0 < h < 1
        b = _at(broker, when)
        row = {"horizon": label, "seconds": h, "time": str(when)[:23],
               "measurable": not (sub and not precise), "spec": h == 0 or h in spec_h}
        if b and row["measurable"]:
            bmid = (b[0] + b[1]) / 2
            mo = d * (bmid - p0) / p0 * 1e4
            row.update(broker_bid=b[0], broker_ask=b[1], broker_mid=round(bmid, 6), broker_quote_time=str(b[2])[:23],
                       markout_bps=round(mo, 3), markout_usd=round(d * (bmid - p0) * usd_per_unit, 2))
        if row["measurable"]:
            per_feed = {}
            for feed, ticks in live:
                r = _at(ticks, when)
                if r:
                    exit_px = r[0] if d > 0 else r[1]
                    per_feed[feed] = {"bid": round(r[0], 6), "ask": round(r[1], 6),
                                      "exec_bps": round(d * (exit_px - p0) / p0 * 1e4, 3),
                                      "mid_bps": round(d * ((r[0] + r[1]) / 2 - p0) / p0 * 1e4, 3)}
                    row.update({f"ref_{feed}_{k}": v for k, v in per_feed[feed].items()})
            if per_feed:
                # Combined reference: the deciding feed's quote and its
                # executable markout; the mid markout is the feeds' average.
                ranked = sorted(per_feed, key=lambda f: per_feed[f]["exec_bps"], reverse=True)
                k = lr.quorum_rank(len(ranked), mode)
                row["ref_mid_bps"] = round(float(np.mean([v["mid_bps"] for v in per_feed.values()])), 3)
                if k <= len(ranked):
                    decider = ranked[k - 1]
                    row.update(ref_bid=per_feed[decider]["bid"], ref_ask=per_feed[decider]["ask"],
                               ref_exec_bps=per_feed[decider]["exec_bps"], ref_feed=decider)
        # Thresholds only where they decide the flag: the sub-second window for
        # millisecond fills, 1 s and 5 s for fills stamped to the second.
        early_set = {float(e) for e in (rules.get("early_horizons_seconds") if precise
                                        else rules.get("fallback_early_horizons_seconds")) or []}
        if h > 0 and h in early_set:
            row["threshold_bps"] = thr.get(la._hkey(h))
            row["clears_threshold"] = (row.get("markout_bps") is not None and row["threshold_bps"] is not None
                                       and row["markout_bps"] >= float(row["threshold_bps"]))
        rows.append(row)

    # Engine classification on this one trade, identical to the scan.
    for row in rows[1:]:
        if row["spec"]:
            frame[la._mcol(row["seconds"])] = row.get("markout_bps", np.nan)
    frame["ms_fill"] = precise
    classified = la._classify_trades(frame.copy(), rules).iloc[0]
    stale_ms = float(rules.get("quote_throttle_ms", 200)) + float(rules.get("quote_age_tolerance_ms", 100))
    b0 = _at(broker, t0)
    lead = pd.to_timedelta(float(rules.get("reference_price_delay_ms", 500)), unit="ms")
    summary = {"precision": "millisecond" if precise else "second",
               "quote_age_ms": round((t0 - b0[2]).total_seconds() * 1000, 1) if b0 else None,
               "stale_quote_threshold_ms": stale_ms, "reference_combine": mode,
               "reference_combine_label": lr.quorum_label(mode)}
    summary["stale_quote"] = bool(precise and summary["quote_age_ms"] is not None and summary["quote_age_ms"] > stale_ms)
    thr_div = float(rules.get("reference_divergence_bps", 1.0))
    thr_lead = float(rules.get("reference_lead_bps", 1.0))
    gaps, moves = {}, {}
    bl = _at(broker, t0 - lead)
    if b0:
        bmid0 = (b0[0] + b0[1]) / 2
        for feed, ticks in live:
            r0 = _at(ticks, t0)
            # Same freshness rule as the scan: a reference quote within 5 s.
            if not r0 or r0[2] < t0 - pd.Timedelta(seconds=5):
                continue
            rmid0 = (r0[0] + r0[1]) / 2
            gaps[feed] = round(d * (rmid0 - bmid0) / p0 * 1e4, 3)
            rl = _at(ticks, t0 - lead)
            if bl and rl:
                moves[feed] = round(d * (rmid0 - (rl[0] + rl[1]) / 2) / p0 * 1e4, 3)
    if gaps:
        summary["gap_to_reference_bps"] = lr.quorum_value(gaps.values(), mode)
        summary["gap_by_reference_bps"] = gaps
        summary["behind_reference"] = summary["gap_to_reference_bps"] >= thr_div
    if moves:
        summary["reference_move_before_fill_bps"] = lr.quorum_value(moves.values(), mode)
        summary["reference_move_by_reference_bps"] = moves
        summary["broker_move_before_fill_bps"] = round(d * (bmid0 - (bl[0] + bl[1]) / 2) / p0 * 1e4, 3)
        summary["reference_led"] = (summary["reference_move_before_fill_bps"] >= thr_lead
                                    and summary["broker_move_before_fill_bps"] <= 0.25 * summary["reference_move_before_fill_bps"])
    early_h = {float(e) for e in (rules.get("early_horizons_seconds") if precise
                                  else rules.get("fallback_early_horizons_seconds")) or []}
    early = [r for r in rows[1:] if r["spec"] and r["seconds"] in early_h]
    late_row = next(r for r in rows if r["seconds"] == max(spec_h))
    peak = max((r["markout_bps"] for r in early if r.get("markout_bps") is not None), default=None)
    late = late_row.get("markout_bps")
    summary.update(early_peak_bps=peak, markout_60s_bps=late,
                   # Only a FADING advantage has a decay; one that held or grew has none.
                   decay_pct=round((1 - late / peak) * 100, 1) if peak and peak > 0 and late is not None and late < peak else None,
                   hold_seconds=round(float(frame.at[0, "hold_seconds"]), 3) if pd.notna(frame.at[0, "hold_seconds"]) else None,
                   latency_flag=bool(classified["flagged"]), directional=bool(classified["directional"]))
    def _span(hs):
        hs = sorted(hs)
        fmt = lambda h: f"{int(round(h * 1000))} ms" if h < 1 else f"{h:g} s"
        return fmt(hs[0]) if len(hs) == 1 else f"{fmt(hs[0])} – {fmt(hs[-1])}"
    window = _span(early_h) if early_h else "the early window"
    if summary["latency_flag"]:
        verdict = (f"Latency-type trade: a favourable move of {peak:.2f} bps within {window} of the fill that faded to "
                   f"{late:.2f} bps by 60 s, on a {summary['hold_seconds']:.0f} s hold.")
    elif summary["directional"]:
        # The engine compares the 60 s markout with the early peak only; the
        # path in between (visible in the 5 s slices) may dip and recover.
        peak_txt = f"an early peak of {peak:+.2f} bps within {window}" if peak is not None else "no measurable early peak"
        verdict = (f"Directional trade: the advantage had not faded by 60 s ({late:+.2f} bps at 60 s against "
                   f"{peak_txt}) — not a latency pattern.")
    elif peak is None:
        verdict = "Not measurable: no broker ticks cover this fill."
    else:
        verdict = (f"No latency signature: early markout peaked at {peak:.2f} bps within {window}, "
                   "below the engine's thresholds or without the fading shape.")
    n_gap = len(summary.get("gap_by_reference_bps") or {})
    which = lr.quorum_label(mode).replace("venue", "independent reference venue") if n_gap > 1 \
        else "the independent reference"
    if summary.get("behind_reference"):
        at_least = "at least " if n_gap > 1 and lr.quorum_rank(n_gap, mode) > 1 else ""
        verdict += (f" At the fill the broker price was {at_least}{summary['gap_to_reference_bps']:.2f} bps "
                    f"behind {which}.")
    if summary.get("reference_led"):
        markets = "markets" if len(summary.get("reference_move_by_reference_bps") or {}) > 1 else "market"
        verdict += f" The reference {markets} had already moved before the fill while the broker quote had not."
    if summary["stale_quote"]:
        verdict += f" The broker quote was {summary['quote_age_ms']:.0f} ms old, beyond the {stale_ms:.0f} ms throttle allowance."
    summary["verdict"] = verdict

    out = {"account": account_key, "order": None if pd.isna(trade.get("order")) else str(trade.get("order")),
           "trade": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in frame.iloc[0][
               ["database", "symbol", "cmd", "volume_lots", "open_time", "close_time", "open_price",
                "close_price", "net_profit", "hold_seconds"]].to_dict().items()},
           "trade_source": source, "rows": rows, "summary": summary,
           "sources": {"broker": broker_src, "broker_ticks": int(len(broker)),
                       # `reference` keeps the first feed for older readers.
                       "reference": ref_metas[0] if ref_metas else {"note": "reference feed disabled"},
                       "references": ref_metas,
                       "reference_ticks": int(sum(len(t) for _, t in live)), "usd_conversion": usd_basis},
           "method": ("Markout = direction × (price at horizon − fill price) ÷ fill price, in basis points. Broker markout "
                      "uses the broker mid; reference markout uses the reference's executable opposite side (a buy exits "
                      "at the reference bid, a sell at the ask), shifted by the day's median broker−reference price "
                      "difference, for each independent reference. "
                      + f"A reference signal counts when {lr.quorum_label(mode)} covering the fill show it; the "
                        "combined columns use the value that many venues reach. "
                      + "Times are UTC. Sub-second horizons are shown only for millisecond-stamped fills."),
           "thresholds": {"flag_markout_bps": thr, "min_decay_pct": rules.get("min_decay_pct"),
                          "max_hold_seconds": rules.get("max_hold_seconds")},
           "notes": notes, "generated_at": str(datetime.utcnow())[:19]}
    out = _jsonable(out)
    # Cache only complete evidence: a transient DB or reference failure must
    # not pin an empty table for ten minutes.
    if len(broker) and len(live) == len(refs) and live:
        _CACHE[key] = (time.time(), out)
    if len(_CACHE) > 300:
        _CACHE.clear()
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NaT:
        return None
    return value
