"""Replay: the biggest / most suspicious recent client wins, as playable cards.

A client win is a firm LOSS, so the biggest wins are exactly what surveillance
wants to watch back frame-by-frame. This module pulls recent CLOSED trades from
every production server, deflates cent accounts, ranks them (by size, or by a
suspicion score that rewards large-and-fast round trips), and returns compact
cards. Each card carries an account snapshot (balance, floating equity, and the
symbol's net exposure in lots and canonical USD) and enough trade detail for the
front-end to replay the entry/exit over minute bars from the tick store.

Everything is cached with a short TTL so the "new content every 5 minutes"
refresh is cheap and every viewer shares one pull rather than hammering MySQL.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from webapp.mysql_extract import MYSQL_DATABASES

# Canonical contract sizes for USD exposure where the table doesn't carry one
# (MT4 orders have no contract_size column). Standard MT sizes; FX majors 100k,
# metals/indices/energy per broker spec. Anything unlisted falls back to 100k.
_CONTRACT_SIZE = {
    "XAUUSD": 100.0, "XAGUSD": 5000.0, "XBRUSD": 1000.0, "UKOIL": 1000.0,
    "USOIL": 1000.0, "WTIUSD": 1000.0, "US30": 1.0, "SPX500": 1.0,
    "NAS100": 1.0, "GER40": 1.0, "UK100": 1.0, "JP225": 1.0, "AUS200": 1.0,
    "BTCUSD": 1.0, "ETHUSD": 1.0,
}
_DEFAULT_CONTRACT = 100000.0

_CACHE: dict = {}                       # key -> (fetched_at, payload)
_LOCK = threading.Lock()
_TTL = 290.0                            # ~5 min, matching the refresh cadence


def _servers() -> list[str]:
    return list(MYSQL_DATABASES)


def _canon(symbol: str) -> str:
    from webapp import trade_feed as tf
    try:
        return tf._canonical(str(symbol))
    except Exception:
        return str(symbol)


def _contract_size(canon: str) -> float:
    return _CONTRACT_SIZE.get(canon, _DEFAULT_CONTRACT)


# --------------------------------------------------------------- movers pull
def _mt4_recent(server: str, since_epoch: int, cap: int) -> list[dict]:
    from webapp import trade_feed as tf
    con = tf._connection(server)
    cents = tf.cent_logins(server)
    rows = []
    with con.cursor() as cur:
        cur.execute(
            "SELECT login, symbol_name, cmd, volume, open_price, open_ts, "
            "close_price, close_ts, profit FROM orders "
            "WHERE close_ts > %s AND cmd IN (0,1) AND open_price > 0 "
            "ORDER BY ABS(profit) DESC LIMIT %s", [since_epoch, cap])
        for (login, sym, cmd, vol, op, ots, cp, cts, profit) in cur.fetchall():
            scale = 100.0 if int(login) in cents else 1.0
            rows.append({
                "account_key": f"{server}:{login}", "symbol_raw": str(sym),
                "symbol": _canon(sym), "side": "buy" if int(cmd) == 0 else "sell",
                "lots": float(vol) / 100.0 / scale,
                "open_price": float(op), "close_price": float(cp),
                "open_time": pd.Timestamp(int(ots), unit="s"),
                "close_time": pd.Timestamp(int(cts), unit="s"),
                "pnl": float(profit) / scale})
    return rows


def _mt5_recent(server: str, since_dt, cap: int) -> list[dict]:
    from webapp import trade_feed as tf
    con = tf._connection(server)
    cents = tf.cent_logins(server)
    # Closing deals (entry 1=out, 3=out-by) carry the round-trip profit.
    with con.cursor() as cur:
        cur.execute(
            "SELECT position_id, login, symbol, action, volume, price, `time`, profit "
            "FROM deals WHERE `time` > %s AND entry IN (1,3) AND action IN (0,1) "
            "ORDER BY ABS(profit) DESC LIMIT %s", [since_dt, cap])
        closers = cur.fetchall()
    if not closers:
        return []
    pids = tuple({int(r[0]) for r in closers if r[0]})
    opens: dict = {}
    if pids:
        placeholders = ",".join(["%s"] * len(pids))
        with con.cursor() as cur:
            cur.execute(
                f"SELECT position_id, price, `time`, volume FROM deals "
                f"WHERE entry = 0 AND position_id IN ({placeholders})", list(pids))
            for (pid, prc, when, vol) in cur.fetchall():
                opens[int(pid)] = (float(prc), pd.Timestamp(when))
    rows = []
    for (pid, login, sym, action, vol, prc, when, profit) in closers:
        scale = 100.0 if int(login) in cents else 1.0
        # a long is closed by a SELL deal (action 1); a short by a BUY (action 0)
        side = "buy" if int(action) == 1 else "sell"
        op, ots = opens.get(int(pid or 0), (float(prc), pd.NaT))
        rows.append({
            "account_key": f"{server}:{login}", "symbol_raw": str(sym),
            "symbol": _canon(sym), "side": side,
            "lots": float(vol) / 10000.0 / scale,
            "open_price": op, "close_price": float(prc),
            "open_time": ots, "close_time": pd.Timestamp(when),
            "pnl": float(profit or 0) / scale})
    return rows


def _pull_all(minutes: int, cap_per_server: int) -> pd.DataFrame:
    since = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=minutes)
    rows: list[dict] = []
    for server in _servers():
        try:
            if server.startswith("mt4"):
                rows += _mt4_recent(server, int(since.timestamp()), cap_per_server)
            else:
                rows += _mt5_recent(server, since.to_pydatetime(), cap_per_server)
        except Exception:
            continue                    # one dead server never blanks the board
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["duration_min"] = (
        (frame["close_time"] - frame["open_time"]).dt.total_seconds() / 60.0)
    frame["usd"] = frame.apply(
        lambda r: abs(r["lots"]) * _contract_size(r["symbol"])
        * (r["close_price"] or 0), axis=1)
    return frame


def _suspicion(frame: pd.DataFrame) -> pd.Series:
    """A large, fast, PROFITABLE-for-the-client round trip is the toxic shape.
    Rank-normalise each component so no single scale dominates, then combine."""
    if frame.empty:
        return pd.Series([], dtype=float)
    size = frame["pnl"].clip(lower=0).rank(pct=True)          # only wins score
    dur = frame["duration_min"].fillna(1e9)
    speed = (1.0 / (1.0 + dur)).rank(pct=True)                # faster = higher
    notional = frame["usd"].rank(pct=True)
    return (0.5 * size + 0.3 * speed + 0.2 * notional).astype(float)


# --------------------------------------------------------------- public API
def board(mode: str = "wins", minutes: int = 1440, limit: int = 30) -> dict:
    """Top cards for the replay wall. mode='wins' ranks by client profit (our
    loss); mode='suspicious' by the toxic-shape score. Cached ~5 min."""
    key = ("board", mode, minutes)
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _TTL:
            return _slice(hit[1], limit)
    frame = _pull_all(minutes, cap_per_server=400)
    payload = {"generated": now, "window_minutes": minutes, "cards": []}
    if not frame.empty:
        if mode == "suspicious":
            frame = frame.assign(score=_suspicion(frame)).sort_values(
                "score", ascending=False)
        else:                            # biggest client wins first
            frame = frame.sort_values("pnl", ascending=False)
        cards = []
        for r in frame.head(120).itertuples():
            cards.append({
                "account_key": r.account_key, "symbol": r.symbol,
                "symbol_raw": r.symbol_raw, "side": r.side,
                "lots": round(float(r.lots), 2), "pnl": round(float(r.pnl), 2),
                "usd": round(float(r.usd), 0),
                "open_price": round(float(r.open_price), 5),
                "close_price": round(float(r.close_price), 5),
                "open_time": None if pd.isna(r.open_time) else r.open_time.strftime("%Y-%m-%d %H:%M"),
                "close_time": None if pd.isna(r.close_time) else r.close_time.strftime("%Y-%m-%d %H:%M"),
                "duration_min": None if pd.isna(r.duration_min) else round(float(r.duration_min), 1),
                "score": round(float(getattr(r, "score", 0.0)), 3) if mode == "suspicious" else None,
            })
        payload["cards"] = cards
    with _LOCK:
        _CACHE[key] = (now, payload)
    return _slice(payload, limit)


def _slice(payload: dict, limit: int) -> dict:
    out = dict(payload)
    out["cards"] = payload["cards"][:limit]
    out["total"] = len(payload["cards"])
    return out


def account_snapshot(account_key: str) -> dict:
    """Balance, floating equity, and per-symbol net exposure (lots + canonical
    USD) from the account's LIVE open positions. Best-effort per platform."""
    from webapp import trade_feed as tf
    try:
        server, login = account_key.split(":", 1)
        login = int(login)
    except Exception:
        return {}
    cents = tf.cent_logins(server)
    scale = 100.0 if login in cents else 1.0
    con = tf._connection(server)
    snap: dict = {"account_key": account_key, "balance": None, "equity": None,
                  "floating": None, "open_positions": 0, "exposure": []}
    try:
        with con.cursor() as cur:
            cur.execute("SELECT balance FROM accounts WHERE login = %s", [login])
            row = cur.fetchone()
            if row and row[0] is not None:
                snap["balance"] = float(row[0]) / scale
    except Exception:
        pass
    exposure: dict = {}                  # canonical -> [signed_lots, usd, floating]
    floating = 0.0
    try:
        if server.startswith("mt5"):
            with con.cursor() as cur:
                cur.execute(
                    "SELECT symbol, action, volume, price_current, contract_size, "
                    "profit FROM positions WHERE login = %s", [login])
                for (sym, action, vol, prc, csize, profit) in cur.fetchall():
                    lots = float(vol) / 10000.0 / scale
                    signed = lots if int(action) == 0 else -lots
                    canon = _canon(sym)
                    usd = signed * float(csize or _contract_size(canon)) * float(prc or 0)
                    slot = exposure.setdefault(canon, [0.0, 0.0, 0.0])
                    slot[0] += signed; slot[1] += usd
                    slot[2] += float(profit or 0) / scale
                    floating += float(profit or 0) / scale
                    snap["open_positions"] += 1
        else:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT symbol_name, cmd, volume, open_price, profit "
                    "FROM orders WHERE login = %s AND close_ts = 0 AND cmd IN (0,1)",
                    [login])
                for (sym, cmd, vol, op, profit) in cur.fetchall():
                    lots = float(vol) / 100.0 / scale
                    signed = lots if int(cmd) == 0 else -lots
                    canon = _canon(sym)
                    usd = signed * _contract_size(canon) * float(op or 0)
                    slot = exposure.setdefault(canon, [0.0, 0.0, 0.0])
                    slot[0] += signed; slot[1] += usd
                    slot[2] += float(profit or 0) / scale
                    floating += float(profit or 0) / scale
                    snap["open_positions"] += 1
    except Exception:
        pass
    snap["floating"] = round(floating, 2)
    if snap["balance"] is not None:
        snap["equity"] = round(snap["balance"] + floating, 2)
    snap["exposure"] = [
        {"symbol": k, "lots": round(v[0], 2), "usd": round(v[1], 0),
         "floating": round(v[2], 2)}
        for k, v in sorted(exposure.items(), key=lambda kv: -abs(kv[1][1]))]
    return snap


def trade_chart(account: str, symbol_raw: str, open_iso: str, close_iso: str) -> dict:
    """Minute bars tightly windowed around ONE trade, plus its entry/exit flags.

    Windowing to the trade (not the account's whole recent history) is the speed
    fix: a fast scalp needs ~30 bars, fetched in well under a second, instead of
    thousands over days. The bar width scales so the count stays playback-sized,
    and the whole result is cached ~5 min so re-expanding a card is instant."""
    from webapp import tick_bars
    key = ("chart", account, symbol_raw, open_iso, close_iso)
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _TTL:
            return hit[1]
    try:
        open_t = pd.Timestamp(open_iso) if open_iso else pd.NaT
        close_t = pd.Timestamp(close_iso) if close_iso else pd.NaT
    except Exception:
        open_t = close_t = pd.NaT
    if pd.isna(close_t):
        close_t = pd.Timestamp.utcnow().tz_localize(None)
    if pd.isna(open_t):
        open_t = close_t - pd.Timedelta(minutes=30)
    span = max(pd.Timedelta(minutes=20), (close_t - open_t))
    server = account.split(":", 1)[0]
    tick_server = server if server.startswith("mt4") else "mt4_live01"
    note = ""

    def _pull(a, b):
        try:
            return tick_bars.fetch_bars(tick_server, (str(symbol_raw),),
                                        a.to_pydatetime(), b.to_pydatetime(),
                                        chunk_hours=4, workers=8)
        except Exception:
            return pd.DataFrame()

    THREE_H = pd.Timedelta(hours=3)
    if span <= THREE_H:
        pad = max(pd.Timedelta(minutes=10), span * 0.25)
        bars = _pull(open_t - pad, close_t + pad)
    else:
        # Long hold: the entry and the exit are what matter. Pull a tight window
        # around each and stitch, rather than scanning days of ticks.
        wing = pd.Timedelta(minutes=45)
        entry_bars = _pull(open_t - pd.Timedelta(minutes=15), open_t + wing)
        exit_bars = _pull(close_t - wing, close_t + pd.Timedelta(minutes=15))
        bars = pd.concat([b for b in (entry_bars, exit_bars) if b is not None and len(b)],
                         ignore_index=True)
        if len(bars):
            bars = bars.drop_duplicates(subset=["minute"]).sort_values("minute")
        note = (f"long hold — middle of a {round(span.total_seconds()/3600,1)}h "
                f"position omitted; entry and exit windows shown")
    if bars is None or bars.empty:
        payload = {"bars": [], "trades": [],
                   "note": f"no tick bars stored for {symbol_raw} in that window"}
        with _LOCK:
            _CACHE[key] = (now, payload)
        return payload
    total_min = max(1.0, (bars["minute"].max() - bars["minute"].min()).total_seconds() / 60.0)
    bar_min = max(1, int(round(total_min / 280.0))) if span <= THREE_H else 1
    if bar_min > 1:
        bars = (bars.set_index("minute").resample(f"{bar_min}min")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last"}).dropna(subset=["close"]).reset_index())
    out_bars = [{"time": b.minute.strftime("%Y-%m-%d %H:%M"),
                 "open": float(b.open), "high": float(b.high),
                 "low": float(b.low), "close": float(b.close)}
                for b in bars.itertuples()]
    # Snap each flag to the nearest rendered bar so it always lands on the axis.
    times = [b["time"] for b in out_bars]

    def _snap(ts):
        if pd.isna(ts) or not times:
            return times[0] if times else None
        target = pd.Timestamp(ts)
        idx = min(range(len(bars)), key=lambda i: abs(bars["minute"].iloc[i] - target))
        return times[idx]

    markers = []
    if not pd.isna(open_t):
        markers.append({"time": _snap(open_t), "kind": "entry"})
    if not pd.isna(close_t):
        markers.append({"time": _snap(close_t), "kind": "exit"})
    payload = {"bars": out_bars, "trades": markers, "bar_min": bar_min, "note": note}
    with _LOCK:
        _CACHE[key] = (now, payload)
    return payload


def account_events(account: str, start_iso: str, end_iso: str) -> dict:
    """Every deal the CLIENT did in the replay window (extended back to catch
    positions already open at the start), so the front-end can reconstruct their
    balance / floating / net exposure at each playhead position and show how they
    managed the trade. Cent-deflated. Returns raw open/close events, the account's
    current balance as an anchor, and per-symbol contract sizes for USD math."""
    from webapp import trade_feed as tf
    try:
        server, login = account.split(":", 1)
        login = int(login)
    except Exception:
        return {"events": [], "balance": None}
    con = tf._connection(server)
    cents = tf.cent_logins(server)
    scale = 100.0 if login in cents else 1.0
    try:
        start = pd.Timestamp(start_iso)
        end = pd.Timestamp(end_iso) if end_iso else pd.Timestamp.utcnow().tz_localize(None)
    except Exception:
        return {"events": [], "balance": None}
    ext = start - pd.Timedelta(days=2)               # catch positions already open
    events = []
    try:
        if server.startswith("mt5"):
            with con.cursor() as cur:
                cur.execute(
                    "SELECT position_id, symbol, action, volume, price, `time`, entry, "
                    "profit, contract_size FROM deals WHERE login=%s AND `time`>=%s "
                    "AND `time`<=%s AND action IN (0,1) ORDER BY `time` LIMIT 4000",
                    [login, ext.to_pydatetime(), end.to_pydatetime()])
                for (pid, sym, action, vol, prc, when, entry, profit, csize) in cur.fetchall():
                    canon = _canon(sym)
                    events.append({
                        "position_id": int(pid or 0), "symbol": canon,
                        "direction": 1 if int(action) == 0 else -1,
                        "lots": float(vol) / 10000.0 / scale,
                        "kind": "in" if int(entry) == 0 else "out",
                        "price": float(prc or 0),
                        "profit": float(profit or 0) / scale,
                        "contract": float(csize or _contract_size(canon)),
                        "time": pd.Timestamp(when).strftime("%Y-%m-%d %H:%M:%S")})
        else:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT symbol_name, cmd, volume, open_price, open_ts, close_price, "
                    "close_ts, profit FROM orders WHERE login=%s AND cmd IN (0,1) "
                    "AND open_ts <= %s AND (close_ts = 0 OR close_ts >= %s) "
                    "ORDER BY open_ts LIMIT 4000",
                    [login, int(end.timestamp()), int(ext.timestamp())])
                for (sym, cmd, vol, op, ots, cp, cts, profit) in cur.fetchall():
                    canon = _canon(sym)
                    lots = float(vol) / 100.0 / scale
                    direction = 1 if int(cmd) == 0 else -1
                    contract = _contract_size(canon)
                    events.append({"position_id": 0, "symbol": canon, "direction": direction,
                                   "lots": lots, "kind": "in", "price": float(op or 0),
                                   "profit": 0.0, "contract": contract,
                                   "time": pd.Timestamp(int(ots), unit="s").strftime("%Y-%m-%d %H:%M:%S")})
                    if cts and int(cts) > 0:
                        events.append({"position_id": 0, "symbol": canon, "direction": direction,
                                       "lots": lots, "kind": "out", "price": float(cp or 0),
                                       "profit": float(profit or 0) / scale, "contract": contract,
                                       "time": pd.Timestamp(int(cts), unit="s").strftime("%Y-%m-%d %H:%M:%S")})
    except Exception as error:
        return {"events": [], "balance": None, "note": f"{type(error).__name__}: {error}"}
    events.sort(key=lambda e: e["time"])
    balance = None
    try:
        with con.cursor() as cur:
            cur.execute("SELECT balance FROM accounts WHERE login=%s", [login])
            row = cur.fetchone()
            if row and row[0] is not None:
                balance = float(row[0]) / scale
    except Exception:
        pass
    return {"events": events, "balance": balance, "window_start": start_iso}


def trade_by_ticket(ticket: str) -> dict:
    """Resolve a TRADE TICKET to one round-trip + its tick chart -- the
    hub's Trade Zoom. Accepts our own MT5 position ticket (via the vantage
    orders ledger) OR a raw source ticket (MT4 order number / MT5
    position_id), searching every server. Returns {card, chart}."""
    ticket = (ticket or "").strip()
    if not ticket.isdigit():
        return {"error": "ticket must be a number"}
    tk = int(ticket)
    from webapp import trade_feed as tf

    # 1) our own ticket first -- the ledger maps it to the client source.
    try:
        import sqlite3
        from pathlib import Path as _P
        cx = sqlite3.connect(_P(__file__).resolve().parent / "app.db")
        cx.row_factory = sqlite3.Row
        led = cx.execute("SELECT source_account, symbol FROM vantage_orders "
                         "WHERE ticket = ? ORDER BY id DESC LIMIT 1",
                         (tk,)).fetchone()
    except Exception:
        led = None
    src_hint = led["source_account"] if led else None

    def _find_mt4(server):
        con = tf._connection(server); cents = tf.cent_logins(server)
        with con.cursor() as cur:
            cur.execute(
                "SELECT login, symbol_name, cmd, volume, open_price, open_ts, "
                "close_price, close_ts, profit FROM orders WHERE `order` = %s "
                "AND cmd IN (0,1) LIMIT 1", [tk])
            r = cur.fetchone()
        if not r:
            return None
        login, sym, cmd, vol, op, ots, cp, cts, profit = r
        scale = 100.0 if int(login) in cents else 1.0
        return {"account_key": f"{server}:{login}", "symbol_raw": str(sym),
                "symbol": _canon(sym), "side": "buy" if int(cmd) == 0 else "sell",
                "lots": float(vol) / 100.0 / scale,
                "open_price": float(op), "close_price": float(cp),
                "open_time": pd.Timestamp(int(ots), unit="s"),
                "close_time": pd.Timestamp(int(cts), unit="s"),
                "pnl": float(profit) / scale}

    def _find_mt5(server):
        con = tf._connection(server); cents = tf.cent_logins(server)
        with con.cursor() as cur:
            cur.execute(
                "SELECT position_id, login, symbol, action, volume, price, "
                "`time`, profit, entry FROM deals WHERE position_id = %s "
                "AND action IN (0,1) ORDER BY `time`", [tk])
            deals = cur.fetchall()
        if not deals:
            return None
        opn = next((d for d in deals if int(d[8]) == 0), None)
        clo = next((d for d in deals if int(d[8]) in (1, 3)), deals[-1])
        login = clo[1]; scale = 100.0 if int(login) in cents else 1.0
        return {"account_key": f"{server}:{login}", "symbol_raw": str(clo[2]),
                "symbol": _canon(clo[2]),
                "side": "buy" if int(clo[3]) == 1 else "sell",
                "lots": float(clo[4]) / 10000.0 / scale,
                "open_price": float(opn[5]) if opn else float(clo[5]),
                "close_price": float(clo[5]),
                "open_time": pd.Timestamp(opn[6]) if opn else pd.NaT,
                "close_time": pd.Timestamp(clo[6]),
                "pnl": float(clo[7] or 0) / scale}

    order = _servers()
    if src_hint and ":" in str(src_hint):        # check the hinted server first
        sh = str(src_hint).split(":")[0]
        order = [sh] + [s for s in order if s != sh]
    card = None
    for server in order:
        try:
            card = _find_mt4(server) if server.startswith("mt4") \
                else _find_mt5(server)
        except Exception:
            card = None
        if card:
            break
    if not card:
        return {"error": f"ticket {tk} not found on any server"}
    card["usd"] = round(abs(card["lots"]) * _contract_size(card["symbol"])
                        * (card["close_price"] or 0), 0)
    card["duration_min"] = (None if pd.isna(card["open_time"])
                            else round((card["close_time"]
                                        - card["open_time"]).total_seconds()
                                       / 60.0, 1))
    ot = None if pd.isna(card["open_time"]) else card["open_time"].strftime("%Y-%m-%d %H:%M")
    ct = None if pd.isna(card["close_time"]) else card["close_time"].strftime("%Y-%m-%d %H:%M")
    chart = trade_chart(card["account_key"], card["symbol_raw"], ot or "", ct or "")
    return {"card": {k: (None if (isinstance(v, float) and pd.isna(v))
                         else (v.strftime("%Y-%m-%d %H:%M")
                               if hasattr(v, "strftime") else v))
                     for k, v in card.items()},
            "chart": chart}


def resolve_login(q: str) -> dict:
    """Map a bare login (or a full server:login) to account_key(s). Searches
    every server's accounts table so the hub's Client Zoom accepts just a number."""
    q = (q or "").strip()
    if ":" in q:
        return {"account_key": q}
    if not q.isdigit():
        return {"candidates": []}
    from webapp import trade_feed as tf
    login = int(q)
    hits = []
    for server in _servers():
        try:
            con = tf._connection(server)
            with con.cursor() as cur:
                cur.execute("SELECT 1 FROM accounts WHERE login = %s LIMIT 1", [login])
                if cur.fetchone():
                    hits.append(f"{server}:{login}")
        except Exception:
            continue
    if len(hits) == 1:
        return {"account_key": hits[0]}
    return {"candidates": hits}
