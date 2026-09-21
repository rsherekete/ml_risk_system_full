"""Production trade feed from MySQL -- the bridge until prod Kafka credentials.

WHY THIS IS SAFE FOR THE BUSINESS

Every query here goes to `ld4-dbproxy` -- the reporting path this project has
already read two-year backfills through -- never to the MT4/MT5 trade servers
themselves. The trade servers journal to these databases regardless of who
reads them; a reader cannot slow order execution or touch a client's session.
On top of that, every poll is a single indexed range scan (`time > cursor` on
MT5 deals, `open_ts > cursor` on MT4 orders), LIMITed, one short-lived
statement per server every few seconds -- load in the noise floor next to the
backfills this proxy already served without complaint. Connect and read
timeouts are tight so a slow proxy degrades THIS feed, never the other way
round.

WHY IT COEXISTS WITH KAFKA UNCHANGED

The engine de-duplicates on trade identity (account, symbol, side, lots,
price, second). A deal seen via MySQL now and via Kafka later -- or both, once
production credentials arrive -- collapses to one decision. So this feed can
simply run alongside the consumer: whichever source delivers a trade first
wins, and nothing double-fires.

LAG

MySQL trails the trade server by seconds. The engine's entry-quality rule
absorbs that structurally: a signal whose price has already run away fails
"enter at or better than the client" and is blocked, so lag can only cost
missed trades, never bad fills.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

#: Per-server state: cursor + a reusable connection. One lock; polls are quick.
_LOCK = threading.Lock()
_CURSORS: dict[str, object] = {}
_CONNECTIONS: dict[str, object] = {}
_FAILURES: dict[str, float] = {}

#: After a failure, leave that server alone this long before retrying --
#: hammering a struggling proxy is exactly the impact to avoid.
FAILURE_BACKOFF_SECONDS = 60.0

#: Cent-account logins per server, cached hourly. A cent account denominates
#: in 1/100 units: its lots and profit arrive inflated 100x, so copying one at
#: face value oversizes by two orders of magnitude and its mirror P&L
#: mis-scales the same way. Membership comes from the broker's own group
#: naming (`%cent%`), the authoritative marker.
_CENT: dict[str, tuple[float, set]] = {}
_CENT_TTL = 3600.0


def _cent_disk_path():
    from pathlib import Path
    return Path(__file__).resolve().parent / "artifacts" / "cent_logins.json"


def _cent_disk_load(server: str) -> set:
    try:
        import json
        stored = json.loads(_cent_disk_path().read_text(encoding="utf-8"))
        return {int(login) for login in stored.get(server, [])}
    except Exception:
        return set()


def _cent_disk_save(server: str, logins: set) -> None:
    try:
        import json
        path = _cent_disk_path()
        stored = {}
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
        stored[server] = sorted(logins)
        path.write_text(json.dumps(stored), encoding="utf-8")
    except Exception:
        pass


def cent_logins(server: str) -> set:
    stamp, cached = _CENT.get(server, (0.0, set()))
    if time.time() - stamp < _CENT_TTL:
        return cached
    logins: set = set()
    try:
        connection = _connection(server)
        # AUTHORITATIVE: the account's group joined to the `groups` config
        # table, whose `currency` column says 'CNT' for the cent tier. Group
        # NAMES are not reliable -- MT5 cent groups happen to contain
        # `Cnt`/`CNT` path segments, but MT4 cent groups are named like
        # `RCm_00_00` with no such marker, so a name pattern found 0 of
        # MT4's 21,328 cent logins. Verified identical counts against
        # `congroup` on both platforms (mt5_live01: 7,782 / mt4_live01:
        # 21,328).
        with connection.cursor() as statement:
            try:
                statement.execute(
                    "SELECT DISTINCT a.login FROM `accounts` a "
                    "JOIN (SELECT DISTINCT `group` g FROM `groups` "
                    "      WHERE currency = 'CNT') x ON a.`group` = x.g")
                logins = {int(r[0]) for r in statement.fetchall()}
            except Exception:
                # fallback for a server without a `groups` table: the
                # (case-sensitive-collation) name pattern
                statement.execute(
                    "SELECT login FROM `accounts` WHERE UPPER(`group`) "
                    "LIKE '%CNT%' OR UPPER(`group`) LIKE '%CENT%'")
                logins = {int(r[0]) for r in statement.fetchall()}
    except Exception:
        # A VPN blip must NOT cache an empty set for an hour -- that is how a
        # cent client's trade went through at 100x size. Serve the last known
        # set from disk and retry the live query in 60 seconds.
        logins = _cent_disk_load(server)
        _CENT[server] = (time.time() - _CENT_TTL + 60.0, logins)
        return logins
    _CENT[server] = (time.time(), logins)
    _cent_disk_save(server, logins)
    return logins


#: The money AND lot columns of a cent account are denominated in cents / cent-
#: lots -- a factor of 100 above their dollar / standard-lot values. Deflate BOTH
#: (verified against the copy-trader's own execution path): a cent client's raw
#: "1.00 lot" is 0.01 real lots and its raw "$500 profit" is $5. Applying this at
#: every point raw MySQL values enter a financial calculation is the ONLY way the
#: numbers are right; ~86k of the population are cent accounts, so skipping it
#: inflates exposure, P&L and every model feature built from money by ~100x on
#: those rows. THE authoritative classifier is cent_logins() (groups.currency).
def deflate_cent(frame, server: str, login_col: str = "login",
                 money_cols=(), lot_cols=()):
    """Divide the named money/lot columns by 100 on cent-account rows, in place.

    `server` is a single MySQL database; for a multi-server frame call once per
    server or use deflate_cent_multi(). Returns the frame for chaining."""
    import pandas as pd
    if frame is None or len(frame) == 0 or login_col not in frame.columns:
        return frame
    cents = cent_logins(server)
    if not cents:
        return frame
    logins = pd.to_numeric(frame[login_col], errors="coerce")
    mask = logins.isin(cents).to_numpy()
    if not mask.any():
        return frame
    for col in list(money_cols) + list(lot_cols):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                dtype="float64", copy=True)
            values[mask] = values[mask] / 100.0
            frame[col] = values
    return frame


def deflate_cent_multi(frame, server_col: str = "database", login_col: str = "login",
                       money_cols=(), lot_cols=()):
    """deflate_cent for a frame spanning several servers (has a server column)."""
    import pandas as pd
    if frame is None or len(frame) == 0 or server_col not in frame.columns:
        return frame
    parts = []
    for server, group in frame.groupby(server_col, observed=True):
        parts.append(deflate_cent(group.copy(), str(server), login_col,
                                   money_cols=money_cols, lot_cols=lot_cols))
    return pd.concat(parts, ignore_index=True) if parts else frame


MT5_OPENINGS = """
    SELECT deal, login, symbol, action, volume, price, `time`
    FROM deals
    WHERE `time` > %s AND entry = 0 AND action IN (0, 1) AND volume > 0
    ORDER BY `time` LIMIT 300
"""
MT5_CLOSINGS = """
    SELECT deal, login, symbol, action, volume, price, profit, `time`
    FROM deals
    WHERE `time` > %s AND entry IN (1, 3) AND action IN (0, 1)
    ORDER BY `time` LIMIT 300
"""
MT4_OPENINGS = """
    SELECT `order`, login, symbol_name AS symbol, cmd, volume, open_price, open_ts
    FROM orders
    WHERE open_ts > %s AND cmd IN (0, 1) AND volume > 0 AND open_price > 0
    ORDER BY open_ts LIMIT 300
"""
MT4_CLOSINGS = """
    SELECT `order`, login, symbol_name AS symbol, cmd, volume, close_price, profit, close_ts
    FROM orders
    WHERE close_ts > %s AND close_ts > 0 AND cmd IN (0, 1)
    ORDER BY close_ts LIMIT 300
"""


def _servers() -> list[str]:
    from webapp.mysql_extract import MYSQL_DATABASES
    return list(MYSQL_DATABASES)


def _connection(server: str):
    """A kept-alive connection per server, tight timeouts, quiet reconnects."""
    connection = _CONNECTIONS.get(server)
    if connection is not None:
        try:
            connection.ping(reconnect=True)
            return connection
        except Exception:
            _CONNECTIONS.pop(server, None)
    from webapp.mysql_extract import connect_settings
    import pymysql
    connection = pymysql.connect(
        **connect_settings(server),
        connect_timeout=5, read_timeout=8, charset="utf8mb4")
    _CONNECTIONS[server] = connection
    return connection


def _query(server: str, sql: str, cursor_value) -> list[tuple]:
    connection = _connection(server)
    with connection.cursor() as statement:
        statement.execute(sql, (cursor_value,))
        return statement.fetchall()


def _rows_mt5(server: str, kind: str, rows: list[tuple], now) -> list[dict]:
    cents = cent_logins(server)
    out = []
    for row in rows:
        if kind == "open":
            _deal, login, symbol, action, volume, price, stamp = row
            profit = None
        else:
            _deal, login, symbol, action, volume, price, profit, stamp = row
        out.append({
            "server": server, "login": int(login), "symbol": str(symbol),
            "action": "DEAL_BUY" if int(action) == 0 else "DEAL_SELL",
            "volume": float(volume) / 10000.0 / (100.0 if int(login) in cents else 1.0),
            "price": float(price or 0),
            "profit": (float(profit) / (100.0 if int(login) in cents else 1.0)
                       if profit is not None else None),
            "event_time": pd.Timestamp(stamp), "ingested_at": now,
        })
    return out


def _rows_mt4(server: str, kind: str, rows: list[tuple], now) -> list[dict]:
    cents = cent_logins(server)
    out = []
    for row in rows:
        if kind == "open":
            _order, login, symbol, cmd, volume, price, stamp = row
            profit = None
        else:
            _order, login, symbol, cmd, volume, price, profit, stamp = row
        out.append({
            "server": server, "login": int(login), "symbol": str(symbol),
            "action": "DEAL_BUY" if int(cmd) == 0 else "DEAL_SELL",
            "volume": float(volume) / 100.0 / (100.0 if int(login) in cents else 1.0),
            "price": float(price or 0),
            "profit": (float(profit) / (100.0 if int(login) in cents else 1.0)
                       if profit is not None else None),
            # MT4 stores unix seconds.
            "event_time": pd.Timestamp(int(stamp), unit="s"), "ingested_at": now,
        })
    return out


def poll(lookback_seconds: float = 90.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(openings, closings) newer than each server's cursor, engine-shaped.

    Cursors advance to the newest row seen per server, so each trade is
    returned once. A failing server backs off rather than being re-hit.
    """
    now = datetime.utcnow()
    openings: list[dict] = []
    closings: list[dict] = []
    with _LOCK:
        for server in _servers():
            failed_at = _FAILURES.get(server, 0.0)
            if time.time() - failed_at < FAILURE_BACKOFF_SECONDS:
                continue
            is_mt5 = server.startswith("mt5")
            cursor = _CURSORS.get(server)
            if cursor is None:
                cursor = (now - timedelta(seconds=lookback_seconds)
                          if is_mt5 else
                          int(time.time() - lookback_seconds))
                _CURSORS[server] = cursor
            try:
                if is_mt5:
                    opened = _rows_mt5(server, "open",
                                       _query(server, MT5_OPENINGS, cursor), now)
                    closed = _rows_mt5(server, "close",
                                       _query(server, MT5_CLOSINGS, cursor), now)
                    stamps = [r["event_time"] for r in opened + closed]
                    if stamps:
                        _CURSORS[server] = max(stamps).to_pydatetime()
                else:
                    opened = _rows_mt4(server, "open",
                                       _query(server, MT4_OPENINGS, cursor), now)
                    closed = _rows_mt4(server, "close",
                                       _query(server, MT4_CLOSINGS, cursor), now)
                    seconds = [int(r["event_time"].timestamp()) for r in opened + closed]
                    if seconds:
                        _CURSORS[server] = max(seconds)
                openings += opened
                closings += closed
            except Exception:
                _FAILURES[server] = time.time()
                _CONNECTIONS.pop(server, None)
                continue

    def frame(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["server", "login", "symbol", "action",
                                         "volume", "price", "profit",
                                         "event_time", "ingested_at"])
        return pd.DataFrame(rows)

    return frame(openings), frame(closings)


# ---------------------------------------------------------------- live views
# The web app's "live" panels originally read the Kafka store, which on this
# UAT cluster is mostly idle -- so live screens showed days-old test flow while
# production traded thousands of times an hour four seconds away in MySQL.
# These produce the SAME shapes the stream aggregates return, so the handlers
# can switch source without any template changing.

import re as _re

#: Same instrument regardless of the suffix a server gives it (matches
#: exposure_policy._SUFFIX): XAUUSD, XAUUSDe, XAUUSDmin, XAUUSD247, XAUUSD+ all
#: move with gold. Treating them separately is what made the live scoring path
#: feed the model an unknown symbol with no market context for a whole "+" book.
_CANON_SUFFIX = _re.compile(r"(MIN|MICRO|247|PRO|ECN|RAW|[EXZCM])+$")
#: Cross-vendor index/oil spellings the training map carries under one name.
_CANON_ALIAS = {"SP500": "SPX500", "US500": "SPX500", "DJ30": "US30",
                "USTEC": "NAS100", "DE40": "GER40", "UKOUSD": "UKOIL"}
_CANON_CACHE: dict[str, str] = {}


def _canonical(symbol: str) -> str:
    """Canonical instrument name, self-contained (the previous import of
    trading_data.research silently fell back to a no-op, so XAUUSD+ etc. never
    canonicalised). Drops a broker dot-suffix (.i/.pro), non-alphanumerics
    (+, _), the size/ECN suffix, then applies the cross-vendor aliases."""
    key = str(symbol)
    cached = _CANON_CACHE.get(key)
    if cached is not None:
        return cached
    base = key.split(".")[0]
    raw = _re.sub(r"[^A-Z0-9]", "", base.upper())
    canon = _CANON_SUFFIX.sub("", raw) or raw
    canon = _CANON_ALIAS.get(canon, canon)
    if len(_CANON_CACHE) < 50000:
        _CANON_CACHE[key] = canon
    return canon


def recent_activity_mysql(hours: int = 24) -> list[dict]:
    """Hourly activity buckets across every server, straight from production."""
    buckets: dict[str, dict] = {}
    for server in _servers():
        try:
            connection = _connection(server)
            with connection.cursor() as statement:
                if server.startswith("mt5"):
                    statement.execute("""
                        SELECT DATE_FORMAT(`time`, '%%Y-%%m-%%dT%%H:00:00') AS bucket,
                               COUNT(*), COUNT(DISTINCT login),
                               SUM(CASE WHEN action IN (0,1) THEN volume ELSE 0 END) / 10000,
                               SUM(CASE WHEN action IN (0,1) THEN profit ELSE 0 END),
                               SUM(CASE WHEN action NOT IN (0,1) THEN profit ELSE 0 END)
                        FROM deals WHERE `time` >= UTC_TIMESTAMP() - INTERVAL %s HOUR
                        GROUP BY 1""", (hours,))
                else:
                    statement.execute("""
                        SELECT DATE_FORMAT(FROM_UNIXTIME(open_ts), '%%Y-%%m-%%dT%%H:00:00'),
                               COUNT(*), COUNT(DISTINCT login),
                               SUM(CASE WHEN cmd IN (0,1) THEN volume ELSE 0 END) / 100,
                               0, 0
                        FROM orders
                        WHERE open_ts >= UNIX_TIMESTAMP(UTC_TIMESTAMP()) - %s * 3600
                        GROUP BY 1""", (hours,))
                for bucket, events, accounts, volume, profit, cash in statement.fetchall():
                    entry = buckets.setdefault(str(bucket), {
                        "bucket": str(bucket), "events": 0, "accounts": 0,
                        "volume": 0.0, "profit": 0.0, "cash_flow": 0.0})
                    entry["events"] += int(events or 0)
                    entry["accounts"] += int(accounts or 0)
                    entry["volume"] += float(volume or 0)
                    entry["profit"] += float(profit or 0)
                    entry["cash_flow"] += float(cash or 0)
        except Exception:
            _FAILURES[server] = time.time()
            _CONNECTIONS.pop(server, None)
    return sorted(buckets.values(), key=lambda b: b["bucket"])


def top_exposure_mysql(limit: int = 25, hours: int = 24) -> list[dict]:
    """Most-traded symbols over the window, across every production server."""
    per_symbol: dict[str, dict] = {}
    for server in _servers():
        try:
            connection = _connection(server)
            with connection.cursor() as statement:
                if server.startswith("mt5"):
                    statement.execute("""
                        SELECT symbol, COUNT(*), COUNT(DISTINCT login),
                               SUM(volume) / 10000, SUM(profit)
                        FROM deals
                        WHERE `time` >= UTC_TIMESTAMP() - INTERVAL %s HOUR
                          AND action IN (0,1) AND symbol <> ''
                        GROUP BY 1""", (hours,))
                else:
                    statement.execute("""
                        SELECT symbol_name, COUNT(*), COUNT(DISTINCT login),
                               SUM(volume) / 100, SUM(profit)
                        FROM orders
                        WHERE open_ts >= UNIX_TIMESTAMP(UTC_TIMESTAMP()) - %s * 3600
                          AND cmd IN (0,1)
                        GROUP BY 1""", (hours,))
                for symbol, events, accounts, volume, profit in statement.fetchall():
                    entry = per_symbol.setdefault(str(symbol), {
                        "symbol": str(symbol), "events": 0, "accounts": 0,
                        "volume": 0.0, "profit": 0.0})
                    entry["events"] += int(events or 0)
                    entry["accounts"] += int(accounts or 0)
                    entry["volume"] += float(volume or 0)
                    entry["profit"] += float(profit or 0)
        except Exception:
            _FAILURES[server] = time.time()
            _CONNECTIONS.pop(server, None)
    return sorted(per_symbol.values(), key=lambda r: -r["volume"])[:limit]


def symbol_var_mysql(horizons: tuple[int, ...] = (1, 5, 20),
                     days: int = 7) -> list[dict]:
    """Per-instrument historical VaR from production daily client P&L.

    Same construction and keys as the stream version: firm P&L is the negative
    of client P&L, VaR is the 5th percentile of daily firm P&L, scaled by
    sqrt(horizon) -- with the same stated caveat that clustering makes this a
    floor, not a ceiling.
    """
    by_symbol: dict[str, dict[str, list]] = {}
    for server in _servers():
        try:
            connection = _connection(server)
            with connection.cursor() as statement:
                if server.startswith("mt5"):
                    statement.execute("""
                        SELECT symbol, DATE(`time`), SUM(profit), SUM(volume) / 10000,
                               COUNT(*), COUNT(DISTINCT login)
                        FROM deals
                        WHERE `time` >= UTC_TIMESTAMP() - INTERVAL %s DAY
                          AND action IN (0,1) AND entry IN (1,3) AND symbol <> ''
                        GROUP BY 1, 2""", (days,))
                else:
                    statement.execute("""
                        SELECT symbol_name, DATE(FROM_UNIXTIME(close_ts)), SUM(profit),
                               SUM(volume) / 100, COUNT(*), COUNT(DISTINCT login)
                        FROM orders
                        WHERE close_ts >= UNIX_TIMESTAMP(UTC_TIMESTAMP()) - %s * 86400
                          AND close_ts > 0 AND cmd IN (0,1)
                        GROUP BY 1, 2""", (days,))
                for symbol, day, pnl, volume, events, accounts in statement.fetchall():
                    canonical = _canonical(symbol)
                    slot = by_symbol.setdefault(canonical, {})
                    row = slot.setdefault(str(day), [0.0, 0.0, 0, 0])
                    row[0] += float(pnl or 0)
                    row[1] += float(volume or 0)
                    row[2] += int(events or 0)
                    row[3] += int(accounts or 0)
        except Exception:
            _FAILURES[server] = time.time()
            _CONNECTIONS.pop(server, None)

    results = []
    for canonical, days_map in by_symbol.items():
        daily = list(days_map.values())
        series = sorted(-value[0] for value in daily)   # firm = -client
        if not series:
            continue
        index = max(0, int(len(series) * 0.05) - 1)
        var_1d = series[index] if len(series) >= 3 else series[0]
        entry = {"symbol": canonical, "days": len(series),
                 "firm_pnl": float(sum(series)),
                 "volume": float(sum(v[1] for v in daily)),
                 "events": int(sum(v[2] for v in daily)),
                 "accounts": int(max(v[3] for v in daily))}
        for horizon in horizons:
            entry[f"var_{horizon}d"] = float(var_1d * (horizon ** 0.5))
        results.append(entry)
    return sorted(results, key=lambda r: -abs(r["firm_pnl"]))


def health() -> dict:
    """Which servers the feed is currently reading, and which are backing off."""
    with _LOCK:
        cursors = {server: str(value) for server, value in _CURSORS.items()}
        cooling = {server: round(FAILURE_BACKOFF_SECONDS - (time.time() - failed), 0)
                   for server, failed in _FAILURES.items()
                   if time.time() - failed < FAILURE_BACKOFF_SECONDS}
    return {"servers": _servers(), "cursors": cursors, "backing_off": cooling}

