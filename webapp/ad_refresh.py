"""Keep the account-day corpus (model_frame.parquet) at YESTERDAY, always.

THE PROBLEM THIS SOLVES

174 of the quant model's 249 features (`ad_*`) as-of join the account-day
corpus. That corpus was built once from a BigQuery snapshot whose records end
2026-08-27, so every live trade joined rows frozen at Aug 26 -- nine days stale
and getting worse daily, while the walk-forward always joined rows ~1 day old.
Live must see what the walk-forward saw: features as of the PREVIOUS trading
day, rolling forward automatically at each new day.

HOW

The pipeline is records -> per-account DAILY rows -> history/lag/lifetime
layers (trading_data.behaviour_features -- the exact functions the corpus was
built with). Records are the BQ snapshot plus production MySQL closed trades
from the snapshot's edge to now (MT4 `orders` carry a trade per row; MT5
`deals` pair entry/out by position_id; volumes /100 MT4 and /10000 MT5 per the
BQ-verified divisors, cent logins deflated /100). The daily aggregation is
sharded by account hash to bound memory; the enrichment layers run on the
small daily frame. Output is written atomically and verified to carry every
column model_features.csv names before it replaces the corpus.

`ensure_fresh` is the auto-update: called at startup and from a daily ticker,
it rebuilds only when the corpus's newest decision_day has fallen behind
yesterday, then drops the in-process caches so the next score joins the new
rows.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

_LOCK = threading.Lock()
_REFRESHING = False

RECORD_COLUMNS = ["database", "platform", "account_key", "timestamp", "symbol",
                  "cmd", "volume_lots", "open_time", "close_time", "open_price",
                  "close_price", "sl", "tp", "state", "net_profit", "profit"]


def _log(message: str) -> None:
    try:
        from webapp import vantage
    except ImportError:
        # Installation without the copy-trading engine (it lives in its own
        # private repository): the console is the log.
        print(f"ad-refresh: {message}", flush=True)
        return
    vantage._log(f"ad-refresh: {message}")


def _paths() -> tuple[Path, Path, Path]:
    from webapp import trade_features as tf
    return (tf._AD_DIR / "model_frame.parquet",
            tf._AD_DIR / "model_features.csv",
            tf._AD_DIR / "bq_90d_records.parquet")


def corpus_max_day() -> pd.Timestamp | None:
    frame_path, _, _ = _paths()
    try:
        import duckdb
        con = duckdb.connect()
        value = con.execute(
            "SELECT max(decision_day) FROM read_parquet(?)",
            [str(frame_path)]).fetchone()[0]
        con.close()
        return pd.Timestamp(value) if value is not None else None
    except Exception:
        return None


def _bulk_connection(server: str):
    """Own connection with timeouts sized for bulk history pulls."""
    import pymysql
    from webapp.mysql_extract import connect_settings
    return pymysql.connect(
        **connect_settings(server),
        connect_timeout=10, read_timeout=120, charset="utf8mb4")


def _pull_mysql_records(since: pd.Timestamp, log=print) -> pd.DataFrame:
    """Closed client trades since `since`, all six servers, records schema."""
    from webapp import trade_feed as tfeed
    out = []
    days = pd.date_range(since.floor("D"), pd.Timestamp.utcnow().tz_localize(None),
                         freq="D")
    for server in tfeed._servers():
        platform = "mt5" if server.startswith("mt5") else "mt4"
        try:
            # DEDICATED connection: the feed's shared one has read_timeout=8,
            # tuned for 300-row polls -- every bulk MT5 chunk timed out on it.
            connection = _bulk_connection(server)
            cents = tfeed.cent_logins(server)
        except Exception as error:
            log(f"{server}: unreachable ({type(error).__name__}) -- skipped")
            continue
        got = 0
        # MT5 full-day scans timed out; 4-hour chunks stay inside the read
        # timeout on every server.
        step_hours = 4 if platform == "mt5" else 24
        chunks = []
        for day_start in days:
            for offset in range(0, 24, step_hours):
                chunks.append((day_start + pd.Timedelta(hours=offset),
                               day_start + pd.Timedelta(hours=offset + step_hours)))
        for chunk_lo, chunk_hi in chunks:
            lo = int(max(chunk_lo, since).timestamp())
            hi = int(chunk_hi.timestamp())
            if lo >= hi:
                continue
            try:
                connection.ping(reconnect=True)   # a dead cursor kills the rest
                with connection.cursor() as cursor:
                    if platform == "mt4":
                        # MT4's swap column is `storage` (SHOW COLUMNS-verified;
                        # both `swap` and `swaps` 1054'd and yielded zero rows).
                        cursor.execute(
                            "SELECT login, symbol_name, cmd, volume, open_price,"
                            " open_ts, close_price, close_ts, profit, storage,"
                            " commission, sl, tp FROM orders WHERE close_ts >= %s"
                            " AND close_ts < %s AND cmd IN (0,1)"
                            " AND open_price > 0", [lo, hi])
                        for (login, sym, cmd, vol, op, ots, cp, cts, profit,
                             swap, comm, sl, tp) in cursor.fetchall():
                            scale = 100.0 if int(login) in cents else 1.0
                            net = (float(profit or 0) + float(swap or 0)
                                   + float(comm or 0)) / scale
                            out.append((server, platform,
                                        f"{server}:{login}",
                                        pd.Timestamp(int(cts), unit="s"),
                                        str(sym),
                                        "buy" if int(cmd) == 0 else "sell",
                                        float(vol) / 100.0 / scale,
                                        pd.Timestamp(int(ots), unit="s"),
                                        pd.Timestamp(int(cts), unit="s"),
                                        float(op), float(cp or 0) or np.nan,
                                        float(sl or 0) or np.nan,
                                        float(tp or 0) or np.nan,
                                        "closed", net, float(profit or 0) / scale))
                            got += 1
                    else:
                        # MT5 `time` is DATETIME -- epoch ints match NOTHING
                        # (two runs of silent zero rows). Bind datetimes.
                        cursor.execute(
                            "SELECT login, position_id, symbol, action, volume,"
                            " price, `time`, entry, profit, storage, commission,"
                            " price_sl, price_tp FROM deals WHERE `time` >= %s"
                            " AND `time` < %s AND action IN (0,1)",
                            [pd.Timestamp(lo, unit="s").to_pydatetime(),
                             pd.Timestamp(hi, unit="s").to_pydatetime()])
                        by_pos: dict = {}

                        def _ts(value):
                            """`time` may arrive as DATETIME or epoch int."""
                            if isinstance(value, (int, float)):
                                return pd.Timestamp(int(value), unit="s")
                            return pd.Timestamp(value)

                        for (login, pid, sym, action, vol, price, when, entry,
                             profit, storage, comm, sl, tp) in cursor.fetchall():
                            key = (int(login), int(pid or 0))
                            rec = by_pos.setdefault(key, {})
                            scale = 100.0 if int(login) in cents else 1.0
                            if int(entry) == 0:
                                rec.update(symbol=str(sym),
                                           cmd="buy" if int(action) == 0 else "sell",
                                           lots=float(vol) / 10000.0 / scale,
                                           open_time=_ts(when),
                                           open_price=float(price),
                                           sl=float(sl or 0) or np.nan,
                                           tp=float(tp or 0) or np.nan)
                            else:
                                rec["net"] = (rec.get("net", 0.0)
                                              + (float(profit or 0) + float(storage or 0)
                                                 + float(comm or 0)) / scale)
                                rec["profit"] = (rec.get("profit", 0.0)
                                                 + float(profit or 0) / scale)
                                rec["close_time"] = _ts(when)
                                rec["close_price"] = float(price)
                        for (login, _pid), rec in by_pos.items():
                            if "open_time" not in rec or "close_time" not in rec:
                                continue
                            out.append((server, platform, f"{server}:{login}",
                                        rec["close_time"], rec["symbol"],
                                        rec["cmd"], rec["lots"], rec["open_time"],
                                        rec["close_time"], rec["open_price"],
                                        rec.get("close_price", np.nan),
                                        rec.get("sl", np.nan), rec.get("tp", np.nan),
                                        "closed", rec.get("net", 0.0),
                                        rec.get("profit", 0.0)))
                            got += 1
            except Exception as error:
                log(f"{server} {chunk_lo}: {type(error).__name__}: {error}")
                try:                       # fresh connection for the next chunk
                    connection = _bulk_connection(server)
                except Exception:
                    break
        log(f"{server}: {got:,} closed trades since {since}")
    return pd.DataFrame(out, columns=RECORD_COLUMNS)


def refresh(shards: int = 8, log=print) -> dict:
    """Rebuild the corpus through yesterday. Returns a summary dict."""
    global _REFRESHING
    with _LOCK:
        if _REFRESHING:
            return {"ok": False, "reason": "already refreshing"}
        _REFRESHING = True
    try:
        return _refresh_inner(shards, log)
    finally:
        with _LOCK:
            _REFRESHING = False


def _refresh_inner(shards: int, log) -> dict:
    import duckdb
    from trading_data.behaviour_features import (
        add_history_features, add_lag_features, add_lifetime_features,
        daily_behaviour_features)

    frame_path, features_path, records_path = _paths()
    ad_columns = [line.strip()
                  for line in features_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    started = time.time()

    con = duckdb.connect()
    base_max = con.execute(
        "SELECT max(timestamp) FROM read_parquet(?)", [str(records_path)]
    ).fetchone()[0]
    base_max = pd.Timestamp(base_max)
    since = base_max - pd.Timedelta(days=1)          # overlap a day, dedupe below
    log(f"base records end {base_max}; pulling MySQL from {since}")

    fresh = _pull_mysql_records(since, log)
    log(f"MySQL append: {len(fresh):,} records")
    if not len(fresh):
        # Nothing came back -- the servers are unreachable (VPN down) or the
        # pull failed. Rebuilding from the base records alone would ROLL THE
        # CORPUS BACK to the base's last day (14 Sep 2026: 12 Sep -> 27 Aug,
        # twice, during a VPN drop). Keep what is on disk; the hourly ticker
        # retries once the servers answer.
        log("no MySQL records fetched -- keeping the existing corpus untouched")
        return {"ok": False, "reason": "no records fetched (servers unreachable?)",
                "minutes": round((time.time() - started) / 60.0, 1)}
    fresh_path = frame_path.parent / "mysql_append_records.parquet"
    if len(fresh):
        fresh.to_parquet(fresh_path, index=False)

    # Cent-account deflation for the BASE records: the BigQuery snapshot
    # carries cent-group values RAW (x100). The quant per-trade pipeline
    # deflates; this corpus did not -- which put a phantom -$68.5M into the
    # trading tab's client P&L (median trading/quant ratio 172x on cent
    # accounts). groups.currency='CNT' is the authority, via cent_logins.
    from webapp.trade_feed import cent_logins
    cent_keys: set[str] = set()
    for server in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04",
                   "mt5_live01"):
        try:
            cent_keys |= {f"{server}:{login}" for login in cent_logins(server)}
        except Exception:
            pass
    log(f"cent deflation: {len(cent_keys):,} cent accounts")

    def _deflate(frame: pd.DataFrame) -> pd.DataFrame:
        # FLOAT64 everywhere first: pymysql hands DECIMAL columns back as
        # Python Decimal, which poisons downstream numpy ("no callable sqrt").
        for column in ("volume_lots", "net_profit", "profit", "open_price",
                       "close_price", "sl", "tp"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce").astype("float64")
        mask = frame["account_key"].isin(cent_keys)
        if mask.any():
            for column in ("volume_lots", "net_profit", "profit"):
                if column in frame.columns:
                    frame.loc[mask, column] = frame.loc[mask, column] / 100.0
        return frame

    # Sharded daily aggregation: 33M+ records never sit in memory at once.
    daily_parts = []
    columns_sql = ", ".join(RECORD_COLUMNS)
    for shard in range(shards):
        base = con.execute(
            f"SELECT {columns_sql} FROM read_parquet(?) "
            f"WHERE hash(account_key) % {shards} = {shard}",
            [str(records_path)]).df()
        base = _deflate(base)
        if len(fresh):
            extra = con.execute(
                f"SELECT {columns_sql} FROM read_parquet(?) "
                f"WHERE hash(account_key) % {shards} = {shard}",
                [str(fresh_path)]).df()
            # extra was cent-deflated at pull time -- coerce dtypes ONLY
            for column in ("volume_lots", "net_profit", "profit", "open_price",
                           "close_price", "sl", "tp"):
                if column in extra.columns:
                    extra[column] = pd.to_numeric(
                        extra[column], errors="coerce").astype("float64")
            base = pd.concat([base, extra], ignore_index=True)
        # the overlap day appears in both sources: one row per trade
        base = base.drop_duplicates(
            ["account_key", "symbol", "open_time", "close_time", "volume_lots"],
            keep="first")
        if not len(base):
            continue
        daily_parts.append(daily_behaviour_features(base))
        log(f"shard {shard + 1}/{shards}: {len(base):,} records -> "
            f"{len(daily_parts[-1]):,} account-days")
        del base
    con.close()

    daily = pd.concat(daily_parts, ignore_index=True)
    del daily_parts
    daily = daily.sort_values(["account_key", "day"]).reset_index(drop=True)
    log(f"daily frame: {len(daily):,} account-days; layering history/lag/lifetime")
    enriched = add_history_features(daily)
    enriched = add_lag_features(enriched)
    enriched = add_lifetime_features(enriched)
    enriched["decision_day"] = enriched["day"]

    missing = [c for c in ad_columns if c not in enriched.columns]
    if missing:
        return {"ok": False,
                "reason": f"rebuilt frame missing columns: {missing[:8]}"}

    keep = ["account_key", "decision_day"] + ad_columns
    out = enriched[keep]
    tmp = frame_path.with_suffix(".parquet.new")
    out.to_parquet(tmp, index=False)
    backup = frame_path.with_suffix(".parquet.bak")
    if not backup.exists():
        frame_path.rename(backup)
    else:
        frame_path.unlink()
    tmp.rename(frame_path)

    _reset_caches()
    summary = {"ok": True, "rows": len(out),
               "max_decision_day": str(out["decision_day"].max()),
               "minutes": round((time.time() - started) / 60.0, 1)}
    log(f"corpus refreshed: {summary}")
    return summary


def _reset_caches() -> None:
    """New corpus on disk -> the in-process snapshots must reload."""
    try:
        from webapp import trade_features as tf
        tf._AD_CORPUS_CACHE = None
        tf._AD_CORPUS_MTIME = None
    except Exception:
        pass
    try:
        from webapp import vantage
        vantage._AD_SNAPSHOT = None
        vantage._PARITY_STATE.clear()
        vantage._ACCOUNT_TAPE.clear()
    except Exception:
        pass


def ensure_fresh(log=None) -> dict:
    """Refresh only when the corpus has fallen behind YESTERDAY."""
    log = log or _log
    newest = corpus_max_day()
    yesterday = (pd.Timestamp.utcnow().tz_localize(None).floor("D")
                 - pd.Timedelta(days=1))
    if newest is not None and newest >= yesterday:
        return {"ok": True, "fresh": True, "max_decision_day": str(newest)}
    log(f"corpus ends {newest}, yesterday is {yesterday.date()} -- refreshing")
    return refresh(log=log)


def start_daily_ticker() -> None:
    """Background thread: check at startup and then hourly; rebuild whenever a
    new trading day has begun. Hourly, not daily, so a laptop waking from
    sleep still catches up promptly."""
    def _tick():
        while True:
            try:
                ensure_fresh()
            except Exception as error:
                _log(f"ticker error: {type(error).__name__}: {error}")
            time.sleep(3600)
    threading.Thread(target=_tick, daemon=True, name="ad-refresh").start()
