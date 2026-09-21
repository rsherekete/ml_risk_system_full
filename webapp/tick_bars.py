"""Minute bars from the MySQL tick table, for path-aware backtesting.

A stop-loss study needs to know whether price TOUCHED a level between entry and
exit. Entry and exit prices alone cannot answer that, which is why the earlier
exit-policy test had to truncate realised losses and produced an impossible
zero-drawdown result.

The tick table has 5.4 billion rows over a full year, indexed on `tm`. Pulling
raw ticks is impractical -- XAUUSD alone is ~312,000 ticks a day -- so bars are
aggregated SERVER-SIDE: one `GROUP BY` minute turns a day into 1,440 rows and
moves a few kilobytes instead of tens of megabytes.

Minute resolution is a deliberate compromise. It can miss a stop touched and
reversed inside a single minute, which makes any stop result slightly
OPTIMISTIC -- stated here because that is the direction that flatters the
policy, and it is the same bias that invalidated the previous attempt. It is
nonetheless far better than no path at all: a minute bar's high and low bound
the excursion, where entry/exit prices bound nothing.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd

BAR_SQL = """
SELECT symbol_name AS symbol,
       DATE_FORMAT(tm, '%%Y-%%m-%%d %%H:%%i:00') AS minute,
       MIN((bid + ask) / 2) AS low,
       MAX((bid + ask) / 2) AS high,
       SUBSTRING_INDEX(GROUP_CONCAT(CAST((bid + ask) / 2 AS CHAR)
                       ORDER BY tm ASC  SEPARATOR ','), ',', 1) AS open,
       SUBSTRING_INDEX(GROUP_CONCAT(CAST((bid + ask) / 2 AS CHAR)
                       ORDER BY tm DESC SEPARATOR ','), ',', 1) AS close,
       COUNT(*) AS ticks
FROM ticks
WHERE tm >= %s AND tm < %s
  AND symbol_name IN ({placeholders})
  AND bid > 0 AND ask > 0
GROUP BY symbol_name, minute
"""


def _fetch_chunk(database: str, sql: str, symbols: tuple[str, ...],
                 window: tuple[datetime, datetime]) -> pd.DataFrame:
    from webapp.mysql_extract import _connection

    for attempt in range(3):
        connection = None
        try:
            connection = _connection(database, timeout=900)
            return pd.read_sql(sql, connection, params=(window[0], window[1], *symbols))
        except Exception:
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(5 * (attempt + 1))
        finally:
            if connection is not None:
                connection.close()
    return pd.DataFrame()


def fetch_bars(database: str, symbols: tuple[str, ...],
               start: datetime, end: datetime, chunk_hours: int = 12,
               workers: int = 8, progress=None) -> pd.DataFrame:
    """Minute OHLC for the given raw tickers over a window.

    Chunked AND parallel. The cost is the index range scan on `tm`, which is
    proportional to the window length and largely independent per chunk, so
    running several windows concurrently is close to a linear speed-up. Serially
    a 90-day pull would take roughly 45 minutes; with eight workers it is a few.

    Symbols match on the RAW ticker, not the canonical one -- XAUUSD and
    XAUUSDmin are distinct instruments. (They happen to share a price feed, so
    one series usually suffices; that is a fact about this venue, not a rule.)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    placeholders = ", ".join(["%s"] * len(symbols))
    sql = BAR_SQL.format(placeholders=placeholders)

    windows, cursor_time = [], start
    while cursor_time < end:
        window_end = min(cursor_time + timedelta(hours=chunk_hours), end)
        windows.append((cursor_time, window_end))
        cursor_time = window_end

    frames, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_chunk, database, sql, symbols, window): window
                   for window in windows}
        for future in as_completed(futures):
            chunk = future.result()
            done += 1
            if not chunk.empty:
                frames.append(chunk)
            if progress and done % 10 == 0:
                progress(f"{done}/{len(windows)} chunks, {sum(len(f) for f in frames):,} bars")

    if not frames:
        return pd.DataFrame(columns=["symbol", "minute", "open", "high", "low", "close", "ticks"])
    bars = pd.concat(frames, ignore_index=True)
    bars["minute"] = pd.to_datetime(bars["minute"])
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.drop_duplicates(["symbol", "minute"]).sort_values(
        ["symbol", "minute"]).reset_index(drop=True)


def excursions(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Maximum adverse and favourable excursion for each trade.

    MAE is the worst the trade ever looked before it closed; MFE is the best.
    Together they say what any stop or target would have done, which entry and
    exit prices alone cannot.

    Computed with a merge_asof-style sweep per symbol rather than a per-trade
    query: at hundreds of thousands of trades, one query each is not viable.
    """
    if trades.empty or bars.empty:
        return trades.assign(mae=pd.NA, mfe=pd.NA, bars_seen=0)

    results = []
    indexed = {symbol: group.reset_index(drop=True)
               for symbol, group in bars.groupby("symbol", observed=True)}

    for symbol, group in trades.groupby("symbol", observed=True):
        series = indexed.get(symbol)
        if series is None or series.empty:
            results.append(group.assign(mae=pd.NA, mfe=pd.NA, bars_seen=0))
            continue

        minutes = series["minute"].to_numpy()
        highs = series["high"].to_numpy()
        lows = series["low"].to_numpy()
        # Running extrema let each trade's window be answered in O(log n) by
        # bracketing its start and end rather than scanning the bars.
        starts = minutes.searchsorted(group["open_time"].to_numpy(), side="left")
        ends = minutes.searchsorted(group["close_time"].to_numpy(), side="right")

        mae, mfe, seen = [], [], []
        direction = group["direction"].to_numpy()
        entry = group["open_price"].to_numpy()
        for i, (lo, hi) in enumerate(zip(starts, ends)):
            if hi <= lo:
                mae.append(None); mfe.append(None); seen.append(0)
                continue
            window_high = highs[lo:hi].max()
            window_low = lows[lo:hi].min()
            if direction[i] > 0:      # long: adverse is down, favourable is up
                mae.append(window_low - entry[i])
                mfe.append(window_high - entry[i])
            else:                     # short: adverse is up
                mae.append(entry[i] - window_high)
                mfe.append(entry[i] - window_low)
            seen.append(int(hi - lo))
        results.append(group.assign(mae=mae, mfe=mfe, bars_seen=seen))

    return pd.concat(results, ignore_index=True)
