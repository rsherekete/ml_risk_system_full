"""The drawdown layer: net exposure by instrument, and what capping it is worth.

WHY THIS EXISTS SEPARATELY FROM THE ROUTING MODEL

The firm's worst drawdown -- $31.3M over sixteen days in January 2026 -- was
84.8% XAUUSD, spread across 12,879 accounts. The per-account model could not
see it and structurally never will: each of those clients looked individually
unremarkable, and what was dangerous was that they agreed with each other.

Meanwhile the net gold position was building in plain sight, from -556 lots on
2 January to 93,759 by the 19th, in the firm's own book.

So there are two different problems wearing the same clothes:

  THE EDGE is that clients lose on average -- roughly $250 per account-day,
  $1.096bn over two years. It is captured by NOT hedging. Every hedge spends
  some of it, which is why the account model can only ever be worth about 1%.

  THE RISK is that clients occasionally agree. That is a portfolio property,
  invisible per account, and it is what actually produces drawdown.

This module addresses the second and leaves the first alone. The results are
cached because the computation reads the full trade history.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import data_store, exposure_policy, model_service

#: Limits swept on screen, in USD of net exposure per instrument.
LIMITS = (5e6, 10e6, 25e6, 50e6, 100e6, 250e6, 500e6)

CACHE = model_service.ARTIFACTS / "book_risk.json"
BOOK = model_service.ARTIFACTS / "daily_book.parquet"

_LOCK = threading.Lock()
_thread: threading.Thread | None = None


def running() -> bool:
    return _thread is not None and _thread.is_alive()


def start_build(history_days: int = 730) -> bool:
    global _thread
    with _LOCK:
        if running():
            return False
        _thread = threading.Thread(target=_build, args=(history_days,),
                                   daemon=True, name="book-risk")
        _thread.start()
        return True


def _build(history_days: int) -> None:
    from . import symbol_specs

    started = time.time()
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=history_days)
        servers = tuple(p.name for p in data_store.WAREHOUSE.iterdir() if p.is_dir())
        specs, rates = symbol_specs.load_all_specs(servers)

        edges = sorted(set(
            [pd.Timestamp(start).tz_localize(None)]
            + pd.date_range(pd.Timestamp(start).tz_localize(None).normalize(),
                            pd.Timestamp(end).tz_localize(None).normalize(),
                            freq="QS").tolist()
            + [pd.Timestamp(end).tz_localize(None)]))

        books = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            chunk = data_store.read_history(
                start=lower.to_pydatetime().replace(tzinfo=timezone.utc),
                end=upper.to_pydatetime().replace(tzinfo=timezone.utc),
                columns=["database", "symbol", "cmd", "volume_lots", "open_time",
                         "close_time", "open_price", "net_profit"])
            if chunk.empty:
                continue
            chunk["net_profit"] = pd.to_numeric(chunk["net_profit"], errors="coerce")
            chunk = chunk.loc[chunk["net_profit"].notna() & chunk["open_time"].notna()]
            if chunk.empty:
                continue
            # `usd_notional` needs the canonical symbol for its asset-class
            # contract-size fallback, and prices from `open_price` directly.
            # Exposure is valued at the OPEN price on purpose: it is what the
            # position is worth while it sits on the book, not what it
            # eventually settled at.
            chunk["canonical_symbol"] = exposure_policy.canonical_symbol(chunk["symbol"])
            # When the spec database is unreachable there are no FX rates, and
            # every JPY/CHF/CAD-quoted instrument would price as NaN -- silently
            # dropping out of the exposure figures. The rates are recoverable
            # from the trades: a USDJPY print at 150 means JPY converts at 1/150.
            chunk_rates = symbol_specs.infer_rates_from_prices(chunk, rates)
            chunk = symbol_specs.usd_notional(chunk, specs, chunk_rates)
            books.append(exposure_policy.daily_book(chunk))
            del chunk

        if not books:
            return
        book = pd.concat(books, ignore_index=True).groupby(
            ["day", "canonical"], observed=True).agg(
            net_usd=("net_usd", "sum"), gross_usd=("gross_usd", "sum"),
            client_pnl=("client_pnl", "sum")).reset_index()
        book.to_parquet(BOOK, index=False)

        table = exposure_policy.sweep(book, LIMITS)
        # The conditional policy: hedge only when exposure is abnormal for that
        # instrument. The static cap fails economically -- it binds on most
        # days and passes the firm's core edge (being short a permanently
        # long-gold crowd) to market along with the risk.
        anomaly = exposure_policy.sweep_anomaly(book)
        instruments = book.groupby("canonical").agg(
            gross_usd=("gross_usd", "sum"),
            mean_net=("net_usd", "mean"),
            max_abs_net=("net_usd", lambda s: float(s.abs().max())),
            client_pnl=("client_pnl", "sum"),
        ).sort_values("gross_usd", ascending=False).head(15).reset_index()

        payload = {
            "flat_profit": float(table.attrs["flat_profit"]),
            "flat_drawdown": float(table.attrs["flat_drawdown"]),
            "rows": table.to_dict("records"),
            "anomaly": anomaly.to_dict("records"),
            "instruments": instruments.to_dict("records"),
            "days": int(book["day"].nunique()),
            "history_days": history_days,
            "seconds": time.time() - started,
            "computed_at": time.time(),
        }
        CACHE.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    except Exception as error:
        CACHE.write_text(json.dumps({
            "error": f"{type(error).__name__}: {error}",
            "computed_at": time.time()}, indent=1), encoding="utf-8")


def report() -> dict:
    """The cached sweep, plus whichever limit is the best trade-off."""
    payload = None
    if CACHE.exists():
        try:
            payload = json.loads(CACHE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            payload = None
    if payload is None:
        return {"available": False, "running": running()}
    if payload.get("error"):
        return {"available": False, "running": running(), "error": payload["error"]}

    rows = payload.get("rows", [])
    dominating = [r for r in rows if r.get("dominates")]
    # Prefer a limit that beats flat on BOTH axes; otherwise the one that buys
    # the most drawdown reduction per dollar of profit given up.
    if dominating:
        best = max(dominating, key=lambda r: r["drawdown_delta"])
    elif rows:
        def efficiency(row):
            given_up = max(1.0, -row["profit_delta"])
            return row["drawdown_delta"] / given_up
        best = max(rows, key=efficiency)
    else:
        best = None

    payload.update({"available": True, "running": running(), "best": best,
                    "dominating": len(dominating)})
    return payload


def current_exposure(limit_usd: float | None = None) -> pd.DataFrame:
    """Latest net exposure per instrument, for the live risk view."""
    if not BOOK.exists():
        return pd.DataFrame()
    book = pd.read_parquet(BOOK)
    if book.empty:
        return book
    latest = book["day"].max()
    today = book.loc[book["day"] == latest].copy()
    today["abs_net_usd"] = today["net_usd"].abs()
    if limit_usd:
        today["over_limit_usd"] = (today["abs_net_usd"] - limit_usd).clip(lower=0)
        today["breach"] = today["over_limit_usd"] > 0
    return today.sort_values("abs_net_usd", ascending=False)
