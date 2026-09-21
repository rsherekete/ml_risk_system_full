"""MT5 copy-trading engine for the Quant book.

This is the only component that moves real money, so its defaults are chosen to
fail safe rather than to be convenient:

* **Paper mode is the default.** Orders are recorded and priced but never sent.
  Going live requires explicitly arming the engine, and arming is a separate
  action from configuring it.
* **Every order passes limit checks first** -- per-symbol exposure, aggregate
  exposure, per-order lot size, and a rate limit. The signal model can emit
  thousands of trades in a burst; without a rate limit a single bad refit could
  fire an unbounded stream of orders at the broker.
* **A kill switch stops execution immediately** and is honoured before every
  order rather than at the top of the loop.
* **Nothing is inferred about fills.** Positions and balances come from MT5
  itself, so the "live" side of the comparison is what the terminal reports,
  never what this module believes it sent.

The dashboard contrasts EXPECTED risk (what the model asked for) against LIVE
risk (what MT5 actually holds), because those diverge in practice -- rejected
orders, partial fills, slippage and margin limits all break the equivalence.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "app.db"


@dataclass
class CopyConfig:
    """Execution settings. Conservative by construction."""

    enabled: bool = False              # armed for LIVE execution
    paper_mode: bool = True            # record only, never send
    login: int = 0
    password: str = ""
    server: str = ""
    #: Multiplier applied to the client's lot size. Below 1 by default so a
    #: mistake is proportionally smaller than the flow it mirrors.
    size_multiplier: float = 0.10
    max_lots_per_order: float = 1.0
    max_open_lots_per_symbol: float = 10.0
    max_total_open_lots: float = 50.0
    #: Burst protection. The model can emit thousands of signals in seconds.
    max_orders_per_minute: int = 30
    min_confidence: float = 0.70
    magic: int = 770001
    deviation_points: int = 20
    allowed_symbols: str = ""          # comma-separated; empty means all


@dataclass
class EngineState:
    status: str = "stopped"            # stopped | running | killed | error
    message: str = ""
    connected: bool = False
    orders_sent: int = 0
    orders_rejected: int = 0
    orders_blocked: int = 0
    last_order_at: float = 0.0
    kill_switch: bool = False
    log: list[str] = field(default_factory=list)


_STATE = EngineState()
_LOCK = threading.Lock()
_RECENT_ORDERS: list[float] = []


def _log(message: str) -> None:
    with _LOCK:
        _STATE.log.append(f"{datetime.now(timezone.utc):%H:%M:%S}  {message}")
        _STATE.log = _STATE.log[-200:]
        _STATE.message = message


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS copy_config (id INTEGER PRIMARY KEY CHECK (id = 1),
            payload TEXT NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS copy_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL, source_account TEXT, symbol TEXT,
            direction INTEGER, lots REAL, confidence REAL, mode TEXT,
            status TEXT, ticket INTEGER, price REAL, comment TEXT);
    """)
    return connection


def load_config() -> CopyConfig:
    with _db() as connection:
        row = connection.execute("SELECT payload FROM copy_config WHERE id = 1").fetchone()
    if row is None:
        return CopyConfig()
    stored = json.loads(row["payload"])
    valid = CopyConfig().__dict__.keys()
    return CopyConfig(**{k: v for k, v in stored.items() if k in valid})


def save_config(config: CopyConfig) -> None:
    with _db() as connection:
        connection.execute(
            "INSERT INTO copy_config (id, payload, updated_at) VALUES (1, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET payload = excluded.payload,"
            " updated_at = excluded.updated_at",
            (json.dumps(asdict(config)), time.time()))


def state() -> EngineState:
    return _STATE


def set_kill_switch(engaged: bool) -> None:
    """Hard stop. Checked before EVERY order, not once per cycle."""
    with _LOCK:
        _STATE.kill_switch = engaged
        _STATE.status = "killed" if engaged else "stopped"
    _log("KILL SWITCH ENGAGED -- execution halted" if engaged else "kill switch released")


# ---------------------------------------------------------------------------
# MT5 terminal
# ---------------------------------------------------------------------------
def _mt5():
    import MetaTrader5 as mt5
    return mt5


def connect(config: CopyConfig) -> tuple[bool, str]:
    """Attach to a running MT5 terminal and log in."""
    try:
        mt5 = _mt5()
    except ImportError:
        return False, "MetaTrader5 package is not installed."
    if not mt5.initialize():
        return False, f"MT5 initialize failed: {mt5.last_error()}"
    if config.login:
        ok = mt5.login(int(config.login), password=config.password, server=config.server)
        if not ok:
            return False, f"MT5 login failed: {mt5.last_error()}"
    _STATE.connected = True
    return True, "connected"


def account_snapshot() -> dict:
    """Balance, equity and margin AS MT5 REPORTS THEM.

    Deliberately read from the terminal rather than derived from our own order
    log: the whole point of the live-versus-expected panel is to surface the
    difference, which is invisible if both sides come from the same place.
    """
    try:
        mt5 = _mt5()
        info = mt5.account_info()
        if info is None:
            return {"available": False, "error": str(mt5.last_error())}
        return {
            "available": True,
            "login": info.login, "currency": info.currency,
            "balance": info.balance, "equity": info.equity,
            "margin": info.margin, "margin_free": info.margin_free,
            "margin_level": info.margin_level, "profit": info.profit,
            "leverage": info.leverage, "server": info.server,
        }
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def open_positions() -> list[dict]:
    try:
        mt5 = _mt5()
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [{
            "ticket": p.ticket, "symbol": p.symbol,
            "direction": 1 if p.type == mt5.POSITION_TYPE_BUY else -1,
            "lots": p.volume, "open_price": p.price_open, "current_price": p.price_current,
            "profit": p.profit, "swap": p.swap, "magic": p.magic,
            "opened_at": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
        } for p in positions]
    except Exception:
        return []


def closed_deals(days: int = 7) -> list[dict]:
    try:
        mt5 = _mt5()
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(now.timestamp() - days * 86400, now.timestamp())
        if deals is None:
            return []
        return [{
            "ticket": d.ticket, "symbol": d.symbol, "volume": d.volume,
            "price": d.price, "profit": d.profit, "commission": d.commission,
            "swap": d.swap, "magic": d.magic,
            "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
        } for d in deals]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# risk limits
# ---------------------------------------------------------------------------
def check_limits(config: CopyConfig, symbol: str, lots: float,
                 confidence: float) -> tuple[bool, str]:
    """Every reason an order may be refused, evaluated before it is built."""
    if _STATE.kill_switch:
        return False, "kill switch engaged"
    if confidence < config.min_confidence:
        return False, f"confidence {confidence:.2f} below minimum {config.min_confidence:.2f}"
    if lots > config.max_lots_per_order:
        return False, f"{lots:.2f} lots exceeds per-order cap {config.max_lots_per_order:.2f}"
    if config.allowed_symbols:
        allowed = {s.strip().upper() for s in config.allowed_symbols.split(",") if s.strip()}
        if symbol.upper() not in allowed:
            return False, f"{symbol} not in the allowed list"

    # Rate limit: the model can emit thousands of signals in a burst, and an
    # unthrottled stream of orders is the failure mode most likely to cause real
    # damage before anyone notices.
    now = time.time()
    global _RECENT_ORDERS
    _RECENT_ORDERS = [t for t in _RECENT_ORDERS if now - t < 60]
    if len(_RECENT_ORDERS) >= config.max_orders_per_minute:
        return False, f"rate limit: {config.max_orders_per_minute} orders/minute reached"

    positions = open_positions()
    symbol_lots = sum(p["lots"] for p in positions if p["symbol"] == symbol)
    total_lots = sum(p["lots"] for p in positions)
    if symbol_lots + lots > config.max_open_lots_per_symbol:
        return False, (f"{symbol} exposure {symbol_lots:.2f}+{lots:.2f} exceeds "
                       f"{config.max_open_lots_per_symbol:.2f}")
    if total_lots + lots > config.max_total_open_lots:
        return False, (f"total exposure {total_lots:.2f}+{lots:.2f} exceeds "
                       f"{config.max_total_open_lots:.2f}")
    return True, "ok"


def submit(config: CopyConfig, source_account: str, symbol: str, direction: int,
           client_lots: float, confidence: float) -> dict:
    """Mirror one client trade, subject to every limit.

    In paper mode the order is recorded with `mode='paper'` and no request is
    sent, so the same code path is exercised and the log is directly comparable
    to a live run.
    """
    lots = round(max(0.01, client_lots * config.size_multiplier), 2)
    allowed, reason = check_limits(config, symbol, lots, confidence)
    if not allowed:
        with _LOCK:
            _STATE.orders_blocked += 1
        _record(source_account, symbol, direction, lots, confidence,
                "paper" if config.paper_mode else "live", "blocked", None, None, reason)
        _log(f"BLOCKED {symbol} {lots} lots -- {reason}")
        return {"sent": False, "reason": reason}

    if config.paper_mode or not config.enabled:
        _record(source_account, symbol, direction, lots, confidence, "paper", "recorded",
                None, None, "paper mode -- not sent")
        _log(f"paper {('BUY' if direction > 0 else 'SELL')} {symbol} {lots} lots")
        return {"sent": False, "reason": "paper mode", "lots": lots}

    try:
        mt5 = _mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"no tick for {symbol}")
        price = tick.ask if direction > 0 else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
            "price": price, "deviation": config.deviation_points,
            "magic": config.magic, "comment": f"zfx-copy {source_account}"[:31],
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        with _LOCK:
            if ok:
                _STATE.orders_sent += 1
                _STATE.last_order_at = time.time()
                _RECENT_ORDERS.append(time.time())
            else:
                _STATE.orders_rejected += 1
        _record(source_account, symbol, direction, lots, confidence, "live",
                "filled" if ok else "rejected",
                getattr(result, "order", None), getattr(result, "price", None),
                getattr(result, "comment", "") if result else "no result")
        _log(f"{'FILLED' if ok else 'REJECTED'} {symbol} {lots} lots @ {price}")
        return {"sent": ok, "ticket": getattr(result, "order", None)}
    except Exception as error:
        with _LOCK:
            _STATE.orders_rejected += 1
        _record(source_account, symbol, direction, lots, confidence, "live", "error",
                None, None, f"{type(error).__name__}: {error}")
        _log(f"ERROR {symbol}: {error}")
        return {"sent": False, "reason": str(error)}


def _record(source_account, symbol, direction, lots, confidence,
            mode, status, ticket, price, comment) -> None:
    with _db() as connection:
        connection.execute(
            "INSERT INTO copy_orders (created_at, source_account, symbol, direction, lots,"
            " confidence, mode, status, ticket, price, comment)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), source_account, symbol, direction, lots, confidence,
             mode, status, ticket, price, comment))


def order_history(limit: int = 200) -> list[dict]:
    with _db() as connection:
        rows = connection.execute(
            "SELECT * FROM copy_orders ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def expected_vs_live(config: CopyConfig) -> dict:
    """The comparison the dashboard exists for.

    EXPECTED is what the engine intended, from its own order log. LIVE is what
    MT5 actually holds. They diverge through rejections, partial fills, slippage
    and margin limits, and that divergence is the number worth watching.
    """
    orders = order_history(2000)
    intended: dict[str, float] = {}
    for order in orders:
        if order["status"] in ("filled", "recorded"):
            intended[order["symbol"]] = intended.get(order["symbol"], 0.0) + \
                order["lots"] * (1 if order["direction"] > 0 else -1)

    live: dict[str, float] = {}
    for position in open_positions():
        live[position["symbol"]] = live.get(position["symbol"], 0.0) + \
            position["lots"] * position["direction"]

    symbols = sorted(set(intended) | set(live))
    rows = []
    for symbol in symbols:
        want, have = intended.get(symbol, 0.0), live.get(symbol, 0.0)
        rows.append({"symbol": symbol, "expected_lots": round(want, 2),
                     "live_lots": round(have, 2), "drift": round(have - want, 2)})
    return {
        "rows": rows,
        "total_expected": round(sum(abs(v) for v in intended.values()), 2),
        "total_live": round(sum(abs(v) for v in live.values()), 2),
        "orders_sent": _STATE.orders_sent,
        "orders_rejected": _STATE.orders_rejected,
        "orders_blocked": _STATE.orders_blocked,
    }
