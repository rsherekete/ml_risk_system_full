"""Net notional exposure -- what the firm is actually carrying, by instrument.

P&L answers what happened; exposure answers what could happen next, and they
are not interchangeable. A book that is flat on the day may be carrying a large
one-sided position into tomorrow.

TWO NUMBERS, DELIBERATELY BOTH SHOWN

* **Gross notional** -- total size regardless of side. What the firm is
  intermediating, and what drives commission and operational load.
* **Net notional** -- direction-weighted. What the firm is actually exposed to.
  A book long 100 lots and short 100 lots has 200 gross and 0 net, and those are
  completely different risks despite identical volume.

CONTRACT SIZES

Notional needs a contract size, which differs by instrument class. Getting this
wrong scales an entire instrument's risk by 100x, so the defaults are explicit
and a symbol nothing recognises is marked rather than silently assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Units per lot by instrument class. FX majors are 100k; metals are 100 oz;
#: indices and crypto are typically 1 unit per lot.
CONTRACT_SIZES: dict[str, float] = {
    "XAUUSD": 100.0, "XAGUSD": 5000.0, "XPTUSD": 100.0, "XPDUSD": 100.0,
    "USOIL": 1000.0, "UKOIL": 1000.0, "NGAS": 10000.0,
    "BTCUSD": 1.0, "ETHUSD": 1.0, "LTCUSD": 1.0, "XRPUSD": 1.0,
    "ADAUSD": 1.0, "DOGEUSD": 1.0, "LINKUSD": 1.0, "SOLUSD": 1.0,
    "USTEC": 1.0, "US30": 1.0, "US500": 1.0, "GER40": 1.0, "UK100": 1.0,
    "JP225": 1.0, "CHINAA50": 1.0, "HK50": 1.0, "AUS200": 1.0,
}
DEFAULT_FX_CONTRACT = 100_000.0


def contract_size(canonical: str) -> tuple[float, bool]:
    """Units per lot, and whether it was recognised.

    The flag matters: an unrecognised symbol falls back to the FX convention,
    which is right for a currency pair and wrong by orders of magnitude for
    anything else. Screens surface the flag rather than presenting a guess as
    a measurement.
    """
    key = str(canonical or "").upper()
    if key in CONTRACT_SIZES:
        return CONTRACT_SIZES[key], True
    # A six-letter all-alpha ticker is almost certainly an FX pair.
    if len(key) == 6 and key.isalpha():
        return DEFAULT_FX_CONTRACT, True
    return DEFAULT_FX_CONTRACT, False


def position_notional(frame: pd.DataFrame, price_column: str = "price_current",
                      symbol_column: str = "canonical_symbol") -> pd.DataFrame:
    """Attach gross and net notional to a frame of positions."""
    working = frame.copy()
    symbols = working[symbol_column].astype(str)
    sizes = symbols.map(lambda s: contract_size(s)[0])
    working["contract_size"] = sizes
    working["contract_known"] = symbols.map(lambda s: contract_size(s)[1])

    # Mark at the current price where the platform provides one. MT5 exposes
    # `price_current`; MT4's open-order rows carry only the entry price, and
    # `.get()` returns None there rather than an empty Series -- so the column
    # has to be checked for presence, not for emptiness.
    price = None
    for candidate in (price_column, "price_current", "open_price"):
        if candidate in working.columns:
            values = pd.to_numeric(working[candidate], errors="coerce")
            if values.notna().any():
                price = values
                break
    if price is None:
        price = pd.Series(np.nan, index=working.index)

    lots = pd.to_numeric(working["volume_lots"], errors="coerce").abs()
    direction = (working["direction"] if "direction" in working.columns
                 else pd.Series(1.0, index=working.index))

    working["gross_notional"] = lots * sizes * price
    working["net_notional"] = working["gross_notional"] * direction
    # Signed lots. Summing `volume_lots` gives GROSS however the positions are
    # oriented, which made net and gross identical in every row and hid the one
    # distinction the panel exists to draw.
    working["signed_lots"] = lots * direction
    return working


def exposure_by_symbol(positions: pd.DataFrame, specs: pd.DataFrame | None = None,
                       rates: dict[str, float] | None = None) -> pd.DataFrame:
    """Current exposure per instrument in USD, largest net first.

    Uses the venue's own contract specs when they are supplied. The hardcoded
    fallback below is a last resort: it cannot know that a given server lists
    gold at 10 oz per lot, and it cannot convert a cross pair at all.
    """
    if positions.empty:
        return pd.DataFrame()
    if specs is not None and not specs.empty:
        from webapp.symbol_specs import usd_notional
        priced = usd_notional(positions, specs, rates)
        return _aggregate_usd(priced)
    priced = position_notional(positions)
    grouped = priced.groupby("canonical_symbol", observed=True).agg(
        positions=("volume_lots", "size"),
        accounts=("account_key", "nunique"),
        gross_lots=("volume_lots", lambda s: float(s.abs().sum())),
        net_lots=("signed_lots", "sum"),
        gross_notional=("gross_notional", "sum"),
        net_notional=("net_notional", "sum"),
        floating_pnl=("profit", "sum"),
        contract_known=("contract_known", "all"),
    ).reset_index()
    grouped["net_share"] = (grouped["net_notional"].abs()
                            / grouped["gross_notional"].replace(0, np.nan))
    grouped = grouped.sort_values("net_notional", key=abs, ascending=False)
    # JSON has no NaN. Instruments the platform gives no price for produce NaN
    # notional, and returning them unconverted fails the whole response rather
    # than the one row -- so they become null and the screen shows them as
    # unknown instead of the panel going blank.
    return grouped.replace({np.nan: None}).reset_index(drop=True)


def _aggregate_usd(priced: pd.DataFrame) -> pd.DataFrame:
    """Roll per-position USD notional up to one row per instrument."""
    grouped = priced.groupby("canonical_symbol", observed=True).agg(
        positions=("volume_lots", "size"),
        accounts=("account_key", "nunique"),
        gross_lots=("volume_lots", lambda s: float(s.abs().sum())),
        net_lots=("signed_lots", "sum"),
        gross_notional=("gross_notional_usd", "sum"),
        net_notional=("net_notional_usd", "sum"),
        floating_pnl=("profit", "sum"),
        contract_size=("contract_size", "first"),
        currency_base=("currency_base", "first"),
        currency_profit=("currency_profit", "first"),
    ).reset_index()

    # Worst status wins per instrument: if any position could not be converted,
    # the instrument's total is incomplete and must say so rather than quietly
    # reporting the subset that happened to convert.
    severity = {"direct_usd_leg": 0, "fallback_contract_direct_usd_leg": 1,
                "cross_rate": 2, "fallback_contract_cross_rate": 3, "unconvertible": 4}
    status = (priced.assign(_rank=priced["notional_status"].map(severity).fillna(4))
                    .groupby("canonical_symbol", observed=True)["_rank"].max())
    reverse = {v: k for k, v in severity.items()}
    grouped["notional_status"] = grouped["canonical_symbol"].map(status).map(reverse)
    grouped["contract_known"] = ~grouped["notional_status"].astype(str).str.startswith("fallback")

    grouped["net_share"] = (grouped["net_notional"].abs()
                            / grouped["gross_notional"].replace(0, np.nan))
    grouped = grouped.sort_values("net_notional", key=lambda s: s.abs(), ascending=False)
    # JSON has no NaN; an unconvertible instrument becomes null and is labelled
    # rather than dropped -- the positions with the worst data quality are
    # exactly the ones risk must still see.
    return grouped.replace({np.nan: None}).reset_index(drop=True)


def historical_exposure(trades: pd.DataFrame, symbols: tuple[str, ...] | None = None,
                        freq: str = "D") -> pd.DataFrame:
    """Net and gross notional over time, from position open/close events.

    Built as a running sum of signed opens and closes rather than by expanding
    every position across every day it was held: at tens of millions of trades
    the expansion is prohibitive, and the cumulative-event form gives the same
    curve.
    """
    if trades.empty:
        return pd.DataFrame()

    working = trades.copy()
    if "canonical_symbol" not in working.columns:
        from webapp.views import add_canonical_symbol
        working = add_canonical_symbol(working)
    if symbols:
        working = working.loc[working["canonical_symbol"].isin(symbols)]
    if working.empty:
        return pd.DataFrame()

    sizes = working["canonical_symbol"].astype(str).map(lambda s: contract_size(s)[0])
    direction = np.where(working["cmd"].astype(str) == "buy", 1.0, -1.0)
    lots = pd.to_numeric(working["volume_lots"], errors="coerce").fillna(0).abs()
    notional = lots * sizes * pd.to_numeric(working["open_price"], errors="coerce").fillna(0)

    # A position adds exposure when it opens and removes it when it closes.
    opens = pd.DataFrame({
        "when": pd.to_datetime(working["open_time"]),
        "symbol": working["canonical_symbol"].to_numpy(),
        "net": notional.to_numpy() * direction,
        "gross": notional.to_numpy(),
    })
    closes = pd.DataFrame({
        "when": pd.to_datetime(working["close_time"]),
        "symbol": working["canonical_symbol"].to_numpy(),
        "net": -notional.to_numpy() * direction,
        "gross": -notional.to_numpy(),
    })
    events = pd.concat([opens, closes], ignore_index=True).dropna(subset=["when"])
    events["bucket"] = events["when"].dt.floor(freq)

    changes = events.groupby(["symbol", "bucket"], observed=True)[["net", "gross"]].sum().reset_index()
    changes = changes.sort_values(["symbol", "bucket"])
    changes["net_notional"] = changes.groupby("symbol", observed=True)["net"].cumsum()
    changes["gross_notional"] = changes.groupby("symbol", observed=True)["gross"].cumsum()
    return changes[["symbol", "bucket", "net_notional", "gross_notional"]]


def exposure_as_at(trades: pd.DataFrame, as_at: pd.Timestamp,
                   specs: pd.DataFrame | None = None,
                   rates: dict[str, float] | None = None) -> pd.DataFrame:
    """End-of-day exposure on a historical date, reconstructed from trades.

    A position contributes to a date's exposure when it opened on or before that
    date and had not yet closed. That is the same definition the live view uses
    -- what the firm was carrying overnight -- but rebuilt from the warehouse so
    any past day can be inspected, not only right now.

    Note the asymmetry with the live view: this can only see positions the
    warehouse holds, so a position opened before the stored history is invisible
    here while the live view (which reads the broker's own position table) would
    show it.
    """
    if trades.empty:
        return pd.DataFrame()

    working = trades.copy()
    opened = pd.to_datetime(working["open_time"], errors="coerce")
    closed = pd.to_datetime(working["close_time"], errors="coerce")
    cutoff = pd.Timestamp(as_at).normalize() + pd.Timedelta(days=1)

    still_open = (opened < cutoff) & (closed.isna() | (closed >= cutoff))
    working = working.loc[still_open].copy()
    if working.empty:
        return pd.DataFrame()

    if "canonical_symbol" not in working.columns:
        from webapp.views import add_canonical_symbol
        working = add_canonical_symbol(working)
    working["direction"] = np.where(
        working["cmd"].astype(str).str.lower() == "buy", 1.0, -1.0)
    working["price_current"] = pd.to_numeric(working["open_price"], errors="coerce")
    if "profit" not in working.columns:
        working["profit"] = 0.0
    if "account_key" not in working.columns:
        working["account_key"] = (working["database"].astype(str) + ":"
                                  + working["login"].astype("int64").astype(str))

    if specs is not None and not specs.empty:
        from webapp.symbol_specs import usd_notional
        return _aggregate_usd(usd_notional(working, specs, rates))
    return exposure_by_symbol(working)


def concentration_alerts(exposure: pd.DataFrame, net_limit_usd: float = 5_000_000,
                         share_limit: float = 0.35) -> list[dict]:
    """Breaches worth a dealer's attention, ranked by severity.

    Two independent tests, because they catch different failures: an absolute
    net limit catches a book that is simply too large one way, and a
    concentration share catches a book that is diversified in name only.
    """
    if exposure.empty:
        return []
    alerts = []
    total_gross = float(exposure["gross_notional"].sum()) or 1.0
    for row in exposure.itertuples():
        net = float(row.net_notional)
        if abs(net) >= net_limit_usd:
            alerts.append({
                "severity": "critical" if abs(net) >= net_limit_usd * 2 else "warning",
                "symbol": row.canonical_symbol,
                "kind": "net exposure",
                "message": (f"Net {'long' if net > 0 else 'short'} "
                            f"${abs(net):,.0f} exceeds the ${net_limit_usd:,.0f} limit"),
                "value": net,
            })
        share = float(row.gross_notional) / total_gross
        if share >= share_limit:
            alerts.append({
                "severity": "warning",
                "symbol": row.canonical_symbol,
                "kind": "concentration",
                "message": f"{share:.0%} of all gross exposure sits in this instrument",
                "value": share,
            })
        if not row.contract_known:
            alerts.append({
                "severity": "info",
                "symbol": row.canonical_symbol,
                "kind": "contract size",
                "message": ("Contract size unrecognised -- notional assumes the FX "
                            "convention and may be wrong by orders of magnitude"),
                "value": 0.0,
            })
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(alerts, key=lambda a: (order[a["severity"]], -abs(a["value"])))


def stress_test(exposure: pd.DataFrame,
                shocks: tuple[float, ...] = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05)) -> list[dict]:
    """What an instantaneous move in each instrument would cost the firm.

    The firm is on the OTHER side of client positions, so a move that profits
    clients costs the firm. Net notional times the shock, sign-flipped.

    Deliberately a simple linear shock: it makes no correlation assumption, and
    an honest first-order number is more useful to a desk than a sophisticated
    one resting on a covariance matrix estimated from three months of data.
    """
    if exposure.empty:
        return []
    results = []
    for shock in shocks:
        per_symbol = {}
        total = 0.0
        for row in exposure.itertuples():
            impact = -float(row.net_notional) * shock
            per_symbol[row.canonical_symbol] = impact
            total += impact
        worst = min(per_symbol.items(), key=lambda kv: kv[1]) if per_symbol else ("--", 0.0)
        results.append({
            "shock": shock,
            "firm_pnl": total,
            "worst_symbol": worst[0],
            "worst_impact": worst[1],
            "by_symbol": per_symbol,
        })
    return results
