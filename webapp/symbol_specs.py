"""Contract specifications and USD conversion, from each server's own tables.

Exposure has to be in USD to be summable across instruments, and getting there
needs three things the server already knows: the contract size, which currency
the contract is denominated in, and a rate to USD.

WHY THE NAIVE FORMULA IS WRONG

`lots x contract_size x price` is right for EURUSD and XAUUSD and wrong for
USDJPY. The distinction is which leg is USD:

* **profit currency is USD** (EURUSD, XAUUSD, US500) -- the contract is in the
  BASE asset, so USD value is `contract_size x price`.
* **base currency is USD** (USDJPY, USDCHF) -- the contract is already 100,000
  USD. Multiplying by 150 overstates it by that factor.
* **neither leg is USD** (GBPJPY, EURGBP) -- the contract is in GBP or EUR and
  needs a genuine cross rate. Treating it as USD overstated GBPJPY exposure as
  $1.375B in an earlier version of this screen.

The specs come from `symbols` on each server rather than a hardcoded table, so
a venue that lists gold at 10 oz per lot is handled correctly instead of being
assumed to be 100.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# Conventional contract sizes come from `trading_data.research`, which already
# owns this: `asset_class()` classifies the instrument, CONTRACT_SIZE_FALLBACK
# gives the per-class default, and SYMBOL_CONTRACT_SIZE_FALLBACK carries the
# exceptions that matter (silver is 5,000oz, not the 100 the metal class
# implies). Re-deriving any of that here would give two tables that drift.
from trading_data.research import asset_class, contract_size_fallback  # noqa: E402

_SPEC_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_SPEC_TTL = 3600.0


def load_specs(database: str) -> pd.DataFrame:
    """Contract metadata for one server, normalised across MT4 and MT5.

    MT4 exposes a single `currency` (the profit currency) plus live `bid`/`ask`;
    MT5 exposes `currency_base`, `currency_profit` and `currency_margin`. Both
    are mapped onto the same three columns so callers need not branch.
    """
    cached = _SPEC_CACHE.get(database)
    if cached and time.time() - cached[0] < _SPEC_TTL:
        return cached[1]

    from webapp.mysql_extract import _connection

    is_mt5 = database.startswith("mt5")
    if is_mt5:
        sql = ("SELECT symbol, contract_size, tick_value, tick_size, digits,"
               " currency_base, currency_profit, currency_margin FROM symbols")
    else:
        # MT4's `currency` is the MARGIN currency, not the profit currency --
        # EURUSD reports EUR. Mapping it to `currency_profit` sent every
        # USD-quoted major down the cross-rate path instead of recognising its
        # direct USD leg.
        sql = ("SELECT symbol, contract_size, tick_value, tick_size, digits,"
               " currency AS currency_margin, bid, ask FROM symbols")

    connection = _connection(database, timeout=120)
    try:
        frame = pd.read_sql(sql, connection)
    finally:
        connection.close()

    frame["database"] = database
    for column in ("currency_base", "currency_profit", "currency_margin", "bid", "ask"):
        if column not in frame.columns:
            frame[column] = pd.NA
    for column in ("contract_size", "tick_value", "tick_size", "bid", "ask"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    from trading_data.research import canonical_symbol
    frame["canonical_symbol"] = frame["symbol"].map(canonical_symbol)

    # For a six-letter alphabetic ticker the legs are unambiguous: EURUSD is
    # EUR against USD. This is how the base and profit currencies are recovered
    # on MT4, which publishes only the margin currency.
    canonical = frame["canonical_symbol"].astype(str).str.upper()
    is_fx = canonical.str.fullmatch(r"[A-Z]{6}").fillna(False)
    frame["currency_base"] = frame["currency_base"].fillna(
        canonical.str[:3].where(is_fx))
    frame["currency_profit"] = frame["currency_profit"].fillna(
        canonical.str[3:].where(is_fx))
    # Anything else quoted by a venue is overwhelmingly USD-settled (metals,
    # indices, crypto, energy); the margin currency is the best evidence there.
    frame["currency_profit"] = frame["currency_profit"].fillna(frame["currency_margin"])
    for column in ("currency_base", "currency_profit", "currency_margin"):
        frame[column] = frame[column].astype("string").str.upper()

    _SPEC_CACHE[database] = (time.time(), frame)
    return frame


def build_fx_rates(specs: pd.DataFrame) -> dict[str, float]:
    """Map each currency to its USD value, from the venue's own live quotes.

    Built from whatever USD pairs the server lists: XXXUSD gives the rate
    directly, USDXXX gives its reciprocal. A currency with no USD pair listed is
    absent from the map, and callers must then report the exposure as
    unconvertible rather than guessing at 1.0 -- a silent 1.0 would value a
    JPY-denominated contract at 150x its true USD size.
    """
    rates: dict[str, float] = {"USD": 1.0}
    priced = specs.loc[specs["bid"].notna() & (specs["bid"] > 0)]
    for row in priced.itertuples():
        symbol = str(row.canonical_symbol or "").upper()
        if len(symbol) != 6 or not symbol.isalpha():
            continue
        base, quote = symbol[:3], symbol[3:]
        mid = float(row.bid if not row.ask or np.isnan(row.ask)
                    else (row.bid + row.ask) / 2)
        if mid <= 0:
            continue
        if quote == "USD" and base not in rates:
            rates[base] = mid
        elif base == "USD" and quote not in rates:
            rates[quote] = 1.0 / mid
    return rates


def usd_notional(positions: pd.DataFrame, specs: pd.DataFrame,
                 rates: dict[str, float] | None = None) -> pd.DataFrame:
    """USD notional per position, with the conversion route recorded.

    `notional_status` says how each value was obtained -- a venue contract size
    with a direct USD leg, a fallback contract size, a cross-rate conversion, or
    no conversion at all. A risk report that cannot distinguish those is
    reporting confidence it does not have.
    """
    rates = rates if rates is not None else build_fx_rates(specs)
    working = positions.copy()

    spec_columns = ["contract_size", "currency_base", "currency_profit",
                    "tick_value", "tick_size"]
    candidates = specs.dropna(subset=["symbol"]).copy()
    candidates["_symbol_key"] = candidates["symbol"].astype(str).str.upper()

    # CONTRACT SIZE IS A PROPERTY OF THE RAW TICKER, NOT THE CANONICAL SYMBOL.
    # The variants that canonicalise together are genuinely different contracts:
    # XAUUSD is 100oz, XAUUSDmin is 10oz and XAUUSD247 is 1oz; EURUSD is 100,000
    # units and EURUSDmin is 10,000. Looking the size up by canonical symbol
    # would value a XAUUSD247 position at 100x its real size. Canonical is the
    # right key for GROUPING the resulting risk -- it is the same underlying --
    # but not for measuring it.
    working["_symbol_key"] = working["symbol"].astype(str).str.upper()

    if "database" in working.columns and "database" in candidates.columns:
        per_server = candidates.drop_duplicates(["database", "_symbol_key"], keep="first")
        working = working.merge(
            per_server[["database", "_symbol_key"] + spec_columns],
            on=["database", "_symbol_key"], how="left", suffixes=("", "_spec"))
    else:
        exact = candidates.drop_duplicates("_symbol_key", keep="first")
        working = working.merge(exact[["_symbol_key"] + spec_columns],
                                on="_symbol_key", how="left", suffixes=("", "_spec"))

    # A ticker this server does not list falls back to the same ticker on any
    # other server before falling back to an asset-class convention.
    gaps = working["contract_size"].isna()
    if gaps.any():
        any_server = candidates.drop_duplicates("_symbol_key", keep="first")
        filled = working.loc[gaps, ["_symbol_key"]].merge(
            any_server[["_symbol_key"] + spec_columns], on="_symbol_key", how="left")
        for column in spec_columns:
            working.loc[gaps, column] = filled[column].to_numpy()
    working = working.drop(columns=["_symbol_key"])

    working["contract_size_source"] = np.where(
        working["contract_size"].notna(), "server", "asset_class_fallback")
    working["contract_size"] = working["contract_size"].where(
        working["contract_size"].notna(),
        working["canonical_symbol"].map(contract_size_fallback))

    lots = pd.to_numeric(working["volume_lots"], errors="coerce").abs().fillna(0.0)
    # Coalesce the price PER ROW. Choosing one column for the whole frame broke
    # as soon as MT4 and MT5 positions were concatenated: MT4 has no
    # `price_current`, so every MT4 row silently priced at NaN and gold came out
    # at $0.
    price = pd.Series(np.nan, index=working.index, dtype="float64")
    for candidate in ("price_current", "price_open", "open_price"):
        if candidate in working.columns:
            price = price.fillna(pd.to_numeric(working[candidate], errors="coerce"))

    profit = working["currency_profit"].astype("string").str.upper().fillna("")

    # Where the venue did not tell us the profit currency -- which is every row
    # when the spec database is unreachable -- infer it from the ticker. The
    # profit currency is the QUOTE side: the last three characters of a six
    # character FX pair, and USD for metals, indices and crypto quoted in
    # dollars. Without this the FX conversion returns NaN and every exposure
    # figure silently becomes blank, which reads as "no risk" rather than "no
    # data" -- the failure mode this whole module exists to avoid.
    missing = profit.isin(["", "<NA>"]) | profit.isna()
    if missing.any():
        canon = working["canonical_symbol"].astype("string").str.upper().fillna("")
        inferred = np.where(canon.str.len() == 6, canon.str[-3:], "USD")
        profit = profit.mask(missing, pd.Series(inferred, index=working.index))
        working["profit_currency_source"] = np.where(missing, "inferred", "venue")

    # ONE UNIVERSAL FORMULA:
    #
    #     USD notional = lots x contract_size x price x rate(profit -> USD)
    #
    # A contract is `contract_size` units of the base asset; multiplying by the
    # price gives its value in the PROFIT currency; converting that to USD
    # finishes the job. This is correct for every instrument type, which the
    # earlier case-by-case version was not:
    #
    #   EURUSD  100,000 x 1.158  x 1.0     = $115,800
    #   USDJPY  100,000 x 150    x 0.00667 = $100,000   (the JPY legs cancel)
    #   XAUUSD  100     x 4,000  x 1.0     = $400,000
    #   GBPJPY  100,000 x 200    x 0.00667 = $133,400   (= 100,000 GBP)
    #
    # The previous "if base is USD, it is already USD" branch skipped the price
    # for XAUUSD -- MT5 reports its base currency as USD -- and valued 768 lots
    # of gold at $76,801 instead of roughly $307m.
    profit_rate = profit.map(lambda c: rates.get(c, np.nan)).to_numpy(dtype="float64")
    contract_notional = lots * working["contract_size"]
    working["notional_usd"] = contract_notional * price * profit_rate

    priced_ok = np.isfinite(price.to_numpy()) & (price.to_numpy() > 0)
    working["notional_status"] = np.select(
        [
            priced_ok & profit.eq("USD").to_numpy(),
            priced_ok & np.isfinite(profit_rate),
        ],
        ["direct_usd_leg", "cross_rate"],
        default="unconvertible",
    )
    working["notional_status"] = np.where(
        (working["contract_size_source"] == "asset_class_fallback")
        & (working["notional_status"] != "unconvertible"),
        "fallback_contract_" + working["notional_status"].astype(str),
        working["notional_status"])

    direction = (pd.to_numeric(working["direction"], errors="coerce").fillna(1.0)
                 if "direction" in working.columns else pd.Series(1.0, index=working.index))
    working["gross_notional_usd"] = working["notional_usd"].abs()
    working["net_notional_usd"] = working["gross_notional_usd"] * np.sign(direction)
    working["signed_lots"] = lots * np.sign(direction)
    return working


#: Contract specifications cached beside the warehouse.
#:
#: These were read from MySQL on every call, so every exposure figure on the
#: site silently became unavailable whenever the VPN dropped -- and because the
#: loader swallowed the connection error and returned an empty frame, the
#: failure showed up as "0 spec rows" rather than "no database". Notional then
#: computed as zero, which is a wrong answer wearing the costume of a right one.
#:
#: Specs change when the broker adds an instrument, not by the minute, so a
#: local copy is both safe and enough.
SPEC_CACHE = Path(__file__).resolve().parent / "spec_cache"


def refresh_spec_cache(databases: tuple[str, ...], progress=None) -> dict:
    """Pull contract specifications from MySQL into the local cache."""
    SPEC_CACHE.mkdir(exist_ok=True)
    written, failures = 0, []
    for database in databases:
        try:
            specs = load_specs(database)
            if specs.empty:
                continue
            specs.to_parquet(SPEC_CACHE / f"{database}.parquet", index=False)
            written += len(specs)
            if progress:
                progress(f"{database}: {len(specs):,} symbol specs cached")
        except Exception as error:
            failures.append(f"{database}: {type(error).__name__}")
            if progress:
                progress(f"{database}: spec refresh FAILED ({type(error).__name__})")
    return {"rows": written, "failures": failures}


def infer_rates_from_prices(trades: pd.DataFrame,
                            rates: dict[str, float] | None = None) -> dict[str, float]:
    """Quote-currency rates read out of the trades themselves.

    Without the spec database there are no FX rates, so every JPY-, CHF- or
    CAD-quoted instrument prices as NaN. But the rates are sitting in the data:
    a USDJPY trade at 150 says one dollar buys 150 yen, so JPY converts to USD
    at 1/150.

    Uses the MEDIAN price per pair rather than the latest, because a single
    fat-fingered or stale print would otherwise reprice the whole book. This is
    a fallback for when the venue's own numbers are unreachable -- it is
    accurate to a few percent, not to the tick, which is the right precision
    for a concentration limit.
    """
    rates = dict(rates or {"USD": 1.0})
    if trades.empty or "symbol" not in trades.columns:
        return rates
    canon = (trades["symbol"].astype(str).str.upper()
             .str.replace(r"[^A-Z0-9]", "", regex=True))
    price = pd.Series(np.nan, index=trades.index, dtype="float64")
    for candidate in ("open_price", "price_open", "price"):
        if candidate in trades.columns:
            price = price.fillna(pd.to_numeric(trades[candidate], errors="coerce"))

    usd_base = canon.str.match(r"^USD[A-Z]{3}$") & price.notna() & (price > 0)
    if not usd_base.any():
        return rates
    quotes = canon.loc[usd_base].str[3:6]
    for currency, group in price.loc[usd_base].groupby(quotes.to_numpy()):
        median = float(group.median())
        if np.isfinite(median) and median > 0 and currency not in rates:
            rates[currency] = 1.0 / median
    return rates


def load_all_specs(databases: tuple[str, ...],
                   allow_remote: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    """Specs across every server, plus one shared FX rate map.

    Prefers the local cache and falls back to MySQL, rather than the reverse:
    the cache is complete, free and always reachable, while the database is
    none of those from outside the VPN.
    """
    frames = []
    for database in databases:
        cached = SPEC_CACHE / f"{database}.parquet"
        if cached.exists():
            try:
                frames.append(pd.read_parquet(cached))
                continue
            except Exception:
                pass
        if allow_remote:
            try:
                specs = load_specs(database)
                if not specs.empty:
                    SPEC_CACHE.mkdir(exist_ok=True)
                    specs.to_parquet(cached, index=False)
                    frames.append(specs)
            except Exception:
                continue
    if not frames:
        # Shaped, not bare. `usd_notional` falls back to asset-class contract
        # sizes when a spec row is missing, which is a usable answer -- but a
        # zero-column DataFrame makes it raise on `dropna(subset=["symbol"])`
        # before it can get there. An empty frame with the right columns lets
        # the fallback run, so exposure degrades in accuracy rather than
        # collapsing to zero the moment the VPN drops.
        return pd.DataFrame(columns=["database", "symbol", "contract_size",
                                     "currency_base", "currency_profit",
                                     "tick_value", "tick_size"]), {"USD": 1.0}
    specs = pd.concat(frames, ignore_index=True)
    return specs, build_fx_rates(specs)
