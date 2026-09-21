"""Per-trade features for the Quant (trade-level) router.

All four groups are knowable at the moment a trade opens, which is when the
copy/hedge decision has to be made:

  * trade-intrinsic -- size, symbol, direction, timing, stop and target placement
  * market context  -- trailing returns and volatility for that symbol
  * account history -- how this client's PREVIOUS trades behaved
  * markout profile -- whether this client's past fills were followed by
    favourable moves (the adverse-selection signature)

THE MARKOUT RULE, stated once because it is the easiest thing here to get
wrong. A trade's OWN post-entry markout is the outcome restated in different
units; including it yields a near-perfect model that cannot be traded. Only the
account's markout profile over EARLIER, ALREADY-CLOSED trades is admissible.

That "already closed" qualifier is doing real work: history is accumulated in
CLOSE-time order and joined as-of each trade's OPEN, so a feature never carries
the result of a position that was still running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: The account-day (watchlist) model's cached feature frame -- the SAME file
#: that model trains on, so both models read identical values by
#: construction. The path mirrors model_service.SCRATCH (not imported: that
#: module imports this one). 37 curated columns of its 174: behavioural
#: signatures, expanding all-history stats, and the lifetime record --
#: everything the per-trade builder does not already derive itself.
#: Durable home for the account-day corpus + its manifest + the records
#: snapshot. Previously a session TEMP scratchpad -- one cleanup away from
#: losing 174 features. webapp/ad_refresh.py keeps the corpus at yesterday.
_AD_DIR = Path(__file__).resolve().parent / "artifacts" / "ad"
#: ALL of the corpus's features, read from its own manifest so the two
#: models can never drift apart. The curated list is only the fallback for
#: an environment without the cache. Row sampling (config.max_trade_rows)
#: pays the memory bill for the full column set -- rows past a few million
#: buy compute, not accuracy; columns are where the information is.
try:
    # Plain file I/O, deliberately not pandas: this runs at import time and
    # a pandas call here detonates the package's latent circular import.
    AD_COLUMNS: list[str] = [
        line.strip()
        for line in (_AD_DIR / "model_features.csv")
        .read_text(encoding="utf-8").splitlines() if line.strip()]
except Exception:
    AD_COLUMNS = [
        "buy_share", "stop_use_rate", "take_profit_use_rate",
        "martingale_rate", "revenge_rate", "scalp_rate", "overnight_rate",
        "concentration", "directional_bias", "lots_dispersion",
        "events_per_symbol", "expanding_win_rate", "expanding_profit_factor",
        "expanding_expectancy_per_trade", "expanding_martingale_rate",
        "expanding_pnl_rate", "client_drawdown", "losing_streak",
        "cumulative_pnl", "payoff_ratio", "profit_factor",
        "expectancy_per_trade", "pnl_rate", "pnl_momentum", "size_momentum",
        "roll5_sharpe", "roll20_sharpe", "roll5_win_rate", "roll20_win_rate",
        "lag_pnl_consistency", "form_vs_life", "life_win_rate",
        "life_profit_factor", "life_sharpe", "life_max_drawdown",
        "life_expectancy", "account_age_days",
    ]
AD_FEATURES: list[str] = ["ad_" + name for name in AD_COLUMNS]

TRADE_FEATURES: list[str] = [
    "hour", "weekday", "log_notional", "log_lots", "has_sl", "has_tp",
    "sl_distance", "tp_distance", "risk_reward", "is_expert", "symbol_code", "direction",
    "ctx_return_1h", "ctx_return_4h", "ctx_return_24h", "ctx_vol_24h", "ctx_flow_prev",
    "trade_index", "hist_win_rate", "hist_mean_pnl", "hist_pnl_std", "hist_duration",
    "hist_edge", "hist_mean_notional", "notional_vs_usual",
    # Funding behaviour, point-in-time as of the trade's open day. A client who
    # deposits, wins and withdraws at once trades nothing like one who funds an
    # account and grinds it down -- and a trade placed right after a fresh
    # deposit is a different proposition from one placed on a nearly-withdrawn
    # account. All cumulative-to-date, carried forward from settled movements
    # only (see `cashflow_store` for the leak test).
    "deposits_to_date", "withdrawals_to_date", "net_funding_to_date",
    "deposit_count_to_date", "withdrawal_count_to_date",
    "days_since_deposit", "days_since_withdrawal",
    "withdrawal_ratio", "funding_churn",
    # NORMALISED market technicals -- every one expressed in units that are
    # comparable ACROSS instruments (volatility units, range fractions, sign
    # agreement), so one tree split serves gold and cable alike. All derive
    # from bar statistics shifted one hour: strictly pre-open, causal.
    "mom_1h_vol", "mom_4h_vol", "mom_24h_vol",   # backward markouts / vol
    "with_momentum",                             # direction x 4h momentum
    "momentum_align",                            # direction x trend agreement
    "zscore_24h", "range_pos_24h", "vol_regime",
    "sl_dist_vol", "tp_dist_vol",                # bracket placement / vol
    # Trade-vs-own-behaviour and account-state. equity_proxy is the client's
    # in-window equity (net funding + realised P&L to date); the margin/impact
    # family expresses THIS trade against it.
    "trades_today", "hours_since_last",
    "equity_proxy", "notional_to_equity", "underwater_frac",
    "hist_recent_pnl5",
    # ACCOUNT-DAY ACTIVITY -- the watchlist model's angle folded into the
    # per-trade router. Daily realised aggregates per account (by CLOSE day,
    # as-of joined strictly before the trade's open day, so nothing still
    # running contributes), over 5/20/60 active-day windows, plus the
    # cross-sectional rank against every other account active that day.
    "acct_pnl_5d", "acct_pnl_20d", "acct_pnl_60d",
    "acct_vol_20d", "acct_winrate_20d",
    "acct_trades_5d", "acct_trades_20d",
    "acct_dd_20d", "acct_best_20d", "acct_worst_20d",
    "acct_rank_20d", "acct_tenure_days", "acct_days_since_active",
    # RECENCY LADDER: win rate and mean P&L over the last 20/10/5/2 percent
    # of the client's closed trades (min 3), beside the full-history
    # aggregates. A lifetime loser on a 60-of-61 heater looks IDENTICAL to a
    # lifetime loser in a slump without these -- the ladder is how the model
    # sees current form and how fast it is diverging from the base rate.
    "hist_wr_20p", "hist_wr_10p", "hist_wr_5p", "hist_wr_2p",
    "hist_mp_20p", "hist_mp_10p", "hist_mp_5p", "hist_mp_2p",
    # REGIME-CONDITIONAL history: this client's record split by the market
    # state at their past entries. The synthetic study measured a 24%
    # recovery shortfall on context-dependent skill without these -- the
    # model could see the regime and the client, but not the interaction.
    "hist_wr_trend", "hist_wr_chop", "hist_mp_trend", "hist_mp_chop",
] + AD_FEATURES

#: Columns that encode the outcome. Asserted absent from the feature set.
OUTCOME_COLUMNS = {"net_profit", "close_price", "close_time", "duration_s", "past_edge"}


#: Crypto CFDs: 1 lot = 1 coin, so the per-unit multiplier is 1, not the
#: 100k a six-letter name would otherwise suggest. BTCUSD priced as forex
#: made a $500 move look like $50M and poisoned every downstream estimate.
CRYPTO_ROOTS = ("BTC", "ETH", "LTC", "XRP", "ADA", "SOL", "DOGE", "AVAX",
                "DOT", "BNB", "LINK", "UNI")


def symbol_class(symbol: str) -> str:
    """gold / silver / fx / crypto / index-other -- ONE classifier shared by
    training, studies and the live gate."""
    root = "".join(c for c in str(symbol).upper() if c.isalnum())
    if root.startswith(CRYPTO_ROOTS):
        return "crypto"
    if root.startswith("XAU"):
        return "gold"
    if root.startswith("XAG"):
        return "silver"
    if len(root) == 6 and root.isalpha():
        return "fx"
    return "index/other"


#: Round-turn commission per lot. Gold, FX and metals are charged (~$6/lot
#: here); index and crypto CFDs are spread-only. ONE source of truth so the
#: magnitude target, the replay and the live hurdle agree.
COMMISSION_PER_LOT = 6.0


def trade_cost(symbol: str, price: float, lots: float,
               include_commission: bool = True) -> float:
    """All-in cost estimate for one trade -- spread toll PLUS round-turn
    commission. ONE implementation shared by the magnitude model's training
    target, the replay, and live expected-value decisions.
    """
    root = "".join(c for c in str(symbol).upper() if c.isalnum())
    price = max(float(price or 0.0), 0.0)
    lots = max(float(lots or 0.0), 0.0)
    if root.startswith(CRYPTO_ROOTS):
        spread_cost = 0.0005 * price * 1.0 * lots
        commission = 0.0
    elif root.startswith("XAU"):
        spread_cost = 0.30 * 100.0 * lots
        commission = COMMISSION_PER_LOT * lots
    elif root.startswith("XAG"):
        spread_cost = 0.03 * 100.0 * lots
        commission = COMMISSION_PER_LOT * lots
    elif len(root) == 6 and root.isalpha():
        spread = 0.00015 * max(price, 1.0) if "JPY" in root else 0.00012
        spread_cost = spread * 100_000.0 * lots
        commission = COMMISSION_PER_LOT * lots
    else:                                   # index / other CFD: spread-only
        spread_cost = 0.0006 * max(price, 1.0) * 100.0 * lots
        commission = 0.0
    return spread_cost + (commission if include_commission else 0.0)


def trade_cost_vector(symbol: pd.Series, price: pd.Series,
                      lots: pd.Series) -> np.ndarray:
    """Vectorised trade_cost for training frames."""
    root = symbol.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    price = pd.to_numeric(price, errors="coerce").fillna(0.0).clip(lower=0.0)
    lots = pd.to_numeric(lots, errors="coerce").fillna(0.0).clip(lower=0.0)
    is_crypto = root.str.startswith(CRYPTO_ROOTS)
    is_xau = root.str.startswith("XAU") & ~is_crypto
    is_xag = root.str.startswith("XAG") & ~is_crypto
    is_fx = (root.str.len().eq(6) & root.str.isalpha()
             & ~is_xau & ~is_xag & ~is_crypto)
    is_jpy = is_fx & root.str.contains("JPY")
    spread_x_unit = np.select(
        [is_crypto, is_xau, is_xag, is_jpy, is_fx],
        [0.0005 * price, 0.30 * 100.0, 0.03 * 100.0,
         0.00015 * price.clip(lower=1.0) * 100_000.0, 0.00012 * 100_000.0],
        default=0.0006 * price.clip(lower=1.0) * 100.0)
    return (spread_x_unit * lots.to_numpy()).astype("float64")


def build_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Attach every feature in `TRADE_FEATURES` to a frame of closed trades."""
    leaked = sorted(set(TRADE_FEATURES) & OUTCOME_COLUMNS)
    if leaked:
        raise RuntimeError(f"outcome columns present in feature set: {leaked}")

    frame = trades.copy()
    # CANONICAL SYMBOL PARITY: the live engine computes market-context bars,
    # symbol_class and symbol_code under the CANONICAL symbol (XAUUSD pools
    # XAUUSDe / XAUUSDmin / XAUUSD247 ...). Training historically grouped by
    # the RAW warehouse spelling, so a variant's ctx/code differed between
    # train and live -- a silent feature-parity break, worst on the
    # high-variant, high-vol instrument (gold). Canonicalise here so the
    # training feature row is built on the SAME instrument identity the live
    # scorer uses. raw_symbol is retained for any downstream display.
    try:
        from webapp.trade_feed import _canonical
        frame["raw_symbol"] = frame["symbol"].astype(str)
        frame["symbol"] = frame["raw_symbol"].map(
            lambda s: _canonical(s) or s)
    except Exception:
        pass
    for column in ("volume_lots", "open_price", "sl", "tp", "net_profit"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["direction"] = np.where(frame["cmd"].astype("string") == "buy", 1.0, -1.0)
    frame["day"] = frame["open_time"].dt.floor("D")
    frame["notional"] = frame["volume_lots"] * 100.0 * frame["open_price"]
    frame = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True)

    frame["hour"] = frame["open_time"].dt.hour
    frame["weekday"] = frame["open_time"].dt.weekday
    frame["log_notional"] = np.log1p(frame["notional"])
    frame["log_lots"] = np.log1p(frame["volume_lots"])
    frame["has_sl"] = (frame["sl"].fillna(0) != 0).astype("float32")
    frame["has_tp"] = (frame["tp"].fillna(0) != 0).astype("float32")
    # Stop and target distance as a fraction of entry, signed so both are
    # positive when placed sensibly for the trade's direction. sl_distance was
    # the single most-used feature in the measured model.
    frame["sl_distance"] = np.where(
        frame["has_sl"] > 0,
        (frame["open_price"] - frame["sl"]) * frame["direction"] / frame["open_price"], np.nan)
    frame["tp_distance"] = np.where(
        frame["has_tp"] > 0,
        (frame["tp"] - frame["open_price"]) * frame["direction"] / frame["open_price"], np.nan)
    frame["risk_reward"] = frame["tp_distance"] / frame["sl_distance"].replace(0, np.nan)
    frame["is_expert"] = (frame["reason"].astype("string") == "expert").astype("float32")
    frame["symbol_code"] = frame["symbol"].astype("category").cat.codes

    # Market context from the trade tape itself: a per-symbol hourly price path,
    # with every statistic shifted one bar so the current hour never informs its
    # own features.
    frame["symbol_hour"] = frame["open_time"].dt.floor("h")
    bars = frame.groupby(["symbol", "symbol_hour"], observed=True).agg(
        price=("open_price", "median"), flow=("direction", "mean"),
    ).reset_index().sort_values(["symbol", "symbol_hour"])
    by_symbol = bars.groupby("symbol", observed=True)["price"]
    for window in (1, 4, 24):
        bars[f"ctx_return_{window}h"] = by_symbol.transform(
            lambda s, w=window: s.pct_change(w)).shift(1)
    bars["ctx_vol_24h"] = by_symbol.transform(
        lambda s: s.pct_change().rolling(24, min_periods=4).std()).shift(1)
    bars["ctx_flow_prev"] = bars.groupby("symbol", observed=True)["flow"].shift(1)
    # Normalised technicals, every statistic through the PREVIOUS completed
    # hour (shift(1)): z-score of price against its 24h band, position within
    # the 24h high-low range, and the volatility regime (today's vol against
    # its own trailing week) -- all unitless, hence cross-symbol.
    mean24 = by_symbol.transform(lambda s: s.rolling(24, min_periods=6).mean()).shift(1)
    std24 = by_symbol.transform(lambda s: s.rolling(24, min_periods=6).std()).shift(1)
    low24 = by_symbol.transform(lambda s: s.rolling(24, min_periods=6).min()).shift(1)
    high24 = by_symbol.transform(lambda s: s.rolling(24, min_periods=6).max()).shift(1)
    prev_price = by_symbol.shift(1)
    bars["zscore_24h"] = (prev_price - mean24) / std24.replace(0, np.nan)
    bars["range_pos_24h"] = ((prev_price - low24)
                             / (high24 - low24).replace(0, np.nan))
    bars["vol_regime"] = bars["ctx_vol_24h"] / bars.groupby(
        "symbol", observed=True)["ctx_vol_24h"].transform(
        lambda s: s.rolling(168, min_periods=24).mean()).replace(0, np.nan)
    frame = frame.merge(
        bars[["symbol", "symbol_hour", "ctx_return_1h", "ctx_return_4h",
              "ctx_return_24h", "ctx_vol_24h", "ctx_flow_prev",
              "zscore_24h", "range_pos_24h", "vol_regime"]],
        on=["symbol", "symbol_hour"], how="left")

    # Backward markouts in volatility units: how far the instrument has run
    # into this entry, measured in units every symbol shares. with_momentum is
    # the interaction that asks the real question -- is the client leaning
    # WITH or AGAINST the move -- and momentum_align its trend-agreement form.
    vol = frame["ctx_vol_24h"].replace(0, np.nan)
    frame["mom_1h_vol"] = (frame["ctx_return_1h"] / vol).clip(-20, 20)
    frame["mom_4h_vol"] = (frame["ctx_return_4h"] / (vol * 2.0)).clip(-20, 20)
    frame["mom_24h_vol"] = (frame["ctx_return_24h"] / (vol * 4.9)).clip(-20, 20)
    frame["with_momentum"] = frame["direction"] * frame["mom_4h_vol"]
    frame["momentum_align"] = frame["direction"] * (
        np.sign(frame["ctx_return_1h"].fillna(0))
        + np.sign(frame["ctx_return_4h"].fillna(0))
        + np.sign(frame["ctx_return_24h"].fillna(0))) / 3.0

    # Bracket placement in volatility units: a 20-pip stop on a quiet pair and
    # a $10 stop on gold become the same number when both are "stops this many
    # daily-vols away" -- the cross-symbol form of the model's top feature.
    frame["sl_dist_vol"] = (frame["sl_distance"] / vol).clip(-50, 50)
    frame["tp_dist_vol"] = (frame["tp_distance"] / vol).clip(-50, 50)

    frame["trade_index"] = frame.groupby("account_key", observed=True).cumcount()
    # Activity shape: the burst trader and the once-a-day trader are different
    # animals even at identical lifetime stats.
    frame["trades_today"] = frame.groupby(
        ["account_key", "day"], observed=True).cumcount()
    frame["hours_since_last"] = (
        frame.groupby("account_key", observed=True)["open_time"].diff()
        .dt.total_seconds() / 3600.0).clip(upper=24 * 14)
    frame["duration_s"] = (frame["close_time"] - frame["open_time"]).dt.total_seconds()
    frame["past_edge"] = frame["net_profit"] / frame["notional"].replace(0, np.nan)

    # Account history in CLOSE-time order, then as-of joined to each OPEN.
    # An expanding mean in open-time order would aggregate outcomes of positions
    # that had not closed yet -- and hist_win_rate is among the most-used
    # features, so that leak would matter.
    closed = frame.sort_values("close_time", kind="mergesort")
    by_close = closed.groupby("account_key", observed=True)
    history = pd.DataFrame({
        "account_key": closed["account_key"].astype(str).to_numpy(),
        "close_time": closed["close_time"].to_numpy(),
        "hist_win_rate": by_close["net_profit"].transform(lambda s: (s > 0).expanding().mean()).to_numpy(),
        "hist_mean_pnl": by_close["net_profit"].transform(lambda s: s.expanding().mean()).to_numpy(),
        "hist_pnl_std": by_close["net_profit"].transform(lambda s: s.expanding().std()).to_numpy(),
        "hist_mean_notional": by_close["notional"].transform(lambda s: s.expanding().mean()).to_numpy(),
        "hist_duration": by_close["duration_s"].transform(lambda s: s.expanding().mean()).to_numpy(),
        "hist_edge": by_close["past_edge"].transform(lambda s: s.expanding().mean()).to_numpy(),
        "hist_cum_pnl": by_close["net_profit"].transform(lambda s: s.expanding().sum()).to_numpy(),
        "hist_peak_pnl": by_close["net_profit"].transform(
            lambda s: s.expanding().sum().cummax()).to_numpy(),
        "hist_recent_pnl5": by_close["net_profit"].transform(
            lambda s: s.rolling(5, min_periods=1).sum()).to_numpy(),
    })

    # The recency ladder, computed per account on the close-ordered arrays
    # (the as-of join below is what keeps it strictly pre-open).
    def _ladder(values: np.ndarray) -> np.ndarray:
        n = len(values)
        out = np.full((n, 8), np.nan)
        cw = np.concatenate([[0.0], np.cumsum((values > 0).astype("float64"))])
        cp = np.concatenate([[0.0], np.cumsum(values)])
        counts = np.arange(1, n + 1)
        for column, fraction in enumerate((0.20, 0.10, 0.05, 0.02)):
            window = np.minimum(np.maximum(3, (counts * fraction).astype(int)),
                                counts)
            lo = counts - window
            out[:, column] = (cw[counts] - cw[lo]) / window
            out[:, 4 + column] = (cp[counts] - cp[lo]) / window
        return out

    ladder = np.full((len(closed), 8), np.nan)
    pnl_all = pd.to_numeric(closed["net_profit"], errors="coerce") \
        .fillna(0.0).to_numpy(dtype="float64")
    # Regime-conditional record: "trend" = the past trade went WITH the
    # 4h momentum, "chop" = against it (with_momentum sign -- cross-symbol
    # by construction). Expanding, per account, causal via the as-of join.
    with_flag = (pd.to_numeric(closed["with_momentum"], errors="coerce")
                 > 0).to_numpy()
    conditional = np.full((len(closed), 4), np.nan)
    for indices in closed.groupby("account_key", observed=True).indices.values():
        ladder[indices] = _ladder(pnl_all[indices])
        flags = with_flag[indices]
        values = pnl_all[indices]
        won = (values > 0).astype("float64")
        for column, mask in ((0, flags), (1, ~flags)):
            count = np.cumsum(mask.astype("float64"))
            wins = np.cumsum(won * mask)
            total = np.cumsum(values * mask)
            with np.errstate(invalid="ignore", divide="ignore"):
                conditional[indices, column] = np.where(
                    count > 0, wins / count, np.nan)
                conditional[indices, 2 + column] = np.where(
                    count > 0, total / count, np.nan)
    for position, name in enumerate(
            ["hist_wr_20p", "hist_wr_10p", "hist_wr_5p", "hist_wr_2p",
             "hist_mp_20p", "hist_mp_10p", "hist_mp_5p", "hist_mp_2p"]):
        history[name] = ladder[:, position]
    for position, name in enumerate(
            ["hist_wr_trend", "hist_wr_chop",
             "hist_mp_trend", "hist_mp_chop"]):
        history[name] = conditional[:, position]
    history = history.sort_values("close_time", kind="mergesort")

    frame = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    frame["account_key"] = frame["account_key"].astype(str)
    frame = pd.merge_asof(
        frame, history, left_on="open_time", right_on="close_time", by="account_key",
        direction="backward", allow_exact_matches=False, suffixes=("", "_hist"))
    frame["notional_vs_usual"] = frame["notional"] / frame["hist_mean_notional"].replace(0, np.nan)

    # ACCOUNT-DAY ACTIVITY. Aggregated by CLOSE day -- realised P&L lands on
    # the day it realises -- then joined to each trade's OPEN day with
    # allow_exact_matches=False on the day grain: a trade only ever sees days
    # that finished BEFORE the day it opens, so today's still-accruing P&L and
    # any position still running are structurally excluded.
    close_day = frame["close_time"].dt.floor("D")
    daily = (pd.DataFrame({"account_key": frame["account_key"],
                           "close_day": close_day,
                           "pnl": frame["net_profit"]})
             .dropna(subset=["close_day"])
             .groupby(["account_key", "close_day"], observed=True)
             .agg(day_pnl=("pnl", "sum"), day_trades=("pnl", "size"),
                  day_wins=("pnl", lambda s: float((s > 0).sum())))
             .reset_index().sort_values(["account_key", "close_day"]))
    grouped_daily = daily.groupby("account_key", observed=True)
    daily["acct_pnl_5d"] = grouped_daily["day_pnl"].transform(
        lambda s: s.rolling(5, min_periods=1).sum())
    daily["acct_pnl_20d"] = grouped_daily["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=1).sum())
    daily["acct_pnl_60d"] = grouped_daily["day_pnl"].transform(
        lambda s: s.rolling(60, min_periods=1).sum())
    daily["acct_vol_20d"] = grouped_daily["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=3).std())
    wins20 = grouped_daily["day_wins"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    trades20 = grouped_daily["day_trades"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    daily["acct_winrate_20d"] = wins20 / trades20.replace(0, np.nan)
    daily["acct_trades_5d"] = grouped_daily["day_trades"].transform(
        lambda s: s.rolling(5, min_periods=1).sum())
    daily["acct_trades_20d"] = trades20
    cumulative = grouped_daily["day_pnl"].cumsum()
    peak20 = cumulative.groupby(daily["account_key"], observed=True).transform(
        lambda s: s.rolling(20, min_periods=1).max())
    daily["acct_dd_20d"] = peak20 - cumulative
    daily["acct_best_20d"] = grouped_daily["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=1).max())
    daily["acct_worst_20d"] = grouped_daily["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=1).min())
    daily["acct_rank_20d"] = daily.groupby("close_day", observed=True)[
        "acct_pnl_20d"].rank(pct=True)
    first_day = grouped_daily["close_day"].transform("min")
    daily["acct_tenure_days"] = (daily["close_day"] - first_day).dt.days

    acct_columns = ["acct_pnl_5d", "acct_pnl_20d", "acct_pnl_60d",
                    "acct_vol_20d", "acct_winrate_20d", "acct_trades_5d",
                    "acct_trades_20d", "acct_dd_20d", "acct_best_20d",
                    "acct_worst_20d", "acct_rank_20d", "acct_tenure_days"]
    frame = pd.merge_asof(
        frame.sort_values("day", kind="mergesort"),
        daily[["account_key", "close_day"] + acct_columns].sort_values(
            "close_day", kind="mergesort"),
        left_on="day", right_on="close_day", by="account_key",
        direction="backward", allow_exact_matches=False)
    frame["acct_days_since_active"] = (
        (frame["day"] - frame["close_day"]).dt.days.clip(upper=90))

    # THE ACCOUNT-DAY CORPUS -- shared verbatim from the watchlist model's
    # cached frame. allow_exact_matches=False is load-bearing: a corpus row
    # carries THAT day's realised outcomes, so a trade may only ever see
    # rows from days strictly before its own open day.
    try:
        corpus = ad_corpus()
        frame = pd.merge_asof(
            frame.sort_values("day", kind="mergesort"), corpus,
            left_on="day", right_on="decision_day", by="account_key",
            direction="backward", allow_exact_matches=False)
    except Exception:
        for name in AD_FEATURES:
            if name not in frame.columns:
                frame[name] = np.nan
    frame = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True)

    # Funding state as of the trade's open day, from the locally cached cash
    # movements. Enrichment, not a dependency: with no cache the columns exist
    # as zeros/NaN and the model simply learns nothing from them, rather than
    # the whole training run failing for want of a deposits table.
    try:
        from webapp import cashflow_store
        daily = _cashflow_daily()
        frame = cashflow_store.attach(frame, daily)
    except Exception:
        for column in ("deposits_to_date", "withdrawals_to_date", "net_funding_to_date",
                       "deposit_count_to_date", "withdrawal_count_to_date",
                       "days_since_deposit", "days_since_withdrawal",
                       "withdrawal_ratio", "funding_churn"):
            if column not in frame.columns:
                frame[column] = np.nan

    # ACCOUNT STATE at the moment of the trade. equity_proxy is what the
    # client demonstrably has in the window: settled funding plus realised
    # P&L to date. notional_to_equity is this trade's footprint against it --
    # the margin-impact signature -- and underwater_frac how deep below their
    # own high-water mark they are trading from (revenge-trading shows up
    # here). All strictly pre-open by construction of the inputs.
    funding = pd.to_numeric(frame.get("net_funding_to_date"), errors="coerce")
    cum_pnl = pd.to_numeric(frame.get("hist_cum_pnl"), errors="coerce")
    frame["equity_proxy"] = funding.fillna(0.0) + cum_pnl.fillna(0.0)
    denominator = frame["equity_proxy"].where(frame["equity_proxy"] > 0)
    frame["notional_to_equity"] = (frame["notional"] / denominator).clip(upper=1000)
    peak = pd.to_numeric(frame.get("hist_peak_pnl"), errors="coerce")
    frame["underwater_frac"] = ((peak - cum_pnl) / denominator).clip(0, 100)
    return frame


_CASHFLOW_DAILY_CACHE = None
_CASHFLOW_DAY = None


def _cashflow_daily():
    """Cashflow daily-features, cached per calendar day. build_trade_features
    runs once in training but per-trade in live scoring; re-reading and
    re-aggregating the cash movements each call would dominate the hot path."""
    global _CASHFLOW_DAILY_CACHE, _CASHFLOW_DAY
    import datetime as _dt
    today = _dt.date.today()
    if _CASHFLOW_DAILY_CACHE is not None and _CASHFLOW_DAY == today:
        return _CASHFLOW_DAILY_CACHE
    from webapp import cashflow_store
    movements = cashflow_store.read_cashflows()
    _CASHFLOW_DAILY_CACHE = (cashflow_store.daily_features(movements)
                             if not movements.empty else movements)
    _CASHFLOW_DAY = today
    return _CASHFLOW_DAILY_CACHE


_AD_CORPUS_CACHE = None
_AD_CORPUS_MTIME = None


def ad_corpus() -> pd.DataFrame:
    """The account-day corpus, prepared ONCE and cached (renamed to AD_FEATURES,
    numeric-coerced, sorted by decision_day). Shared by training's batch build
    and live per-trade scoring so both as-of join the identical frame."""
    global _AD_CORPUS_CACHE, _AD_CORPUS_MTIME
    path = _AD_DIR / "model_frame.parquet"
    mtime = path.stat().st_mtime
    if _AD_CORPUS_CACHE is not None and _AD_CORPUS_MTIME == mtime:
        return _AD_CORPUS_CACHE
    corpus = pd.read_parquet(
        path, columns=["account_key", "decision_day"] + AD_COLUMNS)
    corpus["decision_day"] = pd.to_datetime(corpus["decision_day"])
    corpus["account_key"] = corpus["account_key"].astype(str)
    corpus = corpus.rename(columns=dict(zip(AD_COLUMNS, AD_FEATURES)))
    for name in AD_FEATURES:
        corpus[name] = pd.to_numeric(corpus[name], errors="coerce").astype("float32")
    _AD_CORPUS_CACHE = corpus.sort_values("decision_day", kind="mergesort")
    _AD_CORPUS_MTIME = mtime
    return _AD_CORPUS_CACHE


#: Raw columns build_trade_features consumes -- the schema live must assemble.
SCORING_RAW_COLUMNS = ["database", "account_key", "symbol", "cmd", "volume_lots",
                       "open_time", "close_time", "open_price", "close_price",
                       "sl", "tp", "net_profit", "state", "reason"]


def build_features_for_scoring(new_trades: pd.DataFrame,
                               tape: pd.DataFrame | None) -> pd.DataFrame:
    """PARITY-BY-CONSTRUCTION live scoring builder.

    Runs the SAME build_trade_features the training pipeline uses, on a frame of
    the account's own recent CLOSED trades (`tape` -- providing the expanding
    history + as-of context) plus the OPEN trade(s) being scored (`new_trades`).
    Returns the built rows for `new_trades` only, aligned to their input order,
    every history/AD/funding feature computed by the identical code path.

    Both inputs share SCORING_RAW_COLUMNS. Open trades carry close_time=NaT and
    net_profit=NaN so they never leak into the account's history aggregation.
    """
    new = new_trades.copy().reset_index(drop=True)
    new["_row_id"] = np.arange(len(new))
    new["_score_row"] = True
    if tape is not None and len(tape):
        base = tape.copy()
        base["_score_row"] = False
        base["_row_id"] = -1
        frame = pd.concat([base, new], ignore_index=True, sort=False)
    else:
        frame = new
    # Unify datetime resolution to [us] -- the training cache AND the corpora
    # (model_frame decision_day) are [us]; a stray [ns] NaT from the open rows
    # makes merge_asof reject mismatched keys and silently NaN the AD join.
    for col in ("open_time", "close_time"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col]).astype("datetime64[us]")
    # Open trades have no close yet. A NULL close_time breaks the history
    # merge key AND would let the open trade leak into its own account history.
    # Sentinel it FAR into the future: it then sorts last in close-order (never
    # informs a prior trade's expanding stats) and its own backward/strict
    # as-of join excludes it, so it correctly sees only prior CLOSED trades.
    open_mask = (frame["_score_row"] == True) & frame["close_time"].isna()
    if open_mask.any():
        frame.loc[open_mask, "close_time"] = (
            frame.loc[open_mask, "open_time"] + pd.Timedelta(days=3650))
    built = build_trade_features(frame)
    out = built[built["_score_row"] == True].sort_values("_row_id")
    return out.reset_index(drop=True)
