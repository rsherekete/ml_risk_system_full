"""Synthetic smoke test for the new research.py / risk.py functionality."""
import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")

import numpy as np
import pandas as pd

from trading_data.research import (
    Economics,
    asset_class,
    attach_symbol_specs,
    canonical_symbol,
    contract_size_fallback,
    daily_account_features,
    efficient_frontier,
    evaluate_scenario,
    fit_oos_expected_value,
    fit_oos_predictions,
    normalize_symbol_map,
    prior_day_features,
    recommendation,
    recent_account_features,
    rolling_symbol_exposure,
    supervised_dataset,
    symbol_mapping_table,
)
from trading_data.risk import (
    daily_account_exposure,
    daily_symbol_drivers,
    portfolio_var,
    symbol_price_series,
    symbol_returns,
    symbol_var,
    symbol_volatility,
)

rng = np.random.default_rng(7)

# --- symbol mapping -----------------------------------------------------
tests = {
    "EURUSD": "EURUSD",
    "eurusd.e": "EURUSD",
    "EURUSD.RAW.PRO": "EURUSD",
    "EURUSDm": "EURUSD",
    "GOLD": "XAUUSD",
    "XAUUSD.RAW": "XAUUSD",
    "XAUUSDMIN": "XAUUSD",
    "BTCUSDMIN": "BTCUSD",
    "NZDUSD": "NZDUSD",   # must NOT get eaten by the trailing "D" heuristic
    "USDCAD.ecn": "USDCAD",
    "US30CASH": "US30",
    "USTEC": "USTEC",       # regression: glued-suffix "C" strip must not eat the real index root
    "NAS100": "USTEC",
    "USTECm": "USTEC",
    "USTECc": "USTEC",
    "USTEC.raw": "USTEC",
    # regressions found against real broker data:
    "NGmin": "NG",          # short commodity root + MIN must merge...
    "INTC": "INTC",         # ...without wrongly truncating a real ticker ending in a bare C
    "CUmin": "COPPER",      # venue-specific alternate commodity code, explicit alias
    "XAUUSDt": "XAUUSD",    # trailing lowercase "t" decorator, explicit alias
    "T": "T",               # AT&T's real 1-char ticker must never be touched
    "BABA": "BABA",
    "AMT": "AMT",
}
for raw, expected in tests.items():
    got = canonical_symbol(raw)
    status = "OK" if got == expected else "MISMATCH"
    print(f"canonical_symbol({raw!r}) = {got!r} (expected {expected!r}) [{status}]")
    assert got == expected, f"{raw} -> {got}, expected {expected}"

print("asset_class(EURUSD) =", asset_class("EURUSD"))
print("asset_class(XAGUSD) =", asset_class("XAGUSD"), "fallback contract size =", contract_size_fallback("XAGUSD"))
print("asset_class(BTCUSD) =", asset_class("BTCUSD"))
print("asset_class(EURTRY, cross) =", asset_class("EURTRY"))
print("asset_class(US30) =", asset_class("US30"))

# --- build synthetic trade records ---------------------------------------
n = 4000
days = pd.date_range("2026-05-01", periods=90, freq="D")
symbols_raw = ["EURUSD", "EURUSD.raw", "XAUUSD", "GOLD", "BTCUSDMIN", "US30CASH"]
accounts = [f"acct_{i}" for i in range(40)]

timestamps = rng.choice(days, size=n) + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
symbol_choice = rng.choice(symbols_raw, size=n, p=[0.30, 0.10, 0.20, 0.05, 0.15, 0.20])
cmd_choice = rng.choice(["buy", "sell"], size=n)
account_choice = rng.choice(accounts, size=n)
volume_lots = np.round(rng.uniform(0.01, 5.0, size=n), 2)

base_price = {"EURUSD": 1.08, "EURUSD.raw": 1.08, "XAUUSD": 2350.0, "GOLD": 2350.0, "BTCUSDMIN": 65000.0, "US30CASH": 39000.0}
open_price = np.array([base_price[s] for s in symbol_choice]) * (1 + rng.normal(0, 0.01, n))
close_price = open_price * (1 + rng.normal(0, 0.01, n))
profit = (close_price - open_price) * np.where(cmd_choice == "buy", 1, -1) * volume_lots * 1000
state_choice = rng.choice(["open", "closed"], size=n, p=[0.2, 0.8])

commission = -np.abs(volume_lots) * 7.0  # a per-lot commission cost, always negative
swap = rng.normal(0, 2.0, n)

records = pd.DataFrame({
    "timestamp": pd.to_datetime(timestamps),
    "login": account_choice,
    "account_key": account_choice,
    "database": "mt5_demo01",
    "platform": "mt5",
    "region": "global",
    "symbol": symbol_choice,
    "cmd": cmd_choice,
    "volume_lots": volume_lots,
    "open_time": pd.to_datetime(timestamps) - pd.to_timedelta(rng.integers(1, 5000, n), unit="s"),
    "close_time": pd.to_datetime(timestamps),
    "open_price": open_price,
    "close_price": close_price,
    "price": open_price,
    "state": state_choice,
    "reason": "client",
    "profit": np.where(state_choice == "closed", profit, np.nan),
    "commission": np.where(state_choice == "closed", commission, np.nan),
    "swap": np.where(state_choice == "closed", swap, np.nan),
})
records["net_profit"] = records["profit"].fillna(0.0) + records["commission"].fillna(0.0) + records["swap"].fillna(0.0)
records.loc[records["state"] != "closed", "net_profit"] = np.nan

# Inject money-operation rows (deposits/withdrawals/commission rebates) that a
# real MT4/MT5 venue books as ordinary rows in the same table, with `state`
# looking exactly like a closed trade -- this is precisely the case
# is_realised_trade/realised_trade_pnl must filter out by `cmd`, not `state`.
money_ops = pd.DataFrame({
    "timestamp": [days[10], days[20], days[30]],
    "login": ["acct_1", "acct_2", "acct_3"],
    "account_key": ["acct_1", "acct_2", "acct_3"],
    "database": "mt5_demo01",
    "platform": "mt5",
    "region": "global",
    "symbol": ["", "", ""],
    "cmd": ["balance", "credit", "commission"],
    "volume_lots": [0.0, 0.0, 0.0],
    "open_time": [days[10], days[20], days[30]],
    "close_time": [days[10], days[20], days[30]],
    "open_price": [0.0, 0.0, 0.0],
    "close_price": [0.0, 0.0, 0.0],
    "price": [0.0, 0.0, 0.0],
    "state": ["closed", "closed", "closed"],  # exactly the trap: state looks closed
    "reason": "client",
    "profit": [50_000.0, 25_000.0, -1_000.0],  # a huge "deposit" that must NOT count as trading P/L
    "commission": [0.0, 0.0, 0.0],
    "swap": [0.0, 0.0, 0.0],
})
money_ops["net_profit"] = money_ops["profit"]
records = pd.concat([records, money_ops], ignore_index=True)
records = records.sort_values("timestamp").reset_index(drop=True)

specs = pd.DataFrame({
    "symbol": ["EURUSD", "XAUUSD"],
    "contract_size": [100_000.0, 100.0],
    "tick_value": [1.0, 1.0],
    "tick_size": [0.00001, 0.01],
    "currency_base": ["EUR", "XAU"],
    "currency_profit": ["USD", "USD"],
    "currency_margin": ["EUR", "USD"],
})

records = attach_symbol_specs(records, specs)
print("\nnotional_status counts:\n", records["notional_status"].value_counts())
print("contract_size_source counts:\n", records["contract_size_source"].value_counts())
assert records["notional_usd"].notna().any()

specs_by_db = {"mt5_demo01": specs}
mapping = symbol_mapping_table(records, specs_by_db)
print("\nsymbol_mapping_table:\n", mapping)
assert set(mapping["canonical_symbol"]) == {"EURUSD", "XAUUSD", "BTCUSD", "US30"}

as_of = days[-5]
exposure = daily_account_exposure(records, as_of, top_n=5)
print("\ndaily_account_exposure rows:", len(exposure), "rank types:", exposure["rank_type"].unique())

drivers = daily_symbol_drivers(records, as_of, Economics())
print("\ndaily_symbol_drivers:\n", drivers)

price_series = symbol_price_series(records)
returns = symbol_returns(price_series)
print("\nprice_series rows:", len(price_series), "returns rows:", len(returns))
vol = symbol_volatility(returns, as_of)
print("volatility:\n", vol)

var_table = symbol_var(records, as_of, windows=(1, 7, 20), confidence=0.95)
print("\nsymbol_var:\n", var_table)
assert (var_table["parametric_gross_var_usd"] >= 0).all()
assert set(var_table["window_days"].unique()) == {1, 7, 20}
assert "hist_loss_quantile_return" in var_table.columns and "hist_gain_quantile_return" in var_table.columns

# --- regression: historical VaR must use the tail matching net direction ---
from trading_data.risk import _historical_quantile

fake_returns = pd.DataFrame({
    "canonical_symbol": ["SKEWED"] * 100,
    "day": pd.date_range("2026-01-01", periods=100, freq="D"),
    "log_return": np.concatenate([np.full(85, -0.001), np.full(15, 0.05)]),  # small steady losses, frequent-enough huge spikes up
})
hq = _historical_quantile(fake_returns, window=1, confidence=0.95, as_of=fake_returns["day"].max())
row = hq.loc[hq["canonical_symbol"] == "SKEWED"].iloc[0]
print("\nskewed-series quantiles:", dict(row))
assert row["hist_gain_quantile_return"] > 0.03, "right tail should land inside the +5% spike cluster (15% of the distribution)"
assert row["hist_loss_quantile_return"] < 0, "left tail should be a small steady loss, not a spike"
assert abs(row["hist_gain_quantile_return"]) > abs(row["hist_loss_quantile_return"]), (
    "a short position's risk (right tail) must be recognised as larger than a long's (left tail) for this skew"
)

pf_var = portfolio_var(records, as_of, windows=(1, 7, 20), confidence=0.95)
print("\nportfolio_var:\n", pf_var)
assert (pf_var["diversified_gross_var_usd"] <= pf_var["undiversified_gross_var_usd"] + 1e-6).all()

# --- regression: unmeasured correlation must fall back to rho=+1 (no free
# diversification credit), not rho=0 -- two brand-new symbols with a single
# day of overlap (below min_overlap) should show ZERO diversification benefit ---
thin_day = as_of
thin_records = pd.DataFrame({
    "timestamp": pd.to_datetime([thin_day, thin_day]),
    "symbol": ["EURUSD", "XAUUSD"],
    "canonical_symbol": ["EURUSD", "XAUUSD"],
    "cmd": ["buy", "buy"],
    "volume_lots": [10.0, 10.0],
    "notional_usd": [1_000_000.0, 1_000_000.0],
    "open_time": pd.to_datetime([thin_day, thin_day]),
    "close_time": pd.to_datetime([thin_day, thin_day]),
    "open_price": [1.08, 2350.0],
    "close_price": [1.081, 2360.0],
    "account_key": ["a1", "a2"],
})
thin_pf_var = portfolio_var(thin_records, thin_day, windows=(1,), confidence=0.95, min_overlap=20)
print("\nthin-history portfolio_var (expect ~0 diversification benefit):\n", thin_pf_var)
assert thin_pf_var.loc[0, "gross_diversification_benefit"] < 1e-6, "unmeasured correlation must not grant free diversification credit"

rolling = rolling_symbol_exposure(records, as_of)
print("\nrolling_symbol_exposure:\n", rolling.head())

# --- efficient frontier / evaluate_scenario ---
features = daily_account_features(records)
dataset = supervised_dataset(features)
print("\ndataset rows:", len(dataset))
predictions = fit_oos_predictions(dataset, min_train_days=15)
daily, metrics = evaluate_scenario(predictions, profit_weight=70, drawdown_weight=30)
print("\nmetrics @ 70/30:", metrics)
daily2, metrics2 = evaluate_scenario(predictions, profit_weight=30, drawdown_weight=70)
print("metrics @ 30/70:", metrics2)

frontier = efficient_frontier(predictions, steps=11)
print("\nfrontier:\n", frontier[["profit_weight", "drawdown_weight", "total_proxy_pnl", "max_drawdown", "is_pareto_optimal"]])
assert frontier["is_pareto_optimal"].any() or frontier["oos_days"].max() == 0

# --- regression: oos_days must reflect actually-predicted days, not every
# decision_day in the dataset (burn-in days included) ---
unique_decision_days = pd.to_datetime(dataset["decision_day"]).nunique()
print(f"\noos_days={metrics['oos_days']} vs total unique decision_days={unique_decision_days} (min_train_days=15)")
assert metrics["oos_days"] <= unique_decision_days
starved_predictions = fit_oos_predictions(dataset, min_train_days=10_000)  # far more than any dataset has -> zero predictions
_, starved_metrics = evaluate_scenario(starved_predictions, profit_weight=70, drawdown_weight=30)
print("starved (no-fit) oos_days:", starved_metrics["oos_days"], "(expected 0.0)")
assert starved_metrics["oos_days"] == 0.0, "with zero walk-forward predictions, oos_days must be 0, not the raw day count"

# --- regression: a malformed symbol_map.yaml must not crash canonical_symbol ---
import os
from trading_data.research import reload_symbol_aliases

bad_yaml_path = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bad_symbol_map.yaml"
with open(bad_yaml_path, "w", encoding="utf-8") as fh:
    fh.write("EURUSD.X: EURUSD\n  bad indentation: [unterminated\n")
os.environ["TRADING_SYMBOL_MAP"] = bad_yaml_path
reload_symbol_aliases()
try:
    result = canonical_symbol("EURUSD.E")
    print(f"\nmalformed symbol_map.yaml handled gracefully; canonical_symbol('EURUSD.E') = {result!r}")
    assert result == "EURUSD"
finally:
    del os.environ["TRADING_SYMBOL_MAP"]
    reload_symbol_aliases()

recent = recent_account_features(features, as_of, 7)
rec = recommendation(recent)
print("\nrecommendation counts:\n", rec["recommendation"].value_counts())

prior = prior_day_features(features)
print("\nprior_day_features rows:", len(prior))

# --- book_assignment: new flagship module ---
from trading_data.research import client_intelligence, detect_arbitrage, detect_edge_clients, detect_fraud_review, detect_toxicity
from trading_data.book_assignment import assign_books, daily_intelligence_components, expanding_account_pnl_volatility, expanding_client_profile, firm_daily_pnl

intel = client_intelligence(records)
print("\nclient_intelligence (composed detectors) columns:", list(intel.columns))
for col in ("edge_screen", "concentration_flag", "arbitrage_screen", "fraud_review_screen", "toxicity_score"):
    assert col in intel.columns, f"missing {col} in composed client_intelligence"

components = daily_intelligence_components(records)
print("\ndaily_intelligence_components rows:", len(components), "columns:", list(components.columns))
assert (components["observations"] >= components["close_count"]).all()

profile = expanding_client_profile(components)
print("expanding_client_profile rows:", len(profile))
# regression: expanding stats must be monotonically non-decreasing per account (cumulative)
check = profile.sort_values(["account_key", "decision_day"])
non_decreasing = check.groupby("account_key", observed=True)["cum_observations"].apply(lambda s: (s.diff().dropna() >= 0).all())
assert non_decreasing.all(), "cumulative observation counts must never decrease within an account"
# regression: a decision_day's expanding stats must equal cumsum through THAT day (inclusive), not before it
first_account = profile["account_key"].iloc[0]
acct_rows = profile.loc[profile["account_key"] == first_account].sort_values("decision_day")
acct_components = components.loc[components["account_key"] == first_account].sort_values("day")
assert acct_rows["cum_observations"].iloc[-1] == acct_components["observations"].sum(), (
    "last decision_day's cum_observations must equal the account's all-time total (inclusive of its own day)"
)

predictions_for_books = fit_oos_expected_value(dataset, min_train_days=15)
volatility = expanding_account_pnl_volatility(components)
print("expanding_account_pnl_volatility rows:", len(volatility), "sources:", volatility["pnl_vol_source"].value_counts().to_dict())
assignment = assign_books(predictions_for_books, profile, volatility, profit_weight=70, drawdown_weight=30)
print("\nassign_books book counts:\n", assignment["book"].value_counts())
assert set(assignment["book"].unique()) <= {"A_BOOK", "B_BOOK"}
# regression: an account with a demonstrated persistent edge must never be B_BOOK
edge_rows = assignment.loc[assignment["expanding_edge_flag"].fillna(False)]
if not edge_rows.empty:
    assert (edge_rows["book"] == "A_BOOK").all(), "a demonstrated persistent-edge account must always be A_BOOK regardless of the model score"
# regression: no-prediction rows must default to B_BOOK, never silently A_BOOK
no_prediction_rows = assignment.loc[assignment["low_confidence"] & ~(assignment["expanding_edge_flag"].fillna(False) | assignment["expanding_arbitrage_flag"].fillna(False))]
if not no_prediction_rows.empty:
    assert (no_prediction_rows["book"] == "B_BOOK").all(), "a no-prediction, non-override account must default to B_BOOK, never A_BOOK"
    assert (no_prediction_rows["reason_codes"] == "INSUFFICIENT_HISTORY_DEFAULT_BBOOK").all()

daily_pnl, detail, pnl_metrics = firm_daily_pnl(records, assignment, Economics())
print("\nfirm_daily_pnl metrics:", pnl_metrics)
print("firm_daily_pnl daily columns:", list(daily_pnl.columns))
assert np.allclose(daily_pnl["equity_usd"].diff().dropna().to_numpy(), daily_pnl["firm_pnl_usd"].iloc[1:].to_numpy())
assert (daily_pnl["drawdown_usd"] <= 1e-9).all(), "drawdown must never be positive"
assert pnl_metrics["max_daily_drawdown_usd"] <= 1e-9
# regression: A-book detail rows must never carry a nonzero client-P&L-driven firm P&L formula
a_book_detail = detail.loc[detail["book"] == "A_BOOK"]
if not a_book_detail.empty:
    assert np.allclose(a_book_detail["firm_pnl_usd"], a_book_detail["expected_net_markup_usd"])
b_book_detail = detail.loc[detail["book"] == "B_BOOK"]
if not b_book_detail.empty:
    assert np.allclose(b_book_detail["firm_pnl_usd"], -b_book_detail["client_realised_profit"])

# sanity: different profit/drawdown sliders should be able to change the split
assignment_conservative = assign_books(predictions_for_books, profile, volatility, profit_weight=20, drawdown_weight=80)
print("assign_books @ 20/80 book counts:\n", assignment_conservative["book"].value_counts())

# --- regression: THE core promise of the risk-budget-walk redesign -- expected
# firm P&L and the risk budget consumed must be monotonically non-decreasing as
# profit_weight rises, unlike the old shared-probability-threshold design (which
# could and did invert on real data). Sweep coarsely here; the fine-grained
# 21-step sweep is `real_efficient_frontier`'s job, checked in the real-data script. ---
mono_pnl, mono_risk = [], []
for pw in (0, 20, 40, 60, 80, 100):
    a = assign_books(predictions_for_books, profile, volatility, profit_weight=pw, drawdown_weight=100 - pw)
    b_book = a.loc[a["book"] == "B_BOOK"]
    # B-book's own excess value over hedging -- exactly what the risk-budget walk controls
    mono_pnl.append(b_book["excess_value_usd"].sum())
    mono_risk.append((b_book["sigma_usd"] ** 2).sum())
print("\nrisk-budget sweep -- B-book expected excess value:", [round(v, 2) for v in mono_pnl])
print("risk-budget sweep -- B-book risk (sum sigma^2):", [round(v, 2) for v in mono_risk])
assert all(mono_pnl[i] <= mono_pnl[i + 1] + 1e-6 for i in range(len(mono_pnl) - 1)), "B-book expected excess value must be monotonically non-decreasing in profit_weight"
assert all(mono_risk[i] <= mono_risk[i + 1] + 1e-6 for i in range(len(mono_risk) - 1)), "B-book risk budget consumed must be monotonically non-decreasing in profit_weight"

# --- regression: money-operation rows (deposit/credit/commission-rebate)
# must NEVER count as trading P/L, however their `state` reads ---
from trading_data.research import is_realised_trade, realised_trade_pnl

money_op_rows = records.loc[records["cmd"].isin(["balance", "credit", "commission"])]
assert len(money_op_rows) == 3, "expected the 3 injected money-op rows to survive unfiltered in `records`"
assert not is_realised_trade(money_op_rows).any(), "money operations must never be classified as a realised trade"
assert (realised_trade_pnl(money_op_rows) == 0.0).all(), "money operations must contribute zero to realised trade P/L"

daily_features_all = daily_account_features(records)
acct1_features = daily_features_all.loc[(daily_features_all["account_key"] == "acct_1") & (daily_features_all["day"] == days[10])]
expected_real_only = records.loc[
    (records["account_key"] == "acct_1") & (records["timestamp"].dt.floor("D") == days[10])
    & (records["state"] == "closed") & (~records["cmd"].isin(["balance", "credit", "commission"]))
]["net_profit"].sum()  # daily_account_features now prefers net_profit (price P/L + commission + swap) over raw profit
if not acct1_features.empty:
    realised = float(acct1_features["realised_profit"].iloc[0])
    print(f"\nacct_1 realised_profit on the day of its injected $50,000 'deposit': {realised:,.2f} (independently-computed real-trades-only sum: {expected_real_only:,.2f})")
    assert np.isclose(realised, expected_real_only), "a deposit booked as cmd=balance must not inflate realised trading P/L"
    assert not np.isclose(realised, expected_real_only + 50_000.0), "the $50,000 deposit must not have been added in"

# --- regression: supervised_dataset must label by CALENDAR next day, not by
# position in the account's row sequence -- a gap must drop to NaN, never
# borrow a much-later day's outcome ---
gap_features = pd.DataFrame({
    "account_key": ["gapacct", "gapacct", "gapacct"],
    "database": ["mt5_demo01"] * 3,
    "platform": ["mt5"] * 3,
    "day": pd.to_datetime(["2026-01-03", "2026-01-04", "2026-04-01"]),  # a ~90-day gap after day 2
    "observations": [5, 5, 5], "symbols": [1, 1, 1], "gross_volume_lots": [1.0, 1.0, 1.0],
    "max_position_lots": [1.0, 1.0, 1.0], "close_count": [5, 5, 5], "avg_position_lots": [0.2] * 3,
    "close_rate": [1.0] * 3, "win_rate": [0.5] * 3, "loss_rate": [0.5] * 3, "profit_per_lot": [1.0] * 3,
    "max_position_share": [0.2] * 3, "loss_day": [False] * 3, "high_concentration": [False] * 3,
    "realised_profit": [100.0, -9999.0, 500.0],  # day 1 -> day 2 is real (1-day gap); day 2 -> day 3 is a 90-day gap
})
gap_dataset = supervised_dataset(gap_features)
print("\ngap-labeled supervised_dataset:\n", gap_dataset[["decision_day", "target_profit"]])
day1_row = gap_dataset.loc[gap_dataset["decision_day"] == pd.Timestamp("2026-01-03")]
assert not day1_row.empty and float(day1_row["target_profit"].iloc[0]) == -9999.0, "day 1's real next-calendar-day (day 2) outcome must be used"
day2_present = gap_dataset.loc[gap_dataset["decision_day"] == pd.Timestamp("2026-01-04")]
assert day2_present.empty, "day 2's row must be DROPPED (its calendar next-day, Jan 5, has no data) -- not mislabeled with day 3's (90 days later) outcome"

# --- regression: a normal weekend-sized gap (well within max_label_gap_days)
# must NOT be dropped -- this is exactly the regression that silently starved
# firm_daily_pnl down to a handful of accounts on real data: an earlier,
# overly strict exact-next-calendar-day version required literal back-to-back
# trading days, which almost no real account satisfies every week ---
weekend_gap_features = pd.DataFrame({
    "account_key": ["weekendacct", "weekendacct"],
    "database": ["mt5_demo01"] * 2, "platform": ["mt5"] * 2,
    "day": pd.to_datetime(["2026-01-02", "2026-01-07"]),  # Friday -> the following Wednesday, a 5-day gap
    "observations": [5, 5], "symbols": [1, 1], "gross_volume_lots": [1.0, 1.0],
    "max_position_lots": [1.0, 1.0], "close_count": [5, 5], "avg_position_lots": [0.2, 0.2],
    "close_rate": [1.0, 1.0], "win_rate": [0.5, 0.5], "loss_rate": [0.5, 0.5], "profit_per_lot": [1.0, 1.0],
    "max_position_share": [0.2, 0.2], "loss_day": [False, False], "high_concentration": [False, False],
    "realised_profit": [200.0, -300.0],
})
weekend_dataset = supervised_dataset(weekend_gap_features)
friday_row = weekend_dataset.loc[weekend_dataset["decision_day"] == pd.Timestamp("2026-01-02")]
assert not friday_row.empty, "a 5-day gap (well within the default 7-day tolerance) must NOT drop the row -- most accounts don't trade every calendar day"
assert float(friday_row["target_profit"].iloc[0]) == -300.0, "the label must be the next actual trading day's outcome, found within the gap tolerance"

# --- regression: firm_daily_pnl must score a day's real P&L against the
# MOST RECENT routing decision strictly before it (not necessarily the exact
# prior calendar day -- most accounts don't trade every day), and never
# against that same day's own decision -- otherwise a day's own outcome could
# retroactively justify how it happened to be routed. This also guards the
# regression this exact check caught: an earlier, too-strict exact-day-1 join
# silently dropped almost every account from `detail`. ---
distinct_accounts_in_detail = detail["account_key"].nunique()
distinct_accounts_total = assignment["account_key"].nunique()
print(f"\nfirm_daily_pnl detail covers {distinct_accounts_in_detail} of {distinct_accounts_total} assigned accounts (must be ALL of them, including first-activity-day-only accounts)")
assert distinct_accounts_in_detail == distinct_accounts_total, (
    f"only {distinct_accounts_in_detail}/{distinct_accounts_total} accounts appear in firm_daily_pnl's detail -- "
    "an account's first-activity-day (no prior decision to match) must be defaulted to B_BOOK, never dropped"
)

# --- regression: an account's very first activity day has no prior decision
# to match against and must be defaulted to B_BOOK (not silently dropped) --
# this is the exact bug that broke "A-book accounts + B-book accounts == total
# active accounts" ---
first_days = detail.sort_values("decision_day").groupby("account_key", observed=True).head(1)
no_prior = first_days.loc[first_days["reason_codes"] == "NO_PRIOR_DECISION_DEFAULT_BBOOK"]
print(f"accounts whose first activity day had no prior decision: {len(no_prior)}")
if not no_prior.empty:
    assert (no_prior["book"] == "B_BOOK").all(), "an account's first-ever activity day (no prior decision) must default to B_BOOK"
    assert (no_prior["low_confidence"]).all(), "a no-prior-decision day must be flagged low_confidence"

# --- regression: firm_daily_pnl_fixed_book gives a clean constant-policy
# baseline for the dashboard's comparison equity chart ---
from trading_data.book_assignment import firm_daily_pnl_fixed_book

always_a = firm_daily_pnl_fixed_book(records, "A_BOOK", Economics())
always_b = firm_daily_pnl_fixed_book(records, "B_BOOK", Economics())
print(f"\nfirm_daily_pnl_fixed_book(A_BOOK): {len(always_a)} days, total pnl={always_a['firm_pnl_usd'].sum():,.0f}")
print(f"firm_daily_pnl_fixed_book(B_BOOK): {len(always_b)} days, total pnl={always_b['firm_pnl_usd'].sum():,.0f}")
assert np.allclose(always_a["firm_pnl_usd"], always_a["expected_net_markup_usd"]), "always-A-book must equal markup revenue every day, never client P/L"
assert np.allclose(always_b["firm_pnl_usd"], -always_b["client_realised_profit"]), "always-B-book must equal -client P/L every day, never markup"
assert np.allclose(always_a["equity_usd"].diff().dropna().to_numpy(), always_a["firm_pnl_usd"].iloc[1:].to_numpy())
assert (always_a["drawdown_usd"] <= 1e-9).all() and (always_b["drawdown_usd"] <= 1e-9).all()
try:
    firm_daily_pnl_fixed_book(records, "C_BOOK", Economics())
    raise AssertionError("firm_daily_pnl_fixed_book must reject an invalid book label")
except ValueError:
    pass

sample_keys = detail[["account_key", "decision_day"]].drop_duplicates().head(30)
checked = 0
for _, row in sample_keys.iterrows():
    activity_day = pd.Timestamp(row["decision_day"])
    account_assignments = assignment.loc[assignment["account_key"] == row["account_key"]].copy()
    account_assignments["decision_day"] = pd.to_datetime(account_assignments["decision_day"])
    prior = account_assignments.loc[account_assignments["decision_day"] < activity_day].sort_values("decision_day")
    detail_row = detail.loc[(detail["account_key"] == row["account_key"]) & (detail["decision_day"] == row["decision_day"])]
    if prior.empty or detail_row.empty:
        continue
    expected_book = prior.iloc[-1]["book"]  # the most recent decision strictly before this activity day
    assert detail_row["book"].iloc[0] == expected_book, (
        f"firm_daily_pnl's book for {row['account_key']} on {row['decision_day']} must match the most recent PRIOR decision, not the same day's"
    )
    checked += 1
print(f"firm_daily_pnl as-of day-lag check: verified {checked} account-day(s) against the most recent prior assignment")
assert checked > 0, "the day-lag regression check must actually verify at least one row"

# --- regression: predict_live must cover the account-day(s) supervised_dataset
# dropped for lacking a future label (usually "today"), or the Book Assignment
# tab would show zero activity for today by construction, not because there
# is none -- this is the exact bug found while testing the dashboard end-to-end ---
from trading_data.research import predict_live

live = predict_live(dataset, features, min_train_days=15)
print("\npredict_live rows:", len(live))
assert not live.empty, "predict_live must cover at least the latest day's account-rows dataset dropped"
assert live["target_profit"].isna().all(), "live rows must never carry a fabricated label"
latest_calendar_day = features["day"].max()
assert (pd.to_datetime(live["decision_day"]) == latest_calendar_day).any(), "predict_live must include the latest day"
# and it must NOT duplicate any (account_key, decision_day) already in predictions_for_books
combined_keys = set(zip(predictions_for_books["account_key"], pd.to_datetime(predictions_for_books["decision_day"])))
live_keys = set(zip(live["account_key"], pd.to_datetime(live["decision_day"])))
assert not (combined_keys & live_keys), "predict_live must not overlap fit_oos_predictions's rows"

# --- regression: predict_live's data-sufficiency gate (`len(days) <
# min_train_days or one-class`) is FIRM-WIDE, not per-account. When it trips,
# every account still needing a live decision must still get a row (with
# model_probability_loss left NaN), never be silently omitted entirely --
# the exact bug that could make a whole account vanish from assign_books'
# output with zero trace anywhere ---
tiny_features = pd.DataFrame({
    "account_key": ["thin_a", "thin_a", "thin_b"],
    "database": ["mt5_demo01"] * 3, "platform": ["mt5"] * 3,
    "day": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-05"]),
    "observations": [3, 3, 2], "symbols": [1, 1, 1], "gross_volume_lots": [1.0, 1.0, 1.0],
    "max_position_lots": [1.0, 1.0, 1.0], "close_count": [3, 3, 2], "avg_position_lots": [0.3] * 3,
    "close_rate": [1.0] * 3, "win_rate": [0.5] * 3, "loss_rate": [0.5] * 3, "profit_per_lot": [1.0] * 3,
    "max_position_share": [0.3] * 3, "loss_day": [False] * 3, "high_concentration": [False] * 3,
    "realised_profit": [50.0, -20.0, 30.0],
})
tiny_dataset = supervised_dataset(tiny_features)
distinct_days = pd.to_datetime(tiny_dataset["decision_day"]).nunique()
print(f"\ntiny_dataset distinct decision_days: {distinct_days} (must be < min_train_days=20 to trip the gate)")
assert distinct_days < 20, "this test requires tripping predict_live's min_train_days gate"
tiny_live = predict_live(tiny_dataset, tiny_features, min_train_days=20)
print("predict_live rows when the firm-wide gate trips:", len(tiny_live))
assert not tiny_live.empty, "predict_live must still return a row per account needing a live decision, even when the model can't yet be trusted"
assert tiny_live["model_probability_loss"].isna().all(), "when the gate trips, model_probability_loss must be NaN, not fabricated"
assert (tiny_live["confidence"] == 0.0).all()

# same gate-doesn't-silently-empty check for the regression twin that real
# book routing actually uses
from trading_data.research import predict_live_expected_value

tiny_live_ev = predict_live_expected_value(tiny_dataset, tiny_features, min_train_days=20)
print("predict_live_expected_value rows when the firm-wide gate trips:", len(tiny_live_ev))
assert not tiny_live_ev.empty, "predict_live_expected_value must still return a row per account needing a live decision"
assert tiny_live_ev["model_expected_client_profit"].isna().all(), "when the gate trips, model_expected_client_profit must be NaN, not fabricated"

live_expected_value = predict_live_expected_value(dataset, features, min_train_days=15)
print("predict_live_expected_value rows:", len(live_expected_value))
assert not live_expected_value.empty
live_assignment = assign_books(
    pd.concat([predictions_for_books, live_expected_value], ignore_index=True), profile, volatility, profit_weight=70, drawdown_weight=30,
)
today_assignment_rows = live_assignment.loc[pd.to_datetime(live_assignment["decision_day"]) == latest_calendar_day]
print("today's assignment rows (should be > 0 now):", len(today_assignment_rows))
assert len(today_assignment_rows) > 0, "combining fit_oos_expected_value with predict_live_expected_value must give today's rows a book decision"

from trading_data.research import account_group_lookup, non_client_logins


class _FakeGroupClient:
    """Simulates a userinfo/accounts table: some logins unknown, some flagged."""

    def query(self, sql, **kwargs):
        return pd.DataFrame({
            "login": [101, 102, 103],
            "acct_group": ["RUc_00", "_TEST_bia", "Zs\\Test01\\Std\\Hedged"],
        })


group_lookup = account_group_lookup(_FakeGroupClient(), "mt4", [101, 102, 103, 999])
flagged = non_client_logins(group_lookup)
print("\naccount_group_lookup / non_client_logins:", flagged)
assert flagged == {102, 103}, "must flag only the test-group logins, not the real client or the unresolved one"
assert 999 not in flagged, "an unresolved login (no group found) must never be treated as excluded"

empty_lookup = account_group_lookup(_FakeGroupClient(), "mt4", [])
assert empty_lookup.empty, "an empty login list must short-circuit without querying"
assert non_client_logins(empty_lookup) == set(), "an empty lookup must flag nothing"

print("\nALL SMOKE TESTS PASSED")
