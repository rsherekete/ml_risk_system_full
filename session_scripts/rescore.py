"""Recompute the scores the LIVE engine assigned in the last 8h and compare.

For each order-store row (live_score recorded at decision time) rebuild the
feature row with the SHARED training pipeline: the account's closed-trade tape
(training cache, cut to trades CLOSED before the decision moment) + the open
trade, through build_trade_features. Score with the same model. Then diff.
"""
import sys, sqlite3, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from pathlib import Path
import duckdb
import lightgbm as lgb
from webapp import model_service as ms
from webapp import trade_features as tf

ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")
feats = (ART / "quant_model_features.txt").read_text(encoding="utf-8").splitlines()
model = lgb.Booster(model_file=str(ART / "quant_model.txt"))
CACHE = str(ms.SCRATCH / "quant_feature_cache.parquet")
RAWCOLS = ", ".join(tf.SCORING_RAW_COLUMNS)

cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
cx.row_factory = sqlite3.Row
win = time.time() - 8 * 3600
orders = [dict(r) for r in cx.execute(
    "SELECT created, source_account, symbol, client_direction, client_lots, "
    "live_score, stance, fill_price FROM vantage_orders "
    "WHERE created >= ? AND live_score IS NOT NULL AND source_account IS NOT NULL "
    "AND stance IN ('copy','invert') ORDER BY created DESC LIMIT 400", (win,))]
print(f"orders with live_score in last 8h: {len(orders)}")

con = duckdb.connect()
rows = []
done = 0
for o in orders:
    acct = str(o["source_account"])
    when = pd.Timestamp(o["created"], unit="s")
    try:
        tape = con.execute(
            f"SELECT {RAWCOLS} FROM read_parquet(?) WHERE account_key = ? "
            f"AND close_time < ? ORDER BY open_time", [CACHE, acct, str(when)]).df()
    except Exception:
        continue
    if not len(tape):
        continue
    new = pd.DataFrame([{
        "database": acct.split(":")[0], "account_key": acct,
        "symbol": str(o["symbol"]),
        "cmd": "buy" if int(o["client_direction"] or 0) > 0 else "sell",
        "volume_lots": float(o["client_lots"] or 0.01),
        "open_time": when, "close_time": pd.NaT,
        "open_price": float(o["fill_price"] or 0) or np.nan,
        "close_price": np.nan, "sl": np.nan, "tp": np.nan,
        "net_profit": np.nan, "state": "open", "reason": None}])
    try:
        built = tf.build_features_for_scoring(new, tape)
    except Exception as e:
        continue
    if not len(built):
        continue
    X = built[feats].to_numpy("float64")
    np.putmask(X, ~np.isfinite(X), np.nan)
    p = float(model.predict(X)[0])
    rows.append({"acct": acct, "symbol": o["symbol"], "stance": o["stance"],
                 "live": float(o["live_score"]), "recomputed": p,
                 "tape_n": len(tape)})
    done += 1
    if done >= 150:
        break

df = pd.DataFrame(rows)
print(f"recomputed: {len(df)}")
if len(df):
    df["diff"] = df["live"] - df["recomputed"]
    print(f"\nlive vs recomputed score:")
    print(f"  corr        : {df['live'].corr(df['recomputed']):.3f}")
    print(f"  mean diff   : {df['diff'].mean():+.4f}  (positive = live INFLATED)")
    print(f"  median diff : {df['diff'].median():+.4f}")
    print(f"  |diff|>0.10 : {(df['diff'].abs() > 0.10).mean():.1%} of trades")
    print("\nby stance:")
    for st, g in df.groupby("stance"):
        # would the RECOMPUTED score still clear the anchor the live one did?
        still = ((g["recomputed"] >= 0.85).mean() if st == "copy"
                 else (g["recomputed"] <= 0.157).mean())
        print(f"  {st:7s} n={len(g):3d}  live_med={g['live'].median():.3f}  "
              f"recomp_med={g['recomputed'].median():.3f}  "
              f"diff_med={g['diff'].median():+.3f}  still-clears-anchor={still:.0%}")
    print("\nworst 12 inflations (live >> recomputed):")
    for _, r in df.nlargest(12, "diff").iterrows():
        print(f"  {r['symbol']:10s} {r['stance']:6s} live={r['live']:.3f} "
              f"recomp={r['recomputed']:.3f}  tape={r['tape_n']}")
