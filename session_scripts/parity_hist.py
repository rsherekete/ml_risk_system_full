import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from webapp import model_service as ms
from webapp import trade_features as tf

RAW = ["database","account_key","symbol","cmd","volume_lots","open_time",
       "close_time","open_price","close_price","sl","tp","net_profit","state","reason"]
cache = ms.SCRATCH / "quant_feature_cache.parquet"
df = pd.read_parquet(cache)
print("cache rows:", len(df))

# ctx-INDEPENDENT history features (depend only on the account's own P&L series)
HIST = ["hist_win_rate","hist_mean_pnl","hist_pnl_std","hist_mean_notional",
        "hist_recent_pnl5","hist_wr_20p","hist_wr_10p","hist_wr_5p","hist_wr_2p",
        "hist_mp_20p","hist_mp_10p","hist_mp_5p","hist_mp_2p","trade_index",
        "acct_pnl_5d","acct_pnl_20d","acct_pnl_60d","acct_winrate_20d",
        "acct_trades_20d","acct_dd_20d","acct_tenure_days"]
HIST = [h for h in HIST if h in df.columns]

# pick 30 accounts with a decent number of trades
counts = df["account_key"].value_counts()
picks = counts[(counts >= 30) & (counts <= 400)].head(30).index.tolist()
sub = df[df["account_key"].isin(picks)].copy()
print(f"validation: {len(picks)} accounts, {len(sub)} trades")

# rebuild features from ONLY these accounts' full history via the SAME function
raw_in = sub[RAW].copy()
rebuilt = tf.build_trade_features(raw_in)

# align rebuilt to original by (account_key, open_time)
key = ["account_key","open_time"]
a = sub.set_index(key).sort_index()
b = rebuilt.set_index(key).sort_index()
common = a.index.intersection(b.index)
a, b = a.loc[common], b.loc[common]
print(f"aligned rows: {len(common)}")

print("\nfeature | max_abs_diff | mean_abs_diff | matches(<1e-6)")
worst = []
for f in HIST:
    av = pd.to_numeric(a[f], errors="coerce").to_numpy()
    bv = pd.to_numeric(b[f], errors="coerce").to_numpy()
    both_nan = np.isnan(av) & np.isnan(bv)
    d = np.abs(av - bv)
    d[both_nan] = 0.0
    mad = np.nanmax(d) if len(d) else 0.0
    matches = np.mean((d < 1e-6) | both_nan)
    worst.append((f, mad, np.nanmean(d), matches))
    print(f"  {f:20s} {mad:14.6g} {np.nanmean(d):12.6g}  {matches:.1%}")
allmatch = all(w[3] > 0.999 for w in worst)
print("\nPARITY (history family reproduced by build_trade_features):", "YES" if allmatch else "NO -- see diffs")
