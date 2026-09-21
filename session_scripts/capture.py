import sys, sqlite3
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from pathlib import Path
from webapp import vantage as V
from webapp import model_service as ms

# warm the pieces score() reads
print("warming...")
V.booster(); V._seed_bars(); V._symbol_codes()
V._account_day_snapshot(); V._funding_snapshot()
hist = V.account_history()
print("bars symbols:", len(V._BAR_HISTORY), "| gold bar series:",
      {s: len(V._BAR_HISTORY.get(s, [])) for s in ("XAUUSD+","XAUUSD","XAUUSDmin") if s in V._BAR_HISTORY})

# a real gold-trading account + a plausible live gold price/direction
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
row = cx.execute("SELECT source_account, our_direction, fill_price FROM vantage_orders "
                 "WHERE symbol='XAUUSD+' AND source_account IS NOT NULL "
                 "ORDER BY id DESC LIMIT 1").fetchone()
acct = row[0]; price = float(row[2] or 4470.0)
print(f"probe account {acct} price {price}")

prob, vals = V.score("XAUUSD+", 1, 0.1, price, acct, hist, capture=True)
print(f"\nLIVE score for XAUUSD+ copy: {prob:.3f}")

# training GOLD distribution (symbol_code aside -- compare the shape)
df = pd.read_parquet(ms.SCRATCH / "quant_feature_cache.parquet")
gold = df[df["symbol"].astype(str).str.upper().str.startswith("XAU")]
print(f"training gold trades: {len(gold):,}")
feats = list(V.trade_features_mod.TRADE_FEATURES) if hasattr(V,'trade_features_mod') else None
from webapp import trade_features as tf
feats = list(tf.TRADE_FEATURES)

print("\nfeatures where LIVE value is far from the training-gold median (|robust z|>4) or NaN-mismatch:")
flagged = []
for f in feats:
    lv = vals.get(f, np.nan)
    col = pd.to_numeric(gold[f], errors="coerce")
    med = col.median(); iqr = (col.quantile(.75) - col.quantile(.25)) or 1e-9
    tnan = col.isna().mean()
    if pd.isna(lv):
        if tnan < 0.5:  # live NaN but training usually populated
            flagged.append((f, "LIVE_NaN", lv, med, tnan))
        continue
    z = abs(lv - med) / (iqr if iqr else 1e-9)
    if z > 4:
        flagged.append((f, f"z={z:.1f}", lv, med, tnan))
for f, tag, lv, med, tnan in flagged:
    print(f"  {f:26s} {tag:10s} live={lv!s:>12.10}  train_med={med!s:>12.10}  train_nan={tnan:.0%}")
print(f"\n{len(flagged)} features flagged out of {len(feats)}")
