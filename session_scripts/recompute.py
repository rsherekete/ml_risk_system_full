import sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from pathlib import Path
from webapp import trade_feed as tfeed
from webapp import trade_features as tf
import lightgbm as lgb

ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")
feats = (ART / "quant_model_features.txt").read_text(encoding="utf-8").splitlines()
model = lgb.Booster(model_file=str(ART / "quant_model.txt"))

Q = """
    SELECT `order`, login, symbol_name AS symbol, cmd, volume,
           open_price, open_ts, close_price, close_ts, profit
    FROM orders
    WHERE close_ts > %s AND close_ts > 0 AND cmd IN (0,1) AND open_price > 0
    ORDER BY close_ts DESC LIMIT 8000
"""
since = int(time.time() - 4*24*3600)   # last 4 days
rows = []
for server in ("mt4_live01","mt4_live02","mt4_live03","mt4_live04"):
    try:
        con = tfeed._connection(server)
        with con.cursor() as cur:
            cur.execute(Q, [since])
            data = cur.fetchall()
        cents = tfeed.cent_logins(server)
        for (order, login, symbol, cmd, vol, op, ots, cp, cts, profit) in data:
            scale = 100.0 if int(login) in cents else 1.0
            lot = float(vol) / 10000.0 / (100.0 if int(login) in cents else 1.0)
            rows.append({
                "database": server, "account_key": f"{server}:{login}",
                "symbol": str(symbol), "cmd": "buy" if int(cmd)==0 else "sell",
                "volume_lots": lot, "open_time": pd.to_datetime(int(ots), unit="s"),
                "close_time": pd.to_datetime(int(cts), unit="s"),
                "open_price": float(op), "close_price": float(cp),
                "sl": np.nan, "tp": np.nan,
                "net_profit": float(profit) / scale, "state": "closed", "reason": None})
        print(f"{server}: {len(data)} closed trades")
    except Exception as e:
        print(f"{server}: ERR {type(e).__name__}: {e}")

if not rows:
    print("NO DATA (MySQL unreachable offline?)"); sys.exit()
df = pd.DataFrame(rows)
print(f"\nTOTAL recent closed trades: {len(df):,}  "
      f"range {df['open_time'].min()} -> {df['close_time'].max()}")

built = tf.build_trade_features(df)
X = built[feats].to_numpy("float32"); np.putmask(X, ~np.isfinite(X), np.nan)
p = model.predict(X)
y = (built["net_profit"] > 0).to_numpy()
print(f"\nOUT-OF-SAMPLE recompute ({len(y):,} trades)  base client-win {y.mean():.1%}")
for thr in (0.85, 0.90):
    m = p >= thr
    print(f"  COPY  score>={thr}: n={m.sum():>6,}  client-win={y[m].mean():.1%}  (want ~95%)")
for thr in (0.25, 0.10):
    m = p <= thr
    print(f"  INVERT score<={thr}: n={m.sum():>6,}  client-win={y[m].mean():.1%}  (want LOW)")
# gold specifically
g = built["symbol"].astype(str).str.upper().str.startswith("XAU").to_numpy()
if g.sum():
    pg, yg = p[g], y[g]
    print(f"\n  GOLD only ({g.sum():,}): base-win {yg.mean():.1%}")
    for thr in (0.85, 0.90):
        m = pg >= thr
        if m.sum(): print(f"    copy>={thr}: n={m.sum():,} client-win={yg[m].mean():.1%}")
