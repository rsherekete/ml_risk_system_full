import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from webapp import model_service as ms, vantage

# 1) training-scored trades with outcomes (recent days, active accounts)
q = ms.load_scores(ms.VIEW_QUANT)
q = q[np.isfinite(q["score"]) & np.isfinite(q["pnl"])].copy()
q = q[pd.to_datetime(q["day"]) >= pd.Timestamp("2026-08-20")]
# focus on accounts with many trades so history/parity is well-defined
top = q["account_key"].value_counts()
top = top[top >= 40].index[:12]
q = q[q["account_key"].isin(top)]
parts = []
for acc, d in q.groupby("account_key"):
    parts.append(d.sample(min(40, len(d)), random_state=1))
samp = pd.concat(parts, ignore_index=True)
print("sampled", len(samp), "trades across", samp["account_key"].nunique(), "accounts")

hist = vantage.account_history()
print("account_history rows:", len(hist), "| has sampled accts:",
      sum(a in hist.index for a in top), "/", len(top))

rows = []
for t in samp.itertuples():
    try:
        direction = int(t.direction)
        live = vantage.score(str(t.symbol), direction, float(t.volume_lots),
                             float(t.open_price), str(t.account_key), hist, capture=True)
    except Exception as e:
        live = None
    if isinstance(live, tuple):
        prob, feats = live
    else:
        prob, feats = (live, {})
    rows.append({"account_key": t.account_key, "symbol": t.symbol,
                 "train_score": float(t.score), "live_score": prob,
                 "pnl": float(t.pnl), "win": int(t.pnl > 0),
                 "live_hwr": feats.get("hist_win_rate"),
                 "live_ctx1h": feats.get("ctx_return_1h"),
                 "live_symcode": feats.get("symbol_code")})

r = pd.DataFrame(rows)
r_ok = r[r["live_score"].notna()].copy()
print("\nlive scored:", len(r_ok), "/", len(r))
if len(r_ok):
    r_ok["diff"] = (r_ok["live_score"] - r_ok["train_score"]).abs()
    print("mean|live-train| score diff: %.3f | median: %.3f" % (r_ok["diff"].mean(), r_ok["diff"].median()))
    print("corr(live,train): %.3f" % r_ok[["live_score","train_score"]].corr().iloc[0,1])
    print("\nTRAIN calibration on this sample (score bucket -> winrate):")
    for name, col in (("train","train_score"),("live","live_score")):
        r_ok["b"] = (r_ok[col]*5).clip(0,4).astype(int)
        g = r_ok.groupby("b").agg(n=("win","size"), wr=("win","mean"), mscore=(col,"mean"))
        print(" ", name, dict(zip(g.index, [f"n{int(row.n)}/wr{row.wr:.2f}" for row in g.itertuples()])))
    print("\nlive hist_win_rate NaN frac:", r_ok["live_hwr"].isna().mean(),
          "| ctx_return_1h NaN frac:", r_ok["live_ctx1h"].isna().mean(),
          "| symbol_code NaN frac:", r_ok["live_symcode"].isna().mean())
    print("\nsample rows:")
    print(r_ok[["account_key","symbol","train_score","live_score","win","live_hwr","live_ctx1h"]].head(15).to_string())
