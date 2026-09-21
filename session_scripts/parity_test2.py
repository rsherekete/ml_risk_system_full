import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from webapp import model_service as ms, vantage

print("seeding bars (engine's live context)...")
vantage._seed_bars()
nbars = {k: len(v) for k, v in vantage._BAR_HISTORY.items()}
print("bar series:", len(nbars), "| XAUUSD bars:", nbars.get("XAUUSD"),
      "| top:", sorted(nbars.items(), key=lambda x: -x[1])[:5])

q = ms.load_scores(ms.VIEW_QUANT)
q = q[np.isfinite(q["score"]) & np.isfinite(q["pnl"])].copy()
# most-recent training days = closest to 'live' state
q = q[pd.to_datetime(q["day"]) >= pd.Timestamp("2026-08-29")]
parts = []
for acc, d in q.groupby("account_key"):
    if len(d) >= 15:
        parts.append(d.sample(min(30, len(d)), random_state=3))
samp = pd.concat(parts, ignore_index=True)
print("sampled", len(samp), "recent trades across", samp["account_key"].nunique(), "accounts\n")

hist = vantage.account_history()
rows = []
for t in samp.itertuples():
    try:
        out = vantage.score(str(t.symbol), int(t.direction), float(t.volume_lots),
                            float(t.open_price), str(t.account_key), hist, capture=True)
    except Exception:
        out = None
    prob, feats = out if isinstance(out, tuple) else (out, {})
    rows.append({"train_score": float(t.score), "live_score": prob,
                 "win": int(t.pnl > 0), "symbol": str(t.symbol),
                 "ctx1h": feats.get("ctx_return_1h"), "hwr": feats.get("hist_win_rate")})
r = pd.DataFrame(rows)
r = r[r["live_score"].notna()].copy()
print("live scored:", len(r))
print("ctx_return_1h NaN frac now:", round(r["ctx1h"].isna().mean(), 3))
print("mean|live-train|: %.3f | corr: %.3f" % ((r.live_score-r.train_score).abs().mean(),
      r[["live_score","train_score"]].corr().iloc[0,1]))
print("live_score distribution: min %.2f p10 %.2f median %.2f p90 %.2f max %.2f std %.3f" % (
      r.live_score.min(), r.live_score.quantile(.1), r.live_score.median(),
      r.live_score.quantile(.9), r.live_score.max(), r.live_score.std()))
print("\nCALIBRATION vs CLIENT outcome (does a high live score predict client win?):")
for name, col in (("TRAIN", "train_score"), ("LIVE", "live_score")):
    r["b"] = pd.qcut(r[col].rank(method="first"), 5, labels=False)
    g = r.groupby("b").agg(n=("win", "size"), wr=("win", "mean"), s=(col, "mean"))
    print(" ", name, "  ".join(f"q{int(i)}:wr{row.wr:.2f}(s{row.s:.2f})" for i, row in g.iterrows()))
