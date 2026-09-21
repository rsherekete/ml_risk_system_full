"""Why trader-selection fails: persistence diagnostic + ML ranking arm.
  (A) correlation of train-window trader quality with OOS quality;
  (B) ML: gradient-boost trailing account stats -> forward mean dir-return,
      rank accounts, copy top decile / invert bottom decile, compare to E."""
import sys, time, heapq
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np
import pandas as pd
from webapp.model_service import SCRATCH
from webapp.trade_feed import _canonical
from webapp.views import _contract_units

t0 = time.time()
SPLIT = pd.Timestamp("2026-08-20")
ab = pd.read_parquet(SCRATCH / "entry_ab_test_a35.parquet").set_index("row_id")
p2 = pd.read_parquet(SCRATCH / "persec2_preds.parquet").set_index("row_id")
ab = ab.join(p2[["perlot_E", "hold_s"]], how="inner")
ab["open_time"] = pd.to_datetime(ab["open_time"])
sy = ab["symbol"].astype(str)
scale = {s: _contract_units(s) / max(_contract_units(_canonical(s)), 1e-9)
         for s in sy.unique()}
ab["adj_lots"] = (pd.to_numeric(ab["volume_lots"], errors="coerce")
                  * sy.map(scale)).clip(lower=0.0001)
ab["hold_h"] = ab["hold_s"] / 3600.0
ab["dir_ret"] = ab["direction"] * (ab["close_price"] - ab["open_price"]) \
    / ab["open_price"]
ab["vnp"] = ab["net_profit"] / ab["adj_lots"]

def acct_stats(df):
    df = df.copy()
    df["_win"] = (df["net_profit"] > 0).astype(float)
    df["_mkv"] = (df["dir_ret"] / df["hold_h"].clip(lower=1e-6)) \
        .where((df["hold_s"] >= 240) & (df["hold_s"] <= 4 * 3600))
    g = df.groupby("account_key", observed=True)
    return g.agg(n=("net_profit", "size"), vol_norm_pnl=("vnp", "mean"),
                 win_rate=("_win", "mean"), mk_velocity=("_mkv", "mean"),
                 mean_dir_ret=("dir_ret", "mean"),
                 med_hold=("hold_s", "median"), mean_lots=("adj_lots", "mean"))

tr = acct_stats(ab[ab["open_time"] < SPLIT])
oo = acct_stats(ab[ab["open_time"] >= SPLIT])
tr = tr[tr["n"] >= 20]
both = tr.join(oo[["vol_norm_pnl", "win_rate", "mean_dir_ret"]],
               rsuffix="_oos", how="inner")
print(f"[{time.time()-t0:.0f}s] accounts with train>=20 & oos trades: "
      f"{len(both):,}", flush=True)
print("PERSISTENCE (train -> OOS, Spearman):")
for a, b in (("vol_norm_pnl", "vol_norm_pnl_oos"),
             ("win_rate", "win_rate_oos"),
             ("mean_dir_ret", "mean_dir_ret_oos")):
    rho = both[a].rank().corr(both[b].rank())
    print(f"  {a:14s} -> OOS: rho {rho:+.3f}", flush=True)

# ---- ML arm: learn forward mean_dir_ret from trailing stats ----
import lightgbm as lgb
feats = ["n", "vol_norm_pnl", "win_rate", "mk_velocity", "mean_dir_ret",
         "med_hold", "mean_lots"]
d = both.dropna(subset=["mean_dir_ret_oos"])
# split accounts for honest eval: fit on a random half, rank the other half
rng = np.random.default_rng(0)
mask = rng.random(len(d)) < 0.5
m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                      min_child_samples=40, random_state=0, verbosity=-1)
m.fit(d.loc[mask, feats].fillna(0), d.loc[mask, "mean_dir_ret_oos"])
pred = pd.Series(m.predict(d.loc[~mask, feats].fillna(0)),
                 index=d.index[~mask])
actual = d.loc[~mask, "mean_dir_ret_oos"]
ic = pred.rank().corr(actual.rank())
print(f"[{time.time()-t0:.0f}s] ML forward-return IC (held-out accounts): "
      f"{ic:+.3f}", flush=True)
top = pred.sort_values(ascending=False)
n_dec = max(1, len(top) // 10)
best_ml = set(top.head(n_dec).index)
worst_ml = set(top.tail(n_dec).index)

# ---- replay the ML top/bottom decile on OOS trades ----
oos = ab[ab["open_time"] >= SPLIT].sort_values("open_time").copy()
def pln(c, p):
    c = str(c)
    if c.startswith("XAU"): return 100.0 * p
    if c.startswith("XAG"): return 5000.0 * p
    if c in ("BTCUSD", "ETHUSD"): return 1.0 * p
    if len(c) == 6 and c.isalpha(): return 100_000.0
    return 1.0 * p
COMM = {"XAU": 6, "XAG": 6, "FX": 6, "IDX": 0}
HB = {"XAU": 0.26, "XAG": 1.0, "FX": 0.40, "IDX": 0.50}
def scl(c):
    c = str(c)
    if c.startswith("XAU"): return "XAU"
    if c.startswith("XAG"): return "XAG"
    if len(c) == 6 and c.isalpha(): return "FX"
    return "IDX"
def replay(sub):
    sub = sub.sort_values("open_time")
    oe = sub["open_time"].astype("datetime64[s]").astype("int64").to_numpy()
    ce = sub["close_time"].astype("datetime64[s]").astype("int64").to_numpy()
    en = sub["open_price"].to_numpy(float); ex = sub["close_price"].to_numpy(float)
    d = sub["our_dir"].to_numpy(float); pr = sub["pred_bps"].to_numpy(float)
    wl = sub["win_low"].to_numpy(float); wh = sub["win_high"].to_numpy(float)
    cn = sub["canon"].astype(str).to_numpy()
    eq = peak = mdd = 0.0; heap = []; fl = wn = 0; gw = gl = 0.0
    for i in range(len(sub)):
        while heap and heap[0][0] <= oe[i]:
            _, pu = heapq.heappop(heap); eq += pu
            peak = max(peak, eq); mdd = max(mdd, peak - eq)
        if len(heap) >= 1000: continue
        c = cn[i]; k = scl(c); e = en[i]; mk = True
        ask = 1.3 * pr[i]
        if ask >= 12:
            lim = e * (1 - d[i] * ask / 1e4)
            ok = (wl[i] <= lim) if d[i] > 0 else (wh[i] >= lim)
            if not ok: continue
            e = lim; mk = False
        nt = 0.1 * pln(c, e); cost = COMM[k] * 0.1 + (HB[k]/1e4*nt if mk else 0)
        pu = d[i] * (ex[i] - e) / e * nt - cost
        fl += 1; wn += pu > 0
        if pu > 0: gw += pu
        else: gl += -pu
        heapq.heappush(heap, (ce[i], pu))
    while heap:
        _, pu = heapq.heappop(heap); eq += pu
        peak = max(peak, eq); mdd = max(mdd, peak - eq)
    return {"fl": fl, "win": wn/max(fl,1), "net": eq, "mdd": mdd,
            "pf": gw/max(gl,1e-9)}
sel = oos[oos["account_key"].isin(best_ml | worst_ml)].copy()
isb = sel["account_key"].isin(best_ml)
sel["our_dir"] = np.where(isb, sel["direction"], -sel["direction"])
r = replay(sel)
print(f"=== ML top/bottom-decile select === trades {len(sel):,} | "
      f"filled {r['fl']:,} | win {r['win']:.3f} | net ${r['net']:+,.0f} | "
      f"maxDD ${r['mdd']:,.0f} | PF {r['pf']:.2f} | "
      f"net/DD {r['net']/max(r['mdd'],1e-9):.0f}x", flush=True)
print(f"[{time.time()-t0:.0f}s] done", flush=True)
