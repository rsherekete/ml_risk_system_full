"""TRADER-SELECTION vs per-trade E, like-for-like OOS.

Classify accounts on the TRAINING window (open_time < Aug 10) by three
trailing metrics:
  1. volume-normalised avg P&L  = mean(net_profit / adj_lots)       > 0
  2. win rate                                                       > 0.80
  3. markout velocity (4m-4h band) = mean over trades held 4m..4h of
     dir*(close-open)/open / hold_hours  (directional return per hour)  > 0
BEST = all three pass; WORST = the mirror (<0, <0.20, <0).

OOS replay (open_time >= Aug 10), identical mechanics in every arm (a35
entry-delay, mirror exit at client close, fixed 0.1 canonical lots, E cost
gate, cap 1000) -- arms differ ONLY in how direction/selection is chosen:
  E        : per-trade stance_score (copy>=0.8 / invert<=0.2)   [baseline]
  SELECT   : copy EVERY subsequent trade of a BEST trader; invert EVERY
             subsequent trade of a WORST trader.
Streamed interim prints.
"""
import sys, time, heapq
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np
import pandas as pd
from webapp.model_service import SCRATCH
from webapp.trade_feed import _canonical
from webapp.views import _contract_units

t0 = time.time()
# a35 spans Aug 10-31 only; classify on the first half, trade the second.
# Upstream stance/entry models were fit strictly pre-Aug-10, so the
# classification window carries no model leakage.
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
print(f"[{time.time()-t0:.0f}s] rows {len(ab):,} | "
      f"train {(ab['open_time'] < SPLIT).sum():,} | "
      f"oos {(ab['open_time'] >= SPLIT).sum():,}", flush=True)

# ---- classify on TRAIN window ----
tr = ab[ab["open_time"] < SPLIT].copy()
tr["_vnp"] = tr["net_profit"] / tr["adj_lots"]
tr["_win"] = (tr["net_profit"] > 0).astype(float)
g = tr.groupby("account_key", observed=True)
stat = g.agg(n=("net_profit", "size"), vol_norm_pnl=("_vnp", "mean"),
             win_rate=("_win", "mean"))
band = tr[(tr["hold_s"] >= 240) & (tr["hold_s"] <= 4 * 3600)].copy()
band["_mkv"] = band["dir_ret"] / band["hold_h"].clip(lower=1e-6)
stat["mk_velocity"] = band.groupby(
    "account_key", observed=True)["_mkv"].mean().reindex(stat.index)
stat = stat[stat["n"] >= 20]          # enough history to classify
best = stat.index[(stat["vol_norm_pnl"] > 0) & (stat["win_rate"] > 0.80)
                  & (stat["mk_velocity"] > 0)]
worst = stat.index[(stat["vol_norm_pnl"] < 0) & (stat["win_rate"] < 0.20)
                   & (stat["mk_velocity"] < 0)]
print(f"[{time.time()-t0:.0f}s] classifiable accounts {len(stat):,} | "
      f"BEST {len(best):,} | WORST {len(worst):,}", flush=True)

# ---- OOS book ----
oos = ab[ab["open_time"] >= SPLIT].sort_values("open_time").copy()

def pln(c, p):
    c = str(c)
    if c.startswith("XAU"): return 100.0 * p
    if c.startswith("XAG"): return 5000.0 * p
    if c in ("BTCUSD", "ETHUSD"): return 1.0 * p
    if len(c) == 6 and c.isalpha(): return 100_000.0
    return 1.0 * p
COMM = {"XAU": 6.0, "XAG": 6.0, "FX": 6.0, "IDX": 0.0}
HB = {"XAU": 0.26, "XAG": 1.0, "FX": 0.40, "IDX": 0.50}
def scl(c):
    c = str(c)
    if c.startswith("XAU"): return "XAU"
    if c.startswith("XAG"): return "XAG"
    if len(c) == 6 and c.isalpha(): return "FX"
    return "IDX"

def replay(sub):
    sub = sub.sort_values("open_time")
    open_e = sub["open_time"].astype("datetime64[s]").astype("int64").to_numpy()
    close_e = sub["close_time"].astype("datetime64[s]").astype("int64").to_numpy()
    entry = sub["open_price"].to_numpy(float)
    exitp = sub["close_price"].to_numpy(float)
    dirs = sub["our_dir"].to_numpy(float)
    pred = sub["pred_bps"].to_numpy(float)
    wlo = sub["win_low"].to_numpy(float); whi = sub["win_high"].to_numpy(float)
    canon = sub["canon"].astype(str).to_numpy()
    eq = peak = mdd = 0.0; heap = []
    filled = wins = missed = 0; gwin = gloss = 0.0
    for i in range(len(sub)):
        while heap and heap[0][0] <= open_e[i]:
            _, p_usd = heapq.heappop(heap)
            eq += p_usd; peak = max(peak, eq); mdd = max(mdd, peak - eq)
        if len(heap) >= 1000:
            continue
        c = canon[i]; k = scl(c); d = dirs[i]; en = entry[i]
        askb = 1.3 * pred[i]; is_mkt = True
        if askb >= 12.0:
            limit = en * (1.0 - d * askb / 1e4)
            ok = (wlo[i] <= limit) if d > 0 else (whi[i] >= limit)
            if not ok:
                missed += 1; continue
            en = limit; is_mkt = False
        notion = 0.1 * pln(c, en)
        cost = COMM[k] * 0.1 + (HB[k] / 1e4 * notion if is_mkt else 0.0)
        p_usd = d * (exitp[i] - en) / en * notion - cost
        filled += 1; wins += p_usd > 0
        if p_usd > 0: gwin += p_usd
        else: gloss += -p_usd
        heapq.heappush(heap, (close_e[i], p_usd))
    while heap:
        _, p_usd = heapq.heappop(heap)
        eq += p_usd; peak = max(peak, eq); mdd = max(mdd, peak - eq)
    return {"filled": filled, "win": wins / max(filled, 1), "net": eq,
            "mdd": mdd, "pf": gwin / max(gloss, 1e-9), "missed": missed}

def report(name, sub):
    if not len(sub):
        print(f"=== {name} === (no trades)", flush=True); return
    r = replay(sub)
    print(f"=== {name} === trades {len(sub):,} | filled {r['filled']:,} | "
          f"win {r['win']:.3f} | net ${r['net']:+,.0f} | "
          f"maxDD ${r['mdd']:,.0f} | PF {r['pf']:.2f} | "
          f"net/DD {r['net']/max(r['mdd'],1e-9):.0f}x", flush=True)

# --- E baseline on the SAME oos trades (per-trade stance) ---
st = np.where(oos["stance_score"] >= 0.8, 1,
              np.where(oos["stance_score"] <= 0.2, -1, 0))
e = oos[st != 0].copy()
e["our_dir"] = np.where(e["stance_score"] >= 0.8, 1, -1) * e["direction"]
e["dir_edge"] = np.where(e["stance_score"] >= 0.8, 2 * e["stance_score"] - 1,
                         1 - 2 * e["stance_score"])
n0 = 0.1 * np.array([pln(c, p) for c, p in zip(e["canon"], e["open_price"])])
e_cost = np.array([COMM[scl(c)] for c in e["canon"]]) * 0.1
e = e[e["dir_edge"] * e["perlot_E"] * 0.1 > e_cost]
report("E baseline (per-trade stance, 0.8/0.2)", e)

# --- SELECT: copy best, invert worst ---
bestset, worstset = set(best), set(worst)
sel = oos[oos["account_key"].isin(bestset | worstset)].copy()
is_best = sel["account_key"].isin(bestset)
sel["our_dir"] = np.where(is_best, sel["direction"], -sel["direction"])
report("SELECT copy-best + invert-worst (all their trades)", sel)
report("  SELECT best-only (copy)", sel[is_best.values])
report("  SELECT worst-only (invert)", sel[~is_best.values])
print(f"[{time.time()-t0:.0f}s] done", flush=True)
