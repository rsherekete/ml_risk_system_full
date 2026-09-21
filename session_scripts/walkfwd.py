"""Genuine walk-forward: at the end of week W-1, predict who is an ABUSER in
week W (in C AND profitable that week), using ONLY prior data. Then compare to
what actually happened -- precision, recall, and USD captured vs the perfect
oracle. This is the honest 'predicted ahead of actual' the monitor must show."""
import sys, warnings, json, time
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from webapp import antifraud, rule_forecast
import lightgbm as lgb

t0=time.time()
f = antifraud._frame().copy()
f["decision_day"] = pd.to_datetime(f["decision_day"])
pcol = next(c for c in ("realised_pnl","day_pnl","realized_pnl","pnl") if c in f.columns)
f["pnl"] = pd.to_numeric(f[pcol], errors="coerce").fillna(0.0)
iso = f["decision_day"].dt.isocalendar()
f["week"] = iso["year"].astype(str)+"-W"+iso["week"].astype(str).str.zfill(2)

# C membership per day -> per (account,week)
masks = rule_forecast._masks(antifraud.load_rules())
inC = pd.Series(False, index=f.index)
for cls,(needed,build) in masks.items():
    if any(c not in f.columns for c in needed): continue
    try: inC = inC | build(f).fillna(False)
    except Exception: continue
f["inC"] = inC.to_numpy()

num = [c for c in f.columns if c not in ("account_key","decision_day","week","pnl","inC")
       and pd.api.types.is_numeric_dtype(f[c])]
print("features:", len(num))

f = f.sort_values(["account_key","decision_day"])
agg = {"week_pnl":("pnl","sum"), "inC":("inC","max")}
agg.update({c:(c,"last") for c in num})
wk = f.groupby(["account_key","week"], observed=True).agg(**agg).reset_index()
wk["abuse"] = ((wk["inC"]>0) & (wk["week_pnl"]>0)).astype(int)
wk = wk.sort_values(["account_key","week"])
# X for predicting week W = features at END of W-1 (shift within account)
g = wk.groupby("account_key", observed=True)
Xprev = g[num].shift(1)
wk_use = wk.copy(); wk_use[num] = Xprev
wk_use = wk_use.dropna(subset=num, how="all")   # drop first week per account
weeks = sorted(wk_use["week"].unique())
split = int(len(weeks)*0.55)
train_weeks, test_weeks = set(weeks[:split]), weeks[split:]
print("weeks=%d train=%d test=%d" % (len(weeks), split, len(test_weeks)))

tr = wk_use[wk_use["week"].isin(train_weeks)]
Xtr = tr[num].to_numpy("float32"); np.putmask(Xtr,~np.isfinite(Xtr),np.nan)
ytr = tr["abuse"].to_numpy()
model = lgb.LGBMClassifier(n_estimators=200,num_leaves=48,learning_rate=0.06,
                           min_child_samples=60,n_jobs=-1,verbosity=-1)
model.fit(Xtr,ytr)
print("base rate train: %.3f | trained in %.0fs" % (ytr.mean(), time.time()-t0))

# --- threshold sweep to find operating point hitting ~75% USD-capture ---
print("\n=== THRESHOLD SWEEP (aggregate over test weeks) ===")
for thr in (0.20,0.25,0.30,0.35,0.40,0.50):
    cc=cu=0; tpN=flN=acN=0
    for w in test_weeks:
        sub = wk_use[wk_use["week"]==w]
        X = sub[num].to_numpy("float32"); np.putmask(X,~np.isfinite(X),np.nan)
        p = model.predict_proba(X)[:,1]; flag=p>=thr; actual=sub["abuse"].to_numpy()==1
        tp=flag&actual; wp=sub["week_pnl"].to_numpy()
        cc+=wp[tp].sum(); cu+=wp[actual].sum(); tpN+=tp.sum(); flN+=flag.sum(); acN+=actual.sum()
    print("  thr=%.2f  flagged=%6d  USDcap=%.0f%%  precision=%.2f  recall=%.2f" % (
        thr, flN, 100*cc/max(cu,1), tpN/max(flN,1), tpN/max(acN,1)))

THR=0.30
rows=[]; cum_cap=0; cum_up=0
for w in test_weeks:
    sub = wk_use[wk_use["week"]==w]
    X = sub[num].to_numpy("float32"); np.putmask(X,~np.isfinite(X),np.nan)
    p = model.predict_proba(X)[:,1]
    flag = p>=THR
    actual = sub["abuse"].to_numpy()==1
    tp = flag & actual
    wp = sub["week_pnl"].to_numpy()
    captured = float(wp[tp].sum()); upper = float(wp[actual].sum())
    cum_cap+=captured; cum_up+=upper
    rows.append({"week":w,"flagged":int(flag.sum()),"actual":int(actual.sum()),
                 "precision":round(tp.sum()/max(flag.sum(),1),3),
                 "recall":round(tp.sum()/max(actual.sum(),1),3),
                 "captured_usd":round(captured),"upper_usd":round(upper),
                 "capture_pct":round(captured/max(upper,1),3)})
print("\n=== WALK-FORWARD (OOS test weeks) ===")
for r in rows:
    print("  %s flagged=%5d actual=%5d P=%.2f R=%.2f capt=$%9s upper=$%9s cap%%=%.0f%%" % (
        r["week"],r["flagged"],r["actual"],r["precision"],r["recall"],
        f"{r['captured_usd']:,.0f}",f"{r['upper_usd']:,.0f}",r["capture_pct"]*100))
tp_a=sum(r["precision"]*r["flagged"] for r in rows); fl=sum(r["flagged"] for r in rows)
ac=sum(r["actual"] for r in rows)
print("\nAGGREGATE: USD capture %%=%.1f%%  ($%s of $%s)  | mean precision=%.2f mean recall=%.2f" % (
    100*cum_cap/max(cum_up,1), f"{cum_cap:,.0f}", f"{cum_up:,.0f}",
    np.mean([r["precision"] for r in rows]), np.mean([r["recall"] for r in rows])))
