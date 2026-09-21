"""Tape model: a short-horizon directional signal per instrument from the
client tape, read THROUGH the per-trade model.

The raw tape (who bought, who sold, how much) carries no 15-minute
information (OOS 51%, 14 Sep 2026). What does is the engine's own view of
each trade: the calibrated P(win) and the copy / invert stance it implies.
Rolling sums of those over 5-120 minutes, plus a little price context, give
XAUUSD ~70% directional accuracy on every minute and ~90% on the most
confident decile, 15 minutes ahead.

PARITY BY CONSTRUCTION. Training and live inference call the SAME functions
on the SAME definition of "now":
  * a decision at minute T uses events from COMPLETED minutes [T-w, T-1]
    (never the forming minute T) -- `minute_sums` + `sums_at`;
  * the same per-event terms (`event_terms`, `close_terms`);
  * the same vector builder (`features_from_sums`, vectorised);
  * price context from COMPLETED bars up to T-1 (`price_context` for live,
    the identical formulas in `build_frame` for the grid; `parity_check`
    proves they agree on real data before an artifact is saved);
  * the target is the mid move from minute T to T+h (mid = (high+low)/2).

Artifacts per symbol: `tape_<SYMBOL>_<h>m.txt` (LightGBM booster) and
`tape_<SYMBOL>_<h>m.json` (feature names, confidence threshold = the median
|p-0.5| of the walk-forward OOS predictions, OOS metrics, replay, parity).
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = (5, 15, 30, 60, 120)
PRICE_WINDOWS = (5, 15, 30, 60)
HORIZON = 15
COST_BPS = 1.5
EVENT_COLS = ("o_n", "o_net", "o_net_lots", "o_net_conf", "o_net_conf_lots",
              "o_stance_lots", "o_stance_n", "o_hi_n", "o_lots",
              "c_n", "c_net", "c_net_lots", "c_net_conf", "c_stance_lots")
COPY_FLOOR, INVERT_CEILING = 0.85, 0.16      # the engine's tested anchors
MINUTE = np.timedelta64(1, "m")


def feature_names() -> list[str]:
    names: list[str] = []
    for w in WINDOWS:
        names += [f"{c}_{w}" for c in EVENT_COLS]
        names += [f"o_ratio_{w}", f"o_conf_ratio_{w}", f"flow_{w}", f"stance_flow_{w}"]
    names.append("activity_15_vs_120")
    for w in PRICE_WINDOWS:
        names += [f"ret_{w}", f"range_{w}"]
    names += ["pos_in_60", "hour", "weekday"]
    return names


def artifact_paths(artifacts: Path, symbol: str, horizon: int = HORIZON) -> tuple[Path, Path]:
    return (Path(artifacts) / f"tape_{symbol}_{horizon}m.txt",
            Path(artifacts) / f"tape_{symbol}_{horizon}m.json")


# ------------------------------------------------------------ per-event terms
def _stance(direction, score):
    return np.where(score >= COPY_FLOOR, direction, np.where(score <= INVERT_CEILING, -direction, 0.0))


def event_terms(direction, lots, score) -> dict:
    """Contributions of an OPEN to the rolling sums (scalars or arrays)."""
    direction = np.asarray(direction, dtype=float); lots = np.asarray(lots, dtype=float)
    score = np.asarray(score, dtype=float)
    conf = (score - 0.5) * 2.0
    stance = _stance(direction, score)
    return {"o_n": np.ones_like(direction), "o_net": direction, "o_net_lots": direction * lots,
            "o_net_conf": direction * conf, "o_net_conf_lots": direction * conf * lots,
            "o_stance_lots": stance * lots, "o_stance_n": stance,
            "o_hi_n": (score >= COPY_FLOOR).astype(float), "o_lots": lots}


def close_terms(direction, lots, score) -> dict:
    """A CLOSE of a buy is sell pressure: the open's terms, sign flipped, at
    the open's score."""
    direction = np.asarray(direction, dtype=float); lots = np.asarray(lots, dtype=float)
    score = np.asarray(score, dtype=float)
    conf = (score - 0.5) * 2.0
    stance = _stance(direction, score)
    return {"c_n": np.ones_like(direction), "c_net": -direction, "c_net_lots": -direction * lots,
            "c_net_conf": -direction * conf, "c_stance_lots": -stance * lots}


# ------------------------------------------------------------ windows
def minute_sums(minutes, terms: dict) -> tuple[np.ndarray, np.ndarray]:
    """Group per-event terms by their (floored) minute and return the sorted
    unique minutes plus the CUMULATIVE sums (n+1 x k, EVENT_COLS order, a
    leading zero row) that `sums_at` slices. Missing columns are zero."""
    minutes = np.asarray(pd.to_datetime(np.asarray(minutes)).values.astype("datetime64[m]"))
    k = len(EVENT_COLS)
    if len(minutes) == 0:
        return np.array([], dtype="datetime64[m]"), np.zeros((1, k))
    order = np.argsort(minutes, kind="stable")
    minutes = minutes[order]
    uniq, start = np.unique(minutes, return_index=True)
    mat = np.zeros((len(uniq), k))
    for j, col in enumerate(EVENT_COLS):
        vals = terms.get(col)
        if vals is None:
            continue
        vals = np.asarray(vals, dtype=float)[order]
        mat[:, j] = np.add.reduceat(vals, start)
    cums = np.vstack([np.zeros((1, k)), np.cumsum(mat, axis=0)])
    return uniq, cums


def sums_at(minutes: np.ndarray, cums: np.ndarray, at, w: int) -> np.ndarray:
    """EVENT_COLS sums over the COMPLETED minutes [T-w, T-1] for each decision
    minute T in `at` (datetime64, floored to the minute). Shape (len(at), k)."""
    at = np.asarray(pd.to_datetime(np.asarray(at)).values.astype("datetime64[m]"))
    if len(minutes) == 0:
        return np.zeros((len(at), cums.shape[1]))
    lo = np.searchsorted(minutes, at - w * MINUTE, side="left")
    hi = np.searchsorted(minutes, at - MINUTE, side="right")
    return cums[hi] - cums[lo]


def features_from_sums(sums: dict, price: dict) -> np.ndarray:
    """`sums[w]` = (n, k) array in EVENT_COLS order (or a dict of scalars);
    `price` = arrays/scalars for ret_w, range_w, pos_in_60, hour, weekday.
    Returns (n, F) in feature_names() order. THE one vector builder."""
    def col(w, name):
        block = sums.get(w)
        if block is None:
            return np.zeros(n)
        if isinstance(block, dict):
            return np.full(n, float(block.get(name, 0.0)))
        return np.asarray(block)[:, EVENT_COLS.index(name)]
    first = next(iter(sums.values()))
    n = 1 if isinstance(first, dict) else np.asarray(first).shape[0]
    out = []
    for w in WINDOWS:
        for c in EVENT_COLS:
            out.append(col(w, c))
        o_n = col(w, "o_n")
        out.append(col(w, "o_net") / (o_n + 1.0))
        out.append(col(w, "o_net_conf") / (o_n + 1.0))
        out.append(col(w, "o_net_lots") + col(w, "c_net_lots"))
        out.append(col(w, "o_stance_lots") + col(w, "c_stance_lots"))
    out.append(col(15, "o_n") / (col(120, "o_n") / 8.0 + 1.0))

    def pv(name, default=np.nan):
        v = price.get(name, default)
        v = np.asarray(v, dtype=float)
        return np.full(n, float(v)) if v.ndim == 0 else v
    for w in PRICE_WINDOWS:
        out.append(pv(f"ret_{w}")); out.append(pv(f"range_{w}"))
    out.append(pv("pos_in_60")); out.append(pv("hour", 0.0)); out.append(pv("weekday", 0.0))
    return np.column_stack(out).astype("float64")


def price_context(highs: np.ndarray, lows: np.ndarray, at: datetime) -> dict:
    """Price features from the COMPLETED bars up to T-1 (oldest first, the
    last element is the bar of minute T-1). `at` = the decision minute T."""
    highs = np.asarray(highs, dtype=float); lows = np.asarray(lows, dtype=float)
    mid = (highs + lows) / 2.0
    ctx: dict = {"hour": float(at.hour), "weekday": float(at.weekday())}
    if len(mid) == 0 or not np.isfinite(mid[-1]) or mid[-1] <= 0:
        return ctx
    for w in PRICE_WINDOWS:
        if len(mid) > w:
            ctx[f"ret_{w}"] = (mid[-1] / mid[-1 - w] - 1.0) * 1e4
        if len(mid) >= w:
            ctx[f"range_{w}"] = (np.max(highs[-w:]) - np.min(lows[-w:])) / mid[-1] * 1e4
    if len(mid) >= 60:
        hi60, lo60 = np.max(highs[-60:]), np.min(lows[-60:])
        ctx["pos_in_60"] = (mid[-1] - lo60) / (hi60 - lo60 + 1e-9)
    return ctx


# ------------------------------------------------------------ training
def _load_trades(symbol: str, scratch: Path, artifacts: Path) -> pd.DataFrame | None:
    cache = Path(scratch) / "quant_feature_cache.parquet"
    t = pd.read_parquet(cache, columns=["account_key", "symbol", "open_time", "close_time", "direction", "volume_lots"],
                        filters=[("symbol", "==", symbol)])
    if len(t) < 20_000:
        return None
    t["open_time"] = pd.to_datetime(t["open_time"]); t["close_time"] = pd.to_datetime(t["close_time"])
    sc = pd.read_parquet(Path(artifacts) / "quant_scores.parquet",
                         columns=["account_key", "symbol", "open_time", "close_time", "score_cal"])
    sc = sc.loc[sc["symbol"] == symbol].copy()
    sc["open_time"] = pd.to_datetime(sc["open_time"]); sc["close_time"] = pd.to_datetime(sc["close_time"])
    t = t.merge(sc.drop_duplicates(["account_key", "symbol", "open_time", "close_time"]),
                on=["account_key", "symbol", "open_time", "close_time"], how="left")
    t["score_coverage"] = t["score_cal"].notna()
    t["score_cal"] = t["score_cal"].fillna(0.5)
    return t


def _load_bars(symbol: str, scratch: Path) -> pd.DataFrame:
    import pyarrow.dataset as pads
    from webapp import path_features as pf
    dataset = pf._open_dataset(pf.bars_root(Path(scratch)))
    bars = dataset.to_table(filter=pads.field("symbol") == symbol, columns=["minute", "high", "low"]).to_pandas()
    if bars.empty:
        return bars
    bars["minute"] = pd.to_datetime(bars["minute"])
    return bars.sort_values("minute").drop_duplicates("minute").reset_index(drop=True)


def event_cumsums(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Opens and closes of a trade frame -> (minutes, cumulative sums)."""
    d = t["direction"].to_numpy(float); lots = t["volume_lots"].to_numpy(float); s = t["score_cal"].to_numpy(float)
    opens = event_terms(d, lots, s); closes = close_terms(d, lots, s)
    om = t["open_time"].dt.floor("min").to_numpy(); cm = t["close_time"].dt.floor("min").to_numpy()
    minutes = np.concatenate([om, cm])
    terms = {c: np.concatenate([opens[c], np.zeros(len(t))]) for c in opens}
    terms.update({c: np.concatenate([np.zeros(len(t)), closes[c]]) for c in closes})
    return minute_sums(minutes, terms)


def build_frame(symbol: str, scratch: Path, artifacts: Path, log=print, horizon: int = HORIZON):
    """Minute-grid feature frame + forward-move target for one symbol, built
    with the live-path functions (sums_at / features_from_sums) on completed
    minutes and completed bars."""
    t = _load_trades(symbol, scratch, artifacts)
    if t is None:
        return None, None, None, {"reason": "too few trades"}
    bars = _load_bars(symbol, scratch)
    if bars.empty:
        return None, None, None, {"reason": "no bars"}
    grid = bars.set_index("minute")
    at = grid.index.values.astype("datetime64[m]")
    minutes, cums = event_cumsums(t)
    sums = {w: sums_at(minutes, cums, at, w) for w in WINDOWS}
    # completed bars only: the bar of minute T-1 is the newest a decision at T can see
    high1 = grid["high"].shift(1); low1 = grid["low"].shift(1); mid1 = (high1 + low1) / 2.0
    price = {"hour": grid.index.hour.to_numpy(float), "weekday": grid.index.weekday.to_numpy(float)}
    for w in PRICE_WINDOWS:
        price[f"ret_{w}"] = ((mid1 / mid1.shift(w) - 1.0) * 1e4).to_numpy()
        price[f"range_{w}"] = ((high1.rolling(w).max() - low1.rolling(w).min()) / mid1 * 1e4).to_numpy()
    hi60 = high1.rolling(60).max(); lo60 = low1.rolling(60).min()
    price["pos_in_60"] = ((mid1 - lo60) / (hi60 - lo60 + 1e-9)).to_numpy()
    X = pd.DataFrame(features_from_sums(sums, price), index=grid.index, columns=feature_names())
    X = X.replace([np.inf, -np.inf], np.nan)
    mid = (grid["high"] + grid["low"]) / 2.0
    fwd = (mid.shift(-horizon) / mid - 1.0) * 1e4
    gap_ok = (grid.index.to_series().shift(-horizon) - grid.index.to_series()) <= pd.Timedelta(minutes=horizon + 5)
    y = fwd.where(gap_ok)
    hold = (t["close_time"] - t["open_time"]).dt.total_seconds() / 60.0
    info = {"trades": int(len(t)), "score_coverage": float(t["score_coverage"].mean()), "minutes": int(len(X)),
            "median_hold_min": float(hold.median()), "closed_within_h": float((hold <= horizon).mean())}
    return X, y, (t, bars), info


def parity_check(X: pd.DataFrame, t: pd.DataFrame, bars: pd.DataFrame, n: int = 400, seed: int = 0) -> dict:
    """Rebuild `n` random grid rows EXACTLY the way the live engine does --
    an event buffer of the last 120 minutes filtered to completed minutes,
    `minute_sums` + `sums_at` for a single T, `price_context` on the last 60
    completed bars -- and compare with the training frame. Returns the max
    absolute difference per group; the caller refuses to save on failure."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(np.flatnonzero(X.index >= X.index.min() + pd.Timedelta(hours=3)), size=min(n, len(X)), replace=False)
    om = t["open_time"].dt.floor("min").to_numpy().astype("datetime64[m]")
    cm = t["close_time"].dt.floor("min").to_numpy().astype("datetime64[m]")
    d = t["direction"].to_numpy(float); lots = t["volume_lots"].to_numpy(float); s = t["score_cal"].to_numpy(float)
    bm = bars["minute"].to_numpy().astype("datetime64[m]"); bh = bars["high"].to_numpy(float); bl = bars["low"].to_numpy(float)
    ev_cols = [c for c in X.columns if not c.startswith(("ret_", "range_", "pos_in", "hour", "weekday"))]
    pr_cols = [c for c in X.columns if c not in ev_cols]
    worst_ev = worst_pr = 0.0
    for r in rows:
        T = np.datetime64(X.index[r]).astype("datetime64[m]")
        lo_t = T - 120 * MINUTE
        sel_o = (om >= lo_t) & (om <= T - MINUTE); sel_c = (cm >= lo_t) & (cm <= T - MINUTE)
        opens = event_terms(d[sel_o], lots[sel_o], s[sel_o]); closes = close_terms(d[sel_c], lots[sel_c], s[sel_c])
        mins = np.concatenate([om[sel_o], cm[sel_c]])
        terms = {c: np.concatenate([opens[c], np.zeros(sel_c.sum())]) for c in opens}
        terms.update({c: np.concatenate([np.zeros(sel_o.sum()), closes[c]]) for c in closes})
        m, cs = minute_sums(mins, terms)
        sums = {w: sums_at(m, cs, np.array([T]), w) for w in WINDOWS}
        # completed bars up to T-1, as copy_rates_from_pos(symbol, M1, 1, 60) would return them
        hi_b = np.searchsorted(bm, T - MINUTE, side="right"); lo_b = max(0, hi_b - 61)
        ctx = price_context(bh[lo_b:hi_b], bl[lo_b:hi_b], pd.Timestamp(T).to_pydatetime())
        live = pd.Series(features_from_sums(sums, ctx)[0], index=feature_names())
        train = X.iloc[r]
        worst_ev = max(worst_ev, float(np.nanmax(np.abs(live[ev_cols].to_numpy() - train[ev_cols].to_numpy()))))
        a = live[pr_cols].to_numpy(); b = train[pr_cols].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            worst_pr = max(worst_pr, float(np.max(np.abs(a[both] - b[both]))))
        # a value one side has and the other lacks is a mismatch too (unless the bar history is simply short)
        if (np.isfinite(a) != np.isfinite(b)).any() and hi_b - lo_b >= 61:
            worst_pr = max(worst_pr, 1.0)
    return {"rows": int(len(rows)), "max_abs_diff_events": worst_ev, "max_abs_diff_price": worst_pr,
            "ok": bool(worst_ev < 1e-6 and worst_pr < 1e-6)}


def train_symbol(symbol: str, scratch: Path, artifacts: Path, log=print, horizon: int = HORIZON,
                 first_test_frac: float = 0.5, confidence_share: float = 0.5,
                 save_oos: bool = False) -> dict:
    """Walk-forward by week for the OOS metrics and the confidence threshold,
    a live-path parity check, then a final fit on everything; saves the
    booster + meta only when parity holds. Returns meta."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    started = time.time()
    X, y, raw, info = build_frame(symbol, scratch, artifacts, log, horizon)
    if X is None:
        log(f"{symbol}: skipped ({info.get('reason')})")
        return {"symbol": symbol, "ok": False, **info}
    log(f"{symbol}: {info['trades']:,} trades, {info['minutes']:,} minutes, score coverage {info['score_coverage']:.0%}")
    parity = parity_check(X, raw[0], raw[1])
    log(f"{symbol}: parity live-vs-train on {parity['rows']} rows: events {parity['max_abs_diff_events']:.2e}, "
        f"price {parity['max_abs_diff_price']:.2e} -> {'OK' if parity['ok'] else 'FAILED'}")
    if not parity["ok"]:
        return {"symbol": symbol, "ok": False, "reason": "parity check failed", "parity": parity, **info}
    ok = y.notna() & (y != 0)
    days = pd.DatetimeIndex(np.sort(X.index.normalize().unique()))
    first_test = days[int(len(days) * first_test_frac)]
    weeks = pd.date_range(first_test, days[-1] + pd.Timedelta(days=7), freq="7D")
    params = dict(objective="binary", n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbose=-1,
                  random_state=0)
    preds = pd.Series(np.nan, index=X.index)
    for i in range(len(weeks) - 1):
        lo, hi = weeks[i], weeks[i + 1]
        tr = ok & (X.index < lo); te = ok & (X.index >= lo) & (X.index < hi)
        if te.sum() == 0 or tr.sum() < 5000:
            continue
        m = lgb.LGBMClassifier(**params).fit(X[tr], (y[tr] > 0).astype(int))
        preds[te] = m.predict_proba(X[te])[:, 1]
    sel = preds.notna() & ok
    if sel.sum() < 2000:
        log(f"{symbol}: too few OOS minutes ({int(sel.sum())})")
        return {"symbol": symbol, "ok": False, "reason": "too few OOS minutes", **info}
    p = preds[sel].to_numpy(); yy = (y[sel] > 0).to_numpy().astype(int); fwd = y[sel].to_numpy()
    confv = np.abs(p - 0.5)
    threshold = float(np.quantile(confv, 1.0 - confidence_share))
    tiers = {}
    for share in (0.5, 0.2, 0.1):
        s_ = confv >= np.quantile(confv, 1.0 - share)
        signed = np.where(p[s_] > 0.5, fwd[s_], -fwd[s_])
        tiers[f"top{int(share*100)}"] = {"n": int(s_.sum()), "acc": float(((p[s_] > 0.5) == (yy[s_] == 1)).mean()),
                                         "bps": float(signed.mean())}
    weekly = pd.Series((p > 0.5) == (yy == 1), index=preds[sel].index).groupby(pd.Grouper(freq="7D")).mean().dropna()
    # One-position-at-a-time replay at the threshold: the panel's "expected".
    idx = preds[sel].index
    replay = {"cost_bps": COST_BPS, "trades": 0, "per_day": 0.0, "hit": 0.0, "bps_per_trade_net": 0.0,
              "bps_total_net": 0.0, "maxdd_bps": 0.0, "days": 0}
    pnl, pos_until = [], None
    for i in range(len(idx)):
        if pos_until is not None and idx[i] < pos_until:
            continue
        if confv[i] < threshold:
            continue
        pnl.append((fwd[i] if p[i] > 0.5 else -fwd[i]) - COST_BPS)
        pos_until = idx[i] + pd.Timedelta(minutes=horizon)
    if pnl:
        arr = np.asarray(pnl); cum = np.cumsum(arr); ndays = max(1, (idx[-1] - idx[0]).days)
        replay.update(trades=int(len(arr)), per_day=float(len(arr) / ndays), hit=float((arr > -COST_BPS).mean()),
                      bps_per_trade_net=float(arr.mean()), bps_total_net=float(arr.sum()),
                      maxdd_bps=float((cum - np.maximum.accumulate(cum)).min()), days=int(ndays))
    if save_oos:
        pd.DataFrame({"minute": idx, "p_up": p, "fwd_bps": fwd}).to_parquet(
            Path(scratch) / f"tape_{symbol}_{horizon}m_oos.parquet", index=False)
    final = lgb.LGBMClassifier(**params).fit(X[ok], (y[ok] > 0).astype(int))
    model_path, meta_path = artifact_paths(artifacts, symbol, horizon)
    final.booster_.save_model(str(model_path))
    meta = {"symbol": symbol, "ok": True, "horizon": horizon, "features": feature_names(),
            "threshold": threshold, "confidence_share": confidence_share,
            "oos": {"rows": int(sel.sum()), "acc": float(((p > 0.5) == (yy == 1)).mean()),
                    "auc": float(roc_auc_score(yy, p)), "weeks": len(weekly),
                    "weekly_acc_min": float(weekly.min()), "weekly_acc_max": float(weekly.max()), **tiers},
            "replay": replay, "parity": parity,
            "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "data_end": str(X.index.max()), "train_minutes": float((time.time() - started) / 60.0), **info}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"{symbol}: OOS acc {meta['oos']['acc']:.3f} AUC {meta['oos']['auc']:.3f} | top50 {tiers['top50']['acc']:.3f} "
        f"({tiers['top50']['bps']:+.1f} bps) | top10 {tiers['top10']['acc']:.3f} | threshold {threshold:.3f} | "
        f"replay {replay['per_day']:.0f}/day {replay['bps_per_trade_net']:+.1f} bps net, DD {replay['maxdd_bps']:.0f} bps | "
        f"weekly {meta['oos']['weekly_acc_min']:.2f}-{meta['oos']['weekly_acc_max']:.2f} | {meta['train_minutes']:.1f} min")
    return meta


def load(artifacts: Path, symbol: str, horizon: int = HORIZON):
    """(booster, meta) or (None, None)."""
    import lightgbm as lgb
    model_path, meta_path = artifact_paths(artifacts, symbol, horizon)
    if not model_path.exists() or not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(model_path)), meta
