import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd, numpy as np
from webapp import trade_feed
from webapp.trade_features import _AD_DIR

# cent logins for mt5_live01 (the biggest cent server)
cents = {("mt5_live01", l) for l in trade_feed.cent_logins("mt5_live01")}

def tag_cent(df):
    df = df.copy()
    if "account_key" in df.columns:
        parts = df["account_key"].astype(str).str.split(":", n=1, expand=True)
        df["_srv"] = parts[0]; df["_login"] = pd.to_numeric(parts[1], errors="coerce")
    df["_is_cent"] = [(s, l) in cents for s, l in zip(df.get("_srv", []), df.get("_login", []))]
    return df

# 1) BQ 90d records (training corpus + views)
try:
    rec = pd.read_parquet(_AD_DIR / "bq_90d_records.parquet",
                          columns=["account_key", "symbol", "volume_lots", "net_profit"])
    rec = tag_cent(rec)
    c = rec[rec["_is_cent"]]; nc = rec[~rec["_is_cent"]]
    print("RECORDS corpus (bq_90d_records.parquet):")
    print("  cent rows: mean |net_profit|=%.2f mean lots=%.4f" % (
        c["net_profit"].abs().mean(), c["volume_lots"].mean()))
    print("  non-cent : mean |net_profit|=%.2f mean lots=%.4f" % (
        nc["net_profit"].abs().mean(), nc["volume_lots"].mean()))
    print("  => if cent |pnl|/lots are ~100x non-cent, the corpus is INFLATED")
except Exception as e:
    print("records:", e)

# 2) AD corpus (model_frame) -- check a money feature vs a ratio feature
try:
    ad = pd.read_parquet(_AD_DIR / "model_frame.parquet")
    cols = [c for c in ad.columns if c in ("account_key", "ad_mean_pnl", "ad_day_pnl",
            "ad_pnl_20d", "ad_life_win_rate", "ad_notional")]
    ad = ad[[c for c in cols if c in ad.columns]].copy()
    ad = tag_cent(ad)
    moneycol = next((c for c in ("ad_mean_pnl", "ad_day_pnl", "ad_pnl_20d", "ad_notional") if c in ad.columns), None)
    print("\nAD corpus (model_frame.parquet) money feature:", moneycol)
    if moneycol:
        c = ad[ad["_is_cent"]]; nc = ad[~ad["_is_cent"]]
        print("  cent mean %s=%.2f | non-cent mean=%.2f" % (
            moneycol, c[moneycol].abs().mean(), nc[moneycol].abs().mean()))
except Exception as e:
    print("ad corpus:", e)
