import sqlite3, time
import pandas as pd
from pathlib import Path
ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")

# --- training-derived join tables (what score() looks accounts up in) ---
hist_idx, ad_idx = set(), set()
hcache = ART / "quant_history_cache.parquet"
acache = ART / "quant_ad_cache.parquet"
if hcache.exists():
    h = pd.read_parquet(hcache)
    hist_idx = set(map(str, h.index.tolist()))
    print(f"history cache: {len(hist_idx):,} accounts | sample: {list(hist_idx)[:3]}")
else:
    print("NO history cache")
if acache.exists():
    a = pd.read_parquet(acache)
    ad_idx = set(map(str, a.index.tolist()))
    print(f"AD  cache: {len(ad_idx):,} accounts | sample: {list(ad_idx)[:3]} | cols {len(a.columns)}")
else:
    print("NO AD cache")

# --- accounts we actually traded live (order store, last 4h) ---
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
cx.row_factory = sqlite3.Row
win = time.time() - 4*3600
rows = [dict(r) for r in cx.execute(
    "SELECT DISTINCT source_account FROM vantage_orders WHERE created>=? AND source_account IS NOT NULL",
    (win,))]
live = set(str(r["source_account"]) for r in rows if r["source_account"])
print(f"\nlive-traded accounts (4h): {len(live):,} | sample: {list(live)[:5]}")

if live:
    in_hist = sum(1 for a in live if a in hist_idx)
    in_ad = sum(1 for a in live if a in ad_idx)
    print(f"\n  in history table: {in_hist}/{len(live)} = {in_hist/len(live):.1%}")
    print(f"  in AD  table: {in_ad}/{len(live)} = {in_ad/len(live):.1%}")
    missing = [a for a in live if a not in hist_idx][:8]
    print(f"  examples NOT in history: {missing}")
    # format check: do the key shapes even match?
    print(f"\n  live key shape : {list(live)[0]!r}")
    if hist_idx: print(f"  hist key shape : {list(hist_idx)[0]!r}")
    if ad_idx:   print(f"  ad   key shape : {list(ad_idx)[0]!r}")
