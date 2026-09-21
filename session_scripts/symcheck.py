import sys, sqlite3, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from pathlib import Path
from webapp import model_service as ms
ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")

# 1) the persisted training symbol_code map
syms = (ART / "quant_symbols.txt").read_text(encoding="utf-8").splitlines()
smap = {s: i for i, s in enumerate(syms)}
print(f"quant_symbols.txt: {len(syms)} symbols")
gold_train = [s for s in syms if "XAU" in s.upper() or "GOLD" in s.upper()]
print("  gold-ish in training map:", gold_train)

# 2) symbols the LIVE feed produced (order store)
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
live_syms = [r[0] for r in cx.execute(
    "SELECT DISTINCT symbol FROM vantage_orders WHERE symbol IS NOT NULL")]
print(f"\nlive feed distinct symbols: {len(live_syms)}")
gold_live = [s for s in live_syms if "XAU" in str(s).upper() or "GOLD" in str(s).upper()]
print("  gold-ish live:", gold_live)

# 3) do live symbols resolve in the training map?
print("\n=== live symbol -> training symbol_code resolution ===")
unresolved = [s for s in live_syms if s not in smap]
print(f"  resolve: {len(live_syms)-len(unresolved)}/{len(live_syms)}")
print(f"  UNRESOLVED live symbols (symbol_code=NaN live): {len(unresolved)}")
for s in unresolved[:40]:
    print("    ", s)
