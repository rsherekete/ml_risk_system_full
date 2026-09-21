import sys, sqlite3
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from pathlib import Path
ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")
syms = (ART / "quant_symbols.txt").read_text(encoding="utf-8").splitlines()
smap = {s: i for i, s in enumerate(syms)}

from webapp.exposure_policy import canonical_symbol
from webapp.trade_feed import _canonical

cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
live_syms = [r[0] for r in cx.execute("SELECT DISTINCT symbol FROM vantage_orders WHERE symbol IS NOT NULL")]
unresolved = [s for s in live_syms if s not in smap]
print(f"unresolved live symbols: {len(unresolved)}")

# build canonical -> training code map (from training raw symbols)
train_canon = canonical_symbol(pd.Series(syms))
canon_to_code = {}
for raw, c in zip(syms, train_canon):
    canon_to_code.setdefault(c, smap[raw])   # first variant's code as representative

fixed_exp = fixed_tf = 0
print("\nsymbol | exposure_canon | in_map? | trade_feed_canon | in_map?")
for s in unresolved[:20]:
    ce = canonical_symbol(pd.Series([s])).iloc[0]
    ct = _canonical(s)
    in_e = ce in canon_to_code or ce in smap
    in_t = ct in canon_to_code or ct in smap
    print(f"  {s:12s} {ce:10s} {str(in_e):5s}  {ct:12s} {str(in_t):5s}")
for s in unresolved:
    ce = canonical_symbol(pd.Series([s])).iloc[0]
    if ce in canon_to_code or ce in smap: fixed_exp += 1
    ct = _canonical(s)
    if ct in canon_to_code or ct in smap: fixed_tf += 1
print(f"\nRESOLVED after canonicalization:")
print(f"  exposure_policy.canonical_symbol: {fixed_exp}/{len(unresolved)}")
print(f"  trade_feed._canonical           : {fixed_tf}/{len(unresolved)}")
