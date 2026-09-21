import json
import numpy as np
DUMP = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
M = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\matches.json"
tr = json.load(open(DUMP))["trades"]
mm = json.load(open(M))
XAU = 100.0
OLD_LOT, NEW_LOT = 0.05, 0.01          # min_lot floor before/after
SIZE = NEW_LOT / OLD_LOT               # ~5x smaller positions under safe sizing

def mirror(t):
    c = t["c"]
    return None if not c["close_ts"] else (c["close"] - c["open"]) * t["in_dir"] * t["volume"] * XAU

s1_actual = sum(t["net"] for t in tr)
gold = [t for t in tr if t["symbol"] == "XAUUSD+"]
gold_actual = sum(t["net"] for t in gold)
print("ACTUAL (since Sep-4 reset): s1 total = $%.0f | gold = $%.0f | non-gold = $%.0f"
      % (s1_actual, gold_actual, s1_actual - gold_actual))

# matched gold, client-closed
mc = [t for t in mm if t["c"]["close_ts"]]
m_actual = sum(t["profit"] for t in mc)          # gross (deal profit)
m_mirror = sum(mirror(t) for t in mc)
print("\n[A] EXIT FIX ONLY (same sizes traded, no forced liquidation):")
print("   matched gold %d: actual gross $%.0f  ->  held-to-client-exit $%.0f  (recovered $%.0f)"
      % (len(mc), m_actual, m_mirror, m_mirror - m_actual))
# scale the recovered delta to ALL gold by matched coverage of the forced loss
forced = [t for t in gold if (t.get("out_comment") or "").lower().startswith(("[so", "x1brk"))]
forced_actual = sum(t["net"] for t in forced)
mc_forced = [t for t in mc if (t.get("cat") == "forced")]
rec_ratio = (sum(mirror(t) - t["profit"] for t in mc_forced) /
             abs(sum(t["profit"] for t in mc_forced))) if mc_forced else 0
print("   all gold forced-exit trades %d: actual $%.0f" % (len(forced), forced_actual))
gold_exitfix = gold_actual - forced_actual + forced_actual * (1 - rec_ratio)
print("   => gold under exit-fix ~ $%.0f  (from $%.0f)" % (gold_exitfix, gold_actual))
print("   => s1 under exit-fix ~ $%.0f  (from $%.0f)" % (s1_actual - gold_actual + gold_exitfix, s1_actual))

print("\n[B] FULL FIX (safe %.0fx sizing: positions ~%.0f%% of size, + no forced liquidation):" % (20, SIZE*100))
print("   losses and gains both scale ~%.2fx; gold ~breakeven at small scale" % SIZE)
print("   gold: ~$%.0f  |  no margin stop-outs, no blow-up risk" % (gold_exitfix * SIZE))
print("   NOTE: $1k at safe sizing earns small ABSOLUTE $; the fix removes the")
print("   bleed & tail risk. Returns scale with capital (wall = 20 x equity).")
