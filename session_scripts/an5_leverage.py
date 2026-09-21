import json
import numpy as np
IN = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
tr = json.load(open(IN))["trades"]
gold = [t for t in tr if t["symbol"] == "XAUUSD+"]
XAU_CONTRACT = 100  # oz per lot

lots = [t["our_lots"] or 0 for t in gold]
notionals = [(t["our_lots"] or 0) * (t["in_price"] or 0) * XAU_CONTRACT for t in gold]
print("Fresh balance ~ $1000 (reset 2026-09-04 16:00)")
print("GOLD per-trade lots: min=%.3f med=%.3f max=%.3f" % (
    min(lots), float(np.median(lots)), max(lots)))
print("GOLD per-trade NOTIONAL $: med=$%.0f max=$%.0f" % (
    float(np.median(notionals)), max(notionals)))
print("  => a single median gold position is %.0fx the $1000 equity" % (np.median(notionals)/1000))

# concurrency: how many gold positions open at once (intervals overlap)
events = []
for t in gold:
    events.append((t["in_time"], +1, (t["our_lots"] or 0)*(t["in_price"] or 0)*XAU_CONTRACT))
    events.append((t["out_time"], -1, -(t["our_lots"] or 0)*(t["in_price"] or 0)*XAU_CONTRACT))
events.sort()
cur_n = 0; cur_notional = 0.0; max_n = 0; max_notional = 0.0
for _, dn, dnot in events:
    cur_n += dn; cur_notional += dnot
    max_n = max(max_n, cur_n); max_notional = max(max_notional, cur_notional)
print("\nPeak concurrent GOLD positions: %d" % max_n)
print("Peak concurrent GOLD notional: $%.0f  => %.0fx the $1000 equity" % (
    max_notional, max_notional/1000))

# margin-forced vs natural exits recap
def cat(t):
    o = (t.get("out_comment") or "").lower()
    if o.startswith("[so") or o.startswith("x1brk"): return "margin-forced"
    return "natural"
forced = [t for t in gold if cat(t) == "margin-forced"]
nat = [t for t in gold if cat(t) == "natural"]
print("\nGOLD margin-forced exits: n=%d net=$%.0f (avg $%.2f)" % (
    len(forced), sum(t["net"] for t in forced), np.mean([t["net"] for t in forced])))
print("GOLD natural exits:       n=%d net=$%.0f" % (len(nat), sum(t["net"] for t in nat)))
