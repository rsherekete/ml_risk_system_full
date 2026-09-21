import json, collections
import numpy as np
IN = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
tr = json.load(open(IN))["trades"]
DEADLINES = {300: "5m", 1800: "30m", 7200: "2h", 21600: "6h"}

def hold(t): return t["out_time"] - t["in_time"]

def at_deadline(h, tol=3):
    for s in DEADLINES:
        if abs(h - s) <= tol:
            return DEADLINES[s]
    return None

for label, rows in (("ALL s1", tr),
                    ("GOLD (XAUUSD+)", [t for t in tr if t["symbol"] == "XAUUSD+"]),
                    ("NON-GOLD", [t for t in tr if t["symbol"] != "XAUUSD+"])):
    holds = [hold(t) for t in tr if False]  # placeholder
    n = len(rows)
    at = [at_deadline(hold(t)) for t in rows]
    dl = sum(1 for a in at if a)
    print("\n=== %s  (n=%d) ===" % (label, n))
    print("  held to an EXACT exit-model deadline (5m/30m/2h/6h): %d (%.0f%%)" % (dl, 100*dl/max(n,1)))
    c = collections.Counter(a for a in at if a)
    print("  deadline breakdown:", dict(c))
    # win/loss split at deadline vs not
    dl_rows = [t for t, a in zip(rows, at) if a]
    free_rows = [t for t, a in zip(rows, at) if not a]
    def stat(rs):
        if not rs: return "n=0"
        nets = [t["net"] for t in rs]
        wins = sum(1 for x in nets if x > 0)
        return "n=%d net=$%.0f win=%.0f%% avgwin=$%.2f avgloss=$%.2f" % (
            len(rs), sum(nets), 100*wins/len(rs),
            np.mean([x for x in nets if x > 0]) if wins else 0,
            np.mean([x for x in nets if x <= 0]) if (len(rs)-wins) else 0)
    print("  DEADLINE-closed:", stat(dl_rows))
    print("  free (mirror/other):", stat(free_rows))
    # hold-time percentiles
    hs = sorted(hold(t) for t in rows)
    if hs:
        print("  hold secs p10/50/90/max: %d / %d / %d / %d" % (
            hs[len(hs)//10], hs[len(hs)//2], hs[min(len(hs)-1,9*len(hs)//10)], hs[-1]))
