import json, collections
import numpy as np
IN = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
tr = json.load(open(IN))["trades"]

def cat(t):
    o = (t.get("out_comment") or "").lower()
    if o.startswith("[so"): return "MARGIN STOP-OUT [so]"
    if o.startswith("[sl"): return "broker stop-loss [sl]"
    if o.startswith("[tp"): return "broker take-profit [tp]"
    if o.startswith("x1brk"): return "our bracket SL/TP (x1brk)"
    if o.startswith("x1mir"): return "mirror exit (x1mir)"
    if o.startswith("x1hvs"): return "harvest (x1hvs)"
    if o.startswith("x1net"): return "netting (x1net)"
    if o.startswith("x1rot"): return "rotation (x1rot)"
    if o.startswith("x1rty"): return "retry (x1rty)"
    if o.startswith("x2ttl"): return "exit-model TTL (x2ttl)"
    if o.startswith("x1cls"): return "generic close (x1cls)"
    return "other: " + (o[:10] or "(blank)")

for scope, rows in (("ALL s1", tr), ("GOLD", [t for t in tr if t["symbol"] == "XAUUSD+"]),
                    ("NON-GOLD", [t for t in tr if t["symbol"] != "XAUUSD+"])):
    d = collections.defaultdict(lambda: {"n": 0, "net": 0.0, "wins": 0})
    for t in rows:
        b = d[cat(t)]; b["n"] += 1; b["net"] += t["net"]; b["wins"] += 1 if t["net"] > 0 else 0
    print("\n===== %s  (n=%d, net $%.0f) =====" % (scope, len(rows), sum(t["net"] for t in rows)))
    for k, b in sorted(d.items(), key=lambda kv: kv[1]["net"]):
        print("  %-26s n=%3d  net=$%9.2f  win=%3.0f%%  avg=$%7.2f" % (
            k, b["n"], b["net"], 100*b["wins"]/max(b["n"],1), b["net"]/max(b["n"],1)))

# mirror+harvest vs stop-family, gold
gold = [t for t in tr if t["symbol"] == "XAUUSD+"]
STOPS = ("[so", "[sl", "x1brk")
stop_net = sum(t["net"] for t in gold if (t.get("out_comment") or "").lower().startswith(STOPS))
rest_net = sum(t["net"] for t in gold if not (t.get("out_comment") or "").lower().startswith(STOPS))
print("\nGOLD: stop-family exits net = $%.0f | everything-else (mirror/harvest/etc) net = $%.0f"
      % (stop_net, rest_net))
