import json, collections
import numpy as np
IN = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
tr = json.load(open(IN))["trades"]
gold = [t for t in tr if t["symbol"] == "XAUUSD+"]

def grp(rows, key):
    d = collections.defaultdict(lambda: {"n": 0, "net": 0.0, "wins": 0})
    for t in rows:
        b = d[key(t)]; b["n"] += 1; b["net"] += t["net"]; b["wins"] += 1 if t["net"] > 0 else 0
    return d

def show(title, rows, key):
    print("\n=== %s (n=%d, net $%.0f) ===" % (title, len(rows), sum(t["net"] for t in rows)))
    for k, b in sorted(grp(rows, key).items(), key=lambda kv: kv[1]["net"]):
        print("  %-14s n=%3d net=$%9.2f win=%3.0f%%" % (str(k), b["n"], b["net"], 100*b["wins"]/max(b["n"],1)))

show("GOLD by EXIT reason (out_comment)", gold, lambda t: (t.get("out_comment") or "?")[:6])
show("GOLD by OPEN reason (in_comment prefix)", gold, lambda t: (t.get("in_comment") or "?")[:3])

# Can the OPEN comment recover a source the order store missed?
none_src = [t for t in gold if not t.get("source_account")]
rec = sum(1 for t in none_src if t.get("source_from_comment"))
print("\nGOLD with no order-store source: %d (net $%.0f)" % (len(none_src), sum(t['net'] for t in none_src)))
print("  ...of which OPEN-comment recovers a client source: %d" % rec)
show("  un-sourced GOLD by OPEN comment", none_src, lambda t: (t.get("in_comment") or "?")[:4])

# the big losers: list worst 15 gold trades with full context
print("\n=== WORST 15 GOLD trades ===")
for t in sorted(gold, key=lambda x: x["net"])[:15]:
    hold = t["out_time"] - t["in_time"]
    print("  net=$%8.2f hold=%5ds dir=%+d in='%s' out='%s' src=%s score=%s" % (
        t["net"], hold, t["in_dir"], (t.get("in_comment") or "")[:16],
        (t.get("out_comment") or "")[:8], t.get("source_account"),
        round(t["live_score"],2) if t.get("live_score") is not None else None))
