import sys, json, collections
import numpy as np
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
M = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\matches.json"
mm = json.load(open(M))
mm = [t for t in mm if t["c"]["close_ts"]]          # client-closed only
XAU = 100.0
DELTAS = [-30, -10, -5, 0, 5, 10, 30, 60, 120]

conns = {}
def conn(server):
    if server not in conns:
        conns[server] = mysql_extract._connection(server, timeout=60)
    return conns[server]

def ticks(server, symbol, t0, t1):
    with conn(server).cursor() as cur:
        cur.execute("SELECT tm, bid, ask FROM ticks WHERE symbol_name=%s "
                    "AND tm BETWEEN %s AND %s ORDER BY tm", (symbol, t0, t1))
        return [(int(r[0]), (float(r[1]) + float(r[2])) / 2.0) for r in cur.fetchall()
                if r[1] and r[2]]

# smoke test on the first trade
c0 = mm[0]["c"]
tk = ticks(c0["server"], c0["symbol"], c0["close_ts"] - 30, c0["close_ts"] + 150)
print("tick smoke test: server=%s sym=%s close_ts=%d -> %d ticks in window"
      % (c0["server"], c0["symbol"], c0["close_ts"], len(tk)))
if not tk:
    print("NO TICKS -> aborting latency study"); sys.exit(0)

def price_at(series, t):
    prior = [p for (ts, p) in series if ts <= t]
    return prior[-1] if prior else (series[0][1] if series else None)

sums = {d: 0.0 for d in DELTAS}
sums_f = {d: 0.0 for d in DELTAS}
sums_n = {d: 0.0 for d in DELTAS}
used = 0
for t in mm:
    c = t["c"]
    ser = ticks(c["server"], c["symbol"], c["close_ts"] - 60, c["close_ts"] + 200)
    if not ser:
        continue
    base = price_at(ser, c["close_ts"])
    if base is None:
        continue
    used += 1
    for d in DELTAS:
        p = price_at(ser, c["close_ts"] + d)
        if p is None:
            continue
        delta_pnl = (p - base) * t["in_dir"] * t["volume"] * XAU
        sums[d] += delta_pnl
        (sums_f if t["cat"] == "forced" else sums_n)[d] += delta_pnl

print("\nLATENCY SWEEP -- change in TOTAL mirror P&L vs exiting at the client's instant")
print("(positive delay = we exit LATER than the client)   trades priced: %d" % used)
print("  %6s  %12s  %12s  %12s" % ("delay", "ALL", "forced", "natural"))
for d in DELTAS:
    print("  %+5ds  %+11.0f  %+11.0f  %+11.0f" % (d, sums[d], sums_f[d], sums_n[d]))
