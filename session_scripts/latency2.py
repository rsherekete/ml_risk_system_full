import sys, json, datetime as dt
import numpy as np
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
M = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\matches.json"
mm = [t for t in json.load(open(M)) if t["c"]["close_ts"]]
XAU = 100.0
DELTAS = [-30, -10, -5, 0, 5, 10, 30, 60, 120]
conn = mysql_extract._connection("mt4_live01", timeout=60)   # centralized tick feed

def ticks(dt0, dt1):
    with conn.cursor() as cur:
        cur.execute("SELECT tm, bid, ask FROM ticks WHERE symbol_name=%s "
                    "AND tm BETWEEN %s AND %s ORDER BY tm", ("XAUUSD", dt0, dt1))
        return [(r[0], (float(r[1]) + float(r[2])) / 2.0) for r in cur.fetchall() if r[1] and r[2]]

def price_at(series, when):
    prior = [p for (ts, p) in series if ts <= when]
    return prior[-1] if prior else (series[0][1] if series else None)

# alignment check on 5 trades: mid at close_ts datetime vs client close_price
print("ALIGNMENT check (tick mid at close_ts  vs  client close_price):")
aligned = 0
for t in mm[:8]:
    cdt = dt.datetime.utcfromtimestamp(t["c"]["close_ts"])
    ser = ticks(cdt - dt.timedelta(seconds=30), cdt + dt.timedelta(seconds=30))
    mid = price_at(ser, cdt)
    if mid:
        diff = mid - t["c"]["close"]
        aligned += abs(diff) < 3.0
        print("  close=%.2f  tickmid=%.2f  diff=%+.2f  (%d ticks)" % (t["c"]["close"], mid, diff, len(ser)))
if aligned < 3:
    print("POOR alignment -> timezone still off; aborting"); sys.exit(0)

# latency sweep
sums = {d: 0.0 for d in DELTAS}; sf = {d: 0.0 for d in DELTAS}; sn = {d: 0.0 for d in DELTAS}
used = 0
for t in mm:
    cdt = dt.datetime.utcfromtimestamp(t["c"]["close_ts"])
    ser = ticks(cdt - dt.timedelta(seconds=60), cdt + dt.timedelta(seconds=200))
    base = price_at(ser, cdt)
    if base is None:
        continue
    used += 1
    for d in DELTAS:
        p = price_at(ser, cdt + dt.timedelta(seconds=d))
        if p is None:
            continue
        dp = (p - base) * t["in_dir"] * t["volume"] * XAU
        sums[d] += dp
        (sf if t["cat"] == "forced" else sn)[d] += dp

print("\nLATENCY SWEEP -- change in TOTAL mirror P&L if every exit were delayed by d")
print("(positive delay = exit LATER than the client)   priced: %d trades" % used)
print("  %6s  %10s  %10s  %10s" % ("delay", "ALL", "forced", "natural"))
for d in DELTAS:
    print("  %+5ds  %+9.0f  %+9.0f  %+9.0f" % (d, sums[d], sf[d], sn[d]))
