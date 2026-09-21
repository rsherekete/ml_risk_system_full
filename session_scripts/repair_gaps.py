import sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import backfill_repair as br, data_store
r = br.repair(progress=lambda m: print(f"  {m}", flush=True), vpn_timeout=10800)
print(f"filled {len(r['filled'])} months, {r['rows']:,} rows")
if r["failed"]: print("STILL MISSING:", r["failed"])
c = data_store.coverage()
print(f"coverage: complete {c['complete']}/{c['expected']} | incomplete {c['incomplete'] or 'none'}")
for s in c["servers"]:
    print(f"  {s['server']:<20} {s['files']:>3}mo {s['rows']:>12,}  {'complete' if s['complete'] else str(s['gap_count'])+' missing'}")
