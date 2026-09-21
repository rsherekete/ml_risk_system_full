import sqlite3, time, collections
DB = r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db"
cx = sqlite3.connect(DB); cx.row_factory = sqlite3.Row
now = time.time(); win = now - 4*3600
rows = [dict(r) for r in cx.execute(
    "SELECT * FROM vantage_orders WHERE created >= ? ORDER BY created", (win,))]
print("=== LAST 4H ORDER STORE ===")
print("total rows:", len(rows))
print("by status:", dict(collections.Counter(r['status'] for r in rows)))
print("by stance :", dict(collections.Counter(r['stance'] for r in rows)))
print()
print("=== FILLED (what we took) ===")
for st in ('copy','invert','fade15'):
    f = [r for r in rows if r['status']=='filled' and r['stance']==st]
    if not f: continue
    exp = sum(float(r['expected_usd'] or 0) for r in f)
    lots = sum(float(r['our_lots'] or 0) for r in f)
    print(f"  {st:7s}: {len(f):4d} fills | expected ${exp:+,.2f} | lots {lots:.2f}")
tot_exp_fill = sum(float(r['expected_usd'] or 0) for r in rows if r['status']=='filled')
print(f"  TOTAL expected on fills: ${tot_exp_fill:+,.2f}")
print()
print("=== BLOCKED / NOT FILLED (why we didn't) ===")
nb = [r for r in rows if r['status'] not in ('filled','netted','recycled')]
print("count:", len(nb))
def bucket(detail):
    d = (detail or '').lower()
    if 'cost' in d or 'hurdle' in d or 'spread' in d or 'commission' in d: return 'cost hurdle'
    if 'leverage' in d or 'wall' in d or 'margin' in d: return 'leverage/margin wall'
    if 'venue minimum' in d or '2x intended' in d or 'amplif' in d: return 'min-lot amplification guard'
    if 'position' in d and 'cap' in d: return 'position cap'
    if 'symbol' in d and ('cap' in d or 'share' in d): return 'symbol cap'
    if 'breaker' in d or 'paused' in d: return 'circuit breaker'
    if 'retry' in d or 'requote' in d or 'reject' in d: return 'venue reject/requote'
    if 'pending' in d or 'limit' in d: return 'limit pending/expired'
    return (detail or 'other')[:50]
bc = collections.Counter(bucket(r['detail']) for r in nb)
for reason, n in bc.most_common():
    exp = sum(float(r['expected_usd'] or 0) for r in nb if bucket(r['detail'])==reason)
    print(f"  {n:4d}  {reason:34s} forgone expected ${exp:+,.2f}")
print()
print("=== sample non-fill details ===")
seen=set()
for r in nb:
    b=bucket(r['detail'])
    if b in seen: continue
    seen.add(b)
    print(f"  [{b}] {r['stance']:6s} {r['symbol']:10s} exp {float(r['expected_usd'] or 0):+8.2f}  {(r['detail'] or '')[:90]}")
