import sys, duckdb
STORE = r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\live_stream.duckdb"
try:
    con = duckdb.connect(STORE, read_only=True)
except Exception as e:
    print("locked, cannot read directly:", e); sys.exit()
# pipeline latency = ingested_at - event_time, for recently ingested trade events
q = """
SELECT server,
       COUNT(*) n,
       AVG(EPOCH(ingested_at) - EPOCH(event_time)) avg_lat,
       MEDIAN(EPOCH(ingested_at) - EPOCH(event_time)) med_lat,
       MIN(EPOCH(ingested_at) - EPOCH(event_time)) min_lat
FROM events
WHERE ingested_at > now() - INTERVAL 5 MINUTE
  AND action IN ('DEAL_BUY','DEAL_SELL')
GROUP BY server ORDER BY n DESC
"""
print("KAFKA pipeline latency (ingested_at - event_time), last 5 min of TRADE events:")
for r in con.execute(q).fetchall():
    print("  %-18s n=%5d  avg=%.2fs  median=%.2fs  min=%.2fs" % (r[0], r[1], r[2], r[3], r[4]))
# also quotes freshness
qq = """SELECT COUNT(*) n, MEDIAN(EPOCH(ingested_at)-EPOCH(event_time)) med
        FROM quotes WHERE ingested_at > now() - INTERVAL 2 MINUTE"""
r = con.execute(qq).fetchone()
print("QUOTES last 2min: n=%s median_lat=%.2fs" % (r[0], r[1] if r[1] is not None else -1))
con.close()
