import sys, duckdb
NB = r"c:\Users\RoyVivasi\Documents\notebook"
STORE = NB + r"\webapp\artifacts\live_stream.duckdb"
try:
    con = duckdb.connect(STORE, read_only=True)
except Exception as e:
    print("open failed (app holds lock):", e); sys.exit()
print("columns:", [c[0] for c in con.execute("DESCRIBE events").fetchall()])
print("\ntotal events:", con.execute("SELECT COUNT(*) FROM events").fetchone()[0])
print("time span:", con.execute("SELECT MIN(event_time), MAX(event_time) FROM events").fetchone())
print("\ntoday's events by server:")
for r in con.execute("""SELECT server, COUNT(*) n, COUNT(DISTINCT symbol) syms
    FROM events WHERE event_time >= current_date GROUP BY server ORDER BY n DESC""").fetchall():
    print("  ", r)
print("\nsample today rows:")
for r in con.execute("""SELECT event_time, server, canonical, action, entry, volume, price
    FROM events WHERE event_time >= current_date AND canonical IS NOT NULL
    ORDER BY event_time DESC LIMIT 6""").fetchall():
    print("  ", r)
print("\ntop symbols today by deal count:")
for r in con.execute("""SELECT canonical, COUNT(*) n,
    SUM(CASE WHEN action LIKE '%BUY%' THEN volume ELSE -volume END) net_vol_raw
    FROM events WHERE event_time >= current_date AND canonical IS NOT NULL
    GROUP BY canonical ORDER BY n DESC LIMIT 8""").fetchall():
    print("  ", r)
con.close()
