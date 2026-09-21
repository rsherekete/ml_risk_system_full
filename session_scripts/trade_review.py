import sys, sqlite3, time, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")

# Past hour since 12:45 UTC (13:45 BST). Use last 75 min to be safe.
since = time.time() - 75 * 60
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
cx.row_factory = sqlite3.Row

def counts(table):
    try:
        rows = cx.execute(f"SELECT status, stance, COUNT(*) n FROM {table} "
                          f"WHERE created > ? GROUP BY status, stance ORDER BY n DESC", (since,)).fetchall()
        return [(r["status"], r["stance"], r["n"]) for r in rows]
    except Exception as e:
        return [("ERR", str(e), 0)]

print("=== order outcomes since 12:45 UTC ===")
print("vantage_orders (post-reset):", counts("vantage_orders"))
print("vantage_orders_archived (pre-reset):", counts("vantage_orders_archived"))

# Actual filled orders with detail
print("\n=== FILLED orders (both tables) last 75min ===")
for table in ("vantage_orders", "vantage_orders_archived"):
    try:
        for r in cx.execute(f"SELECT created, stance, symbol, our_lots, our_direction, "
                            f"client_direction, fill_price, live_score, ticket FROM {table} "
                            f"WHERE created > ? AND status='filled' ORDER BY created", (since,)).fetchall():
            print("  %s %-6s %-8s lots=%s ourdir=%s clidir=%s @ %s score=%s tk=%s [%s]" % (
                dt.datetime.utcfromtimestamp(r["created"]).strftime("%H:%M:%S"),
                r["stance"], r["symbol"], r["our_lots"], r["our_direction"],
                r["client_direction"], r["fill_price"],
                round(r["live_score"],3) if r["live_score"] is not None else None,
                r["ticket"], table[:8]))
    except Exception as e:
        print("  ", table, "ERR", e)

# Realized P&L by stance via the reconcile (reads MT5 history, respects reset epoch)
print("\n=== realized P&L by stance (account_reconcile) ===")
try:
    from webapp import vantage
    rec = vantage.account_reconcile(days=1)
    if rec.get("available"):
        print("  balance:", rec.get("balance"), "| deposits:", rec.get("deposits"),
              "| trading total:", rec.get("trading"))
        for st, d in (rec.get("by_stance") or {}).items():
            print("  %-10s pnl=%.2f n=%s" % (st, d.get("pnl", 0), d.get("n")))
    else:
        print("  reconcile not available:", rec.get("reason"))
except Exception as e:
    import traceback; traceback.print_exc()
