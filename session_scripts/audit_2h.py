"""Last-2-hours audit: are we doing the right trades?
Joins MT5 deals with the orders ledger to measure, per trade:
entry drift vs the client's entry (the live-vs-backtest gap suspect),
score/stance mix, close reasons, and per-position round-trip P&L."""
import sys, sqlite3, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
import MetaTrader5 as mt5

mt5.initialize()
a = mt5.account_info()
print(f"account {a.login} | balance {a.balance:.2f} | equity {a.equity:.2f} | "
      f"margin level {a.margin_level:.0f}% | open {mt5.positions_total()}")
now = time.time()
deals = mt5.history_deals_get(now - 2 * 3600 - 3 * 3600, now + 3 * 3600) or []
rows = [{"ticket": d.ticket, "pos": d.position_id, "symbol": d.symbol,
         "type": d.type, "entry": d.entry, "vol": d.volume, "price": d.price,
         "profit": d.profit, "comment": (d.comment or "").strip(),
         "t": d.time} for d in deals if d.symbol]
df = pd.DataFrame(rows)
mt5.shutdown()
if not len(df):
    print("no deals in window"); sys.exit()
df = df[df["t"] >= now - 2 * 3600 + 0]      # server clock ~UTC+3 handled by wide pull
opens = df[df["entry"] == 0].copy()
closes = df[df["entry"] == 1].copy()
print(f"\nlast ~2h: opens {len(opens)} | closes {len(closes)} | "
      f"realized {closes['profit'].sum():+.2f}")
if len(closes):
    print("\nby close reason:")
    print(closes.groupby(closes["comment"].str.slice(0, 12)).agg(
        n=("profit", "size"), pnl=("profit", "sum"),
        win=("profit", lambda s: round((s > 0).mean(), 2))).to_string())
if len(opens):
    opens["stance"] = opens["comment"].str.slice(0, 2)
    opens["score"] = pd.to_numeric(
        opens["comment"].str.extract(r"s(\d\d)")[0], errors="coerce") / 100
    print("\nopens by symbol/side/stance:")
    opens["side"] = opens["type"].map({0: "buy", 1: "sell"})
    print(opens.groupby(["symbol", "side", "stance"]).agg(
        n=("vol", "size"), lots=("vol", "sum"),
        avg_score=("score", "mean")).to_string())

# entry drift vs client's entry, from the ledger
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
led = pd.read_sql_query(
    "SELECT * FROM vantage_orders WHERE created >= datetime('now', '-3 hours') "
    "ORDER BY id DESC", cx)
print(f"\nledger rows (3h): {len(led)} | cols incl: "
      f"{[c for c in led.columns if 'price' in c or 'entry' in c or 'wait' in c]}")
pcols = [c for c in led.columns if "price" in c]
if len(led) and len(pcols) >= 2:
    print(led[["symbol", "stance", "our_lots"] + pcols].head(15).to_string())
