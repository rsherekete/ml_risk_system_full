import sys, traceback
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")

print("=== import trading_data.behaviour_features ===")
try:
    from trading_data import behaviour_features as bf
    print("  OK:", [f for f in ("daily_behaviour_features","add_history_features",
          "add_lag_features","add_lifetime_features") if hasattr(bf, f)])
except Exception:
    traceback.print_exc()

print("\n=== research helpers ===")
try:
    from trading_data.research import is_realised_trade, realised_trade_pnl
    import inspect
    print(inspect.getsource(is_realised_trade)[:600])
    print(inspect.getsource(realised_trade_pnl)[:400])
except Exception:
    traceback.print_exc()

print("\n=== MT5 deals schema probe (position column?) ===")
try:
    from webapp import trade_feed as tfeed
    con = tfeed._connection("mt5_live01")
    with con.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM deals")
        cols = [r[0] for r in cur.fetchall()]
    print("  deals columns:", cols)
except Exception as e:
    print("  ERR:", type(e).__name__, e)

print("\n=== volume divisor check from BQ records ===")
import duckdb, pandas as pd
P = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
con2 = duckdb.connect()
df = con2.execute(f"SELECT platform, median(volume/volume_lots) AS divisor FROM read_parquet('{P}') WHERE volume_lots > 0 GROUP BY 1").df()
print(df)
