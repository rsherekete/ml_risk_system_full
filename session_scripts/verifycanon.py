import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp.trade_feed import _canonical
from webapp import vantage as V

tests = ["XAUUSD+","EURUSD+","GBPJPY+","XAUUSD","XAUUSDmin","NAS100.i","GER40.i",
         "SP500","DJ30","UKOUSD","HK50.i","INTC","AAPL","BTCUSD"]
print("=== canonical ===")
for s in tests:
    print(f"  {s:12s} -> {_canonical(s)}")

print("\n=== symbol_code resolution (was NaN for + symbols) ===")
codes = V._symbol_codes()
for s in ["XAUUSD+","EURUSD+","GBPJPY+","NAS100.i","SP500","DJ30","INTC"]:
    c = codes.get(s) or codes.get(_canonical(s))
    print(f"  {s:12s} code={c}")

print("\n=== bar unification via seed ===")
V._seed_bars()
gold_keys = [k for k in V._BAR_HISTORY if "XAU" in k.upper()]
print("  gold bar keys after seed:", gold_keys)
ctx = V._ctx_features("XAUUSD+", 1)
print("  _ctx_features('XAUUSD+') keys:", sorted(ctx.keys())[:6], "..." if ctx else "EMPTY")
print("  ctx populated?", "YES" if ctx else "NO (still cold)")
