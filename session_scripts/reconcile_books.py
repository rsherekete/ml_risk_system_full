import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import model_service as ms

tr = ms.load_scores(ms.VIEW_TRADING)
qt = ms.load_scores(ms.VIEW_QUANT)
print("TRADING frame:", None if tr is None else f"{len(tr):,} rows")
print("  cols:", sorted(tr.columns)[:25] if tr is not None else "-")
print("QUANT frame:", None if qt is None else f"{len(qt):,} rows")
print("  cols:", sorted(qt.columns)[:25] if qt is not None else "-")

if tr is None or qt is None:
    sys.exit()
tr = tr.copy(); qt = qt.copy()
tr["day"] = pd.to_datetime(tr["day"]).dt.normalize()
qt["day"] = pd.to_datetime(qt["day"]).dt.normalize()
lo = max(tr["day"].min(), qt["day"].min())
hi = min(tr["day"].max(), qt["day"].max())
print(f"\ncommon window: {lo.date()} -> {hi.date()}")
trw = tr[(tr["day"] >= lo) & (tr["day"] <= hi)]
qtw = qt[(qt["day"] >= lo) & (qt["day"] <= hi)]

# candidate pnl columns
tr_pnl = next((c for c in ("pnl", "day_pnl", "client_pnl", "net_profit") if c in trw.columns), None)
qt_pnl = next((c for c in ("pnl", "net_profit") if c in qtw.columns), None)
print(f"trading pnl col: {tr_pnl} | quant pnl col: {qt_pnl}")
ts = pd.to_numeric(trw[tr_pnl], errors="coerce").sum()
qs = pd.to_numeric(qtw[qt_pnl], errors="coerce").sum()
print(f"\nTRADING (account-day) client P&L sum: ${ts:,.0f}")
print(f"QUANT   (per-trade)   client P&L sum: ${qs:,.0f}")
print(f"difference: ${ts-qs:,.0f}  ({(ts-qs)/max(abs(qs),1):+.1%})")

# per-day comparison of the worst 6 mismatch days
td = pd.to_numeric(trw[tr_pnl], errors="coerce").groupby(trw["day"]).sum()
qd = pd.to_numeric(qtw[qt_pnl], errors="coerce").groupby(qtw["day"]).sum()
joined = pd.DataFrame({"trading": td, "quant": qd}).dropna()
joined["diff"] = joined["trading"] - joined["quant"]
print("\nworst mismatch days:")
print(joined.reindex(joined["diff"].abs().sort_values(ascending=False).index).head(6).round(0))
# account universes
print(f"\naccounts: trading {trw['account_key'].nunique():,} | quant {qtw['account_key'].nunique():,}")
common = set(trw['account_key'].unique()) & set(qtw['account_key'].unique())
print(f"common accounts: {len(common):,}")
# databases covered
if "database" in trw.columns and "database" in qtw.columns:
    print("trading dbs:", sorted(trw['database'].astype(str).unique()))
    print("quant   dbs:", sorted(qtw['database'].astype(str).unique()))
