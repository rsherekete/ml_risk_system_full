import sys, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import model_service as ms
from webapp.trade_features import _AD_DIR

ACCT = "mt5_live01:105036806"
LOGIN = 105036806

# 1) live MySQL — is it trading, and how recently?
from webapp import trade_feed as tfeed
con = tfeed._connection("mt5_live01")
with con.cursor() as cur:
    cur.execute("SELECT COUNT(*), MIN(`time`), MAX(`time`) FROM deals WHERE login=%s AND action IN (0,1)", [LOGIN])
    print("MySQL deals:", cur.fetchone())

# 2) AD corpus (behaviour panel source)
adf = pd.read_parquet(_AD_DIR / "model_frame.parquet", columns=["account_key", "decision_day"])
mine = adf[adf["account_key"] == ACCT]
print("AD corpus rows:", len(mine), "| days:", (mine['decision_day'].min(), mine['decision_day'].max()) if len(mine) else "-")

# 3) quant scores frame (per-trade)
qt = ms.load_scores(ms.VIEW_QUANT)
print("QUANT frame has account:", (qt['account_key'] == ACCT).sum() if qt is not None else "no frame")
if qt is not None:
    print("  quant day range:", pd.to_datetime(qt['day']).min(), "->", pd.to_datetime(qt['day']).max())
    print("  quant mt5_live01 accounts:", qt[qt['account_key'].str.startswith('mt5_live01')]['account_key'].nunique())

# 4) trading scores frame
tr = ms.load_scores(ms.VIEW_TRADING)
print("TRADING frame has account:", (tr['account_key'] == ACCT).sum() if tr is not None else "no frame")
if tr is not None:
    print("  trading day range:", pd.to_datetime(tr['day']).min(), "->", pd.to_datetime(tr['day']).max())
