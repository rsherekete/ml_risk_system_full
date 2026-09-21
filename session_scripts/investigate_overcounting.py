import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data import clients_from_yaml
from trading_data.research import platform_for_database

pd.set_option("display.width", 220)

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=10)
end = decision_day + pd.Timedelta(days=1)

for name, client in clients_from_yaml().items():
    platform = platform_for_database(name)
    print("=" * 90)
    print(name, f"({platform})", "-- last 10 days, all diagnostics from the SAME raw table")
    print("=" * 90)

    if platform == "mt4":
        table = "traderecord"
        # cmd: 0=buy,1=sell (real market/pending orders 0-5), 6=balance, 7=credit
        # state: 0-2=open, 3=closed,4=closed_part,5=closed_by (real), 6=deleted (never executed / cancelled)
        q = client.query(f"""
            SELECT
                COUNT(DISTINCT login) AS any_row_logins,
                COUNT(DISTINCT CASE WHEN cmd IN (0,1) THEN login END) AS buy_sell_cmd_logins,
                COUNT(DISTINCT CASE WHEN cmd IN (6,7) THEN login END) AS balance_credit_only_capable_logins,
                COUNT(DISTINCT CASE WHEN state IN (3,4,5) THEN login END) AS genuinely_closed_logins,
                COUNT(DISTINCT CASE WHEN cmd IN (0,1) AND state IN (3,4,5) THEN login END) AS real_closed_trade_logins,
                COUNT(DISTINCT CASE WHEN state = 6 THEN login END) AS deleted_pending_only_logins,
                COUNT(*) AS total_rows
            FROM {table}
            WHERE tm >= %(start)s AND tm < %(end)s
        """, start=start.to_pydatetime(), end=end.to_pydatetime())
    else:
        table = "dealrecord"
        # action: 0=buy,1=sell (real trade actions), 2=balance,3=credit, 4+=other money ops
        # entry: 0=IN(open),1=OUT(close),2=INOUT(reversal),3=OUT_BY(close by)
        q = client.query(f"""
            SELECT
                COUNT(DISTINCT login) AS any_row_logins,
                COUNT(DISTINCT CASE WHEN action IN (0,1) THEN login END) AS buy_sell_cmd_logins,
                COUNT(DISTINCT CASE WHEN action NOT IN (0,1) THEN login END) AS money_op_only_capable_logins,
                COUNT(DISTINCT CASE WHEN entry IN (1,3) THEN login END) AS genuinely_closed_logins,
                COUNT(DISTINCT CASE WHEN action IN (0,1) AND entry IN (1,3) THEN login END) AS real_closed_trade_logins,
                0 AS deleted_pending_only_logins,
                COUNT(*) AS total_rows
            FROM {table}
            WHERE ts >= %(start)s AND ts < %(end)s
        """, start=start.to_pydatetime(), end=end.to_pydatetime())

    print(q.iloc[0].to_dict())
    print()
