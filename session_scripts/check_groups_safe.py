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
    trade_table = "traderecord" if platform == "mt4" else "dealrecord"
    time_col = "tm" if platform == "mt4" else "ts"
    group_table = "userinfo" if platform == "mt4" else "accounts"

    print("=" * 90)
    print(name, f"({platform})")

    # Step 1: small distinct-login list from the trade table only (no join).
    logins = client.query(f"""
        SELECT DISTINCT login FROM {trade_table}
        WHERE {time_col} >= %(start)s AND {time_col} < %(end)s
    """, start=start.to_pydatetime(), end=end.to_pydatetime())
    login_list = logins["login"].dropna().astype(int).tolist()
    print(f"  distinct logins active in window: {len(login_list)}")
    if not login_list:
        print()
        continue

    # Step 2: bounded IN-list lookup against the (large) account/group table --
    # indexed point lookups on a small list, never a full join over both big tables.
    chunk = login_list[:5000]
    placeholders = ",".join(str(x) for x in chunk)
    try:
        groups = client.query(f"""
            SELECT `group` AS acct_group, COUNT(*) AS n
            FROM {group_table}
            WHERE login IN ({placeholders})
            GROUP BY `group`
            ORDER BY n DESC
        """)
        print(f"  group breakdown ({group_table}, {len(chunk)} logins looked up):")
        print(groups.to_string(index=False))
        matched = int(groups["n"].sum())
        print(f"  matched rows in {group_table}: {matched} / {len(chunk)} logins looked up")
    except Exception as exc:
        print(f"  group lookup FAILED -> {type(exc).__name__}: {exc}")
    print()
