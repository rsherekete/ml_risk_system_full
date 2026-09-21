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

TEST_DEMO_MARKERS = ("test", "demo")

for name, client in clients_from_yaml().items():
    platform = platform_for_database(name)
    trade_table = "traderecord" if platform == "mt4" else "dealrecord"
    time_col = "tm" if platform == "mt4" else "ts"
    group_table = "userinfo" if platform == "mt4" else "accounts"

    print("=" * 90)
    print(name, f"({platform})")

    logins = client.query(f"""
        SELECT DISTINCT login FROM {trade_table}
        WHERE {time_col} >= %(start)s AND {time_col} < %(end)s
    """, start=start.to_pydatetime(), end=end.to_pydatetime())
    login_list = logins["login"].dropna().astype(int).tolist()
    total_distinct = len(login_list)
    print(f"  distinct logins active in window: {total_distinct}")
    if not login_list:
        print()
        continue

    chunk = login_list[:5000]
    placeholders = ",".join(str(x) for x in chunk)
    try:
        rows = client.query(f"""
            SELECT DISTINCT login, `group` AS acct_group
            FROM {group_table}
            WHERE login IN ({placeholders})
        """)
    except Exception as exc:
        print(f"  group lookup FAILED -> {type(exc).__name__}: {exc}")
        print()
        continue

    # a login can carry >1 group historically (mt4 userinfo snapshots) -- take
    # the most-recently-seen group per login isn't available cheaply here, so
    # instead just report: of the distinct logins, how many have EVERY known
    # group tagged as test/demo (unambiguous) vs at least one non-test/demo group.
    rows["is_test_demo"] = rows["acct_group"].str.lower().str.contains("|".join(TEST_DEMO_MARKERS), na=False)
    by_login = rows.groupby("login")["is_test_demo"].agg(["all", "any"])
    unambiguous_test_demo = int(by_login["all"].sum())
    ever_test_demo = int(by_login["any"].sum())
    matched_logins = rows["login"].nunique()
    print(f"  logins matched in {group_table}: {matched_logins} / {total_distinct}")
    print(f"  logins whose group is ALWAYS test/demo-tagged: {unambiguous_test_demo}")
    print(f"  logins with ANY test/demo-tagged group in history: {ever_test_demo}")
    print()
