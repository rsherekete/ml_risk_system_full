import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import clients_from_yaml, platform_for_database

pd.set_option("display.width", 220)

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)

for name, client in clients_from_yaml().items():
    platform = platform_for_database(name)
    print("=" * 90)
    print(name, f"({platform})")

    # --- table listing -----------------------------------------------------
    try:
        tables = client.tables()
        print("  tables:")
        print(tables.to_string(index=False))
    except Exception as exc:
        print(f"  tables() FAILED -> {exc}")

    if platform == "mt4":
        # --- ts vs tm divergence, and dual-filter vs tm-only comparison -----
        try:
            div = client.query("""
                SELECT
                    SUM(ABS(ts - UNIX_TIMESTAMP(tm)) > 1)     AS over_1s,
                    SUM(ABS(ts - UNIX_TIMESTAMP(tm)) > 60)    AS over_60s,
                    SUM(ABS(ts - UNIX_TIMESTAMP(tm)) > 86400) AS over_1day,
                    MAX(ABS(ts - UNIX_TIMESTAMP(tm)))          AS max_diff_seconds,
                    COUNT(*) AS n
                FROM traderecord
                WHERE tm >= %(start)s AND tm < %(end)s
            """, start=start.to_pydatetime(), end=end.to_pydatetime())
            print("  ts/tm divergence (tm-bounded, 90d):", div.iloc[0].to_dict())
        except Exception as exc:
            print(f"  divergence check FAILED -> {exc}")

        try:
            tm_only = client.query("""
                SELECT COUNT(DISTINCT login) AS distinct_logins_tm_only
                FROM traderecord
                WHERE tm >= %(start)s AND tm < %(end)s
            """, start=start.to_pydatetime(), end=end.to_pydatetime())
            dual = client.query("""
                SELECT COUNT(DISTINCT login) AS distinct_logins_dual_filter
                FROM traderecord
                WHERE tm >= %(start)s AND tm < %(end)s
                  AND ts >= %(start_ts)s AND ts < %(end_ts)s
            """, start=start.to_pydatetime(), end=end.to_pydatetime(),
                 start_ts=int(start.timestamp()) - 1, end_ts=int(end.timestamp()) + 1)
            print(f"  tm-only distinct logins: {tm_only.iloc[0]['distinct_logins_tm_only']}   "
                  f"dual-filter distinct logins: {dual.iloc[0]['distinct_logins_dual_filter']}")
        except Exception as exc:
            print(f"  tm-only vs dual comparison FAILED -> {exc}")

        # --- does a "users" table exist separate from userinfo? ------------
        try:
            users_count = client.query("SELECT COUNT(*) AS n FROM users")
            print(f"  users table EXISTS, row count: {users_count.iloc[0]['n']}")
        except Exception as exc:
            print(f"  no 'users' table (or inaccessible): {type(exc).__name__}: {exc}")

        # --- requestinfo table? ---------------------------------------------
        try:
            reqinfo = client.query(f"""
                SELECT COUNT(DISTINCT login) AS distinct_logins
                FROM requestinfo
                WHERE tm >= %(start)s AND tm < %(end)s
            """, start=start.to_pydatetime(), end=end.to_pydatetime())
            print(f"  requestinfo distinct logins (90d): {reqinfo.iloc[0]['distinct_logins']}")
        except Exception as exc:
            print(f"  no usable 'requestinfo' table: {type(exc).__name__}: {exc}")

        # --- tradetransinfo (broader request log) distinct logins ----------
        try:
            ttinfo = client.query("""
                SELECT COUNT(DISTINCT login) AS distinct_logins
                FROM tradetransinfo
                WHERE tm >= %(start)s AND tm < %(end)s
            """, start=start.to_pydatetime(), end=end.to_pydatetime())
            print(f"  tradetransinfo distinct logins (90d): {ttinfo.iloc[0]['distinct_logins']}")
        except Exception as exc:
            print(f"  tradetransinfo check FAILED -> {type(exc).__name__}: {exc}")

    else:
        # MT5: accounts table total (no activity filter at all)
        try:
            acct_total = client.query("SELECT COUNT(*) AS n FROM accounts")
            print(f"  accounts table total (NO activity filter): {acct_total.iloc[0]['n']}")
        except Exception as exc:
            print(f"  accounts count FAILED -> {exc}")

        # requestrecord distinct logins over the same window
        try:
            reqrec = client.query("""
                SELECT COUNT(DISTINCT login) AS distinct_logins
                FROM requestrecord
                WHERE ts >= %(start)s AND ts < %(end)s
            """, start=start.to_pydatetime(), end=end.to_pydatetime())
            print(f"  requestrecord distinct logins (90d): {reqrec.iloc[0]['distinct_logins']}")
        except Exception as exc:
            print(f"  requestrecord check FAILED -> {type(exc).__name__}: {exc}")

    # --- all-history distinct-login ceiling (no date filter) ---------------
    trade_table = "traderecord" if platform == "mt4" else "dealrecord"
    try:
        ceiling = client.query(f"SELECT COUNT(DISTINCT login) AS all_time_distinct_logins FROM {trade_table}")
        print(f"  ALL-TIME distinct logins in {trade_table} (no date filter): {ceiling.iloc[0]['all_time_distinct_logins']}")
    except Exception as exc:
        print(f"  all-time ceiling FAILED -> {exc}")

    print()
