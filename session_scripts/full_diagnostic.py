r"""
Ready-to-run diagnostic for the 8,000-12,000-accounts-per-server discrepancy.

BLOCKED IN THIS SESSION: this sandbox has no network route to
ld4-dbproxy.in.zfx.loc (internal-only DNS domain; the only VPN adapter on
this machine -- "Fortinet SSL VPN Virtual Ethernet Adapter" -- is Disabled).
Every client.query() call below will raise
pymysql.err.OperationalError (2003, Can't connect ...) until that VPN is
connected. This script is otherwise complete and safe to run as-is once
connectivity exists (uses the same PK-scoped join pattern as
get_trade_requests()/account_group_lookup() to avoid the known
full-userinfo-join hang).
"""
import sys; sys.path.insert(0, r'c:\Users\RoyVivasi\Documents\notebook')

import calendar
import pandas as pd

from trading_data.research import clients_from_yaml, platform_for_database

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)
start_ts = calendar.timegm(start.to_pydatetime().timetuple())
end_ts = calendar.timegm(end.to_pydatetime().timetuple())

HINT = "/*+ MAX_EXECUTION_TIME(120000) */"  # cap any one query at 120s so a runaway
                                             # full scan fails fast instead of hanging

clients = clients_from_yaml()
report_rows = []

for db, client in clients.items():
    platform = platform_for_database(db)
    print(f"\n===== {db} ({platform}) =====")

    # (1) full table list
    try:
        tables = client.tables()
    except Exception as e:
        print(f"  ERROR listing tables: {e}")
        report_rows.append({"database": db, "check": "tables()", "error": str(e)})
        continue
    names = set(tables["table_name"])
    print(tables.sort_values("table_name").to_string(index=False))

    def row_count_ceiling(table):
        try:
            df = client.query(f"SELECT {HINT} COUNT(*) AS n FROM {table}")
            return int(df["n"].iloc[0])
        except Exception as e:
            return f"ERROR: {e}"

    def distinct_login_windowed(table, login_col, ts_col, extra_where=""):
        try:
            sql = (
                f"SELECT {HINT} COUNT(DISTINCT {login_col}) AS n FROM {table} "
                f"WHERE {ts_col} >= %(start_ts)s AND {ts_col} < %(end_ts)s {extra_where}"
            )
            df = client.query(sql, start_ts=start_ts, end_ts=end_ts)
            return int(df["n"].iloc[0])
        except Exception as e:
            return f"ERROR: {e}"

    def distinct_login_all_history(table, login_col):
        try:
            df = client.query(f"SELECT {HINT} COUNT(DISTINCT {login_col}) AS n FROM {table}")
            return int(df["n"].iloc[0])
        except Exception as e:
            return f"ERROR: {e}"

    if platform == "mt4":
        # (2) true one-row-per-account registry, if it exists
        if "users" in names:
            users_rows = row_count_ceiling("users")
            print(f"  users table total rows: {users_rows}")
            report_rows.append({"database": db, "check": "users_total_rows", "value": users_rows})
        else:
            print("  no 'users' table found (registry candidates seen: "
                  f"{sorted(n for n in names if 'user' in n.lower())})")
            report_rows.append({"database": db, "check": "users_table_present", "value": False})

        # (4) tradetransinfo, broader "any request" definition, 90-day window.
        # tradetransinfo itself carries no login column -- login only exists on
        # the userinfo row written in the SAME ARS transaction (ts, sequence is
        # userinfo's PK), so this mirrors get_trade_requests()'s join exactly:
        # both sides are bounded by the same ts range and joined on the PK
        # (ts, sequence), never a bare login/date cross-join against all of
        # userinfo -- this is the safe pattern, not the banned one.
        try:
            sql = f"""
                SELECT {HINT} COUNT(DISTINCT u.login) AS n
                FROM tradetransinfo x
                JOIN userinfo u
                  ON u.ts = x.ts AND u.sequence = x.sequence
                 AND u.ts >= %(start_ts)s AND u.ts < %(end_ts)s
                WHERE x.ts >= %(start_ts)s AND x.ts < %(end_ts)s
            """
            df = client.query(sql, start_ts=start_ts, end_ts=end_ts)
            tti_logins = int(df["n"].iloc[0])
        except Exception as e:
            tti_logins = f"ERROR: {e}"
        print(f"  tradetransinfo distinct logins (90d, PK-joined to userinfo): {tti_logins}")
        report_rows.append({"database": db, "check": "tradetransinfo_distinct_login_90d", "value": tti_logins})

        # requestinfo -- the request-queue table (DC_* status, dealer, gw fields);
        # only probe it if it actually exists on this server.
        if "requestinfo" in names:
            cols = client.columns("requestinfo")
            login_col = "login" if "login" in set(cols["column_name"]) else None
            ts_col = "tm" if "tm" in set(cols["column_name"]) else ("ts" if "ts" in set(cols["column_name"]) else None)
            print(f"  requestinfo columns: {list(cols['column_name'])}")
            if login_col and ts_col:
                val = distinct_login_windowed("requestinfo", login_col, ts_col)
                print(f"  requestinfo distinct logins (90d): {val}")
                report_rows.append({"database": db, "check": "requestinfo_distinct_login_90d", "value": val})
        else:
            print("  no 'requestinfo' table on this server")

        # (5) traderecord ceiling, no date filter at all
        tr_all = distinct_login_all_history("traderecord", "login")
        print(f"  traderecord distinct logins (ALL history, no filter): {tr_all}")
        report_rows.append({"database": db, "check": "traderecord_distinct_login_all_time", "value": tr_all})

    else:  # mt5
        # (3) accounts table, one-row-per-login, NO activity filter at all
        accounts_rows = row_count_ceiling("accounts")
        print(f"  accounts table total rows (no activity filter): {accounts_rows}")
        report_rows.append({"database": db, "check": "accounts_total_rows", "value": accounts_rows})

        # (4)-equivalent: requestrecord, every request attempt, 90-day window
        # (drop opcode=1 filter here on purpose -- we want the broadest
        # "any request" definition, not just inserts, to see if that's what
        # the other tool is counting)
        req_logins = distinct_login_windowed("requestrecord", "login", "ts")
        print(f"  requestrecord distinct logins (90d, ANY opcode): {req_logins}")
        report_rows.append({"database": db, "check": "requestrecord_distinct_login_90d_any_opcode", "value": req_logins})

        # (5) dealrecord ceiling, no date filter at all
        dr_all = distinct_login_all_history("dealrecord", "login")
        print(f"  dealrecord distinct logins (ALL history, no filter): {dr_all}")
        report_rows.append({"database": db, "check": "dealrecord_distinct_login_all_time", "value": dr_all})

print("\n\n===== SUMMARY =====")
summary = pd.DataFrame(report_rows)
print(summary.to_string(index=False))
summary.to_csv(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\diagnostic_summary.csv", index=False)
