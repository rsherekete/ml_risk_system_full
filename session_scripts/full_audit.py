import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data import clients_from_yaml
from trading_data.research import platform_for_database

pd.set_option("display.width", 200)
print("=" * 80)
print("1. CONNECTION CHECK")
print("=" * 80)
clients = clients_from_yaml()
for name, client in clients.items():
    try:
        r = client.ping()
        print(f"  {name}: OK, server_time={r.iloc[0]['server_time']}")
    except Exception as exc:
        print(f"  {name}: FAILED -> {type(exc).__name__}: {exc}")

print()
print("=" * 80)
print("2. ACTUAL GRANTED PERMISSIONS (SHOW GRANTS)")
print("=" * 80)
for name, client in clients.items():
    try:
        grants = client.query("SHOW GRANTS")
        print(f"  {name}:")
        for _, row in grants.iterrows():
            print(f"    {row.iloc[0]}")
    except Exception as exc:
        print(f"  {name}: FAILED -> {type(exc).__name__}: {exc}")

print()
print("=" * 80)
print("3. RAW TABLE-LEVEL COUNTS (bypasses our client's date-window logic entirely)")
print("=" * 80)
for name, client in clients.items():
    platform = platform_for_database(name)
    table = "traderecord" if platform == "mt4" else "dealrecord"
    try:
        stats = client.query(
            f"SELECT COUNT(*) AS total_rows, COUNT(DISTINCT login) AS distinct_logins, "
            f"MIN(tm) AS min_tm, MAX(tm) AS max_tm FROM {table}"
        ) if platform == "mt4" else client.query(
            f"SELECT COUNT(*) AS total_rows, COUNT(DISTINCT login) AS distinct_logins, "
            f"MIN(ts) AS min_ts, MAX(ts) AS max_ts FROM {table}"
        )
        print(f"  {name} ({table}):")
        print(f"    {stats.iloc[0].to_dict()}")
    except Exception as exc:
        print(f"  {name}: FAILED -> {type(exc).__name__}: {exc}")

print()
print("=" * 80)
print("4. USERINFO / ACCOUNTS TABLE -- total registered client count per server")
print("=" * 80)
for name, client in clients.items():
    platform = platform_for_database(name)
    table = "userinfo" if platform == "mt4" else "accounts"
    try:
        stats = client.query(f"SELECT COUNT(*) AS total_accounts FROM {table}")
        print(f"  {name} ({table}): {stats.iloc[0]['total_accounts']} total registered accounts")
    except Exception as exc:
        print(f"  {name}: FAILED -> {type(exc).__name__}: {exc}")

print()
print("=" * 80)
print("5. YESTERDAY (now fully closed) vs TODAY (brand new) -- via our normal client path")
print("=" * 80)
decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
yesterday = decision_day - pd.Timedelta(days=1)
for name, client in clients.items():
    try:
        y = client.get_trade_records(yesterday.to_pydatetime(), decision_day.to_pydatetime())
        t = client.get_trade_records(decision_day.to_pydatetime(), (decision_day + pd.Timedelta(days=1)).to_pydatetime())
        print(f"  {name}: yesterday={len(y)} rows/{y['login'].nunique() if not y.empty else 0} accounts, "
              f"today-so-far={len(t)} rows/{t['login'].nunique() if not t.empty else 0} accounts")
    except Exception as exc:
        print(f"  {name}: FAILED -> {type(exc).__name__}: {exc}")
