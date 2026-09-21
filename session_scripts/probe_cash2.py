"""Cash-movement tables and region coverage, queried without full scans."""
import sys

import pymysql
import yaml

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)
defaults = config["defaults"]
host = config["servers"]["mt4_live01"]["host"]


def connect(database, timeout=120):
    return pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                           password=defaults["password"], database=database,
                           connect_timeout=20, read_timeout=timeout)


print("=" * 22, "mt4_live01 balance_ops")
connection = connect("mt4_live01")
with connection.cursor() as cursor:
    cursor.execute("SHOW COLUMNS FROM balance_ops")
    print("  columns:", [c[0] for c in cursor.fetchall()])
    cursor.execute("SELECT * FROM balance_ops ORDER BY 1 DESC LIMIT 4")
    for row in cursor.fetchall():
        print("   ", row)
    cursor.execute("SELECT COUNT(*) FROM balance_ops")
    print("  rows:", cursor.fetchone()[0])

    print("\n  region coverage on accounts:")
    cursor.execute("SELECT COUNT(*), SUM(country <> ''), COUNT(DISTINCT country) FROM accounts")
    total, with_country, distinct = cursor.fetchone()
    print(f"    {total:,} accounts | {with_country:,} with country "
          f"({with_country / max(1, total):.0%}) | {distinct} distinct")
    cursor.execute("SELECT country, COUNT(*) n FROM accounts WHERE country <> '' "
                   "GROUP BY country ORDER BY n DESC LIMIT 12")
    for row in cursor.fetchall():
        print("     ", row)
    cursor.execute("SELECT `group`, COUNT(*) n FROM accounts GROUP BY `group` "
                   "ORDER BY n DESC LIMIT 8")
    print("  groups:")
    for row in cursor.fetchall():
        print("     ", row)
connection.close()

print("\n" + "=" * 22, "mt5_live01")
connection = connect("mt5_live01")
with connection.cursor() as cursor:
    cursor.execute("SHOW TABLES")
    tables = [r[0] for r in cursor.fetchall()]
    print("  user/account tables:",
          [t for t in tables if any(k in t.lower() for k in ("user", "account", "client"))])
    for table in ("users", "accounts"):
        if table in tables:
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            columns = [c[0] for c in cursor.fetchall()]
            geo = [c for c in columns if any(k in c.lower() for k in
                   ("country", "city", "state", "zip", "group", "lead", "agent", "language"))]
            print(f"  {table} geo columns: {geo}")
            if geo:
                picks = ", ".join(f"`{c}`" for c in (["login"] + geo)[:7])
                cursor.execute(f"SELECT {picks} FROM `{table}` LIMIT 4")
                for row in cursor.fetchall():
                    print("     ", row)
    # MT5 keeps cash movements in `deals` with a non-trade action.
    cursor.execute("SELECT action, COUNT(*) n, SUM(profit) FROM deals "
                   "WHERE action >= 2 GROUP BY action ORDER BY n DESC LIMIT 8")
    print("  non-trade deal actions:")
    for row in cursor.fetchall():
        print("     ", row)
connection.close()
