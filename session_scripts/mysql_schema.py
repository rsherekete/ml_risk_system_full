"""What databases exist, and what shape are the trade tables?

Two of the eight configured names do not resolve, and MT5 exposes an `orders`
table as well as `deals` -- both need settling before an extractor is written.
"""
import sys

import pymysql
import yaml

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)
defaults = config["defaults"]
host = config["servers"]["mt4_live01"]["host"]

connection = pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                             password=defaults["password"], connect_timeout=15, read_timeout=120)
with connection.cursor() as cursor:
    cursor.execute("SHOW DATABASES")
    databases = [row[0] for row in cursor.fetchall()]
print(f"{len(databases)} databases visible:")
for name in databases:
    print("  ", name)
connection.close()

# MT5 keeps deals and positions separately; find which table carries realised
# P&L with a usable entry timestamp.
connection = pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                             password=defaults["password"], database="mt5_live01",
                             connect_timeout=15, read_timeout=180)
with connection.cursor() as cursor:
    cursor.execute("SHOW TABLES")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"\nmt5_live01 tables ({len(tables)}):")
    print("  ", [t for t in tables if any(k in t.lower()
                                          for k in ("deal", "order", "position", "trade"))])
    for table in ("deals", "orders", "positions"):
        if table not in tables:
            continue
        cursor.execute(f"SHOW COLUMNS FROM `{table}`")
        columns = [c[0] for c in cursor.fetchall()]
        print(f"\n  {table}: {columns[:22]}")

with connection.cursor() as cursor:
    cursor.execute("SHOW COLUMNS FROM `orders`")
    print("\nmt4-style order columns on mt4_live01:")
connection.close()

connection = pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                             password=defaults["password"], database="mt4_live01",
                             connect_timeout=15, read_timeout=180)
with connection.cursor() as cursor:
    cursor.execute("SHOW COLUMNS FROM `orders`")
    print("  ", [c[0] for c in cursor.fetchall()])
    # MT4 stores times as unix ints; confirm the real span excluding sentinels.
    cursor.execute("SELECT MIN(CLOSE_TIME), MAX(CLOSE_TIME) FROM orders "
                   "WHERE CLOSE_TIME > 0 AND CLOSE_TIME < 2147483647")
    print("  close_time span (unix):", cursor.fetchone())
connection.close()
