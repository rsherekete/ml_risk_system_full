"""Where do client region and cash movements live?

Two additions depend on this: breaking every figure down by client region rather
than by server (a server is an operational artefact, a region is a business
one), and a deposits/withdrawals study that can also feed the models.
"""
import sys

import pymysql
import yaml

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)
defaults = config["defaults"]
host = config["servers"]["mt4_live01"]["host"]


def connect(database):
    return pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                           password=defaults["password"], database=database,
                           connect_timeout=20, read_timeout=180)


for database in ("mt4_live01", "mt5_live01"):
    print(f"\n{'=' * 20} {database}")
    connection = connect(database)
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = [r[0] for r in cursor.fetchall()]
        candidates = [t for t in tables if any(k in t.lower() for k in
                      ("user", "account", "client", "group", "balance", "trans"))]
        print("candidate tables:", candidates[:18])

        for table in ("users", "accounts"):
            if table not in tables:
                continue
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            columns = [c[0] for c in cursor.fetchall()]
            geo = [c for c in columns if any(k in c.lower() for k in
                   ("country", "city", "zip", "address", "phone", "lead", "group",
                    "agent", "currency", "language", "state"))]
            print(f"\n  {table}: {len(columns)} columns")
            print(f"    geo/segmentation: {geo}")
            if geo:
                picks = ", ".join(f"`{c}`" for c in (["login"] + geo)[:8])
                cursor.execute(f"SELECT {picks} FROM `{table}` LIMIT 5")
                for row in cursor.fetchall():
                    print("     ", row)

        # --- cash movements ---
        print("\n  cash movements:")
        if database.startswith("mt5"):
            cursor.execute("SELECT action, entry, COUNT(*), SUM(profit) "
                           "FROM deals WHERE action NOT IN (0,1) GROUP BY 1,2 "
                           "ORDER BY 3 DESC LIMIT 8")
            for row in cursor.fetchall():
                print("     action/entry/count/sum:", row)
        else:
            cursor.execute("SELECT cmd, COUNT(*), SUM(profit) FROM orders "
                           "GROUP BY cmd ORDER BY 2 DESC LIMIT 8")
            for row in cursor.fetchall():
                print("     cmd/count/sum:", row)
            cursor.execute("SELECT `order`, login, cmd, profit, comment, open_ts "
                           "FROM orders WHERE cmd = 6 ORDER BY open_ts DESC LIMIT 4")
            for row in cursor.fetchall():
                print("     balance op:", row)
    connection.close()
