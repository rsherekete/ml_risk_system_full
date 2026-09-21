"""Is there tick or bar history in MySQL? It would be free, unlike BigQuery."""
import sys

import pymysql
import yaml

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)
defaults = config["defaults"]
host = config["servers"]["mt4_live01"]["host"]


def connect(database, timeout=90):
    return pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                           password=defaults["password"], database=database,
                           connect_timeout=20, read_timeout=timeout)


for database in ("mt4_live01", "mt5_live01"):
    print(f"\n{'=' * 20} {database}")
    connection = connect(database)
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = [r[0] for r in cursor.fetchall()]
        candidates = [t for t in tables if any(k in t.lower() for k in
                      ("tick", "history", "chart", "bar", "price", "quote", "rate"))]
        print("  price-ish tables:", candidates)

        for table in candidates[:6]:
            try:
                cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                columns = [c[0] for c in cursor.fetchall()]
                print(f"\n  {table}: {columns[:14]}")
                cursor.execute(f"SELECT * FROM `{table}` LIMIT 3")
                for row in cursor.fetchall():
                    print("     ", str(row)[:190])
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                print(f"     rows: {cursor.fetchone()[0]:,}")
            except Exception as error:
                print(f"  {table}: {type(error).__name__}: {str(error)[:80]}")
    connection.close()

# Some deployments keep price history in a separate reporting schema.
print(f"\n{'=' * 20} other schemas")
connection = pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                             password=defaults["password"], connect_timeout=20, read_timeout=90)
with connection.cursor() as cursor:
    for schema in ("rep_general", "reporting", "mt4_svc"):
        try:
            cursor.execute(f"SHOW TABLES FROM `{schema}`")
            tables = [r[0] for r in cursor.fetchall()]
            hits = [t for t in tables if any(k in t.lower() for k in
                    ("tick", "history", "bar", "price", "quote", "rate", "chart"))]
            print(f"  {schema}: {len(tables)} tables | price-ish: {hits[:10]}")
        except Exception as error:
            print(f"  {schema}: {type(error).__name__}")
connection.close()
