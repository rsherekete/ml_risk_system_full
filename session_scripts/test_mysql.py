"""Is MySQL reachable, and can it serve two years cheaply?

MySQL is the right primary source: it is the system of record and costs nothing
per query, where BigQuery bills per byte scanned. This checks connectivity,
then measures how much history each server actually holds.
"""
import sys
import time

import yaml

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pymysql

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)

defaults = config["defaults"]
print(f"connecting as {defaults['user']} on port {defaults['port']}\n")

for name, server in config["servers"].items():
    t0 = time.time()
    try:
        connection = pymysql.connect(
            host=server["host"], port=defaults["port"], user=defaults["user"],
            password=defaults["password"], database=server["database"],
            connect_timeout=10, read_timeout=60)
    except Exception as error:
        print(f"{name:<20} FAILED {type(error).__name__}: {str(error)[:90]}")
        continue

    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            trade_table = next((t for t in ("orders", "deals", "traderecord", "dealrecord")
                                if t in tables), None)
            detail = ""
            if trade_table:
                # Row count and span: the two numbers that decide whether two
                # years is servable from here.
                cursor.execute(f"SELECT COUNT(*) FROM `{trade_table}`")
                rows = cursor.fetchone()[0]
                time_column = None
                cursor.execute(f"SHOW COLUMNS FROM `{trade_table}`")
                columns = [c[0] for c in cursor.fetchall()]
                for candidate in ("close_time", "CLOSE_TIME", "time", "TIME", "close_ts"):
                    if candidate in columns:
                        time_column = candidate
                        break
                if time_column:
                    cursor.execute(
                        f"SELECT MIN(`{time_column}`), MAX(`{time_column}`) FROM `{trade_table}`")
                    lo, hi = cursor.fetchone()
                    detail = f" | {trade_table}: {rows:,} rows, {lo} .. {hi}"
                else:
                    detail = f" | {trade_table}: {rows:,} rows"
        print(f"{name:<20} OK  {len(tables):>3} tables in {time.time() - t0:.1f}s{detail}")
    except Exception as error:
        print(f"{name:<20} query failed: {type(error).__name__}: {str(error)[:80]}")
    finally:
        connection.close()
