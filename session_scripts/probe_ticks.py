"""Can the MySQL ticks table be queried fast enough for a path-aware backtest?

The table is huge, so everything depends on the indexes. A query that cannot use
one will scan billions of rows and time out -- which is what happened to the
naive COUNT(*).
"""
import sys
import time

import pymysql
import yaml

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)
defaults = config["defaults"]
host = config["servers"]["mt4_live01"]["host"]

connection = pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                             password=defaults["password"], database="mt4_live01",
                             connect_timeout=20, read_timeout=240)

with connection.cursor() as cursor:
    print("indexes on ticks:")
    cursor.execute("SHOW INDEX FROM ticks")
    for row in cursor.fetchall():
        print(f"  {row[2]:<24} seq={row[3]} column={row[4]}")

    print("\ntable size:")
    cursor.execute("""SELECT table_rows, ROUND(data_length/1024/1024/1024,1) AS data_gb,
                             ROUND(index_length/1024/1024/1024,1) AS index_gb
                      FROM information_schema.tables
                      WHERE table_schema='mt4_live01' AND table_name='ticks'""")
    print("  approx rows / data GB / index GB:", cursor.fetchone())

    print("\nrange of tm (indexed lookup):")
    for sql, label in (("SELECT MIN(tm) FROM ticks", "earliest"),
                       ("SELECT MAX(tm) FROM ticks", "latest")):
        t0 = time.time()
        cursor.execute(sql)
        print(f"  {label}: {cursor.fetchone()[0]}  [{time.time()-t0:.1f}s]")

    # The query shape the backtest needs: one symbol over one trade's lifetime.
    print("\ntargeted window queries:")
    for window, label in ((("2026-08-25 10:00:00", "2026-08-25 11:00:00"), "1 hour"),
                          (("2026-08-25 00:00:00", "2026-08-26 00:00:00"), "1 day")):
        t0 = time.time()
        cursor.execute("SELECT COUNT(*), MIN(bid), MAX(ask) FROM ticks "
                       "WHERE symbol_name = 'XAUUSD' AND tm >= %s AND tm < %s", window)
        count, low, high = cursor.fetchone()
        print(f"  XAUUSD {label:<7}: {count:>9,} ticks, range {low}..{high}  "
              f"[{time.time()-t0:.1f}s]")

    print("\nEXPLAIN for the window query:")
    cursor.execute("EXPLAIN SELECT tm, bid, ask FROM ticks WHERE symbol_name='XAUUSD' "
                   "AND tm >= '2026-08-25 10:00:00' AND tm < '2026-08-25 11:00:00'")
    for row in cursor.fetchall():
        print("  ", row)

connection.close()
