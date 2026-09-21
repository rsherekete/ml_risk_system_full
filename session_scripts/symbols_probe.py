"""What contract metadata do the servers publish, and can we USD-value everything?"""
import sys

import pandas as pd
import pymysql
import yaml

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)
defaults = config["defaults"]
host = config["servers"]["mt4_live01"]["host"]

for database in ("mt5_live01", "mt4_live01"):
    connection = pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                                 password=defaults["password"], database=database,
                                 connect_timeout=20, read_timeout=120)
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES LIKE '%symbol%'")
        print(f"\n{database} symbol tables:", [r[0] for r in cursor.fetchall()])
        try:
            cursor.execute("SHOW COLUMNS FROM symbols")
            columns = [c[0] for c in cursor.fetchall()]
            print("  columns:", columns)
            wanted = [c for c in columns if any(k in c.lower() for k in
                      ("symbol", "contract", "currency", "tick", "digits", "profit", "margin"))]
            cursor.execute(f"SELECT {', '.join('`' + c + '`' for c in wanted[:12])} FROM symbols LIMIT 6")
            for row in cursor.fetchall():
                print("   ", row)
        except Exception as error:
            print(f"  symbols unavailable: {type(error).__name__}: {str(error)[:100]}")
    connection.close()
