"""What contract size does each server really publish for gold?

A 100x error on the largest exposure in the book is not something to infer, so
this reads the raw rows and cross-checks against tick_value, which encodes the
same information independently: for XAUUSD quoted to 2 decimals, tick_value
should equal contract_size x tick_size.
"""
import sys

import pandas as pd
import pymysql
import yaml

with open(r"c:\Users\RoyVivasi\Documents\notebook\server.yaml") as stream:
    config = yaml.safe_load(stream)
defaults = config["defaults"]
host = config["servers"]["mt4_live01"]["host"]

for database in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04", "mt5_live01"):
    is_mt5 = database.startswith("mt5")
    connection = pymysql.connect(host=host, port=defaults["port"], user=defaults["user"],
                                 password=defaults["password"], database=database,
                                 connect_timeout=20, read_timeout=120)
    columns = ("symbol, contract_size, tick_value, tick_size, digits, currency_base, currency_profit"
               if is_mt5 else
               "symbol, contract_size, tick_value, tick_size, digits, currency")
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {columns} FROM symbols WHERE symbol LIKE 'XAUUSD%' "
                       f"OR symbol LIKE 'EURUSD%' ORDER BY symbol")
        rows = cursor.fetchall()
    connection.close()
    print(f"\n{database}")
    for row in rows[:10]:
        print("   ", row)
