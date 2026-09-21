import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data import clients_from_yaml
from trading_data.research import platform_for_database

pd.set_option("display.width", 220)

clients = clients_from_yaml()
for name, client in clients.items():
    platform = platform_for_database(name)
    trade_table = "traderecord" if platform == "mt4" else "dealrecord"
    group_table = "userinfo" if platform == "mt4" else "accounts"
    print("=" * 80)
    print(name, f"({platform})")
    try:
        idx = client.query(f"SHOW INDEX FROM {group_table}")
        cols = idx[["Key_name", "Column_name", "Non_unique", "Cardinality"]] if "Key_name" in idx.columns else idx
        print(f"  {group_table} indexes:")
        print(cols.to_string(index=False))
    except Exception as exc:
        print(f"  SHOW INDEX FROM {group_table} FAILED -> {exc}")
    try:
        idx2 = client.query(f"SHOW INDEX FROM {trade_table}")
        cols2 = idx2[["Key_name", "Column_name", "Non_unique", "Cardinality"]] if "Key_name" in idx2.columns else idx2
        print(f"  {trade_table} indexes:")
        print(cols2.to_string(index=False))
    except Exception as exc:
        print(f"  SHOW INDEX FROM {trade_table} FAILED -> {exc}")
    print()
