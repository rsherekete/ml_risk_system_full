import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
from trading_data import clients_from_yaml
from trading_data.research import platform_for_database

for name, client in clients_from_yaml().items():
    platform = platform_for_database(name)
    if platform != "mt4":
        continue
    print("=" * 70)
    print(name)
    try:
        r = client.query("SELECT COUNT(*) AS total_rows, COUNT(DISTINCT login) AS distinct_logins FROM userinfo")
        print(r.iloc[0].to_dict())
    except Exception as exc:
        print(f"FAILED -> {exc}")
