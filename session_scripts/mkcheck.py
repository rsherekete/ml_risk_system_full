import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import duckdb
from webapp import model_service as ms
con = duckdb.connect()
for name in ("markout_all_servers.parquet", "markout_trading.parquet"):
    p = ms.SCRATCH / name
    if p.exists():
        print(name, ":")
        for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p.as_posix()}')").fetchall():
            print("  ", r[0], r[1])
        print("  rows:", con.execute(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0])
        row = con.execute(f"SELECT * FROM read_parquet('{p.as_posix()}') LIMIT 1").df()
        for c in row.columns:
            print(f"    {c} = {row[c].iloc[0]!r}")
    else:
        print(name, "MISSING")
