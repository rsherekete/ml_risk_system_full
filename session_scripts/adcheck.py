import duckdb
P = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
con = duckdb.connect()
print("columns:")
for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{P}')").fetchall():
    print("  ", r[0], r[1])
print("\ncount/min/max timestamp:")
print(con.execute(f"SELECT count(*), min(timestamp), max(timestamp) FROM read_parquet('{P}')").fetchone())
print("states:", con.execute(f"SELECT state, count(*) FROM read_parquet('{P}') GROUP BY 1").fetchall())
print("sample row:")
row = con.execute(f"SELECT * FROM read_parquet('{P}') LIMIT 1").df()
for c in row.columns:
    print(f"  {c} = {row[c].iloc[0]!r}")
