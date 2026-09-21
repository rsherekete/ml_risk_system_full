"""Is BigQuery reachable through Google's VPC-SC restricted VIP from inside
the corporate VPN (the public endpoint answers 403 at the edge)? Try DNS,
a raw HTTPS GET, then the client with api_endpoint overridden; on success
search INFORMATION_SCHEMA for rebate_payout / primary_trading_account_number."""
import socket, ssl, sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
for host in ("restricted.googleapis.com", "private.googleapis.com", "bigquery.googleapis.com"):
    try:
        ip = socket.gethostbyname(host)
        print(f"{host} -> {ip}")
        s = socket.create_connection((ip, 443), timeout=6)
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(s, server_hostname=host) as ss:
            ss.sendall(b"GET /bigquery/v2/projects/zfx-dwh-prod/datasets HTTP/1.1\r\nHost: " + host.encode() + b"\r\nConnection: close\r\n\r\n")
            head = ss.recv(300)
            print("   ", head.split(b"\r\n")[0])
    except Exception as e:
        print(f"{host}: {type(e).__name__}: {e}")

for endpoint in ("https://restricted.googleapis.com", "https://private.googleapis.com"):
    try:
        from google.cloud import bigquery
        from google.api_core.client_options import ClientOptions
        client = bigquery.Client(project="zfx-dwh-prod", client_options=ClientOptions(api_endpoint=endpoint))
        t0 = time.time()
        dss = [d.dataset_id for d in client.list_datasets(timeout=30)]
        print(f"\n{endpoint}: datasets ({time.time()-t0:.1f}s):", dss)
        hits = []
        for ds in dss:
            try:
                sql = (f"SELECT table_name, column_name, data_type FROM `zfx-dwh-prod.{ds}.INFORMATION_SCHEMA.COLUMNS` "
                       f"WHERE REGEXP_CONTAINS(LOWER(column_name), r'rebate|primary_trading|trading_account_number|sub_account') LIMIT 300")
                for r in client.query(sql).result(timeout=120):
                    hits.append((ds, r.table_name, r.column_name, r.data_type))
            except Exception as e:
                print(f"  [{ds}] {type(e).__name__}: {str(e)[:120]}")
        for h in hits:
            print("  ", h)
        break
    except Exception as e:
        print(f"\n{endpoint}: {type(e).__name__}: {str(e)[:300]}")
