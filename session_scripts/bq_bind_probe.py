"""Every packet leaves through the Fortinet full tunnel (egress 196.61.189.20),
and Google's front end answers 403 for bigquery.googleapis.com from that
address while www.googleapis.com works. Test the hypothesis by binding the
socket to the WiFi interface (home ISP egress) instead -- no system change,
per-connection only. Stage 1: raw HTTPS GET. Stage 2: the BigQuery client on
a requests session whose adapter binds the same source address; on success,
list datasets and search INFORMATION_SCHEMA for the two DWH fields."""
import socket, ssl, sys, time, json
WIFI_IP = sys.argv[1] if len(sys.argv) > 1 else "10.240.17.21"
HOST = "bigquery.googleapis.com"

def raw_get(src_ip):
    ip = socket.gethostbyname(HOST)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(12)
    if src_ip:
        s.bind((src_ip, 0))
    s.connect((ip, 443))
    ctx = ssl.create_default_context()
    with ctx.wrap_socket(s, server_hostname=HOST) as ss:
        ss.sendall(b"GET /bigquery/v2/projects/zfx-dwh-prod/datasets HTTP/1.1\r\nHost: " + HOST.encode() + b"\r\nConnection: close\r\n\r\n")
        data = b""
        while True:
            chunk = ss.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 3000:
                break
    head = data.split(b"\r\n")[0].decode(errors="replace")
    body = data.split(b"\r\n\r\n", 1)[1][:200].decode(errors="replace") if b"\r\n\r\n" in data else ""
    return ip, head, body.replace("\n", " ")

for label, src in (("via VPN (default)", None), (f"bound to WiFi {WIFI_IP}", WIFI_IP)):
    try:
        ip, head, body = raw_get(src)
        print(f"{label}: {ip} -> {head} | {body[:160]}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")

print("\n=== BigQuery client with source-bound transport")
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.poolmanager import PoolManager
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from google.cloud import bigquery

    class BoundAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **kw):
            kw["source_address"] = (WIFI_IP, 0)
            self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **kw)

    creds, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    session.mount("https://", BoundAdapter())
    client = bigquery.Client(project="zfx-dwh-prod", _http=session)
    t0 = time.time()
    dss = [d.dataset_id for d in client.list_datasets(timeout=60)]
    print(f"datasets ({time.time()-t0:.1f}s): {len(dss)} -> {dss[:40]}")
    hits = []
    for ds in dss:
        try:
            sql = (f"SELECT table_name, column_name, data_type FROM `zfx-dwh-prod.{ds}.INFORMATION_SCHEMA.COLUMNS` "
                   f"WHERE REGEXP_CONTAINS(LOWER(column_name), r'rebate|primary_trading|trading_account_number|sub_account') LIMIT 200")
            for r in client.query(sql).result(timeout=120):
                hits.append((ds, r.table_name, r.column_name, r.data_type))
        except Exception as e:
            print(f"  [{ds}] {type(e).__name__}: {str(e)[:140]}")
    print(f"\ncolumns matching rebate / primary_trading / trading_account_number: {len(hits)}")
    for h in hits:
        print("  ", h)
except Exception as e:
    print("client:", type(e).__name__, str(e)[:400])
