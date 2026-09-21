"""Cloud shard runner: pull one shard of Hyperliquid fills history from a
fresh egress IP, then hand the parquet parts back as a GitHub release asset.

  python hl_shard_run.py --shard 1/4 --token <github token> [--top 14133] [--since 2026-03-16]
                         [--coins xyz:GOLD,BTC,ETH,xyz:SILVER,xyz:SP500] [--hours 5]

Needs hl_collect.py next to it (fetch it from the repository first). Stops
after --hours and uploads whatever it has, so a session time limit never
loses the work; re-running the same shard resumes from its done list.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = "https://api.github.com"
OWNER, REPO = "vivasiRoy", "notebook"


def upload(token: str, tag: str, path: Path) -> None:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    r = s.get(f"{API}/repos/{OWNER}/{REPO}/releases/tags/{tag}", timeout=60)
    if r.status_code == 404:
        r = s.post(f"{API}/repos/{OWNER}/{REPO}/releases", json={"tag_name": tag, "target_commitish": "main", "name": f"Hyperliquid fills {tag}",
                                                                   "body": "Fills history shards pulled from cloud sessions (public data).", "draft": False, "prerelease": True}, timeout=60)
    rel = r.json()
    for a in rel.get("assets", []):
        if a["name"] == path.name:
            s.delete(f"{API}/repos/{OWNER}/{REPO}/releases/assets/{a['id']}", timeout=60)
    upload_url = rel["upload_url"].split("{")[0]
    with open(path, "rb") as fh:
        r = s.post(f"{upload_url}?name={path.name}", data=fh, timeout=(60, 1800),
                   headers={"Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)})
    print(f"upload {path.name}: {r.status_code} {r.text[:120] if r.status_code >= 300 else 'ok'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--top", type=int, default=14133)
    ap.add_argument("--since", default="2026-03-16")
    ap.add_argument("--coins", default="xyz:GOLD,BTC,ETH,xyz:SILVER,xyz:SP500")
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--tag", default="hl-fills-2026-09-15")
    args = ap.parse_args()
    k, n = args.shard.split("/")
    cmd = [sys.executable, str(HERE / "hl_collect.py"), "fills", "--coins", args.coins, "--top", str(args.top), "--since", args.since, "--shard", args.shard]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=sys.stdout, stderr=sys.stderr)
    deadline = time.time() + args.hours * 3600
    while proc.poll() is None and time.time() < deadline:
        time.sleep(30)
    if proc.poll() is None:
        print(f"time budget reached after {args.hours} h -- stopping the puller and uploading what we have", flush=True)
        proc.terminate()
        try:
            proc.wait(60)
        except Exception:
            proc.kill()
    out = HERE / "hl"
    parts = sorted(out.glob(f"fills_part_s{k}of{n}_*.parquet")) + sorted(out.glob(f"fills_done_{k}of{n}.txt"))
    if not parts:
        print("nothing collected", flush=True); return
    zpath = HERE / f"hl_fills_shard_{k}of{n}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in parts:
            zf.write(p, p.name)
    print(f"zipped {len(parts)} files -> {zpath.name} ({zpath.stat().st_size / 1e6:.1f} MB)", flush=True)
    upload(args.token, args.tag, zpath)


if __name__ == "__main__":
    main()
