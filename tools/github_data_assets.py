"""Ship the data files GitHub refuses as commits -- as release assets.

GitHub caps a committed file at 100 MB. The antifraud corpus
(model_frame.parquet, ~390 MB) and the 90-day BigQuery records snapshot
(bq_90d_records.parquet, ~930 MB) are far over it, so they travel attached to
a release on the same private repository (a release asset may be up to 2 GB),
and a fresh clone pulls them back into the folders the app reads with `fetch`.

  python tools/github_data_assets.py status       # what the release holds
  python tools/github_data_assets.py upload       # this machine -> GitHub
  python tools/github_data_assets.py fetch        # fresh clone  <- GitHub

Options: --repo ml_antifraud_system --owner rsherekete --tag data-2026-09-14 --force
(fetch: re-download even when a same-size file is already in place).

Token: env GITHUB_TOKEN, else %USERPROFILE%\\.github_token -- a fine-grained
token with Contents: read (fetch) or read/write (upload) on the repository.
A collaborator fetching on another machine uses a token of their own.

Where the files land (relative to the folder that holds `webapp/`):
  model_frame.parquet          webapp/artifacts/ad/   antifraud account-day corpus
  model_features.csv           webapp/artifacts/ad/   the corpus's feature manifest
  bq_90d_records.parquet       webapp/artifacts/ad/   90-day BigQuery trade records
  markout_all_servers.parquet  webapp/artifacts/ad/   post-trade markout panels
webapp/model_service.py resolves its data folder to webapp/artifacts/ad when
NOTEBOOK_DATA_DIR is unset and the original session scratchpad is absent.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_to_github import token  # noqa: E402  (same token sources as the code push)

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.github.com"
OWNER = "rsherekete"
REPO = "ml_antifraud_system"
TAG = "data-2026-09-14"
AD_DIR = ROOT / "webapp" / "artifacts" / "ad"
#: the session scratchpad the markout panel was first built in (original machine)
LEGACY_SCRATCH = Path(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude"
                      r"\c--Users-RoyVivasi-Documents-notebook"
                      r"\9951e7b4-740a-496a-a92d-689972573193\scratchpad")
CHUNK = 8 * 1024 * 1024

#: asset name -> destination relative to ROOT. Upload looks for the source in
#: the destination first, then NOTEBOOK_DATA_DIR, then the legacy scratchpad.
MANIFEST: dict[str, str] = {
    "model_frame.parquet": "webapp/artifacts/ad/model_frame.parquet",
    "model_features.csv": "webapp/artifacts/ad/model_features.csv",
    "bq_90d_records.parquet": "webapp/artifacts/ad/bq_90d_records.parquet",
    "markout_all_servers.parquet": "webapp/artifacts/ad/markout_all_servers.parquet",
}


def _source(name: str) -> Path | None:
    candidates = [ROOT / MANIFEST[name]]
    env = os.environ.get("NOTEBOOK_DATA_DIR", "").strip()
    if env:
        candidates.append(Path(env) / name)
    candidates.append(LEGACY_SCRATCH / name)
    for p in candidates:
        if p.is_file():
            return p
    return None


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


class _Progress:
    """File-like reader that reports how much of a large upload has gone."""

    def __init__(self, path: Path, label: str):
        self.fh = open(path, "rb")
        self.total = path.stat().st_size
        self.sent = 0
        self.label = label
        self.t0 = time.time()
        self.mark = 0

    def __len__(self) -> int:
        return self.total - self.sent

    def read(self, n: int = -1) -> bytes:
        data = self.fh.read(n)
        self.sent += len(data)
        if self.sent - self.mark >= 50 * 1024 * 1024 or self.sent == self.total:
            self.mark = self.sent
            rate = self.sent / max(time.time() - self.t0, 1e-6) / 1e6
            print(f"    {self.label}: {_mb(self.sent)} / {_mb(self.total)} ({rate:.1f} MB/s)", flush=True)
        return data

    def close(self) -> None:
        self.fh.close()


class GH:
    def __init__(self, tok: str):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def release(self, owner: str, repo: str, tag: str | None) -> dict | None:
        if tag:
            r = self.s.get(f"{API}/repos/{owner}/{repo}/releases/tags/{tag}", timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            sys.exit(f"cannot read release {tag}: {r.status_code} {r.text[:200]}")
        r = self.s.get(f"{API}/repos/{owner}/{repo}/releases", timeout=60, params={"per_page": 50})
        if r.status_code != 200:
            sys.exit(f"cannot list releases: {r.status_code} {r.text[:200]}")
        data = [x for x in r.json() if str(x.get("tag_name", "")).startswith("data-")]
        return sorted(data, key=lambda x: x["tag_name"])[-1] if data else None

    def create_release(self, owner: str, repo: str, tag: str) -> dict:
        body = ("Data files the app reads that are too large to commit. Fetch them into place with\n"
                "`python tools/github_data_assets.py fetch` (needs a token with Contents: read).\n\n"
                + "\n".join(f"- `{name}` -> `{dest}`" for name, dest in MANIFEST.items()))
        r = self.s.post(f"{API}/repos/{owner}/{repo}/releases", timeout=60,
                        json={"tag_name": tag, "target_commitish": "main",
                              "name": f"Antifraud data snapshot {tag.removeprefix('data-')}",
                              "body": body, "draft": False, "prerelease": False})
        if r.status_code not in (200, 201):
            sys.exit(f"cannot create release {tag}: {r.status_code} {r.text[:300]}")
        print(f"created release {tag}")
        return r.json()

    def delete_asset(self, owner: str, repo: str, asset_id: int) -> None:
        r = self.s.delete(f"{API}/repos/{owner}/{repo}/releases/assets/{asset_id}", timeout=60)
        if r.status_code not in (204, 404):
            sys.exit(f"cannot delete asset {asset_id}: {r.status_code} {r.text[:200]}")

    def upload_asset(self, rel: dict, name: str, path: Path) -> dict:
        upload_url = rel["upload_url"].split("{")[0]
        size = path.stat().st_size
        last: Exception | None = None
        for attempt in range(3):
            body = _Progress(path, name)
            try:
                r = self.s.post(f"{upload_url}?name={name}", data=body, timeout=(60, 1800),
                                headers={"Content-Type": "application/octet-stream",
                                         "Content-Length": str(size)})
            except requests.RequestException as error:
                last = error
                print(f"    attempt {attempt + 1} failed: {error}; retrying")
                time.sleep(5)
                continue
            finally:
                body.close()
            if r.status_code in (200, 201):
                info = r.json()
                if int(info.get("size") or 0) != size:
                    sys.exit(f"{name}: GitHub stored {info.get('size')} bytes, expected {size}")
                return info
            if r.status_code == 422 and "already_exists" in r.text:
                sys.exit(f"{name}: an asset with this name already exists -- run `status`, then re-run "
                         "(a partial upload is deleted automatically next time)")
            last = RuntimeError(f"{r.status_code} {r.text[:200]}")
            print(f"    attempt {attempt + 1} failed: {last}; retrying")
            time.sleep(5)
        sys.exit(f"{name}: upload failed: {last}")


def cmd_status(gh: GH, args) -> None:
    rel = gh.release(args.owner, args.repo, args.tag)
    if rel is None:
        print(f"no data release on {args.owner}/{args.repo}" + (f" with tag {args.tag}" if args.tag else ""))
        return
    print(f"release {rel['tag_name']} ({rel.get('name')}) -- {rel['html_url']}")
    have = {a["name"]: a for a in rel.get("assets", [])}
    for name, dest in MANIFEST.items():
        a = have.get(name)
        local = ROOT / dest
        local_note = f"local {_mb(local.stat().st_size)}" if local.is_file() else "local missing"
        if a is None:
            print(f"  {name:<30} not uploaded            {local_note}")
        else:
            print(f"  {name:<30} {a['state']:<9} {_mb(a['size']):>12}  {local_note}")
    for name in sorted(set(have) - set(MANIFEST)):
        print(f"  {name:<30} (not in the manifest)")


def cmd_upload(gh: GH, args) -> None:
    tag = args.tag or TAG
    rel = gh.release(args.owner, args.repo, tag) or gh.create_release(args.owner, args.repo, tag)
    have = {a["name"]: a for a in rel.get("assets", [])}
    for name in MANIFEST:
        src = _source(name)
        if src is None:
            print(f"  {name}: no local copy found, skipped")
            continue
        size = src.stat().st_size
        old = have.get(name)
        if old and old.get("state") == "uploaded" and int(old.get("size") or 0) == size:
            print(f"  {name}: already uploaded ({_mb(size)})")
            continue
        if old:
            print(f"  {name}: replacing the {old.get('state')} copy of {_mb(int(old.get('size') or 0))}")
            gh.delete_asset(args.owner, args.repo, old["id"])
        print(f"  {name}: uploading {_mb(size)} from {src}")
        info = gh.upload_asset(rel, name, src)
        print(f"  {name}: done ({info['state']}, {_mb(info['size'])})")
    print(f"release {tag}: https://github.com/{args.owner}/{args.repo}/releases/tag/{tag}")


def cmd_fetch(gh: GH, args) -> None:
    rel = gh.release(args.owner, args.repo, args.tag)
    if rel is None:
        sys.exit(f"no data release found on {args.owner}/{args.repo}")
    print(f"fetching release {rel['tag_name']}")
    for a in rel.get("assets", []):
        dest_rel = MANIFEST.get(a["name"])
        if dest_rel is None:
            print(f"  {a['name']}: not in the manifest, skipped")
            continue
        dest = ROOT / dest_rel
        if dest.is_file() and dest.stat().st_size == a["size"] and not args.force:
            print(f"  {a['name']}: already in place ({_mb(a['size'])})")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        print(f"  {a['name']}: downloading {_mb(a['size'])} -> {dest_rel}")
        t0 = time.time()
        # The asset URL answers a 302 to a signed object-store URL; requests
        # drops the Authorization header on that cross-host hop, as it must.
        with gh.s.get(a["url"], headers={"Accept": "application/octet-stream"}, stream=True,
                      allow_redirects=True, timeout=(60, 600)) as r:
            if r.status_code != 200:
                sys.exit(f"{a['name']}: download failed: {r.status_code} {r.text[:200]}")
            got, mark = 0, 0
            with open(part, "wb") as out:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    out.write(chunk)
                    got += len(chunk)
                    if got - mark >= 100 * 1024 * 1024:
                        mark = got
                        print(f"    {_mb(got)} / {_mb(a['size'])} ({got / max(time.time() - t0, 1e-6) / 1e6:.1f} MB/s)",
                              flush=True)
        if got != a["size"]:
            part.unlink(missing_ok=True)
            sys.exit(f"{a['name']}: got {got} bytes, expected {a['size']}")
        part.replace(dest)
        print(f"  {a['name']}: done in {time.time() - t0:.0f}s")
    print("all assets in place; restart the app so it reads them")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "upload", "fetch"])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--owner", default=OWNER)
    ap.add_argument("--tag", default=None, help=f"release tag (upload default {TAG}; fetch/status default: newest data-* release)")
    ap.add_argument("--force", action="store_true", help="fetch: overwrite same-size files already in place")
    args = ap.parse_args()
    gh = GH(token())
    {"status": cmd_status, "upload": cmd_upload, "fetch": cmd_fetch}[args.command](gh, args)


if __name__ == "__main__":
    main()
