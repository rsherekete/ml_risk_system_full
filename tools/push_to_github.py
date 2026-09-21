"""Push this folder (and the session's scratch scripts) to GitHub without git.

Git is not installed on this machine, so the upload goes through the GitHub
REST API: blobs -> tree -> commit -> ref, one commit per run, into a PRIVATE
repository under the user's account. Two guards decide what leaves the box:

1. `.gitignore` in the folder root (credentials, client data stores, venvs).
2. A CONTENT scan: every candidate file is searched for the secret values
   found in the credential files (server.yaml, vantage.yaml, kafka/*.yaml)
   and for anything that looks like a token / password assignment; a file
   that carries one is skipped and listed, never uploaded.

Usage
  python tools/push_to_github.py --dry-run             # list what would go
  python tools/push_to_github.py --repo ml_antifraud_system  # upload (needs token)

Token: env GITHUB_TOKEN, else %USERPROFILE%\\.github_token (one line). A
classic token with `repo` scope or a fine-grained token with Contents:
read/write and Administration: read/write (to create the repository).
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.github.com"
OWNER = "rsherekete"
MAX_FILE = 95 * 1024 * 1024               # GitHub refuses blobs over 100 MB
SCRATCH = Path(os.environ.get("CLAUDE_SCRATCHPAD", "")) if os.environ.get("CLAUDE_SCRATCHPAD") else None
SESSION_DIR = "session_scripts"           # where the scratch scripts land in the tree

#: Only a HARD-CODED literal counts (a quoted value after password= or
#: token=); reading a password from config (password=defaults["password"])
#: or generating a token (secrets.token_urlsafe) is the application, not a leak.
#: The copy-trading engine, split out so the application can be shared
#: without it: `--profile app` pushes everything EXCEPT these paths,
#: `--profile engine` pushes ONLY these paths (into a separate private repo).
ENGINE_PATHS = (
    "webapp/vantage.py", "webapp/tape_model.py", "webapp/copytrader.py", "webapp/replay.py",
    "webapp/templates/vantage.html", "webapp/templates/copytrader*.html",
    "webapp/artifacts/tape_*", "webapp/artifacts/entry_model*", "webapp/artifacts/exit_model*",
    "webapp/artifacts/strategy2_*", "webapp/artifacts/vantage*", "webapp/artifacts/quant_perlot*",
    "webapp/artifacts/quant_mae_*", "webapp/artifacts/quant_exit_fe*", "webapp/artifacts/quant_magnitude*",
    "vantage.example.yaml", "revert_vantage.ps1",
    "session_scripts/*vantage*", "session_scripts/tape*", "session_scripts/entry_*", "session_scripts/exit_*",
    "session_scripts/replay*", "session_scripts/persec*", "session_scripts/s3_*", "session_scripts/strategy2*",
    "session_scripts/gate_*", "session_scripts/xau5_*", "session_scripts/sizing*", "session_scripts/calibrate_dd*",
    "session_scripts/risk_budget*", "session_scripts/cvar_budget*", "session_scripts/*engine*",
    "session_scripts/verify_s1*", "session_scripts/verify_sizer*", "session_scripts/test_sizer*",
    "session_scripts/trade_accuracy*", "session_scripts/wf_accuracy*", "session_scripts/pos_check*",
    "session_scripts/orderstate*", "session_scripts/close_all_demo*", "session_scripts/enable_autotrading*",
    "session_scripts/flatten_retry*", "session_scripts/turn_on*", "session_scripts/dump_vantage*",
    "session_scripts/vdiag*", "session_scripts/vperf*", "session_scripts/policy_lab*", "session_scripts/pareto_*",
    "session_scripts/book_hedge*", "session_scripts/trade_routing*", "session_scripts/horizon_*",
    "session_scripts/synthetic_*", "session_scripts/trace_pipeline*", "session_scripts/fresh_account*",
    "session_scripts/ftmo_*", "session_scripts/shot_vantage*", "session_scripts/mt5_*", "session_scripts/livescores*",
    "session_scripts/experiments*", "session_scripts/invert_*", "session_scripts/entry_*", "session_scripts/train_perlot*",
    "session_scripts/equiv_curve*", "session_scripts/edge*", "session_scripts/two_sided*", "session_scripts/windowed*",
    "session_scripts/stopout*", "session_scripts/ablate*", "session_scripts/chain_*", "session_scripts/seq_quant_exit*",
)


def matches_engine(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in ENGINE_PATHS)


#: `--source-only` keeps the repository a CODE backup. Everything below is
#: derived output that the engine rebuilds from MySQL and the tick tape: model
#: dumps (~96 MB of LightGBM text), scan results, warehouse partitions. Backing
#: them up costs a hundred megabytes a push and restores nothing that a rescan
#: would not.
DERIVED_PATHS = (
    "webapp/artifacts/*", "webapp/artifacts/**/*",
    "webapp/warehouse/*", "webapp/warehouse/**/*",
    "trading_data/*", "trading_data/**/*",
)


def matches_derived(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in DERIVED_PATHS)


SECRET_LINE = re.compile(r"(password|passwd|api[_-]?key|secret|token)\s*[:=]\s*['\"]([^'\"\s]{6,})['\"]", re.I)
PLACEHOLDER = re.compile(r"change[-_ ]?me|example|placeholder|your[-_ ]|xxx+|<[^>]+>|\$\{|%s|\{\}", re.I)
TOKEN_SHAPES = re.compile(r"\b(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ya29\.[A-Za-z0-9._-]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})\b")


# ------------------------------------------------------------- ignore rules
def load_ignore() -> list[str]:
    pats = []
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pats.append(line.rstrip("/"))
    return pats


def ignored(rel: str, pats: list[str]) -> bool:
    """Git semantics: the LAST matching rule wins, and a rule starting with
    `!` un-ignores (so `webapp/artifacts/ad/` followed by
    `!webapp/artifacts/ad/model_features.csv` ships just that one file)."""
    parts = rel.split("/")
    result = False
    for p in pats:
        negate = p.startswith("!")
        pat = p[1:] if negate else p
        if "/" in pat:
            hit = fnmatch.fnmatch(rel, pat) or rel.startswith(pat + "/")
        else:
            hit = any(fnmatch.fnmatch(part, pat) for part in parts)
        if hit:
            result = not negate
    return result


# ------------------------------------------------------------ secret values
def secret_values() -> set[str]:
    """Every password / token / key VALUE in the credential files, so a copy
    of one anywhere else (a scratch script, a notebook output) is caught."""
    vals: set[str] = set()
    import yaml
    for name in ("server.yaml", "vantage.yaml", "kafka/clusters.yaml", "kafka/config.yaml",
                 "docs/replication/_servers_local.yaml"):
        p = ROOT / name
        if not p.exists():
            continue
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(node, key=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, str(k))
            elif isinstance(node, list):
                for v in node:
                    walk(v, key)
            elif isinstance(node, str) and re.search(r"pass|secret|token|key", key, re.I) and len(node) >= 6:
                vals.add(node)
        walk(doc)
    for extra in (ROOT / "webapp" / "artifacts" / "agent_key.txt",):
        if extra.exists():
            text = extra.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) >= 8:
                vals.add(text)
    return vals


#: A client account key, e.g. "mt4_live01:1054821". Real client identifiers.
ACCOUNT_KEY = re.compile(rb"mt[45]_(?:live|demo|dubai_live)[0-9]*:[0-9]{4,}")

#: How many DISTINCT account keys make a file per-client output rather than
#: code. Source files legitimately carry one or two as worked examples; the
#: scan artefacts carry hundreds to tens of thousands.
MAX_ACCOUNT_KEYS = 5


def carries_client_data(data: bytes) -> str:
    """Per-client analysis output must never leave the machine.

    The `.gitignore` blocks client data by EXTENSION (*.csv, *.parquet, ...),
    which missed the scan artefacts entirely because they are .json. A dry run
    on 17 Sep 2026 was about to upload ~37,000 unique client account keys,
    including `event_impact_test_summary.json`'s "abusive_rows" -- identified
    accounts labelled as abusive, with their P&L. This is the content guard
    that stops that class of file whatever it is called.
    """
    keys = set(ACCOUNT_KEY.findall(data))
    if len(keys) > MAX_ACCOUNT_KEYS:
        return (f"per-client data: {len(keys):,} distinct client account keys "
                f"(limit {MAX_ACCOUNT_KEYS})")
    return ""


def carries_secret(data: bytes, secrets: set[str]) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ""                            # binary: no text secrets to find
    for s in secrets:
        if s in text:
            return "contains a credential value from the config files"
    m = TOKEN_SHAPES.search(text)
    if m:
        return f"contains a token-shaped string ({m.group(1)[:8]}...)"
    for line in text.splitlines():
        m = SECRET_LINE.search(line)
        if m and not PLACEHOLDER.search(m.group(2)):
            return f"has a hard-coded credential: {line.strip()[:60]}"
    return ""


# ---------------------------------------------------------------- the walk
def walk_tree(root: Path, pats: list[str], skipped: list[str]):
    """Every file under `root`, surviving directories that will not scan.

    `Path.rglob` aborts the WHOLE walk on the first unreadable directory, which
    on this tree is a matter of when, not if: it lives in OneDrive (cloud-only
    placeholders raise on scandir), it holds a 2 GB DuckDB the app keeps open,
    and the venvs nest deep enough to pass MAX_PATH. Losing the entire push to
    one such directory -- with no indication of which -- is the failure this
    avoids. It also PRUNES ignored directories instead of walking into them, so
    .launch-venv and webapp/warehouse are never descended at all.
    """
    for parent, dirs, names in os.walk(root, onerror=lambda e: skipped.append(
            f"{Path(getattr(e, 'filename', '?'))}: directory unreadable ({e.strerror})")):
        here = Path(parent)
        rel_dir = here.relative_to(root).as_posix()
        dirs[:] = sorted(
            d for d in dirs
            if not ignored(f"{rel_dir}/{d}".lstrip("./") if rel_dir != "." else d, pats)
            and d != ".git")
        for name in sorted(names):
            yield here / name


def collect(dry: bool) -> tuple[list[tuple[str, Path]], list[str]]:
    pats = load_ignore()
    secrets = secret_values()
    files: list[tuple[str, Path]] = []
    skipped: list[str] = []
    for path in walk_tree(ROOT, pats, skipped):
        rel = path.relative_to(ROOT).as_posix()
        # stat() is the second thing that can throw rather than answer: a
        # cloud-only OneDrive placeholder, a file the app holds open, a path
        # past MAX_PATH. Report it against the file and carry on -- one
        # unreadable entry must never cost the other 502.
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
        except OSError as error:
            skipped.append(f"{rel}: unreadable ({error.strerror})")
            continue
        if rel.startswith(".git/") or ignored(rel, pats):
            continue
        if size > MAX_FILE:
            skipped.append(f"{rel}: {size/1e6:.0f} MB is over GitHub's 100 MB blob limit")
            continue
        try:
            data = path.read_bytes()
        except OSError as error:
            skipped.append(f"{rel}: unreadable ({error.strerror}) -- locked by a running process")
            continue
        why = carries_secret(data, secrets) or carries_client_data(data)
        if why:
            skipped.append(f"{rel}: {why}")
            continue
        files.append((rel, path))
    if SCRATCH and SCRATCH.is_dir():
        for path in sorted(SCRATCH.glob("*")):
            if not path.is_file() or path.suffix.lower() not in (".py", ".md", ".txt", ".json", ".ps1", ".sql"):
                continue
            rel = f"{SESSION_DIR}/{path.name}"
            data = path.read_bytes()
            if len(data) > MAX_FILE:
                continue
            why = carries_secret(data, secrets) or carries_client_data(data)
            if why:
                skipped.append(f"{rel}: {why}")
                continue
            files.append((rel, path))
    return files, skipped


# ---------------------------------------------------------------- GitHub
def token() -> str:
    """GITHUB_TOKEN, else %USERPROFILE%\\.github_token, else the GitHub CLI's
    stored login (`gh auth token`) -- the CLI's browser login is the easiest
    way to authorise this machine."""
    t = os.environ.get("GITHUB_TOKEN", "").strip()
    if not t:
        p = Path.home() / ".github_token"
        if p.exists():
            t = p.read_text(encoding="utf-8").strip()
    if not t:
        import shutil, subprocess
        gh = shutil.which("gh") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "gh" / "bin" / "gh.exe")
        try:
            t = subprocess.run([gh, "auth", "token"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            t = ""
    if not t:
        sys.exit("no token: log in with `gh auth login --web`, or set GITHUB_TOKEN, or write it to %USERPROFILE%\\.github_token")
    return t


class GH:
    def __init__(self, tok: str):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def call(self, method, path, **kw):
        for attempt in range(5):
            r = self.s.request(method, API + path, timeout=120, **kw)
            if r.status_code in (403, 429) and "rate" in r.text.lower():
                wait = int(r.headers.get("Retry-After", "30"))
                print(f"  rate limited, waiting {wait}s"); time.sleep(wait); continue
            if r.status_code >= 500:
                time.sleep(2 + attempt * 3); continue
            return r
        return r

    def ensure_repo(self, repo: str, owner: str) -> tuple[str, str | None]:
        # Say which repository, as the tool resolved it. A 404 here is
        # ambiguous between "wrong name", "wrong owner" and "token cannot see
        # it", and without the resolved path in the log all three look the same.
        print(f"repo: GET /repos/{owner}/{repo}")
        r = self.call("GET", f"/repos/{owner}/{repo}")
        print(f"  -> {r.status_code}")
        if r.status_code == 200:
            info = r.json()
            # REFUSE A PUBLIC REPOSITORY. A repo this tool CREATES is private,
            # but one that already exists was never checked -- so pushing to a
            # name that happens to be public would publish the copy-trading
            # strategy (vantage.py, tape_model.py) and every client account key
            # that slipped under the 5-key limit, irreversibly and to everyone.
            # Visibility is not something to discover after the push.
            if not info.get("private", True):
                sys.exit(
                    f"REFUSING: {owner}/{repo} is PUBLIC.\n"
                    f"  This push carries proprietary strategy code.\n"
                    f"  Make it private at https://github.com/{owner}/{repo}/settings\n"
                    f"  (Danger Zone -> Change repository visibility), then re-run.")
        elif r.status_code == 404:
            me = self.call("GET", "/user").json().get("login")
            body = {"name": repo, "private": True, "auto_init": False,
                    "description": "Broker risk-analytics web app, copy-trading engine and research notebooks"}
            r = self.call("POST", "/user/repos", json=body) if me and me.lower() == owner.lower() \
                else self.call("POST", f"/orgs/{owner}/repos", json=body)
            if r.status_code not in (200, 201):
                # "Already exists" after a 404 read is not a naming clash -- it
                # is a PERMISSION problem wearing a confusing hat. The account
                # owns the repo; the token simply cannot see it, and GitHub
                # returns 404 rather than 403 for private repos a token has no
                # scope for. Say so, because the raw 422 sends you looking for a
                # duplicate name that does not exist.
                if r.status_code == 422 and "already exists" in r.text:
                    scopes = r.headers.get("X-OAuth-Scopes", "(none reported)")
                    sys.exit(
                        f"{owner}/{repo} EXISTS but this token cannot see it.\n"
                        f"  Token scopes: {scopes}\n"
                        f"  A private repo needs the top-level `repo` scope. Ticking\n"
                        f"  only `public_repo` is not enough, and a fine-grained token\n"
                        f"  must list this repository under 'Only select repositories'.\n"
                        f"  Fix at https://github.com/settings/tokens, then re-run.")
                sys.exit(f"cannot create {owner}/{repo}: {r.status_code} {r.text[:200]}")
            info = r.json()
            print(f"created private repository {info['full_name']}")
        else:
            sys.exit(f"cannot read {owner}/{repo}: {r.status_code} {r.text[:200]}")
        branch = info.get("default_branch") or "main"
        r = self.call("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
        head = r.json()["object"]["sha"] if r.status_code == 200 else None
        if head is None:
            # A repository with no commits refuses the git-data API ("Git
            # Repository is empty", 409): seed it with one file through the
            # contents API, which creates the branch and the first commit.
            seed = base64.b64encode(b"# notebook\n\nSeeded by tools/push_to_github.py; the full tree follows in the next commit.\n").decode()
            r = self.call("PUT", f"/repos/{owner}/{repo}/contents/README.md",
                          json={"message": "Initialise repository", "content": seed, "branch": branch})
            if r.status_code not in (200, 201):
                sys.exit(f"cannot seed the empty repository: {r.status_code} {r.text[:200]}")
            head = r.json()["commit"]["sha"]
            print(f"seeded empty repository with a first commit on {branch}")
        return branch, head

    def blob(self, owner, repo, data: bytes) -> str:
        r = self.call("POST", f"/repos/{owner}/{repo}/git/blobs",
                      json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
        if r.status_code != 201:
            raise RuntimeError(f"blob failed: {r.status_code} {r.text[:200]}")
        return r.json()["sha"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ml_antifraud_system")
    ap.add_argument("--owner", default=OWNER)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source-only", action="store_true",
                    help="code backup: drop derived artefacts (model dumps, "
                         "scan results, warehouse partitions)")
    ap.add_argument("--profile", choices=["all", "app", "engine"], default="all",
                    help="all = whole folder; app = everything except the copy-trading engine; engine = the engine only")
    ap.add_argument("--message", default=None)
    args = ap.parse_args()
    if args.message is None:
        args.message = {"all": "Risk analytics app, Vantage copy-trading engine, Event Impact rebuild, session scripts",
                        "app": "Risk analytics web app, Event Impact rebuild, research notebooks, session scripts (engine kept separate)",
                        "engine": "Vantage copy-trading engine: strategy, tape model, sizing, studies"}[args.profile]

    files, skipped = collect(args.dry_run)
    if args.profile == "app":
        held = [rel for rel, _ in files if matches_engine(rel)]
        files = [(rel, p) for rel, p in files if not matches_engine(rel)]
        print(f"profile app: {len(held)} engine files held back")
    elif args.profile == "engine":
        files = [(rel, p) for rel, p in files if matches_engine(rel) or rel in (".gitignore", "requirements.txt")]
        print(f"profile engine: {len(files)} engine files selected")
    if args.source_only:
        dropped = [rel for rel, _ in files if matches_derived(rel)]
        bytes_dropped = sum(p.stat().st_size for rel, p in files if matches_derived(rel))
        files = [(rel, p) for rel, p in files if not matches_derived(rel)]
        print(f"source-only: {len(dropped)} derived files held back "
              f"({bytes_dropped/1e6:.1f} MB of model dumps / scan output)")
    total = sum(p.stat().st_size for _, p in files)
    print(f"{len(files):,} files, {total/1e6:.1f} MB to upload; {len(skipped)} skipped")
    for s in skipped:
        print("  skipped:", s)
    if args.dry_run:
        by_dir: dict[str, int] = {}
        for rel, p in files:
            by_dir[rel.split('/')[0]] = by_dir.get(rel.split('/')[0], 0) + p.stat().st_size
        for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1]):
            print(f"  {d:<28} {n/1e6:8.1f} MB")
        return

    gh = GH(token())
    branch, head = gh.ensure_repo(args.repo, args.owner)
    print(f"repository {args.owner}/{args.repo}, branch {branch}, head {head or '(empty)'}")
    tree = []
    t0 = time.time()
    for i, (rel, p) in enumerate(files, 1):
        sha = gh.blob(args.owner, args.repo, p.read_bytes())
        tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": sha})
        if i % 100 == 0:
            print(f"  {i}/{len(files)} blobs ({time.time()-t0:.0f}s)")
    # BUILD THE TREE IN CHUNKS, chaining each onto the last with base_tree.
    # One 503-entry request returned "tree.sha <sha> is not a valid blob" --
    # a single opaque SHA with no indication of which of the 503 paths it
    # belonged to, after four minutes of uploading. Chunking bounds the request,
    # and a failure now names the batch it happened in, so the search is 25
    # paths rather than every file in the push.
    TREE_CHUNK = 25
    tree_sha = None
    for start in range(0, len(tree), TREE_CHUNK):
        batch = tree[start:start + TREE_CHUNK]
        body = {"tree": batch}
        if tree_sha:
            body["base_tree"] = tree_sha
        r = gh.call("POST", f"/repos/{args.owner}/{args.repo}/git/trees", json=body)
        if r.status_code != 201:
            print(f"tree failed on entries {start + 1}-{start + len(batch)} of {len(tree)}:")
            for e in batch:
                print(f"    {e['sha'][:8]}  {e['path']}")
            sys.exit(f"tree failed: {r.status_code} {r.text[:300]}")
        tree_sha = r.json()["sha"]
        if (start // TREE_CHUNK) % 5 == 0:
            print(f"  tree {min(start + TREE_CHUNK, len(tree))}/{len(tree)} entries")
    body = {"message": args.message, "tree": tree_sha}
    if head:
        body["parents"] = [head]
    r = gh.call("POST", f"/repos/{args.owner}/{args.repo}/git/commits", json=body)
    if r.status_code != 201:
        sys.exit(f"commit failed: {r.status_code} {r.text[:300]}")
    commit = r.json()["sha"]
    ref = f"/repos/{args.owner}/{args.repo}/git/refs/heads/{branch}"
    r = gh.call("PATCH", ref, json={"sha": commit, "force": False}) if head else \
        gh.call("POST", f"/repos/{args.owner}/{args.repo}/git/refs", json={"ref": f"refs/heads/{branch}", "sha": commit})
    if r.status_code not in (200, 201):
        sys.exit(f"ref update failed: {r.status_code} {r.text[:300]}")
    print(f"pushed {len(files):,} files as {commit[:10]} -> https://github.com/{args.owner}/{args.repo}")


if __name__ == "__main__":
    main()
