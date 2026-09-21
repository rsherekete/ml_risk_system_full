"""Sync code from this working tree into the SHARED repository clone.

The shared repo (rsherekete/antifraud_engine_new) is not a fork of this one. It
is an assembled subset: code only, with the copy-trading engine replaced by a
stub and client identifiers stripped out. Doing that by hand is how something
eventually leaks, so it lives here instead.

What this does, in order:

1. Copies the application source across (webapp, trading_data, kafka).
2. Drops the copy-trading engine and installs the stub in its place.
3. Redacts real `server:login` account identifiers and internal hosts that sit
   in source comments and templates. The replacement is derived from a hash of
   the original, so the same real account always maps to the same placeholder
   and files do not churn between runs.
4. Scans the result and **refuses to continue** if anything that should not
   leave is still present.

It never commits or pushes; run it, read the diff, then commit yourself.

Usage
    python tools/sync_shared_repo.py --target ../path/to/antifraud_engine_new
    python tools/sync_shared_repo.py --target ... --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: (directory, filename patterns) copied across. Anything not listed stays here.
INCLUDE = [
    ("webapp", ("*.py",)),
    ("webapp/templates", ("*.html",)),
    ("webapp/static", ("*",)),
    ("trading_data", ("*.py",)),
    ("kafka", ("*.py",)),
]

ROOT_FILES = ("requirements-webapp.txt", "server.example.yaml", "symbol_map.example.yaml")
TOOL_FILES = ("make_sample_artifacts.py", "sync_shared_repo.py")

#: The copy-trading engine. Proprietary, kept in its own private repository, and
#: not needed by the beta profile -- its routes are gated off.
ENGINE_FILES = (
    "webapp/vantage.py", "webapp/tape_model.py", "webapp/copytrader.py",
    "webapp/templates/vantage.html", "webapp/templates/quant/copytrader.html",
)

#: `main.py` imports `copytrader` at module scope, so the name must exist in the
#: target even though the real module does not travel. This stub is the canonical
#: copy and is reinstalled on every sync, after the engine files are removed.
STUB = "webapp/copytrader.py"
STUB_SOURCE = ROOT / "tools" / "copytrader_stub.py"

ACCOUNT_RE = re.compile(r"\b(mt[45]_(?:live|dubai_live)\d+):(\d+)\b", re.I)
#: Real internal hosts that appear in docstrings and example config.
HOST_SUBS = {"10.135.51.17": "db-host.internal"}


def placeholder(match: re.Match) -> str:
    """Stable pseudonym: same input -> same output on every run, so a re-sync
    produces no spurious diff."""
    server, login = match.group(1), match.group(2)
    if login.startswith("1000") and len(login) == 6:
        return match.group(0)                      # already redacted
    return f"{server}:{100000 + zlib.crc32(login.encode()) % 900000}"


def redact(text: str) -> str:
    text = ACCOUNT_RE.sub(placeholder, text)
    for real, fake in HOST_SUBS.items():
        text = text.replace(real, fake)
    return text


GATE = {
    "real account identifier": re.compile(r"mt[45]_(?:live|dubai_live)\d+:(?!\d{6}\b)\d+", re.I),
    "internal IP": re.compile(r"\b10\.\d+\.\d+\.\d+\b"),
    "internal hostname": re.compile(r"[a-z0-9\-]+\.(?:zfx|zeal)[a-z0-9\-]*\.(?:com|net|local)", re.I),
    "credential token": re.compile(r"(ghp_|github_pat_|xox[baprs]-|AKIA[0-9A-Z]{16})"),
    "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
}


def scan(target: Path) -> dict:
    findings: dict[str, list] = {}
    for path in target.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix == ".parquet":
            continue
        # This file necessarily contains the very patterns it searches for.
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, rx in GATE.items():
            for hit in rx.findall(text):
                findings.setdefault(label, []).append(
                    (str(path.relative_to(target)), str(hit)[:40]))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="the shared repo clone")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not (target / ".git").exists():
        print(f"error: {target} is not a git clone", file=sys.stderr)
        return 2

    copied = 0
    for rel, pats in INCLUDE:
        source = ROOT / rel
        if not source.exists():
            continue
        for path in source.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or "artifacts" in path.parts:
                continue
            if not any(fnmatch.fnmatch(path.name, p) for p in pats):
                continue
            relative = path.relative_to(ROOT)
            if str(relative).replace("\\", "/") in ENGINE_FILES:
                continue
            out = target / relative
            if args.dry_run:
                copied += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in (".py", ".html", ".md", ".txt", ".yaml"):
                # newline="" on both sides so line endings pass through
                # untouched. Without it every file is rewritten LF->CRLF and
                # shows as modified with an empty diff, burying the few files
                # that actually changed.
                with open(path, encoding="utf-8", errors="ignore", newline="") as handle:
                    new = redact(handle.read())
                old = None
                if out.exists():
                    with open(out, encoding="utf-8", errors="ignore", newline="") as handle:
                        old = handle.read()
                if new == old:
                    continue                      # unchanged; leave it alone
                with open(out, "w", encoding="utf-8", newline="") as handle:
                    handle.write(new)
            else:
                if out.exists() and out.read_bytes() == path.read_bytes():
                    continue
                shutil.copy2(path, out)
            copied += 1

    for name in ROOT_FILES:
        if (ROOT / name).exists() and not args.dry_run:
            shutil.copy2(ROOT / name, target / name)
    (target / "tools").mkdir(exist_ok=True)
    for name in TOOL_FILES:
        if (ROOT / "tools" / name).exists() and not args.dry_run:
            shutil.copy2(ROOT / "tools" / name, target / "tools" / name)
    if (ROOT / "docs" / "DEPLOYMENT.md").exists() and not args.dry_run:
        (target / "docs").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "docs" / "DEPLOYMENT.md", target / "docs" / "DEPLOYMENT.md")

    # The engine must not survive a re-copy, and the stub must be reinstated.
    if not args.dry_run:
        for name in ENGINE_FILES:
            victim = target / name
            if victim.exists():
                victim.unlink()
                print(f"  removed engine file: {name}")
        if STUB_SOURCE.exists():
            shutil.copy2(STUB_SOURCE, target / STUB)
            print(f"  installed stub: {STUB}")
        else:
            print(f"  WARNING: {STUB_SOURCE.name} is missing, so {STUB} is absent "
                  f"from the target and the app will not import there.")

    print(f"copied {copied} file(s) into {target}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    findings = scan(target)
    total = sum(len(v) for v in findings.values())
    print(f"\ngate scan: {total} finding(s)")
    for label, hits in findings.items():
        print(f"  {label}: {len(hits)}")
        for where, what in hits[:5]:
            print(f"      {where} -> {what}")
    if total:
        print("\nREFUSING to call this done. Fix the findings above, then re-run.",
              file=sys.stderr)
        return 1
    print("\nClean. Review `git status` in the target, then commit and push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
