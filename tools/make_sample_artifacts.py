"""Build ANONYMISED sample artefacts so a fresh deployment renders something.

A forked deployment with an empty `webapp/artifacts/` shows empty screens, which
reads as broken. This produces a small, structurally-valid artefact set that
makes the Latency and Toxic panes render, carrying no real client data.

WHAT IS REMOVED, and why each one matters:

* **Account keys** -> deterministic pseudonyms (`mt4_demo01:1000042`). Stable
  across files via one salted hash, so an account flagged in the latency scan is
  the same account in the toxic features -- the screens stay coherent -- while
  the real `server:login` never appears. The salt is random per run and thrown
  away, so the mapping cannot be reversed even by re-running this.
* **Money** -> scaled by one random factor in [0.3, 3.0] plus per-value jitter.
  Shape and rank order survive; the firm's real exposure does not. A single
  scale factor would be trivially undone from any one known figure, hence the
  jitter.
* **Countries** -> generic labels. The real distribution is commercially
  sensitive in aggregate (it names where the abuse concentrates), and no screen
  needs the true value to demonstrate the feature.
* **Dates** -> shifted by a whole number of days so weekday structure survives.

Usage
    python tools/make_sample_artifacts.py                  # -> sample_artifacts/
    python tools/make_sample_artifacts.py --out some/dir --accounts 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "webapp" / "artifacts"

#: `server:login` as it appears throughout the artefacts.
ACCOUNT_RE = re.compile(r"^(mt[45]_[a-z0-9_]+):(\d+)$", re.I)

#: Keys whose values are money and must not survive in the clear.
MONEY_KEYS = {
    "econ_usd", "realized_pnl", "adverse_usd", "toxic_profit", "markout_usd",
    "gross_profit", "gross_loss", "net_profit", "adverse_usd_total",
    "early_mo_usd", "avg_adverse_usd", "profit_per_min", "equity", "balance",
}

#: Country is generalised rather than dropped: the field must stay populated or
#: the templates that group by it render an empty panel.
GENERIC_COUNTRIES = ["AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH"]


class Anonymiser:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        # Thrown away when the process exits -- nothing persists that could
        # turn a pseudonym back into a real account.
        self.salt = os.urandom(16).hex()
        self.scale = self.rng.uniform(0.3, 3.0)
        self._accounts: dict[str, str] = {}
        self._countries: dict[str, str] = {}
        self.day_shift = self.rng.randint(120, 400)

    def account(self, value: str) -> str:
        match = ACCOUNT_RE.match(str(value))
        if not match:
            return value
        if value not in self._accounts:
            digest = hashlib.sha256((self.salt + value).encode()).hexdigest()
            platform = "mt5" if match.group(1).lower().startswith("mt5") else "mt4"
            server = f"{platform}_demo{int(digest[:2], 16) % 3 + 1:02d}"
            self._accounts[value] = f"{server}:{1_000_000 + int(digest[2:8], 16) % 900_000}"
        return self._accounts[value]

    def country(self, value: str) -> str:
        key = str(value)
        if key not in self._countries:
            self._countries[key] = GENERIC_COUNTRIES[
                len(self._countries) % len(GENERIC_COUNTRIES)]
        return self._countries[key]

    def money(self, value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return value
        return round(float(value) * self.scale * self.rng.uniform(0.85, 1.15), 2)

    def walk(self, node, key: str | None = None):
        """Recurse a decoded-JSON structure, rewriting in place."""
        if isinstance(node, dict):
            return {self.account(k) if ACCOUNT_RE.match(str(k)) else k:
                    self.walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [self.walk(v, key) for v in node]
        if isinstance(node, str):
            if ACCOUNT_RE.match(node):
                return self.account(node)
            if key in ("country", "country_code"):
                return self.country(node)
            return node
        if key in MONEY_KEYS:
            return self.money(node)
        return node

    def frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if out.index.name in ("account_key", "account"):
            out.index = [self.account(v) for v in out.index]
        for column in out.columns:
            low = column.lower()
            if low in ("account", "account_key"):
                out[column] = [self.account(v) for v in out[column]]
            elif low in ("country", "country_code"):
                out[column] = [self.country(v) for v in out[column]]
            elif low in MONEY_KEYS:
                out[column] = [self.money(v) for v in out[column]]
            elif pd.api.types.is_datetime64_any_dtype(out[column]):
                out[column] = out[column] - pd.Timedelta(days=self.day_shift)
        return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "sample_artifacts"))
    parser.add_argument("--accounts", type=int, default=250,
                        help="cap rows in the per-account tables")
    parser.add_argument("--orders", type=int, default=4000)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    anon = Anonymiser()
    written = []

    for name in ("latency_scan.json", "toxic_scan.json"):
        path = SRC / name
        if not path.exists():
            print(f"  skip {name} (absent)")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            data["rows"] = data["rows"][: args.accounts]
        data = anon.walk(data)
        (out / name).write_text(json.dumps(data, indent=1), encoding="utf-8")
        written.append(name)

    for name, cap in (("toxic_features.parquet", args.accounts),
                      ("toxic_orders.parquet", args.orders)):
        path = SRC / name
        if not path.exists():
            print(f"  skip {name} (absent)")
            continue
        frame = pd.read_parquet(path)
        if "adverse_usd" in frame.columns:      # keep the interesting tail
            frame = frame.sort_values("adverse_usd", ascending=False)
        frame = frame.head(cap)
        anon.frame(frame).to_parquet(out / name)
        written.append(name)

    (out / "README.md").write_text(
        "# Sample artefacts\n\n"
        "Anonymised, generated by `tools/make_sample_artifacts.py`.\n\n"
        "Account identifiers are pseudonyms, monetary values are scaled and\n"
        "jittered, countries are generic labels and dates are shifted. Nothing\n"
        "here is a real client, a real position or a real amount. They exist so\n"
        "a fresh deployment renders populated screens instead of empty ones.\n\n"
        "**Do not draw conclusions from these numbers.** Copy into\n"
        "`webapp/artifacts/` to use them, and replace them with output from a\n"
        "real scan as soon as the deployment has database access.\n",
        encoding="utf-8")

    print(f"wrote {len(written)} artefact(s) to {out}")
    for name in written:
        print(f"  {name}  ({(out / name).stat().st_size / 1024:.0f} KB)")
    print(f"  pseudonymised accounts: {len(anon._accounts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
