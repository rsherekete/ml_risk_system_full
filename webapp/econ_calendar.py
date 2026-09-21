"""Dynamic economic-events calendar.

Pulls the ForexFactory weekly calendar JSON feeds (last/this/next week) --
the same macro schedule investing.com renders -- normalises times to UTC,
and serves them to the AntiFraud tabs: the Event Impact window picker, the
latency profiles' event-proximity feature, and the News/Event/Vol category.

Every event carries a link to the investing.com economic calendar for the
human-readable view; the feed itself is the machine-readable source (no
public investing.com API exists, and their pages are scrape-blocked).
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "artifacts" / "econ_calendar.json"
FEEDS = [
    "https://nfs.faireconomy.media/ff_calendar_lastweek.json",
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
LDN = ZoneInfo("Europe/London")
INVESTING_URL = "https://www.investing.com/economic-calendar/"

_LOCK = threading.Lock()
_MEM: dict = {}


def _fetch() -> list[dict]:
    import requests
    rows: list[dict] = []
    for url in FEEDS:
        try:
            data = requests.get(url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"}).json()
            for e in data:
                try:
                    ts = datetime.fromisoformat(str(e.get("date")))
                    ts_utc = ts.astimezone(timezone.utc)
                except Exception:
                    continue
                rows.append({
                    "title": str(e.get("title") or ""),
                    "country": str(e.get("country") or ""),
                    "impact": str(e.get("impact") or ""),
                    "utc": ts_utc.isoformat(),
                    "ldn": ts_utc.astimezone(LDN).strftime("%Y-%m-%d %H:%M:%S"),
                    "forecast": str(e.get("forecast") or ""),
                    "previous": str(e.get("previous") or ""),
                    "link": INVESTING_URL,
                    "source": "forexfactory",
                })
        except Exception:
            continue
    rows.sort(key=lambda r: r["utc"])
    return rows


HIST_PATH = ROOT / "artifacts" / "econ_calendar_history.json"


def _accumulate(rows: list[dict]) -> None:
    """Append newly-seen events to the FIXED history file. The live feed only
    serves three rolling weeks; the history file is what training targets and
    historical analysis pin to, and it only ever grows."""
    try:
        hist = json.loads(HIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        hist = {"rows": []}
    seen = {(r["utc"], r["title"]) for r in hist["rows"]}
    added = 0
    for r in rows:
        key = (r["utc"], r["title"])
        if key not in seen:
            hist["rows"].append(r)
            seen.add(key)
            added += 1
    if added:
        hist["rows"].sort(key=lambda r: r["utc"])
        HIST_PATH.write_text(json.dumps(hist), encoding="utf-8")


def _schedule_backfill() -> list[dict]:
    """Deterministically-scheduled majors for the training window BEFORE the
    accumulator existed: US Non-Farm Payrolls (first Friday, 13:30 London).
    Honest scope: only events whose timing is a fixed public rule are
    backfilled; everything else accrues from the live feed going forward."""
    out = []
    for month in range(1, 13):
        try:
            d = datetime(2026, month, 1, tzinfo=LDN)
        except ValueError:
            continue
        while d.weekday() != 4:
            d += timedelta(days=1)
        ts = d.replace(hour=13, minute=30)
        out.append({"title": "Non-Farm Employment Change (schedule-derived)",
                    "country": "USD", "impact": "High",
                    "utc": ts.astimezone(timezone.utc).isoformat(),
                    "ldn": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "forecast": "", "previous": "",
                    "link": INVESTING_URL, "source": "schedule"})
    return out


def historical_events() -> list[dict]:
    """The FIXED calendar: accumulated feed history + schedule-derived
    backfill, deduped -- stable for training and historical analysis."""
    try:
        hist = json.loads(HIST_PATH.read_text(encoding="utf-8"))["rows"]
    except Exception:
        hist = []
    seen = {(r["utc"][:16], r["title"][:20]) for r in hist}
    for r in _schedule_backfill():
        if (r["utc"][:16], r["title"][:20]) not in seen:
            hist.append(r)
    hist.sort(key=lambda r: r["utc"])
    return hist


def events(refresh: bool = False, max_age_s: float = 3600.0) -> list[dict]:
    """The three-week calendar, cached in memory and on disk for an hour."""
    now = time.time()
    with _LOCK:
        if not refresh and _MEM.get("rows") is not None \
                and now - _MEM.get("stamp", 0) < max_age_s:
            return _MEM["rows"]
    rows = _fetch()
    if rows:
        _accumulate(rows)
    if not rows:
        try:
            disk = json.loads(CACHE.read_text(encoding="utf-8"))
            rows = disk.get("rows", [])
        except Exception:
            rows = []
    else:
        try:
            CACHE.write_text(json.dumps({"rows": rows, "at": now}),
                             encoding="utf-8")
        except Exception:
            pass
    with _LOCK:
        _MEM.update(rows=rows, stamp=now)
    return rows


def high_impact(refresh: bool = False) -> list[dict]:
    return [e for e in events(refresh) if e["impact"].lower() == "high"]


def event_times_utc(min_impact: str = "high",
                    historical: bool = True) -> list[datetime]:
    """Naive-UTC datetimes of calendar events at or above the impact level --
    the anchor list for event-proximity features. `historical` uses the FIXED
    accumulated calendar (training-stable); else the live 3-week feed."""
    order = {"low": 0, "medium": 1, "high": 2}
    floor = order.get(min_impact.lower(), 2)
    src = historical_events() if historical else events()
    seen_live = {(e["utc"], e["title"]) for e in src}
    for e in events():
        if (e["utc"], e["title"]) not in seen_live:
            src.append(e)
    out = []
    for e in src:
        if order.get(e["impact"].lower(), -1) >= floor:
            try:
                out.append(datetime.fromisoformat(e["utc"])
                           .astimezone(timezone.utc).replace(tzinfo=None))
            except Exception:
                pass
    return sorted(out)


def near_share(times_utc, tolerance_minutes: float = 10.0,
               min_impact: str = "high") -> float:
    """Share of the given naive-UTC timestamps that fall within +/- tolerance
    of a calendar event. The latency profiles use it: entries clustered on
    macro prints are event-driven flow, not random scalping."""
    anchors = event_times_utc(min_impact)
    if not anchors or times_utc is None or not len(times_utc):
        return 0.0
    import numpy as np
    import pandas as pd
    stamps = pd.to_datetime(pd.Series(list(times_utc))).astype("datetime64[s]")
    a = np.array(sorted(anchors), dtype="datetime64[s]")
    t = stamps.to_numpy()
    idx = np.searchsorted(a, t)
    tol = np.timedelta64(int(tolerance_minutes * 60), "s")
    near = np.zeros(len(t), dtype=bool)
    for shift in (0, 1):
        j = np.clip(idx - shift, 0, len(a) - 1)
        near |= np.abs(t - a[j]) <= tol
    return float(near.mean())


def ldn_to_utc(text: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM[:SS]' as Europe/London wall time -> naive UTC."""
    text = str(text).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            local = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"unparseable time: {text!r}")
    return local.replace(tzinfo=LDN).astimezone(timezone.utc) \
        .replace(tzinfo=None)
