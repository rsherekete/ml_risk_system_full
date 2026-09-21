"""Fill holes in the warehouse, and survive a VPN that comes and goes.

WHY A SEPARATE REPAIR PATH

`mysql_extract.backfill` walks a whole window month by month. That is right for
a first load and wrong for a repair: re-fetching twenty-five months to recover
nine wastes an hour of VPN time and rewrites files that were already correct.

It also assumes the network stays up. It does not. A backfill of mt5_live01 ran
for over an hour, degraded (one month took 65 minutes against a normal 30
seconds), and then failed outright when the tunnel dropped -- leaving that
server with sixteen of its twenty-five months and a nine-month hole through the
most recent and busiest period. Because the partitions had been cleared first,
the warehouse was briefly in a worse state than before the repair began.

So this module does three things differently:

* it asks `data_store.coverage` which months are ACTUALLY missing and fetches
  only those, so it is safe and cheap to re-run;
* it waits for the VPN rather than failing when the host stops resolving, and
  keeps waiting, because the useful behaviour overnight is to finish when the
  link returns rather than to give up at the first drop;
* it writes each month as it arrives, so an interruption costs one month rather
  than the whole run.
"""

from __future__ import annotations

import socket
import time
from datetime import datetime, timezone

import pandas as pd

#: The host every MySQL extract goes through.
DB_HOST = "ld4-dbproxy.in.zfx.loc"


def vpn_up(host: str = DB_HOST) -> bool:
    """Does the database host resolve? The cheapest proxy for 'VPN is up'."""
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


def wait_for_vpn(timeout_seconds: float = 3600.0, poll_seconds: float = 30.0,
                 progress=None) -> bool:
    """Block until the tunnel returns, or give up after `timeout_seconds`."""
    deadline = time.time() + timeout_seconds
    announced = False
    while time.time() < deadline:
        if vpn_up():
            if announced and progress:
                progress("VPN back up, resuming")
            return True
        if not announced and progress:
            progress(f"VPN down -- waiting up to {timeout_seconds / 60:.0f} min")
            announced = True
        time.sleep(poll_seconds)
    return False


def month_bounds(period: str) -> tuple[datetime, datetime]:
    """(start, end) datetimes for a 'YYYY-MM' partition label."""
    start = pd.Period(period, freq="M").start_time
    end = pd.Period(period, freq="M").end_time
    return (start.to_pydatetime().replace(tzinfo=timezone.utc),
            (end + pd.Timedelta(nanoseconds=1)).to_pydatetime().replace(tzinfo=timezone.utc))


def repair(database: str | None = None, progress=None,
           vpn_timeout: float = 3600.0) -> dict:
    """Fetch every month the warehouse is missing, waiting out VPN drops.

    Returns what it filled and what it could not, rather than raising: a repair
    that fixed eight of nine months has done real work and the caller needs to
    know which one is still open.
    """
    from webapp import data_store, mysql_extract

    def say(message):
        if progress:
            progress(message)

    filled, failed, rows_total = [], [], 0
    coverage = data_store.coverage()
    targets = [s for s in coverage["servers"]
               if s["gaps"] and (database is None or s["server"] == database)]

    if not targets:
        say("no gaps to repair")
        return {"filled": [], "failed": [], "rows": 0}

    for server in targets:
        name = server["server"]
        if name not in mysql_extract.MYSQL_DATABASES:
            # dubai is BigQuery-only; its repair path is `dubai_backfill`.
            say(f"{name}: {len(server['gaps'])} gaps, but not a MySQL server -- skipped")
            failed.extend(f"{name} {m}" for m in server["gaps"])
            continue

        say(f"{name}: {len(server['gaps'])} missing months to fetch")
        for period in server["gaps"]:
            if not vpn_up() and not wait_for_vpn(vpn_timeout, progress=progress):
                say(f"{name} {period}: VPN never returned -- stopping")
                failed.append(f"{name} {period}")
                continue

            start, end = month_bounds(period)
            frame = None
            for attempt in range(5):
                try:
                    began = time.time()
                    frame = mysql_extract.fetch_month(name, start, end)
                    say(f"{name} {period}: {len(frame):,} rows in {time.time() - began:.0f}s")
                    break
                except Exception as error:
                    # A dropped tunnel is not a query error -- wait for it rather
                    # than burning the remaining attempts against a dead socket.
                    if not vpn_up():
                        wait_for_vpn(vpn_timeout, progress=progress)
                        continue
                    if attempt == 4:
                        say(f"{name} {period}: FAILED -- {type(error).__name__}")
                        failed.append(f"{name} {period}")
                    else:
                        time.sleep(10 * (attempt + 1))
            if frame is None or frame.empty:
                continue

            data_store.write_partitions(name, frame, time_column="close_time")
            rows_total += len(frame)
            filled.append(f"{name} {period}")

    if rows_total:
        # Only touch the watermark for servers that are now whole; a partial
        # repair that claimed completeness would hide the remaining hole.
        after = {s["server"]: s for s in data_store.coverage()["servers"]}
        for server in targets:
            entry = after.get(server["server"])
            if entry and entry["complete"]:
                data_store.set_watermark(
                    server["server"], datetime.now(timezone.utc), entry["rows"],
                    detail="gap repair")

    return {"filled": filled, "failed": failed, "rows": rows_total}
