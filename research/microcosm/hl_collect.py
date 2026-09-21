"""Hyperliquid public-data collectors -- the scored tape, the ladder and the
matrix from a venue where every participant is a public address.

  python hl_collect.py candles --coins xyz:GOLD,BTC,ETH --since 2026-03-16
  python hl_collect.py fills   --coins xyz:GOLD,BTC,ETH,xyz:SILVER,xyz:SP500 --top 12000 --since 2026-03-16
  python hl_collect.py live    --coins xyz:GOLD,BTC,ETH          # runs until stopped

fills:   the leaderboard's most active addresses (by monthly volume), each
         address's fills since `since` paginated through userFillsByTime,
         kept for the requested coins. Every fill carries the address, coin,
         side, price, size, time, the position BEFORE the fill, the direction
         (Open Long / Close Short ...), closed P&L and fee -- a trade log with
         identities. Resumable; checkpoints every 100 addresses.
live:    websocket trades (with buyer and seller addresses) for each coin,
         plus the L2 book sampled once a minute -> daily parquet files.
Rate budget: the info endpoint allows ~1200 weight/min per IP; fills pages
weigh 20, so the puller paces itself at ~50 pages/min.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SCRATCH = Path(__file__).resolve().parent
OUT = SCRATCH / "hl"
OUT.mkdir(exist_ok=True)
API = "https://api.hyperliquid.xyz/info"
WS = "wss://api.hyperliquid.xyz/ws"
LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
PAGE = 2000


SESSION = requests.Session()


def bind_source(ip: str) -> None:
    """Route this process's requests through one local interface (a second
    public IP doubles the per-IP rate budget)."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.connection import create_connection as _cc

    class _Bound(HTTPAdapter):
        def init_poolmanager(self, *a, **kw):
            kw["source_address"] = (ip, 0)
            return super().init_poolmanager(*a, **kw)
    SESSION.mount("https://", _Bound())
    print(f"requests bound to source ip {ip}", flush=True)


def info(body: dict, retries: int = 6) -> object:
    for attempt in range(retries):
        try:
            r = SESSION.post(API, json=body, timeout=30)
        except requests.RequestException:
            time.sleep(2 + attempt * 2); continue
        if r.status_code == 429:
            time.sleep(5 + attempt * 5); continue
        if r.status_code >= 500:
            time.sleep(2 + attempt * 2); continue
        if r.status_code != 200:
            raise RuntimeError(f"{body.get('type')}: {r.status_code} {r.text[:160]}")
        return r.json()
    raise RuntimeError(f"{body.get('type')}: gave up")


# ------------------------------------------------------------------ candles
def candles(coins: list[str], since: str) -> None:
    start = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000); now = int(time.time() * 1000)
    for coin in coins:
        rows = []; t = start
        while t < now:
            c = info({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m", "startTime": t, "endTime": min(t + 5000 * 60_000, now)}})
            if not c:
                t += 5000 * 60_000; continue
            rows += c
            t = int(c[-1]["t"]) + 60_000
            time.sleep(0.3)
        df = pd.DataFrame(rows).drop_duplicates("t").sort_values("t")
        df["minute"] = pd.to_datetime(df["t"], unit="ms")
        for k in ("o", "h", "l", "c", "v"):
            df[k] = df[k].astype(float)
        df[["minute", "o", "h", "l", "c", "v", "n"]].to_parquet(OUT / f"candles_{coin.replace(':', '_')}_1m.parquet", index=False)
        print(f"{coin}: {len(df):,} 1m candles {df['minute'].min()} .. {df['minute'].max()}", flush=True)


# ------------------------------------------------------------------ fills
def _window(row: dict, name: str, key: str) -> float:
    for w in row.get("windowPerformances", []):
        if w[0] == name:
            return float(w[1].get(key) or 0)
    return 0.0


def fills(coins: list[str], top: int, since: str, max_pages: int = 8, min_vlm: float = 5e4, max_vlm: float = 3e7,
          seed: int = 1, shard: str = "0/1") -> None:
    """Addresses in the ACTIVE-TRADER band of monthly volume (the very top of
    the leaderboard is market makers with millions of fills -- slow to pull and
    not the population the tape reads), shuffled so every checkpoint is a
    representative sample, each capped at `max_pages` x 2,000 fills.
    `shard` k/n takes every n-th address starting at k so several pullers
    (on different IPs) split the work; every shard keeps its own done list."""
    import random
    lb = requests.get(LEADERBOARD, timeout=120).json().get("leaderboardRows", [])
    band = [r for r in lb if min_vlm <= _window(r, "month", "vlm") <= max_vlm]
    random.Random(seed).shuffle(band)
    addresses = [r["ethAddress"] for r in band[:top]]
    k, n = (int(x) for x in shard.split("/"))
    addresses = addresses[k::n]
    print(f"fills: {len(band):,} addresses in the ${min_vlm:,.0f}..${max_vlm:,.0f} monthly-volume band; shard {shard} -> {len(addresses):,}", flush=True)
    # every done list (any shard) counts, so a re-sharded run never repeats an address
    done: set = set()
    for p in OUT.glob("fills_done*.txt"):
        done |= set(p.read_text().split())
    done_path = OUT / (f"fills_done_{k}of{n}.txt" if n > 1 else "fills_done.txt")
    mine = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [a for a in addresses if a not in done]
    print(f"fills: {len(addresses):,} addresses selected, {len(done):,} already done, {len(todo):,} to pull", flush=True)
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    coin_set = set(coins)
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    tag = f"s{k}of{n}_" if n > 1 else ""
    state = {"n_req": 0, "kept": 0, "buffer": [], "part": len(list(OUT.glob(f"fills_part_{tag}*.parquet"))), "done_n": 0}
    t0 = time.time()
    budget = 55.0                      # pages per minute across all workers (weight 20 each, limit ~1200/min)

    def pull(addr: str) -> list[dict]:
        rows = []; t = since_ms; pages = 0
        while pages < max_pages:
            with lock:
                elapsed = time.time() - t0
                if state["n_req"] / max(elapsed, 1e-9) * 60 > budget:
                    wait = state["n_req"] / (budget / 60) - elapsed
                else:
                    wait = 0.0
                state["n_req"] += 1
            if wait > 0:
                time.sleep(min(wait, 5.0))
            try:
                page = info({"type": "userFillsByTime", "user": addr, "startTime": t})
            except RuntimeError as error:
                print(f"{addr[:10]}: {error}", flush=True); break
            pages += 1
            if not page:
                break
            for f in page:
                if f.get("coin") in coin_set:
                    rows.append({"address": addr, "coin": f["coin"], "side": f["side"], "px": float(f["px"]), "sz": float(f["sz"]),
                                 "time": int(f["time"]), "start_position": float(f.get("startPosition") or 0), "dir": f.get("dir"),
                                 "closed_pnl": float(f.get("closedPnl") or 0), "fee": float(f.get("fee") or 0), "crossed": bool(f.get("crossed")),
                                 "liquidation": bool(f.get("liquidation")), "tid": int(f.get("tid") or 0)})
            if len(page) < PAGE:
                break
            t = int(page[-1]["time"]) + 1
        return rows

    def finish(addr: str, rows: list[dict]) -> None:
        with lock:
            state["buffer"] += rows; state["kept"] += len(rows); mine.add(addr); state["done_n"] += 1
            m_ = state["done_n"]
            if m_ % 100 == 0 or m_ == len(todo):
                if state["buffer"]:
                    pd.DataFrame(state["buffer"]).to_parquet(OUT / f"fills_part_{tag}{state['part']:04d}.parquet", index=False)
                    state["part"] += 1; state["buffer"] = []
                done_path.write_text("\n".join(sorted(mine)))
                print(f"fills: {m_:,}/{len(todo):,} addresses, {state['kept']:,} fills kept, {state['n_req']:,} requests, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for addr, rows in zip(todo, pool.map(pull, todo)):
            finish(addr, rows)
    print("fills: DONE", flush=True)


# ------------------------------------------------------------------ live
async def _live(coins: list[str]) -> None:
    import websockets
    trades_buf: dict[str, list] = {c: [] for c in coins}
    books: dict[str, dict] = {}
    last_flush = time.time(); last_book = 0.0
    async def flush():
        day = time.strftime("%Y-%m-%d", time.gmtime())
        for c, rows in trades_buf.items():
            if rows:
                p = OUT / f"live_trades_{c.replace(':', '_')}_{day}.parquet"
                df = pd.DataFrame(rows)
                if p.exists():
                    df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
                df.to_parquet(p, index=False); trades_buf[c] = []
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, max_size=None) as ws:
                for c in coins:
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": c}}))
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": c}}))
                print(f"live: subscribed {coins}", flush=True)
                async for raw in ws:
                    msg = json.loads(raw)
                    ch = msg.get("channel"); data = msg.get("data")
                    if ch == "trades" and isinstance(data, list):
                        for t in data:
                            c = t.get("coin")
                            if c in trades_buf:
                                u = t.get("users") or ["", ""]
                                trades_buf[c].append({"time": int(t["time"]), "side": t["side"], "px": float(t["px"]), "sz": float(t["sz"]),
                                                      "buyer": u[0], "seller": u[1], "tid": int(t.get("tid") or 0)})
                    elif ch == "l2Book" and isinstance(data, dict):
                        books[data.get("coin")] = data
                    now = time.time()
                    if now - last_book >= 60 and books:
                        day = time.strftime("%Y-%m-%d", time.gmtime())
                        rows = [{"time": int(now * 1000), "coin": c, "levels": json.dumps(b.get("levels"))} for c, b in books.items()]
                        p = OUT / f"live_books_{day}.parquet"
                        df = pd.DataFrame(rows)
                        if p.exists():
                            df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
                        df.to_parquet(p, index=False); last_book = now
                    if now - last_flush >= 60:
                        await flush(); last_flush = now
        except Exception as error:
            print(f"live: reconnecting after {type(error).__name__}: {error}", flush=True)
            await asyncio.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["candles", "fills", "live"])
    ap.add_argument("--coins", default="xyz:GOLD,BTC,ETH")
    ap.add_argument("--since", default="2026-03-16")
    ap.add_argument("--top", type=int, default=12000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--source-ip", default="")
    args = ap.parse_args()
    coins = [c for c in args.coins.split(",") if c]
    if args.source_ip:
        bind_source(args.source_ip)
    if args.command == "candles":
        candles(coins, args.since)
    elif args.command == "fills":
        fills(coins, args.top, args.since, shard=args.shard)
    else:
        asyncio.run(_live(coins))


if __name__ == "__main__":
    main()
