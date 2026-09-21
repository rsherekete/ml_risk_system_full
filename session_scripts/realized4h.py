import json, time
J = json.load(open(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\status.json", encoding="utf-8-sig"))
now = time.time(); win = now - 4*3600
for key in ("s1","s2"):
    perf = J["report"]["strategies"][key]["perf"]
    rec = perf.get("recent", [])
    span = [r["close_time"] for r in rec if r.get("close_time")]
    print(f"=== {key.upper()} recent closes: {len(rec)} (span {(max(span)-min(span))/3600:.1f}h)" if span else f"=== {key.upper()} no closes")
    r4 = [r for r in rec if r.get("close_time",0) >= win]
    if not r4:
        print("  none closed in last 4h in the recent buffer"); continue
    net = sum(r["net"] for r in r4)
    exp = sum(r["expected"] for r in r4)
    cli = sum(r["client_equiv"] for r in r4)
    cost = sum(r["cost"] for r in r4)
    wins = sum(1 for r in r4 if r["net"]>0)
    matched = [r for r in r4 if r["expected"]]
    print(f"  closed in 4h (buffer): {len(r4)}  wins {wins} ({wins/len(r4):.0%})")
    print(f"  REALIZED net   : ${net:+,.2f}")
    print(f"  EXPECTED (matched {len(matched)}): ${exp:+,.2f}")
    print(f"  GAP realized-expected: ${net-exp:+,.2f}")
    print(f"  client @ our size: ${cli:+,.2f}   cost(swap+comm): ${cost:+,.2f}")
    # biggest losers
    worst = sorted(r4, key=lambda r: r["net"])[:6]
    print("  biggest realized losers:")
    for r in worst:
        print(f"    {r['symbol']:10s} net {r['net']:+8.2f}  exp {r['expected']:+7.2f}  cost {r['cost']:+6.2f}  lots {r['our_lots']}")
    print()
