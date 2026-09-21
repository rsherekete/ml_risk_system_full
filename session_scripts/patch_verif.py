import json
p = r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\rule_verification.json"
d = json.loads(open(p).read())
for k, r in d.get("rules", {}).items():
    pf = r.get("predicted_flag_active"); af = r.get("actually_flagged")
    rec = r.get("recall"); prec = r.get("precision")
    if pf is None or af is None:
        continue
    tp = round(rec * af) if rec is not None else None
    r["true_positives"] = tp
    r["scope"] = ("BACKFILLED ML model over ALL active accounts "
                  "(distinct from the tape-scan flagged table below)")
    r["explain"] = (
        f"BACKFILLED walk-forward (model cutoff {d.get('model_cutoff')}, "
        f"never saw these days). Latest day "
        f"{r.get('day') or d.get('verified_against_day')}: ML model over all "
        f"{r.get('active_accounts')} active accounts predicted {pf} would "
        f"flag, {af} actually did; {tp} correct -> caught {tp}/{af} real "
        f"flags (recall {rec}) and {tp}/{pf} picks right (precision {prec}). "
        f"14-day mean P {r.get('mean_precision')} / R {r.get('mean_recall')}.")
open(p, "w").write(json.dumps(d))
for k, r in d["rules"].items():
    print(k, "tp", r["true_positives"], "of", r["actually_flagged"],
          "actual /", r["predicted_flag_active"], "predicted")
