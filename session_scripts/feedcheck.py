import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import trade_feed

print("=== MySQL prod feed (trade_feed.poll) ===")
t = time.time()
try:
    op, cl = trade_feed.poll()
    print("poll() returned in %.1fs | openings=%d closings=%d" % (time.time() - t, len(op), len(cl)))
    for name, f in (("openings", op), ("closings", cl)):
        if len(f):
            tcol = next((c for c in ("event_time", "open_time", "close_time", "time") if c in f.columns), None)
            newest = pd.to_datetime(f[tcol]).max() if tcol else None
            print(f"  {name}: newest {tcol}={newest} | sample symbols={f['symbol'].head(3).tolist() if 'symbol' in f else '?'}")
        else:
            print(f"  {name}: EMPTY")
    # second poll to see if cursor advances / fresh flow
    time.sleep(3)
    op2, cl2 = trade_feed.poll()
    print("2nd poll (after 3s): openings=%d closings=%d" % (len(op2), len(cl2)))
except Exception as e:
    import traceback; traceback.print_exc()
