import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import ad_refresh
print("corpus max day BEFORE:", ad_refresh.corpus_max_day())
result = ad_refresh.refresh(log=print)
print("RESULT:", result)
print("corpus max day AFTER:", ad_refresh.corpus_max_day())
