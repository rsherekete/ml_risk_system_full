import sys, time, socket
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import symbol_specs, data_store
servers = tuple(p.name for p in data_store.WAREHOUSE.iterdir() if p.is_dir())
r = symbol_specs.refresh_spec_cache(servers, progress=lambda m: print("  ", m, flush=True))
print(f"cached {r['rows']:,} spec rows | failures: {r['failures']}")
specs, rates = symbol_specs.load_all_specs(servers, allow_remote=False)
print(f"from cache: {len(specs):,} spec rows | {len(rates):,} FX rates")
