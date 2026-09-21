import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import vantage, agent

# 1) live config picked up the netting fix
cfg = vantage._load_config() if hasattr(vantage, "_load_config") else None
if cfg is None:
    # find the loader
    for name in ("load_config", "read_config", "current_config", "_config"):
        if hasattr(vantage, name):
            obj = getattr(vantage, name)
            cfg = obj() if callable(obj) else obj
            break
print("config loader used:", type(cfg).__name__ if cfg else "none")
if cfg is not None:
    print("  net_by_symbol:", getattr(cfg, "net_by_symbol", "?"),
          "| invert_mirror_only:", getattr(cfg, "invert_mirror_only", "?"),
          "| use_exit_model:", getattr(cfg, "use_exit_model", "?"),
          "| mode:", getattr(cfg, "mode", "?"))

# 2) agent tool-round limit + cancel plumbing
print("MAX_TOOL_ROUNDS:", agent.MAX_TOOL_ROUNDS)
print("has request_cancel:", hasattr(agent, "request_cancel"))
