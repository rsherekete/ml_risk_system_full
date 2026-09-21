import MetaTrader5 as mt5, time
mt5.initialize()
print("Market Watch symbols total:", mt5.symbols_total())
for s in ("XAUUSD", "GBPJPY", "EURUSD", "USDJPY", "GBPUSD", "XAUUSDe"):
    info = mt5.symbol_info(s)
    if info is None:
        print(f"  {s}: NOT FOUND on this account")
        continue
    tick = mt5.symbol_info_tick(s)
    age = (time.time() - tick.time) if tick and tick.time else None
    print(f"  {s}: visible={info.visible} "
          f"bid={getattr(tick,'bid',None)} ask={getattr(tick,'ask',None)} "
          f"tick_age={round(age,1) if age is not None else None}s "
          f"trade_mode={info.trade_mode}")
a = mt5.account_info()
print("account:", a.login, "| server:", a.server, "| trade_allowed:",
      mt5.terminal_info().trade_allowed)
mt5.shutdown()
