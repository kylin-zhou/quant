import akshare as ak
import numpy as np
import talib
from prettytable import PrettyTable

table = PrettyTable(["symbol", "time", "ER", "close", "trend", "macdhist", "kd_cross", "macd_cross", "atr", "signal"])

symbols = ["MA2301", "TA2301", "RM2301", "RB2301"]

for symbol in symbols:
    df = ak.futures_zh_minute_sina(symbol=symbol, period="30")
    """
    ER = Direction / Volatility

    Where:
    Direction = ABS(Close – Close[n])
    Volatility = sum(ABS(Close – Close[1]))
    n = The efficiency ratio period.
    """
    n = 20
    close, high, low = df.close.values, df.high.values, df.low.values

    direction = np.abs(close[-1] - close[-n])
    volatility = np.abs(close[-n:] - np.array([0] + close[-n:-1].tolist()))
    er = direction / sum(volatility)

    k = 1 if close[-1] > close[-n] else -1

    # y:close, y_hat:kx+b, y_mean:avg(close)
    # y_hat = np.array([(k * direction / n * i + close[-n]) for i, price in enumerate(close[-n:])])
    # sse = sum((close[-n:] - y_hat) ** 2)
    # sst = sum((close[-n:] - np.mean(close[-n:])) ** 2)
    # r2 = 1 - sse / sst

    ma20 = talib.SMA(close, 20)
    ma50 = talib.SMA(close, 50)
    ma100 = talib.SMA(close, 100)
    ma200 = talib.SMA(close, 200)
    macd, macdsignal, macdhist = talib.MACD(close)
    slowk, slowd = talib.STOCH(high, low, close)
    atr = talib.ATR(high, low, close, timeperiod=20)

    # "symbol", "up/down", "ER", "close", "ma20","ma50","ma100","ma200","macdhist","kdj","atr", "1%"
    # signal, trend, momentum
    trend_dict = {"close": close[-1], "ma20": ma20[-1], "ma50": ma50[-1], "ma100": ma100[-1], "ma200": ma200[-1]}
    trend_dict = sorted(trend_dict.items(), key=lambda x: x[1])
    trend = " > ".join([i[0] for i in trend_dict])
    kd_cross = ""
    if slowk[-1] > slowd[-1] and slowk[-1] < slowd[-1]:
        kd_cross = "Golden"
    elif slowk[-1] < slowd[-1] and slowk[-1] > slowd[-1]:
        kd_cross = "Death"

    macd_hist = ""
    if macdhist[-1] > macdhist[-2]:
        macd_hist = "up"
    elif macdhist[-1] < macdhist[-2]:
        macd_hist = "down"

    macd_cross = ""
    if macdhist[-1] > 0 and macdhist[-2] < 0:
        macd_cross = "Golden"
    elif macdhist[-1] < 0 and macdhist[-2] > 0:
        macd_cross = "Death"

    signals = ["↑", "↓"]

    table.add_row(
        [
            symbol,
            df.datetime.values[-1],
            k * round(er, 2),
            close[-1],
            trend,
            macd_hist,
            kd_cross,
            macd_cross,
            round(atr[-1], 2),
            signals[0],
        ]
    )

print(table)
