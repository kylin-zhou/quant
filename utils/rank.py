import sys
import akshare as ak
import numpy as np
import talib
from prettytable import PrettyTable

table = PrettyTable(["symbol", "time", "ER","trend","atr"])

# boli, caipo, luowen, pvc, chuanjian, niaosu, doupo, yumi
symbols = {
    "future": [
        "MA2301", "v2301", "RB2301", "RM2301", "c2301", "m2301","fg2301", "l2301"
        ],
    # "etf": [
    # "sh513050", "sh515790", "sh510300", "sh512170", "sh512690",
    # "sh513100", "sh588000", "sh510500"],
}

def get_er(close, n):
    """
    ER = Direction / Volatility

    Where:
    Direction = ABS(Close – Close[n])
    Volatility = sum(ABS(Close – Close[1]))
    n = The efficiency ratio period.
    """

    direction = np.abs(close[-1] - close[-n])
    volatility = np.abs(close[-(n-1):-1] - close[-n:-2])
    er = direction / sum(volatility)
    k = 1 if close[-1] > close[-n] else -1

    return k*er

def get_r2(close, n):
    # y:close, y_hat:kx+b, y_mean:avg(close)
    direction = np.abs(close[-1] - close[-n])
    k = 1 if close[-1] > close[-n] else -1

    y_hat = np.array([(k * direction / n * i + close[-n]) for i, price in enumerate(close[-n:])])
    sse = sum((close[-n:] - y_hat) ** 2)
    sst = sum((close[-n:] - np.mean(close[-n:])) ** 2)
    r2 = 1 - sse / sst
    return r2

def calculate_indicator(df):
    close, high, low = df.close.values, df.high.values, df.low.values

    ma = talib.SMA(close, 3)
    ma50 = talib.SMA(close, 50)
    ma100 = talib.SMA(close, 100)
    atr = talib.ATR(high, low, close, timeperiod=20)

    er = get_er(ma, 100)

    trend_dict = {"close": close[-1], "ma50": ma50[-1], "ma100": ma100[-1]}
    trend_dict = sorted(trend_dict.items(), key=lambda x: x[1])
    trend = " < ".join([i[0] for i in trend_dict])

    signals = ["-", "↑", "↓"]

    table.add_row(
        [
            symbol,
            df.datetime.values[-1],
            round(er,3),
            trend,
            round(atr[-1],3),
        ]
    )


if __name__ == "__main__":
    for market in symbols.keys():
        for symbol in symbols[market]:
            if market == "future":
                df = ak.futures_zh_minute_sina(symbol=symbol, period="30").iloc[:, :6]
            else:
                df = ak.fund_etf_hist_sina(symbol=symbol).iloc[:, :6]
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
            calculate_indicator(df)

    print(table)

