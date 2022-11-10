import akshare as ak
import numpy as np
import talib
from prettytable import PrettyTable

table = PrettyTable(["symbol", "time", "close", "ER", "trend", "trend_strength", "atr", "signal"])

# boli, caipo, luowen, pvc, chuanjian, niaosu, doupo, yumi
symbols = {
    "future": ["TA2301", "MA2301", "FG2301", "RM2301", "RB2301", "v2301", "sa2301", "ur2301", "m2301", "c2301"],
    "etf": ["sh513050", "sh515790", "sh510300", "sh512170", "sh512690", "sh513100", "sh588000", "sh510500"],
}


def calculate_indicator(df):
    """
    ER = Direction / Volatility

    Where:
    Direction = ABS(Close – Close[n])
    Volatility = sum(ABS(Close – Close[1]))
    n = The efficiency ratio period.
    """
    n = 50
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
    trend_strength = sum([(np.abs(close[-1]) - np.abs(i)) / np.abs(close[-1]) for i in trend_dict.values()])
    trend_dict = sorted(trend_dict.items(), key=lambda x: x[1])
    trend = " < ".join([i[0] for i in trend_dict])

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

    signals = ["-", "↑", "↓"]

    table.add_row(
        [
            symbol,
            df.datetime.values[-1],
            close[-1],
            k * round(er, 2),
            trend,
            round(trend_strength, 2),
            round(atr[-1], 2),
            signals[0],
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

