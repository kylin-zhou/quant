import sys
import akshare as ak
import numpy as np
import talib
# from prettytable import PrettyTable

# table = PrettyTable(["symbol", "time", "ER","trend","atr"])

# 
symbols = {
    "future": [
        "MA2305", "v2305", "RB2305", "c2305"
        ],
    "etf": [
    "sh513050", "sh515790", "sh512170", "sh512690",
    "sh510300", "sh588000", "sh510500","sz159915"],
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


def calculate_long_short(df):
    # 计算 MACD 指标
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
        df["close"].values, fastperiod=20, slowperiod=50, signalperiod=15
    )

    # 计算 KDJ 指标
    df["kdj_k"], df["kdj_d"] = talib.STOCH(
        df["high"].values, df["low"].values, df["close"].values, fastk_period=9, slowk_period=3, slowd_period=3
    )
    df["ma"] = talib.SMA(df["close"].values, 20)

    # 创建long和short列，初始值为false
    df["long"] = False
    df["short"] = False
    for i in range(1, len(df)):
        # 如果在0轴上方macd金叉或macdhist大于0且kdj金叉，将long对应的行变为true
        if (
            df.loc[i, "macd_hist"] > 0
            and df.loc[i - 1, "macd_hist"] < 0
            and df.loc[i, "kdj_k"] > df.loc[i, "kdj_d"]
            and df.loc[i - 1, "kdj_k"] < df.loc[i - 1, "kdj_d"]
        ):
            df.loc[i, "long"] = True
        elif df.loc[i, "macd"] > 0 and df.loc[i, "macd_hist"] > 0 and df.loc[i - 1, "macd_hist"] < 0:
            df.loc[i, "long"] = True
        # 如果在0轴下方macd死叉或macdhist小于0且kdj死叉，将short对应的行变为ture
        elif (
            df.loc[i, "macd_hist"] < 0
            and df.loc[i - 1, "macd_hist"] > 0
            and df.loc[i, "kdj_k"] < df.loc[i, "kdj_d"]
            and df.loc[i - 1, "kdj_k"] > df.loc[i - 1, "kdj_d"]
        ):
            df.loc[i, "short"] = True
        elif df.loc[i, "macd"] < 0 and df.loc[i, "macd_hist"] < 0 and df.loc[i - 1, "macd_hist"] > 0:
            df.loc[i, "short"] = True

    return df


def analyze_win_rate(df):
    win_rate = {}
    for days in [5, 10, 20, 30]:
        long_win_count, short_win_count = 0, 0
        long_total_count, short_total_count = 0, 0
        long_up_changes, long_down_changes = [0], [0]
        short_up_changes, short_down_changes = [0], [0]
        for i in range(len(df) - days):
            if df.loc[i, "long"]:
                long_total_count += 1
                if df.loc[i + days, "close"] > df.loc[i, "close"]:
                    long_win_count += 1

                change = df.loc[i + days, "close"] / df.loc[i, "close"] - 1
                if change > 0:
                    long_up_changes.append(change)

            elif df.loc[i, "short"]:
                short_total_count += 1
                if df.loc[i + days, "close"] < df.loc[i, "close"]:
                    short_win_count += 1

                change = df.loc[i + days, "close"] / df.loc[i, "close"] - 1
                if change < 0:
                    short_down_changes.append(change)

        rate = (long_win_count + short_win_count) / (long_total_count + short_total_count)
        win_rate[days] = rate

        print(
            "The {} win rate is: {:.2f}% long win rate: {:.2f} short win rate:{:.2f}".format(
                days, rate * 100, long_win_count / long_total_count, short_win_count / short_total_count
            )
        )

        print(
            "long avg up:{:.4f} , short avg down:{:.4f}".format(
                np.mean(long_up_changes),
                np.mean(short_down_changes)
            )
        )

    return win_rate


def backtest(df):
    """get buy/sell signal 5/10/20/50 Win/Loss Ratio

    long:
        1. macd_hist > 0, kdj gold
        2. macd_dif > 0, macd gold

    short:
        1. macd_hist < 0, kdj dead
        2. macd_dif < 0, macd dead
    """

    # 新增 long 和 short 列
    df = calculate_long_short(df)

    # 计算 long 变为 True 时的价格变化
    win_rate = analyze_win_rate(df)

if __name__ == "__main__":
    for market in symbols.keys():
        for symbol in symbols[market]:
            print(symbol)
            if market == "future":
                df = ak.futures_zh_minute_sina(symbol=symbol, period="30").iloc[:, :6]
            else:
                df = ak.fund_etf_hist_sina(symbol=symbol).iloc[:, :6]
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]

            # calculate_indicator(df)
            backtest(df)

    # print(table)

