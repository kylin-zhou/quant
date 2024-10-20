import sys
import akshare as ak
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
# from prettytable import PrettyTable

# table = PrettyTable(["symbol", "time", "ER","trend","atr"])

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

def get_volatility(df, n):
    open, close, high, low = df.open.values[-n:], df.close.values[-n:], df.high.values[-n:], df.low.values[-n:]

    avg = (open + close) / 2
    max_min = (high - low)+1
    day_vola = np.max([np.abs(avg-high), np.abs(avg-low)], axis=0)
    day_vola = [raw/base for raw,base in zip(day_vola,max_min)]
    vola = np.mean(day_vola)
    return vola


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

def kdj(df, n=9, m1=3, m2=3):
    """
    计算 KDJ 指标
    """
    # 计算最低价的 N 日移动最小值
    low_list = df["low"].rolling(window=n, min_periods=1).min()
    # 计算最高价的 N 日移动最大值
    high_list = df["high"].rolling(window=n, min_periods=1).max()
    # 计算当日的 RSV 值
    rsv = (df["close"] - low_list) / (high_list - low_list) * 100

    # 计算 K 值
    k = np.zeros_like(rsv)
    k[0] = 50  # 初始值为 50
    for i in range(1, len(rsv)):
        k[i] = (2 * k[i - 1] + rsv[i]) / 3
    # 计算 D 值
    d = np.zeros_like(rsv)
    d[0] = 50  # 初始值为 50
    for i in range(1, len(rsv)):
        d[i] = (2 * d[i - 1] + k[i]) / 3

    # 计算 J 值
    j = 3 * k - 2 * d
    # 将 K 值、D 值和 J 值存入 DataFrame 中
    df["kdj_k"] = k
    df["kdj_d"] = d
    df["kdj_j"] = j
    return df

def calculate_long_short(df):
    # 计算 MACD 指标
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
        df["close"].values, fastperiod=20, slowperiod=50, signalperiod=15
    )

    # 计算 KDJ 指标
    kdj(df)
#     df["kdj_k"], df["kdj_d"] = talib.STOCH(
#         df["high"].values, df["low"].values, df["close"].values, fastk_period=9, slowk_period=3, slowd_period=3
#     )

    # 均线指标
    df["ma0"] = talib.SMA(df["close"].values, 50)
    df["ma1"] = talib.SMA(df["close"].values, 120)
    df["ma2"] = talib.SMA(df["close"].values, 150)

    # 创建long和short列，初始值为false
    df["signal"] = "--"
    position = False
    for i in range(1, len(df)):
        # 如果在0轴上方macd金叉或macdhist大于0且kdj金叉，将long对应的行变为true
        if (
            df.loc[i, "macd_hist"] > 0
            and df.loc[i, "kdj_k"] > df.loc[i, "kdj_d"]
            and df.loc[i - 1, "kdj_k"] < df.loc[i - 1, "kdj_d"]
            and df.loc[i,"close"] > df.loc[i,"ma1"]
        ):
            df.loc[i, "signal"] = "long"
        if df.loc[i, "ma1"] > df.loc[i, "ma2"] and df.loc[i, "macd_hist"] > 0 and df.loc[i - 1, "macd_hist"] < 0:
            df.loc[i, "signal"] = "long"
            
        # 如果在0轴下方macd死叉或macdhist小于0且kdj死叉，将short对应的行变为ture
        if (
            df.loc[i, "macd_hist"] < 0
            and df.loc[i, "kdj_k"] < df.loc[i, "kdj_d"]
            and df.loc[i - 1, "kdj_k"] > df.loc[i - 1, "kdj_d"]
            and df.loc[i,"close"] < df.loc[i,"ma1"]
        ):
            df.loc[i, "signal"] = "short"
        if df.loc[i, "ma1"] < df.loc[i, "ma2"] and df.loc[i, "macd_hist"] < 0 and df.loc[i - 1, "macd_hist"] > 0:
            df.loc[i, "signal"] = "short"

    return df


def calculate_indicator_sar(df):
    # 计算 MACD 指标
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
        df["close"].values, fastperiod=20, slowperiod=50, signalperiod=10
    )

    df["sar"] = talib.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)

    # 均线指标
    df["ma0"] = talib.SMA(df["close"].values, 50)
    df["ma1"] = talib.SMA(df["close"].values, 120)
    df["ma2"] = talib.SMA(df["close"].values, 150)

    # 平均穿透

    # 创建long和short列，初始值为false
    df["signal"] = "--"
    position = False
    for i in range(150, len(df)):
        # macd 和 sar 金，做多，将long对应的行变为true
        if df.loc[i, "ma1"] > df.loc[i, "ma2"] and (
            (
                df.loc[i, "macd_hist"] > 0
                and df.loc[i, "close"] > df.loc[i, "sar"]
                and df.loc[i - 1, "close"] < df.loc[i - 1, "sar"]
            )
            or (df.loc[i, "close"] > df.loc[i, "sar"] and df.loc[i, "macd_hist"] > 0 and df.loc[i - 1, "macd_hist"] < 0)
        ):
            df.loc[i, "signal"] = "long"
        # 将short对应的行变为ture
        if df.loc[i, "ma1"] < df.loc[i, "ma2"] and (
            (
                df.loc[i, "macd_hist"] < 0
                and df.loc[i, "close"] < df.loc[i, "sar"]
                and df.loc[i - 1, "close"] > df.loc[i - 1, "sar"]
            )
            or (df.loc[i, "close"] < df.loc[i, "sar"] and df.loc[i, "macd_hist"] < 0 and df.loc[i - 1, "macd_hist"] > 0)
        ):
            df.loc[i, "signal"] = "short"

    return df

def analyze_win_rate(df):
    win_rate = {}
    for days in [10, 20, 30]:
        long_win_count, short_win_count = 0, 0
        long_total_count, short_total_count = 0, 0
        long_up_changes, long_down_changes = [0], [0]
        short_up_changes, short_down_changes = [0], [0]
        for i in range(len(df) - days):
            if df.loc[i, "signal"] == "long":
                long_total_count += 1
                if df.loc[i + days, "close"] > df.loc[i, "close"]+1:
                    long_win_count += 1

                change = df.loc[i + days, "close"] / df.loc[i, "close"] - 1
                if change > 0:
                    long_up_changes.append(change)
                else:
                    long_down_changes.append(change)

            elif df.loc[i, "signal"] == "short":
                short_total_count += 1
                if df.loc[i + days, "close"] < df.loc[i, "close"]-1:
                    short_win_count += 1

                change = df.loc[i + days, "close"] / df.loc[i, "close"] - 1
                if change > 0:
                    short_up_changes.append(change)
                else:
                    short_down_changes.append(change)

        rate = (long_win_count + short_win_count) / (long_total_count + short_total_count)
        win_rate[days] = rate

        print(
            "The {} win rate is: {:.2f}% long win rate: {:.2f} short win rate:{:.2f}".format(
                days, rate * 100, long_win_count / long_total_count, short_win_count / short_total_count
            )
        )

        print(
            "long avg up:{:.4f} long avg down:{:.4f} short avg up:{:.4f} short avg down:{:.4f}".format(
                sum(long_up_changes) / len(long_up_changes),
                sum(long_down_changes) / len(long_down_changes),
                sum(short_up_changes) / len(short_up_changes),
                sum(short_down_changes) / len(short_down_changes),
            )
        )

    return win_rate



def plot_signal(df, name=""):
    states_buy, states_sell = [], []
    for i in range(len(df)):
        if df.loc[i, "signal"] == "long":
            states_buy.append(i)
        elif df.loc[i, "signal"] == "short":
            states_sell.append(i)

    close = df["close"]
    fig = plt.figure(dpi=300,figsize=(30, 10))
    plt.plot(close, color="r", lw=2.0)
    plt.plot(df["ma1"], color="y", lw=2.0, label="ma1")
    plt.plot(df["ma2"], color="b", lw=2.0, label="ma2")
    plt.plot(close, "^", markersize=10, color="m", label="buying signal", markevery=states_buy)
    plt.plot(close, "v", markersize=10, color="k", label="selling signal", markevery=states_sell)
    plt.title(f"{name} buy/sell signal")

    # 添加网格虚线栅格，网格间隔为价格的1%
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.yticks(range(int(min(close)), int(max(close)+1), int(max(close)*0.01)))
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)

    plt.legend()
    plt.savefig(f"tmp/{name}.png")

def plot_f(df, name=""):
    import mplfinance as mpf

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    # 准备交易信号数据，这里假设有三种信号：开多仓，开空仓
    df["buy"] = np.nan
    df.loc[df["signal"] == "long", "buy"] = df.loc[df["signal"] == "long", "low"]
    df["sell"] = np.nan
    df.loc[df["signal"] == "short", "sell"] = df.loc[df["signal"] == "short", "high"]


    apd = [
        mpf.make_addplot(df["buy"], scatter=True, markersize=30, marker='^', color='m'),
        mpf.make_addplot(df["sell"], scatter=True, markersize=30, marker='v', color='k')
    ]

    save = dict(fname=f"tmp/{name}.png", dpi=300, bbox_inches='tight')

    mpf.plot(
        data=df,
        type="candle",
        title=f"{name} trading signals",
        ylabel="price",
        style="binance",
        volume=True,
        mav=(50, 120, 150),
        addplot=apd,
        tight_layout=False, 
        savefig=save,
        figsize=(30, 15),
        warn_too_much_data=10000
    )

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
    
    symbols = {
        "future": [
            "ma2305","v2305", "RB2310", "CF2309","sr2307", "eb2401", "pp2401","c2401","sp2401","i2401"
            ],
        # "etf": [
        # "sh513050", "sh515790", "sh512170", "sh512690",
        # "sh510300", "sh588000", "sh510500","sz159915"],
    }
    
    for market in symbols.keys():
        for symbol in symbols[market]:
            print(symbol)
            if market == "future":
                df = ak.futures_zh_minute_sina(symbol=symbol, period="30").iloc[-1000:, :6].reset_index(drop=True)
            else:
                df = ak.fund_etf_hist_sina(symbol=symbol).iloc[:, :6]
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]

            # # print(df.head(1), "\n", df.tail(1))
            # backtest(df)

            # # plot_signal(df, name=symbol)
            # plot_f(df, name=symbol)

            # df.drop(["ma1", "ma2"], axis=1).round(5).to_csv(f"tmp/{symbol}.csv", index=False)

            print("volatility:\t", get_volatility(df, 100))

    # print(table)

