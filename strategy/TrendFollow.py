import numpy as np
import talib

for symbol in ["FG9999.XZCE", "MA9999.XZCE", "TA9999.XZCE", "UR9999.XZCE", "RM9999.XZCE", "C9999.XDCE", "V9999.XDCE"]:
    df = get_bars(
        symbol,
        12 * 20,
        unit="30m",
        fields=["date", "open", "high", "low", "close"],
        include_now=False,
        end_dt="2022-12-31",
    )
    # print(df.head())

    """
    ER = Direction / Volatility

    Where:
    Direction = ABS(Close – Close[n])
    Volatility = sum(ABS(Close – Close[1]))
    n = The efficiency ratio period.
    """
    n = 90
    close, high, low = df.close.values, df.high.values, df.low.values

    direction = np.abs(close[-1] - close[-n])
    volatility = np.abs(close[-n:] - np.array([0] + close[-n:-1].tolist()))
    er = direction / sum(volatility)

    k = 1 if close[-1] > close[-n] else -1

    # y:close, y_hat:kx+b, y_mean:avg(close)
    y_hat = np.array([(k * direction / n * i + close[-n]) for i, price in enumerate(close[-n:])])
    sse = sum((close[-n:] - y_hat) ** 2)
    sst = sum((close[-n:] - np.mean(close[-n:])) ** 2)
    r2 = 1 - sse / sst

    atr = talib.ATR(high, low, close, timeperiod=20)

    print(symbol, k, er, r2 * direction, atr[-1])
