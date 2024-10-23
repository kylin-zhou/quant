from backtesting import Backtest
from backtesting.lib import crossover
import pandas as pd

from strategy.backtesting import DualMovingAverageStrategy

df = pd.read_csv("D:/trading/quant/data/futures/RM/RM2301.csv")
df.columns = [
        'date',
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'Hold'
    ]

# 创建回测对象
bt = Backtest(df, DualMovingAverageStrategy, cash=10_000, commission=.002)
stats = bt.run()
print(stats)

bt.plot(filename="DualMovingAverageStrategy.html")