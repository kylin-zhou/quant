from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta

from .ta import sma

class DualMovingAverageStrategy(Strategy):

    long_stop_loss = 0.010  # 多头止损比例
    long_take_profit = 0.020  # 多头止盈比例
    short_stop_loss = 0.010  # 空头止损比例
    short_take_profit = 0.020  # 空头止盈比例
    entry_price = 0
    position_size = 0

    def init(self):
        # 计算快慢均线
        self.ma1 = self.I(sma, self.data.Close, 10)
        self.ma2 = self.I(sma, self.data.Close, 20)
        self.ma3 = self.I(sma, self.data.Close, 50)
        self.ma4 = self.I(sma, self.data.Close, 100)

    def next(self):

        if self.position_size == 0:  # 如果没有持仓
            # 如果快线上穿慢线,买入
            if self.ma2 > self.ma3:
                if crossover(self.ma1, self.ma2):
                    self.entry_price = self.data.Close[-1]
                    self.buy()
                    self.position_size += 1
                    print(self.position_size, self.entry_price, "buy")

            # if self.ma3 < self.ma4:
            #     if crossover(self.ma2, self.ma1):
            #         self.entry_price = self.data.Close[-1]
            #         self.sell()
        else:
            if self.position_size > 0:
                if (self.data.Close[-1] <= self.entry_price * (1 - self.long_stop_loss) or 
                    self.data.Close[-1] >= self.entry_price * (1 + self.long_take_profit)):
                    self.position.close()
                    self.entry_price, self.position_size = 0, 0

            elif self.position.is_short:
                if (self.data.Close[-1] >= self.entry_price * (1 + self.short_stop_loss) or 
                    self.data.Close[-1] <= self.entry_price * (1 - self.short_take_profit)):
                    self.position.close()
                    self.entry_price = 0