import sys
from sys import float_info as sflt
import numpy as np
import pandas as pd


def zero(x):
    """If the value is close to zero, then return zero. Otherwise return itself."""
    return 0 if abs(x) < sflt.epsilon else x
    
def calculate_sar(high, low, N=4, step=2, mvalue=20):
    step1 = step/100
    mvalue1 = mvalue/100
    ep, af, sar = [0]*len(high), [0]*len(high), [0]*len(high)
    low_pre = float('inf') #上一个周期最小值
    high_pre = float('-inf') #上一个周期最大值
    
    def _falling(high, low, drift:int=1):
        """Returns the last -DM value"""
        # Not to be confused with ta.falling()
        up = high - high.shift(drift)
        dn = low.shift(drift) - low
        _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
        return _dmn > 0

    # Falling if the first NaN -DM is positive
    trend = _falling(high.iloc[:2], low.iloc[:2])

    for i in range(N,len(high)):
        #先确定sr0
        if trend and i == N :#涨势 
            ep[i] = min(low[0:N-1])
            sar[i] = ep[i]
            continue
        elif trend == False and i == N:#跌势 
            ep[i] = max(high[0:N-1])
            sar[i] = ep[i]*-1
            continue
              
        af[i] = af[i-1]+step1
        if  af[i] > mvalue1:
             af[i] = mvalue1
        if trend:    
            sar[i] = abs(sar[i-1]) + af[i]*(high[i-1] - abs(sar[i-1]))
            ep[i] = max(ep[i-1], high[i])
            high_pre = max(high_pre, high[i])

            if low[i] < abs(sar[i]): # 趋势反转
               trend = not trend
               af[i] = 0
               low_pre = low[i]
               sar[i] = max(high[0:i]) if low_pre == 0 else high_pre*-1

        else :
            sar[i] = -1 * (abs(sar[i-1]) + af[i]*(low[i-1] - abs(sar[i-1])))
            ep[i] = min(ep[i-1],low[i])
            low_pre = min(low_pre,low[i])
            
            if high[i] > abs(sar[i]): # 趋势反转
               trend = not trend
               high_pre = high[i] 
               af[i] = 0
               sar[i] = min(low[0:i]) if high_pre == 0 else low_pre
 
    return sar

# 计算 EMA
def calculate_ema(prices, period):
    return pd.Series(prices).ewm(span=period, adjust=False).mean()

# 计算 MACD, Signal Line, and Histogram
def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """
    Returns the MACD Line, Signal Line, and Histogram
    """
    # 计算快线和慢线的EMA
    short_ema = calculate_ema(prices, short_period)  # 快线EMA
    long_ema = calculate_ema(prices, long_period)    # 慢线EMA
    
    # MACD Line = 快线EMA - 慢线EMA
    macd_diff = short_ema - long_ema
    
    # Signal Line = MACD Line的9周期EMA
    macd_dea = calculate_ema(macd_diff, signal_period)
    
    # MACD Histogram = MACD Line - Signal Line
    macd_histogram = (macd_diff - macd_dea) * 2
    
    return pd.DataFrame({'macd_diff': macd_diff, 'macd_dea': macd_dea, 'macd_histogram': macd_histogram})


import backtrader as bt

class ParabolicSAR(bt.Indicator):
    lines = ('sar',)
    params = (('step', 0.02), ('maxaf', 0.2), ('period', 4))

    plotinfo = dict(subplot=False)

    def __init__(self):
        self.ep = [0.0]
        self.af = [0.0]
        self.trend = None
        self.high_prev = float('-inf')
        self.low_prev = float('inf')

    def zero(self, x):
        return 0 if abs(x) < 1e-8 else x

    def _falling(self, drift=1):
        if len(self) <= drift:
            return False
        up = self.data.high[0] - self.data.high[-drift]
        dn = self.data.low[-drift] - self.data.low[0]
        _dmn = self.zero(dn if (dn > up and dn > 0) else 0)
        return _dmn > 0

    def prenext(self):
        # 在指标开始计算之前，用最高价填充SAR值
        self.lines.sar[0] = self.data.high[0]

    def next(self):
        if len(self) <= self.p.period:
            if self.trend is None:
                self.trend = self._falling(1)
            if len(self) == self.p.period:
                if self.trend:
                    self.ep.append(min(self.data.low.get(size=self.p.period)))
                    self.lines.sar[0] = self.ep[-1]
                else:
                    self.ep.append(max(self.data.high.get(size=self.p.period)))
                    self.lines.sar[0] = -self.ep[-1]
            else:
                self.lines.sar[0] = self.data.high[0]
            return

        self.af.append(min(self.af[-1] + self.p.step, self.p.maxaf))

        if self.trend:
            sar = abs(self.lines.sar[-1]) + self.af[-1] * (self.data.high[-1] - abs(self.lines.sar[-1]))
            self.ep.append(max(self.ep[-1], self.data.high[0]))
            self.high_prev = max(self.high_prev, self.data.high[0])

            if self.data.low[0] < abs(sar):
                self.trend = not self.trend
                self.af[-1] = 0
                self.low_prev = self.data.low[0]
                sar = max(self.data.high.get(size=len(self))) if self.low_prev == 0 else -self.high_prev
        else:
            sar = -1 * (abs(self.lines.sar[-1]) + self.af[-1] * (self.data.low[-1] - abs(self.lines.sar[-1])))
            self.ep.append(min(self.ep[-1], self.data.low[0]))
            self.low_prev = min(self.low_prev, self.data.low[0])

            if self.data.high[0] > abs(sar):
                self.trend = not self.trend
                self.high_prev = self.data.high[0]
                self.af[-1] = 0
                sar = min(self.data.low.get(size=len(self))) if self.high_prev == 0 else self.low_prev

        self.lines.sar[0] = sar