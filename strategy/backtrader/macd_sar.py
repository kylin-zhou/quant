# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt  # 引入backtrader框架

import os, sys

import akshare as ak
import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime
 
from .base import BaseStrategyClass
from .indicator import ParabolicSAR
 
""" ma+MACD+sar
trend long: ma50 > ma120, sar > close, macd 金叉做多
trend short: ma50 < ma120, sar < close, macd 死叉做空

止损：ATR止损
"""
 
class MACDSARStrategyClass(BaseStrategyClass):
    '''#平滑异同移动平均线MACD
    '''

    def __init__(self):
        # sma源码位于indicators\macd.py
        # 指标必须要定义在策略类中的初始化函数中
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        macd = bt.ind.MACD(self.close,
            period_me1=20,
            period_me2=50,
            period_signal=10,
            subplot=False
        )
        self.macd_diff = macd.macd
        self.macd_dea = macd.signal
 
        self.ma10 = bt.talib.SMA(self.close, timeperiod=10, subplot=False)
        self.ma20 = bt.talib.SMA(self.close, timeperiod=20, subplot=False)
        self.ma50 = bt.talib.SMA(self.close, timeperiod=50, subplot=False)
        self.ma250 = bt.talib.SMA(self.close, timeperiod=250, subplot=False)
        self.ma150 = bt.talib.SMA(self.close, timeperiod=150, subplot=False)

        self.sar = ParabolicSAR(self.datas[0], step=0.02, maxaf=0.2, period=4)
        # self.sar = bt.talib.SAR(self.high, self.low)
        
        atr_period = 14
        self.atr = bt.talib.ATR(self.high, self.low, self.close, timeperiod=atr_period)

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.entry_price = 0 # 买入价格
        self.stop_loss_price = 0
        self.max_cash = 0

        self.highest_price = 0  # 多单建仓后的最高价
        self.lowest_price = float('inf')  # 空单建仓后的最低价

        self.buyStopLoss = False
        self.shortStopLoss = False
        self.buyTakeProfit = False
        self.shortTakeProfit = False

        self.atr_rate_low = 2
        self.atr_rate_high = 3
        self.atr_gap = self.atr_rate_high - self.atr_rate_low
 
    def next(self):

        self.log(f"price {self.close[0]} macd_diff {self.macd_diff[0]} macd_dea {self.macd_dea[0]} sar {self.sar[0]}")
        self.log(f"entry price {self.entry_price},  stoploss {self.stop_loss_price}")
        
        if self.order: # 检查是否有指令等待执行,
            return

        # 在 Backtrader 中，[0] 表示当前时刻的值, [-1] 为前一个时刻的值
        bias = (self.close[0] - self.ma50[0]) / self.ma50[0] * 100
        buy_signal, sell_signal = False, False
        
        # 做多信号计算
        if (
            # self.ma20[0] > self.ma50[0] > self.ma150[0]
            self.ma50[0] > self.ma150[0] > self.ma250[0]
            and self.ma150[0] > self.ma150[-1]
            and bias < 1
            and ((self.macd_diff[0] > self.macd_dea[0] and self.macd_diff[-6] < self.macd_dea[-6] and self.sar[0] > 0 and self.sar[-1] < 0) or
            (self.sar[0] > 0 and self.sar[-6] < 0 and self.macd_diff[-1] < self.macd_dea[-1] and self.macd_diff[0] > self.macd_dea[0]))
            ):
            buy_signal = True
        
        # 做空信号计算
        if (
            # self.ma20[0] < self.ma50[0] < self.ma150[0]
            self.ma50[0] < self.ma150[0] < self.ma250[0]
            and self.ma150[0] < self.ma150[-1]
            and bias > -1
            and ((self.macd_diff[0] < self.macd_dea[0] and self.macd_diff[-6] > self.macd_dea[-6] and self.sar[0] < 0 and self.sar[-1] > 0) or
            (self.sar[0] < 0 and self.sar[-6] > 0 and self.macd_diff[-1] > self.macd_dea[-1] and self.macd_diff[0] < self.macd_dea[0]))
            ):
            sell_signal = True

        if self.ma20[0] > self.ma50[0] > self.ma150[0] and self.ma50[0] > self.ma50[-1]:
            sell_signal = False
        if self.ma20[0] < self.ma50[0] < self.ma150[0] and self.ma50[0] < self.ma50[-1]:
            buy_signal = False
        
        if self.position.size == 0:
            size = 1
            if buy_signal:
                self.log(f'BUY CREATE: {self.close[0]}, Pos size {self.position.size}')
                self.order = self.buy(self.datas[0],size=size)
                self.stop_loss_price = self.close[0] - self.atr[0] * self.atr_rate_low

            # 做空
            elif sell_signal:
                self.log(f'SELL CREATE: {self.close[0]}, Pos size {self.position.size}')
                self.order = self.sell(self.datas[0],size=size)
                self.stop_loss_price = self.close[0] + self.atr[0] * self.atr_rate_low

        # 执行开平仓，如果当前持有多单
        elif self.position.size > 0:
            if self.close[0] > self.position.price + self.atr_gap*self.atr[0]:
                self.highest_price = max(self.highest_price, self.close[0])
                self.stop_loss_price = self.highest_price - self.atr[0] * self.atr_rate_high

            if self.close[0] < self.stop_loss_price:
                self.order = self.sell(size=abs(self.position.size))
                self.max_cash = self.broker.getvalue()

                self.highest_price, self.lowest_price = 0, float('inf')

        elif self.position.size < 0:
            if self.close[0] < self.position.price - self.atr_gap*self.atr[0]:
                self.lowest_price = min(self.lowest_price, self.close[0])
                self.stop_loss_price = self.lowest_price + self.atr[0] * self.atr_rate_high

            if self.close[0] > self.stop_loss_price:
                self.order = self.buy(size=abs(self.position.size))
                self.max_cash = self.broker.getvalue()

                self.highest_price, self.lowest_price = 0, float('inf')
