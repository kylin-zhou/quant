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
trend long: ma1 > ma2, sar > close, macd 金叉做多
trend short: ma1 < ma2, sar < close, macd 死叉做空

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
            period_me2=40,
            period_signal=10
        )
        self.macd_diff = macd.macd
        self.macd_dea = macd.signal
 
        self.ma1 = bt.talib.SMA(self.close, timeperiod=50, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=120, subplot=False)

        self.sar = ParabolicSAR(self.datas[0], step=0.02, maxaf=0.2, period=4)
        # self.sar = bt.talib.SAR(self.high, self.low)
        
        atr_period = 15
        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=atr_period)

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.entry_price = 0 # 买入价格
        self.max_cash = 0

        self.highest_price = 0  # 多单建仓后的最高价
        self.lowest_price = float('inf')  # 空单建仓后的最低价

        self.buyStopLoss = False
        self.shortStopLoss = False
        self.buyTakeProfit = False
        self.shortTakeProfit = False

        self.atr_rate_low = 1.5
        self.atr_rate_high = 3

        self.stop_loss = 0.01
        self.stop_win = 0.02
 
    def next(self):

        self.log(f"price {self.close[0]} macd_diff {self.macd_diff[0]} macd_dea {self.macd_dea[0]} sar {self.sar[0]} ma1 {self.ma1[0]}")
        
        if self.order: # 检查是否有指令等待执行,
            return

        # 在 Backtrader 中，[0] 表示当前时刻的值, [-1] 为前一个时刻的值
        # 做多信号计算
        if self.position.size == 0:
            size = 1
            if (self.ma1[0] > self.ma2[0] and
                ((self.macd_diff[0] > self.macd_dea[0] and self.macd_diff[-5] < self.macd_dea[-5] and self.sar[0] > 0 and self.sar[-1] < 0) or
                (self.sar[0] > 0 and self.sar[-5] < 0 and self.macd_diff[-1] < self.macd_dea[-1] and self.macd_diff[0] > self.macd_dea[0]))
            ):
                self.log(f'BUY CREATE: {self.close[0]}, Pos size {self.position.size}')
                self.order = self.buy(self.datas[0],size=size)
                self.entry_price = self.position.price
                self.highest_price = self.close[0]

            # 做空
            elif (self.ma1[0] < self.ma2[0] and
                ((self.macd_diff[0] < self.macd_dea[0] and self.macd_diff[-5] > self.macd_dea[-5] and self.sar[0] < 0 and self.sar[-1] > 0) or
                (self.sar[0] < 0 and self.sar[-5] > 0 and self.macd_diff[-1] > self.macd_dea[-1] and self.macd_diff[0] < self.macd_dea[0]))
            ):
                self.log(f'SELL CREATE: {self.close[0]}, Pos size {self.position.size}')
                self.order = self.sell(self.datas[0],size=size)
                self.entry_price = self.position.price
                self.lowest_price = self.close[0]

        # 执行开平仓，如果当前持有多单
        elif self.position.size > 0:
            # self.highest_price = max(self.highest_price, self.close[0])
            # if (self.highest_price - self.close[0]) / self.highest_price >= 0.02:
            #     self.order = self.sell(size=abs(self.position.size))

            self.stop_loss_profit_atr_long()
            if self.buyStopLoss or self.buyTakeProfit:
                self.order = self.sell(size=abs(self.position.size))
                self.max_cash = self.broker.getvalue()

        elif self.position.size < 0:
            # self.lowest_price = min(self.lowest_price, self.close[0])
            # if (self.close[0] - self.lowest_price) / self.lowest_price >= 0.02:
            #     self.order = self.buy(size=abs(self.position.size))
            self.stop_loss_profit_atr_short()
            if self.shortStopLoss or self.shortTakeProfit:
                self.order = self.buy(size=abs(self.position.size))
                self.max_cash = self.broker.getvalue()

    def stop_loss_profit_atr_long(self):
        if (self.entry_price - self.close[0] ) >self.atr_rate_low*self.ATR[0]:
            self.buyStopLoss = True
            self.log("多单止损")
        elif (self.close[0] - self.entry_price) > self.atr_rate_high*self.ATR[0]:
            self.buyTakeProfit = True
            self.log("多单止盈")

    def stop_loss_profit_atr_short(self):
        if (self.close[0] - self.entry_price) > self.atr_rate_low*self.ATR[0]:
            self.shortStopLoss = True
            self.log("空单止损")
        elif (self.entry_price - self.close[0]) > self.atr_rate_high*self.ATR[0]:
            self.shortTakeProfit = True
            self.log("空单止盈")

    def stop_loss_profit_fixed_long(self):
        if (self.close[0]/self.last_price) < (1-self.stop_loss):
            self.buyStopLoss = True
            self.log("多单止损")
        if self.close[0]/self.last_price > (1+self.stop_win): # 止盈
            self.buyTakeProfit = True
            self.log("多单止盈")

    def stop_loss_profit_fixed_short(self):
        if (self.close[0]/self.last_price) > (1+self.stop_loss):
            self.shortStopLoss = True
            self.log("空单止损")
        if self.close[0]/self.last_price < (1-self.stop_win): # 止损
            self.shortTakeProfit = True
            self.log("空单止盈")

    def stop_loss_profit_move_long(self):
        pass