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
trend long: ma100 > ma200, sar > close, macd 金叉做多
trend short: ma100 < ma200, sar < close, macd 死叉做空

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
            period_signal=10
        )
        self.macd_diff = macd.macd
        self.macd_dea = macd.signal
 
        self.ma1 = bt.talib.SMA(self.close, timeperiod=120, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=200, subplot=False)

        self.sar = ParabolicSAR(self.datas[0], step=0.02, maxaf=0.2, period=4)
        # self.sar = bt.talib.SAR(self.high, self.low)
        
        atr_period = 15
        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=atr_period)

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.last_price = 0
        self.max_cash = 0
        self.atr_rate_low = 1.5
        self.atr_rate_high = 3

        self.stop_loss = 0.01
        self.stop_win = 0.02
 
    def next(self):

        self.log(f"price {self.close[0]} macd_diff {self.macd_diff[0]} macd_dea {self.macd_dea[0]} sar {self.sar[0]} ma1 {self.ma1[0]}")
        
        if self.order: # 检查是否有指令等待执行,
            return

        self.last_price = self.position.price # 持仓价
        
        self.buySig = False
        self.shortSig = False
        self.buyStopLoss = False
        self.shortStopLoss = False
        self.buyTakeProfit = False
        self.shortTakeProfit = False

        # 在 Backtrader 中，[0] 表示当前时刻的值, [-1] 为前一个时刻的值
        # 做多信号计算
        if (self.ma1[0] > self.ma2[0] and
            ((self.macd_diff[0] > self.macd_dea[0] and self.sar[0] > 0 and self.sar[-1] < 0) or
            (self.sar[0] > 0 and self.macd_diff[-1] < self.macd_dea[-1] and self.macd_diff[0] > self.macd_dea[0]))
        ):
            self.buySig = True
            
        # 做空
        if (self.ma1[0] < self.ma2[0] and
            ((self.macd_diff[0] < self.macd_dea[0] and self.sar[0] < 0 and self.sar[-1] > 0) or
            (self.sar[0] < 0 and self.macd_diff[-1] > self.macd_dea[-1] and self.macd_diff[0] < self.macd_dea[0]))
        ):
            self.shortSig = True

        # 多单卖出信号计算
        if self.position.size > 0:
            if (self.close[0] - self.last_price) < -self.atr_rate_low*self.ATR[0]:
                self.buyStopLoss = True
            if (self.close[0] - self.last_price) > self.atr_rate_high*self.ATR[0]:
                self.buyTakeProfit = True
            # if (self.close[0]/self.last_price) < (1-self.stop_loss):
            #     self.buyStopLoss = True
            # if self.close[0]/self.last_price > (1+self.stop_win): # 止盈
            #     self.buyTakeProfit = True
        if self.position.size < 0:
            if (self.close[0] - self.last_price) > self.atr_rate_low*self.ATR[0]:
                self.shortStopLoss = True
            if  (self.last_price - self.close[0]) > self.atr_rate_high*self.ATR[0]:
                self.shortTakeProfit = True
            # if self.close[0]/self.last_price > (1+self.stop_loss):
            #     self.shortStopLoss = True
            # if (self.close[0]/self.last_price) < (1-self.stop_win):
            #     self.shortTakeProfit = True

        # 执行开平仓，如果当前持有多单
        if self.position.size > 0:
            if self.buyStopLoss or self.shortSig:
                self.log("多单止损")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            elif self.buyTakeProfit:
                self.log("多单止盈")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            else:
                pass

        # 如果当前持有空单
        elif self.position.size < 0: 
            if self.shortStopLoss or self.buySig:
                self.log("空单止损")
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            elif self.shortTakeProfit:
                self.log("空单止盈")
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            else:
                pass

        # 如果没有持仓，等待入场时机
        else:
            # Check if we are in the market
            size = 1
            if self.buySig:
                self.log(f'BUY CREATE: {self.close[0]}, Pos size {self.position.size}')
                self.buySig = True
                self.order = self.buy(self.datas[0],size=size)

            if self.shortSig:
                self.log(f'SELL CREATE: {self.close[0]}, Pos size {self.position.size}')
                self.shortSig = True
                self.order = self.sell(self.datas[0],size=size)