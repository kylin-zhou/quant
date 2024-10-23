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
 
""" 均线策略
"""
 
class MAStrategyClass(BaseStrategyClass):
 
    def __init__(self):
        # sma源码位于indicators\macd.py
        # 指标必须要定义在策略类中的初始化函数中
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=20)

        self.ma1 = bt.talib.SMA(self.close, timeperiod=10, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=20, subplot=False)
        self.ma3 = bt.talib.SMA(self.close, timeperiod=100, subplot=False)
        self.ma4 = bt.talib.SMA(self.close, timeperiod=200, subplot=False)
        self.crossover = bt.indicators.CrossOver(self.ma1, self.ma2)

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.initial_cash = self.broker.get_cash()
        self.max_cash = 0
        self.hard_loss = 0.2
        self.atr_rate_low = 1.5
        self.atr_rate_high = 3
        self.std_rate = 3
 
    def next(self):
        size = 1
 
        if self.order: # 检查是否有指令等待执行,
            return
 
        self.max_cash = max(self.broker.getvalue(), self.max_cash)
        # current_cash = self.broker.getvalue()
        # if current_cash/self.initial_cash < 0.8:
        #     return
        
        self.buySig = False
        self.shortSig = False
        self.long_stop_loss = False
        self.short_stop_loss = False

        # 如果没有持仓，等待入场时机
        if self.position.size == 0:
            # Simply log the closing price of the series from the reference
            # Check if we are in the market
            # self.data.close是表示收盘价

            if (
                self.ma1 > self.ma2 and self.ma1[-1] < self.ma2[-1] and self.ma3 > self.ma4
            ):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('BUY Price,{}'.format(self.position.price))
                self.log("做多")
                self.buySig = True
                self.order = self.buy(self.datas[0],size=size)
                self.last_price = self.position.price

            elif (
                self.ma1 < self.ma2 and self.ma1[-1] > self.ma2[-1] and self.ma3 < self.ma4
            ):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('Pos size %s' % self.position.size)
                self.log("做空")
                self.shortSig = True
                self.order = self.sell(self.datas[0],size=size)
                self.last_price = self.position.price

        elif self.position.size > 0:
            # 多单止损
            if (self.last_price - self.close[0]) > self.atr_rate_low*self.ATR[0]:
                self.long_stop_loss = True
            if (self.close[0] - self.last_price) > self.atr_rate_high*self.ATR[0]:
                self.long_stop_loss = True

            if self.long_stop_loss:
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()

        elif self.position.size < 0:
            # 空单止损：当价格上涨至ATR时止损平仓
            if (self.close[0] - self.last_price) > self.atr_rate_low*self.ATR[0]:
                self.short_stop_loss = True
            if (self.last_price - self.close[0]) > self.atr_rate_high*self.ATR[0]:
                self.short_stop_loss = True

            # 空单止损：当价格上涨至ATR时止损平仓
            if self.short_stop_loss:
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
