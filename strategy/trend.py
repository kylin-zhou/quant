# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt  # 引入backtrader框架

import os, sys

import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime
 
from ._base import BaseStrategyClass

""" 趋势策略
1. 以长期均线作为趋势滤波，只在上升趋势做多，下降趋势做空
2. 以支撑位、阻力位突破交易
3. ATR动态止损、信号止损
"""
 
class TrendStrategyClass(BaseStrategyClass):
    '''
    '''
 
    def __init__(self):
        # 指标必须要定义在策略类中的初始化函数中
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        self.ma1 = bt.talib.SMA(self.close, timeperiod=20, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=50, subplot=False)
        self.ma3 = bt.talib.SMA(self.close, timeperiod=200, subplot=False)

        atr_period = 20
        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=atr_period)
        
        period = 50
        self.min = bt.talib.MIN(self.close, timeperiod=period)
        self.max = bt.talib.MAX(self.close, timeperiod=period)
 
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.last_price = 0
        self.max_cash = 0
        self.atr_rate_low = 1
        self.atr_rate_high = 2
 
    def next(self):
 
        if self.order: # 检查是否有指令等待执行,
            return
 
        # self.log("last_price {} close {} atr {}"
        #     .format(self.last_price, self.close[0], self.ATR[0])
        # )

        self.last_price = self.position.price
        
        self.buySig = False
        self.shortSig = False
        self.buyStopLoss = False
        self.shortStopLoss = False

        # 计算信号
        # 做多
        if (
            self.close>self.ma2 > self.ma3 and self.ma2 > self.ma2[-1] and
            self.close[0] > self.max[-1]
        ):
            self.buySig = True
        # 做空
        if (
            self.close<self.ma2 < self.ma3 and self.ma2 < self.ma2[-1] and
            self.close[0] < self.min[-1]
        ):
            self.shortSig = True

        # 多单止损
        if (self.close[0] - self.last_price) < -self.atr_rate_low*self.ATR[0]:
            self.buyStopLoss = True
        if (self.close[0] - self.close[-1]) < -self.atr_rate_high*self.ATR[0]:
            self.buyStopLoss = True
        if self.close < self.ma2:
            self.buyStopLoss = True
        if self.close < self.min[-1]:
            self.buyStopLoss = True
        if self.shortSig:
            self.buyStopLoss = True

        # 空单止损
        if (self.close[0] - self.last_price) > self.atr_rate_low*self.ATR[0]:
            self.shortStopLoss = True
        if (self.close[0] - self.close[-1]) > self.atr_rate_high*self.ATR[0]:
            self.shortStopLoss = True
        if self.close > self.ma2:
            self.shortStopLoss = True
        if self.close > self.max[-1]:
            self.shortStopLoss = True
        if self.buySig:
            self.shortStopLoss = True
        

        # 如果当前持有多单
        if self.position.size > 0:
            if self.buyStopLoss:
                self.log("多单止损")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()

        # 如果当前持有空单
        elif self.position.size < 0: 
            if self.shortStopLoss:
                self.log("空单止损")
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()

        # 如果没有持仓，等待入场时机
        else:
            # Check if we are in the market
            # self.data.close是表示收盘价

            size = 1
            if self.buySig:
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('BUY Price,{}'.format(self.position.price))
                self.log("做多")
                self.buySig = True
                self.order = self.buy(self.datas[0],size=size)

            if self.shortSig:
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('Pos size %s' % self.position.size)
                self.log("做空")
                self.shortSig = True
                self.order = self.sell(self.datas[0],size=size)

