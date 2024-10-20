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
 
""" 多均线策略
"""
 
class MAStrategyClass(BaseStrategyClass):
    '''
    '''
 
    def __init__(self):
        # sma源码位于indicators\macd.py
        # 指标必须要定义在策略类中的初始化函数中
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=20)

        self.ma1 = bt.talib.SMA(self.close, timeperiod=5, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=20, subplot=False)
        self.ma3 = bt.talib.SMA(self.close, timeperiod=50, subplot=False)
        self.ma4 = bt.talib.SMA(self.close, timeperiod=200, subplot=False)

        period = 20
        self.ma2_std = bt.talib.STDDEV(self.ma2, timeperiod=period, nbdev=1.0)
        self.ma2_min = bt.talib.MIN(self.ma2, timeperiod=period)
        self.ma2_max = bt.talib.MAX(self.ma2, timeperiod=period)
        self.adx = bt.talib.ADX(self.high, self.low, self.close, timeperiod=10)
 
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.max_cash = 0
        self.hard_loss = 0.2
        self.atr_rate_low = 2
        self.atr_rate_high = 3
        self.std_rate = 3
 
    def next(self):
 
        if self.order: # 检查是否有指令等待执行,
            return
 
        self.last_price = self.position.price
        self.max_cash = max(self.broker.getvalue(), self.max_cash)
        
        self.buySig = False
        self.shortSig = False
        self.long_stop_loss = False
        self.short_stop_loss = False

        # get sig
        # 多单止损
        if (self.close[0] - self.last_price) < -self.atr_rate_low*self.ATR[0]:
            self.long_stop_loss = True
        if (self.close[0] - self.close[-1]) < -self.atr_rate_high*self.ATR[0]:
            self.long_stop_loss = True
        if self.close<self.ma3:
            self.long_stop_loss = True

        # 空单止损：当价格上涨至ATR时止损平仓
        if (self.close[0] - self.last_price) > self.atr_rate_low*self.ATR[0]:
            self.short_stop_loss = True
        if (self.close[0] - self.close[-1]) > self.atr_rate_high*self.ATR[0]:
            self.short_stop_loss = True
        if self.close>self.ma3:
            self.short_stop_loss = True


        # 如果当前持有多单
        if self.position.size > 0:
            # 多单止损
            if self.long_stop_loss or self.shortSig:
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
                
        # 如果当前持有空单
        elif self.position.size < 0:
            # 空单止损：当价格上涨至ATR时止损平仓
            if self.short_stop_loss or self.buySig:
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()

        # 如果没有持仓，等待入场时机
        else:
            # Simply log the closing price of the series from the reference
            # Check if we are in the market
            # self.data.close是表示收盘价

            size = 1

            if (
                (self.ma1>self.ma2)
                # and (self.ma2-self.ma2_min)>self.std_rate*self.ma2_std
                # and self.ma1 > self.ma1[-1]
                # and self.ma2 > self.ma2[-1]
                and self.close > self.ma4 > self.ma4[-1]
                and self.adx>25
            ):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('BUY Price,{}'.format(self.position.price))
                self.log("做多")
                self.buySig = True
                self.order = self.buy(self.datas[0],size=size)

            if (
                (self.ma1<self.ma2)
                # and (self.ma2_max-self.ma2)>self.std_rate*self.ma2_std
                # and self.ma1 < self.ma1[-1]
                # and self.ma2 < self.ma2[-1]
                # and self.ma3 < self.ma3[-1]
                and self.close < self.ma4 < self.ma4[-1] 
                and self.adx>25
            ):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('Pos size %s' % self.position.size)
                self.log("做空")
                self.shortSig = True
                self.order = self.sell(self.datas[0],size=size)


class ETFMAStrategyClass(BaseStrategyClass):
    '''
    '''
 
    def __init__(self):
        # sma源码位于indicators\macd.py
        # 指标必须要定义在策略类中的初始化函数中
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=8)

        self.ma1 = bt.talib.SMA(self.close, timeperiod=10, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=20, subplot=False)
        self.ma3 = bt.talib.SMA(self.close, timeperiod=30, subplot=False)
        self.ma4 = bt.talib.SMA(self.close, timeperiod=40, subplot=False)
        period = 20
        self.ma2_std = bt.talib.STDDEV(self.ma3, timeperiod=period, nbdev=1.0)
        self.ma2_min = bt.talib.MIN(self.ma3, timeperiod=period)
        self.ma2_max = bt.talib.MAX(self.ma3, timeperiod=period)
        self.adx = bt.talib.ADX(self.high, self.low, self.close, timeperiod=10)
 
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.max_cash = 0
        self.hard_loss = 0.1
        self.atr_rate_low = 1
        self.atr_rate_high = 1
 
    def next(self):
 
        if self.order: # 检查是否有指令等待执行,
            return
 
        self.last_price = self.position.price
        self.max_cash = max(self.broker.getvalue(), self.max_cash)
        
        self.buySig = False
        self.shortSig = False

        # 如果当前持有多单
        if self.position.size > 0 :
            self.log("last_price {} close {} max_cash {} atr {} cash {}"
                .format(self.last_price, self.close[0], self.max_cash, self.ATR[0], self.broker.getvalue())
            )
            # self.order = self.sell(size=abs(self.position.size))
            # 多单止损
            if (self.close[0] - self.last_price) < -self.atr_rate_low*self.ATR[0]:
                self.log("多单止损")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            # 多单止盈
            elif 1-self.broker.getvalue()/self.max_cash > self.hard_loss:
                self.log("多单止盈")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            elif (self.close[0] - self.close[-1]) < -self.atr_rate_high*self.ATR[0]:
                self.log("多单止盈")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            elif self.shortSig:
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
                

        # 如果没有持仓，等待入场时机
        else:
            # Simply log the closing price of the series from the reference
            self.log('Close, %.2f' % self.close[0])
            # Check if we are in the market
    
            # self.data.close是表示收盘价
            # 收盘价大于histo，做多

            size = 10

            if ((self.ma1>self.ma2>self.ma3) and (self.ma3-self.ma2_min)>3*self.ma2_std
                and self.ma4 >= self.ma4[-1] and self.adx>=25):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('BUY Price,{}'.format(self.position.price))
                self.log("做多")
                self.buySig = True
                self.order = self.buy(self.datas[0],size=size)
