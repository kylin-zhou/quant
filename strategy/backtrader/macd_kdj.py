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
 
""" ma+MACD+kdj
close > ma100, macd 金叉做多
macd > 0, kdj 金叉做多

close < ma100, macd 死叉做空
macd < 0, kdj 死叉做空

止损：ATR止损、macd叉止损、多空止损
"""
 
class MACDKDJStrategyClass(BaseStrategyClass):
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
        self.diff = macd.macd
        self.dea = macd.signal
 
        self.ma0 = bt.talib.SMA(self.close, timeperiod=50, subplot=False)
        self.ma1 = bt.talib.SMA(self.close, timeperiod=120, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=240, subplot=False)

        self.rsi1 = bt.talib.RSI(self.close, timeperiod=6)
        self.rsi2 = bt.talib.RSI(self.close, timeperiod=24)
        
        atr_period = 20
        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=atr_period)

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.last_price = 0
        self.max_cash = 0
        self.atr_rate_low = 2
        self.atr_rate_high = 4

        self.stop_loss = 0.01
        self.stop_win = 0.02
 
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
        # if abs(self.ma1 -self.ma1[-50]) > 0.5*self.ATR[0]:
        # macd金叉做多
        if (self.diff[-1]<self.dea[-1] and self.diff>self.dea
            and self.ma1 > self.ma2
        ):
            self.buySig = True
        if (self.diff>self.dea
            and self.rsi1[-1]<self.rsi2[-1] and self.rsi1>self.rsi2
            and self.close > self.ma0
        ):
            self.buySig = True
        # macd死叉做空
        if (self.diff[-1]>self.dea[-1] and self.diff<self.dea
            and self.ma1 < self.ma2
        ):
            self.shortSig = True
        if (self.diff<self.dea
            and self.rsi1[-1]>self.rsi2[-1] and self.rsi1<self.rsi2
            and self.close < self.ma0
        ):
            self.shortSig = True

        if self.last_price != 0:
            # 多单止损
            # if (self.close[0] - self.last_price) < -self.atr_rate_low*self.ATR[0]:
            #     self.buyStopLoss = True
            # if (self.close[0] - self.last_price) > self.atr_rate_high*self.ATR[0]:
            #     self.buyStopLoss = True
            # if (self.diff[-1]>self.dea[-1] and self.diff<self.dea):
            #     self.buyStopLoss = True
            # if self.shortSig:
            #     self.buyStopLoss = True
            # if (self.diff[-1]>self.dea[-1] and self.diff<self.dea):
            #     self.buyStopLoss = True
            if (self.close[0]/self.last_price) < (1-self.stop_loss):
                self.buyStopLoss = True
            if self.close[0]/self.last_price > (1+self.stop_win): # 止盈
                self.buyStopLoss = True


            # 空单止损
            # if (self.close[0] - self.last_price) > self.atr_rate_low*self.ATR[0]:
            #     self.shortStopLoss = True
            # if (self.close[0] - self.last_price) < -self.atr_rate_high*self.ATR[0]:
            #     self.shortStopLoss = True
            # if (self.diff[-1]<self.dea[-1] and self.diff>self.dea):
            #     self.shortStopLoss = True
            # if self.buySig:
            #     self.shortStopLoss = True
            # if (self.diff[-1]<self.dea[-1] and self.diff>self.dea):
            #     self.shortStopLoss = True
            if (self.close[0]/self.last_price) < (1-self.stop_win): # 止盈, 价格变化比
                self.shortStopLoss = True
            if self.close[0]/self.last_price > (1+self.stop_loss):
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

