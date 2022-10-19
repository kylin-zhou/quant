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
 
from ._base import BaseStrategyClass
 
""" AMA均线策略
"""
 
class AMAStrategyClass(BaseStrategyClass):
    '''
    AMA均线策略
    '''
 
    def __init__(self):
        # sma源码位于indicators\macd.py
        # 指标必须要定义在策略类中的初始化函数中
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=8)

        self.ama = bt.talib.KAMA(self.close, timeperiod=10, subplot=False)
        
        period = 20
        self.ama_std = bt.talib.STDDEV(self.ama, timeperiod=period, nbdev=1.0)
        self.ama_min = bt.talib.MIN(self.ama, timeperiod=period)
        self.ama_max = bt.talib.MAX(self.ama, timeperiod=period)
 
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.max_cash = 0
        self.atr_rate_low = 1
        self.atr_rate_high = 1
        self.std_rate = 3
 
    def next(self):
 
        if self.order: # 检查是否有指令等待执行,
            return
 
        self.last_price = self.position.price
        
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
            elif (self.close[0] - self.close[-1]) < -self.atr_rate_high*self.ATR[0]:
                self.log("多单止盈")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            elif self.shortSig:
                self.log("多单止盈")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
        # 如果当前持有空单
        elif self.position.size < 0 :  
            self.log("last_price {} close {} max_cash {} atr {}"
                .format(self.last_price, self.close[0], self.max_cash, self.ATR[0])
            )
            # self.order = self.buy(size=abs(self.position.size))
            # 空单止损：当价格上涨至ATR时止损平仓
            if (self.close[0] - self.last_price) > self.atr_rate_low*self.ATR[0]:
                self.log("空单止损")
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            # 空单止盈：当价格突破20日最高点时止盈平仓
            elif (self.close[0] - self.close[-1]) > self.atr_rate_high*self.ATR[0]:
                self.log("空单止盈")
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                self.max_cash = self.broker.getvalue()
            elif self.buySig:
                self.log("空单止盈")
                self.order = self.buy(size=abs(self.position.size))
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

            if (self.ama-self.ama_min>self.std_rate*self.ama_std):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('BUY Price,{}'.format(self.position.price))
                self.log("做多")
                self.buySig = True
                self.order = self.buy(self.datas[0],size=size)

            # 收盘价小于等于histo，做空
            if (self.ama_max-self.ama>self.std_rate*self.ama_std):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('Pos size %s' % self.position.size)
                self.log("做空")
                self.shortSig = True
                self.order = self.sell(self.datas[0],size=size)


class ETFAMAStrategyClass(BaseStrategyClass):
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

        self.ama = bt.talib.KAMA(self.close, timeperiod=10, subplot=False)
        
        period = 20
        self.ama_std = bt.talib.STDDEV(self.ama, timeperiod=period, nbdev=1.0)
        self.ama_min = bt.talib.MIN(self.ama, timeperiod=period)
        self.ama_max = bt.talib.MAX(self.ama, timeperiod=period)
 
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.max_cash = 0
        self.atr_rate_low = 0.4
        self.atr_rate_high = 1
        self.std_rate = 2
 
    def next(self):
 
        if self.order: # 检查是否有指令等待执行,
            return
 
        self.last_price = self.position.price
        
        self.buySig = False
        self.sellSig = False

        # 如果当前持有多单
        if self.position.size > 0 :
            # self.order = self.sell(size=abs(self.position.size))
            # 多单止损
            if (self.close[0] - self.last_price) < -self.atr_rate_low*self.ATR[0]:
                self.log("多单止损")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.sellSig = True
                self.max_cash = self.broker.getvalue()
            # 多单止盈
            elif (self.close[0] - self.close[-1]) < -self.atr_rate_high*self.ATR[0]:
                self.log("多单止盈")
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
                self.sellSig = True
                self.max_cash = self.broker.getvalue()

        # 如果没有持仓，等待入场时机
        else:
            # Simply log the closing price of the series from the reference
            self.log('Close, %.2f' % self.close[0])
            # Check if we are in the market
    
            # self.data.close是表示收盘价
            # 收盘价大于histo，做多

            size = 1000

            if (self.ama-self.ama_min>self.std_rate*self.ama_std):
                self.log('BUY CREATE,{}'.format(self.close[0]))
                self.log('BUY Price,{}'.format(self.position.price))
                self.log("做多")
                self.buySig = True
                self.order = self.buy(self.datas[0],size=size)


        self.log("last_price {} close {} atr {} buy {} sell {}"
            .format(self.last_price, self.close[0], self.ATR[0],self.buySig, self.sellSig)
        )