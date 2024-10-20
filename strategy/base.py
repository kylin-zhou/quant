# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt  # 引入backtrader框架

import akshare as ak
import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime
from loguru import logger
 
import os, sys
 
"""base
"""
 
class BaseStrategyClass(bt.Strategy):
    '''策略基类
    '''
    def __init__(self):
        # 指标必须要定义在策略类中的初始化函数中, 初始化策略中使用的各项指标、变量
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.max_cash = 0
        self.hard_loss = 0.02
        self.atr_rate = 1
 
    def log(self, txt):
        ''' Logging function for this strategy'''
        dt = self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()}, {txt}')
 
    def notify_cashvalue(self, cash, value):
        pass
        # self.log('Cash %s Value %s' % (cash, value))
 
    def notify_order(self, order):
        print(type(order), 'Is Buy ', order.isbuy())
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
 
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
 
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
 
            self.bar_executed = len(self)
 
 
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
 
        self.order = None
 
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
 
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
 
    def next(self):
        # 策略的主逻辑，在每个新的数据点（例如每个新的价格条）到达时被调用
        pass