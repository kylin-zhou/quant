# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt  # 引入backtrader框架

import akshare as ak
import pandas as pd
 
from datetime import datetime
 
import os, sys
 
""" MACD 经典策略
macd_dif > macd_ema, 金叉买入
macd_dif < macd_emd, 死叉卖出
"""
 
class MACDStrategyClass(bt.Strategy):
    '''#平滑异同移动平均线MACD
        DIF(蓝线): 计算12天平均和26天平均的差，公式：EMA(C,12)-EMA(c,26)
       Signal(DEM或DEA或MACD) (红线): 计算macd9天均值，公式：Signal(DEM或DEA或MACD)：EMA(MACD,9)
        Histogram (柱): 计算macd与signal的差值，公式：Histogram：MACD-Signal
        period_me1=12
        period_me2=26
        period_signal=9
        macd = ema(data, me1_period) - ema(data, me2_period)
        signal = ema(macd, signal_period)
        histo = macd - signal
    '''
 
    def __init__(self):
        # sma源码位于indicators\macd.py
        # 指标必须要定义在策略类中的初始化函数中
        macd = bt.ind.MACD()
        self.macd = macd.macd
        self.signal = macd.signal
        self.histo = bt.ind.MACDHisto()
 
        self.dataclose = self.datas[0].close
 
        self.order = None
        self.buyprice = None
        self.buycomm = None
 
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
 
    def notify_cashvalue(self, cash, value):
        self.log('Cash %s Value %s' % (cash, value))
 
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
 
        if self.order: # 检查是否有指令等待执行,
            return
 
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        # Check if we are in the market
        if not self.getposition(self.datas[0]):
 
            # self.data.close是表示收盘价
            # 收盘价大于histo，买入
            # if self.macd > 0 and self.signal > 0 and self.histo > 0:
            if self.macd > self.signal:
                self.log('BUY CREATE,{}'.format(self.dataclose[0]))
                self.log('BUY Price,{}'.format(self.position.price))
                self.order = self.buy(self.datas[0])
 
        else:
 
            # 收盘价小于等于histo，卖出
            # if self.macd <= 0 or self.signal <= 0 or self.histo <= 0:
            if self.macd < self.signal:
                self.log('BUY CREATE,{}'.format(self.dataclose[0]))
                self.log('Pos size %s' % self.position.size)
                self.order = self.sell(self.datas[0])