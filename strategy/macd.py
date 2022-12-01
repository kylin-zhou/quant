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
 
""" MACD 经典策略
macd_dif > macd_ema, 金叉买入, 做多
macd_dif < macd_emd, 死叉卖出, 做空
TR止损
"""
 
class MACDStrategyClass(BaseStrategyClass):
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
        self.close = self.datas[0].close
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        macd = bt.ind.MACD(self.close,
            period_me1=20,
            period_me2=40,
            period_signal=14
        )
        self.histo = bt.ind.MACDHisto()
        # macd, macdsignal, self.histo = bt.talib.MACD(
        #     self.close, fastperiod=12, slowperiod=26, signalperiod=9
        # )

        atr_period = 20
        self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=atr_period)
 
        self.ma1 = bt.talib.SMA(self.close, timeperiod=50, subplot=False)
        self.ma2 = bt.talib.SMA(self.close, timeperiod=100, subplot=False)

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.last_price = 0
        self.max_cash = 0
        self.atr_rate_low = 2
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
            self.histo>0 and self.ma1 > self.ma2 and
            (self.ma1 - self.ma1[-1])>0 and (self.ma2 - self.ma2[-1])>0
        ):
            self.buySig = True
        # 做空
        if (
            self.histo<0 and self.ma1 < self.ma2 and
            (self.ma1 - self.ma1[-1])<0 and (self.ma2 - self.ma2[-1])<0
        ):
            self.shortSig = True

        # 多单止损
        if (self.close[0] - self.last_price) < -self.atr_rate_low*self.ATR[0]:
            self.buyStopLoss = True
        if (self.close[0] - self.close[-1]) < -self.atr_rate_high*self.ATR[0]:
            self.buyStopLoss = True
        if self.histo<0:
            self.buyStopLoss = True
        if self.shortSig:
            self.buyStopLoss = True

        # 空单止损
        if (self.close[0] - self.last_price) > self.atr_rate_low*self.ATR[0]:
            self.shortStopLoss = True
        if (self.close[0] - self.close[-1]) > self.atr_rate_high*self.ATR[0]:
            self.shortStopLoss = True
        if self.histo>0:
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

