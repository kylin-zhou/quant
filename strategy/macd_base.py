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
 
class StrategyClass(bt.Strategy):
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
            if self.macd > 0 and self.signal > 0 and self.histo > 0:
                self.log('BUY CREATE,{}'.format(self.dataclose[0]))
                self.order = self.buy(self.datas[0])
 
        else:
 
            # 收盘价小于等于histo，卖出
            if self.macd <= 0 or self.signal <= 0 or self.histo <= 0:
                self.log('BUY CREATE,{}'.format(self.dataclose[0]))
                self.log('Pos size %s' % self.position.size)
                self.order = self.sell(self.datas[0])
 
 
def get_data(trader_code="AU0", start_date='2022-01-01', end_date='2022-08-01'):
 
    history_df = ak.futures_main_sina(trader_code, start_date=start_date, end_date=end_date).iloc[:, :6]
    # 处理字段命名，以符合 Backtrader 的要求
    history_df.columns = [
        'date',
        'open',
        'high',
        'low',
        'close',
        'volume',
    ]
    # 把 date 作为日期索引，以符合 Backtrader 的要求
    history_df.index = pd.to_datetime(history_df['date'])
 
    # Create a Data Feed
    data = bt.feeds.PandasData(dataname=history_df,
                                fromdate=pd.to_datetime(start_date),
                                todate=pd.to_datetime(end_date))
 
    return data
 
cerebro = bt.Cerebro()
cerebro.adddata(get_data(trader_code="MA0"), name='IF')

# 初始资金 100,000
start_cash = 100000
cerebro.broker.setcash(start_cash)  # 设置初始资本为 100000
cerebro.broker.setcommission(commission=0.1, # 按 0.1% 来收取手续费
                             mult=300, # 合约乘数
                             margin=0.1, # 保证金比例
                             percabs=False, # 表示 commission 以 % 为单位
                             commtype=bt.CommInfoBase.COMM_FIXED,
                             stocklike=False)

# 加入策略
cerebro.addstrategy(StrategyClass)
# 回测时需要添加 PyFolio 分析器
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl') # 返回收益率时序数据
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn') # 年化收益率
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio') # 夏普比率
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown') # 回撤

result = cerebro.run() # 运行回测系统
# 从返回的 result 中提取回测结果
strat = result[0]
# # 返回日度收益率序列
daily_return = pd.Series(strat.analyzers.pnl.get_analysis())
# 打印评价指标
print("--------------- AnnualReturn -----------------")
print(strat.analyzers._AnnualReturn.get_analysis())
print("--------------- SharpeRatio -----------------")
print(strat.analyzers._SharpeRatio.get_analysis())
print("--------------- DrawDown -----------------")
print(strat.analyzers._DrawDown.get_analysis())


port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
pnl = port_value - start_cash  # 盈亏统计

print(f"初始资金: {start_cash}")
print(f"总资金: {round(port_value, 2)}")
print(f"净收益: {round(pnl, 2)}")

cerebro.plot(style='candlestick')  # 画图