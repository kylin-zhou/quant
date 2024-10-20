# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt  # 引入backtrader框架

import akshare as ak
import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime
from copy import deepcopy
import os, sys
 
import torch
from ..strategy_ai.models import LSTMModel, LSTM, ALSTM, TCN


""" 深度学习策略
预测上涨, 买入, 做多
预测下跌, 卖出, 做空
TR止损
"""
 

def split_sequences(df, window=60, step=1):
    X = []
    for i in range(0, df.shape[0]-window, step):
        X.append(df.iloc[i:window+i,:].values)

    return np.array(X)

def get_feature(df, window=60):
    # 相对前日涨跌幅
    close_change = df.loc[1:,"close"].values - df.loc[:(df.shape[0]-2),"close"].values
    close_change = np.hstack([[0], close_change])
    df["close_change"] = (close_change / df.loc[:,"close"].values) * 100

    # open 相对前日 close
    open_close_change = df.loc[1:,"open"].values - df.loc[:(df.shape[0]-2),"close"].values
    open_close_change = np.hstack([[0], open_close_change])
    df["open_close_change"] = (open_close_change / df.loc[:,"close"].values) * 100

    def get_change(df, base="open"):
        base_change = df.loc[1:,base].values - df.loc[:(df.shape[0]-2),base].values
        base_change = np.hstack([[0], base_change])
        return (base_change / df.loc[:,base].values) * 100
        
    for col in ["open","high","low","volume"]:
        df[col+"_change"] = get_change(df, base=col)
    df["volume_change"].replace(-np.inf, 0, inplace=True)

    # 当日：开盘 收盘 最高 最低 差值 波动, 以开盘价和收盘价为基准
    df["high_open_rate"] = (df["high"] - df["open"]) / df["open"]
    df["low_open_rate"] = (df["low"] - df["open"]) / df["open"]
    df["close_open_rate"] = (df["close"] - df["open"]) / df["open"]
    df["high_close_rate"] = (df["high"] - df["close"]) / df["close"]
    df["low_close_rate"] = (df["low"] - df["close"]) / df["close"]

    df = df.drop(["open","high","low","close","volume"], axis=1)

    test_x = split_sequences(df=df.drop(["date"], axis=1), window=window)

    return test_x

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
        # 准备第一个标的沪深300主力合约的close、high、low 行情数据
        self.close = self.datas[0].close
        self.high = self.datas[0].high
        self.low = self.datas[0].low

        self.signal = self.datas[0].signal

        self.TR = ta.TRANGE(np.array(self.high), np.array(self.low), np.array(self.close))
 
        self.order = None     
        self.buy_count = 0 # 记录买入次数
        self.last_price = 0 # 记录买入价格
                
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
 
        if self.order: # 检查是否有指令等待执行, 如果还有订单在执行中，就不做新的仓位调整
            return
 
        # 如果当前持有多单
        if self.position.size > 0 :
            #多单止损：当价格回落2倍ATR时止损平仓
            if (self.close[-1] - self.close[-2]) <= -0.92*self.TR[-1]: 
            # if self.datas[0].close < (self.last_price - 2*self.ATR[0]):
                print('elif self.datas[0].close < (self.last_price - 2*self.ATR[0]):')
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
            # 反手做空
            elif self.signal[0] < 0.5:
                print('self.CrossoverL < 0 and self.buy_count == 0')
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.TR[-1]*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                self.order = self.sell(size=self.buy_unit)
                self.last_price = self.position.price # 记录买入价格
                self.buy_count = 1  # 记录本次交易价格
                
        # 如果当前持有空单
        elif self.position.size < 0 :          
            #空单止损：当价格上涨至2倍ATR时止损平仓
            if (self.close[-1] - self.close[-2]) <= -0.92*self.TR[-1]: 
            # if self.datas[0].close < (self.last_price+2*self.ATR[0]):
                print('self.datas[0].close < (self.last_price+2*self.ATR[0])')
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
            # 反手做多
            elif  self.signal[0] > 0.5:
                print('if self.CrossoverH > 0 and self.buy_count == 0:')
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.TR[-1]*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                self.order = self.buy(size=self.buy_unit)
                self.last_price = self.position.price # 记录买入价格
                self.buy_count = 1  # 记录本次交易价格
                
        else: # 如果没有持仓，等待入场时机
            #入场: 价格突破上轨线且空仓时，做多
            if  self.signal[0] > 0.5:
                print('if self.CrossoverH > 0 and self.buy_count == 0:')
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.TR[-1]*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                self.order = self.buy(size=self.buy_unit)
                self.last_price = self.position.price # 记录买入价格
                self.buy_count = 1  # 记录本次交易价格
            #入场: 价格跌破下轨线且空仓时，做空
            elif self.signal[0] < 0.5:
                print('self.CrossoverL < 0 and self.buy_count == 0')
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.TR[-1]*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                self.order = self.sell(size=self.buy_unit)
                self.last_price = self.position.price # 记录买入价格
                self.buy_count = 1  # 记录本次交易价格

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

class PandasData_more(bt.feeds.PandasData):
    lines = ('signal',) # 要添加的线
    # -1表示自动按列明匹配数据，也可以设置为线在数据源中列的位置索引 (('pe',6),('pb',7),) 
    params=(('signal', -1),)

def get_data(trader_code="AU0", start_date='2020-08-01', end_date='2022-08-01'):
    """https://akshare.akfamily.xyz/data/futures/futures.html#id54
    """
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
    
    window = 90
    print(history_df.head())
    feature_data = get_feature(deepcopy(history_df), window=window)
    model = LSTMModel(11, 16)
    model.load_state_dict(torch.load("D:\quant\checkpoint\w60_mse_model.pt"))
    pred = model(torch.tensor(feature_data, dtype=torch.float32)).detach().numpy()
    print(history_df.head())

    # 把 date 作为日期索引，以符合 Backtrader 的要求
    history_df.index = pd.to_datetime(history_df['date'])
    history_df["signal"] = np.hstack([np.zeros(window)+0.5, pred])

    # Create a Data Feed
    data = PandasData_more(dataname=history_df,
                                fromdate=pd.to_datetime(start_date),
                                todate=pd.to_datetime(end_date))
 
    return data
 
cerebro = bt.Cerebro()
cerebro.adddata(get_data(trader_code="AU0"), name='AU')

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



# 计算累计收益
cumulative = (daily_return + 1).cumprod()
# 计算回撤序列
max_return = cumulative.cummax()
drawdown = (cumulative - max_return) / max_return
# 计算收益评价指标
import pyfolio as pf
# 按年统计收益指标
perf_stats_year = (daily_return).groupby(daily_return.index.to_period('y')).apply(lambda data: pf.timeseries.perf_stats(data)).unstack()
# 统计所有时间段的收益指标
perf_stats_all = pf.timeseries.perf_stats((daily_return)).to_frame(name='all')
perf_stats = pd.concat([perf_stats_year, perf_stats_all.T], axis=0)
perf_stats_ = round(perf_stats,4).reset_index()
 
 
# 绘制图形
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import matplotlib.ticker as ticker # 导入设置坐标轴的模块
plt.style.use('seaborn') # plt.style.use('dark_background')
 
fig, (ax0, ax1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1.5, 4]}, figsize=(16,9))
cols_names = ['date', 'Annual\nreturn', 'Cumulative\nreturns', 'Annual\nvolatility',
       'Sharpe\nratio', 'Calmar\nratio', 'Stability', 'Max\ndrawdown',
       'Omega\nratio', 'Sortino\nratio', 'Skew', 'Kurtosis', 'Tail\nratio',
       'Daily value\nat risk']
 
# 绘制表格
ax0.set_axis_off() # 除去坐标轴
table = ax0.table(cellText = perf_stats_.values, 
                bbox=(0,0,1,1), # 设置表格位置， (x0, y0, width, height)
                rowLoc = 'right', # 行标题居中
                cellLoc='right' ,
                colLabels = cols_names, # 设置列标题
                colLoc = 'right', # 列标题居中
                edges = 'open' # 不显示表格边框
                )
table.set_fontsize(13)
 
# 绘制累计收益曲线
ax2 = ax1.twinx()
ax1.yaxis.set_ticks_position('right') # 将回撤曲线的 y 轴移至右侧
ax2.yaxis.set_ticks_position('left') # 将累计收益曲线的 y 轴移至左侧
# 绘制回撤曲线
drawdown.plot.area(ax=ax1, label='drawdown (right)', rot=0, alpha=0.3, fontsize=13, grid=False)
# 绘制累计收益曲线
(cumulative).plot(ax=ax2, color='#F1C40F' , lw=3.0, label='cumret (left)', rot=0, fontsize=13, grid=False)
# 不然 x 轴留有空白
ax2.set_xbound(lower=cumulative.index.min(), upper=cumulative.index.max())
# 主轴定位器：每 5 个月显示一个日期：根据具体天数来做排版
ax2.xaxis.set_major_locator(ticker.MultipleLocator(100)) 
# 同时绘制双轴的图例
h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
plt.legend(h1+h2,l1+l2, fontsize=12, loc='upper left', ncol=1)
 
fig.tight_layout() # 规整排版
plt.show()