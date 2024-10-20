# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt  # 引入backtrader框架
import matplotlib.pyplot as plt

import os, sys
import akshare as ak
import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime
import argparse
from loguru import logger

import quantstats as qs
# extend pandas functionality with metrics, etc.
qs.extend_pandas()

from strategy import get_strategy_cls

logger.add('backtest.log', level='INFO', encoding='utf-8', format='{message}', mode='w')
 
def get_data(symbol, period=5, start_date='2022-01-01', end_date='2023-09-27'):
    """https://akshare.akfamily.xyz/data/futures/futures.html#id54
    """
    # history_df = ak.futures_main_sina(trader_code, start_date=start_date, end_date=end_date).iloc[:, :6]
    history_df = ak.futures_zh_minute_sina(symbol=symbol, period=period).iloc[:, :6]
    # history_df = ak.fund_etf_hist_sina(symbol="sh588000")
    # history_df = pd.read_csv("D:/quant/data/futures/dominant/TA9999.XZCE.30m.csv")

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
                                # fromdate=pd.to_datetime(start_date),
                                # todate=pd.to_datetime(end_date)
                                )
 
    return data

def main(StrategyClass, symbol):
    cerebro = bt.Cerebro()
    cerebro.adddata(get_data(symbol), name=f'{symbol}')

    # 初始资金 10,000
    start_cash = 10000
    cerebro.broker.setcash(start_cash)  # 设置初始资本为 10000
    cerebro.broker.setcommission(commission=0.001, # 按 0.1% 来收取手续费
                                margin=0.1, # 保证金比例
                                mult=10, # 合约乘数
                                percabs=False, # 表示 commission 以 % 为单位
                                commtype=bt.CommInfoBase.COMM_FIXED,
                                stocklike=True)

    # 加入策略
    cerebro.addstrategy(StrategyClass)
    # 回测时需要添加 PyFolio 分析器
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl') # 返回收益率时序数据
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn') # 年化收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio') # 夏普比率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown') # 回撤
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer') # 交易统计

    # cerebro.addwriter(bt.WriterFile, csv=True, out='log.csv')

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
    
    print(" win rate\t{:.3f}\n win_loss_ratio\t{:.3f}\n avg_return\t{:.3f}\n avg_win\t{:.3f}\n avg_loss\t{:.3f}".format(
        qs.stats.win_rate(daily_return), qs.stats.win_loss_ratio(daily_return),
        qs.stats.avg_return(daily_return), qs.stats.avg_win(daily_return), qs.stats.avg_loss(daily_return)
    ))
    qs.reports.html(daily_return, output='stats.html', title='Stock Sentiment')
    # qs.reports.metrics(daily_return, mode="basic")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="backtest")
    parser.add_argument(
        "-s",
        "--strategy",
        help="strategy name",
        default="macd_sar",
    )
    parser.add_argument(
        "-f",
        "--future",
        help="future contract",
        default="rb0",
    )
    args = parser.parse_args()

    strategy = get_strategy_cls[args.strategy]
    
    main(StrategyClass=strategy, symbol=args.future)