# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import quantstats as qs
# extend pandas functionality with metrics, etc.
qs.extend_pandas()

from strategy.backtrader import get_strategy_cls

 
def get_data(symbol, period=5, start_date='2022-01-01', end_date='2023-09-27'):
    """https://akshare.akfamily.xyz/data/futures/futures.html#id54
    """
    try:
        history_df = pd.read_csv(f"D:/trading/quant/data/futures/{symbol}.csv")
    except:
        history_df = ak.futures_zh_minute_sina(symbol=symbol, period=period).iloc[:, :6]
        history_df.to_csv(f"D:/trading/quant/data/futures/{symbol}.csv", index=False)
    # history_df = ak.fund_etf_hist_sina(symbol="sh588000")

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
    logger.info(f'backtesting {symbol}...')
    cerebro = bt.Cerebro()
    cerebro.adddata(get_data(symbol), name=f'{symbol}')

    # 初始资金 10,000
    start_cash = 10000
    cerebro.broker.setcash(start_cash)  # 设置初始资本为 10000
    cerebro.broker.setcommission(commission=0.00001, # 按 0.1% 来收取手续费
                                margin=0.09, # 保证金比例
                                mult=10, # 合约乘数
                                percabs=False, # 表示 commission 以 % 为单位
                                commtype=bt.CommInfoBase.COMM_FIXED,
                                stocklike=False)

    # 加入策略
    cerebro.addstrategy(StrategyClass)
    cerebro.addwriter(bt.WriterFile, csv=True, out=f'./backtest_result/trading_log_{symbol}.csv')
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

    port_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
    pnl = port_value - start_cash  # 盈亏统计

    trade_analysis = strat.analyzers._TradeAnalyzer.get_analysis()

    # 打印评价指标
    # logger.info("--------------- AnnualReturn -----------------")
    # logger.info(strat.analyzers._AnnualReturn.get_analysis())
    # logger.info("--------------- SharpeRatio -----------------")
    # logger.info(strat.analyzers._SharpeRatio.get_analysis())

    logger.info(f"初始资金: {start_cash}")
    logger.info(f"总资金: {round(port_value, 2)}")
    logger.info(f"净收益: {round(pnl, 2)}")
    logger.info(f"交易次数: {trade_analysis.total.total}")
    logger.info(f"胜率: {trade_analysis.won.total / trade_analysis.total.total:.2f}")
    logger.info(f"盈亏比: {trade_analysis.won.pnl.average / abs(trade_analysis.lost.pnl.average):.2f}")  # 需要处理除零错误
    logger.info(f'回撤: {strat.analyzers._DrawDown.get_analysis().max.drawdown/100:.2f}')


    # logger.info("win rate\t{:.3f}\nwin_loss_ratio\t{:.3f}\navg_return\t{:.3f}\navg_win\t{:.3f}\navg_loss\t{:.3f}".format(
    #     qs.stats.win_rate(daily_return), qs.stats.win_loss_ratio(daily_return),
    #     qs.stats.avg_return(daily_return), qs.stats.avg_win(daily_return), qs.stats.avg_loss(daily_return)
    # ))
    # qs.reports.html(daily_return, output='stats.html', title='Stock Sentiment')
    # qs.reports.metrics(daily_return, mode="basic")
    
    if args.future is not None:
        try:
            fig = cerebro.plot(style='candlestick', dpi=600)[0][0]  # 画图
            fig.savefig(f'./backtest_result/backtest_{strategy.__name__}_{symbol}.png', dpi=1200)
        except Exception as e:
            pass
            print(f"画图失败, {e}")

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
        help="future contract, example: -f v0 rb0",
        nargs="+",
        default=None,
        required=False
    )
    parser.add_argument(
        "-m",
        "--comment",
        help="comment for backtest, example: -m test",
        default="",
        required=True
    )
    args = parser.parse_args()

    logger.add(f'backtest_{args.comment}.log', level='INFO', encoding='utf-8', format='{message}', mode='w')

    strategy = get_strategy_cls[args.strategy]
    
    if args.future is None:
        futures = []
        for s in ['v','pp','rb','ma','c']: # ,'fg','eb','sr','sa'
        # for s in ['v']:
            for t in ["20","21", "22", "23", "24", "25"]:
                for post in ['01', '05', '09']:
                    if t == "25" and post != "01": continue
                    futures.append(f"{s}{t}{post}")
    else:
        futures = args.future

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(main, [strategy] * len(futures), futures)

    with open(f'backtest_{args.comment}.log', "r", encoding='utf-8') as f:
        data = f.readlines()

    profit = 0
    total_trades, win_rates, win_loss_ratios = [], [], []
    drawdowns = []
    for line in data:
        if line.startswith("净收益"):
            profit += float(line.split(":")[-1].strip())
        if line.startswith("胜率"):
            win_rate = float(line.split(":")[-1].strip())
            if win_rate > 0 and win_rate < 1:
                win_rates.append(win_rate)
        if line.startswith("盈亏比"):
            win_loss_ratio = float(line.split(":")[-1].strip())
            if win_loss_ratio > 0 and win_loss_ratio < 10:
                win_loss_ratios.append(win_loss_ratio)
        if line.startswith("交易次数"):
            total_trades.append(float(line.split(":")[-1].strip()))
        if line.startswith("回撤"):
            drawdown = float(line.split(":")[-1].strip())
            if drawdown > 0:
                drawdowns.append(drawdown)

    logger.info("--------------- Strategy analysis -----------------")
    logger.info(f"--------------- {strategy.__name__} {args.comment}-----------------")
    logger.info(
        (
            f"\n总收益: {profit}"
            f"\n交易次数: {np.sum(total_trades)}\n胜率: {np.mean(win_rates)}\n盈亏比: {np.mean(win_loss_ratios)}"
            f"\n最大回撤: {np.max(drawdowns)}\n平均回撤: {np.mean(drawdowns)}"
        )
    )