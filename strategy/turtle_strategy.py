from datetime import datetime

import backtrader as bt  # 升级到最新版
import matplotlib.pyplot as plt  # 由于 Backtrader 的问题，此处要求 pip install matplotlib==3.2.2
import akshare as ak  # 升级到最新版
import pandas as pd

class TurtleTradingStrategy(bt.Strategy):
    params = dict(
        N1= 20, # 唐奇安通道上轨的t
        N2=10, # 唐奇安通道下轨的t
        )
    
    def log(self, txt, dt=None):           
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self): 
        self.order = None                   
        self.buy_count = 0 # 记录买入次数
        self.last_price = 0 # 记录买入价格
        # 准备第一个标的沪深300主力合约的close、high、low 行情数据
        self.close = self.datas[0].close
        self.high = self.datas[0].high
        self.low = self.datas[0].low
        # 计算唐奇安通道上轨：过去20日的最高价
        self.DonchianH = bt.ind.Highest(self.high(-1), period=self.p.N1, subplot=True)
        # 计算唐奇安通道下轨：过去10日的最低价
        self.DonchianL = bt.ind.Lowest(self.low(-1), period=self.p.N2, subplot=True)
        # 生成唐奇安通道上轨突破：close>DonchianH，取值为1.0；反之为 -1.0
        self.CrossoverH = bt.ind.CrossOver(self.close(0), self.DonchianH, subplot=False)
        # 生成唐奇安通道下轨突破:
        self.CrossoverL = bt.ind.CrossOver(self.close(0), self.DonchianL, subplot=False)
        # 计算 ATR
        self.TR = bt.ind.Max((self.high(0)-self.low(0)), # 当日最高价-当日最低价
                                    abs(self.high(0)-self.close(-1)), # abs(当日最高价−前一日收盘价)
                                    abs(self.low(0)-self.close(-1))) # abs(当日最低价-前一日收盘价)
        self.ATR = bt.ind.SimpleMovingAverage(self.TR, period=self.p.N1, subplot=False)
        # 计算 ATR，直接调用 talib ，使用前需要安装 python3 -m pip install TA-Lib
        # self.ATR = bt.talib.ATR(self.high, self.low, self.close, timeperiod=self.p.N1, subplot=True)
    
    
    def next(self): 
        # 如果还有订单在执行中，就不做新的仓位调整
        if self.order:
            return  
                
        # 如果当前持有多单
        if self.position.size > 0 :
            # 多单加仓:价格上涨了买入价的0.5的ATR且加仓次数少于等于3次
            if self.datas[0].close >self.last_price + 0.5*self.ATR[0] and self.buy_count <= 4:
                print('if self.datas[0].close >self.last_price + 0.5*self.ATR[0] and self.buy_count <= 4:')
                print('self.buy_count',self.buy_count)
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.ATR*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                # self.sizer.p.stake = self.buy_unit
                self.order = self.buy(size=self.buy_unit)
                self.last_price = self.position.price # 获取买入价格
                self.buy_count = self.buy_count + 1
            #多单止损：当价格回落2倍ATR时止损平仓
            elif self.datas[0].close < (self.last_price - 2*self.ATR[0]):
                print("多单止损")
                print('elif self.datas[0].close < (self.last_price - 2*self.ATR[0]):')
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0
            #多单止盈：当价格突破10日最低点时止盈离场 平仓
            elif self.CrossoverL < 0:
                print("多单止盈")
                print('self.CrossoverL < 0')
                self.order = self.sell(size=abs(self.position.size))
                self.buy_count = 0 
                
        # 如果当前持有空单
        elif self.position.size < 0 :
            # 空单加仓:价格小于买入价的0.5的ATR且加仓次数少于等于3次
            if self.datas[0].close<self.last_price-0.5*self.ATR[0] and self.buy_count <= 4:
                print('self.datas[0].close<self.last_price-0.5*self.ATR[0] and self.buy_count <= 4')
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.ATR*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                # self.sizer.p.stake = self.buy_unit
                self.order = self.sell(size=self.buy_unit)
                self.last_price = self.position.price # 获取买入价格
                self.buy_count = self.buy_count + 1              
            #空单止损：当价格上涨至2倍ATR时止损平仓
            elif self.datas[0].close < (self.last_price+2*self.ATR[0]):
                print('self.datas[0].close < (self.last_price+2*self.ATR[0])')
                print("空单止损")
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
            #空单止盈：当价格突破20日最高点时止盈平仓
            elif self.CrossoverH>0:
                print('self.CrossoverH>0')
                print("空单止盈")
                self.order = self.buy(size=abs(self.position.size))
                self.buy_count = 0
                
        else: # 如果没有持仓，等待入场时机
            #入场: 价格突破上轨线且空仓时，做多
            if self.CrossoverH > 0 and self.buy_count == 0:
                print('if self.CrossoverH > 0 and self.buy_count == 0:')
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.ATR*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                self.order = self.buy(size=self.buy_unit)
                self.last_price = self.position.price # 记录买入价格
                self.buy_count = 1  # 记录本次交易价格
            #入场: 价格跌破下轨线且空仓时，做空
            elif self.CrossoverL < 0 and self.buy_count == 0:
                print('self.CrossoverL < 0 and self.buy_count == 0')
                # 计算建仓单位：self.ATR*期货合约乘数300*保证金比例0.1
                self.buy_unit = max((self.broker.getvalue()*0.005)/(self.ATR*300*0.1),1)
                self.buy_unit = int(self.buy_unit) # 交易单位为手
                self.order = self.sell(size=self.buy_unit)
                self.last_price = self.position.price # 记录买入价格
                self.buy_count = 1  # 记录本次交易价格
        
    # 打印订单日志
    def notify_order(self, order):
        order_status = ['Created','Submitted','Accepted','Partial',
                        'Completed','Canceled','Expired','Margin','Rejected']
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            self.log('ref:%.0f, name: %s, Order: %s'% (order.ref,
                                                   order.data._name,
                                                   order_status[order.status]))
            return
        # 已经处理的订单
        if order.status in [order.Partial, order.Completed]:
            if order.isbuy():
                self.log(
                        'BUY EXECUTED, status: %s, ref:%.0f, name: %s, Size: %.2f, Price: %.2f, Cost: %.2f, Comm %.2f' %
                        (order_status[order.status], # 订单状态
                         order.ref, # 订单编号
                         order.data._name, # 股票名称
                         order.executed.size, # 成交量
                         order.executed.price, # 成交价
                         order.executed.value, # 成交额
                         order.executed.comm)) # 佣金
            else: # Sell
                self.log('SELL EXECUTED, status: %s, ref:%.0f, name: %s, Size: %.2f, Price: %.2f, Cost: %.2f, Comm %.2f' %
                            (order_status[order.status],
                             order.ref,
                             order.data._name,
                             order.executed.size,
                             order.executed.price,
                             order.executed.value,
                             order.executed.comm))
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            # 订单未完成
            self.log('ref:%.0f, name: %s, status: %s'% (
                order.ref, order.data._name, order_status[order.status]))
            
        self.order = None
        
    def notify_trade(self, trade):
        # 交易刚打开时
        if trade.justopened:
            self.log('Trade Opened, name: %s, Size: %.2f,Price: %.2f' % (
                    trade.getdataname(), trade.size, trade.price))
        # 交易结束
        elif trade.isclosed:
            self.log('Trade Closed, name: %s, GROSS %.2f, NET %.2f, Comm %.2f' %(
            trade.getdataname(), trade.pnl, trade.pnlcomm, trade.commission))
        # 更新交易状态
        else:
            self.log('Trade Updated, name: %s, Size: %.2f,Price: %.2f' % (
                    trade.getdataname(), trade.size, trade.price))
   
  
# 准备股票日线数据，输入到backtrader
# 利用 AKShare 获取股票的后复权数据，这里只获取前 6 列
IF_price = ak.futures_main_sina("MA0", start_date='2022-01-01', end_date='2023-01-01').iloc[:, :6]
# 处理字段命名，以符合 Backtrader 的要求
IF_price.columns = [
    'date',
    'open',
    'high',
    'low',
    'close',
    'volume',
]
# 把 date 作为日期索引，以符合 Backtrader 的要求
IF_price.index = pd.to_datetime(IF_price['date'])

datafeed = bt.feeds.PandasData(dataname=IF_price,
                           fromdate=pd.to_datetime('2017-01-01'),
                           todate=pd.to_datetime('2022-04-30'))
                           
# 创建主控制器
cerebro = bt.Cerebro()
cerebro.adddata(datafeed, name='IF')

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
cerebro.addstrategy(TurtleTradingStrategy)
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