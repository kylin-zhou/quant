
import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
from sys import float_info as sflt
from loguru import logger

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def zero(x):
    """If the value is close to zero, then return zero. Otherwise return it"""
    return 0 if abs(x) < sflt.epsilon else x
    
def calculate_sar(high, low, N=4, step=2, mvalue=20):
    step1 = step/100
    mvalue1 = mvalue/100
    ep, af, sar = [0]*len(high),[0]*len(high),[0]*len(high)
    low_pre = float('inf') #上一个周期最小值
    high_pre = float('-inf') #上一个周期最大值
    
    def _falling(high, low, drift:int=1):
        """Returns the last -DM value"""
        # Not to be confused with ta.falling()
        up = high - high.shift(drift)
        dn = low.shift(drift) - low
        _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
        return _dmn > 0

    # Falling if the first NaN -DM is positive
    trend = _falling(high.iloc[:2], low.iloc[:2])

    for i in range(N,len(high)):
        #先确定sr0
        if trend and i == N :#涨势 
            ep[i] = min(low[0:N-1])
            sar[i] = ep[i]
            continue
        elif trend == False and i == N:#跌势 
            ep[i] = max(high[0:N-1])
            sar[i] = ep[i]*-1
            continue
              
        af[i] = af[i-1]+step1
        if  af[i] > mvalue1:
             af[i] = mvalue1
        if trend:    
            sar[i] = abs(sar[i-1])+af[i]*(high[i-1]-abs(sar[i-1]))
            ep[i] = max(ep[i-1],high[i])
            high_pre = max(high_pre,high[i])

            if low[i] < abs(sar[i]): # 趋势反转
               trend = not trend
               af[i] = 0
               low_pre = low[i]
               sar[i] = max(high[0:i]) if low_pre == 0 else high_pre*-1

        else :
            sar[i] = -1 * (abs(sar[i-1])+af[i]*(low[i-1]-abs(sar[i-1])))
            ep[i] = min(ep[i-1],low[i])
            low_pre = min(low_pre,low[i])
            
            if high[i] > abs(sar[i]): # 趋势反转
               trend = not trend
               high_pre = high[i] 
               af[i] = 0
               sar[i] = min(low[0:i]) if high_pre == 0 else low_pre
 
    return sar


def calculate_signal(df):
    # params
    macd_fast, macd_slow, macd_signal = 20, 50, 10
    ma0_length, ma1_length, ma2_length = 20, 50, 150
    atr_multiplier_loss, atr_multiplier_profit = 2,3
    atr_gap = atr_multiplier_profit - atr_multiplier_loss
    highest_price, lowest_price = float('-inf'), float('inf')
    position_size, position_size = 0,0
    total_profit, profit = 0, 0

    # indicator
    sar = calculate_sar(df['high'], df['low'], N=4, step=2, mvalue=20)
    macd_df = df.ta.macd(close=df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    macd, macdhist, macdsignal = (
        macd_df[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'], 
        macd_df[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'], 
        macd_df[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'] 
    )
    atr = df.ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14).values
    ma10 = df.ta.sma(length=10).values
    ma0 = df.ta.sma(length=ma0_length).values
    ma1 = df.ta.sma(length=ma1_length).values
    ma2 = df.ta.sma(length=ma2_length).values
    ma3 = df.ta.sma(length=250).values
    bias = df.ta.bias(length=50, mamode="sma").values * 100

    df['macdhist'], df['sar'] = macdhist, np.abs(sar)
    df['ma1'], df['ma2'], df['ma3'] = ma1, ma2, ma3

    close, open = df.close.values, df.open.values
    macdhist, sar = np.array(macdhist.values), np.array(sar)

    # ========================= calculate signal

    # init value
    df['signal'] = 0

    for index in range(50, len(df)):
        buy_signal, sell_signal = False, False
        signal = 0

        if ma1[index] > ma2[index] > ma3[index] and bias[index] < 1 and ma2[index] > ma2[index-1]:
        # cond 1, 均线多头, 上升趋势, 强势上涨, sar做多
            if (
                ((macdhist[index] > 0 and np.any(macdhist[index-6:index] < 0) and sar[index] > 0 and sar[index-1] < 0) or
                (sar[index] > 0 and np.any(sar[index-6:index] < 0) and macdhist[index] > 0 and macdhist[index-1] < 0))
            ):
                buy_signal = True

        elif ma1[index] < ma2[index] < ma3[index] and bias[index] > -1 and ma2[index] < ma2[index-1]:
        # cond 2, 均线空头, 下降趋势, 强势下跌, sar做空
            if (
                ((macdhist[index] < 0 and np.any(macdhist[index-6:index] > 0) and sar[index] < 0 and sar[index-1] > 0) or
                (sar[index] < 0 and np.any(sar[index-6:index] > 0) and macdhist[index] < 0 and macdhist[index-1] > 0))
            ):
                sell_signal = True

        if ma0[index] > ma1[index] > ma2[index] and ma1[index] > ma1[index-1] and ma2[index] > ma2[index-1]:
            sell_signal = False
        if ma0[index] < ma1[index] < ma2[index] and ma1[index] < ma1[index-1] and ma2[index] < ma2[index-1]:
            buy_signal = False
        
        if position_size == 0: # 空仓
            if buy_signal:
                stop_loss_price = close[index] - atr[index] * atr_multiplier_loss
                # position_size = 1
                position_price = close[index]
                signal = 1
                logger.info(f"buy long, price: {position_price}")
            elif sell_signal:
                stop_loss_price = close[index] + atr[index] * atr_multiplier_loss
                # position_size = -1
                position_price = close[index]
                signal = -1
                logger.info(f"sell short, price: {position_price}")

        # elif position_size > 0: # 持多
        #     if close[index] > position_price + atr_gap*atr[index]: # price increase, update stop loss
        #         highest_price = max(close[index], highest_price)
        #         stop_loss_price = highest_price - atr[index] * atr_multiplier_profit
        #         # logger.info("increase stop loss price")

        #     if close[index] < stop_loss_price: # close
        #         signal = -1
        #         profit = close[index] < stop_loss_price
        #         position_size, highest_price, lowest_price = 0, float('-inf'), float('inf')
        #         profit = close[index] - position_price
        #         total_profit += profit
        #         logger.info(f"close long, price: {close[index]} profit: {profit}")
        #     # if buy_signal:
        #     #     signal = 1

        # else: # 持空
        #     if close[index] < position_price - atr_gap*atr[index]: # price decrease, update stop loss
        #         lowest_price = min(close[index], lowest_price)
        #         stop_loss_price = lowest_price + atr[index] * atr_multiplier_profit
        #         # logger.info("decrease stop loss price")
        #     if close[index] > stop_loss_price: # close
        #         signal = 1
        #         position_size, highest_price, lowest_price = 0, float('-inf'), float('inf')
        #         profit = position_price - close[index]
        #         total_profit += profit
        #         logger.info(f"close short, price: {close[index]} profit: {profit}")
            # if sell_signal:
            #     signal = -1

        df.loc[index, "signal"] = signal
        
    logger.info(f"total profit: {total_profit}")

    return df

def plot_signal_interactive(new, ticker=""):
    # 创建带有3个子图的图表，设置高度比例
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.02,  # 减小子图间距
        subplot_titles=('Price & SAR', 'MACD Histogram', 'Moving Averages'),
        row_heights=[0.6, 0.2, 0.2]  # 设置每个子图的高度比例
    )
    
    # 第一个子图：价格、SAR和信号
    fig.add_trace(
        go.Scatter(
            x=new.index, 
            y=new['close'], 
            name=ticker,
            line=dict(width=3)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=new.index, 
            y=new['sar'], 
            name='Parabolic SAR',
            mode='markers',
            marker=dict(
                size=4,
                color = ['red' if price > sar else 'green' for price, sar in zip(new['close'],new['sar'])],
                symbol='circle'
            )
        ), row=1, col=1
    )
    
    # 添加做多信号
    fig.add_trace(
        go.Scatter(
            x=new.loc[new['signal']==1].index, 
            y=new['close'][new['signal']==1],
            name='LONG', 
            mode='markers',
            marker=dict(
                size=12, 
                symbol='triangle-up', 
                color='red'
            )
        ), row=1, col=1
    )
    
    # 添加做空信号
    fig.add_trace(
        go.Scatter(
            x=new.loc[new['signal']==-1].index, 
            y=new['close'][new['signal']==-1],
            name='SHORT', 
            mode='markers',
            marker=dict(
                size=12, 
                symbol='triangle-down', 
                color='black'
            )
        ), row=1, col=1
    )

    for index, date in zip(new.index,new.datetime):
        if index % 180 == 0:
            fig.add_annotation(
                x=index,
                y=min(new.close)-100,
                text=date,  # 格式化日期文本
                showarrow=False,
                yshift=10  # 控制文本的垂直位置
            )
    
    # 第二个子图：MACD柱状图
    fig.add_trace(
        go.Bar(
            x=new.index, 
            y=new['macdhist'], 
            name='MACD Histogram',
            marker_color='red'
        ), row=2, col=1
    )
    
    # 第三个子图：移动平均线
    # fig.add_trace(
    #     go.Scatter(
    #         x=new.index, 
    #         y=new['ma1'], 
    #         name='MA1'
    #     ), row=3, col=1
    # )
    
    fig.add_trace(
        go.Scatter(
            x=new.index, 
            y=new['ma2'], 
            name='MA2',
            line=dict(dash='dot')
        ), row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=new.index, 
            y=new['ma3'], 
            name='MA3',
            line=dict(dash='dot')
        ), row=3, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=900,  # 总高度
        showlegend=True,
        title={
            'text': f"{ticker} Technical Analysis",
            'y':0.98,  # 调整标题位置
            'x':0.01,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        margin=dict(t=30, l=50, r=50, b=30),  # 调整边距
        xaxis3_title="Date",
        hovermode='x unified',
        template='plotly_white',
        dragmode='pan',  # 默认拖动模式设为平移
        modebar=dict(
            add=[
                'scrollZoom', 
                'zoomIn', 
                'zoomOut', 
                'resetScale'
            ],
            remove=['lasso', 'select']
        )
    )
    
    # 为所有子图添加网格线和配置缩放选项
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='LightGray',
        rangeslider=dict(visible=False),
        scaleanchor="x",
        showspikes=True,  # 添加十字准线
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikedash='solid'
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='LightGray',
        fixedrange=False,
        showspikes=True,  # 添加十字准线
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikedash='solid'
    )
    
    # 配置具体的交互行为
    config = {
        'scrollZoom': True,  # 启用滚轮缩放
        'modeBarButtonsToAdd': [
            'drawopenpath',  # 自由画线
            'drawline',      # 直线
            'eraseshape'     # 橡皮擦
        ],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{ticker}_analysis',
            'height': 900,
            'width': 1200,
            'scale': 2
        }
    }
    
    # 显示图表
    fig.show(config=config)
    fig.write_html('analysis.html', config=config)


if __name__=="__main__":
    df = pd.read_csv(f"D:/trading/quant/data/futures/{sys.argv[1]}.csv")
    df = calculate_signal(df)
    # plot_signal(df.iloc[100:], sys.argv[1])
    plot_signal_interactive(df.iloc[100:], sys.argv[1])