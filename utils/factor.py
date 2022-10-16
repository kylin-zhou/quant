import os
import sys
import glob
import pathlib
from copy import deepcopy
from collections import Counter
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import talib
from sklearn import preprocessing

FUTURE = 10
TASK = "clf"

def get_ma_feature(df):
    periods = np.arange(3,30)
    for i in periods:
        df[f"ma{i}"] = talib.SMA(df["close"], timeperiod=i)
        # ma = talib.EMA(df["close"], timeperiod=i)
        # df[f"kama{i}"] = talib.KAMA(df["close"], timeperiod=i)
        # df[f"open_ma{i}"] = talib.EMA(df["open"], timeperiod=i)
        # df[f"ma{i}_close"] = ma - df["close"]
        
    # combines = combinations(periods, 2)
    # for i in combines:
    #     if i[0] < i[1]:
    #         df[f"ma{i[0]}-{i[1]}"] = df[f"ma{i[0]}"] - df[f"ma{i[1]}"]
            
    # combines = combinations(periods, 3)
    # for i in combines:
    #     if i[0] < i[1] < i[2]:
    #         df[f"ma{i[0]}-{i[1]}-{i[2]}"] = df[f"ma{i[0]}"] - df[f"ma{i[1]}"]- df[f"ma{i[2]}"]

def get_add_feature(df):
    low, high, close = df["low"], df["high"], df["close"]

    # Support and resistance
    periods = np.arange(3,30)
    for i in periods:
        df[f"support_{i}"] = df["close"].rolling(window=i).min() - df["close"]
        df[f"resistance_{i}"] = df["close"].rolling(window=i).max() - df["close"]
        df[f"shock_{i}"] = df["close"].rolling(window=i).std() - df["close"]
        
        df[f"up_count_{i}"] = (df["close"] - df["close"].shift(1)).apply(lambda x:1 if x>0 else 0)\
        .rolling(window=i).apply(lambda x:Counter(x)[1]) / i
        
        df[f"momentum_{i}"] = df["close"] / df["close"].shift(i) - 1
        

def get_momentum_feature(df):
    low, high, close = df["low"], df["high"], df["close"]
    
    # ADX
    adx = talib.ADX(high, low, close, timeperiod=5)
    adx2 = talib.ADX(high, low, close, timeperiod=14)
    df["adx"] = adx
    df["adx2"] = adx2
    
    # AROON
    aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
    df["aroondown"], df["aroonup"] = aroondown, aroonup
    
    # CCI
    cci = talib.CCI(high, low, close, timeperiod=7)
    cci2 = talib.CCI(high, low, close, timeperiod=14)
    df["cci"] = cci
    df["cci2"] = cci2
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"], df["macdsignal"], df["macdhist"] = macd, macdsignal, macdhist
    
    # MOM
    mom = talib.MOM(close, timeperiod=5)
    mom2 = talib.MOM(close, timeperiod=10)
    df["mom"] = mom
    df["mom"] = mom2
    
    # ROC
    roc = talib.ROC(close, timeperiod=5)    
    roc2 = talib.ROC(close, timeperiod=10)
    df["roc"] = roc
    df["roc2"] = roc2
    
    # RSI
    rsi = talib.RSI(close, timeperiod=7)
    rsi2 = talib.RSI(close, timeperiod=14)
    df["rsi"] = rsi
    df["rsi2"] = rsi2
    
    # KD
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df["slowk"], df["slowd"] = slowk, slowd
    
    # WILLR
    willr = talib.WILLR(high, low, close, timeperiod=7)
    willr2 = talib.WILLR(high, low, close, timeperiod=14)
    df["willr"] = willr
    df["willr2"] = willr2


def get_volume_feature(df):
    low, high, close, volume = df["low"], df["high"], df["close"], df["volume"]
    
    # AD
    ad = talib.AD(high, low, close, volume)
    df["ad"] = ad
    
    # ADOSC
    adsoc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df["adsoc"] = adsoc
    
    # OBV
    obv = talib.OBV(close, volume)
    df["obv"] = obv


def get_volatility_feature(df):
    low, high, close = df["low"], df["high"], df["close"]
    
    # ATR
    atr = talib.ATR(high, low, close, timeperiod=14)
    df["atr"] = atr
    
    # TRANGE
    trange = talib.TRANGE(high, low, close)
    df["trange"] = trange


def get_close_change(df, base="close"):
    # close change, use for label
    gap = FUTURE
    base_change = df.loc[gap:,base].values / df.loc[:(df.shape[0]-1-gap),base].values - 1
    base_change_rate = base_change * 100
    return np.hstack([base_change_rate,[0]*gap])

def get_label(close, avg_close, std_close):
    if (avg_close > close+30):
        return 1
    elif (avg_close < close-30):
        return 0
    else:
        return 2

def get_feature_df(df):
    df = deepcopy(df)
    
    # close_change = get_close_change(df, base="close")
    # close_open_rate = (df["close"] - df["open"]) / df["open"] * 100
        
    assert TASK in ("clf","reg")
    # calculate up/down
    if TASK=="clf":
        # labels = []
        # for cc,co in zip(np.hstack([close_change[1:], [0]]),
        #                  np.hstack([close_open_rate[1:], [0]])):
        #     if cc > 0 and co > 0:
        #         label = 1 # buy
        #     elif cc < 0 and co < 0:
        #         label = 0 # short
        #     else:
        #         label = 2 # no change
        #     labels.append(label)
        # df["label"] = np.array(labels)
        # labels = close_change
        # df["label"] = np.array(list(map(lambda x:1 if x>0 else 0, labels)))
        window = FUTURE
        avg_close = df["close"].rolling(window=window).mean().shift(-window)
        # std_close = df["close"].rolling(window=window).std().shift(window)
        # future_df = pd.concat([df["close"], avg_close, std_close], axis=1)
        # future_df.columns = ["close","avg_close","std_close"]
        # df["label"] = future_df.apply(lambda x: get_label(x.close,x.avg_close,x.std_close), axis=1)
        df["label"] = (avg_close - df["close"]).apply(lambda x: 1 if x>0 else 0)
    if TASK=="reg":
        df["label"] = close_change
    
    # calculate factor
    # get_ma_feature(df)
    get_add_feature(df)
    # get_momentum_feature(df)
    # get_volume_feature(df)
    # get_volatility_feature(df)
    
    # drop anomaly value
    df = df.dropna().reset_index(drop=True)
    df = df.drop(0, axis=0).reset_index(drop=True)
    df = df.drop([len(df)-1], axis=0).reset_index(drop=True)
    return df

def get_feature_label(df):
    feature_df = df.drop(["date","open","high","low","close","volume","label"],axis=1)
    # feature_df = (feature_df-feature_df.min())/(feature_df.max()-feature_df.min())
    feature_df = (feature_df-feature_df.mean())/(feature_df.std())
    feature_df = feature_df.dropna().reset_index(drop=True)
    features = feature_df.values
    labels = df["label"].values
    return features, labels