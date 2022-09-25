import os
import sys
import glob
import pathlib
from copy import deepcopy
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import talib
from sklearn import preprocessing

FUTURE = 10
TASK="clf"

def get_ma_feature(df):
    periods = np.arange(2,25)
    for i in periods:
        df[f"ma{i}"] = talib.SMA(df["close"], timeperiod=i)
        df[f"ema{i}"] = talib.EMA(df["close"], timeperiod=i)
        df[f"kama{i}"] = talib.KAMA(df["close"], timeperiod=i)
        # df[f"open_ma{i}"] = talib.EMA(df["open"], timeperiod=i)
        
    combines = combinations(periods, 2)
    for i in combines:
        if i[0] < i[1]:
            df[f"ma{i[0]}-{i[1]}"] = df[f"ma{i[0]}"] - df[f"ma{i[1]}"]
        
    return df

def get_add_feature(df):
    low, high, close = df["low"], df["high"], df["close"]
    
    # Bollinger Bands 
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df["upperband"], df["middleband"], df["lowerband"] = upperband, middleband, lowerband
    
    # SAR
    SAR = talib.SAR(high, low, acceleration=0, maximum=0)
    df["sar"] = SAR
    
    # Support and resistance
    for i in [5,10,15,30]:
        df[f"support_{i}"] = df["close"].rolling(window=i).min()
        df[f"resistance_{i}"] = df["close"].rolling(window=i).max()

def get_momentum_feature(df):
    low, high, close = df["low"], df["high"], df["close"]
    
    # ADX
    adx = talib.ADX(high, low, close, timeperiod=14)
    df["adx"] = adx
    
    # AROON
    aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
    df["aroondown"], df["aroonup"] = aroondown, aroonup
    
    # CCI
    cci = talib.CCI(high, low, close, timeperiod=14)
    df["cci"] = cci
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"], df["macdsignal"], df["macdhist"] = macd, macdsignal, macdhist
    
    # MOM
    mom = talib.MOM(close, timeperiod=10)
    df["mom"] = mom
    
    # ROC
    roc = talib.ROC(close, timeperiod=10)
    df["roc"] = roc
    
    # RSI
    rsi = talib.RSI(close, timeperiod=14)
    df["rsi"] = rsi
    
    # KD
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df["slowk"], df["slowd"] = slowk, slowd
    
    # WILLR
    willr = talib.WILLR(high, low, close, timeperiod=14)
    df["willr"] = willr

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

def get_change(df, base="open"):
    gap = FUTURE
    base_change = df.loc[gap:,base].values - df.loc[:(df.shape[0]-1-gap),base].values
    base_change_rate = base_change / df.loc[:(df.shape[0]-1-gap),base].values*100
    return np.hstack([[0]*gap, base_change_rate])


def get_feature_df(df):
    df = deepcopy(df)
    
    close_change = get_change(df, base="close")
    close_open_rate = (df["close"] - df["open"]) / df["open"] * 100
        
    assert TASK in ("clf","reg")
    # calculate up/down
    if TASK=="clf":
        labels = np.hstack([close_change[1:], [0]*1])
        df["label"] = np.array(list(map(lambda x:1 if x>0 else 0, labels)))
    if TASK=="reg":
        df["label"] = np.hstack([close_open_rate[1:], [0]])
    
    # calculate factor
    df = pd.concat([get_ma_feature(df), get_add_feature(df),
                    get_momentum_feature(df),
                    get_volume_feature(df), get_volatility_feature(df)
                   ])
    
    # drop anomaly value
    df = df.dropna().reset_index(drop=True)
    df = df.drop(0, axis=0).reset_index(drop=True)
    df = df.drop([len(df)-1], axis=0).reset_index(drop=True)
    return df

def get_feature_label(df):
    feature_df = df.drop(["date","label"],axis=1)
    # feature_df = (feature_df-feature_df.min())/(feature_df.max()-feature_df.min())
    # feature_df = (feature_df-feature_df.mean())/(feature_df.std())
    features = feature_df.values
    labels = df["label"].values
    return features, labels