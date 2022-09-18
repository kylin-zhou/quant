import pandas as pd
import numpy as np
from copy import deepcopy
import os
import sys
import glob
import pathlib
import datetime as dt


dfs = {}
symbols = []
for i in glob.glob("data/futures/*"):
    symbol = pathlib.Path(i).name[:-4]
    # if symbol == "AU0.shfe":
    dfs[symbol] = pd.read_csv(i)
    symbols.append(symbol)
    break

print(symbols)

def get_feature_df(df):
    # 相对前日涨跌幅
    close_change = df.loc[1:,"close"].values - df.loc[:(df.shape[0]-2),"close"].values
    close_change = np.hstack([[0], close_change])
    df["label"] = (df["close"] - df["open"]).apply(lambda x:1 if x>0 else 0)
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
    df = df[df["volume_change"] != 0]

    # 当日：开盘 收盘 最高 最低 差值 波动, 以开盘价和收盘价为基准
    df["high_open_rate"] = (df["high"] - df["open"]) / df["open"]
    df["low_open_rate"] = (df["low"] - df["open"]) / df["open"]
    df["close_open_rate"] = (df["close"] - df["open"]) / df["open"]
    df["high_close_rate"] = (df["high"] - df["close"]) / df["close"]
    df["low_close_rate"] = (df["low"] - df["close"]) / df["close"]

    df = df.drop(["open","high","low","close","volume"], axis=1)
    df = df.dropna()
    return df

def split_sequences(df, window=90, step=1):
    X,y = [],[]
    for i in range(0, df.shape[0]-window-1, step):
        X.append(df.drop("label",axis=1).iloc[i:window+i,:].values)
        y.append(df.loc[i+window,"label"])

    return np.array(X), np.array(y)
    

from functools import reduce
def vstack_array(arrays):
    return reduce(lambda x, y: np.vstack([x,y]), arrays)
def hstack_array(arrays):
    return reduce(lambda x, y: np.hstack([x,y]), arrays)
    
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = [],[],[],[],[],[]
for i,df in enumerate(dfs.values()):
    df = get_feature_df(df)
    train_df = df[df["datetime"] < '2021-07-01'].reset_index(drop=True)
    valid_df = df[(df["datetime"] > "2021-07-01") & (df["datetime"] < "2022-01-01")].reset_index(drop=True)
    test_df = df[df["datetime"] > "2022-01-01"].reset_index(drop=True)

    train_x, train_y = split_sequences(df=deepcopy(train_df.drop(["datetime"], axis=1)))
    valid_x, valid_y = split_sequences(df=deepcopy(valid_df.drop(["datetime"], axis=1)))
    test_x, test_y = split_sequences(df=deepcopy(test_df.drop(["datetime"], axis=1)))
    
    if len(train_y) > 0:
        train_X.append(train_x)
        train_Y.append(train_y)
    if len(valid_y) > 0:
        valid_X.append(valid_x)
        valid_Y.append(valid_y)
    if len(test_y) > 0:
        test_X.append(test_x)
        test_Y.append(test_y)
    

if len(train_Y) > 1:
    train_data, valid_data, test_data = (
                (vstack_array(train_X), hstack_array(train_Y)),
                (vstack_array(valid_X), hstack_array(valid_Y)),
                (vstack_array(test_X), hstack_array(test_Y)),
            )
else:
    train_data, valid_data, test_data = (
                (np.array(train_X), np.array(train_Y)),
                (np.array(valid_X), np.array(valid_Y)),
                (np.array(test_X), np.array(test_Y)),
            )
        
sys.path.append("D:/quant")
from models import LSTM, ALSTM, TCN

strtime = dt.datetime.now().strftime("%Y%m%d%H%M")
log_path = f"logs/{strtime}"

model = LSTM(
    d_feat=train_data[0].shape[2],
    hidden_size=16,
    batch_size=32,
    loss='bce',
    lr=0.001,
    log_path=log_path
)
model.fit(train_data,
        valid_data,
        save_path="D:/quant/checkpoint/w90_bce_model.pt")

preds = model.predict(test_data[0])
print(preds.shape)

import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

import matplotlib.pyplot as plt

plt.plot(test_data[1], label="true")
plt.plot(sigmoid(preds), label="pred")
plt.legend()
plt.show()