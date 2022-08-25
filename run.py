import pandas as pd
import numpy as np

import sys
sys.version

if0_df = pd.read_csv("D:/quant/data/IF0.cffex.csv") # 沪深300指数期货连续

df = if0_df

# 相对前日涨跌幅
close_change = df.loc[1:,"close"].values - df.loc[:(df.shape[0]-2),"close"].values
close_change = np.hstack([[0], close_change])
df["close_change"] = (close_change / df.loc[:,"close"].values) * 100

df["label"] = df["close_change"]# (df["close"] - df["open"]).apply(lambda x:1 if x>0 else 0)

# volume 波动
# volume_change = df.loc[1:,"volume"].values - df.loc[:(df.shape[0]-2),"volume"].values
# volume_change = np.hstack([[0], volume_change])
# df["volume_change"] = (volume_change / df.loc[:,"volume"].values) * 100

def get_change(df, base="open"):
    base_change = df.loc[1:,base].values - df.loc[:(df.shape[0]-2),base].values
    base_change = np.hstack([[0], base_change])
    return (base_change / df.loc[:,base].values) * 100
    
for col in ["open","high","low","volume"]:
    df[col+"_change"] = get_change(df, base=col)

# 当日：开盘 收盘 最高 最低 差值 波动, 以开盘价和收盘价为基准
df["high_rate"] = (df["high"] - df["open"]) / df["open"]
df["low_rate"] = (df["low"] - df["open"]) / df["open"]
df["close_rate"] = (df["close"] - df["open"]) / df["open"]
df["high_rate2"] = (df["high"] - df["close"]) / df["close"]
df["low_rate2"] = (df["low"] - df["close"]) / df["close"]

df = df.drop(["open","high","low","close","volume"], axis=1)
df.head()


def split_sequences(df, window=60, step=1):
    X,y = [],[]
    for i in range(0, df.shape[0]-window-1, step):
        X.append(df.iloc[i:window+i,:].values)
        y.append(df.loc[i+window,"label"])

    return np.array(X), np.array(y)

from copy import deepcopy

train_df = df[df["datetime"] < '2021-07-01'].reset_index(drop=True)
valid_df = df[(df["datetime"] > "2021-07-01") & (df["datetime"] < "2022-01-01")].reset_index(drop=True)
test_df = df[df["datetime"] > "2022-01-01"].reset_index(drop=True)

print(train_df.shape,valid_df.shape, test_df.shape)

train_data = split_sequences(df=deepcopy(train_df.drop(["datetime"], axis=1)))
valid_data = split_sequences(df=deepcopy(valid_df.drop(["datetime"], axis=1)))
test_data = split_sequences(df=deepcopy(test_df.drop(["datetime"], axis=1)))


sys.path.append("D:/quant")
from models import LSTM, ALSTM, TCN

model = LSTM(d_feat=train_data[0].shape[2], log_path="./")
model.fit(train_data,
        valid_data,
        save_path="D:/quant/checkpoint/model.pt")

preds = model.predict(test_data[0])
print(preds.shape)

import matplotlib.pyplot as plt

plt.plot(range(len(preds)), test_data[1], label="true")
plt.plot(range(len(preds)), preds, label="pred")
plt.legend()
plt.show()