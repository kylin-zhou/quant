import pandas as pd
import numpy as np

import sys
sys.version

if0_df = pd.read_csv("D:/quant/data/IF0.cffex.csv") # 沪深300指数期货连续

df = if0_df

# 相对前日涨跌幅
close_change = df.loc[1:,"close"].values - df.loc[:(df.shape[0]-2),"close"].values
close_change = np.hstack([[0], close_change])
df["label"] = (df["close"] - df["open"]).apply(lambda x:1 if x>0 else 0)
df["close_change"] = (close_change / df.loc[:,"close"].values) * 100

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


train_df = df[df["datetime"] < '2020-07-01']
train_data = train_df.drop(["datetime","label"], axis=1).values, train_df["label"].values

valid_df = df[(df["datetime"] > "2020-07-01") & (df["datetime"] < "2021-01-01")]
valid_data = valid_df.drop(["datetime","label"], axis=1).values, valid_df["label"].values

test_df = df[df["datetime"] > "2022-01-01"]
test_data = test_df.drop(["datetime","label"], axis=1).values, test_df["label"].values

print(train_df.tail(),valid_df.tail(), test_df.tail())

sys.path.append("D:/quant")
from models import LSTMModel, LSTM

lstm = LSTM(train_data[0].shape[1], log_path="./")
lstm.fit(train_data,
        valid_data,
        save_path="D:/quant/checkpoint/model.pt")

preds = lstm.predict(test_data[0])
print(preds)