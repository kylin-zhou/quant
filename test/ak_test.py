import akshare as ak

print("a stock")
stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20210301", end_date='20210907', adjust="hfq")
print(stock_zh_a_hist_df)

print("futures")
futures_df = ak.futures_main_sina("IF0", start_date="20170101", end_date="20220801").iloc[:,:6]
print(futures_df.columns)
futures_df.columns = ['datetime','open','high','low','close','volume']

print(futures_df.head())

futures_df.to_csv("data/IF0.csv", index=False)