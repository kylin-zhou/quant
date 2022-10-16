import os
import time

import schedule
import akshare as ak
import pandas as pd

from logger import get_logger

def job():
    symbols = {'MA2301','TA2301',"RM2301"}

    logger.info("data working...")
    for symbol in symbols:
        logger.info(symbol)
        data = ak.futures_zh_minute_sina(symbol=symbol, period="5")

        dir = f"data/futures/{symbol[:-4]}"
        file = os.path.join(dir, f"{symbol}.csv")
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif os.path.exists(file):
            history_data = pd.read_csv(file)
            data = pd.concat([history_data, data], axis=0).reset_index(drop=True)
            data = data.drop_duplicates()
            data.to_csv(file, index=False)
            logger.info("{}".format(data.shape))
        else:
            logger.info("{}".format(data.shape))
            data.to_csv(file, index=False)

def alive_job():
    logger.info("I'm working...")

""" example
schedule.every(10).seconds.do(job)
schedule.every(10).minutes.do(job)               # 每隔 10 分钟运行一次 job 函数
schedule.every().hour.do(job)                    # 每隔 1 小时运行一次 job 函数
schedule.every().day.at("10:30").do(job)         # 每天在 10:30 时间点运行 job 函数
schedule.every().monday.do(job)                  # 每周一 运行一次 job 函数
schedule.every().wednesday.at("13:15").do(job)   # 每周三 13：15 时间点运行 job 函数
schedule.every().minute.at(":17").do(job)        # 每分钟的 17 秒时间点运行 job 函数
"""

logger = get_logger("logs/data_log.txt")

logger.info("start task data...")

schedule.every(21).days.do(job)         # 每隔 21 天运行一次
schedule.every(1).days.do(alive_job)

while True:
    schedule.run_pending()   # 运行所有可以运行的任务
    time.sleep(3600)