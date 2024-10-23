import pandas_ta as ta
import pandas as pd

def sma(data, window=14):
    data = pd.Series(data)
    return ta.sma(data, window)