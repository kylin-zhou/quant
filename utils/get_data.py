import akshare as ak

symbol_exchange = {'V0': 'dce',
    'P0': 'dce',
    'B0': 'dce',
    'M0': 'dce',
    'I0': 'dce',
    'JD0': 'dce',
    'L0': 'dce',
    'PP0': 'dce',
    'FB0': 'dce',
    'BB0': 'dce',
    'Y0': 'dce',
    'C0': 'dce',
    'A0': 'dce',
    'J0': 'dce',
    'JM0': 'dce',
    'CS0': 'dce',
    'EG0': 'dce',
    'RR0': 'dce',
    'EB0': 'dce',
    'LH0': 'dce',
    'TA0': 'czce',
    'OI0': 'czce',
    'RS0': 'czce',
    'RM0': 'czce',
    'ZC0': 'czce',
    'JR0': 'czce',
    'SR0': 'czce',
    'CF0': 'czce',
    'RI0': 'czce',
    'MA0': 'czce',
    'FG0': 'czce',
    'LR0': 'czce',
    'SF0': 'czce',
    'SM0': 'czce',
    'CY0': 'czce',
    'AP0': 'czce',
    'CJ0': 'czce',
    'UR0': 'czce',
    'SA0': 'czce',
    'PF0': 'czce',
    'PK0': 'czce',
    'FU0': 'shfe',
    'SC0': 'ine',
    'AL0': 'shfe',
    'RU0': 'shfe',
    'ZN0': 'shfe',
    'CU0': 'shfe',
    'AU0': 'shfe',
    'RB0': 'shfe',
    'WR0': 'shfe',
    'PB0': 'shfe',
    'AG0': 'shfe',
    'BU0': 'shfe',
    'HC0': 'shfe',
    'SN0': 'shfe',
    'NI0': 'shfe',
    'SP0': 'shfe',
    'NR0': 'ine',
    'SS0': 'shfe',
    'LU0': 'ine',
    'BC0': 'ine',
    'IF0': 'cffex',
    'TF0': 'cffex',
    'IH0': 'cffex',
    'IC0': 'cffex',
    'TS0': 'cffex',
    'IM0': 'cffex'}

symbol = "SC0"
exchange = symbol_exchange[symbol]
start_date = "20100101"
end_date="20220801"

print("futures:", symbol, exchange)
futures_df = ak.futures_main_sina(symbol, start_date=start_date, end_date=end_date).iloc[:,:6]
print(futures_df.columns)
futures_df.columns = ['datetime','open','high','low','close','volume']

print(futures_df.head())

futures_df.to_csv(f"data/{symbol}.{exchange}.csv", index=False)