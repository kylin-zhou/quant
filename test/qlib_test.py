import qlib
qlib.init(provider_uri='data/cn_data')

from qlib.data import D
instruments = ['SH600000']
fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
df = D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day')
print(df.head())
