import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')

# 数据加载
df = pd.read_csv('./bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv')
# print(df.head())

df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
df.index = df.Timestamp
df = df.resample('D').mean()
df_month = df.resample('M').mean()
df_year = df.resample('A-DEC').mean()
df_Q = df.resample('Q-DEC').mean()

plt.rcParams['font.sans-serif'] = ['SimHei']

# fig = plt.figure(figsize=[15, 7])
# plt.suptitle('比特币金额走势（美金）', fontsize=22)

# plt.subplot(221)
# plt.plot(df.Weighted_Price, '-', label='按天')
# plt.legend()

# plt.subplot(222)
# plt.plot(df_month.Weighted_Price, '-', label='按月')
# plt.legend()

# plt.subplot(223)
# plt.plot(df_Q.Weighted_Price, '-', label='按季度')
# plt.legend()

# plt.subplot(224)
# plt.plot(df_year.Weighted_Price, '-', label='按年')
# plt.legend()

# plt.show()

# plt.figure(figsize=[15, 7])
# sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()
# print("Dickey–Fuller test: p=%f" %
#       sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
# # plt.show()

df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)
# print("Dickey–Fuller test: p=%f" %
#       sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])

# df_month['prices_box_diff'] = df_month.Weighted_Price_box - \
#     df_month.Weighted_Price_box.shift(12)
# print("Dickey–Fuller test: p=%f" %
#       sm.tsa.stattools.adfuller(df_month.prices_box_diff[12:])[1])

# df_month['prices_box_diff2'] = df_month.prices_box_diff - \
#     df_month.prices_box_diff.shift(1)
# plt.figure(figsize=(15, 7))

# sm.tsa.seasonal_decompose(df_month.prices_box_diff2[13:]).plot()
# print("Dickey–Fuller test: p=%f" %
#       sm.tsa.stattools.adfuller(df_month.prices_box_diff2[13:])[1])

# plt.show()

Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D = 1
d = 1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model = sm.tsa.statespace.SARIMAX(df_month.Weighted_Price_box, order=(param[0], d, param[1]),
                                          seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
# print(result_table.sort_values(by='aic', ascending=True).head())
# print(best_model.summary())


def invboxcox(y, lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))


df_month2 = df_month[['Weighted_Price']]
date_list = [datetime(2020, 10, 31), datetime(2020, 11, 30), datetime(2020, 12, 31), datetime(2021, 1, 31), datetime(2021, 2, 28), datetime(
    2021, 3, 31), datetime(2021, 4, 30), datetime(2021, 5, 31), datetime(2021, 6, 30)]
future = pd.DataFrame(index=date_list, columns=df_month.columns)
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=115), lmbda)
plt.figure(figsize=(15, 7))
df_month2.Weighted_Price.plot(label='实际金额')
df_month2.forecast.plot(color='r', ls='--', label='预测金额')
plt.legend()
plt.title('比特币价格趋势（按月）')
plt.xlabel('时间')
plt.ylabel('美金')
plt.show()
