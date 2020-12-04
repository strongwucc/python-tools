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
df = pd.read_csv('./shanghai_1990-12-19_to_2019-2-28.csv')

df.Timestamp = pd.to_datetime(df.Timestamp)
df.index = df.Timestamp

f = df.resample('D').mean()
df_month = df.resample('M').mean()
df_year = df.resample('A-DEC').mean()
df_Q = df.resample('Q-DEC').mean()

plt.rcParams['font.sans-serif'] = ['SimHei']
# print(df_month)
# fig = plt.figure(figsize=[15, 7])
# plt.suptitle('沪市走势', fontsize=22)

# plt.subplot(221)
# plt.plot(df.Price, '-', label='按天')
# plt.legend()

# plt.subplot(222)
# plt.plot(df_month.Price, '-', label='按月')
# plt.legend()

# plt.subplot(223)
# plt.plot(df_Q.Price, '-', label='按季度')
# plt.legend()

# plt.subplot(224)
# plt.plot(df_year.Price, '-', label='按年')
# plt.legend()

# plt.show()

df_month['Price_box'], lmbda = stats.boxcox(df_month.Price)

Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D = 1
d = 1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model = sm.tsa.statespace.SARIMAX(df_month.Price_box, order=(param[0], d, param[1]),
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


def invboxcox(y, lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))


df_month2 = df_month[['Price']]
date_list = [datetime(2019, 3, 31), datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30), datetime(2019, 7, 31), datetime(2019, 8, 31), datetime(2019, 9, 30), datetime(2019, 10, 31), datetime(2019, 11, 30), datetime(2019, 12, 31), datetime(2020, 1, 31), datetime(2020, 2, 29), datetime(2020, 3, 31), datetime(2020, 4, 30), datetime(2020, 5, 31), datetime(2020, 6, 30), datetime(2020, 7, 31), datetime(2020, 8, 31), datetime(2020, 9, 30), datetime(2020, 10, 31), datetime(2020, 11, 30), datetime(2020, 12, 31), datetime(2021, 1, 31), datetime(2021, 2, 28), datetime(2021, 3, 31), datetime(2021, 4, 30), datetime(2021, 5, 31), datetime(2021, 6, 30)]
future = pd.DataFrame(index=date_list, columns=df_month.columns)
df_month2 = pd.concat([df_month2, future])
# print(df_month2.shape)
df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=367), lmbda)
plt.figure(figsize=(15, 7))
df_month2.Price.plot(label='实际走势')
df_month2.forecast.plot(color='r', ls='--', label='预测走势')
plt.legend()
plt.title('沪市价格走势（按月）')
plt.xlabel('时间')
plt.ylabel('价格')
plt.show()
