
# データの分割
# 
# データを以下の３つに分ける。
# - 学習用
# - チューニングの参考にする検証用
# - チューニング後のテスト用
#
# 検証用とテスト用が同じデータだとテスト用に都合の良いように調整するため、検証用とテスト用とでは別データを使う。


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Boston.csv')

crime = pd.get_dummies(df['CRIME'], drop_first = True, dtype=int)
df2 = pd.concat([df, crime], axis = 1)
df2 = df2.drop(['CRIME'], axis = 1)


train_val, test = train_test_split(df2,test_size = 0.2,
random_state = 0)

train_val

'''
Out[11]: 
      ZN  INDUS  CHAS    NOX     RM  ...  PTRATIO  LSTAT  PRICE  low  very_low
43  22.0   5.86     0  0.431  6.108  ...     19.1   9.16   24.3    1         0
62  22.0   5.86     0  0.431  6.957  ...     19.1   3.53   29.6    0         1
3    0.0  21.89     0  0.624  6.151  ...     21.2  18.46   17.8    1         0
71  80.0   3.64     0  0.392  6.108  ...     16.4   6.57   21.9    0         1
45  21.0   5.64     0  0.439  5.963  ...     16.8  13.45   19.7    0         1
..   ...    ...   ...    ...    ...  ...      ...    ...    ...  ...       ...
96   0.0  18.10     0  0.655  5.759  ...     20.2  14.13   19.9    0         0
67   0.0   5.19     0  0.515  6.310  ...     20.2   6.75   20.7    0         1
64  28.0  15.04     0  0.464  6.249  ...     18.2  10.59   20.6    0         1
47   0.0   2.89     0  0.445  6.625  ...     18.0   6.65   28.4    0         1
44   0.0   5.96     0  0.499  5.850  ...     19.2   8.77   21.0    0         1

[80 rows x 14 columns]
'''