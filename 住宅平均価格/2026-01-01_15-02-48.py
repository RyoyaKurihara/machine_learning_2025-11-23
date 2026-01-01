
# ダミー変数の設定
#
# 犯罪数は文字列だから、ダミー変数を用いて整数に置き換える。
# highならばlow=0, very_low=0となるようにする。


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Boston.csv')

crime = pd.get_dummies(df['CRIME'], drop_first = True, dtype=int)
df2 = pd.concat([df, crime], axis = 1)
df2 = df2.drop(['CRIME'], axis = 1)
df2.head(10)

'''
Out[6]: 
     ZN  INDUS  CHAS    NOX     RM  ...  PTRATIO  LSTAT  PRICE  low  very_low
0   0.0  18.10     0  0.718  3.561  ...     20.2   7.12   27.5    0         0
1   0.0   8.14     0  0.538  5.950  ...     21.0  27.71   13.2    1         0
2  82.5   2.03     0  0.415  6.162  ...     14.7   7.43   24.1    0         1
3   0.0  21.89     0  0.624  6.151  ...     21.2  18.46   17.8    1         0
4   0.0  18.10     0  0.614  6.980  ...     20.2  11.66   29.8    0         0
5   0.0   6.20     0  0.507  6.086  ...     17.4  10.88   24.0    1         0
6  22.0   5.86     0  0.431  6.438  ...     19.1   3.59   24.8    0         1
7   0.0   4.39     0  0.442  6.014  ...     18.8  10.53   17.5    0         1
8   0.0   9.90     0  0.544  6.113  ...     18.4  12.73   21.0    1         0
9   0.0  18.10     0  0.583  5.905  ...     20.2  11.45   20.6    0         0

[10 rows x 14 columns]
'''
