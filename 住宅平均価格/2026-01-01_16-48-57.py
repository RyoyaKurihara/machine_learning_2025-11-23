
# 外れ値の処理
# 
# 特徴量に使う列の外れ値を除去する。
# また、テストデータに外れ値があるとモデルの正解率がさがってしまうため、ある程度残しておく。
#
# 外れ値の行（インデックスを指定）を削除する。


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Boston.csv')

crime = pd.get_dummies(df['CRIME'], drop_first = True, dtype=int)
df2 = pd.concat([df, crime], axis = 1)
df2 = df2.drop(['CRIME'], axis = 1)


train_val, test = train_test_split(df2,test_size = 0.2,
random_state = 0)

train_val_mean = train_val.mean() 
train_val2=train_val.fillna(train_val_mean) 

train_val3 = train_val2.drop([76], axis = 0)

col = ['INDUS', 'NOX', 'RM', 'PTRATIO', 'LSTAT', 'PRICE']

train_val4 = train_val3[col]
train_val4.head(3)

'''
Out[18]: 
    INDUS    NOX     RM  PTRATIO  LSTAT  PRICE
43   5.86  0.431  6.108     19.1   9.16   24.3
62   5.86  0.431  6.957     19.1   3.53   29.6
3   21.89  0.624  6.151     21.2  18.46   17.8
'''
