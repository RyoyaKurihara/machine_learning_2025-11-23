
# 外れ値の処理
# 
# 特徴量に使う列の外れ値を除去する。
# また、テストデータに外れ値があるとモデルの正解率がさがってしまうため、ある程度残しておく。
#
# RM: 部屋数
# PTRATIO: 教員一人当たりの生徒数


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

out_line1 = train_val2[(train_val2['RM'] < 6) & (train_val2['PRICE'] > 40)].index
out_line2 = train_val2[(train_val2['PTRATIO'] > 18) &
(train_val2['PRICE'] > 40)].index

print(out_line1, out_line2)

'''
Int64Index([76], dtype='int64') Int64Index([76], dtype='int64')
'''
