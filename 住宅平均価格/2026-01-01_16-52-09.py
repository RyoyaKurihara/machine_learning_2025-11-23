
# 相関係数を調べる。
#
# 各列同士の相関係数を出力して、モデルに対する影響度合いを調べる。
# 特徴量と正解データの相関係数が大きいほど、モデルに大きな影響を与える。
#
# 相関行列: 各列同士の相関係数を一覧化したもの


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

train_val4.corr()

'''
Out[20]: 
            INDUS       NOX        RM   PTRATIO     LSTAT     PRICE
INDUS    1.000000  0.785722 -0.403129  0.249438  0.578406 -0.470889
NOX      0.785722  1.000000 -0.272996  0.077533  0.484295 -0.325289
RM      -0.403129 -0.272996  1.000000 -0.404568 -0.560454  0.753771
PTRATIO  0.249438  0.077533 -0.404568  1.000000  0.326563 -0.542449
LSTAT    0.578406  0.484295 -0.560454  0.326563  1.000000 -0.693490
PRICE   -0.470889 -0.325289  0.753771 -0.542449 -0.693490  1.000000
'''
