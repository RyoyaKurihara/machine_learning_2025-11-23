
# 外れ値の処理
# 
# グラフで確認
# 重回帰分析の場合は1つの外れ値が大きな影響を与えるため、取り除いておく。


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

colname = train_val2.columns
for name in colname:
    train_val2.plot(kind = 'scatter', x = name, y = 'PRICE')

