
# 欠損値の処理
# 
# 欠損値を平均値で穴埋め


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

train_val2.isnull().sum()

'''
Out[13]: 
ZN          0
INDUS       0
CHAS        0
NOX         0
RM          0
AGE         0
DIS         0
RAD         0
TAX         0
PTRATIO     0
LSTAT       0
PRICE       0
low         0
very_low    0
dtype: int64
'''

