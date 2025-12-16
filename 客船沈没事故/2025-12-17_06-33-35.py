
# 特徴量と、正解データに分割する。
# さらに、学習用データとテスト用データに分割する。

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('./Survived.csv')

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df['Embarked'].fillna(df['Embarked'].mode()[0])

x = df[['Pclass','Age','SibSp','Parch','Fare']]
t = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,t,
test_size = 0.2,random_state = 0)

x_train.shape

'''
Out[5]: (712, 5)
'''