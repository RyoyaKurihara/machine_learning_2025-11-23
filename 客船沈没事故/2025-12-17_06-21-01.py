
# 欠損値を埋める
# null部分に値をセットする。。
# `cabin`は使わないからスルーする

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('./Survived.csv')

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()


# Out[3]: 
# PassengerId      0
# Survived         0
# Pclass           0
# Sex              0
# Age              0
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         0
# dtype: int64
