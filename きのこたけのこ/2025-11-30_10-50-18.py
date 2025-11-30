
import pandas as pd
from sklearn import tree


df = pd.read_csv('./KvsT.csv')
# df

x = df[['身長', '体重', '年代']]
# x

t = df[['派閥']]
# t

model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(x, t)

model.score(x, t)
