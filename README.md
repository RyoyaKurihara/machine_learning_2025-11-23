# machine_learning_2025-11-23

「スッキリわかる Pythonによる機械学習入」を参考にする。

# chapter 4 "きのこ派とたけのこ派に分類する"

modelに学習と予測、モデルの評価について(scikit-learn)

```python
import pandas as pd
from sklearn import tree


df = pd.read_csv('./KvsT.csv')

x = df[['身長', '体重', '年代']]

t = df[['派閥']]

model = tree.DecisionTreeClassifier(random_state = 0) # モデル
model.fit(x, t) # 学習

```

```python
import pandas as pd
from sklearn import tree


df = pd.read_csv('./KvsT.csv')

x = df[['身長', '体重', '年代']]

t = df[['派閥']]

model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(x, t)

taro = [[170, 70, 20]]
taro_df = pd.DataFrame(taro, columns=x.columns)
model.predict(taro_df) # 予測
```

```shell
Out[12]: array(['きのこ', 'たけのこ'], dtype=object)
```

```python

import pandas as pd
from sklearn import tree


df = pd.read_csv('./KvsT.csv')

x = df[['身長', '体重', '年代']]

t = df[['派閥']]

model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(x, t)

model.score(x, t) # 評価

```

```shell
Out[13]: 1.0
```

# chapter 5 "あやめの分類"

欠損値のあるデータでのモデルの学習と、データを学習用データとテスト用データにわけることについて。

## 欠損あるデータ行を削除する

```python
import pandas as pd
from sklearn import tree

df = pd.read_csv('./iris.csv')
df2 = df.dropna(how = 'any', axis = 0) # nullのある行を削除

x = df2[['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']]
t = df2['種類']

model = tree.DecisionTreeClassifier(max_depth = 2, random_state=0)
model.fit(x, t)
model.score(x, t)

```

```shell
Out[33]: 0.951048951048951
```

## 欠損部分に平均値をいれる

中央値などの値を入れる場合もある。

```python
import pandas as pd
from sklearn import tree

df = pd.read_csv('./iris.csv')
column_mean = df.mean(numeric_only=True)

# print(column_mean)
df2 = df.fillna(column_mean) # null部分に平均値を代入

x = df2[['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']]
t = df2['種類']

model = tree.DecisionTreeClassifier(max_depth = 2, random_state=0)
model.fit(x, t)
model.score(x, t)

```

```shell
Out[30]: 0.94
```

## 学習用データとテスト用データの分割

データを分割することで、学習用の結果をより正確にスコアにできる

```python
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('./iris.csv')
df2 = df.dropna(how = 'any', axis = 0)

x = df2[['がく片長さ', 'がく片幅', '花弁長さ', '花弁幅']]
t = df2['種類']

x_train, x_test, y_train, y_test = train_test_split(x, t, test_size = 0.3, random_state = 0)

# print(x_train.shape)
# print(x_test.shape) 

model = tree.DecisionTreeClassifier(max_depth = 2, random_state=0)
model.fit(x_train, y_train) # 学習用データを使う
model.score(x_test, y_test) # テスト用データを使う

```

```shell
Out[5]: 0.9302325581395349
```

