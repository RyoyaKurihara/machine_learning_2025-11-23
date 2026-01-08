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

前処理の手順
1. どんな予測モデルを作成するのか明確にするために、種類列のデータにどんな値があるか確認する。
2. 収集したデータに欠損値があるとデータ分析ができないため、欠損値があるかないか確認する。
	- 欠損値の行が少ない場合、欠損値の行を削除する。
	- 欠損値の場所に代表値（平均値、中央値、最頻値）を埋める。
3. 前処理した学習用のデータを特徴量と正解データに分割する。

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

# chapter 6 "映画の興行収入" 

前処理の手順
1. 収集したデータに欠損値があるとデータ分析ができないため、欠損値があるかないか確認する。
	- 欠損値の場所に代表値（平均値、中央値、最頻値）を埋める。
2. 全体の傾向からかけ離れたデータ（外れ値）があると作成したモデルの性能が上がりにくくなるため、外れ値が含まれているか確認する。
	- 散布図を使うと、ラクに見つけれれる。
	- あったら、削除する。
	- 決定木だと外れ値の影響が少ないけど、重回帰分析では影響が大きいため今回は実施する。（モデルによって違うため、使うモデルの性質をよく調べる。）
3. 前処理した学習用のデータを特徴量と正解データに分割する。
	- 映画IDのような列は興行収入に関係なさそうなので、特徴量にも含めない。
4. 特徴量と正解データを訓練データとテストデータに分割する。


# chapter 7 "客船沈没事故での生存予想"

不均衡データ: 正解データの数に偏りがあるデータ。"客船沈没事故での生存予想"において、死亡数が549、生存数が342と1.6倍の差がある。
大きく偏りがあると、モデルが「とりあえず多い方に入れる」という学習をしてしまう。

ピポットテーブルを使っいデータの欠損値を埋めるときに、データの分布に影響がないように穴埋めするために列の値ごとに穴埋めする値を分けること。

# chapter 8 "住宅平均価格"

データの分割を学習用、検証用、テスト用にわける。
検証用は学習用を使ったモデルに対して、チューニングに用いる。

データの標準化によって、各特徴量の平均とばらつきを統一させる。
データの標準化をすることで適切な比較ができる。また、標準化したデータで学習したほうが予測性能があがることもある。


