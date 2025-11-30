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
# df

x = df[['身長', '体重', '年代']]
# x

t = df[['派閥']]
# t

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


