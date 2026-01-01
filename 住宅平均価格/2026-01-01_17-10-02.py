
# データの標準化
#
# 各特徴量の平均とばらつきを統一させること。
# 
# 各特徴量の平均値やばらつきが大きく異なると特徴量の比較しずらいため、
# データの標準化をすることで適切な比較ができる。また、標準化したデータで学習したほうが予測性能があがることもある。



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

col =['RM', 'LSTAT', 'PTRATIO']
x = train_val4[col]
t = train_val4[['PRICE']]

x_train, x_val, y_train, y_val = train_test_split(x, t, test_size = 0.2, random_state = 0)

sc_model_x = StandardScaler()
sc_model_x.fit(x_train)

sc_x = sc_model_x.transform(x_train)

tmp_df = pd.DataFrame(sc_x, columns = x_train.columns)

tmp_df.mean()
tmp_df.std()

'''
Out[36]: 
RM        -3.559763e-16
LSTAT      1.727014e-16
PTRATIO   -1.436241e-16
dtype: float64

Out[37]: 
RM         1.008032
LSTAT      1.008032
PTRATIO    1.008032
dtype: float64
'''