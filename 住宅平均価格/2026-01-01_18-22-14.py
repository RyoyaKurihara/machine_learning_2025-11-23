
# データの標準化
#
# 各特徴量の平均とばらつきを統一させること。
# 
# 各特徴量の平均値やばらつきが大きく異なると特徴量の比較しずらいため、
# データの標準化をすることで適切な比較ができる。また、標準化したデータで学習したほうが予測性能があがることもある。
#
# 正解データも標準化しておく。


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

sc_model_y = StandardScaler()
sc_model_y.fit(y_train)

sc_y = sc_model_y.transform(y_train)

sc_y

'''
Out[39]: 
array([[-0.05270192],
       [-0.44023557],
       [-1.82278806],
       [ 2.2096567 ],
       [-0.33549674],
       [ 0.12535409],
       [ 2.89045906],
       [-0.14696686],
       [-0.52402663],
       [ 0.01014138],
       [-0.58686992],
       [-0.10507133],
       [ 0.25104068],
       [ 2.73335082],
       [-0.32502286],
       [-0.28312733],
       [-0.71255651],
       [-0.97440357],
       [ 0.16724962],
       [-0.20981015],
       [-1.0162991 ],
       [ 0.7747348 ],
       [ 0.08345856],
       [ 0.5338355 ],
       [-1.2571984 ],
       [ 0.75378703],
       [ 0.75378703],
       [-1.05819463],
       [-0.36691839],
       [ 0.20914515],
       [-0.4821311 ],
       [-0.96392969],
       [-0.29360121],
       [-0.02128027],
       [ 2.89045906],
       [-0.34597063],
       [ 0.19867126],
       [ 0.62810044],
       [-1.34098946],
       [ 2.24107835],
       [ 0.13582797],
       [-1.0791424 ],
       [-0.60781769],
       [ 0.25104068],
       [-0.18886239],
       [ 0.25104068],
       [-0.29360121],
       [ 0.96326468],
       [-0.86966475],
       [ 0.06251079],
       [ 0.24056679],
       [-0.55544828],
       [-1.10009016],
       [-0.08412356],
       [-0.3040751 ],
       [ 1.82212305],
       [-0.95345581],
       [-0.38786616],
       [-0.17838851],
       [-1.10009016],
       [-0.9010864 ],
       [-0.19933627],
       [ 1.28795504]])
'''