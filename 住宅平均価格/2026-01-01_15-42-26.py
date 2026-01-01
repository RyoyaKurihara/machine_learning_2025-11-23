
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Boston.csv')
df['CRIME'].value_counts()

'''
Out[3]: 
very_low    50
high        25
low         25
Name: CRIME, dtype: int64
'''




