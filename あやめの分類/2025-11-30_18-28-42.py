
import pandas as pd

df = pd.read_csv('./iris.csv')
df.isnull().any(axis = 0)


