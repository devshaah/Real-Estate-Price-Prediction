import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Bengaluru_House_Data.csv')
df.drop(['area_type','society','balcony','availability'],axis='columns',inplace=True)
# print(df.isnull().sum())

df.dropna(inplace=True)
# print(df.isnull().sum())

# print(df['size'].unique())
#some value were 2bhk some were 3 bedroom so bringing consistency
df['bhk'] = df['size'].apply(lambda x : int(x.split(' ')[0]) )     
df.drop(['size'],axis='columns',inplace=True)
print(df)
