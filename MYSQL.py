import pandas as pd
df=pd.read_csv('rules.csv')
print(df.head())
df1=df['antecedents']
print(df1.head())
df2=df['
