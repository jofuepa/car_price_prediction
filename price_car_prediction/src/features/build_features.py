import pandas as pd

df = pd.read_csv('../../data/data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


df.to_csv('../../data/data_processed.csv', index=False)