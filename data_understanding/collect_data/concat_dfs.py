import pandas as pd

# Levanto datasets a concatenar
df_1 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/liga-profesional_2022_argentina.xlsx')
df_2 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/liga-profesional_2017_2018_argentina.xlsx')
df_3 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/liga-profesional_2003_2004_argentina.xlsx')

# Imprimo caracteristicas de cada dataframe a concatenar
print(df_1.head(1))
print(df_1.shape)

print(df_2.head(1))
print(df_2.shape)

print(df_3.head(1))
print(df_3.shape)

# Concateno dataframes
df_concat = pd.concat([df_1, df_2, df_3], axis=0)
print(df_concat.head(1))
print(df_concat.shape)

df_concat.to_excel('./data_understanding/collect_data/data_seg/entidad_partido_argentina.xlsx', index=False)