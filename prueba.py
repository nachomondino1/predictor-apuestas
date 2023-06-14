import pandas as pd

# Demostracion de por que falla construct data en la construccion de variables historicas...
# No calcula la cuenta cuando hay nan...
n_ult_part = 5
l = [1, 2, 3, None, 5]

print(sum(l) / n_ult_part)

'''
pais = 'argentina'
l_paises = ['argentina', 'inglaterra']

df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/df_competencias.xlsx')
print(df_comp)

df = df_comp[df_comp['pais'].isin(l_paises)]

for competicion, categoria in zip(df['nombre'], df['categoria']):
    print(competicion, categoria)
'''

''' # Probando integracion de datos con +1 fecha de act por fifa
from data_preparation import format_data, clean_data, integrate_data

df_part = pd.read_excel("/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_part_cleaned.xlsx")
df_jug = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/enticed_jugadores.xlsx', index_col=0)
print(df_jug.head(1))

# Entidad jugador: fecha y valor de mercado
df_jug = format_data.convert_fecha_to_datetime(df_jug, string_format='%b %d, %Y')
df_jug = format_data.convert_valor_mercado_to_int(df_jug)

df_jug = clean_data.prepare_text_columns(df_jug, l_col_to_except=['id', 'fifa'])  # df_jug = clean_data.prepare_text_columns(df_jug)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

# Integro datasets
df_integrated = integrate_data.player_data_in_match(df_part, df_jug)
df_integrated.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=False)
'''


''' 
# Fecha
import datetime

fecha_str = "AUG 30, 2008"
fecha_dt = datetime.datetime.strptime(fecha_str, '%b %d, %Y')
print(fecha_dt)
print(fecha_dt.month)
'''


''' concatenar df_seg...
import pandas as pd

# Levanto datasets a concatenar
df_1 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/inglaterra/entidad_partido.xlsx')
df_2 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/data_seg/premier-league_2018_2019_inglaterra.xlsx')
df_3 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/data_seg/premier-league_2006_2007_inglaterra.xlsx')

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

df_concat.to_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/data_seg/entidad_partido_inglaterra.xlsx', index=False)
'''