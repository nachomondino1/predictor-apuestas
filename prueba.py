import pandas as pd
from p3_data_preparation import format_data


import pandas as pd
import numpy as np

# Supongamos que tienes un DataFrame llamado df
df = pd.DataFrame({'goles_loc': [1.0, 'A', 2.0,  4.0],
                   'goles_vis': [5.0, 6.0, 7.0, 'B']})

l_filas_a_borrar = []

for i, row in df.iterrows():

    try:
        int(row['goles_loc'])
        int(row['goles_vis'])

    except:
        l_filas_a_borrar.append(i)

print(f"Cantidad de partidos eliminados por no tener goles integer: {len(l_filas_a_borrar)/len(df)*100:.0f}%")
df = df.drop(l_filas_a_borrar)

# Mostrar el DataFrame resultante
print(df)

'''
df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx')
print(df_comp)

# print(df_comp.competicion.values[2:])
for competicion, is_cup in zip(df_comp['competicion'].values[2:], df_comp['is_cup'].values[2:]):
    print(competicion, is_cup)
'''

"""
from fuzzywuzzy import fuzz

n1 = "chilwell benjamin"
n2 = "Ben Chilwell"
nombre1 = "Benjamin White"
nombre2 = "white ben"
umbral = 85

print(fuzz.token_set_ratio(n1, n2) >= umbral)
"""


"""
import pandas as pd

# Supongamos que tienes un DataFrame llamado df
# y deseas reemplazar todos los valores iguales a 1 por 5

# Crear un DataFrame de ejemplo
data = {'Columna1': [1, 2, 1, 4, 5],
        'Columna2': [1, 6, 7, 8, 9],
        'Columna3': [10, 1, 12, 13, 14]}

df = pd.DataFrame(data)

# Reemplazar todos los valores iguales a 1 por 5
df_reemplazado = df.replace(1, 5)
df_reemplazado = df_reemplazado.replace(10, 100)

# Mostrar el DataFrame resultante
print("DataFrame original:")
print(df)
print("\nDataFrame con reemplazo:")
print(df_reemplazado)
"""


"""
import itertools

l_ids = [1, 2, 3]
actual_temp = 2023
output_list = list(itertools.chain.from_iterable([[actual_temp]] * len(l_ids)))
print(output_list)

def extract_id_from_url(url):

    pos_ini = url.find('/Matches/') + len('/Matches/')
    pos_fin = url.find('/Live/')
    id = url[pos_ini: pos_fin]
    return id

# url = 'https://www.whoscored.com/Matches/1703386/Live/Argentina-Liga-Profesional-2023-Sarmiento-Newell-s-Old-Boys'
# id = extract_id_from_url(url)
# print(id)
"""


"""
# DataFrame original
df = pd.DataFrame({'col1': [1, 3, 3], 'col2': [4, 5, 6], 'col3': [1, 4, 5]}, index=[10, 20, 30])
print(df)

# Unir los dataframes por el índice correspondiente
# df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]}, index=[20, 30, 40])
# df_merged = df1.merge(df2, left_index=True, right_index=True)

df_uniques = df.drop_duplicates(subset=['col1'])

print(df_uniques)
"""


'''
import math

# Demostracion de por que falla construct data en la construccion de variables historicas...
# No calcula la cuenta cuando hay nan...
n_ult_part = 5
l = [1, 2, 3, None, 5]
l = [None, None, None, None, None]

l_sin_nan = list(filter(lambda x: x is not None and not math.isnan(x), l))  # l_sin_nan = list(filter(lambda x: not math.isnan(x), l))

print(l_sin_nan)

print(sum(l_sin_nan) / n_ult_part)
'''

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