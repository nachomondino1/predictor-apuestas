import pandas as pd
from data_preparation import format_data

concatenar = False

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



'''
# DataFrame original
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=[10, 20, 30])

# DataFrame a unir
df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]}, index=[20, 30, 40])

# Unir los dataframes por el índice correspondiente
df_merged = df1.merge(df2, left_index=True, right_index=True)

print(df_merged)
'''

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


# concatenar df_seg...
if concatenar:
    # Levanto datasets a concatenar
    df_1 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/argentina/entidad_partido.xlsx')
    df_2 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/sudamerica/entidad_partido.xlsx')
    # df_3 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/inglaterra/premier-league_1989_1990_inglaterra.xlsx')

    df_1['goles_loc'] = df_1['goles_loc'].astype(int)
    df_2['goles_loc'] = df_2['goles_loc'].astype(int)

    # df_1 = format_data.convert_fecha_to_datetime(df_1,string_format='%d.%m.%Y %H:%M')  # ya lo voy a extraer datetime... # Fundamental para poder ordenar el df por 'fecha'

    # Imprimo caracteristicas de cada dataframe a concatenar
    print(df_1.head(1))
    print(df_1.shape)

    print(df_2.head(1))
    print(df_2.shape)

    # print(df_3.head(1))
    # print(df_3.shape)

    # Concateno dataframes
    # df_concat = pd.concat([df_1, df_2, df_3], axis=0)
    df_concat = pd.concat([df_1, df_2], axis=0)

    # # Antes del cambio de tipo
    # print(df_concat.dtypes)
    #
    # # Cambiar el tipo de datos de la columna "goles_loc" a int
    # df_concat['goles_loc'] = df_concat['goles_loc'].astype(int)
    #
    # # Después del cambio de tipo
    # print(df_concat.dtypes)

    print(df_concat.head(1))
    print(df_concat.shape)

    df_concat.to_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/argentina_sudamerica/entidad_partido.xlsx', index=False)
