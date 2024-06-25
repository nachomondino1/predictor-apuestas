import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import datetime
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_matches_result
from p3_data_preparation.construct_data import determine_result


# id_country = 167
# id_competicion = 1671
# df_comp = pd.read_excel('p2_data_understanding/data/df_competencies.xlsx', index_col=0)  # Para garantizar que tengo todas las predicciones
# country = 
# competition = 

country = 'usa'
competition = 'mls'

# Levantar predicciones.xlsx
df = pd.read_excel('p6_deployment/data/historial_predicciones.xlsx', index_col=0)  # Para garantizar que tengo todas las predicciones
print(df.shape)

# Seleccionar los partidos de ayer
fecha_hoy = datetime.datetime.now()
n_days_to_extract = 7  # 7 por pruebas sino es 1
fecha_limite = fecha_hoy - datetime.timedelta(days=n_days_to_extract)  # ATENCION! n_dias_ult_part desde el partido missing mas viejo
print(f"Fechas a filtrar: {fecha_limite} --> {fecha_hoy}")

df_filt = df[(df['date'] > fecha_limite) & (df['date'] <= fecha_hoy)]
print(df_filt.shape)

l_ids = df_filt.index
print(f"Lista de ids a los que extraer resultado: {l_ids}")


# Por competicion
# Extraer el resultado y goles de dichos partidos
df_results = extract_matches_result(country, competition, l_ids)
# df_results = pd.read_excel('p6_deployment/data/results.xlsx', index_col=0)
print(df_results)

df_results = determine_result(df_results, var_resp='result')

# Agrego columnas 'result' y 'score' a predicciones.xlsx
df_concat = pd.concat([df_filt, df_results], axis=1)
print(df_concat)

# Exportar dataset results.xlsx
df_concat.to_excel('p6_deployment/data/predicciones.xlsx')

