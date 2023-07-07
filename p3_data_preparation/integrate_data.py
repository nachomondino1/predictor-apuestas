import pandas as pd
import math
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz

# RELLENO DE DATOS DE WHOSCORED A PARTIR DE FLASHCORE
def fill_whoscored_with_flashscore(df_part_who, df_part_flash):

    # Defino columnas a rellenar
    cont_part_rell = 0
    l_col = ['posesion_loc', 'posesion_vis', 'remates_loc', 'remates_vis', 'remates_a_puerta_loc',
             'remates_a_puerta_vis', 'faltas_loc', 'faltas_vis', 'pases_loc', 'pases_vis', 'pases_comp_loc',
             'pases_comp_vis', 'offsides_loc', 'offsides_vis']

    # Selecciono partidos sin estadisticas
    df_part_who_filt = df_part_who[df_part_who['posesion_loc'].isnull()]  # Funciona bien
    # df_part_who_filt.to_excel('/Users/nachomondino/Desktop/df_part_who_a_rellenar.xlsx', index=False)

    # Saco horas de fechas para poder compararlas (hay diferencia porque Whoscored tiene otro huso horario)
    df_part_who_filt['fecha_sin_hora'] = df_part_who_filt['fecha'].dt.date
    df_part_flash['fecha_sin_hora'] = df_part_flash['fecha'].dt.date

    # Por partido sin estadisticas en Whoscored
    for idx, row in df_part_who_filt.iterrows():

        fecha_who = row['fecha_sin_hora']
        equipo_loc_who = row['equipo_loc']
        equipo_vis_who = row['equipo_vis']

        # Selecciono partidos en Flashscore con la misma fecha
        df_part_flash_filt = df_part_flash[df_part_flash['fecha_sin_hora'] == fecha_who]

        # Por partido con la misma fecha
        for index, fila in df_part_flash_filt.iterrows():

            equipo_loc_fs = fila['equipo_loc']
            equipo_vis_fs = fila['equipo_vis']

            # Si los equipos local y visitante son similares
            if buscar_coincidencias(equipo_loc_fs, str2=equipo_loc_who, umbral=50) and buscar_coincidencias(str1=equipo_vis_fs, str2=equipo_vis_who, umbral=50): # if buscar_coincidencias(str1=equipo_loc_fs, str2=equipo_loc_who) or buscar_coincidencias(str1=equipo_vis_fs, str2=equipo_vis_who):  # introduce FP
                # print("Match:")
                # print(f"Fecha: {fecha_who} = {fila['fecha_sin_hora']}")
                # print(f"Equipo local: {equipo_loc_who} = {equipo_loc_fs}")
                # print(f"Equipo vis: {equipo_vis_who} = {equipo_vis_fs}")
                cont_part_rell += 1

                # Relleno con columnas de df_part_flash en caso que tenga datos
                for col in l_col:
                    df_part_who.loc[idx, col] = df_part_flash.loc[index, col]

    print(f"Cantidad de partidos rellenados: {cont_part_rell} sobre {len(df_part_who_filt)} posibles.")
    return df_part_who

def buscar_coincidencias(str1, str2, umbral):
    """
    Determina si hay similaridad entre strings.

    :param str1: Primer string a comparar. (string)
    :param str2: Segundo string a comparar. (string)
    :param umbral: Porecentaje minimo de similaridad entre strings de 0 a 100. (int)
    :return: True si se asemejan en mas del valor umbral, de lo contrario, False.
    """
    return fuzz.token_set_ratio(str1, str2) < umbral


# Integracion entre datos de WHOSCORED
def map_data(df_jug, df_jug_part):
    # Filtrar las columnas necesarias de df_jug_part
    df_jug_filtered = df_jug[['id_jug', 'altura', 'fecha_nac']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_merged = pd.merge(df_jug_part, df_jug_filtered, on='id_jug', how='left')

    return df_merged

def integrate_player_to_part(df_jug_part, df_part):
    # De df_jug quiero prom_edad_{tit, sup}_{loc, vis}, prom_alt_{tit, sup}_{loc, vis}
    # De df_jug_part quiero prom_rat_{tit, sup}_{loc, vis} y ponderarlo por min played...

    # Calculo edad
    df_jug_part = determine_edad(df_part, df_jug_part)

    # Calculo min_played
    df_jug_part = determine_min_played(df_jug_part)

    # Por jugador, construyo sum_min_played y prom_rating en ultimos n partidos --> para no requerir equipo, uso fecha
    df_jug_part = determine_var_en_ult_partidos(df_jug_part, 'min_played', type='sum')
    df_jug_part = determine_var_en_ult_partidos(df_jug_part, 'rating', type='mean_pond', var_pond='min_played')
    df_jug_part.to_excel('/Users/nachomondino/Desktop/df_jug_part_rating_ult_part.xlsx')

    # df_jug_part = pd.read_excel('/Users/nachomondino/Desktop/df_jug_part_rating_ult_part.xlsx', index_col=0)

    # Construyo prom_edad_{tit, sup}_{loc, vis}, prom_alt_{tit, sup}_{loc, vis}     # Agregar construccion de rating
    df_part = construct_variables_jug_in_part(df_part, df_jug_part)
    df_part.to_excel('/Users/nachomondino/Desktop/df_part_int.xlsx')

def determine_min_played(df):

    def calculate_minutes_played(row):
        if row['titularidad'] == 'titular':
            if pd.isna(row['min_cambio']) or row['min_cambio'] >=90:
                return 90
            else:
                return row['min_cambio']
        elif row['titularidad'] == 'suplente':
            if pd.isna(row['min_cambio']) or row['min_cambio'] >=90:
                return 0
            else:
                return 90 - row['min_cambio']
        else:
            return None

    # Aplicar la función a cada fila del DataFrame para calcular los minutos jugados.
    df['min_played'] = df.apply(calculate_minutes_played, axis=1)
    return df

def determine_edad(df_part, df_jug_part):

    # Filtrar las columnas necesarias de df_jug_part
    df_part_filtered = df_part[['id_part', 'fecha']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_merged = pd.merge(df_jug_part, df_part_filtered, on='id_part', how='left')

    # Calculo edad (fecha_hora - fecha nac)
    diferencia_dias = (df_merged['fecha'] - df_merged['fecha_nac']).dt.days  # Calcular la diferencia en días
    df_merged['edad'] = diferencia_dias // 365  # Calcular la edad en años
    # df_merged = df_merged.drop(['fecha_nac'], axis=1)  # Borro columnas fecha_hora y fecha_nac

    return df_merged

def determine_var_en_ult_partidos(df, variable, type='mean', var_pond=None):  # Calcula bien.
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos

    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :param variable: String. Nombre de variable a promediar
    :return: Dataframe con estadisticas promediadas
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por jugador
    for id_jug in df['id_jug'].unique():
        # print(f"ID JUGADOR: {id_jug}")

        # Obtengo los partidos que jugó el jugador
        df_filt = df[df['id_jug'] == id_jug]
        # print(df_filt)

        # Recorrer los partidos del jugador
        for idx, row in df_filt.iterrows():

            # Filtro para seleccionar los ultimos partidos del jugador en los ultimos n dias
            fecha_part = row['fecha']
            fecha_limite = fecha_part - timedelta(days=30)
            df_seleccionados = df_filt.copy()
            df_seleccionados = df_seleccionados.dropna(subset=[variable])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)
            df_seleccionados = df_seleccionados.loc[(df_seleccionados['fecha'] >= fecha_limite) & (df_seleccionados['fecha'] < fecha_part)]
            largo = len(df_seleccionados)

            if largo > 0:
                if type == "mean_pond":
                    df_seleccionados = df_seleccionados.dropna(subset=[var_pond])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)

                    if len(df_seleccionados) > 0:
                        promedio_ponderado = (df_seleccionados[var_pond] * df_seleccionados[variable]).sum() / df_seleccionados[var_pond].sum()
                        df.loc[idx, f'prom_pond_{variable}_ult_part'] = promedio_ponderado

                elif type == "mean":
                    promedio = df_seleccionados[variable].sum() / len(df_seleccionados[variable])  # df_seleccionados[variable].dropna().sum() / len(df_seleccionados[variable].dropna())
                    df.loc[idx, f'prom_{variable}_ult_part'] = promedio

                elif type == "sum":
                    suma = df_seleccionados[variable].sum()  # df_seleccionados[variable].dropna().sum()
                    df.loc[idx, f'sum_{variable}_ult_part'] = suma

    return df

def construct_variables_jug_in_part(df_part, df_jug_part):  # Agregar calculo de rating y min played

    l_condiciones = df_jug_part['condicion'].unique()
    l_titularidades = df_jug_part['titularidad'].unique()

    # Por partido en df_part
    for index, row in df_part.iterrows():

        df_jug_part_filt = df_jug_part.copy()
        df_jug_part_filt_1 = df_jug_part_filt[df_jug_part_filt['id_part'] == row['id_part']]

        # Por condicion (home, away)
        for condicion in l_condiciones:
            df_jug_part_filt_2 = df_jug_part_filt_1[df_jug_part_filt_1['condicion'] == condicion]

            # Por titularidad (tit, sup)
            for titularidad in l_titularidades:

                # Selecciono registros
                df_jug_part_filt_3 = df_jug_part_filt_2[df_jug_part_filt_2['titularidad'] == titularidad]
                largo = len(df_jug_part_filt_3)
                # print(df_jug_part_filt_3)
                # print(sum(df_jug_part_filt_3['edad']))
                # print(sum(df_jug_part_filt_3['altura']))
                # print(sum(df_jug_part_filt_3['prom_rating_ult_part']))
                # print(sum(df_jug_part_filt_3['sum_min_played_ult_part']))

                # Calculo promedio de edad y altura
                if largo > 0:
                    prom_edad = df_jug_part_filt_3['edad'].dropna().sum() / len(df_jug_part_filt_3['edad'].dropna())
                    prom_alt = df_jug_part_filt_3['altura'].dropna().sum() / len(df_jug_part_filt_3['altura'].dropna())
                    sum_rat = df_jug_part_filt_3['prom_pond_rating_ult_part'].dropna().sum()
                    sum_min_played = df_jug_part_filt_3['sum_min_played_ult_part'].dropna().sum()

                    # Guardo columna en df_part
                    df_part.loc[index, f'prom_edad_{condicion}_{titularidad}'] = prom_edad
                    df_part.loc[index, f'prom_alt_{condicion}_{titularidad}'] = prom_alt
                    df_part.loc[index, f'sum_rat_{condicion}_{titularidad}'] = sum_rat
                    df_part.loc[index, f'sum_min_{condicion}_{titularidad}'] = sum_min_played

    return df_part

def prueba():
    pais = "argentina"

    # Levanto datasets de prueba
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_part_cleaned.xlsx')
    df_jug_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_jug_part.xlsx')
    df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_jug_formated.xlsx')
    df_part_flash = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_part_fs_cleaned.xlsx')

    # Relleno estadisticas en partidos de Whoscored usando datos de Flashscore
    df_part = fill_whoscored_with_flashscore(df_part, df_part_flash)

    # Integro df_jug a df_jug_part
    # df_jug_part_integ = map_data(df_jug, df_jug_part)
    # df_jug_part_integ.to_excel('/Users/nachomondino/Desktop/df_prueba.xlsx')

    # Integro df_jug_part_integ (df_jug_part + df_jug) a df_part
    # df_part_integ = integrate_player_to_part(df_jug_part_integ, df_part)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()