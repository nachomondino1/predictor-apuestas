import pandas as pd
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
def map_player_entities(df_jug, df_jug_part):
    """
    Integro entidad de jugador con la entidad de jugador por partido.
    :param df_jug: Dataframe jugador
    :param df_jug_part: Dataframe jugador por partido
    :return: Dataframe jugador por partido con columnas altura y fecha_nac por jugador.
    """
    # Filtrar las columnas necesarias de df_jug_part
    df_jug_filtered = df_jug[['id_jug', 'altura', 'fecha_nac']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_merged = pd.merge(df_jug_part, df_jug_filtered, on='id_jug', how='left')
    return df_merged

def map_player_to_part(df_part, df_jug_part):  # Agregar calculo de rating y min played
    # De df_jug quiero prom_edad_{tit, sup}_{loc, vis}, prom_alt_{tit, sup}_{loc, vis}
    # De df_jug_part quiero prom_rat_{tit, sup}_{loc, vis} y ponderarlo por min played...

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
    from p3_data_preparation import construct_data

    pais = "argentina"

    # Levanto datasets de prueba
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_part_cleaned.xlsx')
    df_jug_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_jug_part.xlsx')
    df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_jug_formated.xlsx')
    df_part_flash = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_part_fs_cleaned.xlsx')

    # Integro Flashscore a Whoscored para rellenar estadisticas en partidos de Whoscored
    df_part = fill_whoscored_with_flashscore(df_part, df_part_flash)

    # Integro df_jug a df_jug_part
    df_jug_part = map_player_entities(df_jug, df_jug_part)

    # Calculo edad y minutos jugados
    df_jug_part = construct_data.determine_edad(df_part, df_jug_part)  # construct_data
    df_jug_part = construct_data.determine_min_played(df_jug_part)  # construct_data
    # Por jugador, construyo sum_min_played y prom_rating en ultimos n partidos --> para no requerir equipo, uso fecha
    df_jug_part = construct_data.determine_var_en_ult_partidos(df_jug_part, 'min_played', type='sum')
    df_jug_part = construct_data.determine_var_en_ult_partidos(df_jug_part, 'rating', type='mean_pond', var_pond='min_played')

    # Integro df_jug_part_integ (df_jug_part + df_jug) a df_part
    df_part_integ = map_player_to_part(df_jug_part, df_part)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()