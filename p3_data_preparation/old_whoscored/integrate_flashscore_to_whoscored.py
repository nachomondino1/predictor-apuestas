import pandas as pd
import numpy as np
from p3_data_preparation import clean_data
from fuzzywuzzy import fuzz


def fill_whoscored_with_flashscore(df_match_who, df_match_flash):
    """
    Relleno NaN values en algunas columnas del dataset partido de Whoscored mediante los datos de Flashscore.

    :param df_match_who: Dataframe de partidos de Whoscored. (DataFrame)
    :param df_match_flash: Dataframe de partidos de Flashscore. (DataFrame)
    :return: Dataframe de partidos de Whoscored rellenado con datos de Flashscore. (DataFrame)
    """
    # Defino columnas a rellenar
    cont_part_rell = 0
    l_col = ['arbitro', 'dt_loc', 'dt_vis', 'posesion_loc', 'posesion_vis', 'remates_loc', 'remates_vis', 'remates_a_puerta_loc',
             'remates_a_puerta_vis', 'faltas_loc', 'faltas_vis', 'pases_loc', 'pases_vis', 'pases_comp_loc',
             'pases_comp_vis', 'offsides_loc', 'offsides_vis']

    # Preparo dfs para facilitar y mejorar integracion
    df_match_flash = preparate_to_integrate(df_match_flash)
    df_match_who = clean_data.prepare_text_columns(df_match_who, l_col_to_except=['temporada'])  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

    # Hago copias para evitar SettingWithCopyWarning al crear columnas "fecha_sin_hora"
    df_match_who_filt = df_match_who.copy()
    df_match_flash_filt = df_match_flash.copy()

    # Selecciono partidos sin estadisticas de Whoscored y con estadisticas de Flashscore
    df_match_who_filt = df_match_who_filt[df_match_who_filt['posesion_loc'].isnull()]  # Funciona bien
    df_match_flash_filt = df_match_flash_filt.dropna(subset="remates_loc")  # Funciona bien

    # Saco horas de fechas para poder compararlas (hay diferencia porque Whoscored tiene otro huso horario)
    df_match_who_filt.loc[:, 'fecha_sin_hora'] = df_match_who_filt['fecha'].dt.date
    df_match_flash_filt['fecha_sin_hora'] = df_match_flash_filt['fecha'].dt.date

    # Por partido sin estadisticas en Whoscored
    for idx_ws, row_ws in df_match_who_filt.iterrows():

        # Selecciono partidos en Flashscore con la misma fecha
        df_match_flash_filt_fecha = df_match_flash_filt[df_match_flash_filt['fecha_sin_hora'] == row_ws['fecha_sin_hora']]

        # Por partido con la misma fecha
        for idx_fs, row_fs in df_match_flash_filt_fecha.iterrows():

            # Si los equipos local y visitante son similares
            if buscar_coincidencias(row_fs['equipo_loc'], row_ws['equipo_loc'], umbral=70) and buscar_coincidencias(row_fs['equipo_vis'], row_ws['equipo_vis'], umbral=70):
                cont_part_rell += 1

                # Relleno con columnas de df_match_flash en caso que tenga datos
                for col in l_col:
                    df_match_who.loc[idx_ws, col] = df_match_flash.loc[idx_fs, col]

    df_match_who.to_excel('/Users/nachomondino/Desktop/df_rellenado.xlsx', index=False)
    print(f"Cantidad de partidos rellenados: {cont_part_rell} sobre {len(df_match_who_filt)} posibles.")
    return df_match_who

def buscar_coincidencias(str1, str2, umbral):
    """
    Determina si hay similaridad entre strings.

    :param str1: Primer string a comparar. (string)
    :param str2: Segundo string a comparar. (string)
    :param umbral: Porecentaje minimo de similaridad entre strings de 0 a 100. (int)
    :return: True si se asemejan en mas del valor umbral, de lo contrario, False.
    """
    return fuzz.token_set_ratio(str1, str2) >= umbral

def preparate_to_integrate(df_match_flash):

    # Format data: fecha y posesion
    df_match_flash['fecha'] = pd.to_datetime(df_match_flash['fecha'], format='%d.%m.%Y %H:%M')  # ya lo voy a extraer datetime... # Fundamental para poder ordenar el df por 'fecha'
    df_match_flash = convert_ball_possession_to_int(df_match_flash)

    # Clean data
    # Hago limpieza de variables object antes de integrar para facilitar la integracion de datos
    df_match_flash = clean_data.prepare_text_columns(df_match_flash, l_col_to_except=['id', 'temporada'])  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

    # Remuevo strings adicionales en los nombres de los equipos
    df_match_flash = clean_teams_names(df_match_flash)

    return df_match_flash

def convert_ball_possession_to_int(df):
    """
     Transforma la posesión de string a float.

     :param df: Dataframe con columnas 'posesion_loc' y 'posesion_vis', donde la posesión se representa como un string,
                por ejemplo, '65%'.
     :return: Dataframe con la columna 'posesion' interpretada como float, por ejemplo, 0.65.
     """
    df['posesion_loc'] = df['posesion_loc'].apply(lambda x: int(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan)
    df['posesion_vis'] = df['posesion_vis'].apply(lambda x: int(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan)
    return df

def clean_teams_names(df):
    """
    Limpia y cambia el nombre de algunos equipos en el dataframe.
    :param df: Dataframe con columnas "equipo_loc" y "equipo_vis"
    :return: Dataframe con nombres de equipos modificados y limpios
    """
    # Limpio string 'Vencedor' en el nombre de algunos equipos.
    d_sub_strings_adic = {'vencedor': '', 'equipo que avanza': ''}  # Tengo que tener cuidado, reemplazo strings... pueden ser substring y cambiarlo sin querer hacerlo.
    df['equipo_loc'] = df['equipo_loc'].replace(d_sub_strings_adic, regex=True).str.strip()
    df['equipo_vis'] = df['equipo_vis'].replace(d_sub_strings_adic, regex=True).str.strip()

    # Quitar abreviaturas en nombres de equipos (NO FUNCIONA...)
    d_abrev_team_names = {' l p': ' la plata', 'atl ': 'atletico ', ' jrs': ' juniors', ' utd': ' united'}  # Tengo que tener cuidado, reemplazo strings... pueden ser substring y cambiarlo sin querer hacerlo.
    df['equipo_loc'] = df['equipo_loc'].replace(d_abrev_team_names, regex=True).str.strip()
    df['equipo_vis'] = df['equipo_vis'].replace(d_abrev_team_names, regex=True).str.strip()

    # Reemplazo nombres enteros de equipos para que sea igual a los de la entidad jugador
    d_team_names = {'estudiantes la plata': 'estudiantes', 'qpr': 'queens park rangers',  'wolves': 'wolverhampton',
                    'west brom': 'west bromwich albion'}
    for equipo_part, equipo_jug in d_team_names.items():
        df['equipo_loc'] = df['equipo_loc'].replace(equipo_part, equipo_jug)
        df['equipo_vis'] = df['equipo_vis'].replace(equipo_part, equipo_jug)

    return df