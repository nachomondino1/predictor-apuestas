import pandas as pd
import numpy as np
from p3_data_preparation import clean_data

def preparate_to_integrate(df_part_flash):

    # Format data: fecha y posesion
    df_part_flash['fecha'] = pd.to_datetime(df_part_flash['fecha'], format='%d.%m.%Y %H:%M')  # ya lo voy a extraer datetime... # Fundamental para poder ordenar el df por 'fecha'
    df_part_flash = convert_posesion_to_int(df_part_flash)

    # Clean data
    # Hago limpieza de variables object antes de integrar para facilitar la integracion de datos
    df_part_flash = clean_data.prepare_text_columns(df_part_flash, l_col_to_except=['id', 'temporada'])  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

    # Remuevo strings adicionales en los nombres de los equipos
    df_part_flash = clean_teams_names(df_part_flash)

    return df_part_flash

def convert_posesion_to_int(df):
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