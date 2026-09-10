import pandas as pd
from utils.set_up_logging import logger
import os
from dotenv import load_dotenv
import utils.directories as directories

"""
El objetivo es concatenar los df_map de todos los paises para tener un mapeo centralizado y ser inmune al traspaso de jugadores entre ligas. 
(sobretodo a mitad temporada y en el cambio de temporada hasta que sale el nuevo fifa)
Incluso, la integracion podria ser mas robusta
"""

def concat_integrate_data_by_country(d_countries: dict, d_dates: dict, verbose: int = 0):
    """
    Concatenacion de df_map_players de varios paises.

    # Parameters:
        l_countries: Paises a concatenar su mapeo.
    """
    # Defino variables
    df_map, df_player_sofifa, df_player_fifa_sofifa = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Por pais
    for id_country in d_countries.keys():
        country = d_countries[id_country]
        date = d_dates[id_country]
        logger.info(f"Country: {country} Date: {date}")

        # Levanto su df_map
        df_player_sofifa_p = pd.read_excel(f'data/{country}/p3_data_preparation/{date}/clean_data/df_player_sofifa_cleaned.xlsx', index_col=0)
        df_player_fifa_sofifa_p = pd.read_excel(f'data/{country}/p3_data_preparation/{date}/clean_data/df_player_fifa_sofifa_cleaned.xlsx', index_col=0)
        df_map_p = pd.read_excel(f'data/{country}/p3_data_preparation/{date}/integrate_data/df_map_players_fs_so.xlsx', index_col=0)
        
        if 'Unnamed: 0' in df_player_fifa_sofifa_p.columns:
            df_player_fifa_sofifa_p = df_player_fifa_sofifa_p.drop(columns=['Unnamed: 0'])

        logger.info(df_player_sofifa_p.shape)
        logger.info(df_player_fifa_sofifa_p.shape)
        logger.info(df_map_p.shape)

        if verbose >= 2:
            print("Df_player_sofifa")
            print(df_player_sofifa_p)
            print("Df_player_fifa_sofifa")
            print(df_player_fifa_sofifa_p)
            print("Df_map")
            print(df_map_p)
            print(f"\n\nNº de filas: {len(df_map_p)}")

        # Concateno
        df_map = pd.concat([df_map, df_map_p], axis=0)
        df_player_sofifa = pd.concat([df_player_sofifa, df_player_sofifa_p], axis=0)
        df_player_fifa_sofifa = pd.concat([df_player_fifa_sofifa, df_player_fifa_sofifa_p], axis=0)

        logger.info("Concat countries:")
        logger.info(df_player_sofifa.shape)
        logger.info(df_player_fifa_sofifa.shape)
        logger.info(df_map.shape)

    # Elimino duplicados
    df_player_sofifa = df_player_sofifa.loc[~df_player_sofifa.index.duplicated(keep='first')]
    df_player_fifa_sofifa = df_player_fifa_sofifa.drop_duplicates(subset=['id_player', 'fifa_year'])  # No uso id_competition? Creo que no pues ya mapié por competicion antes... y no lo hare a futuro pues usare este df
    df_map = drop_duplicate_maps(df_map)

    logger.info("Sin duplicados:")
    logger.info(df_player_sofifa.shape)
    logger.info(df_player_fifa_sofifa.shape)
    logger.info(df_map.shape)

    # Exporto datos
    df_map.to_excel(f'{base_path_int}/df_map_players_fs_so.xlsx')
    df_player_sofifa.to_excel(f'{base_path_clean}/df_player_sofifa_cleaned.xlsx')
    df_player_fifa_sofifa.to_excel(f'{base_path_clean}/df_player_fifa_sofifa_cleaned.xlsx')

    return df_map, df_player_sofifa, df_player_fifa_sofifa
    
def drop_duplicate_maps(df):

    # Elimino duplicados (x traspasos de jugadores ppalmente) (28588 --> 29198)
    # Paso 1: Calcular el porcentaje de NaN por fila
    df['nan_percentage'] = df.isna().mean(axis=1)

    # Paso 2: Ordenar el DataFrame basado en el porcentaje de NaN (de menor a mayor)
    # df_sorted = df.sort_values(by='nan_percentage')
    df_sorted = df.sort_values(by=['nan_percentage', "porcentaje_coincidencia", 'tipo'], ascending=[True, False, True])

    # Paso 3: Eliminar duplicados y quedarte con los que tienen menos NaN
    df_sin_dup = df_sorted.drop_duplicates(subset=['id_player_fs'])

    # Opcional: Eliminar la columna auxiliar 'nan_percentage'
    df_sin_dup = df_sin_dup.drop(columns=['nan_percentage'])
    # print(f"\n\nNº de filas repetidas: {len(df_concat) - len(df_sin_duplicados)}")
    return df_sin_dup

def integrate_with_new_map(l_countries):
    """
    Generar el df_integrated con el nuevo mapeo completo (para poder hacer main_best_model directamente)
    """
    from main import DataPreparation
    dp = DataPreparation()

    # Por pais
    for country in l_countries:
        logger.info(f"Country: {country}")

        # Levanto datos del pais (de Sofifa levanta en integrate...)
        df_match = pd.read_excel(f'./data/{country}/p3_data_preparation/clean_data/df_match_cleaned.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./data/{country}/p3_data_preparation/clean_data/df_match_player_cleaned.xlsx', index_col=0)

        # Integro datos
        df_p = dp.integrate_data(df_match, df_match_player, df_player_sofifa=pd.DataFrame(), df_player_fifa_sofifa=pd.DataFrame(), prod=True, export=False) 

        # Exporto datos
        df_p.to_excel(f'./data/all/df_integrated_{country}.xlsx', index=True)


if __name__ == "__main__":

    # Definicion de parametros
    d_countries = {48: 'england', 55: 'france', 59: 'germany', 77: 'italy', 148: 'spain', 167: 'usa', 6: 'argentina'}
    d_dates = {48: '2025-08-14', 55: '2025-08-14', 59: '2025-05-29', 77: '2025-05-29', 148: '2025-08-14', 167: '2025-05-29', 6: '2025-02-06'}

    # Creo directorio donde guardar datos
    base_path_clean = 'data/data_preparation/clean_data'
    base_path_int = 'data/data_preparation/integrate_data'
    directories.make_directories(l_directorios=[base_path_clean, base_path_int])

    # Concatendo df_map
    df_map, df_player_sofifa, df_player_fifa_sofifa = concat_integrate_data_by_country(d_countries=d_countries, d_dates=d_dates)

    # Integro datos usando el df_map completo
    # integrate_with_new_map(l_countries=d_countries.values())