import pandas as pd
from predictor.utils.set_up_logging import logger
import os
from dotenv import load_dotenv
import predictor.utils.directories as directories


def concat_raw_data_by_competition(id_country, country, l_dataframes, export=True):
    """
    Concatena dfs de distintas competiciones.
    :return: Dataframe. Contiene todas las competiciones de un country (DataFrame)
    """
    # Levanto competiciones del country
    df_comp = pd.read_excel('./data/_shared/master_tables/df_competencies.xlsx')
    df_comp_pais = df_comp[df_comp['id_country'] == id_country]
    print("Competiciones del country:\n", df_comp_pais)

    # Por dataframe (e.f. df_match, df_match_player)
    for dataframe in l_dataframes:

        ruta_base_comp = f'./data/{country.lower()}/data_understanding/data_seg/per_competition/{dataframe}'
        df_concat = pd.DataFrame()
        print("\n", f"\n Dataframe: {dataframe} \n".center(240, "#"))

        # Por competition del country
        for i, row in df_comp_pais.iterrows():
            # d_comps = {'laliga': 'la-liga', 'laliga2': 'la-liga-2'}

            competicion_form = row['competition_flashscore'].lower().replace(" ", "-")  # competicion_form = row['competition_sofifa'].lower().replace(" ", "-")
            # if competicion_form in d_comps.keys():
                # competicion_form = d_comps[competicion_form]
            print(f" Competition: {row['competition_flashscore']} --> form {competicion_form} ".center(120, "-"))

            # Levanto su dataframe
            try:
                if dataframe == "df_player_fifa_sofifa":
                    df = pd.read_excel(f'{ruta_base_comp}/{competicion_form}.xlsx')
                else:
                    df = pd.read_excel(f'{ruta_base_comp}/{competicion_form}.xlsx', index_col=0)
                print(df.head(2))
                print(f"Shape df: {df.shape}")

                # Concateno dataframes
                df_concat = pd.concat([df_concat, df], axis=0)
                print(f"Shape df_concat: {df_concat.shape}")

            except:
                print(f"No existe el dataframe para la competition {competicion_form}")

        df_sin_duplicados = df_concat.drop_duplicates() if dataframe == "df_player_fifa_sofifa" else df_concat[~df_concat.index.duplicated()]

        # Si hay duplicados
        if len(df_concat)-len(df_sin_duplicados) > 0:
            logger.info(f"Cuidado! Hay filas repetidas. Hay filas repetidas en {dataframe}. Nº de filas repetidas: {len(df_concat)-len(df_sin_duplicados)}")
            logger.info(df_sin_duplicados)

        if export:
            df_sin_duplicados.to_excel(f'./data/{country}/data_understanding/{dataframe}.xlsx', index=True)

def concat_raw_data_by_season(country, l_dataframes, export=True):
    """
    Concatena DataFrames de distintas temporadas automáticamente.
    
    :param country: str, País correspondiente a los datos.
    :param l_dataframes: list, Lista de nombres de subcarpetas donde están los archivos por temporada.
    :param export: bool, Si es True, exporta el resultado como archivo Excel.
    :return: None
    """
    for dataframe in l_dataframes:
        print(f"\nProcesando dataframe: {dataframe}")
        df_concat = pd.DataFrame()
        ruta_base = f'./data/{country}/data_understanding/data_seg/per_season/{dataframe}'

        # Obtener automáticamente todos los archivos dentro de la ruta
        try:
            l_filenames = [f for f in os.listdir(ruta_base) if f.endswith('.xlsx')]
        except FileNotFoundError:
            print(f"Ruta no encontrada: {ruta_base}")
            continue
        
        print(f"Archivos encontrados ({len(l_filenames)}): {l_filenames}")

        # Concatenar todos los archivos
        for filename in l_filenames:
            print(f"Procesando archivo: {filename}")
            try:
                filepath = os.path.join(ruta_base, filename)
                
                # Cargar el archivo Excel
                if dataframe == "df_player_sofifa":
                    df = pd.read_excel(filepath)  
                else:
                    df = pd.read_excel(filepath, index_col=0)
                
                print(df.head(1))
                print(f"Shape df: {df.shape}")

                # Concatenar al DataFrame final
                df_concat = pd.concat([df_concat, df], axis=0)
                print(f"Shape acumulado: {df_concat.shape}")
            except Exception as e:
                print(f"Falló procesando {filename}: {e}")

        # Eliminar duplicados
        df_sin_duplicados = df_concat.drop_duplicates()
        print(f"\n\nNº de filas repetidas eliminadas: {len(df_concat) - len(df_sin_duplicados)}")

        # Exportar el resultado si se requiere
        if export:
            export_path = f'./data/{country}/data_understanding/{dataframe}_{country}.xlsx'
            os.makedirs(os.path.dirname(export_path), exist_ok=True)  # Crear directorios si no existen
            df_sin_duplicados.to_excel(export_path, index=True)
            print(f"Archivo exportado: {export_path}")

def concat_raw_data_by_country(d_countries, verbose: int = 1):
    """
    Concatenacion de datos extraidos de varios paises.
    """
    df_ct1, df_ct2, df_ct3, df_ct4, df_ct5, df_ct6 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Por pais
    for id_country, country in d_countries.items():

        # Levanto datasets de Flashscore
        df1 = pd.read_excel(f'data/{country}/data_understanding/df_match.xlsx', index_col=0)
        df2 = pd.read_excel(f'data/{country}/data_understanding/df_match_player.xlsx', index_col=0)
        df3 = pd.read_excel(f'data/{country}/data_understanding/df_match_odds.xlsx', index_col=0)
        df4 = pd.read_excel(f'data/{country}/data_understanding/df_player_sofifa.xlsx', index_col=0)
        df5 = pd.read_excel(f'data/{country}/data_understanding/df_player_fifa_sofifa.xlsx', index_col=0)
        df6 = pd.read_excel(f'data/{country}/data_understanding/df_teams_sofifa.xlsx', index_col=0)

        if verbose >= 1:
            logger.info(f"Country: {country}. df_match: {df1.shape}. df_match_player: {df2.shape} df_match_odds: {df3.shape}")

        # Concateno datos
        df_ct1 = pd.concat([df_ct1, df1], axis=0)
        df_ct2 = pd.concat([df_ct2, df2], axis=0)
        df_ct3 = pd.concat([df_ct3, df3], axis=0)
        df_ct4 = pd.concat([df_ct4, df4], axis=0)
        df_ct5 = pd.concat([df_ct5, df5], axis=0)
        df_ct6 = pd.concat([df_ct6, df6], axis=0)

        if verbose >= 1:
            logger.info(f"df_match_ct: {df_ct1.shape}. df_match_player_ct: {df_ct2.shape} df_match_odds_ct: {df_ct3.shape}")

    return df_ct1, df_ct2, df_ct3, df_ct4, df_ct5, df_ct6 

def concat_missing_data_by_country(d_countries, verbose: int = 1):
    """
    Concatenacion de datos extraidos de varios paises.
    """
    df_ct1, df_ct2 = pd.DataFrame(), pd.DataFrame()

    # Por pais
    for id_country, country in d_countries.items():

        # Levanto datasets de Flashscore
        df1 = pd.read_excel(f'data/{country}/deployment/missing/data_understanding/all/df_match_odds_miss.xlsx', index_col=0)
        df2 = pd.read_excel(f'data/{country}/deployment/missing/data_preparation/all/df_integrated_missing.xlsx', index_col=0)
       
        if verbose >= 1:
            logger.info(f"Country: {country}. df1: {df1.shape}. df2: {df2.shape}")

        # Concateno datos
        df_ct1 = pd.concat([df_ct1, df1], axis=0)
        df_ct2 = pd.concat([df_ct2, df2], axis=0)

    return df_ct1, df_ct2


if __name__ == "__main__":

    # Definicion de parametros
    raw_data, integrate_data, missing_data = False, False, False
    d_countries = {6: 'argentina', 48: 'england', 55: 'france', 59: 'germany', 77: 'italy', 148: 'spain', 167: 'usa'}
    d_countries = {48: 'england', 55: 'france', 59: 'germany', 77: 'italy', 148: 'spain'}

    # Flashscore
    if raw_data:
        df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = concat_raw_data_by_country(d_countries)
        
        # Exporto datos
        df_match.to_excel('data/all/data_understanding/df_match.xlsx')
        df_match_player.to_excel('data/all/data_understanding/df_match_player.xlsx')
        df_match_odds.to_excel('data/all/data_understanding/df_match_odds.xlsx')
        df_player_sofifa.to_excel('data/all/data_understanding/df_player_sofifa.xlsx')
        df_player_fifa_sofifa.to_excel('data/all/data_understanding/df_player_fifa_sofifa.xlsx')
        df_teams_sofifa.to_excel('data/all/data_understanding/df_teams_sofifa.xlsx')

    if integrate_data:
        df_map, df_player_sofifa, df_player_fifa_sofifa, df_teams = concat_integrate_data_by_country(d_countries)

        # Exporto datos
        df_map.to_excel('data/all/data_preparation/integrate_data/df_map_players_fs_so.xlsx')
        df_player_sofifa.to_excel('data/all/data_preparation/clean_data/df_player_sofifa_cleaned.xlsx')
        df_player_fifa_sofifa.to_excel('data/all/data_preparation/clean_data/df_player_fifa_sofifa_cleaned.xlsx')
        df_teams.to_excel('data/all/data_preparation/integrate_data/df_teams.xlsx')

    if missing_data:
        directories.make_directories(l_directorios=['data/all/deployment/missing/data_understanding/all', 'data/all/deployment/missing/data_preparation/all'])
        df_match_odds_miss, df_integrated_miss = concat_missing_data_by_country(d_countries)

        # Exporto datos
        df_match_odds_miss.to_excel('data/all/deployment/missing/data_understanding/all/df_match_odds_miss.xlsx')
        df_integrated_miss.to_excel('data/all/deployment/missing/data_preparation/all/df_integrated_missing.xlsx')


    # CONCATENACION DE DATA SEG X SEASON O COMPETICION
    # Definicion de variables
    competicion, temporada = True, False
    id_country = 1000
    export = True

    # Levento df_countries y obtengo id
    df_countries = pd.read_excel(f'./data/_shared/master_tables/df_countries.xlsx')
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0]
    print(country)

    # Concateno competiciones del country
    if competicion:
        # l_dataframes =  ['df_teams_sofifa'] 
        l_dataframes = ["df_match", "df_match_player", 'df_match_odds'] # df_teams', 'df_player', 'df_coaches', 'df_player_sofifa', 'df_player_fifa_sofifa']
        concat_raw_data_by_competition(id_country, country, l_dataframes, export)

    # Concateno temporadas de una misma competition del country
    if temporada:
        l_dataframes = ['df_player_sofifa', 'df_player_fifa_sofifa']  # ["df_match", "df_match_player", 'df_match_odds']
        concat_raw_data_by_season(country, l_dataframes, export)  
    