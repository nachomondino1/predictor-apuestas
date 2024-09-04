import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from set_up_logging import logger
import os
from dotenv import load_dotenv

def concat_integrate_dfs(l_countries):

    df = pd.DataFrame()

    # Por pais
    for id_country, country in l_countries.items():

        # Levanto su df_integrated.xlsx
        ruta_to_int_country = f'data/{country}/p3_data_preparation/integrate_data/df_map_players_fs_so.xlsx'
        df_int_country = pd.read_excel(ruta_to_int_country, index_col=0)
        print(df_int_country)
        print(f"\n\nNº de filas: {len(df_int_country)}")

        # Concateno
        df = pd.concat([df, df_int_country], axis=0)
        print(f"\n\nNº de filas concat: {len(df)}")

    # Elimino duplicados (x traspasos de jugadores ppalmente) (28588 --> 29198)
    # Paso 1: Calcular el porcentaje de NaN por fila
    df['nan_percentage'] = df.isna().mean(axis=1)

    # Paso 2: Ordenar el DataFrame basado en el porcentaje de NaN (de menor a mayor)
    df_sorted = df.sort_values(by='nan_percentage')

    # Paso 3: Eliminar duplicados y quedarte con los que tienen menos NaN
    df_sin_dup = df_sorted.drop_duplicates(subset=['id_player_fs'])

    # Opcional: Eliminar la columna auxiliar 'nan_percentage'
    df_sin_dup = df_sin_dup.drop(columns=['nan_percentage'])
    # print(f"\n\nNº de filas repetidas: {len(df_concat) - len(df_sin_duplicados)}")
    return df, df_sin_dup

def concat_dfs_per_competition(id_country, country, l_dataframes, export=True):
    """
    Concatena dfs de distintas competiciones.
    :return: Dataframe. Contiene todas las competiciones de un country (DataFrame)
    """
    # Levanto competiciones del country
    df_comp = pd.read_excel('./data/df_competencies.xlsx')
    df_comp_pais = df_comp[df_comp['id_country'] == id_country]
    print("Competiciones del country:\n", df_comp_pais)

    # Por dataframe (e.f. df_match, df_match_player)
    for dataframe in l_dataframes:

        ruta_base_comp = f'./p2_data_understanding/data/{country.lower()}/data_seg/per_competition/{dataframe}'
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
            df_sin_duplicados.to_excel(f'./data/{country}/p2_data_understanding/{dataframe}.xlsx', index=True)

def concat_dfs_per_season(country, competition, l_dataframes, l_filenames, export=True):
    """
    Concatena dfs de distintas temporadas
    :return: Dataframe. Contiene todas las temporadas especificadas.
    """
    # competition = l_filenames[0][:l_filenames[0].find("_") + 1] + country

    for dataframe in l_dataframes:

        print(f"\nDataframe: {dataframe}")
        df_concat = pd.DataFrame()
        ruta_base = f'./data/{country}/p2_data_understanding/data_seg/per_season/{dataframe}'

        # Por dataframe a concatenar
        for filename in l_filenames:
            print(f"Filename: {filename}")

            try:
                # Levanto el dataframe
                if dataframe == "df_player":
                    df = pd.read_excel(f'{ruta_base}/{filename}')  
                else:
                    df = pd.read_excel(f'{ruta_base}/{filename}', index_col=0)  
            
                print(df.head(1))
                print(f"Shape df: {df.shape}")

                # Concateno
                df_concat = pd.concat([df_concat, df], axis=0)
                print(f"Shape df_concat: {df_concat.shape}")

            except:
                print("Falló")

        # Me fijo si hay duplicados (no deberia)
        df_sin_duplicados = df_concat.drop_duplicates()
        print(f"\n\nNº de filas repetidas: {len(df_concat) - len(df_sin_duplicados)}")

        if export:
            df_sin_duplicados.to_excel(f'./data/{country}/p2_data_understanding/data_seg/per_competition/{dataframe}/{competition}.xlsx', index=False)

if __name__ == "__main__":
    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')
    env = os.getenv('ENVIRONMENT')

    # Definicion de variables
    id_country = 148
    export = True
    competicion, temporada, integracion = False, False, True
    d_countries = {6: 'argentina', 48: 'england', 55: 'france', 59: 'germany', 77: 'italy', 148: 'spain', 167: 'usa'}

    # Levento df_countries y obtengo id
    df_countries = pd.read_excel(f'./data/df_countries.xlsx')
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0]
    print(country)

    # Integracion
    if integracion:
        df, df_sin_dup = concat_integrate_dfs(d_countries)

        # Exporto datos 
        if env == 'dev':
            df.to_excel(f'{BASE_DIR_LOCAL}/df_integrated_all_con_dup.xlsx')
            df_sin_dup.to_excel(f'{BASE_DIR_LOCAL}/df_integrated_all.xlsx')

        elif env == "prod":
            df_sin_dup.to_excel('data/df_map_players_fs_so.xlsx')
    
    # Concateno competiciones del country
    if competicion:
        l_dataframes =  ['df_teams_sofifa'] # ["df_match", "df_match_player", 'df_match_odds', 'df_teams', 'df_player', 'df_coaches', 'df_player_sofifa', 'df_player_fifa_sofifa']
        concat_dfs_per_competition(id_country, country, l_dataframes, export)

    # Concateno temporadas de una misma competition del country
    if temporada:
        l_dataframes = ["df_match", "df_match_player", 'df_match_odds']  # ["df_player"]
        l_filenames =  ["FIFA 18_24.xlsx", "FIFA 07_17.xlsx"]   # l_files = [col for col in df_last_old_matches.columns if re.search(r'_player_', col) and "_miss" not in col]  # Selecciono las variables que corresponden a jugadores
        competition = "premier_league"
        concat_dfs_per_season(country, competition, l_dataframes, l_filenames, export)  
