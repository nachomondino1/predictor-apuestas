import pandas as pd
import warnings

def concat_dfs_per_competition(id_country, country, l_dataframes, export=True):
    """
    Concatena dfs de distintas competiciones.
    :return: Dataframe. Contiene todas las competiciones de un country (DataFrame)
    """
    # Levanto competiciones del country
    df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
    df_comp_pais = df_comp[df_comp['id_country'] == id_country]
    print("Competiciones del country:\n", df_comp_pais)

    # Por dataframe (e.f. df_match, df_match_player)
    for dataframe in l_dataframes:

        ruta_base_comp = f'./p2_data_understanding/data/{country.lower()}/data_seg/per_competition/{dataframe}'
        df_concat = pd.DataFrame()
        print("\n", f"\n Dataframe: {dataframe} \n".center(240, "#"))

        # Por competition del country
        for i, row in df_comp_pais.iterrows():

            competicion_form = row['competition_flashscore'].lower().replace(" ", "-")
            print(f" Competition: {row['competition_flashscore']} ".center(120, "-"))

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

        if dataframe == "df_player_fifa_sofifa":
            df_sin_duplicados = df_concat.drop_duplicates()        
        else:
             # Identificar los índices duplicados
            # indices_duplicados = df_concat.index[df_concat.index.duplicated()]
            # print(len(indices_duplicados))

            # Filtrar el DataFrame para mantener solo las filas cuyo índice no está duplicado
            df_sin_duplicados = df_concat[~df_concat.index.duplicated()]
            # print(df_sin_duplicados.shape)

        # Si hay duplicados
        if len(df_concat)-len(df_sin_duplicados) > 0:
            text = f"Cuidado! Hay filas repetidas. Hay filas repetidas en {dataframe}. Nº de filas repetidas: {len(df_concat)-len(df_sin_duplicados)}"
            warnings.warn(text)
            print(df_sin_duplicados)

        if export:
            df_sin_duplicados.to_excel(f'./p2_data_understanding/data/{country}/{dataframe}.xlsx', index=True)

def concat_dfs_per_season(country, competition, l_dataframes, l_filenames, export=True):
    """
    Concatena dfs de distintas temporadas
    :return: Dataframe. Contiene todas las temporadas especificadas.
    """
    # competition = l_filenames[0][:l_filenames[0].find("_") + 1] + country

    for dataframe in l_dataframes:

        print(f"\nDataframe: {dataframe}")
        df_concat = pd.DataFrame()
        ruta_base = f'./p2_data_understanding/data/{country}/data_seg/per_season/{dataframe}'

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
            df_sin_duplicados.to_excel(f'./p2_data_understanding/data/{country}/data_seg/per_competition/{dataframe}/{competition}.xlsx', index=False)

def prueba():
     # Definicion de variables
    country = "England"
    export = True
    competicion, temporada = True, False
    
    # Levento df_countries y obtengo id
    df_countries = pd.read_excel(f'./p2_data_understanding/data/df_countries.xlsx')
    id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]
    print(id_country)

    # Concateno competiciones del country
    if competicion:
        l_dataframes = ["df_match", "df_match_player", 'df_match_odds', 'df_teams', 'df_player', 'df_coaches']  # ['df_player', 'df_player_temp']  # 
        concat_dfs_per_competition(id_country, country, l_dataframes, export)

    # Concateno temporadas de una misma competition del country
    if temporada:
        l_dataframes = ["df_player"]  #  ["df_player"]
        l_filenames = ["FIFA 18_24.xlsx", "FIFA 07_17.xlsx"]
        competition = "premier_league"
        concat_dfs_per_season(country, competition, l_dataframes, l_filenames, export)

if __name__ == "__main__":
    prueba()
