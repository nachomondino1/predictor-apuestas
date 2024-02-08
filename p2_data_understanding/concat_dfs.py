import pandas as pd


def concat_dfs_por_competicion(pais, l_dataframes, export=True):
    """
    Concatena dfs de distintas competiciones.
    :return: Dataframe. Contiene todas las competiciones de un pais (DataFrame)
    """
    # Levanto competiciones del pais
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx')
    df_comp_pais = df_comp[df_comp['pais_flashscore'] == pais.capitalize()]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
    print("Competiciones del pais:\n", df_comp_pais)

    # Por dataframe (e.f. df_part, df_part_jug)
    for dataframe in l_dataframes:

        ruta_base_comp = f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/data_seg/por_competicion/{dataframe}'
        df_concat = pd.DataFrame()
        print(f"\n\nDataframe: {dataframe}")

        # Por competicion del pais
        for i, row in df_comp_pais.iterrows():

            competicion_form = row['competicion_flashscore'].lower().replace(" ", "-")
            print(f" Competicion: {row['competicion_flashscore']} ".center(120, "$"))

            # Levanto su dataframe
            try:
                df = pd.read_excel(f'{ruta_base_comp}/{competicion_form}_{pais.lower()}.xlsx')
                print(f"Shape df: {df.shape}")

                # Concateno dataframes
                df_concat = pd.concat([df_concat, df], axis=0)
                print(f"Shape df_concat: {df_concat.shape}")

            except:
                print(f"No existe el dataframe para la competicion {competicion_form}")

        # Me fijo si hay duplicados (no deberia)
        df_sin_duplicados = df_concat.drop_duplicates().reset_index(drop=True)
        print(f"Nº de filas repetidas: {len(df_concat)-len(df_sin_duplicados)}")

        if export:
            df_sin_duplicados.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/{dataframe}.xlsx', index=False)

def concat_dfs_por_temporada(pais, l_dataframes, l_filenames, export=True):
    """
    Concatena dfs de distintas temporadas
    :return: Dataframe. Contiene todas las temporadas especificadas.
    """

    competicion = l_filenames[0][:l_filenames[0].find("_") + 1] + pais

    for dataframe in l_dataframes:

        print(f"\nDataframe: {dataframe}")
        df_concat = pd.DataFrame()
        ruta_base = f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/data_seg/por_temporada/{dataframe}'

        # Por dataframe a concatenar
        for filename in l_filenames:
            print(f"Filename: {filename}")

            try:
                # Levanto el dataframe
                df = pd.read_excel(f'{ruta_base}/{filename}') # index_col=0
                print(df.head(1))
                print(f"Shape df: {df.shape}")

                # Concateno
                df_concat = pd.concat([df_concat, df], axis=0)
                print(f"Shape df_concat: {df_concat.shape}")

            except:
                print("Falló")

        # Me fijo si hay duplicados (no deberia)
        df_sin_duplicados = df_concat.drop_duplicates().reset_index(drop=True)
        print(f"\n\nNº de filas repetidas: {len(df_concat) - len(df_sin_duplicados)}")

        if export:
            df_sin_duplicados.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/data_seg//por_competicion/{dataframe}/{competicion}.xlsx', index=False)

if __name__ == "__main__":

    # Definicion de variables
    pais = "Argentina"
    export = True

    # Concateno competiciones del pais
    l_dataframes = ["df_part", "df_part_jug"]
    concat_dfs_por_competicion(pais, l_dataframes, export)

    # Concateno temporadas de una misma competicion del pais
    # l_dataframes = ["df_part", "df_part_jug"]  #  ["df_jug"]
    # l_filenames = ["copa-italia_2015_2016_italia.xlsx", "copa-italia_2008_2009_italia.xlsx"]
    # concat_dfs_por_temporada(pais, l_dataframes, l_filenames, export)
