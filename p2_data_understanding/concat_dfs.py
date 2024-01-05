import pandas as pd


def concat_dfs():

    # Levanto datasets a concatenar
    df_1 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/England/entidad_jugadores (premier).xlsx', index_col=0)
    df_2 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/England/entidad_jugadores_FIFA 08.xlsx', index_col=0)
    # df_3 = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/inglaterra/premier-league_1989_1990_inglaterra.xlsx')

    # Imprimo caracteristicas de cada dataframe a concatenar
    print(df_1.head(1))
    print(df_1.shape)

    print(df_2.head(1))
    print(df_2.shape)
    # df_2['fecha'] = pd.to_datetime(df_2['fecha'], format='%d.%m.%Y %H:%M')  # ya lo voy a extraer datetime... # Fundamental para poder ordenar el df por 'fecha'

    # print(df_3.head(1))
    # print(df_3.shape)

    # Concateno dataframes
    # df_concat = pd.concat([df_1, df_2, df_3], axis=0)
    df_concat = pd.concat([df_1, df_2], axis=0)
    print(df_concat.head(1))
    print(df_concat.shape)
    # df_concat.to_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/argentina_south_america/df_jug.xlsx', index=False)

    # Eliminar los duplicados
    df_sin_duplicados = df_concat.drop_duplicates().reset_index(drop=True)
    print(df_sin_duplicados.shape)

    df_sin_duplicados.to_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/england/entidad_jugadores.xlsx', index=False)


concat_dfs()
