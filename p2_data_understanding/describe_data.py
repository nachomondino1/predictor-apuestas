# Importo librerias
import pandas as pd


def getting_to_know_data(df):
    """
    Describe dataframe pasado como parametro
    :param df: Dataframe
    :return: funcion sin retorno
    """
    # Data frame's dimensionality
    print("\nDataframe shape: ", df.shape)

    # See firsts 2 Dataframe's rows
    print("\nPrimeras 2 filas del dataframe:")
    pd.set_option("display.precision", 2)  # mostrar maximo dos decimales
    if len(df.columns) < 50:
        pd.set_option("display.max.columns", None)  # para ver todas las columnas del df y no que las colapse
    print(df.head(2))  # y .tail es para ver las ultimas filas

    # Displaying Data Types
    print("\nDataframe info:")
    df.info()

    # Showing Basics Statistics
    if len(df.columns) > 0:
        print("\nDataframe basic statistics:")
        print(df.describe(include='all'))  # basic descriptive statistics for all numeric columns

def verificar_unicidad_registros(df):

    print(f"\nAnalisis de unicidad de registros...")
    indices_duplicados = df.index.duplicated()
    df_dup = df.index[indices_duplicados]

    if len(df_dup) > 0:
        print(f"\t El índice tiene valores duplicados.") # Indices duplicados: {list(indices_duplicados)}")
    else:
        print("\tEl índice no tiene valores duplicados.")

def check_ids_in_both_dataframes(df1, df2, column: str = None):

    if column is None:
        todos_en_df2 = df1.index.isin(df2.index).all()
    else:
        todos_en_df2 = df1.index.isin(df2[column]).all()

    if todos_en_df2:
        print("\tTodos los valores del índice de df1 están en el índice de df2.")
    else:
        print("\tAl menos un valor del índice de df1 no está en el índice de df2.")

def prueba():

    country = "England"

    # Levanto datasets
    df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx')  #     df_match = pd.read_excel(f'./p3_data_preparation/data/{country}/df_match_formated.xlsx')
    df_match_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx')
    df_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx', index_col=0)  #     df_player = pd.read_excel(f'./p3_data_preparation/data/{country}/df_player_formated.xlsx', index_col=0)

    getting_to_know_data(df_match)
    getting_to_know_data(df_match_player)
    getting_to_know_data(df_player)

    # Verifico unicidad de registros segun campos id
    verificar_unicidad_registros(df_match)

    # Verifico consistencia en campos que relacionan entidades
    check_ids_in_both_dataframes(df_match, df_match_player)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de jugadores: verificar_relacion_entidades(df_player_part, df_match)

if __name__ == "__main__":
    prueba()