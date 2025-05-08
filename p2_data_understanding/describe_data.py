# Importo librerias
import pandas as pd
from utils.set_up_logging import logger
import seaborn as sns
import matplotlib.pyplot as plt


def getting_to_know_data(df, verbose: int = 1):
    """
    Describe dataframe pasado como parametro
    :param df: Dataframe
    :return: funcion sin retorno
    """
    # Display de Dataframes
    if verbose >= 1:
        pd.set_option("display.precision", 2)  # mostrar maximo dos decimales
        if len(df.columns) < 50:
            pd.set_option("display.max.columns", None)  # para ver todas las columnas del df y no que las colapse

    # Data frame's dimensionality
    if verbose >= 1:
        logger.info(f"\nDataframe shape: {df.shape}")

    # Displaying Data Types
    if verbose >= 1:
        logger.info("\nDataframe info:")
        df.info()

    # See firsts 2 Dataframe's rows and statistics
    if verbose >= 2 and len(df.columns) > 0:
        print("\nPrimeras 2 filas del dataframe:")
        print(df.head(2))  # y .tail es para ver las ultimas filas

        # Showing Basics Statistics
        print("\nDataframe basic statistics:")
        print(df.describe(include='all'))  # basic descriptive statistics for all numeric columns

def verificar_unicidad_registros(df):

    logger.info(f"\nAnalisis de unicidad de registros...")
    indices_duplicados = df.index.duplicated()
    df_dup = df.index[indices_duplicados]

    if len(df_dup) > 0:
        logger.warning(f"\t El índice tiene valores duplicados.") # Indices duplicados: {list(indices_duplicados)}")
    # else:
    #     logger.info("\tEl índice no tiene valores duplicados.")

def scatter_plot(df: pd.DataFrame, name: str):
    sns.pairplot(df, diag_kind='kde')
    plt.savefig(f"images/{name}.png")
    plt.close()

def check_ids_in_both_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, column: str = None) -> None:
    """
    Verifica si todos los índices de df1 están presentes en df2.

    Args:
        df1 (pd.DataFrame): Primer DataFrame.
        df2 (pd.DataFrame): Segundo DataFrame.
        column (str, optional): Nombre de la columna en df2 para comparar con el índice de df1. Si es None, se comparan los índices de ambos DataFrames.

    Returns:
        None
    """
    reference = df2.index if column is None else df2[column]
    all_in_df2 = df1.index.isin(reference).all()

    if all_in_df2:
        logger.info("\tTodos los valores del índice de df1 están en el índice de df2.")
    else:
        logger.error("\tAl menos un valor del índice de df1 no está en el índice de df2.")

if __name__ == "__main__":
    country = "England"

    # Levanto datasets
    df_match = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match.xlsx')  #     df_match = pd.read_excel(f'./data/{country}/p3_data_preparation/df_match_formated.xlsx')
    df_match_player = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match_player.xlsx')
    df_player = pd.read_excel(f'./data/{country}/p2_data_understanding/df_player.xlsx', index_col=0)  #     df_player = pd.read_excel(f'./data/{country}/p3_data_preparation/df_player_formated.xlsx', index_col=0)

    getting_to_know_data(df_match)
    getting_to_know_data(df_match_player)
    getting_to_know_data(df_player)

    # Verifico unicidad de registros segun campos id
    verificar_unicidad_registros(df_match)

    # Verifico consistencia en campos que relacionan entidades
    check_ids_in_both_dataframes(df_match, df_match_player)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de jugadores: verificar_relacion_entidades(df_player_part, df_match)
