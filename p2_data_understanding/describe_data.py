# Importo librerias
import pandas as pd


def getting_to_know_data(df):
    """
    Describe dataframe pasado como parametro
    :param df: Dataframe
    :return: funcion sin retorno
    """
    print("DESCRIPCION DE DATAFRAME:".center(120))

    # Data frame's dimensionality
    print("\nDataframe shape: ", df.shape)

    # See firsts 5 Dataframe's rows
    print("\nPrimeras 5 filas del dataframe:")
    pd.set_option("display.max.columns", None)  # para ver todas las columnas del df y no que las colapse
    pd.set_option("display.precision", 2)  # mostrar maximo dos decimales
    print(df.head())  # y .tail es para ver las ultimas filas

    # Displaying Data Types
    print("\nDataframe info:")
    df.info()

    # Showing Basics Statistics
    print("\nDataframe basic statistics:")
    print(df.describe(include='all'))  # basic descriptive statistics for all numeric columns


def verificar_unicidad_registros(df, columns_id):

    df_duplicados = df.duplicated(subset=columns_id)
    hay_duplicados = df_duplicados.any()

    if hay_duplicados:
        print(f"Lo/s campo/s {columns_id} no hace cada registro unico. Hay solo {len(df_duplicados) - df_duplicados.sum()} unicos sobre {len(df)} posibles.")
    else:
        print(f"Se verifica que todo registro es unico segun {columns_id}")


def verificar_relacion_entidades(df_part, df_jug_part): # notas
    """
    Verifica unicidad de ids y consistencia entre dataframes
    :param df_alt:
    :param df_opi:
    :return:
    """
    # Obtengo cantidad de ids unicos en cada dataframe
    ids = list(df_part['id_part'])  # ids en dataframe altenativas
    ids_con_opi = df_jug_part['id_part'].unique()  # ids en dataframe opiniones

    # Verifico que ids con opinion tengan id en Dataframe alternativas
    i = 0
    for id_con_opi in ids_con_opi:
        if id_con_opi not in ids:
            i += 1
    print(f"Hay {i} ids que estan en df_part_jug y no en df_part")

def prueba():

    # Levanto datasets
    df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/argentina/data_seg/df_part_Liga_Profesional_2005.xlsx')
    df_jug_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/argentina/data_seg/df_jug_part_Liga_Profesional_2005.xlsx')

    getting_to_know_data(df_part)
    getting_to_know_data(df_jug_part)

    # Verifico unicidad de registros segun campos id
    verificar_unicidad_registros(df_part, columns_id='id_part')
    verificar_unicidad_registros(df_jug_part, columns_id=['id_jug', 'id_part'])

    # Verifico consistencia en campos que relacionan entidades
    verificar_relacion_entidades(df_part, df_jug_part)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de jugadores: verificar_relacion_entidades(df_jug_part, df_part)


    '''
    print(" c) Analisis de cantidad de opiniones por valor de cada campo especifico ".center(120))
    n_opi_by_value(df_alt, df_opi)
    '''

if __name__ == "__main__":
    prueba()