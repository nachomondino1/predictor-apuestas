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
    print(df.describe(include=object))  # basic descriptive statistics for all columns

def main():
    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/flashscore/liga_argentina_historico.xlsx')

    # Describo dataset
    getting_to_know_data(df)

main()
