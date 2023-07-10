import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def convert_columns_to_int(df, df_etiquetas=None):
    """
    Convierte las variables categóricas de tipo string a numéricas utilizando LabelEncoder y guarda los valores
    originales y enteros correspondientes.

    :param df: DataFrame que contiene las variables a convertir. (DataFrame)
    :param df_etiquetas: DataFrame adicional con las etiquetas originales y enteros correspondientes.
                         Si se proporciona, se utilizará para la conversión en lugar de ajustar un nuevo LabelEncoder.
                         (DataFrame, opcional)
    :return: DataFrame con las variables convertidas y un DataFrame adicional con las etiquetas originales y
             enteros correspondientes.
    """
    if df_etiquetas is None:
        df_etiquetas = pd.DataFrame(columns=['variable', 'valor_orig', 'valor_int'])

    le = LabelEncoder()

    # Por variable string
    for col in df.select_dtypes(include=['object']).columns:
        print("Columna: ", col)

        if col not in df_etiquetas['variable'].unique():

            # Convierto columna a int
            df[col] = le.fit_transform(df[col])

            # Guardo etiquetas
            l_valor_orig = le.classes_
            l_valor_int = le.transform(l_valor_orig)

            # Guardo en dataframe
            for valor_orig, valor_int in zip(l_valor_orig, l_valor_int):
                df_etiquetas.loc[len(df_etiquetas)] = [col, valor_orig, valor_int]
    return df, df_etiquetas

def revert_columns_from_int(df, df_etiquetas, columns=None):
    """
    Convierte las variables numéricas a sus valores originales utilizando el DataFrame df_etiquetas.

    :param df: DataFrame que contiene las variables a revertir. (DataFrame)
    :param df_etiquetas: DataFrame que contiene las etiquetas originales y los valores enteros correspondientes. (DataFrame)
    :param columns: Lista de columnas a revertir. Si no se proporciona, se revertirán todas las columnas en df_etiquetas.
                    (list, opcional)
    :return: DataFrame con las variables revertidas a sus valores originales.
    """
    if columns is None:
        columns = df_etiquetas['variable'].unique()

    for col in columns:
        col_etiquetas = col if col != 'y_pred' else 'equipo_ganador'
        mapping = df_etiquetas.loc[df_etiquetas['variable'] == col_etiquetas].set_index('valor_int')['valor_orig']
        df[col] = df[col].map(mapping)

    return df

def prueba():
    # Levanto datasets
    pais = 'argentina'
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_partido.xlsx')
    df_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_jugadores.xlsx", index_col=0)

    # Entidad partido: fecha de string a datetime, posesion de str a float
    df_part['fecha'] = pd.to_datetime(df_part['fecha'], format='%d.%m.%Y %H:%M')  # df_part = convert_fecha_to_datetime(df_part, string_format='%d.%m.%Y %H:%M')
    df_part = convert_posesion_to_int(df_part)
    df_part['es_copa'] = df_part['es_copa'].replace(True, 1).replace(False, 0)

    # Entidad jugador: fecha de string a datetime y convierto valor de mercado en entero
    df_jug['fecha'] = pd.to_datetime(df_jug['fecha'], format='%b %d, %Y')  # df_jug = convert_fecha_to_datetime(df_jug, string_format='%b %d, %Y')
    df_jug = convert_valor_mercado_to_int(df_jug)

    df_part.to_excel('/Users/nachomondino/Desktop/df_part_formated.xlsx', index=False)
    df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_formated.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()