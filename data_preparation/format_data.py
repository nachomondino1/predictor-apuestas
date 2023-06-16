import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def convert_posesion_to_int(df):
    """
    Transformo posesion de string a float
    :param df: Dataframe. Con columnas 'posesion_loc' y 'posesion_vis' donde la posesion se interpreta como string. Por
    ejemplo '65%'.
    :return: Dataframe. Con columna 'fecha' interpretada como float. Por ejemplo, '0.65'
    """
    df['posesion_loc'] = df['posesion_loc'].apply(lambda x: int(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan)
    df['posesion_vis'] = df['posesion_vis'].apply(lambda x: int(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan)
    return df

def convert_fecha_to_datetime(df, string_format):
    """
    Transformo fecha de string a datetime
    :param df: Dataframe. Con columna 'fecha' interpretada como string
    :return: Dataframe. Con columna 'fecha' interpretada como datetime
    """
    df['fecha'] = pd.to_datetime(df['fecha'], format=string_format)
    return df

def convert_valor_mercado_to_int(df):
    """
    Transformo fecha de string a datetime
    :param df: Dataframe.
    :return: Dataframe.
    """
    d = {'M': 1000000, 'K': 1000}

    def convertir_valor_mercado(valor_mercado_str):
        if valor_mercado_str == "€0":
            return None
        else:
            for elem in d.keys():
                if elem in valor_mercado_str:
                    valor_mercado_int = float(valor_mercado_str.replace("€", "").replace(elem, "")) * d[elem]
                    return valor_mercado_int
            return None

    df['valor_mercado'] = df['valor_mercado'].apply(convertir_valor_mercado)
    return df

def convert_columns_to_int(df, df_etiquetas=None):  #Funciona?
    """
    Convierte las variables categóricas de tipo string a numéricas utilizando LabelEncoder y guarda los valores originales y enteros correspondientes.

    :param df: DataFrame que contiene las variables a convertir.
    :param pais: País para el cual se realiza la conversión.
    :return: DataFrame con las variables convertidas y un DataFrame adicional con las etiquetas originales y enteros correspondientes.
    """
    # Si aun no tengo un sistema de codificacion
    if df_etiquetas is None:

        # Convertir variables categóricas string a categóricas numéricas
        le = LabelEncoder()
        df_etiquetas = pd.DataFrame(columns=['variable', 'valor_orig', 'valor_int'])

        # Por variable string
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])

            l_valor_orig = le.classes_
            l_valor_int = le.transform(l_valor_orig)

            # Guardo en dataframe
            for valor_orig, valor_int in zip(l_valor_orig, l_valor_int):
                df_etiquetas.loc[len(df_etiquetas)] = [col, valor_orig, valor_int]

        return df, df_etiquetas

    # Si ya tengo un sistema de codificacion
    else:
        # Por variable a codificar
        for col in df_etiquetas['variable'].unique():

            l_valor_orig = df_etiquetas[df_etiquetas['variable'] == col]['valor_orig']
            l_valor_int = df_etiquetas[df_etiquetas['variable'] == col]['valor_int']

            # Guardo en dataframe
            for valor_orig, valor_int in zip(l_valor_orig, l_valor_int):
                df[col] = df[col].replace(valor_orig, valor_int)

        return df

def target_to_object(df, df_etiquetas):

    # Reemplazo codigos por etiquetas
    df_etiquetas_filt = df_etiquetas[df_etiquetas['variable'] == 'equipo_ganador']
    l_valor_orig = list(df_etiquetas_filt['valor_orig'])
    l_valor_int = list(df_etiquetas_filt['valor_int'])
    mapping = dict(zip(l_valor_int, l_valor_orig))
    df['y_pred_etiqueta'] = df['y_pred'].map(mapping)
    return df

def revert_columns_from_int(df, df_etiquetas, columns=None):  # Podria reemplazar target_to_object() pero no puedo usar columns = ['y_pred']
    """
    Convierte las variables numéricas a sus valores originales utilizando el DataFrame df_etiquetas.

    :param df: DataFrame que contiene las variables a revertir.
    :param df_etiquetas: DataFrame que contiene las etiquetas originales y los valores enteros correspondientes.
    :return: DataFrame con las variables revertidas a sus valores originales.
    """
    columns = df_etiquetas['variable'].unique() if columns is None else columns

    for col in columns:
        df_etiquetas_filt = df_etiquetas[df_etiquetas['variable'] == col]
        l_valor_orig = list(df_etiquetas_filt['valor_orig'])  # list() Para evitar TypeError: 'numpy.int64' object is not iterable
        l_valor_int = list(df_etiquetas_filt['valor_int'])
        mapping = dict(zip(l_valor_int, l_valor_orig))
        df[col] = df[col].map(mapping)
    return df

def prueba():
    # Levanto datasets
    df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_partido_argentina.xlsx')
    df_jug = pd.read_excel("/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_jugadores.xlsx", index_col=0)

    # Entidad partido: fecha de string a datetime, posesion de str a float
    df_part = convert_fecha_to_datetime(df_part, string_format='%d.%m.%Y %H:%M')
    df_part = convert_posesion_to_int(df_part)
    df_part['es_copa'] = df_part['es_copa'].replace(True, 1).replace(False, 0)

    # Entidad jugador: fecha de string a datetime y convierto valor de mercado en entero
    df_jug = convert_fecha_to_datetime(df_jug, string_format='%b %d, %Y')
    df_jug = convert_valor_mercado_to_int(df_jug)

    df_part.to_excel('/Users/nachomondino/Desktop/df_part_formated.xlsx', index=False)
    df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_formated.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()