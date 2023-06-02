import pandas as pd
import numpy as np

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

# prueba()