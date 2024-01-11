import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime

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

def convert_valor_mercado_to_int(df):
   """
    Transforma el valor de mercado de string a float.

    :param df: Dataframe con columna 'valor_mercado' cuyos valores son un string, por ejemplo, '€1.2M'.
    :return: Dataframe con la columna 'valor_mercado' interpretada como float, por ejemplo, 1.200.000.
    """
   d = {'M': 1000000, 'K': 1000}

   def convertir_valor_mercado(valor_mercado_str):

       # Si no se tiene el dato del valor de mercado
       if valor_mercado_str == "€0":
           return None

       # Si se tiene el dato del valor de mercado
       else:
           for elem in d.keys():
               if elem in valor_mercado_str:
                   valor_mercado_int = float(valor_mercado_str.replace("€", "").replace(elem, "")) * d[elem]
                   return valor_mercado_int
           return None

   df['valor_mercado'] = df['valor_mercado'].apply(convertir_valor_mercado)
   return df

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

        if col not in df_etiquetas['variable'].unique():

            # Quito NaN de la columna para evitar codificar el valor NaN
            df_sin_na = df.dropna(subset=[col])

            # Convierto columna a int
            df_sin_na[col] = le.fit_transform(df_sin_na[col])

            # Reemplazar los valores de la columna en los índices sin Nan
            df.loc[df_sin_na.index, col] = df_sin_na[col]
            df[col] = df[col].astype("float64")  # Convertir el dtype a int64

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
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_part.xlsx')
    df_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_jug.xlsx", index_col=0)

    # Entidad partido WhoScored: fecha, resultados de medio tiempo y final
    df_part['fecha'] = pd.to_datetime(df_part['fecha'] + ' ' + df_part['hora'], format='%a, %d-%b-%y %H:%M')
    df_part['fecha'] = df_part['fecha'] - datetime.timedelta(hours=4)  # Resto 4 horas a la columna 'fecha' para que este en horario argentino
    df_part[['ht_goles_loc', 'ht_goles_vis']] = df_part['ht_result'].str.split(' : ', expand=True)  # Separar ht_result en ht_goles_loc y ht_goles_vis
    df_part[['goles_loc', 'goles_vis']] = df_part['ft_result'].str.split(' : ', expand=True)  # Separar ft_result en goles_loc y goles_vis
    df_part = df_part.drop(['hora', 'ht_result', 'ft_result'], axis=1)

    # Entidad jugador: fecha
    df_jug['fecha_nac'] = pd.to_datetime(df_jug['fecha_nac'], format='%d-%m-%Y')

    # Exporto pruebas
    df_part.to_excel('/Users/nachomondino/Desktop/df_part_formated.xlsx', index=False)
    df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_formated.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()