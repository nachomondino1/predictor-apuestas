import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
import warnings

def convert_posesion_to_int(df):
    """
    Transformo posesion de string a float
    :param df: Dataframe. Con columnas 'posesion_loc' y 'posesion_vis' donde la posesion se interpreta como string. Por
    ejemplo '65%'.
    :return: Dataframe. Con columna 'fecha' interpretada como float. Por ejemplo, '0.65'
    """
    posesion = "ball_possession"
    l_tit = ['home', 'away']

    for tit in l_tit:
        df[f'{posesion}_{tit}'] = df[f'{posesion}_{tit}'].apply(
            lambda x: int(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan)
        
        # Verifica si todos los elementos de la columna son de tipo float
        if not all(isinstance(value, (float, np.floating)) for value in df[f'{posesion}_{tit}']):
            # Si no todos los elementos son de tipo float, raise una advertencia
            warnings.warn(f"Not all elements in the column '{posesion}' are float.")
    
    return df

def convert_value_to_int(df):
    """
    Transforma el valor de mercado de string a float.

    :param df: Dataframe con columna 'value' cuyos valores son un string, por ejemplo, '€1.2M'.
    :return: Dataframe con la columna 'value' interpretada como float, por ejemplo, 1.200.000.
    """
    def convertir_value(value_str):
        d = {'M': 1000000, 'K': 1000}

        # Si no se tiene el dato del valor de mercado
        if value_str == "€0":
            return None

        # Si se tiene el dato del valor de mercado
        else:
            for elem in d.keys():
                if elem in value_str:
                    value_int = float(value_str.replace("€", "").replace(elem, "")) * d[elem]
                    return value_int
            return None
    
    # Reemplazo strings por numbers
    df['value'] = df['value'].apply(convertir_value)

    # Verifica si todos los elementos de la columna son de tipo float
    if not all(isinstance(value, (float, np.floating)) for value in df['value']):
        # Si no todos los elementos son de tipo float, raise una advertencia
        warnings.warn(f"Not all elements in the column 'value' are float.")

    return df

def convert_goles_to_int(df):
    """
    Elimina las filas que hacen que goals_home y goals_away no sea integer como debe. Puede ser por NaN o por string "-".
    :param df:
    :return:
    """
    l_filas_a_borrar = []

    # Por partido
    for i, row in df.iterrows():
        try:
            int(row['goals_home'])
            int(row['goals_away'])
        # Si los goles no pueden ser transofrmados a integer
        except:
            # Guardo indice para eliminar la fila
            l_filas_a_borrar.append(i)

    # Elimino filas del dataframe
    print(f"Cantidad de partidos eliminados por no tener goles integer: {len(l_filas_a_borrar)/len(df)*100:.1f}%")
    df = df.drop(l_filas_a_borrar)

    # Convierto columnas goles a integer
    df['goals_home'] = df['goals_home'].astype(int)
    df['goals_away'] = df['goals_away'].astype(int)
    print(f"Verificacion de dtype de goles (deberia ser int):", df[f"goals_home"].dtype, df[f"goals_away"].dtype)
    return df

def convert_capacity_to_int(df):

    l_columns = ['capacity', 'attendance']

    for col in l_columns:

         # Comprobar si la columna contiene valores de tipo cadena (string)
        if df[col].dtype == 'object':
            # Reemplazar los espacios en blanco en los valores de la columna
            df[col] = df[col].str.replace(' ', '')

            # Convertir la columna al tipo de datos correcto (entero)
            df[col] = df[col].astype(float) # Pues si tiene nan, es float.
        else:
            print(f"Fallo la conversion de la columna {col} a float")
    return df

def convert_columns_to_int(df, df_etiquetas=None):
    """
    Convierte las variables string a numéricas.

    :param df: DataFrame que contiene las variables a convertir. (DataFrame)
    :param df_etiquetas: DataFrame adicional con las etiquetas originales y enteros correspondientes. Si se proporciona, se utilizará 
    para la conversión en lugar de ajustar un nuevo LabelEncoder.(DataFrame, opcional)
    :return: DataFrame con las variables convertidas y un DataFrame adicional con las etiquetas originales y enteros correspondientes.
    """
    # Definicion de variables
    le = LabelEncoder()

    # Obtener columnas de tipo objeto
    l_columnas_a_codificar = df.select_dtypes(include=['object']).columns
    print(f"Columnas str a convertir a int: {l_columnas_a_codificar}")

    # Codifico variables string en numericas
    if df_etiquetas is None:
        df_etiquetas = pd.DataFrame(columns=['variable', 'str_value', 'int_value'])

        # Por variable string
        for col in l_columnas_a_codificar:

            # Quito NaN de la columna para evitar codificar el valor NaN
            df_sin_na = df.dropna(subset=[col])

            # Convierto columna a int
            df_sin_na[col] = le.fit_transform(df_sin_na[col].astype(str)) # agregue el "as_type(str)" porque tiraba error sino desde que uso vsc en vez Pycharm

            # Reemplazar los valores de la columna en los índices sin Nan
            df.loc[df_sin_na.index, col] = df_sin_na[col]
            df[col] = df[col].astype("float64")  # Convertir el dtype a int64

            # Guardo etiquetas
            l_valor_str = le.classes_
            l_valor_int = le.transform(l_valor_str)

            # Guardo string y su equivalente numerico
            df_etiquetas_col = pd.DataFrame({'variable': col, 'str_value': l_valor_str, 'int_value': l_valor_int})
            df_etiquetas = pd.concat([df_etiquetas, df_etiquetas_col], axis=0)
            # print(df_etiquetas)
            
    # Por fila
    for i, row in df_etiquetas.iterrows():

        # Obtener el nombre de la columna a la que se le realizará el reemplazo
        nombre_columna = row['variable']

        # Reemplazar valores en la columna específica
        df.loc[:, nombre_columna] = df[nombre_columna].replace(row['str_value'], row['int_value'])
        
    return df, df_etiquetas

def convert_pred_str_to_int(df, name_var_str, name_var_int, df_etiquetas_y): # Esto funciona bien
    """
    Obtengo las predicciones segun la casa de apuestas
    """
    # Convertir de str a num
    for idx, row in df_etiquetas_y.iterrows():
        # print(f"{row['str_value']} --> {row['int_value']}")
        
        # Obtengo indices de partidos con determinado resultado
        idx_to_change = df[df[name_var_str] == row['str_value']].index

        # Reemplazar valores en la columna específica
        df.loc[idx_to_change, name_var_int] = row['int_value']

    return df

def convert_pred_int_to_str(df, name_var_int, name_var_str, df_etiquetas_y): # Esto funciona bien
    """
    Obtengo las predicciones segun la casa de apuestas
    """
    # Convertir de str a num
    for idx, row in df_etiquetas_y.iterrows():
        # print(f"{row['int_value']} --> {row['str_value']}")
        
        # Obtengo indices de partidos con determinado resultado
        idx_to_change = df[df[name_var_int] == row['int_value']].index

        # Reemplazar valores en la columna específica
        df.loc[idx_to_change, name_var_str] = row['str_value']

    return df

def prueba():
    # Levanto datasets
    country = 'argentina'
    df_match = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{country}/df_match.xlsx')
    df_player = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{country}/df_player.xlsx", index_col=0)

    # Entidad partido WhoScored: fecha, resultados de medio tiempo y final
    df_match['fecha'] = pd.to_datetime(df_match['fecha'] + ' ' + df_match['hora'], format='%a, %d-%b-%y %H:%M')
    # df_match['fecha'] = df_match['fecha'] - datetime.timedelta(hours=4)  # Resto 4 horas a la columna 'fecha' para que este en horario argentino
    df_match[['ht_goals_home', 'ht_goals_away']] = df_match['ht_result'].str.split(' : ', expand=True)  # Separar ht_result en ht_goals_home y ht_goals_away
    df_match[['goals_home', 'goals_away']] = df_match['ft_result'].str.split(' : ', expand=True)  # Separar ft_result en goals_home y goals_away
    df_match = df_match.drop(['hora', 'ht_result', 'ft_result'], axis=1)

    # Entidad jugador: fecha
    df_player['fecha_nac'] = pd.to_datetime(df_player['fecha_nac'], format='%d-%m-%Y')

    # Exporto pruebas
    df_match.to_excel('/Users/nachomondino/Desktop/df_match_formated.xlsx', index=False)
    df_player.to_excel('/Users/nachomondino/Desktop/df_player_formated.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()