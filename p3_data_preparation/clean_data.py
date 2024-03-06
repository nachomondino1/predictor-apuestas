import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from p4_modeling.build_model import select_best_hiperparameters
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings

# Importo librerias
import string
import requests

class TextPreparation:
    """Techniques to prepare text"""

    def __init__(self):  # no defino df como atributo puesto que es muy importante que el usuario reciba el df de cada funcion.
        pass

    def to_lower(self, df, columns):
        """Convierte a minúscula los textos"""
        for col in columns:
            df[col] = df[col].str.lower()
        return df

    def delete_accent(self, df, columns):
        """Remueve acentos de los textos"""
        d = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'}
        for col in columns:
            df[col] = df[col].replace(d, regex=True)
        return df

    def delete_special_characters(self, df, columns):
        """Remueve caracteres especiales de los textos"""
        d = {'ã': 'a', 'â': 'a', 'ä': 'a', 'ê': 'e', 'ë': 'e', 'î': 'i', 'ï': 'i', 'ô': 'o', 'ö': 'o', 'ø': 'o',
             'û': 'u', 'ü': 'u', 'ñ': 'n', 'č': 'c', 'ć': 'c', 'ğ': 'g', 'ß': 'ss', 'ń': 'n', 'š': 's'}
        for col in columns:
            df[col] = df[col].replace(d, regex=True)
        return df

    def delete_punctuation(self, df, columns):
        """Elimina signos de puntuación de los textos"""
        for col in columns:
            df[col] = df[col].str.replace(f'[{string.punctuation}]', ' ', regex=True)
        return df

    def tokenize(self, df, columns):
        """Tokeniza los textos"""
        for col in columns:
            df[col] = df[col].str.split()
        return df

    def stop_word_removal(self, df, columns):
        """Elimina palabras vacías de los textos"""
        url = 'https://raw.githubusercontent.com/7PartidasDigital/AnaText/master/datos/diccionarios/vacias.txt'
        palabras_vacias = pd.read_csv(url, squeeze=True)
        for col in columns:
            df[col] = df[col].apply(lambda x: [word for word in x if word not in palabras_vacias])
        return df

    def download_stop_word_removal_file(self):
        """Descarga el archivo de palabras vacías"""
        url = 'https://raw.githubusercontent.com/7PartidasDigital/AnaText/master/datos/diccionarios/vacias.txt'
        r = requests.get(url, allow_redirects=True)
        open('vacias.txt', 'wb').write(r.content)

    def stemming(self, df, columns):
        # from nltk.stem import SnowballStemmer

        """Aplica stemming a los textos"""
        # spanish_stemmer = SnowballStemmer('spanish')
        # for col in columns:
        #     df[col] = df[col].apply(lambda x: [spanish_stemmer.stem(word) for word in x])
        # return df
        pass

def prepare_text_columns(df, l_cols_to_process=[], l_col_to_except=[]):
    '''
    Prepara el texto de las columns que contengan strings.
    :param df: Dataframe.
     :param l_cols_to_process: Lista. Columns del tipo object a procesar.
    :param l_col_to_except: Lista. Columns del tipo object que omitir en el procesamiento.
    :return: Dataframe con columns que contienen strings ya preparados para ser analizados
    '''
    # Si l_cols_to_process está vacía, procesar todas las columns de texto
    if not l_cols_to_process:
        l_cols_to_process = df.select_dtypes(include='object').columns.tolist()

    # Filtrar las columns que se procesarán y que no están en la lista de excepciones
    l_text_columns = [col for col in l_cols_to_process if col not in l_col_to_except]
    print("\nColumns tipo object a preparar:", l_text_columns)

    # Creo objeto de la clase
    tp = TextPreparation()
    df = tp.to_lower(df, columns=l_text_columns)
    df = tp.delete_accent(df, columns=l_text_columns)
    df = tp.delete_special_characters(df, columns=l_text_columns)
    df = tp.delete_punctuation(df, columns=l_text_columns)
    return df

def clean_teams_names(df):
    """
    Limpia y cambia el nombre de algunos equipos en el dataframe.
    :param df: Dataframe. Unidad de analisis: partido. Columns: al menos "team_home" y "team_away"
    :return: Dataframe. El pasado por parametro con nombres de equipos modificados y limpios
    """
    # Limpio string 'Vencedor' en el nombre de algunos equipos.
    d_sub_strings_adic = {"winner": '', "advancing to next round": ''}  # Tengo que tener cuidado, reemplazo strings... pueden ser substring y cambiarlo sin querer hacerlo.
    df['team_name'] = df['team_name'].replace(d_sub_strings_adic, regex=True).str.strip()
    # df['team_home'] = df['team_home'].replace(d_sub_strings_adic, regex=True).str.strip()
    # df['team_away'] = df['team_away'].replace(d_sub_strings_adic, regex=True).str.strip()
    return df

def fill_nan_values(X, l_columns_to_fill, fill_type: str = "mode"): 
    """
    Relleno NaN values en un Dataframe.
    1) dropna teniendo en cuenta solo las columnas con menos nan + 2) imput (o fillna) solo de las columnas con mayor cant de nan  

    :param X: (Dataframe)
    :param y: (Dataframe)
    :param fill_type: Tipo de relleno de datos como mode o ml. (String)
    :param percentil_nan: A mayor valor, mas alto el porc_nan_max_col y, por ende, menos columnas son consideradas con mucho nan (es decir, menos relleno de datos).
    :return: (Dataframe)
    """
    X_filled = X.copy()
    
    # Por columna a rellenar
    for col in l_columns_to_fill:
        # print(f"Columna a rellenar: {col}")

        # OPCION 1: Llenar los valores faltantes con el valor más frecuente en cada columna
        if fill_type == "mode":  #     raise KeyError(key) from err --> KeyError: 0
            X_filled[col].fillna(X[col].mode()[0], inplace=True)

        # OPCION 2: Llenar los valores faltantes con ML
        elif fill_type == "ml":

            # Dividir el dataframe en conjunto de entrenamiento, validacion y prueba
            ## Separo test de train y val puesto que test tendra los NaN values para la columna
            X_train_val = X.loc[X[col].notnull()]  # df con columna!=nan # e.g. (2728, 11)
            X_train_val = X_train_val.drop(columns=l_columns_to_fill) 
            y_train_val = X.loc[X_train_val.index, col]  # y_train_val = X.loc[X[col].notnull(), col]  # Solo la columna donde columna!=nan # e.g. (2728,)

            ## Dejo en X_test los registros donde la columna es nan    
            X_test = X.loc[X[col].isnull()]
            X_test = X_test.drop(columns=l_columns_to_fill)  # e.g. (378, 11)

            ## Separo en train y val
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, shuffle=True)

            # Selecciono los mejores hiperparametros usando el set de validacion
            model = select_best_hiperparameters(RandomForestRegressor(), X_val, y_val, k=3)

            # Entrenar el modelo con los datos de entrenamiento
            model.fit(X_train, y_train)

            # Predecir los valores faltantes
            if len(X_test) > 0:
                predicted_values = model.predict(X_test)

                # Rellenar los valores faltantes en el dataframe
                predicted_values_index = X_test.index
                X_filled.loc[predicted_values_index, col] = predicted_values  # Creo que funciona
            else:
                text = f"En la columna {col} no hay nan values para rellenar. X_test no tiene registros a los cuales predecir. "
                warnings.warn(text)
        else:
            text = f"No se rellenaron los datos puesto que el tipo='{fill_type}' no es una opcion. Las opciones son 'mode' y 'ml'."
            warnings.warn(text)
    return X_filled

def replace_nan_with_zero(df, col1, col2):  # Esto solo para las columnas n_player_miss_home y n_player_miss_away. Si un equipo no tiene jug asusentes pero el otro si, entonces que reemplece nan por 0 (asi puedo restar home y away evitando el nan puesto que 7 - nan = nan)
    """
    Replace NaN values with 0 if one of the variables has an integer value and the other is NaN.
    If both variables are NaN, do not replace any values.
    If both variables take integer values, do not replace any values.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the columns.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.

    Returns:
    DataFrame: The DataFrame with NaN values replaced by 0 according to the specified conditions.
    """
    # Replace NaN with 0 if one variable has an integer value and the other is NaN
    condition_1 = df[col1].notnull() & df[col2].isnull()
    condition_2 = df[col2].notnull() & df[col1].isnull()

    df[col1] = np.where(condition_2, 0, df[col1])
    df[col2] = np.where(condition_1, 0, df[col2])
    return df

def delete_rows_nan(df: pd.DataFrame, porc_nan_max: float, _print: bool = False):
    """
    Elimina las rows de un DataFrame que contienen un percentage alto de valores NaN.

    :param df: DataFrame de entrada. (DataFrame)
    :param porc_nan_max: Percentage maximo tolerado de NaN values en una fila (Float) [0-1]
    :return: DataFrame resultante después de delete las rows con valores NaN. (DataFrame)
    """
    # Elimino rows segun umbral
    df_nan_rows = df.isnull().mean(axis=1)  # Calcula el percentage de valores NaN en cada fila
    rows_to_delete = df_nan_rows[df_nan_rows > porc_nan_max].index  # Obtiene las rows que superan el umbral  # Esta bien > pues sino el 0 borra todas...
   
    # Elimino rows segun umbral
    df_filtrado = df.drop(rows_to_delete)  # Elimina las rows con valores NaN
    if _print:
        print(f"De las {len(df)} rows, se delete {len(rows_to_delete)/len(df)*100:.0f}%, quedan {len(df) - len(rows_to_delete)} rows.")
    return df_filtrado

def delete_columns_nan(df: pd.DataFrame, porc_nan_max: float, _print: bool = False):
    """
    Elimina las columns de un DataFrame que contienen un percentage alto de valores NaN.

    :param df: DataFrame de entrada. (DataFrame)
    :param porc_nan_max: Percentage maximo tolerado de NaN values en una columna. (Float) [0-1]
    :return: DataFrame resultante después de delete las columns con valores NaN. (DataFrame)
    """
    # Calcula la proporción de NaN en cada columna
    df_nan_col = df.isna().mean()

    # Identifica las columns con una proporción de NaN mayor al umbral
    columns_delete = df_nan_col[df_nan_col > porc_nan_max].index 

    # Elimina las columns identificadas del DataFrame
    df_sin_nan = df.drop(columns_delete, axis=1)
    if _print:
        print(f"De las {len(df.columns)} columns, se eliminaron {len(list(columns_delete))} por tener un % NaN mayor a thr_nan_col={porc_nan_max*100:.0f}%: {list(columns_delete)}")
    return df_sin_nan

def determine_columns_to_fill(df, percentil_nan, _print: bool = False): # Funciona perfecto
    """
    Determinar que columnas del dataframe son consideradas con mucho nan y cuales con poco nan
    """
    # Calcula porcentaje de nan para cada columna
    df_nan = df.isna().mean()

    # Determino porc_nan_max_col segun percentil 
    porc_nan_max_col = np.percentile(df_nan.sort_values(), percentil_nan) # Ordena el DataFrame df_porc_nan antes de tomar el percentil (no hace falta pero bueno, para mas seguridad)

    # Diferencio entre columnas con mucho nan y poco nan
    l_columns_con_mucho_nan = df.columns[df_nan > porc_nan_max_col].tolist() 
    l_columns_con_poco_nan = df.columns.difference(l_columns_con_mucho_nan)
    if _print:
        print(f"{len(l_columns_con_mucho_nan)} de las {len(df.columns)} columnas son consideradas con mucho NaN (+{porc_nan_max_col*100:.0f}% de NaN): {l_columns_con_mucho_nan}")
    return l_columns_con_poco_nan, l_columns_con_mucho_nan

def drop_columns_until_drop_nan_not_empty(df, porc_nan_max: float = 0.95, n_reg_min: int = 100, _print: bool = True):
    """
    Elimina columnas con mucho nan hasta que el dataframe tenga al menos un registro para poder entrenar el modelo
    Es clave hacerlo en nan para no eliminar columnas en el modeling. 
    """
    warnings.simplefilter("always")

    # Elimino filas con al menos un nan (tal como lo haria en Modeling)
    df_drop_na = delete_rows_nan(df, 0, _print=False)
    if _print:
        print("\n Cantidad de filas df: ", df_drop_na.shape[0])

    # Si quedan menos registros que n_reg_min
    if len(df_drop_na) <= n_reg_min:
        # text = "Cuidado, el dataframe podria generar error en Modeling por no quedar registros con los cuales entrenar el modelo"
        # warnings.warn(text, UserWarning)

        porc_nan_max = porc_nan_max - 0.05

        # Elimino columnas con mucho nan
        df = delete_columns_nan(df, porc_nan_max)
        if _print:
            print("porc_nan_max: ", porc_nan_max)
            print("Columnas restantes en df", df.shape[1])

        # Vuelvo a verificar si quedan filas nan          
        df = drop_columns_until_drop_nan_not_empty(df, porc_nan_max, n_reg_min=n_reg_min)
    return df

def prueba():
    country = 'England'

    # Levanto dataset
    df = pd.read_excel(f'p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
    print(df.head(2))

    # Eliminacion de NaN values
    # largo_inicial = len(df)
    # df = df.dropna(subset=['dif_prom_ult_part_dif_remates'], how='any')
    # print(f"Se eliminó el {(largo_inicial - len(df)) / largo_inicial * 100:.0f}% de rows, quedan {len(df)} rows.")

    # Preparacion de texto
    df = prepare_text_columns(df, l_cols_to_process=['team_home', 'team_away'])  # Preparacion texto para facilitar construccion de datos bassado en equipos
    df = clean_teams_names(df)  # Eliminar strings adicionales en names de equipos
    print(df.head(2))

    # Verificar que no haya outliers
    # ...

    # Normalizo columns con valores mas grandes para evitar ValueError: Solver produced non-finite parameter weights. The input data may contain large values and need to be preprocessed.
    # scaler = StandardScaler()  # Crea un objeto StandardScaler
    # df['dif_sum_min_titular'] = scaler.fit_transform(df['dif_sum_min_titular'].values.reshape(-1, 1))
    # df['dif_sum_min_suplente'] = scaler.fit_transform(df['dif_sum_min_suplente'].values.reshape(-1, 1))
    
    df.to_excel('/Users/nachomondino/Desktop/df_cleaned_prueba.xlsx', index=True)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()