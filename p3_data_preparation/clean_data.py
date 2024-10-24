import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from set_up_logging import logger
import string
import requests

# 1) Preparacion de columnas string
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
            try:
                df[col] = df[col].replace(d, regex=True)
            except ValueError:
                pass
        return df

    def delete_special_characters(self, df, columns):
        """Remueve caracteres especiales de los textos"""
        d = {'ã': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ê': 'e', 'ë': 'e', 'è': 'e','î': 'i', 'ï': 'i', 'ì': 'i', 'ô': 'o', 'ö': 'o', 'ø': 'o', 
            'ó': 'o', 'û': 'u', 'ü': 'u', 'ù': 'u', 'ñ': 'n', 'č': 'c', 'ć': 'c', 'ğ': 'g', 'ß': 'ss', 'ń': 'n', 'š': 's'}
        for col in columns:
            try:
                df[col] = df[col].replace(d, regex=True)
            except ValueError:
                pass
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

def prepare_text_columns(df: pd.DataFrame, l_cols_to_process: list = None):
    """
    Prepara el texto de las columns que contengan strings.

    # Parameters:
        df: Dataframe con columnas string (DataFrame)
        l_cols_to_process: Columns del tipo object a procesar. (list)
    
    # Returns:
        Dataframe pasado como parametro con columns strings ya preparadas para ser analizadas. (DataFrame)
    """
    # Si l_cols_to_process está vacía, procesar todas las columns de texto
    if l_cols_to_process is None:
        l_cols_to_process = df.select_dtypes(include='object').columns.tolist()
    print("\nColumns tipo object a preparar:", l_cols_to_process)

    # Creo objeto de la clase
    tp = TextPreparation()
    df = tp.to_lower(df, columns=l_cols_to_process)
    df = tp.delete_accent(df, columns=l_cols_to_process)
    df = tp.delete_special_characters(df, columns=l_cols_to_process)
    df = tp.delete_punctuation(df, columns=l_cols_to_process)
    return df

def clean_teams_names(df):
    """
    Limpia y cambia el nombre de algunos equipos en el dataframe.
    :param df: Dataframe. Unidad de analisis: partido. Columns: al menos "team_home" y "team_away"
    :return: Dataframe. El pasado por parametro con nombres de equipos modificados y limpios
    """
    # Limpio string 'Vencedor' en el nombre de algunos equipos.
    d_sub_strings_adic = {"winner": '', "advancing to next round": ''}  # Tengo que tener cuidado, reemplazo strings... pueden ser substring y cambiarlo sin querer hacerlo.
    df['team_home'] = df['team_home'].replace(d_sub_strings_adic, regex=True).str.strip()
    df['team_away'] = df['team_away'].replace(d_sub_strings_adic, regex=True).str.strip()
    return df

# 2) Eliminacion de columnas no relevantes
def delete_not_relevant_stats(df, stats_columns, relevant_stats_columns):
    """
    Elimina las columnas especificadas de un DataFrame.

    Args:
        df (DataFrame): El DataFrame del que se eliminarán las columnas.
        columns_to_drop (list): Una lista de listas, donde cada sublista contiene los nombres de las columnas a eliminar.

    Returns:
        DataFrame: El DataFrame con las columnas especificadas eliminadas.
    """  
    # Determino cuales son las estadisticas a eliminar
    stats_to_drop = list(set(stats_columns).difference(set(relevant_stats_columns)))
    n_cols_inic = len(df.columns)

    # Eliminar las columnas especificadas
    for stat in stats_to_drop:

        l_stat = [f'{stat}_home', f'{stat}_away']
        columns_to_remove = [col for col in l_stat if col in df.columns]
        df = df.drop(columns_to_remove, axis=1)

    n_cols_fin = len(df.columns)
    n_removed_cols = n_cols_inic - n_cols_fin
    if n_removed_cols > 0:
        logger.warning(f"Eliminacion de columnas irrelevantes: {n_cols_inic} --> {n_cols_fin}")

    return df

# 3) Tratamiento de NaN values
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
        logger.warning(f"De las {len(df)} rows, se delete {len(rows_to_delete)/len(df)*100:.0f}%, quedan {len(df) - len(rows_to_delete)} rows.")
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
        logger.warning(f"De las {len(df.columns)} columns, se eliminaron {len(list(columns_delete))} por tener un % NaN mayor a thr_nan_col={porc_nan_max*100:.0f}%: {list(columns_delete)}")
    return df_sin_nan

def determine_columns_to_fill(df, percentil_nan, porc_max: float = 0.3, _print: bool = False):
    """
    Determinar que columnas del dataframe son consideradas con mucho nan y cuales con poco nan
    # Parameters
        df: Dataframe a rellenar NaN values.
        percentil_nan: Percentil para determinar porcentaje de nan umbral para decidir si una columna se rellenara a no.
        porc_max: Porcentaje de nan umbral para decidir si una columna se rellenara a no.
    # Return
        df: Dataframe habiendo rellenado NaN values de las columnas consideradas con mucho NaN.
    """
    # Calcula porcentaje de nan para cada columna
    df_nan = df.isna().mean()

    # Determino porc_nan_max_col segun percentil 
    perc_max = np.percentile(df_nan.sort_values(), percentil_nan) # Ordena el DataFrame df_porc_nan antes de tomar el percentil (no hace falta pero bueno, para mas seguridad)

    # Determino porcentaje min de nan (En caso que el percentil sea muy grande, uso el porcentaje fijo mas pequeño de manera de rellenar mas)
    porc_nan_max_col = min(porc_max, perc_max)
    if _print:
        df_nan.to_excel('/Users/nachomondino/Desktop/df_nan.xlsx')
        logger.info(f"Porcentaje min de nan para considerar con mucho nan: {perc_max}")
        logger.info(f"Porcentaje min de nan para considerar con mucho nan: {porc_max}")
        logger.info(f"Porcentaje a considerar: {porc_nan_max_col}")

    # Diferencio entre columnas con mucho nan y poco nan
    l_columns_con_mucho_nan = df.columns[df_nan > porc_nan_max_col].tolist() 
    l_columns_con_poco_nan = df.columns.difference(l_columns_con_mucho_nan)
    if _print:
        print(f"{len(l_columns_con_mucho_nan)} de las {len(df.columns)} columnas son consideradas con mucho NaN (+{porc_nan_max_col*100:.0f}% de NaN): {l_columns_con_mucho_nan}")

    return l_columns_con_poco_nan, l_columns_con_mucho_nan

def drop_columns_until_drop_na_min_rows(df, porc_nan_max: float = 0.95, n_reg_min: int = 100, _print: bool = False):
    """
    Elimina columnas con mucho nan hasta que el dataframe tenga al menos un registro para poder entrenar el modelo
    Es clave hacerlo en nan para no eliminar columnas en el modeling. 
    """
    # Elimino filas con al menos un nan (tal como lo haria en Modeling)
    df_drop_na = delete_rows_nan(df, 0, _print=False)
    if _print:
        print("\n Cantidad de filas df: ", df_drop_na.shape[0])

    # Si quedan menos registros que n_reg_min
    if len(df_drop_na) <= n_reg_min:

        porc_nan_max = porc_nan_max - 0.05

        # Elimino columnas con mucho nan
        df = delete_columns_nan(df, porc_nan_max)
        if _print:
            print("porc_nan_max: ", porc_nan_max)
            print("Columnas restantes en df", df.shape[1])

        # Vuelvo a verificar si quedan filas nan          
        df = drop_columns_until_drop_na_min_rows(df, porc_nan_max, n_reg_min=n_reg_min)

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
    # Importar solo cuando es necesario
    from p4_modeling.build_model import select_best_hiperparameters
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from tqdm import tqdm

    X_filled = X.copy()

    # Progress bar
    logger.warning(f"Se rellenaran {len(l_columns_to_fill)} columnas.")
    progress_bar = tqdm(total=len(l_columns_to_fill), ncols=80)  # Inicializo barra de progreso

    # Por columna a rellenar
    for col in l_columns_to_fill:
        # logger.info(f"Columna a rellenar: {col}")

        # OPCION 1: Llenar los valores faltantes con el valor más frecuente en cada columna
        if fill_type == "mode":  #     raise KeyError(key) from err --> KeyError: 0

            mode_value = X[col].mode()[0]
            X_filled[col] = X_filled[col].fillna(mode_value)

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

            # Despues de definir columns con mucho NaN, elimino registros con nan en las otras columnas y puede que una columna con mucho nan ya no tenga nan.
            if len(X_test) > 0:
                ## Separo en train y val
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, shuffle=True)

                # Selecciono los mejores hiperparametros usando el set de validacion
                model = select_best_hiperparameters(RandomForestRegressor(), X_val, y_val, k=3, n_iter=30, _print=False) #  Con 50, tarda 1 min/columna. Con 25, tarda 0.3 min/columna.

                # Entrenar el modelo con los datos de entrenamiento
                model.fit(X_train, y_train)

                # Predecir los valores faltantes
                predicted_values = model.predict(X_test)

                # Rellenar los valores faltantes en el dataframe
                predicted_values_index = X_test.index
                X_filled.loc[predicted_values_index, col] = predicted_values  # Creo que funciona

            else:
                logger.error(f"En la columna {col} no hay nan values para rellenar. X_test no tiene registros a los cuales predecir. No hacer nada.")
        else:
            logger.error(f"No se rellenaron los datos puesto que el tipo='{fill_type}' no es una opcion. Las opciones son 'mode' y 'ml'.")

        progress_bar.update(1)

    progress_bar.close()
    return X_filled

def drop_and_fill_nan_values(X, percentil_nan: int = 75, fill_type: str = "mode"):
    """
    Elimino columnas con muy alto porcentaje de NaN values. Luego, elimino filas con NaN considerando solo las columnas con menos % de NaN values. 
    En las filas restantes, relleno las columnas con mucho NaN con la moda.
    """
    print("\tReemplazo y remuevo NaN values (uso ML y no puede tener input NaN)... ")
    # Determino las columns con mucho NaN (mas de nan_threshold%)
    l_columns_poco_nan, l_columns_mucho_nan = determine_columns_to_fill(X, percentil_nan=percentil_nan)

    # Elimino registros NaN en las columns con bajo % de NaN 
    X = X.dropna(subset=l_columns_poco_nan)  # df = clean_data.delete_rows_nan(X_sin_col_mucho_nan, porc_nan_max=0)
    # print(f"De las {len(X_sin_col_mucho_nan)} filas, se han eliminado {len(X_sin_col_mucho_nan)-len(X)} por tener al menos un Nan value. Quedan {len(X)} filas. Shape final: {X.shape}") 

    ## Relleno filas
    X = fill_nan_values(X, l_columns_mucho_nan, fill_type=fill_type)
    print(f"Tras eliminar y reemplazar nan values, se hara el feature selection con {X.shape[0]} filas y {X.shape[1]} columnas")
    return X

def replace_nan_with_zero(df, col1, col2):  # Si un equipo no tiene jug asusentes pero el otro si, entonces que reemplece nan por 0 (asi puedo restar home y away evitando el nan puesto que 7 - nan = nan)
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

def replace_infinite(df):
    
    # Reemplazar los valores infinitos por NaN para luego eliminarlos --> Para evitar error en scaler: (ValueError: Input X contains infinity or a value too large for dtype('float64')) 
    df_numeric = df.select_dtypes(include=[np.number]) # Seleccionar solo las columnas numéricas
    # logger.info(np.isinf(df_numeric).sum()) # Verificar si existen valores infinitos en las columnas numéricas
    # Calcular el porcentaje de valores infinitos por columna
    inf_percentages = (np.isinf(df_numeric).sum() / len(df_numeric)) * 100

    # Crear una lista con los nombres de las columnas y su porcentaje de valores infinitos
    inf_columns_info = [(col, inf_percentages[col]) for col in df_numeric.columns if inf_percentages[col] > 0]

    # Loguear la información de las columnas con valores infinitos
    for col, perc in inf_columns_info:
        logger.info(f"Columna: {col}, Porcentaje de valores infinitos: {perc:.2f}%")

    df[df_numeric.columns] = df_numeric.replace([np.inf, -np.inf], np.nan) # Reemplazar los valores infinitos por NaN en las columnas numéricas
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    country = 'England'

    # Levanto dataset
    df = pd.read_excel(f'data/{country}/p2_data_understanding/df_match.xlsx', index_col=0)
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
    
    df.to_excel(f'{BASE_DIR_LOCAL}/df_cleaned_prueba.xlsx', index=True)