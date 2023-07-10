import pandas as pd
from dspy.data_preparation.text_preparation import TextPreparation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def prepare_text_columns(df, l_col_to_except):
    '''
    Prepara el texto de las columnas que contengan strings.
    :param df: Dataframe.
    :param l_col_to_except: Lista. Columnas del tipo object que omitir en el procesamiento.
    :return: Dataframe con columnas que contienen strings ya preparados para ser analizados
    '''
    # Obtengo columnas a procesar
    l_cols_to_process = set_columns_to_process(df, l_col_to_except)

    # Creo objeto de la clase
    tp = TextPreparation()
    df = tp.to_lower(df, columns=l_cols_to_process)
    df = tp.delete_accent(df, columns=l_cols_to_process)
    df = tp.delete_special_characters(df, columns=l_cols_to_process)
    df = tp.delete_punctuation(df, columns=l_cols_to_process)
    return df

def set_columns_to_process(df, l_col_to_except):
    """Define las columnas a procesar"""
    l_cols_object = df.select_dtypes(include='object').columns
    l_cols_to_process = [col for col in l_cols_object if col not in l_col_to_except]
    print("Columnas tipo object a preparar:", l_cols_to_process)
    return l_cols_to_process

# TRATAMIENTO DE NAN VALUES
def fill_nan_values(X, y, type):

    # Definivion de variable
    # l_columnas_con_nan = X.columns[X.isna().any()].tolist()
    nan_threshold = 0.05  # cuidado que si hago eliminacion de col antes por un valor inferior, esta lista esta vacia y no hace fillna...
    l_columnas_con_nan = X.columns[X.isna().mean() > nan_threshold].tolist()  # e.g. ['historial_entre_si', 'dif_edad_tit', 'dif_alt_tit', 'dif_rat_tit', 'dif_edad_sup', 'dif_alt_sup', 'dif_rat_sup']
    print("Columnas consideradas con mucho NaN:", l_columnas_con_nan)

    param_grid = {
        'n_estimators': [100, 300], # 200
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 10] #  5
    }

    # Crear una copia del dataframe original
    X_filled = X.copy()

    for col in l_columnas_con_nan:
        # print(f"Columna a rellenar: {col}")

        # OPCION 1: Llenar los valores faltantes con el valor más frecuente en cada columna
        if type == "mode":
            X_filled[col].fillna(X_filled[col].mode()[0], inplace=True)

        # OPCION 2: Llenar los valores faltantes con ML
        elif type == "ml":

            # Elimino registros NaN en las columnas con bajo % de NaN (para poder usarlas en X_train)
            # print("Shape X_filled antes:", X_filled.shape)  # e.g. (3279, 18)
            X_filled_dropna = X_filled.dropna(subset=X_filled.drop(l_columnas_con_nan, axis=1).columns)
            # print("Shape X_filled despues:", X_filled_dropna.shape) # e.g. (3106, 18)

            # Dividir el dataframe en conjunto de entrenamiento y prueba
            X_train = X_filled_dropna.loc[X[col].notnull()].drop(columns=l_columnas_con_nan)  # df con columna!=nan
            # print(X_train.shape) # e.g. (2728, 11)
            y_train = X_filled_dropna.loc[X[col].notnull(), col]  # Solo la columna donde columna!=nan
            # print(y_train.shape) # e.g. (2728,)
            X_test = X_filled_dropna.loc[X[col].isnull()].drop(columns=l_columnas_con_nan)
            # print(X_test.shape) # e.g. (378, 11)

            # Crear un modelo RandomForestRegressor
            model = RandomForestRegressor()

            # Realizar la búsqueda de cuadrícula para encontrar los mejores hiperparámetros
            grid_search = GridSearchCV(model, param_grid, cv=2)
            grid_search.fit(X_train, y_train)

            # Obtener los mejores hiperparámetros encontrados
            best_params = grid_search.best_params_

            # Crear un nuevo modelo RandomForestRegressor con los mejores hiperparámetros
            model_best = RandomForestRegressor(**best_params)

            # Entrenar el modelo con los datos de entrenamiento
            model_best.fit(X_train, y_train)

            # Predecir los valores faltantes
            predicted_values = model_best.predict(X_test)

            # Rellenar los valores faltantes en el dataframe
            predicted_values_index = X_test.index
            X_filled.loc[predicted_values_index, col] = predicted_values  # Creo que funciona

    # Elimino los registros NaN en las columnas que preferi usar para entrenar en vez de rellenar
    X_filled = X_filled.dropna(subset=X_filled.drop(l_columnas_con_nan, axis=1).columns)
    y = y.loc[X_filled.index]  # Selecciono las y solo de los registros en X_filled
    X_filled = X_filled.reset_index(drop=True)  # Reseteo index en X
    y = y.reset_index(drop=True) # Reseteo index en y
    return X_filled, y

def eliminar_filas_nan(df, umbral):
    """
    Elimina las filas de un DataFrame que contienen un porcentaje alto de valores NaN.

    :param df: DataFrame de entrada. (DataFrame)
    :param umbral: Umbral en forma de porcentaje (0-100) para determinar el límite de NaN en una fila. (float)
    :return: DataFrame resultante después de eliminar las filas con valores NaN. (DataFrame)
    """
    # Elimino filas segun umbral
    porcentaje_nan = df.isnull().mean(axis=1)  # Calcula el porcentaje de valores NaN en cada fila
    filas_a_eliminar = porcentaje_nan[porcentaje_nan > umbral].index  # Obtiene las filas que superan el umbral
    print(f"Se eliminó el {len(filas_a_eliminar)/len(df)*100:.0f}% de filas, quedan {len(df) - len(filas_a_eliminar)} filas.")

    # Elimino filas segun umbral
    df_filtrado = df.drop(filas_a_eliminar)  # Elimina las filas con valores NaN
    return df_filtrado

def eliminar_columnas_nan(df, umbral):
    """
    Elimina las columnas de un DataFrame que contienen un porcentaje alto de valores NaN.

    :param df: DataFrame de entrada. (DataFrame)
    :param umbral: Umbral en forma de porcentaje (0-100) para determinar el límite de NaN en una columna. (float)
    :return: DataFrame resultante después de eliminar las columnas con valores NaN. (DataFrame)
    """
    # Calcula la proporción de NaN en cada columna
    prop_nan = df.isna().mean()

    # Identifica las columnas con una proporción de NaN mayor al umbral
    columnas_eliminar = prop_nan[prop_nan > umbral].index

    # Elimina las columnas identificadas del DataFrame
    df_sin_nan = df.drop(columnas_eliminar, axis=1).reset_index(drop=True)  # es clave el drop=True para eliminar el indice viejo sino agrega la columna "index"
    print(f"Se eliminaron {len(list(columnas_eliminar))} de {len(df.columns)} columnas por tener un % NaN mayor a thr_nan_col={umbral*100:.0f}%: {list(columnas_eliminar)}")
    return df_sin_nan

def prueba():
    pais = 'argentina'

    # Levanto dataset
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_part_formated.xlsx')
    df_jug =pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_jug_formated.xlsx')

    # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
    l_col_to_except = ['id', 'temporada']
    df_part = prepare_text_columns(df_part, l_col_to_except)
    df_jug = prepare_text_columns(df_jug, l_col_to_except)

    # Remuevo strings adicionales en los nombres de los equipos
    df_part = clean_teams_names(df_part)

    df_part.to_excel('/Users/nachomondino/Desktop/df_part_cleaned_prueba.xlsx', index=False)
    df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_cleaned_prueba.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()