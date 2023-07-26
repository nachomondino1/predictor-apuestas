import pandas as pd
from dspy.data_preparation.text_preparation import TextPreparation
from p4_modeling.build_model import select_best_hiperparameters
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


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

    # Definicion de variables
    X_filled = X.copy()    # Crear una copia del dataframe original dado que realizare cambios en las columnas y valores
    nan_threshold = 0.05  # cuidado que si hago eliminacion de col antes por un valor inferior, esta lista esta vacia y no hace fillna...

    # Determino las columnas con mucho NaN (mas de nan_threshold%)
    l_columnas_con_nan = X.columns[X.isna().mean() > nan_threshold].tolist()  # e.g. ['historial_entre_si', 'dif_edad_tit', 'dif_alt_tit', 'dif_rat_tit', 'dif_edad_sup', 'dif_alt_sup', 'dif_rat_sup']
    print("Columnas consideradas con mucho NaN:", l_columnas_con_nan)

    # Por columna a rellenar
    for col in l_columnas_con_nan:
        # print(f"Columna a rellenar: {col}")

        # OPCION 1: Llenar los valores faltantes con el valor más frecuente en cada columna
        if type == "mode":
            X_filled[col].fillna(X_filled[col].mode()[0], inplace=True)

        # OPCION 2: Llenar los valores faltantes con ML
        elif type == "ml":

            # Si hay al menos una columna sin NaN (sino no tengoo columnas para X_train y falla con ValueError)
            if len(l_columnas_con_nan) < len(X.columns):

                # Elimino registros NaN en las columnas con bajo % de NaN (para poder usarlas en X_train)
                X_filled_dropna = X_filled.dropna(subset=X_filled.drop(l_columnas_con_nan, axis=1).columns)

                # Dividir el dataframe en conjunto de entrenamiento, validacion y prueba
                # Separo test de train y val puesto que test tendra los NaN values para la columna
                X_train_val = X_filled_dropna.loc[X[col].notnull()].drop(columns=l_columnas_con_nan)  # df con columna!=nan # e.g. (2728, 11)
                y_train_val = X_filled_dropna.loc[X[col].notnull(), col]  # Solo la columna donde columna!=nan # e.g. (2728,)
                X_test = X_filled_dropna.loc[X[col].isnull()].drop(columns=l_columnas_con_nan)  # e.g. (378, 11)
                # Separo en train y val
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, shuffle=True)

                # Selecciono los mejores hiperparametros usando el set de validacion
                model = select_best_hiperparameters(RandomForestRegressor(), X_val, y_val, k=10)  # Falla x2

                # Entrenar el modelo con los datos de entrenamiento
                model.fit(X_train, y_train)

                # Predecir los valores faltantes
                predicted_values = model.predict(X_test)

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
    pais = 'argentina_south_america'
    thr_nan_col = 0.2

    # Levanto dataset
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_constructed_False_180_30_3.xlsx')

    # Eliminacion de NaN values
    # Elimino filas y columnas con alto porcentaje de NaN values
    largo_inicial = len(df)
    df = df.dropna(subset=['dif_prom_ult_part_dif_remates'], how='any')
    print(f"Se eliminó el {(largo_inicial - len(df)) / largo_inicial * 100:.0f}% de filas, quedan {len(df)} filas.")

    if thr_nan_col is not None:
        prop_nan = df.isna().mean()
        print(prop_nan)
        df = eliminar_columnas_nan(df, umbral=thr_nan_col)  # 2º elimino columnas con mucho NaN # ojo que asi puede borrar odds

    # Verificar que no haya outliers
    # ...

    # Normalizo columnas con valores mas grandes para evitar ValueError: Solver produced non-finite parameter weights. The input data may contain large values and need to be preprocessed.
    # scaler = StandardScaler()  # Crea un objeto StandardScaler
    # df['dif_sum_min_titular'] = scaler.fit_transform(df['dif_sum_min_titular'].values.reshape(-1, 1))
    # df['dif_sum_min_suplente'] = scaler.fit_transform(df['dif_sum_min_suplente'].values.reshape(-1, 1))
    #
    # df.to_excel('/Users/nachomondino/Desktop/df_cleaned_prueba.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()