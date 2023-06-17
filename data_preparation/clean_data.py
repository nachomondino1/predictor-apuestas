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
    # Creo objeto de la clase
    tp = TextPreparation(df, l_col_to_except)

    # Uso metodos de la clase
    tp.to_lower()
    tp.delete_accent()
    tp.delete_special_characters()
    tp.delete_punctuation()
    return tp.df

def clean_teams_names(df):
    """
    Limpia y cambia el nombre de algunos equipos en el dataframe.
    :param df: Dataframe con columnas "equipo_loc" y "equipo_vis"
    :return: Dataframe con nombres de equipos modificados y limpios
    """
    # Limpio string 'Vencedor' en el nombre de algunos equipos.
    df['equipo_loc'] = df['equipo_loc'].replace({'vencedor': '', 'equipo que avanza': ''}).str.strip()
    df['equipo_vis'] = df['equipo_vis'].replace({'vencedor': '', 'equipo que avanza': ''}).str.strip()

    # Quitar abreviaturas en nombres de equipos (NO FUNCIONA...)
    d_abrev_team_names = {' l p': ' la plata', 'atl ': 'atletico ', ' jrs': ' juniors', ' utd': ' united'}  # Tengo que tener cuidado, reemplazo strings... pueden ser substring y cambiarlo sin querer hacerlo.
    df['equipo_loc'] = df['equipo_loc'].replace(d_abrev_team_names, regex=True).str.strip()
    df['equipo_vis'] = df['equipo_vis'].replace(d_abrev_team_names, regex=True).str.strip()

    # Reemplazo nombres enteros de equipos para que sea igual a los de la entidad jugador
    d_team_names = {'estudiantes la plata': 'estudiantes', 'qpr': 'queens park rangers',  'wolves': 'wolverhampton',
                    'west brom': 'west bromwich albion'}
    for equipo_part, equipo_jug in d_team_names.items():
        df['equipo_loc'] = df['equipo_loc'].replace(equipo_part, equipo_jug)
        df['equipo_vis'] = df['equipo_vis'].replace(equipo_part, equipo_jug)

    return df

# TRATAMIENTO DE NAN VALUES
def fill_nan_values(df, type):

    # Definivion de variables
    l_columnas_con_nan = df.columns[df.isna().any()].tolist()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Crear una copia del dataframe original
    df_filled = df.copy()

    for col in l_columnas_con_nan:

        # OPCION 1: Llenar los valores faltantes con el valor más frecuente en cada columna
        if type == "mode":
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

        # OPCION 2: Llenar los valores faltantes con ML
        elif type == "ml":

            # Dividir el dataframe en conjunto de entrenamiento y prueba
            X_train = df_filled.loc[df[col].notnull()].drop(columns=l_columnas_con_nan)
            y_train = df_filled.loc[df[col].notnull(), col]
            X_test = df_filled.loc[df[col].isnull()].drop(columns=l_columnas_con_nan)

            # Crear un modelo RandomForestRegressor
            model = RandomForestRegressor()

            # Realizar la búsqueda de cuadrícula para encontrar los mejores hiperparámetros
            grid_search = GridSearchCV(model, param_grid, cv=3)
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
            df_filled.loc[df[col].isnull(), col] = predicted_values

    # Imprimir el dataframe después de la imputación
    return df_filled

def eliminar_filas_nan(df, umbral):
    """
    Elimina las filas de un DataFrame que contienen un porcentaje alto de valores NaN.

    Args:
        df (pandas.DataFrame): DataFrame de entrada.
        umbral (float): Umbral en forma de porcentaje (0-100) para determinar el límite de NaN en una fila.

    Returns:
        pandas.DataFrame: DataFrame resultante después de eliminar las filas con valores NaN.

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

    Args:
        df (pandas.DataFrame): DataFrame de entrada.
        umbral (float): Umbral en forma de porcentaje (0-100) para determinar el límite de NaN en una columna.

    Returns:
        pandas.DataFrame: DataFrame resultante después de eliminar las columnas con valores NaN.

    """
    # Calcula la proporción de NaN en cada columna
    prop_nan = df.isna().mean()

    # Identifica las columnas con una proporción de NaN mayor al umbral
    columnas_eliminar = prop_nan[prop_nan > umbral].index

    # Elimina las columnas identificadas del DataFrame
    df_sin_nan = df.drop(columnas_eliminar, axis=1)
    print(f"Columnas eliminadas por % NaN mayor a thr_nan_col={umbral*100:.0f}%: {list(columnas_eliminar)}")
    return df_sin_nan

def prueba():
    # Levanto dataset
    df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/inglaterra/df_part_formated.xlsx')
    df_jug =pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/inglaterra/df_jug_formated.xlsx')

    # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
    l_col_to_except = ['id', 'temporada']
    df_part = prepare_text_columns(df_part, l_col_to_except)
    df_jug = prepare_text_columns(df_jug, l_col_to_except)

    # Remuevo strings adicionales en los nombres de los equipos
    df_part = clean_teams_names(df_part)

    df_part.to_excel('/Users/nachomondino/Desktop/df_part_cleaned.xlsx', index=False)
    df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_cleaned.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()