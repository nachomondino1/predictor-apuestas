import pandas as pd
from dspy.data_preparation.text_preparation import TextPreparation
from sklearn.ensemble import RandomForestRegressor

def prepare_text_columns(df, l_col_to_except):  # Podria agregar un l_except_columns para eevitar analizar alguna columna de strings que no quiera preparar...
    '''
    Prepara el texto de las columnas que contengan strings.
    :param df: Dataframe.
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
    d_abrev_team_names = {'l p ': 'la plata', 'atl ': 'atletico ', ' jrs': ' juniors', ' utd': ' united'}  # Tengo que tener cuidado, reemplazo strings... pueden ser substring y cambiarlo sin querer hacerlo.
    df['equipo_loc'] = df['equipo_loc'].replace(d_abrev_team_names, regex=True).str.strip()
    df['equipo_vis'] = df['equipo_vis'].replace(d_abrev_team_names, regex=True).str.strip()

    # Reemplazo nombres enteros de equipos para que sea igual a los de la entidad jugador
    d_team_names = {'qpr': 'queens park rangers',  'wolves': 'wolverhampton', 'west brom': 'west bromwich albion'}
    for equipo_part, equipo_jug in d_team_names.items():
        df['equipo_loc'] = df['equipo_loc'].replace(equipo_part, equipo_jug)
        df['equipo_vis'] = df['equipo_vis'].replace(equipo_part, equipo_jug)

    return df

def treat_nan_values(df, type):
    columnas_con_nan = df.columns[df.isna().any()].tolist()

    # Crear una copia del dataframe original
    df_filled = df.copy()

    # OPCION 1: Eliminar cualquier registro con al menos un nan
    if type == "drop":
        df_filled = df.dropna().reset_index()  # inplace=True  # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)

    # OPCION 2: Llenar los valores faltantes con el valor más frecuente en cada columna
    elif type == "fillna_with_mode":
        for col in columnas_con_nan:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

    # OPCION 3: Llenar los valores faltantes con ML
    elif type == "fillna_with_ml":

        # Iterar sobre las columnas con valores faltantes
        for col in columnas_con_nan:

            # Dividir el dataframe en conjunto de entrenamiento y prueba
            X_train = df_filled.loc[df[col].notnull()].drop(columns=columnas_con_nan)
            y_train = df_filled.loc[df[col].notnull(), col]
            X_test = df_filled.loc[df[col].isnull()].drop(columns=columnas_con_nan)

            # Crear un modelo RandomForestRegressor
            model = RandomForestRegressor()

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Predecir los valores faltantes
            predicted_values = model.predict(X_test)

            # Rellenar los valores faltantes en el dataframe
            df_filled.loc[df[col].isnull(), col] = predicted_values

    # Imprimir el dataframe después de la imputación
    return df_filled


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