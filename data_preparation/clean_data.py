import pandas as pd
from dspy.data_preparation.text_preparation import TextPreparation
from sklearn.ensemble import RandomForestRegressor


def prepare_text_columns(df):  # Podria agregar un l_except_columns para eevitar analizar alguna columna de strings que no quiera preparar...
    '''
    Prepara el texto de las columnas que contengan strings.
    :param df: Dataframe.
    :return: Dataframe con columnas que contienen strings ya preparados para ser analizados
    '''
    # Convertir variables categoricas string a categoricas numericas
    for var in df.select_dtypes(include=['object']).columns:

        prepare_text = TextPreparation(textos=df[var])
        prepare_text.to_lower()
        prepare_text.delete_accent()
        prepare_text.delete_special_characters()
        df[var] = prepare_text.textos

    return df

def clean_teams_names(df):  # Funcion verificada
    """
    Limpia y cambia el nombre de algunos equipos en el dataframe.
    :param df: Dataframe con columnas "equipo_loc" y "equipo_vis"
    :return: Dataframe con nombres de equipos modificados y limpios
    """
    # Limpio string 'Vencedor' en el nombre de algunos equipos.
    df['equipo_loc'] = df['equipo_loc'].str.replace('vencedor', '').str.replace('equipo que avanza', '').str.strip()
    df['equipo_vis'] = df['equipo_vis'].str.replace('vencedor', '').str.replace('equipo que avanza', '').str.strip()

    # Cambiar nombres de equipos
    d = {'gimnasia l.p.': 'gimnasia la plata', 'atl. tucuman': 'atletico tucuman', 'argentinos jrs.': 'argentinos juniors',
         'boca jrs.': 'boca juniors', 'estudiantes l.p.': 'estudiantes'}

    for equipo_part, equipo_jug in d.items():
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
    df_part = pd.read_excel('data/df_part_formated.xlsx')
    df_jug =pd.read_excel('data/df_jug_formated.xlsx')

    # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
    df_part = prepare_text_columns(df_part)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales
    df_jug = prepare_text_columns(df_jug)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

    # Remuevo strings adicionales en los nombres de los equipos
    df_part = clean_teams_names(df_part)

    df_part.to_excel('/Users/nachomondino/Desktop/df_part_cleaned.xlsx', index=False)
    df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_cleaned.xlsx', index=False)


# prueba()