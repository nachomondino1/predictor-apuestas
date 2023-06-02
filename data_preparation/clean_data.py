import pandas as pd
from dspy.data_preparation.text_preparation import TextPreparation
from sklearn.preprocessing import LabelEncoder

def remove_strings_from_teams(df):
    """
    Limpio string 'Vencedor' en el nombre de algunos equipos.
    :param df: Dataframe. Con columnas "equipo_loc" y "equipo_vis"
    :return: Dataframe pasado por parametro sin strings "Vencedor" y "Equipo que avanza" en las columnas "equipo_loc"
    y "equipo_vis".
    """
    df['equipo_loc'] = df['equipo_loc'].str.replace('Vencedor', '').str.replace('Equipo que avanza', '').str.strip()
    df['equipo_vis'] = df['equipo_vis'].str.replace('Vencedor', '').str.replace('Equipo que avanza', '').str.strip()
    return df

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

def convert_columns_to_int(df):
    # Convertir variables categoricas string a categoricas numericas
    le = LabelEncoder()

    # Por variable string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

        # Verificar si la variable es "equipo_ganador"
        if col == "equipo_ganador":
            etiquetas = le.classes_
            codigos = le.transform(etiquetas)
            df_etiquetas = pd.DataFrame({"Etiqueta": etiquetas, "Código": codigos})
            df_etiquetas.to_excel("/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/etiquetas_equipo_ganador.xlsx", index=False)

    return df

def change_teams_names(df_part):
    """
    Cambio el nombre de algunos equipos de la entidad partido puesto que falla su busqueda en la entidad jugador por
    estar abreviados. Por ejemplo, "atl. tucuman" en vez de "atletico tucuman"
    :param df_part: Dataframe
    :return: Dataframe con nombres de equipos modificados para facilitar la integracion
    """
    # Definicion de variables
    d = {'gimnasia l.p.': 'gimnasia la plata', 'atl. tucuman': 'atletico tucuman', 'argentinos jrs.': 'argentinos juniors',
         'boca jrs.': 'boca juniors', 'estudiantes l.p.': 'estudiantes'}

    for equipo_part, equipo_jug in d.items():

        df_part['equipo_loc'] = df_part['equipo_loc'].replace(equipo_part, equipo_jug)
        df_part['equipo_vis'] = df_part['equipo_vis'].replace(equipo_part, equipo_jug)

    return df_part

def prueba():
    # Levanto dataset
    df = pd.read_excel('data/df_constructed.xlsx')

    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

    # Remover NaN values
    df = df.dropna()  # inplace=True  # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)

    # Convertir variables categoricas string a categoricas numericas
    df = convert_columns_to_int(df)

# prueba()


''' fill NaN values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:

        mean = df[col].mean()  # Calcula la media de una columna
        df[col] = df[col].fillna(mean)  # Rellena los NaN en esa columna con la media

        print(f"Columna: {col} \nMedia: {mean}")
'''