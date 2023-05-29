import pandas as pd
import numpy as np

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

def convert_fecha_to_datetime(df, string_format):
    """
    Transformo fecha de string a datetime
    :param df: Dataframe. Con columna 'fecha' interpretada como string
    :return: Dataframe. Con columna 'fecha' interpretada como datetime
    """
    df['fecha'] = pd.to_datetime(df['fecha'], format=string_format)
    return df

def separate_lists_in_columns(df, variable):  # Si bien es ineficiente, no me conviene mejorarla puesto que extraere ya las columnas separadas... ( tampoco tanto, tarda 4.2 seg, 4.1, 2.3, 1.9, 0.6, 0.6) Cuando extraiga cada jugador en vez de la lista, podre borrala
    """
    Convierto columnas que contienen listas en multiples columnas de un solo elemento
    :param df: Dataframe.
    :param variable: String. Nombre de la variable
    :return: Dataframe.
    """
    # Por registro
    for i in range(len(df)):

        # Obtengo lista
        str_with_list = df.loc[i, variable]

        # Verificar si el elemento es un string (Evito nan)
        if isinstance(str_with_list, str):

            # Convierto string a lista
            l_jug = eval(str_with_list)  # e.g. ["Dibu", ..., "Messi"]

            # Por elemento de la lista
            for j in range(len(l_jug)):

                # Guardo jugador en columna nueva
                df.loc[i, f'{variable[2:]}_{j+1}'] = l_jug[j]

    df = df.drop([variable], axis=1)
    return df

def main():
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_partido_argentina.xlsx')
    df_jug = pd.read_excel("/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_jugadores.xlsx")

    # Convierto posesion de string a integer
    df = remove_percent_sign(df)

    # Convierto fecha de string a datetime
    df = transform_date_column(df, string_format='%d.%m.%Y %H:%M')  # Fundamental para poder ordenar el df por 'fecha'
    df_jug = transform_date_column(df_jug, string_format='%b %d, %Y')

    # Remuevo strings adicionales en los nombres de los equipos
    df = remove_strings_from_teams(df)

    # Ordeno por campo 'fecha'
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)
    df.to_excel('./df_formated.xlsx', index=False)
    df_jug.to_excel('./df_jug_formated.xlsx', index=False)

# main()