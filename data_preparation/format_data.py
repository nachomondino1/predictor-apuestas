import pandas as pd
import numpy as np

def remove_percent_sign(df):
    """
    Transformo posesion de string a float
    :param df: Dataframe. Con columnas 'posesion_loc' y 'posesion_vis' donde la posesion se interpreta como string. Por
    ejemplo '65%'.
    :return: Dataframe. Con columna 'fecha' interpretada como float. Por ejemplo, '0.65'
    """
    df['posesion_loc'] = df['posesion_loc'].apply(lambda x: int(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan)
    df['posesion_vis'] = df['posesion_vis'].apply(lambda x: int(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan)
    return df

def transform_date_column(df):
    """
    Transformo fecha de string a datetime
    :param df: Dataframe. Con columna 'fecha' interpretada como string
    :return: Dataframe. Con columna 'fecha' interpretada como datetime
    """
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d.%m.%Y %H:%M')
    return df

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


def main():
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/argentina/argentina.xlsx')

    # Convierto posesion de string a integer
    df = remove_percent_sign(df)

    # Convierto fecha de string a datetime
    df = transform_date_column(df)  # Fundamental para poder ordenar el df por 'fecha'

    # Remuevo strings adicionales en los nombres de los equipos
    df = remove_strings_from_teams(df)

    # Ordeno por campo 'fecha'
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)
    df.to_excel('/Users/nachomondino/Desktop/df_formated.xlsx')

# main()
