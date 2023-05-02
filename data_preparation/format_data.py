import pandas as pd
import datetime

def posesion_balon(df):
    """
    Transformo posesion de string a float
    :param df: Dataframe. Con columnas 'posesion_loc' y 'posesion_vis' donde la posesion se interpreta como string. Por
    ejemplo '65%'.
    :return: Dataframe. Con columna 'fecha' interpretada como float. Por ejemplo, '0.65'
    """
    l_variables = ['posesion_loc', 'posesion_vis']

    # Por fila (partido)
    for idx in df.index:

        # Por variable de posesion
        for variable in l_variables:

            # Obtengo posesion
            pos_balon = df.loc[idx, variable]

            # Si es un string (evito nan que son float)
            if isinstance(pos_balon, str):
                # Reemplazo "%", convierto a entero y guardo el nuevo valor
                df.loc[idx, variable] = int(pos_balon.replace("%", ""))
    return df

def format_column_date(df):
    """
    Transformo fecha de string a datetime
    :param df: Dataframe. Con columna 'fecha' interpretada como string
    :return: Dataframe. Con columna 'fecha' interpretada como datetime
    """
    # Por registro
    for i in range(len(df)):

        fecha_str = df.loc[i, 'fecha']
        fecha_form = datetime.datetime.strptime(fecha_str, '%d.%m.%Y %H:%M').date()

        df.loc[i, 'fecha'] = fecha_form
    return df

def clean_teams(df):
    """
    Limpio string 'Vencedor' en el nombre de algunos equipos.
    :param df:
    :return:
    """
    # Definicion de variables
    l_variables = ['equipo_loc', 'equipo_vis']

    # Por partido
    for i in range(len(df)):

        # Por variable de equipo
        for variable in l_variables:

            # Remuevo 'vencedor' o 'equipo que avanza' del nombre del equipo
            str_equipo = df.loc[i, variable].replace('Vencedor', '').replace('Equipo que avanza','')
            df.loc[i, variable] = str_equipo
    return df

def main():
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/flashscore/liga_argentina_historico.xlsx')

    df = posesion_balon(df)

    df = format_column_date(df)  # Fundamental para poder ordenar el df por 'fecha'
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    df.to_excel('/Users/nachomondino/Desktop/df_formated_2.xlsx')

# main()
