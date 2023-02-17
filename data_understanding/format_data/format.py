import pandas as pd
import datetime
import re

def format_date(df):  # Aun no la probe pues deberia hice el format de la fecha junto con la extraccion pero bueno
    """
    Convierte fecha del formato "14.02.2023 00:00" a formato "dd/mm/yyyy hh:mm", es decir, "14/02/2023 20:00"
    :param df: Dataframe. Fecha en formato "14.02.2023 20:00"
    :return: String. Fecha y hora con formato "dd/mm/yyyy", por ejemplo, "14/02/2023 20:00"
    """
    # Por registro (partido)
    for i in range(len(df)):

        # PASO 1: SPLIT FECHA (STRING) EN ELEMENTOS (DIA, MES, AÑO, HORA Y MIN)
        n_dia, n_mes, n_año, hora, min = re.split(r':|.', df.loc[i, 'fecha'])

        # Paso 2: convierto string a datetime
        fecha_datetime = datetime.datetime(int(n_año), int(n_mes), int(n_dia), int(hora), int(min))  # Convierto variable de clase 'str' a clase 'datetime.datetime'
        fecha_form = fecha_datetime.strftime("%Y/%m/%d %H:%M")  # Convierto el formato de datetime del default al formato deseado

        # Guardo fecha formateada
        df.loc[i, 'Fecha'] = fecha_form

    return df


def convert_fecha_to_datetime(df):

    # Ordeno dataframe por fecha (de mas reciente a mas viejo)
    print(type(df.fecha[0]))

    # convert to date
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)

    # verify datatype
    print(type(df.fecha[0]))
    return df

def main():
    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/flashscore/liga_argentina_historico.xlsx')
    print(df.head())

    # Formateo fecha
    # df = format_date(df)

    # Guardo el dataset
    df.to_excel('./df_formated.xlsx')

main()