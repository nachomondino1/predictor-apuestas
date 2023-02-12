import pandas as pd

def date_to_datetime(df):

    # Ordeno dataframe por fecha (de mas reciente a mas viejo)
    print(type(df.fecha[0]))

    # convert to date
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)

    # verify datatype
    print(type(df.fecha[0]))
    return df


def main():
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data understanding/collect_data/liga_argentina_historico.xlsx')
    df = date_to_datetime(df)
    df.to_excel('./format_data/df_formated.xlsx')

main()