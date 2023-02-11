import pandas as pd

def date_to_datetime(df):

    # Ordeno dataframe por fecha (de mas reciente a mas viejo)
    print(type(df.Fecha[0]))

    # convert to date
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)

    # verify datatype
    print(type(df.Fecha[0]))
    return df


def main():
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data understanding/collect initial data/liga_argentina_historico.xlsx')
    df = date_to_datetime(df)
    df.to_excel('./df_formated.xlsx')

main()