import pandas as pd
from main_next_matches import main


def collect_data(l_countries):
    """
    Recoleccion de predicciones de todos los paises
    """
    # Defino condiciones del analisis
    n_days_max_next_matches = 3 # Numero de dias maximo desde hoy para extraer partidos
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': True}  # momentaneamente data_unders es False por pruebas

    # Levanto predicciones.xlsx
    try:
        df = pd.read_excel(f'./p6_deployment/data/predicciones.xlsx', index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Por pais
    for country in l_countries:

        # Extraigo, preparo y predigo proximos partidos
        df_country = main(d_run, country, n_days_max_next_matches, export=d_run['export'])

        # Guardo predicciones en historial
        df = load_and_update_predictions(df_country, df)
    
    # Exporto datos
    df.index.name = 'id_match'
    df.to_excel(f'./p6_deployment/data/predicciones.xlsx', index=True)

def load_and_update_predictions(df, df_hist):
    """
    Cargar predicciones de proximos partidos a historial de predicciones
    """
    # Por proximo partido
    for idx, row in df.iterrows():
        print(f'Partido Nº {idx}')

        # Convertir row a DataFrame
        row_df = pd.DataFrame([row])
        row_df.index = [idx]

        # Si ya esta en df_hist
        if idx in df_hist.index:
            copiado1 = row['copiado_formaciones']
            copiado2 = df_hist.loc[idx, 'copiado_formaciones']
            print(copiado1, copiado2)

            # Si tengo el partido con formaciones y no esta en df_hist
            if (copiado2 == 1) and (pd.isna(copiado1)):
                df_hist = df_hist.drop([idx])
                df_hist = pd.concat([df_hist, row_df], axis=0)
        else:
            # Lo cargo (ya sea con o sin formaciones)
            df_hist = pd.concat([df_hist, row_df], axis=0)

    return df_hist

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    l_countries = ["england", 'germany', 'france', 'italy'] 
    #  'spain' --> volver a entrenar modelos.

    collect_data(l_countries)