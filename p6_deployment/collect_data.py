import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p6_deployment import main_next_matches
from datetime import datetime

def collect_data(id_country, n_days=7):
    """
    Recoleccion de predicciones de todos los paises
    """
    # Defino condiciones del analisis
    d_run = {'run_missing': False, 'data_unders': True, 'data_prep': True, 'modeling': True, 'export': True}  # momentaneamente data_unders es False por pruebas

    # Levanto predicciones.xlsx
    try:
        df = pd.read_excel(f'./p6_deployment/data/predicciones.xlsx', index_col=0)

        # Elimino partidos viejos de predicciones --> Ver si funciona ok
        df = df[df['date'] >=  datetime.now()]

    except FileNotFoundError:
        df = pd.DataFrame()

    # Extraigo, preparo y predigo proximos partidos
    df_country = main_next_matches.main(d_run, id_country, n_days, export=d_run['export'])

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

    # Defino argumentos
    id_country = 6
    n_days_max_next_matches = 3 # Numero de dias maximo desde hoy para extraer partidos

    collect_data(id_country, n_days=n_days_max_next_matches)