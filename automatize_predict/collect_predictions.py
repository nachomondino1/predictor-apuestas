import sys
import ast
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p6_deployment import main_next_matches
from datetime import datetime

# Mejoras a implementar
# # - Tengo que actualizar historial_predicciones.xlsx con el resultado del partido una vez que ya termino.
# Chequear que historial_predicciones.xslx esta bien construido y se actualiza correctamente.

def collect_predictions(l_countries: list, n_days: float):
    """
    Recoleccion de predicciones de todos los paises
    """
    # Defino condiciones del analisis
    d_run = {'run_missing': False, 'data_unders': True, 'data_prep': True, 'modeling': True, 'export': True}  # momentaneamente data_unders es False por pruebas
    df = pd.DataFrame()

    # Levanto historial_predicciones.xlsx
    try:
        df_hist = pd.read_excel(f'./p6_deployment/data/historial_predicciones.xlsx', index_col=0)
    except FileNotFoundError:
        df_hist = pd.DataFrame()

    # Por country
    for id_country in l_countries:

        # Extraigo, preparo y predigo proximos partidos
        df_country = main_next_matches.main(d_run, id_country, n_days, export=d_run['export'])

        # Concatenar df_countries....
        df = pd.concat([df, df_country], axis=0)

        # Guardo predicciones en historial
        df_hist = load_and_update_predictions(df_country, df)

    # Exporto datos
    df.index.name = 'id_match'
    df.to_excel(f'./p6_deployment/data/predicciones.xlsx', index=True)
    df_hist.to_excel(f'./p6_deployment/data/historial_predicciones.xlsx', index=True)

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

    # l_countries, n_days = prueba()
    n_days = float(sys.argv[1])  # e.g. 7  # Numero de dias maximo desde hoy para extraer partidos
    l_countries = ast.literal_eval(sys.argv[2])  # e.g. [48, 55, 59, 77, 167]

    collect_predictions(l_countries, n_days)    
