import sys
sys.path.append('.')  # Fallaba el import de main
import ast
import json
import pandas as pd
from p6_deployment import main_next_matches

def collect_predictions(d_run: dict, l_countries:list, n_days:float, df_hist):
    """
    Recoleccion de predicciones de todos los paises
    """
    # Defino condiciones del analisis
    df = pd.DataFrame()

    # Por country
    for id_country in l_countries:

        # Extraigo, preparo y predigo proximos partidos
        df_country = main_next_matches.main(d_run, id_country, n_days, export=d_run['export'])

        # Concatenar df_countries....
        df = pd.concat([df, df_country], axis=0)

    # Guardo predicciones en historial
    print(f"Antes de cargar historial: {df_hist.shape}")
    df_hist = pd.concat([df_hist, df], axis=0)
    df_hist.drop_duplicates(keep='last')  # Asi, si cargo con formaciones, me quedo con ese en vez de sin.
    print(f"Luego de cargar country al historial: {df_hist.shape}")

    # Exporto datos
    df.index.name = 'id_match'  # Es importante para la base de datos MySQL # df.set_index('id_match', inplace=True) # Establecer 'n_iteration' como índice del DataFrame
    df.to_excel(f'p6_deployment/data/predicciones.xlsx', index=True)
    df_hist.to_excel(f'p6_deployment/data/historial_predicciones.xlsx', index=True)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Defino argumentos
    n_days = float(sys.argv[1])  # e.g. 7  # Numero de dias maximo desde hoy para extraer partidos
    l_countries = ast.literal_eval(sys.argv[2])  # e.g. [48, 55, 59, 77, 167]
    d_run = json.loads(sys.argv[3])  # Convertir la cadena JSON de vuelta a un diccionario  {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True}

    # Levanto historial_predicciones.xlsx
    df_hist = pd.read_excel(f'p6_deployment/data/historial_predicciones.xlsx', index_col=0)

    collect_predictions(d_run, l_countries, n_days, df_hist)    
