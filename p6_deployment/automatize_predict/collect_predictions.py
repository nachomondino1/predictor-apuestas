import sys
sys.path.append('.')  # Fallaba el import de main
import os
from dotenv import load_dotenv
import ast
import json
import pandas as pd
from p6_deployment import main_next_matches
from set_up_logging import logger


def collect_predictions(d_run: dict, l_countries:list, n_days:float, df_historial_predicciones:pd.DataFrame):
    """
    Recoleccion de predicciones de todos los paises
    """
    # Defino condiciones del analisis
    df_predicciones = pd.DataFrame()

    # Por country
    for id_country in l_countries:

        # Extraigo, preparo y predigo proximos partidos
        df_predicciones_country = main_next_matches.main(d_run, id_country, n_days, export=d_run['export'])

        # Elimino predicciones sin id_country y columnas vacias (las variables predictoras como referee)
        df_predicciones_country = df_predicciones_country.dropna(subset=['id_country'])
        df_predicciones_country = df_predicciones_country.dropna(axis=1, how='all')

        # Concatenar df_countries....
        df_predicciones = pd.concat([df_predicciones, df_predicciones_country], axis=0)

        # Exporto por seguridad (x si falla un pais, no haberlo corrido la action al re pedo)
        df_predicciones.to_excel(f'data/predicciones.xlsx', index=True)
        df_historial_predicciones.to_excel(f'data/historial_predicciones.xlsx', index=True)

    # Guardo predicciones en historial_predicciones
    largo_inic = len(df_historial_predicciones)
    df_historial_predicciones = pd.concat([df_historial_predicciones, df_predicciones], axis=0)
    df_historial_predicciones = df_historial_predicciones[~df_historial_predicciones.index.duplicated(keep='last')]  # Eliminar filas con índices duplicados, manteniendo la ultima aparición (la + actualizada)
    largo_fin = len(df_historial_predicciones)
    if largo_inic == largo_fin:
        logger.warning(f"No se han agregado partidos en historial_predicciones.xlsx. Sigue teniendo {largo_inic}")
    elif largo_fin > largo_inic:
        logger.critical(f"Se han agregado {largo_fin - largo_inic} partidos en historial_predicciones.xlsx. {largo_inic} --> {largo_fin}")

    # Exporto datos
    df_predicciones.index.name = 'id_match'  # Es importante para la base de datos MySQL
    df_predicciones.to_excel(f'data/predicciones.xlsx', index=True)
    df_historial_predicciones.to_excel(f'data/historial_predicciones.xlsx', index=True)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Cargo variables entorno
    load_dotenv()
    env = os.getenv('ENVIRONMENT')

    # Defino argumentos (no puedo usar variables entorno por update_predictions.yml que usa parametros especificos para cada corrida)
    if env == 'dev':
        # Definir condiciones del análisis
        n_days = 15
        l_countries = [48, 55, 59, 77, 148]
        d_run = {'run_missing': False, 'data_unders': True, 'data_prep': True, 'modeling': True, 'export': True}

    elif env == 'prod':
        n_days = float(sys.argv[1])  # Numero de dias maximo desde hoy para extraer partidos (e.g. 7)
        l_countries = ast.literal_eval(sys.argv[2])  # Lista de paises a los cuales extraer proximos partidos (e.g. [48, 55, 59, 77, 167])
        d_run = json.loads(sys.argv[3])  # Parametros de ejecucion (e.g. {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True})

    # Levanto historial_predicciones.xlsx
    df_historial_predicciones = pd.read_excel(f'data/historial_predicciones.xlsx', index_col=0)

    collect_predictions(d_run, l_countries, n_days, df_historial_predicciones)