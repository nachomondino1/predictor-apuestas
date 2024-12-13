import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import os
from dotenv import load_dotenv
import pandas as pd
import datetime
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_matches_result
from p3_data_preparation.construct_data import determine_result
from p4_modeling.betting_strategy import BettingStrategy


def determine_last_matches(df: pd.DataFrame, n_days: float = 1):
    """
    Selecciono los partidos jugados en los ultimos <n_days>.
    """
    # Seleccionar los partidos de ultimo/s dia/s
    fecha_hoy = datetime.datetime.now()
    fecha_limite = fecha_hoy - datetime.timedelta(days=n_days)
    print(f"Fechas a filtrar: {fecha_limite} --> {fecha_hoy}")

    df_filt = df[(df['date'] > fecha_limite) & (df['date'] <= fecha_hoy)]
    print(df_filt.shape)
    return df_filt

def collect_results(df: pd.DataFrame, df_countries: pd.DataFrame, df_comp_public: pd.DataFrame):
    """
    Extraigo de Flashscore el resultado de los partidos pasados como parametro (df) y lo agrego como columna.
    """
    # Definicion de variables
    df_results = pd.DataFrame()
    bs = BettingStrategy()

    # Por competicion
    for idx, row in df_comp_public.iterrows():
        competition = row['competition_flashscore']
        l_ids_country = df[df['id_country'] == row['id_country']].index
        logger.info(f"ID_COUNTRY: {row['id_country']} COMPETITION: {competition}")
        logger.info(f"Cantidad de partidos a los que extraer resultado: {len(l_ids_country)}")

        # Definicion de variables
        country = df_countries[df_countries['id_country']==row['id_country']]['country_name'].values[0]
        
        # Extraer goles home y away en los partidos desde Flashscore
        df_results_competition = extract_matches_result(country, competition, l_ids_country)

        if len(df_results_competition) > 0:

            # Elimino partidos con goals_home y goals_away None (A PRUEBA, NO SE SI FUNCIONA)
            # Elimino partidos con goals_home "-" puesto que son partidos suspendidos
            df_results_competition = drop_suspended_matches(df_results_competition)
            # df_results_competition = df_results_competition.dropna(subset=["goals_home"])

            # Guardo results de competencia
            df_results = pd.concat([df_results, df_results_competition], axis=0)
            logger.info(df_results)
        
    # Si hay algun partido:
    if len(df_results) > 0:

        # Agrego columnas 'goals_home' y 'goals_away' a predicciones.xlsx
        df_pred_with_goals = pd.concat([df, df_results], axis=1)

        # Determino ganador y si acerté
        df_pred_with_result = determine_result(df_pred_with_goals, var_resp='result')
        df_pred_with_result = bs.determine_winning_bets(df_pred_with_result)  # Determino acierto o fallo
        df_pred_with_result.index.name = 'id_match'  # Es importante para la base de datos MySQL

        # Exportar dataset
        df_pred_with_result.to_excel('data/predicciones.xlsx') # Los partidos que tiene son de historial_predicciones en realidad pero uso predicciones.xlsx para poder activar dispatch y enviar datos a VPS?
        logger.critical(f"Se recolecto el resultado de {len(df_results)} partidos.")

    else:
        logger.warning("Se evitó el update de resultados puesto que no se detectaron partidos jugados ayer")

def drop_suspended_matches(df):
    """
    Si no los elimina, falla la determinacion de si acerte o no pues los goals no son integer.
    """
    # Convertir la columna 'goals_home' a numérica, convirtiendo valores no numéricos en NaN
    df['goals_home'] = pd.to_numeric(df['goals_home'], errors='coerce')
    df['goals_away'] = pd.to_numeric(df['goals_away'], errors='coerce')

    # Eliminar las filas donde 'goals_home' es NaN (es decir, no era un número)
    df = df.dropna(subset=['goals_home', 'goals_away'])

    # Opcional: Convertir de nuevo a entero si los valores en 'goals_home' deben ser enteros
    df['goals_home'] = df['goals_home'].astype(int)
    df['goals_away'] = df['goals_away'].astype(int)
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Cargo variables entorno
    load_dotenv()
    env = os.getenv('ENVIRONMENT')
    
    # Parametros de ejecución
    if env == 'dev':
        n_days = 5
    
    elif env == 'prod':
        n_days = float(sys.argv[1])  # Numero de dias maximo desde hoy para extraer partidos (e.g. 7)

    # Levanto datasets
    df_historial_predicciones = pd.read_excel('data/historial_predicciones.xlsx', index_col=0)  # Para garantizar que tengo todas las predicciones.  
    df_countries = pd.read_excel('data/df_countries.xlsx')
    df_comp = pd.read_excel('data/df_competencies.xlsx')
    df_comp_public = df_comp[df_comp['is_public'] == 1]  # Determino competencias a extraer

    # Selecciono los partidos de los ultimos <n_days>
    df_last_matches = determine_last_matches(df_historial_predicciones, n_days)

    # Agrego columnas 'goals_home', 'goals_away', 'result' y 'acerte'
    collect_results(df_last_matches, df_countries, df_comp_public)