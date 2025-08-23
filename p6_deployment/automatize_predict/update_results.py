import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import os
from dotenv import load_dotenv
import pandas as pd
import datetime
from p2_data_understanding.collect_initial_data.scraper_flashscore import FlashscoreCrawler
from p3_data_preparation.construct_data import determine_result
from p4_modeling.betting_strategy import BettingStrategy
from tqdm import tqdm


def determine_last_matches(df: pd.DataFrame, n_days: float = 1):
    """
    Selecciono los partidos jugados en los ultimos <n_days>.
    """
    fecha_hoy = datetime.datetime.now()
    fecha_hoy_ajustada = fecha_hoy - datetime.timedelta(hours=2) # Para confirmar que el partido ya terminó
    fecha_limite = fecha_hoy_ajustada - datetime.timedelta(days=n_days)
    print(f"Fechas a filtrar: {fecha_limite} --> {fecha_hoy}")

    # Seleccionar los partidos de ultimo/s dia/s
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
    df_filt = df[(df['date'] > fecha_limite) & (df['date'] <= fecha_hoy_ajustada)]
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
            df_results_competition = drop_suspended_matches(df_results_competition) # df_results_competition.dropna(subset=["goals_home"])

            # Guardo results de competencia
            df_results = pd.concat([df_results, df_results_competition], axis=0)
            logger.info(df_results)
        
    # Si hay algun partido:
    if len(df_results) > 0:

        # Agrego columnas 'goals_home' y 'goals_away' a predicciones.xlsx
        df.loc[df_results.index, ['goals_home', 'goals_away']] = df_results
        df = df.dropna(subset=['goals_home', 'goals_away'])  # Eliminar filas con NaN en goles

        # Determino ganador y si acerté
        df_pred_with_result = determine_result(df, var_resp='result')
        df_pred_with_result = bs.determine_winning_bets(df_pred_with_result)  # Determino acierto o fallo
        df_pred_with_result.index.name = 'id_match'  # Es importante para la base de datos MySQL
        logger.info(df_pred_with_result)

        # Exportar dataset
        # df_pred_with_result.to_excel('data/predicciones.xlsx') # Los partidos que tiene son de historial_predicciones en realidad pero uso predicciones.xlsx para poder activar dispatch y enviar datos a VPS?
        logger.critical(f"Se recolecto el resultado de {len(df_results)} partidos.")
    else:
        logger.warning("Se evitó el update de resultados puesto que no se detectaron partidos jugados ayer")
        df_pred_with_result = pd.DataFrame()
    
    return df_pred_with_result

def extract_matches_result(country: str, competition: str, l_ids:list):

    # DEFINCION DE PARAMETROS & VARIABLES
    crawler = FlashscoreCrawler() # headless=False
    df = pd.DataFrame()

    # Formateo variables para construir url
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(" ", "-")  # formateo competition para las rutas de archivo y urls
    url = f'https://www.flashscore.com/football/{country_form}/{competition_form}/'
    crawler.driver.get(url)
    print(f'URL competición: {url}')

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()

    # Click en hoja "Results"
    crawler.click_results_page()
    crawler.click_show_more_matches() # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la season)
    progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

    # POR MATCH (c/u identificado con un id)
    for id_match in l_ids:

        # Ingreso a pagina de informacion del match
        url_match = f'https://www.flashscore.com/match/{id_match}/#/match-summary'
        crawler.driver.get(url_match)

        # Extraigo todos los datos del partido
        d_row = {}
        d_row.update(crawler.extract_result()) # {'goals_home': 2, 'goals_away': 1}
            
        # Guardo datos del partido
        df = pd.concat([df, pd.DataFrame(d_row, index=[id_match])])
        progress_bar.update(1)

    # Finalizada la extraccion, cierro el web browser automático
    progress_bar.close()
    crawler.driver.close()
    return df

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
        n_days = 10
    
    elif env == 'prod':
        n_days = float(sys.argv[1])  # Numero de dias maximo desde hoy para extraer partidos (e.g. 7)

    # Levanto datasets
    df_historial_predicciones = pd.read_excel('data/historial_predicciones.xlsx', index_col=0)  # Para garantizar que tengo todas las predicciones.  
    df_countries = pd.read_excel('data/df_countries.xlsx')
    df_comp = pd.read_excel('data/df_competencies.xlsx')
    df_comp_public = df_comp[df_comp['is_public'] == 1]  # Determino competencias a extraer

    # Selecciono los partidos de los ultimos <n_days>
    df_last_matches = determine_last_matches(df_historial_predicciones, n_days)
    logger.info(df_last_matches)

    # Agrego columnas 'goals_home', 'goals_away', 'result' y 'acerte'
    df = collect_results(df_last_matches, df_countries, df_comp_public)

    # Actualizo historial_predicciones con resultados
    if len(df) > 0:
        df_nuevos_resultados = df.loc[:, ['goals_home', 'goals_away', 'result', 'acerte']]
        df_historial_predicciones.loc[df_nuevos_resultados.index, ['goals_home', 'goals_away', 'result', 'acerte']] = df_nuevos_resultados
        df_historial_predicciones.to_excel('data/historial_predicciones.xlsx')  # Para garantizar que tengo todas las predicciones.  
