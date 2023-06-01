# Importo librerias
import pandas as pd
import datetime
import time
import random
import warnings
from dspy.data_understanding.web_scraping.selenium import Crawler


def extract_cuota(crawler, SEC_WAIT, i):  # Probar funcion... NO ESTA VERIFICADA...  # Puedo volver a la anterior, solo fallaron 41 cuotas por "Cuotas retiradas por la casa de apuestas."

    # Busco odd suponiendo que cambio durante el partido
    try:
        text_with_odds = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]', attribute='title', sec_wait=SEC_WAIT)  # 3.00 » 2.25
        odds = float(text_with_odds.split('»')[0].strip())  # 3.00
        return odds

    # Si la odd no cambio durante el partido, o bien, aparece "Cuotas retiradas por la casa de apuestas."
    except ValueError:  # could not convert string to float:
        try:
            odds_str = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]//span[@class="oddsValueInner"]', text=True, sec_wait=SEC_WAIT)
            odds = float(odds_str)
            return odds

        except ValueError:  # could not convert string to float
            print("Fallo extraccion de la cuota")
            return None

def extract_flashscore():
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    crawler = Crawler(headless=True, path=None) # Creo objeto de clase CrawlerActions()

    df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_partido_argentina.xlsx')

    # Por partido
    for i in range(len(df_part)):
        print(f"Partido Nº{i+1}")
        id = df_part.loc[i, 'id']

        # Reinicio diccionario en el que guardar la nueva fila
        d_nueva_fila = {}

        # Ingreso a pagina de informacion del partido
        crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

        # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
        if crawler.extract_tag(xpath='.//div[@class="oddsRow"]') is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...

            d_nueva_fila['odds_loc'] = extract_cuota(crawler, SEC_WAIT, i=1)
            d_nueva_fila['odds_emp'] = extract_cuota(crawler, SEC_WAIT, i=2)
            d_nueva_fila['odds_vis'] = extract_cuota(crawler, SEC_WAIT, i=3)
            print(id, d_nueva_fila['odds_loc'], d_nueva_fila['odds_emp'], d_nueva_fila['odds_vis'])

        # GUARDADO DE DATOS EN DATAFRAME
        df_part.loc[i, 'odds_loc'] = d_nueva_fila['odds_loc']
        df_part.loc[i, 'odds_emp'] = d_nueva_fila['odds_emp']
        df_part.loc[i, 'odds_vis'] = d_nueva_fila['odds_vis']

    # Guardado de archivo excel en computadora
    df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_partido_argentina_prueba.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()


extract_flashscore()