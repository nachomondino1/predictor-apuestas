import sys
sys.path.append('/Users/nachomondino/Documents/GitHub/predictor-apuestas')  # Fallaba el import de main
import pandas as pd
from p2_data_understanding.collect_initial_data.scraper_flashscore import FlashscoreCrawler
from tqdm import tqdm

def extract_data_flashscore(pais, competicion, is_cup, export=True):
    
    # Definicion de variables
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")
    df_part, df_part_jug = pd.DataFrame(), pd.DataFrame()
    pais_form = pais.lower().replace(' ', "_")
    competicion_form = competicion.lower().replace(" ", "-")  # formateo competicion para las rutas de archivo y urls
    ruta_base = f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p6_deployment/data_next_matches/{pais_form}/data_understanding"

     # Ingreso a pagina
    url = f'https://www.flashscore.com.ar/futbol/{pais.lower()}/{competicion_form}'
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    print(f' Competicion: {competicion} '.center(120, '+'), f"\nURL competición: {url}")

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()

    temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
    print(f" {temp_year} ".center(120, "-"))
    
    # Extraigo partidos (items) y sus ids
    l_items = crawler.extract_tags(xpath='.//div[@id="live-table"]//div[@class="event__match event__match--scheduled event__match--last event__match--twoLine"]', sec_wait=crawler.SEC_WAIT_MAX)
    l_ids = [item.get_attribute('id') for item in l_items]
    progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

    # POR PARTIDO (c/u identificado con un id)
    for id_part in l_ids:

        # Ingreso a pagina de informacion del partido
        id_part = id_part[id_part.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")
        url_partido = f'https://www.flashscore.com.ar/partido/{id_part}/#/resumen-del-partido'
        crawler.driver.get(url_partido)

         # Reinicio diccionario en el que guardar datos del nuevo partido
        d_new_row_df_part = {'id_part': id_part, 'competicion': competicion, 'temporada': temp_year, 'pais': pais, 'es_copa': is_cup}
        d_new_row_df_part_jug = {'id_part': id_part}

        # EXTRACCION DE CAMPOS
        ## Extraigo campos de hoja "Resumen"
        d_new_row_df_part.update(crawler.extract_basic_data_from_resumen(extract_goles=False))

        ## Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
        if crawler.extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=crawler.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...
            d_new_row_df_part.update(crawler.extract_cuota())

        ## Si tiene hoja "Formaciones", extraigo campos
        d_new_row_df_part_jug.update(crawler.extract_bajas_pre_partido()) # FALTARIA TMB SECCION POSIBLES BAJAS.

        # GUARDADO DE DATOS EN DATAFRAME
        df_part = pd.concat([df_part, pd.DataFrame(d_new_row_df_part, index=[0])])
        df_part_jug = pd.concat([df_part_jug, pd.DataFrame(d_new_row_df_part_jug, index=[0])])
        progress_bar.update(1)

    # Cerrar la barra de progreso al finalizar
    progress_bar.close()

    if export:
        # Guardo partidos de la temporada (por seguridad)
        df_part.to_excel(f'{ruta_base}/df_part_{competicion_form}_{pais_form}.xlsx', index=False)
        df_part_jug.to_excel(f'{ruta_base}/df_part_jug_{competicion_form}_{pais_form}.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part, df_part_jug

def prueba():
    # Selecciono pais a extraer y obtengo las competencias y su categoria
    pais = 'england'  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    competicion = 'Premier League'
    is_cup = 0

    # Extraigo partidos
    extract_data_flashscore(pais, competicion, is_cup, export=True)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()