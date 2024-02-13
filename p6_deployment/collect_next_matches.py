import sys
sys.path.append('/Users/nachomondino/Documents/GitHub/predictor-apuestas')  # Fallaba el import de main
import pandas as pd
from p2_data_understanding.collect_initial_data.scraper_flashscore import FlashscoreCrawler
from tqdm import tqdm
from datetime import datetime, timedelta


def extract_next_matches_flashscore(pais: str, competicion: str, is_cup: int): # -> tuple[pd.DataFrame, pd.DataFrame]
    
    # Definicion de variables
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")
    df_part, df_part_jug = pd.DataFrame(), pd.DataFrame()
    pais_form = pais.lower().replace(' ', "_")
    competicion_form = competicion.lower().replace(" ", "-")  # formateo competicion para las rutas de archivo y urls
    ruta_base = f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p6_deployment/data_next_matches/{pais_form}/deployment"

     # Ingreso a pagina
    url = f'https://www.flashscore.com.ar/futbol/{pais.lower()}/{competicion_form}/partidos/'
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    print(f' Competicion: {competicion} '.center(120, '+'), f"\nURL competición: {url}")

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()    

    temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
    print(f" {temp_year} ".center(120, "-"))
    
    # Extraigo partidos (items) y sus ids --> NO PUDE EXTRAER LOS SVG.. PERO SI EL DIV DE EVENT_TIME... VER 
    l_items = crawler.extract_tags(xpath='.//div[@id="live-table"]//div[@class="sportName soccer"]//div[contains(@class, "event__match--scheduled")]', sec_wait=crawler.SEC_WAIT_MAX)  # 

    # Filtro partidos por fecha
    l_items_filt = select_items_by_date(crawler, l_items, days=7)

    # Obtengo u
    l_ids = [item.get_attribute('id') for item in l_items_filt]
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

    # if export:
    #     # Guardo partidos de la temporada (por seguridad)
    #     df_part.to_excel(f'{ruta_base}/df_part_{competicion_form}_{pais_form}.xlsx', index=False)
    #     df_part_jug.to_excel(f'{ruta_base}/df_part_jug_{competicion_form}_{pais_form}.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part, df_part_jug

def select_items_by_date(crawler, l_items: list, days: int = 7):
    """
    Selecciona los partidos que se juegan entre hoy y los proximos n dias.
    """
    # Definicion de variables
    l_items_filt = []

    # Determino fecha umbral y año actual
    fecha_umbral = datetime.now() + timedelta(days=days)
    anio_actual = datetime.now().year

    # Por partido
    for item in l_items:

        # Obtengo su fecha 
        fecha_str = crawler.extract_tag(tag_inicial=item, xpath='./div[@class="event__time"]', text=True, sec_wait=1)

        # Si el partido tiene div con fecha
        if fecha_str is not None:
            
            # Formateo fecha de str a datetime (agregandole el año pues sino toma 1900)
            fecha_str = f"{anio_actual}.{fecha_str}"
            fecha_objeto = datetime.strptime(fecha_str, "%Y.%d.%m. %H:%M")
            # print("Fecha convertida:", fecha_objeto)
        
            # Si el partido es dentro de los proximos dias
            if fecha_objeto <= fecha_umbral:
                l_items_filt.append(item)
            else:
                # Dejo de extraer partidos puesto que luego del primer partido que es posterior a fecha_umbral, todos lo son (estan ordenados por fecha en Flashscore)
                break
        else:
            print("El partido no tiene div con class 'event_time', por ende, no pude obtener fecha_str", fecha_str)

    return l_items_filt

def prueba():
    # Selecciono pais a extraer y obtengo las competencias y su categoria
    pais = 'inglaterra'  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    export = True

    print(" Collecting data... ")
    # Selecciono competencias del pais
    df_comp = pd.read_excel('./p2_data_understanding/data/df_competencias.xlsx')
    df_comp_pais = df_comp[df_comp['pais_flashscore'] == pais.capitalize()]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
    print(f' PAIS: {pais} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_pais['competicion_flashscore']}")

    # Definicion de variables
    df_part_concat, df_part_jug_concat, df_jug_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # POR COMPETICION
    for i, row in df_comp_pais.iterrows():

        print(f" Competicion: {row['competicion_flashscore']} ".center(120, '+'))
 
        if row['competicion_flashscore'] not in ["Championship"]:

            # Extraigo proximos partidos
            df_part_next_matches, df_part_jug_next_matches = extract_next_matches_flashscore(pais, row['competicion_flashscore'], row['is_cup'])
            df_part_concat = pd.concat([df_part_concat, df_part_next_matches], axis=0)
            df_part_jug_concat = pd.concat([df_part_jug_concat, df_part_jug_next_matches], axis=0)
        
    # Exporto datasets de prueba
    df_part_concat.to_excel(f'/Users/nachomondino/Desktop/df_part_next_matches.xlsx', index=False)
    df_part_jug_concat.to_excel(f'/Users/nachomondino/Desktop/df_part_jug_next_matches.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()