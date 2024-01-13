# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
import time
import random
from tqdm import tqdm
import warnings
import re


class FlashscoreCrawler(Crawler):
    # Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper normal, scraper de fields especificos poro falla y scraper de proximos partidos...

    def __init__(self, headless, path):
        super().__init__(headless, path)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 1  # antes 0.2 pero fallaba extraccion de campos que si estaban como goles
        self.SEC_WAIT_MAX = 2  # antes estaba en 1.5

    def accept_cookies(self):
        # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
        boton_cookies = super().extract_tag(xpath='.//button[@id="onetrust-accept-btn-handler"]')
        super().click_boton(boton_cookies)

    def extract_urls_temporadas(self):

        l_tag_temporadas = super().extract_tags(xpath='.//section[@id="tournament-page-archiv"]//div[@class="archive__row"]/div[@class="archive__season"]/a')
        l_urls_temporadas = [tag.get_attribute('href') for tag in l_tag_temporadas]
        return l_urls_temporadas

    def extract_basic_data_from_resumen(self):

        d_nueva_fila = {}
        d_field_xpath = {
            'equipo_loc': './/div[starts-with(@class, "duelParticipant__home")]',
            'equipo_vis': './/div[starts-with(@class, "duelParticipant__away")]',
            'goles_loc': './/div[@class="detailScore__wrapper"]/span[1]',
            'goles_vis': './/div[@class="detailScore__wrapper"]/span[3]',
            'arbitro': './/span[contains(text(), "Árbitro")]/following-sibling::span',
            'cancha': './/span[contains(text(), "Estadio")]/following-sibling::span'
        }

        # Extraigo el primer campo con espera para evitar extraer sin que haya cargado la pagina
        d_nueva_fila['fecha'] = super().extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=10)

        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_nueva_fila[field] = value

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_nombre(self, tag):  # Funcion auxiliar de extract_formacion()
        try:
            url_with_nombre = tag.get_attribute('href')
            nombre = url_with_nombre.split('/')[4].replace('-', ' ')
        except:
            nombre = tag.text
        return nombre

    def extract_formacion(self):

        # Definicion de variables
        d_nueva_fila = {}
        d_formaciones = {"Formaciones iniciales": "tit", 'Suplentes': 'sup', 'Jugadores reemplazados': 'sup_ing',
                         'Jugadores ausentes': 'aus'}  # 'Alineaciones iniciales': 'tit', 'Jugadores sustituidos': 'sup_ing',

        # Por formacion ("Formacion inicial", "Suplentes" y  "Ausentes")
        for formacion, titularidad in d_formaciones.items():

            # Si exista dicha formacion
            if super().extract_tag(xpath=f'.//div[text()="{formacion}"]', sec_wait=self.SEC_WAIT_MAX):

                # Extraigo listado de jugadores
                l_tags_jug_loc = super().extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]',sec_wait=self.SEC_WAIT_MIN * 2)  # Es lista de tags o None
                l_tags_jug_vis = super().extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]',sec_wait=self.SEC_WAIT_MIN * 2)  # Es lista de tags o None

                if l_tags_jug_loc:
                    l_nombres_loc = [self.extract_nombre(tag) for tag in l_tags_jug_loc]
                    for i, nombre in enumerate(l_nombres_loc):
                        d_nueva_fila[f'jug_{titularidad}_loc_{i + 1}'] = nombre

                if l_tags_jug_vis:
                    l_nombres_vis = [self.extract_nombre(tag) for tag in l_tags_jug_vis]
                    for i, nombre in enumerate(l_nombres_vis):
                        d_nueva_fila[f'jug_{titularidad}_vis_{i + 1}'] = nombre

        return d_nueva_fila

    def extract_dts(self):
        d_nueva_fila = {}

        # Extraigo entrenadores
        dt_loc_tag = super().extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]//a[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN)
        dt_vis_tag = super().extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]//a[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN)

        if dt_loc_tag:
            d_nueva_fila['dt_loc'] = self.extract_nombre(dt_loc_tag)

        if dt_vis_tag:
            d_nueva_fila['dt_vis'] = self.extract_nombre(dt_vis_tag)

        return d_nueva_fila

    def extract_estadisticas(self):

        d_nueva_fila = {}
        d_estadisticas = {'posesion': 'Posesión de balón', 'remates': 'Remates', 'remates_a_puerta': 'Remates a puerta',
                          'tarjetas_amarillas': 'Tarjetas amarillas', 'faltas': 'Faltas', 'pases': 'Pases totales',
                          'pases_comp': 'Pases completados', 'offsides': 'Fueras de juego', 'ataques': 'Ataques',
                          'ataques_pelig': 'Ataques peligrosos'}

        # Por estadistica (posesion, remates, etc)
        for field, field_page in d_estadisticas.items():

            # Extraigo dicha estadistica tanto para el equipo local como para el visitante
            d_nueva_fila[f'{field}_loc'] = super().extract_tag(xpath=f'.//strong[text()="{field_page}"]//parent::div//preceding-sibling::div', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_nueva_fila[f'{field}_vis'] = super().extract_tag(xpath=f'.//strong[text()="{field_page}"]//parent::div//following-sibling::div', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_cuota(self):

        # Definicion de variables
        d_nueva_fila = {}
        l_odds = ['odds_loc', 'odds_emp', 'odds_vis']

        # Extraer las cuotas en una lista
        odds_elements = super().extract_tags(xpath='.//div[@class="cellWrapper"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)

        # Por cuota (local, emp y vis)
        for i, columna in enumerate(l_odds, start=0):

            odds_element = odds_elements[i]
            odds_str = odds_element.get_attribute('title')  # odds_str = super().extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]', attribute='title', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # 3.00 » 2.25

            # Si cambió durante el partido
            if '»' in odds_str:
                d_nueva_fila[columna] = odds_str.split('»')[0].strip()  # 3.00

            # Si no cambió durante el partido
            else:
                d_nueva_fila[columna] = super().extract_tag(tag_inicial=odds_element, xpath='.//span[@class="oddsValueInner"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # d_nueva_fila[columna] = super().extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]//span[@class="oddsValueInner"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)

        # print(d_nueva_fila)
        return d_nueva_fila

def extract_data_flashscore(pais):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    crawler = FlashscoreCrawler(headless=True, path=None)
    df_part, df_part_jug= pd.DataFrame(), pd.DataFrame()
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx')
    print(f' PAIS: {pais} '.center(120, '#'))

    # Selecciono competencias del pais
    df_comp = df_comp[df_comp['pais'] == pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
    pais_form = df_comp.iloc[0]['pais'].lower().replace(' ', "_")
    print(f"Competiciones a extraer: {df_comp['competicion']}")

    # POR COMPETICION
    for competicion, is_cup in zip(df_comp['competicion'], df_comp['is_cup']):

        # Ingreso a pagina
        competicion_form = competicion.lower().replace(" ", "-")  # formateo competicion para las rutas de archivo y urls
        url = f'https://www.flashscore.es/futbol/{pais.lower()}/{competicion_form}/archivo/'
        crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
        print(f' Competicion: {competicion}. URL: {url} '.center(120, '+'))

        # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
        crawler.accept_cookies()

        # Extraigo urls de las distintas temporadas (años) de la competicion
        l_urls_temporadas = crawler.extract_urls_temporadas()
        print(f'Cantidad de temporadas: {len(l_urls_temporadas)}')

        # POR TEMPORADA
        for url_temp in l_urls_temporadas[:10]:  # De mas reciente a menos reciente

            # Ingreso a pagina de temporada e imprimo año de la temporada
            crawler.driver.get(url_temp)
            temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
            print(f" {temp_year} ".center(120, "-"))

            # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
            while True:
                boton_mostrar = crawler.extract_tag(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=crawler.SEC_WAIT_MAX * 3)
                if crawler.click_boton(boton_mostrar) is False:
                    break

            # Extraigo partidos (items) y sus ids
            l_items = crawler.extract_tags(xpath='.//div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]', sec_wait=crawler.SEC_WAIT_MAX*5)
            l_ids = [item.get_attribute('id') for item in l_items]
            print(f"Partidos recolectados de la temporada {temp_year} (e.g. en premier league deberian ser 380): {len(l_ids)}")
            progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

            # POR PARTIDO (c/u identificado con un id)
            for id in l_ids:

                # Ingreso a pagina de informacion del partido
                id = id[id.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")
                crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

                # Reinicio diccionario en el que guardar datos del nuevo partido
                d_nueva_fila = {'id_part': id, 'competicion': competicion, 'temporada': temp_year, 'pais': pais, 'es_copa': is_cup}
                d_nueva_fila_2 = {'id_part': id}

                # EXTRACCION DE CAMPOS
                # Extraigo campos de hoja "Resumen"
                d_nueva_fila.update(crawler.extract_basic_data_from_resumen())

                # Si tiene hoja "Estadisticas", extraigo campos
                boton_estadisticas = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Estadísticas"]', sec_wait=crawler.SEC_WAIT_MAX, print_fail=False)
                if crawler.click_boton(boton_estadisticas) is not False:
                    time.sleep(random.uniform(crawler.SEC_WAIT_MIN + 3, crawler.SEC_WAIT_MAX + 3))  # Falla el campo posesion_loc puesto que es el primero en ser extraido y aun no cargo...
                    d_nueva_fila.update(crawler.extract_estadisticas())

                # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                if crawler.extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=crawler.SEC_WAIT_MIN) is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...
                    d_nueva_fila.update(crawler.extract_cuota())

                # Si tiene hoja "Formaciones", extraigo campos
                boton_formaciones = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Formaciones" or text()="Alineaciones"]', sec_wait=crawler.SEC_WAIT_MAX)
                if crawler.click_boton(boton_formaciones) is not False:
                    time.sleep(random.uniform(crawler.SEC_WAIT_MIN + 3, crawler.SEC_WAIT_MAX + 3)) # Por posible falla en el primer campo a extraer  # WebDriverWait(crawler.driver, SEC_WAIT_LONG + 3).until(EC.presence_of_element_located((By.XPATH, './/div[@class="preMatchTabCnt preMatchTabCnt1"]')))
                    d_nueva_fila.update(crawler.extract_dts())
                    d_nueva_fila_2.update(crawler.extract_formacion())

                # GUARDADO DE DATOS EN DATAFRAME
                df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
                df_part_jug = pd.concat([df_part_jug, pd.DataFrame(d_nueva_fila_2, index=[0])])
                progress_bar.update(1)

            # Cerrar la barra de progreso al finalizar
            progress_bar.close()

            # Guardo partidos de la temporada (por seguridad)
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_part/{competicion_form}_{temp_year.replace("/", "_")}_{pais_form}.xlsx', index=False)
            df_part_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_part_jug/{competicion_form}_{temp_year.replace("/", "_")}_{pais_form}.xlsx', index=False)

        # Guardo partidos de la competicion (por seguridad)
        df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_part/{competicion_form}_{pais_form}.xlsx', index=False)
        df_part_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_part_jug/{competicion_form}_{pais_form}.xlsx', index=False)

    # Guardado datos a nivel pais
    df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/df_part.xlsx', index=False)
    df_part_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/df_part_jug.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part, df_part_jug

def prueba():
    # Selecciono pais a extraer y obtengo las competencias y su categoria
    # pais = 'Argentina' # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    pais = 'England'

    # Extraigo partidos
    extract_data_flashscore(pais)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()