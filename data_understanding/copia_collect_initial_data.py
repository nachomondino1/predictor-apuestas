# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
from selenium.webdriver.common.keys import Keys
import datetime
import time
import random
import warnings
from tqdm import tqdm
import itertools


# Que agregar?
# Barra de progreso
# sacar ip de vietnam para extraer hora bien...

def extract_partidos_whoscored(pais):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    crawler = WhoScoredCrawler(headless=False, path=None)

    df_part = pd.DataFrame()  # No hace falta definir columnas por mas que no haya extraido partidos
    df_jug_part = pd.DataFrame()  # No hace falta definir columnas por mas que no haya extraido partidos
    df_jug = pd.DataFrame(columns=['id_jug', 'nombre', 'nacionalidad', 'posicion', 'altura', 'fecha_nac'])  # No hace falta definir columnas por mas que no haya extraido partidos
    df_comp = pd.read_excel('/Users/nachomondino/Desktop/df_competencias_who_scored.xlsx')

    # Selecciono competencias del pais
    df_comp = df_comp[df_comp['pais'] == pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
    print(f' PAIS: {pais} '.center(120, '#'))

    # POR COMPETICION
    for cod_pais, cod_comp, competicion, tipo_comp in zip(df_comp['cod_pais'], df_comp['cod_competicion'], df_comp['competicion'], df_comp['tipo_comp']): # for cod_pais, cod_comp, competicion, tipo_comp in zip(['11'], ['504'], ['Cup'], ['1']):

        # Ingreso a pagina
        url_comp = f'https://www.whoscored.com/Regions/{cod_pais}/Tournaments/{cod_comp}/{pais}-{competicion.replace(" ", "-")}'
        crawler.driver.get(url_comp)  # hasta que no se carga toda la pagina, no sigue...
        print(f' Competicion: {competicion} '.center(120, '#'))
        print(url_comp)
        time.sleep(random.uniform(crawler.SEC_WAIT_MIN, crawler.SEC_WAIT_MAX))  # simular comportamiento humano
        crawler.driver.get_screenshot_as_file('screenshot.png')

        # Extraigo temporada inicial
        temp_year = crawler.extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="seasons"]/option[@selected="selected"]', text=True)

        # POR TEMPORADA
        while True:

            # Extriago urls de items
            l_urls_items = crawler.extract_urls_partidos()
            print(f' Temporada: {temp_year} '.center(120, '+'))
            print(f"Cantidad de partidos: {len(l_urls_items)}")

            # POR PARTIDO (c/u identificado con un id)
            for i, url in enumerate(l_urls_items, start=0): #  url in l_urls_items:  # Si no uso cont_part: for id in l_ids:

                # Intrego a pagina de partido
                crawler.driver.get(url)
                print(f" Partido Nº {i} ".center(120, '-'))

                # Extraigo campos
                id_part = extract_str_from_url(url, str_ini='/Matches/', str_fin='/Live/')
                d_new_row = {'id_part': id_part, 'pais': pais, 'competicion': competicion, 'temporada': temp_year,  'es_copa': tipo_comp}  # Reinicio diccionario en el que guardar datos del nuevo partido
                print(id_part, url)

                # Extraigo campos de la tabla principal
                d_new_row.update(crawler.extract_fields_from_head_table())

                dinamic_table = crawler.extract_tag(xpath='.//div[@class="panel-container"]//div[@class="match-centre-header-team"]', sec_wait=crawler.SEC_WAIT_MAX, print_fail=False)
                if dinamic_table is not None:
                    # Extraigo campos de la table dinamica
                    d_new_row.update(crawler.extract_fields_from_head_dinamic_table())

                    # Datos de jugadores --> hoja "Player statistics"  # Alineaciones + ratings
                    df_jug_match = crawler.extract_player_in_match_data()
                    df_jug_match['id_part'] = id_part  # agrego id de partido...
                    df_jug_part = pd.concat([df_jug_part, df_jug_match], axis=0)  # Guardo datos
                    df_jug_part.to_excel('/Users/nachomondino/Desktop/df_jug_match_prueba.xlsx')
                    print(df_jug_part.shape)

                    # Extraigo estadisticas del partido
                    d_new_row.update(crawler.extract_estadisticas())
                    # Estilo de juego a partir de % de sides ataques y

                '''
                # Datos de jugadores --> hoja "Player statistics"  # Alineaciones + ratings
                boton_player_stat = crawler.extract_tag(xpath='.//div[@id="sub-sub-navigation"]//a[text()="Player Statistics" and not(@class="inactive")]', sec_wait=crawler.SEC_WAIT_MAX, print_fail=False)
                no_results = crawler.extract_tag(xpath='.//div[@id="live-player-stats"]//tbody[@id="player-table-statistics-body"]/tr[text()="There are no results to display"]', sec_wait=crawler.SEC_WAIT_MIN, print_fail=False)

                if boton_player_stat is not None and no_results is None:
                    crawler.click_boton(boton_player_stat)

                    # time.sleep(2)  # A veces no agarra los jugadores del equipo visitante...
                    df_jug_match = crawler.extract_player_in_match_data()
                    df_jug_match['id_part'] = id_part  # agrego id de partido...

                    # Guardo datos
                    df_jug_part = pd.concat([df_jug_part, df_jug_match], axis=0)
                    print(df_jug_part.shape)
                    # df_jug_part.to_excel('/Users/nachomondino/Desktop/df_jug_match_prueba.xlsx')
                '''

                # GUARDADO DE DATOS EN DATAFRAME
                df_part = pd.concat([df_part, pd.DataFrame(d_new_row, index=[i])], axis=0)
                df_part.to_excel('/Users/nachomondino/Desktop/df_part_prueba.xlsx')
                print(df_part.shape)

            # Guardo partidos de la temporada por seguridad
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/data_seg/df_part_{competicion}_{temp_year}.xlsx')
            df_jug_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/data_seg/df_jug_part_{competicion}_{temp_year}.xlsx')

            # Si hay siguiente temporada
            tag_next_temp = crawler.extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="seasons"]/option[@selected="selected"]//following-sibling::option', sec_wait=crawler.SEC_WAIT_MAX, print_fail=False)
            if tag_next_temp is not None:
                temp_year = tag_next_temp.text
                crawler.click_boton(tag_next_temp)

            else:
                print("Ya no hay mas temporadas para esta competicion")
                break
            # break  # para pruebas

    # Extraigo jugadores
    # df_jug = crawler.extract_player_data_(df_jug_part)  # Falla...   for url in df_jug_match['url'].unique():  raise KeyError(key) from err # KeyError: 'url'
    # df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_prueba.xlsx')

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()


def extract_str_from_url(url, str_ini, str_fin):

    pos_ini = url.find(str_ini) + len(str_ini)
    pos_fin = url.find(str_fin)
    id = url[pos_ini: pos_fin]
    return id

class WhoScoredCrawler(Crawler):
    # Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper normal, scraper de fields especificos poro falla y scraper de proximos partidos...

    def __init__(self, headless, path):
        super().__init__(headless, path)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 1
        self.SEC_WAIT_MAX = 3

    def extract_urls_partidos(self):
        # EXTRAIGO URL DE ITEMS...

        # Definicion de variables
        l_url_items_temp = []

        # Clickeo en "Fixture" para ver partidos por mes
        boton_fixture = super().extract_tag(xpath='.//div[@class="with-single-level"]//a[text()="Fixtures"]', sec_wait=self.SEC_WAIT_MAX)
        super().click_boton(boton_fixture)
        time.sleep(random.uniform(self.SEC_WAIT_MIN, self.SEC_WAIT_MAX))

        # Por mes
        while True:

            # Extraigo partidos del mes
            l_tag_items = super().extract_tags(xpath='.//div[@id="tournament-fixture"]//div[@data-id]//a[@class="result-1 rc"]', sec_wait=self.SEC_WAIT_MAX)
            l_url_items = [tag.get_attribute('href') for tag in l_tag_items]  # Stale Element Exception (x2)
            # print(f"Cantidad de partidos: {len(l_url_items)}")
            l_url_items_temp += l_url_items

            # Si hay siguiente mes
            boton_prev_month = super().extract_tag(xpath='.//div[@class="listbox fixture-calendar"]//a[@class="previous button ui-state-default rc-l is-default"]', sec_wait=self.SEC_WAIT_MAX, print_fail=False)

            if boton_prev_month is not None:
                super().click_boton(boton_prev_month)
                time.sleep(random.uniform(self.SEC_WAIT_MIN, self.SEC_WAIT_MAX))  # Clave para que no extraiga x veces un mismo mes...
            else:
                print("Ya no hay mas partidos para esta temporada.")
                break

        return l_url_items_temp

    def extract_fields_from_head_table(self):

        d_nueva_fila = {}
        d_fields_xpath = {
            # 'fecha': {'xpath': './/div[@id="match-header"]//dt[text()="Kick off:"]/following-sibling::dd'},  # falla por tiempo de espera bajo
            'fecha': {'xpath': './/div[@id="match-header"]//dt[text()="Date:"]/following-sibling::dd'}, # falla  por tiempo de espera bajo
            'equipo_loc': {'xpath': './/div[@id="match-header"]//td[@class="team"][1]/a[@class="team-link"]',
                           'xpath_alt': './/span[@class="col12-lg-4 col12-m-4 col12-s-0 col12-xs-0 home team"]'},
            'equipo_vis': {'xpath': './/div[@id="match-header"]//td[@class="team"][2]/a[@class="team-link"]',
                           'xpath_alt': './/span[@class="col12-lg-4 col12-m-4 col12-s-0 col12-xs-0 away team"]'},
            'ft_result': {'xpath': './/div[@id="match-header"]//dt[text()="Full time:"]/following-sibling::dd'},
            'ht_result': {'xpath': './/div[@id="match-header"]//dt[text()="Half time:"]/following-sibling::dd'}
        }

        # Extraigo fecha antes para poder hacer el WebDriverWAit alto y evitar Stale Element Exception (es fundamental)
        # time.sleep(random.uniform(self.SEC_WAIT_MIN + 2, self.SEC_WAIT_MAX + 2))
        d_nueva_fila['hora'] = super().extract_tag(xpath='.//div[@id="match-header"]//dt[text()="Kick off:"]/following-sibling::dd', text=True, sec_wait=10)  # por que no sirve el Driver Wait de dspy?

        for field, dict in d_fields_xpath.items():

            xpath_alt = dict['xpath_alt'] if "xpath_alt" in dict.keys() else None
            value = super().extract_tag(xpath=dict['xpath'], text=True, xpath_alt=xpath_alt, sec_wait=self.SEC_WAIT_MIN)
            d_nueva_fila[field] = value

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_fields_from_head_dinamic_table(self):

        d_nueva_fila = {}
        d_fields_xpath = {
            'arbitro': {'xpath': './/span[@class="referee"]', 'attribute': 'title'},  # falla por tiempo de espera bajo
            'cancha': {'xpath': './/span[@class="venue"]', 'attribute': 'title'},  # falla  por tiempo de espera bajo
            'dt_loc': {'xpath': './/div[@class="match-centre-header-team" and @data-field="home"]//span[@class="manager-name"]', 'attribute': 'innerText'},
            'dt_vis': {'xpath': './/div[@class="match-centre-header-team" and @data-field="away"]//span[@class="manager-name"]', 'attribute': 'innerText'}
        }

        for field, dict in d_fields_xpath.items():
            value = super().extract_tag(xpath=dict['xpath'], attribute=dict['attribute'], sec_wait=self.SEC_WAIT_MIN)
            d_nueva_fila[field] = value

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_estadisticas(self):

        d_nueva_fila = {}
        d_fields_page = {
            'ratings': 'rating',
            'possession': 'posesion',
            'shotsTotal': {
                'shotsTotal': 'remates',
                'shotsOnTarget': 'remates_a_puerta',
                'shotsOnPost': 'remates_palos',
                'shotsOffTarget': 'remates_fuera',
                'shotsBlocked': 'remates_block'
            },
            'passSuccess': {
                'passSuccess': 'porc_pases_comp',
                'passesTotal': 'total_pases',
                'passesAccurate': 'pases_acer',
                'passesKey': 'pases_clave'
            },
            'dribblesWon': 'amagues',
            'aerialsWon': 'duelos_aereos',
            'tackleSuccessful': {
                'tackleSuccessful': 'tackles',
                'interceptions': 'intercepciones',
            },
            'cornersTotal': 'corners',
            'dispossessed': {
                'foulsCommited': 'faltas',
                'offsidesCaught': 'offsides'
            }
        }

        for field_page, value in d_fields_page.items():

            if isinstance(value, str):
                d_nueva_fila[f'{value}_loc'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]//span[@data-field="home"]', text=True, sec_wait=self.SEC_WAIT_MIN)
                d_nueva_fila[f'{value}_vis'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]//span[@data-field="away"]', text=True, sec_wait=self.SEC_WAIT_MIN)

            elif isinstance(value, dict):

                # Abro seccion de detalle haciendo click en "More" (es necesario hacerlo)
                boton_more = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]/div[2]', sec_wait=self.SEC_WAIT_MIN)
                super().click_boton(boton_more)

                # Extraigo campos en el detalle
                for sub_field_page, field in value.items():
                    d_nueva_fila[f'{field}_loc'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="{sub_field_page}"]//span[@data-field="home"]', text=True, sec_wait=self.SEC_WAIT_MIN)
                    d_nueva_fila[f'{field}_vis'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="{sub_field_page}"]//span[@data-field="away"]', text=True, sec_wait=self.SEC_WAIT_MIN)

                # Cierro seccion de detalle haciendo click en "Less" (es necesario hacerlo)
                boton_less = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]/div[2]', sec_wait=self.SEC_WAIT_MIN)
                super().click_boton(boton_less)

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_player_in_match_data(self):
        player_data = []

        start = time.time()
        SEC_WAIT = 0.001

        l_tags_jug_tit = super().extract_tags(xpath='.//div[@class="pitch-field"]/div[@data-player-id]')  #  sec_wait=self.SEC_WAIT_MAX + 2
        print(f"Cantidad de jugadores titulares: {len(l_tags_jug_tit)}")

        for tag in l_tags_jug_tit:
            id_jug = tag.get_attribute('data-player-id')
            nombre = super().extract_tag(tag_inicial=tag, xpath='.//div[@class="player-name-wrapper"]', attribute='title', sec_wait=self.SEC_WAIT_MIN)
            rating = super().extract_tag(tag_inicial=tag, xpath='./div[@class="player-stat"]', text=True, sec_wait=self.SEC_WAIT_MIN)
            condicion = "titular"
            salio = tag.get_attribute('data-is-subbed-off')
            if salio == "true":
                min_cambio = super().extract_tag(tag_inicial=tag, xpath='.//div[@class="player-key-incidents"]/div[@title="Sub out"]', attribute="data-minute", print_fail=False, sec_wait=SEC_WAIT)
            else:
                min_cambio = None

            player_data.append({'id_jug': id_jug, 'nombre_jug': nombre, 'rating': rating, 'condicion': condicion,
                                'min_cambio': min_cambio})  # 'url': url_with_id

        l_tags_jug_sup = super().extract_tags(xpath='.//div[@class="bench"]//div[@data-player-id and @data-field]')  #  sec_wait=self.SEC_WAIT_MAX + 2
        print(f"Cantidad de jugadores suplentes: {len(l_tags_jug_sup)}")
        for tag in l_tags_jug_sup:
            id_jug = tag.get_attribute('data-player-id')
            nombre = super().extract_tag(tag_inicial=tag, xpath='.//div[@class="player-name-wrapper"]', attribute='title', sec_wait=self.SEC_WAIT_MIN)
            rating = super().extract_tag(tag_inicial=tag, xpath='./div[@class="player-stat"]', text=True, sec_wait=self.SEC_WAIT_MIN)
            condicion = "suplente"
            ingreso = tag.get_attribute('data-subbed-in')
            if ingreso == "true":
                min_cambio = super().extract_tag(tag_inicial=tag, xpath='.//div[@class="player-key-incidents"]/div[@title="Sub out"]', attribute="data-minute", print_fail=False, sec_wait=SEC_WAIT)
            else:
                min_cambio = None

            player_data.append({'id_jug': id_jug, 'nombre_jug': nombre, 'rating': rating, 'condicion': condicion,
                                'min_cambio': min_cambio})  # 'url': url_with_id

        end = time.time()
        print(f'Tardo {end-start} segundos')
        df_jug_match = pd.DataFrame(player_data)
        return df_jug_match

    def extract_player_in_match_data_old(self):  # + 50 s por partido por el calculo de min_played
        player_data = []

        start = time.time()

        l_tags_jug = super().extract_tags(xpath='.//div[@id="live-player-stats"]//tbody[@id="player-table-statistics-body"]/tr')  #  sec_wait=self.SEC_WAIT_MAX + 2
        print(f"Cantidad de jugadores: {len(l_tags_jug)}")

        for tag in l_tags_jug:
            url_with_id = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]', attribute='href', sec_wait=self.SEC_WAIT_MIN)
            id_jug = extract_str_from_url(url_with_id, str_ini='/Players/', str_fin='/Show/')
            edad = super().extract_tag(tag_inicial=tag, xpath='.//td[1]/span/span[1]', text=True, sec_wait=self.SEC_WAIT_MIN)
            rating = super().extract_tag(tag_inicial=tag, xpath='.//td[@class="rating "]', text=True, sec_wait=self.SEC_WAIT_MIN)
            condicion, min_played = self.determine_condicion_and_min_played(tag)

            player_data.append({'id_jug': id_jug, 'edad': edad, 'rating': rating, 'condicion': condicion,
                                'min_played': min_played, 'url': url_with_id})

        end = time.time()
        print(f'Tardo {end-start} segundos')
        df_jug_match = pd.DataFrame(player_data)
        return df_jug_match

    def extract_player_data_(self, df_jug_match):
        # Si el id es nuevo, guardo url o la visito directamente? --> visitarla ahora no...

        df_jug = pd.DataFrame()

        # Por jugador unico
        for url in df_jug_match['url'].unique():

            # Ingreso a pagina de jugador
            self.child_driver.get(url)

            # Puedo construir df_jug despues, guardar la url del jug...
            id_jug = extract_str_from_url(url, str_ini='/Players/', str_fin='/Show/')
            nombre = super().extract_tag(xpath='.//h1//following-sibling::div//span[text()="Name: "]//parent::div', attribute='textContent')  # no cambia
            nacionalidad = super().extract_tag(xpath='.//h1//following-sibling::div//span[text()="Nationality: "]//parent::div', attribute='textContent')  # no cambia
            posicion = super().extract_tag(xpath='.//h1//following-sibling::div//span[text()="Positions: "]//following-sibling::span/span[1]', text=True)  # no cambia
            altura = super().extract_tag(xpath='.//h1//following-sibling::div//span[text()="Height: "]//parent::div', attribute='textContent')  # no cambia
            fecha_nac = super().extract_tag(xpath='.//h1//following-sibling::div//span[text()="Age: "]//following-sibling::i', text=True)  # no hace falta pues tengo la edad por partido..

            # Guardo datos
            l_data = [id_jug, nombre, nacionalidad, posicion, altura, fecha_nac]
            print(l_data)
            df_jug.loc[len(df_jug)] = l_data

            # Salgo de pagina de jugador
            # self.child_driver.back()  # no hace falta creo
            # time.sleep(2)

        return df_jug


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Selecciono pais a extraer y obtengo las competencias y su categoria
    # pais = "argentina"  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    pais = 'Argentina'

    # Extraigo partidos
    df = extract_partidos_whoscored(pais)


'''

    def determine_condicion_and_min_played(self, tag):

        SEC_WAIT =  0.001 #self.SEC_WAIT_MIN / 6
        # Determino si el jugador es suplente o titular
        suplente = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[text()=",  Sub  "]', sec_wait=SEC_WAIT, print_fail=False)
        condicion = 'titular' if suplente is None else 'suplente'

        # Determino si el jugador ingreso o salio
        cambio = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[@class="incident-wrapper"]/span', sec_wait=SEC_WAIT, print_fail=False)
        min_cambio = int(cambio.get_attribute('data-minute')) if cambio else 0
        min_cambio = min_cambio if min_cambio <= 90 else 90

        # Si fue titular
        if condicion == "titular":
            salio = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[@class="incident-wrapper"]/span[@data-type="18"]', sec_wait=SEC_WAIT, print_fail=False)
            min_played = min_cambio if salio else 90

        # Si fue suplente
        else:
            ingreso = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[@class="incident-wrapper"]/span[@data-type="19"]', sec_wait=SEC_WAIT, print_fail=False)
            min_played = 90 - min_cambio if ingreso else 0

        return condicion, min_played
        
        
    def determine_condicion_and_min_played_old(self, tag):

        # Determino si el jugador es suplente o titular
        suplente = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[text()=",  Sub  "]', sec_wait=self.SEC_WAIT_MIN/2, print_fail=False)
        condicion = 'titular' if suplente is None else 'suplente'  # 1 es titular y 0 es

        # Determino si el jugador ingreso o salio
        cambio = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[@class="incident-wrapper"]/span', sec_wait=self.SEC_WAIT_MIN/2, print_fail=False)

        # Si fue titular
        if condicion == "titular":
            salio = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[@class="incident-wrapper"]/span[@data-type="18"]', sec_wait=self.SEC_WAIT_MIN/2, print_fail=False)

            # Si salió
            if salio is not None:
                min_cambio = int(cambio.get_attribute('data-minute'))
                min_played = min_cambio if min_cambio <= 90 else 90

            # y si no salió
            else:
                min_played = 90

        # Si fue suplente
        else:
            ingreso = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="player-link"]//following-sibling::span/span[@class="incident-wrapper"]/span[@data-type="19"]', sec_wait=self.SEC_WAIT_MIN/2, print_fail=False)

            # Si ingresó
            if ingreso is not None:
                min_cambio = int(cambio.get_attribute('data-minute'))
                min_played = 90 - min_cambio if min_cambio <= 90 else min_cambio-90

            # Si no ingresó
            else:
                min_played = 0

        return condicion, min_played
'''