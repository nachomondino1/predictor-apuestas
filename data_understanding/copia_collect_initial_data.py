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


def extract_id_from_url(url):

    pos_ini = url.find('/Matches/') + len('/Matches/')
    pos_fin = url.find('/Live/')
    id = url[pos_ini: pos_fin]
    return id

def extract_partidos_whoscored(pais):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    crawler = WhoScoredCrawler(headless=False, path=None)  # Hacer que no falle con headless=True
    df = pd.DataFrame()  # No hace falta definir columnas por mas que no haya extraido partidos
    df_comp = pd.read_excel('/Users/nachomondino/Desktop/df_competencias_who_scored.xlsx')
    print(f' PAIS: {pais} '.center(120, '#'))

    # Selecciono competencias del pais
    df_comp = df_comp[df_comp['pais'] == pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]

    # POR COMPETICION
    for cod_pais, cod_comp, competicion, tipo_comp in zip(df_comp['cod_pais'], df_comp['cod_competicion'], df_comp['competicion'], df_comp['tipo_comp']):

        # Ingreso a pagina
        url = f'https://www.whoscored.com/Regions/{cod_pais}/Tournaments/{cod_comp}/{pais}-{competicion.replace(" ", "-")}'
        crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
        print(f' Competicion: {competicion} '.center(120, '+'))
        print(url)
        time.sleep(random.uniform(4,10))  # simular comportamiento humano

        # Extraigo temporada inicial
        temp_year = crawler.extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="seasons"]/option[@selected="selected"]', text=True)

        # POR TEMPORADA
        while True:

            # Extriago urls de items
            l_urls_items = crawler.extract_urls_items()
            print(f' Temporada: {temp_year} '.center(120, '+'))
            print(f"Cantidad de partidos: {len(l_urls_items)}")

            # POR PARTIDO (c/u identificado con un id)
            #  url in l_urls_items:  # Si no uso cont_part: for id in l_ids:
            for i, url in enumerate(l_urls_items, start=0):  # Si no uso cont_part: for id in l_ids:

                # Intrego a pagina de partido
                crawler.driver.get(url)

                # Extraigo campos
                id = extract_id_from_url(url)
                print(id, url)
                d_new_row = {'id': id, 'pais': pais, 'competicion': competicion, 'temporada': temp_year,  'es_copa': tipo_comp}  # Reinicio diccionario en el que guardar datos del nuevo partido

                # Extraigo campos de la tabla principal
                d_new_row.update(crawler.extract_fields_from_head_table())

                # Extraigo campos de la table dinamica
                d_new_row.update(crawler.extract_fields_from_dinamic_table())

                # Extraigo estadisticas del partido  # Dribbles, aerials won, tackles? #  'ataques': 'Ataques', 'ataques_pelig': 'Ataques peligrosos'
                d_new_row.update(crawler.extract_estadisticas_chat_GPT())


                # Datos de jugadores --> hoja "Player statistics"
                # Alineaciones + ratings
                boton_player_stat = crawler.extract_tag(xpath='.//div[@id="sub-sub-navigation"]//a[text()="Player Statistics"]')
                crawler.click_boton(boton_player_stat)

                nombre = crawler.extract_tag(xpath='')
                rating = crawler.extract_tag(xpath='')
                nacionalidad = crawler.extract_tag(xpath='')  # no cambia
                equipo_act = crawler.extract_tag(xpath='')
                posicion = crawler.extract_tag(xpath='')  # no cambia
                edad = crawler.extract_tag(xpath='')
                condicion = crawler.extract_tag(xpath='')
                key_events = crawler.extract_tag(xpath='')


                '''
                # odds --> hoja "Betting"
                '''

                # GUARDADO DE DATOS EN DATAFRAME
                df = pd.concat([df, pd.DataFrame(d_new_row, index=[i])], axis=0)
                print(df)
                df.to_excel('/Users/nachomondino/Desktop/df_prueba.xlsx')

            '''
            # Si hay siguiente temporada
            tag_next_temp = crawler.extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="seasons"]/option[@selected="selected"]//following-sibling::option')
            if tag_next_temp is not None:
                temp_year = tag_next_temp.text
                crawler.click_boton(tag_next_temp)
            else:
                print("Ya no hay mas temporadas para esta competicion")
                break
            '''
            break

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()


class WhoScoredCrawler(Crawler):
    # Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper normal, scraper de fields especificos poro falla y scraper de proximos partidos...

    def __init__(self, headless, path):
        super().__init__(headless, path)
        self.child_driver = self.driver
        # Definir SEC_WAIT, SEC_WAIT_LONG como atrib..

    def extract_urls_items(self):
        # EXTRAIGO URL DE ITEMS...

        # Definicion de variables
        l_url_items_temp = []

        time.sleep(random.uniform(1, 2))
        # Clickeo en "Fixture" para ver partidos por mes
        boton_fixture = super().extract_tag(xpath='.//div[@class="with-single-level"]//a[text()="Fixtures"]')
        super().click_boton(boton_fixture)
        time.sleep(random.uniform(2, 4))

        # Por mes
        while True:

            # Extraigo partidos del mes
            l_tag_items = super().extract_tags(xpath='.//div[@id="tournament-fixture"]//a[@class="result-1 rc"]')
            l_url_items = [tag.get_attribute('href') for tag in l_tag_items]
            # print(f"Cantidad de partidos: {len(l_url_items)}")
            l_url_items_temp += l_url_items

            # Si hay siguiente mes
            boton_prev_month = super().extract_tag(xpath='.//div[@class="listbox fixture-calendar"]//a[@class="previous button ui-state-default rc-l is-default"]')

            if boton_prev_month is not None:
                super().click_boton(boton_prev_month)
                time.sleep(random.uniform(2, 4))  # Clave para que no extraiga x veces un mismo mes...

            else:
                print("Ya no hay mas partidos para esta temporada.")
                break

        return l_url_items_temp

    def extract_fields_from_head_table(self):

        SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
        d_nueva_fila = {}
        d_fields_xpath = {
            'fecha': './/div[@id="match-header"]//tr[2]/td[2]/div[3]//dd[2]',  # falla por tiempo de espera bajo
            'hora': './/div[@id="match-header"]//tr[2]/td[2]/div[3]//dd[1]', # falla  por tiempo de espera bajo
            'equipo_loc': './/div[@id="match-header"]//td[@class="team"][1]',
            'equipo_vis': './/div[@id="match-header"]//td[@class="team"][2]',
            'ft_result': './/div[@id="match-header"]//td[@class="result"]',
            'ht_result': './/div[@id="match-header"]//tr[2]/td[2]/div[2]//dd[1]',
        }

        for field, xpath in d_fields_xpath.items():
            value = super().extract_tag(xpath=f"{xpath}", text=True)
            d_nueva_fila[field] = value

        return d_nueva_fila

    def extract_fields_from_dinamic_table(self):

        d_nueva_fila = {}
        d_fields_xpath = {
            'arbitro': {'xpath': './/span[@class="referee"]', 'title': False, 'attribute': 'title'},  # falla por tiempo de espera bajo
            'cancha': {'xpath': './/span[@class="venue"]', 'title': False, 'attribute': 'title'},  # falla  por tiempo de espera bajo
            'dt_loc': {'xpath': './/span[@class="manager-name"][1]', 'title': True, 'attribute': None},
            'dt_vis': {'xpath': './/span[@class="manager-name"][2]', 'title': True, 'attribute': None},
        }

        for field, dict in d_fields_xpath.items():

            value = super().extract_tag(xpath=f"{dict['xpath']}", text=dict['title'], attribute=dict['attribute'])
            d_nueva_fila[field] = value

        # arbitro = super().extract_tag(xpath='.//span[@class="referee"]', attribute='title')
        # cancha = super().extract_tag(xpath='.//span[@class="venue"]', attribute='title')
        # dt_loc = super().extract_tag(xpath='.//span[@class="manager-name"][1]', text=True)
        # dt_vis = super().extract_tag(xpath='.//span[@class="manager-name"][2]', text=True)

        return d_nueva_fila

    def extract_estadisticas_chat_GPT(self):

        SEC_WAIT = 0.2
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
                d_nueva_fila[f'{value}_loc'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]//span[@data-field="home"]', text=True, sec_wait=SEC_WAIT)
                d_nueva_fila[f'{value}_vis'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]//span[@data-field="away"]', text=True, sec_wait=SEC_WAIT)

            elif isinstance(value, dict):

                # Abro seccion de detalle haciendo click en "More" (es necesario hacerlo)
                boton_more = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]/div[2]')
                super().click_boton(boton_more)

                # Extraigo campos en el detalle
                for sub_field_page, field in value.items():
                    d_nueva_fila[f'{field}_loc'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="{sub_field_page}"]//span[@data-field="home"]', text=True, sec_wait=SEC_WAIT)
                    d_nueva_fila[f'{field}_vis'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="{sub_field_page}"]//span[@data-field="away"]', text=True, sec_wait=SEC_WAIT)

                # Cierro seccion de detalle haciendo click en "Less" (es necesario hacerlo)
                boton_less = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{field_page}"]/div[2]')
                super().click_boton(boton_less)
        return d_nueva_fila

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Selecciono pais a extraer y obtengo las competencias y su categoria
    # pais = "argentina"  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    pais = 'Argentina'

    # Extraigo partidos
    df = extract_partidos_whoscored(pais)
