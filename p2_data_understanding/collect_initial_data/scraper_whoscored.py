# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
from selenium.common.exceptions import TimeoutException
import time
import random
from tqdm import tqdm
import re


class WhoScoredCrawler(Crawler):
    """
    Interacciones con sitio WhoScored.com para extraer datos.
    """
    def __init__(self, headless, path):
        super().__init__(headless, path)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 1
        self.SEC_WAIT_MAX = 3

    def extract_url_temporadas(self):
        """
        En la pagina web de una competicion, extrae año y url de cada una de sus temporadas.

        :return: Diccionario con año y url de cada temporada como key y value respectivamente. (dict)
        """
        d_temps = {}

        # Extraigo tags de temporadas
        l_tag_temps = super().extract_tags(xpath='.//div[@id="breadcrumb-nav"]//select[@id="seasons"]/option', sec_wait=10, print_fail=False)

        for tag in l_tag_temps:

            temp_year = tag.text  # año en texto del tag (e.g text = 2023)
            url_temp = f"https://www.whoscored.com{tag.get_attribute('value')}"  # url en atributo "value" del tag (e.g value = /Regions/11/Tournaments/68/Seasons/9081/Argentina-Liga-Profesional)
            d_temps[temp_year] = url_temp

        return d_temps

    def extract_url_partidos(self):
        """
        Extrae id y url de cada partido de la temporada.

        :return: Diccionario con id como key y url como value. (dict)
        """
        # Definicion de variables
        d_parts = {}
        max_clicks = 3

        # Hago click en primera etapa de la temporada (si hay), (para garantizar recorrer todas las etapas)
        boton_first_stage = super().extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="stages"]/option[1]', sec_wait=self.SEC_WAIT_MAX, print_fail=False)
        if boton_first_stage is not None:
            super().click_boton(boton_first_stage)
            time.sleep(random.uniform(self.SEC_WAIT_MIN, self.SEC_WAIT_MAX))  # Clave. Sino tenia Stale Element Exception en  "id_part = tag.get_attribute('data-id')"

        # Por etapa de temporada (si hay) (e.g. clausura, apertura, etc)
        while True:

            n_clicks = 0  # Reinicio el numero de clicks por etapa

            # Clickeo en "Fixture" para ver partidos por mes
            boton_fixture = super().extract_tag(xpath='.//div[@class="with-single-level"]//a[text()="Fixtures"]', sec_wait=self.SEC_WAIT_MAX)
            super().click_boton(boton_fixture)
            time.sleep(random.uniform(self.SEC_WAIT_MIN, self.SEC_WAIT_MAX))

            # Por mes (desde el mas reciente al mas antiguo)
            while True:

                # Extraigo partidos del mes
                l_tag_items = super().extract_tags(xpath='.//div[@id="tournament-fixture"]/div[@class="divtable-body"]/div[@data-id]//a[@class="result-1 rc"]', sec_wait=self.SEC_WAIT_MAX, print_fail=False)  #                 # l_tag_items = super().extract_tags(xpath='.//div[@id="tournament-fixture"]/div[@class="divtable-body"]/div[@data-id]', sec_wait=self.SEC_WAIT_MAX, print_fail=False)  # no evita proximos partidos... y falla al intentar ingresar a su url

                # Guardo id y url por partido
                for tag in l_tag_items:
                    id_part = super().extract_tag(tag_inicial=tag, xpath='./parent::div/parent::div', attribute='data-id')  # id_part = tag.get_attribute('data-id')
                    url = tag.get_attribute('href')  # url = super().extract_tag(tag_inicial=tag, xpath='.//a[@class="result-1 rc"]', attribute='href')  # Stale Element Exception (x2)
                    d_parts[id_part] = url

                # Hago click en flechita para ir al mes anterior (si hay)
                boton_prev_month = super().extract_tag(xpath='.//div[@class="listbox fixture-calendar"]//a[@class="previous button ui-state-default rc-l is-default"]', sec_wait=self.SEC_WAIT_MAX, print_fail=False)
                n_clicks = n_clicks + 1 if not l_tag_items else 0  # Si no encontró partidos, suma 1 --> limite de clicks en boton flechita

                if boton_prev_month is not None and n_clicks < max_clicks:
                    super().click_boton(boton_prev_month)
                    time.sleep(random.uniform(self.SEC_WAIT_MIN, self.SEC_WAIT_MAX))  # Clave para que no extraiga x veces un mismo mes...
                else:
                    # print("Ya no hay mas partidos para esta temporada.")
                    break

            # Extraigo la siguiente etapa si hay
            boton_sig_stage = super().extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="stages"]/option[@selected="selected"]/following-sibling::option[1]', sec_wait=self.SEC_WAIT_MAX, print_fail=False)

            if boton_sig_stage is not None:
                super().click_boton(boton_sig_stage)
                time.sleep(random.uniform(self.SEC_WAIT_MIN, self.SEC_WAIT_MAX))  # Clave. Sino tenia Stale Element Exception en  "id_part = tag.get_attribute('data-id')"
                # print("Hago click en siguiente etapa")
            else:
                break

        return d_parts

    def extract_basic_data_from_static_table(self):
        """
        En la pagina web de un partido, especificamente en la tabla estatica principal, extrae datos basicos del partido.

        :return: Diccionario con datos basicos de un partido: fecha, hora, equipo local, equipo visitante, resultado de
        medio tiempo y resultado final. (dict)
        """
        d_nueva_fila = {}
        d_fields_xpath = {
            'fecha': {'xpath': './/div[@id="match-header"]//dt[text()="Date:"]/following-sibling::dd'}, # falla  por tiempo de espera bajo
            'equipo_loc': {'xpath': './/div[@id="match-header"]//td[@class="team"][1]/a[@class="team-link"]',
                           'xpath_alt': './/span[@class="col12-lg-4 col12-m-4 col12-s-0 col12-xs-0 home team"]'},
            'equipo_vis': {'xpath': './/div[@id="match-header"]//td[@class="team"][2]/a[@class="team-link"]',
                           'xpath_alt': './/span[@class="col12-lg-4 col12-m-4 col12-s-0 col12-xs-0 away team"]'},
            'ft_result': {'xpath': './/div[@id="match-header"]//dt[text()="Full time:"]/following-sibling::dd'},
            'ht_result': {'xpath': './/div[@id="match-header"]//dt[text()="Half time:"]/following-sibling::dd'}
        }

        # Extraigo fecha antes para poder hacer el WebDriverWait alto y evitar Stale Element Exception (es fundamental)
        d_nueva_fila['hora'] = super().extract_tag(xpath='.//div[@id="match-header"]//dt[text()="Kick off:"]/following-sibling::dd', text=True, sec_wait=10)  # por que no sirve el Driver Wait de dspy?

        for field, dict in d_fields_xpath.items():

            xpath_alt = dict['xpath_alt'] if "xpath_alt" in dict.keys() else None
            value = super().extract_tag(xpath=dict['xpath'], text=True, xpath_alt=xpath_alt, sec_wait=self.SEC_WAIT_MIN)
            d_nueva_fila[field] = value

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_basic_data_from_dinamic_table(self):
        '''
        En la pagina web de un partido, especificamente en la tabla dinamica, extrae datos basicos del partido.

        :return: Diccionario con datos basicos de un partido: arbitro, cancha, dt local, dt visitante, promedio de edad
        de equipo local y promedio de edad de equipo visitante. (dict)
        '''
        d_nueva_fila = {}
        d_fields_xpath = {
            'arbitro': {'xpath': './/span[@class="referee"]', 'attribute': 'title'},  # falla por tiempo de espera bajo
            'cancha': {'xpath': './/span[@class="venue"]', 'attribute': 'title'},  # falla  por tiempo de espera bajo
            'dt_loc': {'xpath': './/div[@class="match-centre-header-team" and @data-field="home"]//span[@class="manager-name"]', 'attribute': 'innerText'},
            'dt_vis': {'xpath': './/div[@class="match-centre-header-team" and @data-field="away"]//span[@class="manager-name"]', 'attribute': 'innerText'},
            'prom_edad_loc': {'xpath': './/div[@class="compared" and @data-field="home"]/div[@class="average-age"]', 'attribute': 'innerText'},
            'prom_edad_vis': {'xpath': './/div[@class="compared" and @data-field="away"]/div[@class="average-age"]', 'attribute': 'innerText'}
        }

        for field, dict in d_fields_xpath.items():
            value = super().extract_tag(xpath=dict['xpath'], attribute=dict['attribute'], sec_wait=self.SEC_WAIT_MIN)
            d_nueva_fila[field] = value

        return d_nueva_fila

    def extract_player_in_match_data_from_dinamic_table(self):
        """
        En la pagina web de un partido, especificamente en la tabla dinamica, extrae datos de cada jugadores implicado
        en el partido.

        :return: DataFrame con datos de cada jugador en el partido: id del jugador, nombre, rating, condicion, minuto
        del cambio (si hubo). (DataFrame)
        """
        # Definicion de variables
        player_data = []
        SEC_WAIT = 0.001

        d = {'titular': {'class_tag': 'pitch-field', 'attr_cambio': 'data-is-subbed-off', 'title_min_cambio': 'Sub out'},
             'suplente': {'class_tag': 'bench', 'attr_cambio': 'data-subbed-in', 'title_min_cambio': 'Sub in'}}

        for titularidad, dict in d.items():

            l_tags_jug = super().extract_tags(xpath=f'.//div[@class="{dict["class_tag"]}"]//div[@data-player-id and @data-field]')

            for tag in l_tags_jug:
                id_jug = tag.get_attribute('data-player-id')
                nombre = super().extract_tag(tag_inicial=tag, xpath='.//div[@class="player-name-wrapper"]', attribute='title', sec_wait=self.SEC_WAIT_MIN)
                rating = super().extract_tag(tag_inicial=tag, xpath='./div[@class="player-stat"]', text=True, sec_wait=self.SEC_WAIT_MIN)
                figura = 1 if tag.get_attribute('class') == "player is-man-of-the-match" else 0
                condicion = tag.get_attribute('data-field')  # 'home' o 'away'
                cambio = tag.get_attribute(dict['attr_cambio'])
                if cambio == "true":
                    min_cambio = super().extract_tag(tag_inicial=tag, xpath=f'.//div[@class="player-key-incidents"]/div[@title="{dict["title_min_cambio"]}"]', attribute="data-minute", print_fail=False, sec_wait=SEC_WAIT)
                else:
                    min_cambio = None

                # print(f"'id_jug': {id_jug}, 'nombre_jug': {nombre}, 'rating': {rating}, 'condicion': {condicion}, 'titularidad': {titularidad}, 'figura': {figura}, 'min_cambio': {min_cambio}")
                player_data.append({'id_jug': id_jug, 'nombre_jug': nombre, 'rating': rating, 'condicion': condicion,
                                    'titularidad': titularidad, 'figura': figura, 'min_cambio': min_cambio})

        df_jug_match = pd.DataFrame(player_data)
        return df_jug_match

    def extract_estadisticas_from_dinamic_table(self):
        """
        En la pagina web de un partido, especificamente en la tabla dinamica, extrae las estadisticas del partido.

        :return: Diccionario con estadisticas del partido: ratings, posesion, remates y más, tanto para el equipo local
        como para el equipo visitante. (dict)
        """
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
                'passesTotal': 'pases',
                'passesAccurate': 'pases_comp',
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

    def extract_player_in_match_data_from_static_table(self):
        """
        En la pagina web de un partido, especificamente en la tabla dinamica, extrae datos de cada jugadores implicado
        en el partido.

        :return: DataFrame con datos de cada jugador en el partido: id del jugador, nombre, rating, condicion, minuto
        del cambio (si hubo). (DataFrame)
        """
        # Definicion de variables
        player_data = []
        SEC_WAIT = 0.001

        # Hago diccionario para reducir el codigo...
        l_condiciones = ['home', 'away']
        d = {'titular': {'id_tags': 'link-players-lineup', 'class_cambio': 'incidents-icon ui-icon subst-out'},
             'suplente': {'id_tags': 'link-player-substitutes', 'class_cambio': 'incidents-icon ui-icon subst-in'}
             }

        for condicion in l_condiciones:
            # print(f"Condicion: {condicion}")

            for titularidad, dict in d.items():
                # print(f"Titularidad: {titularidad}")

                # Extraigo datos de jugadores titulares en el partido
                l_tags_jug = super().extract_tags(xpath=f'.//div[@id="{dict["id_tags"]}"]/div[@class="{condicion}"]//tr')

                for tag in l_tags_jug:
                    tag_with_url = super().extract_tag(tag_inicial=tag, xpath='.//span[@class="s-off xs-off"]/a[@class="player-link "]', sec_wait=self.SEC_WAIT_MIN)
                    url_jug = tag_with_url.get_attribute('href')
                    id_jug = url_jug.split('/Players/')[1].split('/Show/')[0]  # id_jug = url_jug[url_jug.find('/Players/') + len('/Players/'): url_jug.find('/Show/')]
                    nombre = tag_with_url.text
                    cambio = super().extract_tag(tag_inicial=tag, xpath=f'.//span[@class="{dict["class_cambio"]}"]/parent::span', sec_wait=SEC_WAIT, print_fail=False)
                    min_cambio = cambio.text.replace("'", "") if cambio is not None else None

                    player_data.append({'id_jug': id_jug, 'nombre_jug': nombre, 'condicion': condicion,
                                        'titularidad': titularidad, 'min_cambio': min_cambio})

        return pd.DataFrame(player_data)


def extract_partidos_whoscored(pais):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    crawler = WhoScoredCrawler(headless=False, path=None)
    df_part = pd.DataFrame()
    df_jug_part = pd.DataFrame()
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx')
    pais_form = pais.lower().replace(' ', "_")

    # Selecciono competencias del pais
    df_comp = df_comp[df_comp['pais'] == pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
    print(f' PAIS: {pais} '.center(120, '#'))

    # POR COMPETICION
    for cod_pais, cod_comp, competicion, is_cup in zip(df_comp['cod_pais'], df_comp['cod_competicion'], df_comp['competicion'], df_comp['is_cup']): # for cod_pais, cod_comp, competicion, is_cup in zip(['11'], ['504'], ['Cup'], ['1']):

        # Ingreso a pagina
        url_comp = f'https://www.whoscored.com/Regions/{cod_pais}/Tournaments/{cod_comp}/{pais}-{competicion.replace(" ", "-")}'
        crawler.driver.get(url_comp)  # hasta que no se carga toda la pagina, no sigue...
        time.sleep(random.uniform(crawler.SEC_WAIT_MIN, crawler.SEC_WAIT_MAX))  # simular comportamiento humano
        competicion_form = competicion.lower().replace(" ", "_")
        print(f' Competicion: {competicion} '.center(120, '#'))
        # crawler.driver.get_screenshot_as_file('screenshot.png')

        # Extraigo urls de temporadas
        d_temps = crawler.extract_url_temporadas()  # d_temps = {'2010': 'https://www.whoscored.com/Regions/11/Tournaments/68/Seasons/2118/Argentina-Liga-Profesional', '2013/2014': 'https://www.whoscored.com/Regions/11/Tournaments/68/Seasons/3924/Argentina-Liga-Profesional', '2014': 'https://www.whoscored.com/Regions/11/Tournaments/68/Seasons/4308/Argentina-Liga-Profesional'}
        print(d_temps)

        # POR TEMPORADA
        for temp_year, url_temp in d_temps.items():

            # Ingreso a pagina de temporada
            crawler.driver.get(url_temp)
            print(f' Temporada: {temp_year} '.center(120, '+'))
            print(url_temp)

            # Extriago urls de partidos
            d_parts = crawler.extract_url_partidos()
            progress_bar = tqdm(total=len(d_parts.keys()), ncols=80)  # Inicializo barra de progreso

            # POR PARTIDO
            for id_part, url_part in d_parts.items():

                # Intrego a pagina de partido
                crawler.driver.get(url_part)
                d_new_row = {'id_part': id_part, 'pais': pais, 'competicion': competicion, 'temporada': temp_year,  'es_copa': is_cup}  # Reinicio diccionario en el que guardar datos del nuevo partido

                # Extraigo campos de la tabla principal
                d_new_row.update(crawler.extract_basic_data_from_static_table())

                # Extraigo campos de la table dinamica
                dinamic_table = crawler.extract_tag(xpath='.//div[@class="panel-container"]//div[@class="match-centre-header-team"]', sec_wait=crawler.SEC_WAIT_MIN, print_fail=False)
                if dinamic_table is not None:

                    # Extraigo datos basicos del head de la tabla dinamica
                    d_new_row.update(crawler.extract_basic_data_from_dinamic_table())

                    # Extraigo datos de jugadores en el partido
                    df_jug_match = crawler.extract_player_in_match_data_from_dinamic_table()
                    df_jug_match['id_part'] = id_part  # agrego id de partido...
                    df_jug_part = pd.concat([df_jug_part, df_jug_match], axis=0)  # Guardo datos
                    # df_jug_part.to_excel('/Users/nachomondino/Desktop/df_jug_match_prueba.xlsx', index=False)

                    # Extraigo estadisticas del partido
                    d_new_row.update(crawler.extract_estadisticas_from_dinamic_table())
                    # Estilo de juego a partir de % de sides ataques y

                else:
                    # Extraigo datos de jugadores si existe la seccion "Lineups"
                    lineups = crawler.extract_tag(xpath='.//div[@id="live-linueps"]', sec_wait=crawler.SEC_WAIT_MIN, print_fail=False)
                    if lineups is not None:
                        # Extraigo datos de jugadores en el partido  (sin rating)
                        df_jug_match = crawler.extract_player_in_match_data_from_static_table()
                        df_jug_match['id_part'] = id_part  # agrego id de partido...
                        df_jug_part = pd.concat([df_jug_part, df_jug_match], axis=0)  # Guardo datos

                # GUARDADO DE DATOS EN DATAFRAME
                df_part = pd.concat([df_part, pd.DataFrame(d_new_row, index=[0])], axis=0)
                progress_bar.update(1)

            # Guardo partidos de la temporada por seguridad
            temp_year_form = temp_year.replace("/", "_")
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_part_{competicion_form}_{temp_year_form}.xlsx', index=False)
            df_jug_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_jug_part_{competicion_form}_{temp_year_form}.xlsx', index=False)
            print(df_part.shape)
            print(df_jug_part.shape)

            # Cerrar la barra de progreso al finalizar
            progress_bar.close()

        # Guardo partidos de la competicion por seguridad
        df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_part_{competicion_form}_completo.xlsx', index=False)
        df_jug_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg/df_jug_part_{competicion_form}_completo.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part, df_jug_part

def extract_player_data(df_jug_part):
    """
    Extraccion de datos de jugadores

    :param df_jug_part: DataFrame con datos de cada jugador en cada partido: id del partido, id del jugador, nombre,
    rating, condicion, minuto del cambio (si hubo). (DataFrame)
    :return: DataFrame con datos de cada jugador de la competicion: id, nombre, nacionalidad, posicion
    princiapal, altura y fecha de nacimiento. (DataFrame)
    """
    # Responder en documentacion por que lo hago aparte. Es mucho mas larga la extraccion y da posibilidad a Stale Element Exception
    # Si el id es nuevo, guardo url o la visito directamente? --> visitarla ahora no..
    # Defino variables
    player_data = []
    crawler = Crawler(headless=False, path=None) # Creo objeto de clase  # crawler = WhoScoredCrawler(headless=False, path=None)
    sec_wait = 0.01  # Ojo que no sea tan bajo que no llega a cargar y falla la extraccion...
    cont_fails = 0

    # Selecciono dataframe con jugadores unicos
    df_unique = df_jug_part.drop_duplicates(subset='id_jug').reset_index(drop=True)

    # Construyo la url a partir del id del jugador y su nombre
    df_unique['url'] = 'https://www.whoscored.com/Players/' + df_unique['id_jug'].astype(str) + '/Show/' + df_unique['nombre_jug'].str.replace(' ', '-')   # https://www.whoscored.com/Players/462994/Show/Bruno-Zapelli
    progress_bar = tqdm(total=len(df_unique), ncols=80)    # Inicializo barra de progreso

    # Por jugador unico
    for _, row in df_unique.iterrows():

        # Ingreso a url del jugador
        try:
            crawler.driver.get(row['url'])

            # Extraigo datos y los guardo
            nacionalidad = crawler.extract_tag(xpath='.//h1//following-sibling::div//span[text()="Nationality: "]//parent::div', attribute='textContent', sec_wait=sec_wait, print_fail=False)
            posicion = crawler.extract_tag(xpath='.//h1//following-sibling::div//span[text()="Positions: "]//following-sibling::span/span[1]', text=True, sec_wait=sec_wait, print_fail=False)
            altura = crawler.extract_tag(xpath='.//h1//following-sibling::div//span[text()="Height: "]//parent::div', attribute='textContent', sec_wait=sec_wait, print_fail=False)
            fecha_nac = crawler.extract_tag(xpath='.//h1//following-sibling::div//span[text()="Age: "]//following-sibling::i', text=True, sec_wait=sec_wait, print_fail=False)  # no hace falta pues tengo la edad por partido..

            if nacionalidad is not None:
                l = [elem.strip() for elem in nacionalidad.split('\n') if elem.strip() != ""]  # ['', '            Nationality: ', '            Argentina ', '        '] --> "Argentina"
                if len(l) > 1:
                    nacionalidad = l[1]
                else:
                    nacionalidad = None

            if altura is not None:  # Height: 177cm--> 177cm
                altura = re.split(' |cm', altura)[1]

            player_data.append({'id_jug': row['id_jug'], 'nombre_jug': row['nombre_jug'], 'nacionalidad': nacionalidad,
                'posicion': posicion, 'altura': altura, 'fecha_nac': fecha_nac})
            # print(row['id_jug'], row['nombre_jug'], nacionalidad, posicion, altura, fecha_nac)

        except TimeoutException:
            print(f"Es posible que no exista la url del jugador {row['url']} cuyo id es {row['id_jug']} no tenga pagina")
            cont_fails += 1

        progress_bar.update(1)

    print(f"Falló el ingreso a la url de {cont_fails} jugadores")

    # Creo dataframe
    df_jug = pd.DataFrame(player_data)

    # Cierro la barra de progreso y el driver
    progress_bar.close()
    crawler.driver.close()

    return df_jug

def prueba():
    # Selecciono pais a extraer y obtengo las competencias y su categoria
    pais = 'England' # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    pais_form = pais.lower().replace(" ", "_")

    # Extraigo partidos
    df_part, df_jug_part = extract_partidos_whoscored(pais)
    df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/df_part.xlsx', index=False)
    df_jug_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/df_jug_part.xlsx', index=False)

    # Levanto df_part para extraer player data
    # df_jug_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/df_jug_part.xlsx')

    # Extraigo datos de jugadores
    # df_jug = extract_player_data(df_jug_part)
    # df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/df_jug.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()