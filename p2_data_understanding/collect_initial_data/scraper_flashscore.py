# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p2_data_understanding.collect_initial_data.web_scraping_selenium import Crawler
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta
import re

class FlashscoreCrawler(Crawler):
    """
    Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper
    normal, scraper de fields especificos poro falla y scraper de proximos partidos...
    Contiene todo los xpath.
    """
    def __init__(self, headless: bool = True, path: str = None, browser: str = "Chrome", _print: bool = False):
        super().__init__(headless, path, browser)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 0.8  # Espera para elementos que muchas veces no estan # con 0.2 fallaba extraccion de campos que si estaban como goals
        self.SEC_WAIT_MED = 1.5  # Espera para elementos que casi siempre estan
        self.SEC_WAIT_MAX = 5  # Espera para elementos que casi siempre estan
        self._print = _print # Para imprimir el funcionamiento de cada funcion y poder hacer pruebas...

    def accept_cookies(self):
        """
        Aceptar ventana emergente de cookies
        :return:
        """
        # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
        boton_cookies = super().extract_tag(xpath='.//button[@id="onetrust-accept-btn-handler"]')
        super().click_boton(boton_cookies)

    def extract_urls_seasons(self, n_seasons_max):
        """
        Extrae listado de urls de las seasons de una competition.
        :return: List. Urls de seasons de la competition.
        """
        # Extraigo tags
        l_tag_seasons = super().extract_tags(xpath='.//section[@id="tournament-page-archiv"]//div[@class="archive__row"]/div[@class="archive__season"]/a')

        # Obtengo href de cada tag y lo hago lista
        l_urls_seasons = [tag.get_attribute('href') for tag in l_tag_seasons]

        # Selecciono hasta la temporada maxima a extraer
        if n_seasons_max !=0:
            l_urls_seasons = l_urls_seasons[:n_seasons_max]
        return l_urls_seasons

    def extract_season_year(self):
        
        season_year = super().extract_tag(xpath='.//div[@class="heading__info"]', text=True)

        if isinstance(season_year, str):
            season_year = season_year.replace("/", "_")
        else:
            print("The season is not a string. Probably it failed the extraction.")
        return season_year

    def click_results_page(self):
        xpath_button = './/div[@class="container__heading"]//div[@class="tabs__group"]/a[@class="tabs__tab results"]'
        boton_mostrar = super().extract_tag(xpath=xpath_button, sec_wait=self.SEC_WAIT_MAX*2, print_fail=False)
        super().click_boton(boton_mostrar)

    def click_show_more_matches(self):
        """
        Click en boton "Show more matches" hasta que ya no haya mas. Es decir, carga todos los partidos.
        """
        n_clicks = 0
        while True:
            xpath_button = './/div[@id="live-table"]//div[@class="leagues--static event--leagues summary-results" or @class="leagues--static event--leagues results"]//a[@class="event__more event__more--static"]' 
            boton_mostrar = super().extract_tag(xpath=xpath_button, sec_wait=self.SEC_WAIT_MAX*2, print_fail=False)

            if super().click_boton(boton_mostrar) is False:
                print(f"Hizo {n_clicks} clicks en el boton 'Show More Matches'.")
                break

            n_clicks += 1
    
    def extract_id_matches(self):
        """
        Obtiene todos los partidos de la temporada y luego sus ids
        """
        l_items = super().extract_tags(xpath='.//div[@id="live-table"]//div[starts-with(@class, "event__match ")]', sec_wait=self.SEC_WAIT_MAX)
        l_ids = [item.get_attribute('id') for item in l_items]
        l_ids_clean = clean_id(l_ids)
        return l_ids_clean
    
    def extract_id_next_matches(self, n_days):
            
        # Extraigo partidos (items) y sus ids --> NO PUDE EXTRAER LOS SVG.. PERO SI EL DIV DE EVENT_TIME... VER 
        l_items = super().extract_tags(xpath='.//div[@id="live-table"]//div[@class="sportName soccer"]//div[contains(@class, "event__match--scheduled")]', sec_wait=self.SEC_WAIT_MAX)
        # print(f"Cantidad de proximos partidos en total: {len(l_items)}")

        # Filtro partidos por fecha
        l_items_filt = self.select_items_by_date(l_items, n_days)
        # print(f"Cantidad de proximos partidos dentro de {n_days}: {len(l_items_filt)}")

        # Obtengo ids
        l_ids = [item.get_attribute('id') for item in l_items_filt]
        l_ids_clean = clean_id(l_ids)
        # print(f"Cantidad de ids extraidos dentro de {n_days}: {len(l_ids_clean)}")
        return l_ids_clean

    def extract_match_data(self, next_matches:bool = False):
        """
        Extrae datos del partido siendo este un partido ya jugado.
        """
        # Reinicio diccionario en el que guardar datos del nuevo match
        d_row_match = {}
        d_row_match_player = {}
        d_row_match_odds = {}

        # EXTRACCION DE CAMPOS
        ## Extraigo campos de hoja "summary"
        if self._print:
            print("Obtengo datos de hoja 'Summary'...")
        ### Match information
        d_row_match.update(self.extract_match_information())
        ### Goals
        if not next_matches:
            d_row_match.update(self.extract_result())
        ### Teams
        d_row_match.update(self.extract_teams())

        ## Si tiene hoja "stats", extraigo campos
        if self._print:
            print("Obtengo datos de hoja 'Stats'...")
        if not next_matches:
            boton_stats = super().extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Stats"]', sec_wait=self.SEC_WAIT_MED, print_fail=True)
            if super().click_boton(boton_stats) is not False:
                d_row_match.update(self.extract_stats())

        ## Si existe la seccion "odds pre-match", extraigo odds de Bet365
        if super().extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=self.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las oddss pero no hay valores...
            d_row_match_odds.update(self.extract_odds())
        
        ## Si tiene hoja "Formations", extraigo campos
        if self._print:
            print("Obtengo datos de hoja 'Lineups'...")
        boton_formations = super().extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Lineups"]', sec_wait=self.SEC_WAIT_MED, print_fail=True)
        if super().click_boton(boton_formations) is not False:
            ### Alineaciones titulares, suplentes y ausentes
            d_row_match_player.update(self.extract_lineups())
            
            ### Coaches
            d_row_match.update(self.extract_coaches())

        # Si no hay boton lineup y es un proximo partido 
        elif next_matches:
            d_row_match_player.update(self.extract_bajas_pre_partido()) # FALTARIA TMB SECCION POSIBLES BAJAS.
        
        return d_row_match, d_row_match_player, d_row_match_odds

    def extract_match_information(self):
        """
        Extrae datos basicos de un match como equipos, fecha, cancha, goals, etc.
        :return: Diccionario.
        """
        d_row = {}
        d_field_xpath = {
            'referee': './/div[@data-testid="wcl-summaryMatchInformation"]//span[contains(text(), "Referee")]/parent::div/following-sibling::div', # Intente el svg (para evitar el texto "Referee") pero no lo encuentra
            'venue': './/div[@data-testid="wcl-summaryMatchInformation"]//span[contains(text(), "Venue")]/parent::div/following-sibling::div', 
            'capacity': './/div[@data-testid="wcl-summaryMatchInformation"]//span[contains(text(), "Capacity")]/parent::div/following-sibling::div', 
            'attendance': './/div[@data-testid="wcl-summaryMatchInformation"]//span[contains(text(), "Attendance")]/parent::div/following-sibling::div'
        }

        # Extraigo el primer campo con espera para evitar extraer sin que haya cargado la pagina
        d_row['date'] = super().extract_tag(xpath='.//div[@class="duelParticipant"]/div[@class="duelParticipant__startTime"]', text=True, sec_wait=self.SEC_WAIT_MED)
        
        # Por campo a extraer
        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_row[field] = value

        if self._print:
            print(f"Extracting match information: {d_row}")
        return d_row

    def extract_teams(self):
        """
        Extrae id y nombre de los dos equipos que jugaron o juegan el partido.
        """
        d_row = {}

        tag_team_home =  super().extract_tag(xpath='.//div[@class="duelParticipant"]/div[starts-with(@class, "duelParticipant__home")]', sec_wait=self.SEC_WAIT_MIN, print_fail=True)
        tag_team_away =  super().extract_tag(xpath='.//div[@class="duelParticipant"]/div[starts-with(@class, "duelParticipant__away")]', sec_wait=self.SEC_WAIT_MIN, print_fail=True)

        if tag_team_home is not None:
            url_team_home = super().extract_tag(tag_inicial=tag_team_home, xpath='./a', attribute='href', sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            team_home = tag_team_home.text

            if url_team_home is not None:
                id_team_home = extract_id_from_href(url_team_home)
                d_row.update({"id_team_home": id_team_home, "team_home": team_home}) # "url_team": url_team_home

        if tag_team_away is not None:
            url_team_away = super().extract_tag(tag_inicial=tag_team_away, xpath='./a', attribute='href', sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            team_away = tag_team_away.text

            if url_team_away is not None:
                id_team_away = extract_id_from_href(url_team_away)
                d_row.update({"id_team_away": id_team_away, "team_away": team_away}) #  "url_team": url_team_away

        if self._print:
            print(f"Extracting teams: {d_row}")
        return d_row

    def extract_result(self):
        d_row = {}
        d_field_xpath = {
            'goals_home': './/div[@class="detailScore__wrapper"]/span[1]', # './/div[@class="detailScore__wrapper"]/div[@class="detailScore__divider"]/preceding-sibling::span',
            'goals_away': './/div[@class="detailScore__wrapper"]/span[3]' #  './/div[@class="detailScore__wrapper"]/div[@class="detailScore__wrapper"]/following-sibling::span'
        }

        # Por campo a extraer
        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_row[field] = value    
    
        if self._print:
            print(f"Extracting goals: {d_row}")
        return d_row
    
    def extract_lineups(self):
        """
        Extrae de jugadores titulares, suplentes y ausentes de cada equipo.
        :return: Diccionario.
        """
        # Definicion de variables
        d_row = {}
        d_formations = {"Starting Lineups": "start", 'Substitutes': 'sub', 'Substituted players': 'sub_enter', 'Missing Players': 'miss'}

        # Por formation ("Formation inicial", "Suplentes" y  "Ausentes")
        for formation, titularidad in d_formations.items():

            SEC_WAIT = self.SEC_WAIT_MAX if formation=="Starting Lineups" else self.SEC_WAIT_MIN  # Jugadores ausentes muchas veces no esta. Y suplentes en partidos viejos tampocoEsto agiliza la extraccion.

            # Si existe dicha formation
            tag_lineup = super().extract_tag(xpath=f'.//div[@class="lf__lineUp"]/div[@class="section"]/div[text()="{formation}"]', sec_wait=SEC_WAIT, print_fail=True)
            if self._print:
                print(formation, SEC_WAIT)

            if tag_lineup:

                # Extraigo listado de jugadores
                l_tags_player_home = super().extract_tags(tag_inicial=tag_lineup, xpath='.//following-sibling::div//div[@class="lf__side"][1]//a[starts-with(@href, "/player/")]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None
                l_tags_player_away = super().extract_tags(tag_inicial=tag_lineup, xpath='.//following-sibling::div//div[@class="lf__side"][2]//a[starts-with(@href, "/player/")]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None

                if l_tags_player_home:
                    # Obtengo urls de jugadores
                    l_urls_home = [tag.get_attribute('href') for tag in l_tags_player_home]
                    
                    # Obtengo id y name de dichas urls
                    l_ids_home = [extract_id_from_href(url) for url in l_urls_home]
                    l_names_home = [extract_name_from_href(url) for url in l_urls_home]

                    # Guardo datos
                    for i, url in enumerate(l_urls_home):
                        d_row.update({f'id_player_{titularidad}_home_{i + 1}': l_ids_home[i], f'player_name_{titularidad}_home_{i + 1}': l_names_home[i]}) #  "player_url": url
                
                if l_tags_player_away:
                    l_urls_away = [tag.get_attribute('href') for tag in l_tags_player_away]

                    l_ids_away = [extract_id_from_href(url) for url in l_urls_away]
                    l_names_away = [extract_name_from_href(url) for url in l_urls_away]

                    for i, url in enumerate(l_urls_away):
                        d_row.update({f'id_player_{titularidad}_away_{i + 1}': l_ids_away[i], f'player_name_{titularidad}_away_{i + 1}': l_names_away[i]}) #  "player_url": url

        if self._print:
            print(f"Extracting lineups: {d_row}")
        return d_row
        
    def extract_coaches(self):
        """
        Extrae entrenadores tanto del equipo local como del equipo visitante.
        :return: Diccionario. Tanto para el equipo local como para el visitante, entrenador y su valor como key y value.
        """        
        d_row = {}

        seccion_entrenadores = super().extract_tag(xpath='.//div[@class="lf__lineUp"]/div[@class="section"]/div[text()="Coaches"]', sec_wait=self.SEC_WAIT_MIN)

        # Si existe la seccion de entrenadores
        if seccion_entrenadores:

            # Extraigo entrenadores
            coach_home_tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='./following-sibling::div//div[@class="lf__side"][1]//a[starts-with(@href, "/player/")]', sec_wait=self.SEC_WAIT_MIN)
            coach_away__tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='./following-sibling::div//div[@class="lf__side"][2]//a[starts-with(@href, "/player/")]', sec_wait=self.SEC_WAIT_MIN)

            if coach_home_tag:
                # Obtengo url de coach home
                url_coach_home = coach_home_tag.get_attribute('href')
                id_coach_home = extract_id_from_href(url_coach_home)
                coach_home = extract_name_from_href(url_coach_home)
                d_row.update({"id_coach_home": id_coach_home, "coach_home": coach_home})#  "url_coach": url_coach_home

            if coach_away__tag:
                # Obtengo url de coach away
                url_coach_away = coach_away__tag.get_attribute('href')
                id_coach_away = extract_id_from_href(url_coach_away)
                coach_away = extract_name_from_href(url_coach_away)
                d_row.update({"id_coach_away": id_coach_away, "coach_away": coach_away}) #  "url_coach": url_coach_away

        if self._print:
            print(f"Extracting coaches: {d_row}")
        return d_row

    def extract_stats(self):
        """
        Extrae todas las stats del match que haya
        :return: Diccionario. Tanto para el equipo local como para el visitante, stat y su valor como key y value.
        """
        d_row = {}

        # Extraigo tags de stats
        l_tags_stats = super().extract_tags(xpath=f'.//div[@data-testid="wcl-statistics"]', sec_wait=self.SEC_WAIT_MAX, print_fail=True)
        if self._print:
            print("Cantidad de stats a recolectar", len(l_tags_stats))

        # Por stat
        for tag in l_tags_stats:

            # Extraigo el name de la stat
            name_stat = super().extract_tag(tag_inicial=tag, xpath='.//div[@data-testid="wcl-statistics-category"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            name_stat_form = name_stat.lower().replace(" ", "_").replace('á', 'a').replace('é', 'e').replace('í', 'i').replace("ó", "o").replace('ú', 'u')
            if self._print:
                print(f"Stat a recolectar: {name_stat} --> Name formateado: {name_stat_form}")

            # Extraigo valores de la stat para el local y para el visitante
            d_row[f'{name_stat_form}_home'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "homeValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_row[f'{name_stat_form}_away'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "awayValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)

        if self._print:
            print(f"Extracting stats: {d_row}")
        return d_row

    def extract_odds(self):
        """
        Extrae cuotas de casa de apuestas Bet365. 
        Tambien se podria obtener de la hoja "Odds".
        """
        # Definicion de variables
        d_row = {}

        # Extraer las oddss en una lista
        l_odds_elements = super().extract_tags(xpath='.//div[@class="oddsRowContent"]//div[@class="cellWrapper"]//span[@class="oddsValueInner"]', sec_wait=self.SEC_WAIT_MAX, print_fail=True)

        if len(l_odds_elements) > 0: 
            l_text_odds_elements = [elem.text for elem in l_odds_elements]

            try: # Fallo para argentina en LPG 2010/11    d_row['odds_draw'] = l_text_odds_elements[1] IndexError: list index out of range
                d_row['odds_home'] = l_text_odds_elements[0]
                d_row['odds_draw'] = l_text_odds_elements[1]
                d_row['odds_away'] = l_text_odds_elements[2]
            except IndexError:
                pass

        if self._print:
            print(f"Extracting odds: {d_row}")
        return d_row

    def extract_bajas_pre_partido(self):
        """
        Extrae listado de jugadores ausentes para el proximo partido.
        """
        d_row = {}

        # Extraer bajas
        tag_bajas = super().extract_tag(xpath=f'.//div[@id="detail"]//div[text()="Will not play"]', sec_wait=self.SEC_WAIT_MIN, print_fail=True)

        if tag_bajas:

            # Extraigo listado de jugadores
            l_tags_player_home = super().extract_tags(tag_inicial=tag_bajas, xpath='.//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None
            l_tags_player_away = super().extract_tags(tag_inicial=tag_bajas, xpath='.//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None

            if l_tags_player_home:
                # Obtengo urls de jugadores
                l_urls_home = [tag.get_attribute('href') for tag in l_tags_player_home]
                
                # Obtengo id y name de dichas urls
                l_ids_home = [extract_id_from_href(url) for url in l_urls_home]
                l_names_home = [extract_name_from_href(url) for url in l_urls_home]

                # Guardo datos
                for i, url in enumerate(l_urls_home):
                    d_row.update({f'id_player_miss_home_{i + 1}': l_ids_home[i], f'player_name_miss_home_{i + 1}': l_names_home[i]}) #  "player_url": url
                
            if l_tags_player_away:
                l_urls_away = [tag.get_attribute('href') for tag in l_tags_player_away]

                l_ids_away = [extract_id_from_href(url) for url in l_urls_away]
                l_names_away = [extract_name_from_href(url) for url in l_urls_away]

                for i, url in enumerate(l_urls_away):
                    d_row.update({f'id_player_miss_away_{i + 1}': l_ids_away[i], f'player_name_miss_away_{i + 1}': l_names_away[i]}) #  "player_url": url

        if self._print:
            print(f"Extracting lineups: {d_row}")
        return d_row

    def select_items_by_date(self, l_items, n_days):
        """
        Selecciona los partidos que se juegan entre hoy y los proximos n dias.
        """
        # Definicion de variables
        l_items_filt = []

        # Determino fecha umbral y año actual
        fecha_umbral = datetime.now() + timedelta(days=n_days)
        anio_actual = datetime.now().year

        # Por partido
        for item in l_items:

            # Obtengo su fecha 
            fecha_str = super().extract_tag(tag_inicial=item, xpath='./div[@class="event__time"]', text=True, sec_wait=1)

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

def clean_id(l_ids: list):
    """
    Limpia todos los ids contenidos en una lista al remover el string adicional que cada uno tiene.
    :param l_ids: Lista cuyos elementos contienen ids de partidos. (list)
    :return: Lista cuyos elementos son ids de partidos. (list)
    """
    l_ids_clean = []
    for id_match in l_ids:
        id_match = id_match[id_match.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")
        l_ids_clean.append(id_match)
    return l_ids_clean
    
def extract_id_from_href(href: str) -> str:
    """
    Extrae el identificador del atributo href.

    Parameters:
        href (str): La cadena href del cual se extraerá el identificador.

    Returns:
        str: El identificador extraído o None si no se encuentra.
    """
    pattern = r"/(team|player)/[^/]+/([^/]+)/?" # (e.g. "/player/raya-david/nkVV0IXb", "/player/raya-david/nkVV0IXb/", "/player/lionel-messi/erigoeriowgjwi", "/team/arsenal/asfjsiafjis/"
    match = re.search(pattern, href)
    if match:
        return match.group(2)
    return None

def extract_name_from_href(href: str) -> str:
    """
    Extrae el nombre del equipo del atributo href.

    Parameters:
        href (str): La cadena href del cual se extraerá el nombre del equipo. (e.g. 'https://www.flashscore.com/player/roerslev-rasmussen-mads/pp1zpsrr/'). (String)

    Returns:
        str: El nombre del equipo extraído o None si no se encuentra. (e.g. roerslev-rasmussen-mads). (String)
    """
    pattern = r"/(team|player)/([^/]+)/[^/]+/?"
    match = re.search(pattern, href)
    if match:
        name = match.group(2).replace('-', ' ')
        return name
    return None

def extract_data(id_country, country: str, id_competicion, competition: str, is_cup: int, n_seasons_max: int = 0, l_ids_already_collected: list = None, export: bool = True):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.

    # Parameters:
        id_country: Id del pais a extraer.
        country: Nombre del pais a extraer. (str)
        id_competicion: Id de la competicion del pais a extraer
        competition: Nombre de la competicion del pais a extraer. (str)
        n_season_max: Cantidad de temporadas desde la actual para extraer. 0 para extraer todas las temporadas disponibles. (int)
    
    # Returns:
        df_match: Dataframe.
        df_match_player: Dataframe.
        df_match_odds: Dataframe.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    crawler = FlashscoreCrawler(headless=True)
    df_match, df_match_player, df_match_odds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Formateo variables para guardado de datos
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(".", "").replace(" ", "-")  # formateo competition para las rutas de archivo y urls
    ruta_base = f"./p2_data_understanding/data/{country_form}/data_seg"

    # Ingreso a pagina
    url = f'https://www.flashscore.com/football/{country_form}/{competition_form}/archive/'
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    print(f"URL competition: {url}")

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()

    # Extraigo urls de las distintas seasons (años) de la competition
    l_urls_seasons = crawler.extract_urls_seasons(n_seasons_max)
    print(f'Cantidad de seasons: {len(l_urls_seasons)}')

    # POR season
    for n_season, url_season in enumerate(l_urls_seasons):  # for url_season in l_urls_seasons:  # De mas reciente a menos reciente

        if n_season > 9:
            crawler.SEC_WAIT_MED = crawler.SEC_WAIT_MIN
        
        # Ingreso a pagina de season e imprimo año de la season
        crawler.driver.get(url_season)
        season_year = crawler.extract_season_year()
        print(f" {season_year} ".center(120, "-"))

        # Click en hoja "Results"
        crawler.click_results_page()

        # Si ya extraje la season
        check = not check_if_season_already_extracted(ruta_base, competition_form, season_year) if l_ids_already_collected is None else True 
        if check:

            df_match_season, df_match_player_season, df_match_odds_season = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la season)
            crawler.click_show_more_matches()

            # Extraigo partidos (items) y sus ids
            l_ids = crawler.extract_id_matches()
            if l_ids_already_collected is not None:
                l_ids_filt = [id_match for id_match in l_ids if id_match not in l_ids_already_collected]
                print(f"De los {len(l_ids)} partidos de la temporada, se recolectan solo los {len(l_ids_filt)} que faltan ")
                l_ids = l_ids_filt

            print(f"Partidos recolectados de la season {season_year} (e.g. en premier league deberian ser 380): {len(l_ids)}")
            progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

            # POR MATCH (c/u identificado con un id)
            for id_match in l_ids:

                # Ingreso a pagina de informacion del match
                url_match = f'https://www.flashscore.com/match/{id_match}/#/match-summary'
                crawler.driver.get(url_match)

                # Extraigo todos los datos del partido
                d_row_match, d_row_match_player, d_row_match_odds = crawler.extract_match_data()
                d_row_match.update({'id_country': id_country, 'id_competition': id_competicion, 'is_cup': is_cup, 'season': season_year})

                # Guardo datos del partido
                df_match_season = pd.concat([df_match_season, pd.DataFrame(d_row_match, index=[id_match])])
                df_match_player_season = pd.concat([df_match_player_season, pd.DataFrame(d_row_match_player, index=[id_match])])
                df_match_odds_season = pd.concat([df_match_odds_season, pd.DataFrame(d_row_match_odds, index=[id_match])])
                progress_bar.update(1)

            progress_bar.close()
            if export:
                # Guardo partidos de la season (por seguridad)
                df_match_season.to_excel(f'{ruta_base}/per_season/df_match/{competition_form}_{season_year}.xlsx', index=True)
                df_match_player_season.to_excel(f'{ruta_base}/per_season/df_match_player/{competition_form}_{season_year}.xlsx', index=True)
                df_match_odds_season.to_excel(f'{ruta_base}/per_season/df_match_odds/{competition_form}_{season_year}.xlsx', index=True)
            
        else:
            print(f"La season {season_year} de la competicion {competition_form} ya fue extraida, por lo que, se evita su nueva extraccion.")
            df_match_season = pd.read_excel(f'{ruta_base}/per_season/df_match/{competition_form}_{season_year}.xlsx', index_col=0)
            df_match_player_season = pd.read_excel(f'{ruta_base}/per_season/df_match_player/{competition_form}_{season_year}.xlsx', index_col=0)
            df_match_odds_season = pd.read_excel(f'{ruta_base}/per_season/df_match_odds/{competition_form}_{season_year}.xlsx', index_col=0)
            print(f"Match: {df_match_season.shape}, Match_player: {df_match_player_season.shape}, Odds: {df_match_odds_season.shape}")

        # Concateno seasons
        df_match = pd.concat([df_match, df_match_season], axis=0)
        df_match_player = pd.concat([df_match_player, df_match_player_season], axis=0)
        df_match_odds = pd.concat([df_match_odds, df_match_odds_season], axis=0)

    if export:
        # Guardo partidos de la competition
        df_match.to_excel(f'{ruta_base}/per_competition/df_match/{competition_form}.xlsx', index=True)
        df_match_player.to_excel(f'{ruta_base}/per_competition/df_match_player/{competition_form}.xlsx', index=True)
        df_match_odds.to_excel(f'{ruta_base}/per_competition/df_match_odds/{competition_form}.xlsx', index=True)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_match, df_match_player, df_match_odds

def check_if_season_already_extracted(ruta_base, competition_form, season_year):
    """
    Verificacion de que aun no extraje la temporada
    """
    try:
        pd.read_excel(f'{ruta_base}/per_season/df_match/{competition_form}_{season_year}.xlsx', index_col=0)
        pd.read_excel(f'{ruta_base}/per_season/df_match_player/{competition_form}_{season_year}.xlsx', index_col=0)
        pd.read_excel(f'{ruta_base}/per_season/df_match_odds/{competition_form}_{season_year}.xlsx', index_col=0)
        return True
    except:
        return False

def extract_next_matches(id_country, country: str, id_competicion, competition: str, is_cup, n_days: int): # -> tuple[pd.DataFrame, pd.DataFrame]
    """
    Extraccion de los partidos de los proximos <n_days> dias en la competicion <competition> del pais <country>.
    """
    # Definicion de variables
    df_match, df_match_player, df_match_odds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    crawler = FlashscoreCrawler(headless=True)

    # Formateo variables para construir url
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(" ", "-")  # formateo competition para las rutas de archivo y urls
    url = f'https://www.flashscore.com/football/{country_form}/{competition_form}/fixtures/'
    print(f'URL competición: {url}')

    # Ingreso a pagina
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()  
    season_year = crawler.extract_season_year() 
    print(f" {season_year} ".center(120, "-"))

    # Extraigo partidos
    l_ids = crawler.extract_id_next_matches(n_days)
    progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

    # POR PARTIDO (c/u identificado con un id)
    for id_match in l_ids:

        # Construyo url de partido
        url_match = f'https://www.flashscore.com/match/{id_match}/#/summary-match'
        crawler.driver.get(url_match)

        # Extraigo datos del partido
        d_row_match, d_row_match_player, d_row_match_odds = crawler.extract_match_data(next_matches=True)
        d_row_match.update({'id_country': id_country, 'id_competition': id_competicion, 'is_cup': is_cup, 'season': season_year}) # 'competition': competition, 'country': country

        # GUARDADO DE DATOS EN DATAFRAME
        df_match = pd.concat([df_match, pd.DataFrame(d_row_match, index=[id_match])])
        df_match_player = pd.concat([df_match_player, pd.DataFrame(d_row_match_player, index=[id_match])])
        df_match_odds = pd.concat([df_match_odds, pd.DataFrame(d_row_match_odds, index=[id_match])])
        progress_bar.update(1)

    # Cerrar la barra de progreso al finalizar
    progress_bar.close()

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_match, df_match_player, df_match_odds

def extract_matches_result(country: str, competition: str, l_ids:list):

    # DEFINCION DE PARAMETROS & VARIABLES
    crawler = FlashscoreCrawler(headless=True)
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
        d_row.update({'country': country, 'competition': competition})

        # Guardo datos del partido
        df = pd.concat([df, pd.DataFrame(d_row, index=[id_match])])
        progress_bar.update(1)

    # Finalizada la extraccion, cierro el web browser automático
    progress_bar.close()
    crawler.driver.close()
    return df


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    # Selecciono country a extraer y obtengo las competencias y su categoria
    country = 'england'  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    id_competicion = 481
    competition = 'Premier League'
    is_cup = 0
    n_seasons_max = 1

    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    id_country = df_countries[df_countries['country_name'] == country.capitalize()]['id_country'].values[0]

    # Extraigo partidos
    # df_match, df_match_player, df_match_odds = extract_data(id_country, country, id_competicion, competition, is_cup, n_seasons_max, export=False)
    # df_match.to_excel('/Users/nachomondino/Desktop/df_match.xlsx', index=True)
    # df_match_player.to_excel('/Users/nachomondino/Desktop/df_match_player.xlsx', index=True)
    # df_match_odds.to_excel('/Users/nachomondino/Desktop/df_match_odds.xlsx', index=True)

    # Extraer partidos 
    df_match = pd.read_excel(f"p2_data_understanding/data/{country}/data_seg/per_season/df_match/premier-league_2023_2024.xlsx", index_col=0)
    df_match_miss, df_match_player_miss, df_match_odds_miss = extract_data(id_country, country, id_competicion, competition, is_cup, n_seasons_max=1, l_ids_already_collected=list(df_match.index), export=False)
    df_match_miss.to_excel('/Users/nachomondino/Desktop/df_match_miss.xlsx', index=True)
    df_match_player_miss.to_excel('/Users/nachomondino/Desktop/df_match_player_miss.xlsx', index=True)
    df_match_odds_miss.to_excel('/Users/nachomondino/Desktop/df_match_odds_miss.xlsx', index=True)
