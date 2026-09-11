# Importo librerias
import pandas as pd
from predictor.data_understanding.web_scraping_selenium import Crawler
from tqdm import tqdm
from datetime import datetime, timedelta
import re
from predictor.utils.set_up_logging import logger
from predictor.utils.directories import make_directories
from time import sleep
import random

class FlashscoreCrawler(Crawler):
    """
    Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper
    normal, scraper de fields especificos poro falla y scraper de proximos partidos...
    Contiene todo los xpath.
    """
    def __init__(self, headless: bool = True, browser: str = "Chrome", verbose: int = 0):
        super().__init__(headless, browser)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 1  # Espera para elementos que muchas veces no estan # con 0.2 fallaba extraccion de campos que si estaban como goals
        self.SEC_WAIT_MED = 3  # Espera para elementos que casi siempre estan
        self.SEC_WAIT_MAX = 5 # Espera para elementos que casi siempre estan
        self.verbose =  verbose # Para imprimir el funcionamiento de cada funcion y poder hacer pruebas...

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
            print(f"The season ({season_year})) is not a string. Probably it failed the extraction.")
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
    
    def extract_id_next_matches(self, n_days, verbose: int = 1):
            
        # Extraigo partidos (items) y sus ids --> NO PUDE EXTRAER LOS SVG.. PERO SI EL DIV DE EVENT_TIME... VER 
        l_items = super().extract_tags(xpath='.//div[@id="live-table"]//div[@class="sportName soccer"]//div[contains(@class, "event__match--scheduled")]', sec_wait=self.SEC_WAIT_MAX)
        if self.verbose >= verbose:
            print(f"Cantidad de proximos partidos en total: {len(l_items)}")

        # Filtro partidos por fecha
        l_items_filt = self.select_items_by_date(l_items, n_days)
        if self.verbose >= verbose:
            print(f"Cantidad de proximos partidos dentro de {n_days}: {len(l_items_filt)}")

        # Obtengo ids
        l_ids = [item.get_attribute('id') for item in l_items_filt]
        l_ids_clean = clean_id(l_ids)
        if self.verbose >= verbose:
            print(f"Cantidad de ids extraidos dentro de {n_days}: {len(l_ids_clean)}")

        return l_ids_clean

    def extract_match_data(self, next_matches:bool = False):
        """
        Extrae datos del partido siendo este un partido ya jugado.

        Mejoras: Ponerle nombre "main" o algo asi.
        """
        # Reinicio diccionario en el que guardar datos del nuevo match
        d_row_match = {}
        d_row_match_player = {}
        d_row_match_odds = {}
        print_not_next_matches = not next_matches

        # MATCH INFORMATION (de hoja "summary")
        d_row_match.update(self.extract_match_information(next_matches))
        ### Goals
        if not next_matches:
            d_row_match.update(self.extract_result())
        ### Teams
        d_row_match.update(self.extract_teams())
        d_row_match.update(self.extract_disclaimer_data())

        # Si es un prox partido, determino cuanto falta para el partido
        if next_matches:
            date_dt = pd.to_datetime(d_row_match['date'], format='%d.%m.%Y %H:%M') 
            date_limit = date_dt - datetime.now()
            days_diff = date_limit.days
            hours_diff = date_limit.total_seconds() / 3600  # Convierte segundos a horas
        else: 
            hours_diff = None

        # STATS
        ## Si tiene hoja "stats", extraigo campos
        if not next_matches:
            boton_stats = super().extract_tag(xpath='//a[@data-analytics-alias="match-statistics"]/button', sec_wait=self.SEC_WAIT_MAX, print_fail=True)
            if super().click_boton(boton_stats) is not False:
                d_row_match.update(self.extract_stats())

        # ODDS
        ## Si existe la seccion "odds pre-match", extraigo odds de Bet365
        if super().extract_tag(xpath='.//div[@class="odds"]', sec_wait=self.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las oddss pero no hay valores...
            d_row_match_odds.update(self.extract_odds())

        # LINE UPS
        if next_matches:
            xpath_lineups_button = '//a[@data-analytics-alias="predicted-lineups"]/button'
        else:
            xpath_lineups_button ='//a[@data-analytics-alias="lineups"]/button' # //a[@href="#/match-summary/lineups"]/button

        boton_formations = super().extract_tag(xpath=xpath_lineups_button, sec_wait=self.SEC_WAIT_MAX, print_fail=print_not_next_matches)

        ## Si tiene hoja "Lineups"
        if super().click_boton(boton_formations) is not False:

            # Determino formaciones a extraer
            d_formations = self.determine_formations_to_extract(next_matches=next_matches, hours_diff=hours_diff)

            ### Alineaciones titulares, suplentes y ausentes
            d_row_match_player.update(self.extract_lineups(d_formations))
            
            ### Coaches
            d_row_match.update(self.extract_coaches())

        ## Si no tiene hoja "Lineups" y es un proximo partido 
        elif next_matches:
            d_row_match_player.update(self.extract_bajas_pre_partido(days_diff=days_diff))
            if self.verbose >= 1:
                logger.info("Aun no existe la hoja 'Lineups', por lo que, no puedo obtener formaciones ni coaches.")
                
        else:
            logger.warning("El partido ya se jugó pero no existe la hoja 'Lineups' de la cual extraer las formaciones")
        
        return d_row_match, d_row_match_player, d_row_match_odds

    def extract_new_data(self):
        """
        Inicializo funcion para obtener nuevos campos respecto de los datos ya extraidos 
        """
        d_row = {}
        d_field_xpath = {
            'result_1st_half': '//span[text()="1st Half"]/parent::div/span[2]', # Con este genero 1st_ht_goals_home y 1st_half_goals_away y result_1st_half (1, 0 o 2) 
            'result_2nd_half': '//span[text()="2nd Half"]/parent::div/span[2]', # Con esto genero 2nd_ht_goals_home y 2nd_half_goals_away y result_2nd_half (1, 0 o 2)
            }
        
        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, sec_wait=self.SEC_WAIT_MIN, text=True, print_fail=False)

            print(field, value)
            d_row[field] = value

        return d_row

    def extract_match_information(self, next_matches):
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

        if next_matches:
            d_field_xpath.pop("attendance", None)  # Elimina "attendance" si existe, evitando errores

        # Extraigo el primer campo con espera para evitar extraer sin que haya cargado la pagina
        d_row['date'] = super().extract_tag(xpath='.//div[@class="duelParticipant"]/div[@class="duelParticipant__startTime"]', text=True, sec_wait=self.SEC_WAIT_MED)
        
        # Por campo a extraer
        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_row[field] = value

        if self.verbose >= 1:
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

        if self.verbose >= 1:
            print(f"Extracting teams: {d_row}")
        return d_row

    def extract_result(self):
        d_row = {}
        d_field_xpath = {
            'goals_home': './/div[@class="detailScore__wrapper"]/span[1]',
            'goals_away': './/div[@class="detailScore__wrapper"]/span[3]'
        }

        # Por campo a extraer
        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_row[field] = value    
    
        if self.verbose >= 1:
            print(f"Extracting goals: {d_row}")
        return d_row
    
    def determine_formations_to_extract(self, next_matches, hours_diff):

        if next_matches:
            # Definir conjuntos de formaciones
            formations_less_than_one_hour = {
                "Starting Lineups": "start",
                "Predicted starting lineups": "start",
                "Substitutes": "sub",
                "Will not play": "miss",
                "Missing Players": "miss"
            }

            formations_more_than_one_hour = {
                "Predicted starting lineups": "start",
                "Will not play": "miss"
            }

            # Asignar d_formations según el tiempo restante
            d_formations = formations_less_than_one_hour if hours_diff <= 1 else formations_more_than_one_hour

            if self.verbose >= 1:
                message = (
                    "Existe el botón Lineups y falta menos de una hora, por lo que se intentará obtener las formaciones iniciales"
                    if hours_diff <= 1
                    else "Existe el botón Lineups pero, dado que falta más de una hora, solo se buscarán 'Predicted starting lineups' y 'Will not play'"
                )
                logger.warning(message) if hours_diff <= 1 else logger.info(message)

        else:
            d_formations = {
                "Starting Lineups": "start",
                "Substitutes": "sub", # "Substituted players": "sub_enter", # ya estan en substitutes
                "Missing Players": "miss"
            }
        return d_formations

    def extract_lineups(self, d_formations):
        """
        Extrae de jugadores titulares, suplentes y ausentes de cada equipo.
        :return: Diccionario.
        """
        # Definicion de variables
        d_row = {}

        # Por formation ("Formation inicial", "Suplentes" y  "Ausentes")
        for formation, titularidad in d_formations.items():

            # Si existe dicha formation
            tag_lineup = super().extract_tag(
                xpath=f'//span[text()="{formation}"]/ancestor::div[@class="section"]',  # f'//div[contains(@class, "lf__lineUp")]//span[contains(text(), "{formation}")]/ancestor::div[contains(@class, "wcl-headerSection")]/following-sibling::div', 
                sec_wait=self.SEC_WAIT_MAX, 
                print_fail=True
            )

            if self.verbose >= 1:
                print(formation, self.SEC_WAIT_MAX)

            if tag_lineup:

                # Por equipo (home o away)
                for team in ['1', '2']:

                    team_side = "home" if team == '1' else "away"

                    # --- INICIO DEL INTENTO PRIMARIO ---
                    # Intento 1: Extraer jugadores usando el XPath de a[@href]
                    l_tags_players = super().extract_tags(
                                            tag_inicial=tag_lineup, 
                                            xpath=f'.//div[@class="lf__side"][{team}]//a[starts-with(@href, "/player/")]', # //span[text()="Missing Players"]//ancestor::div[@class="section"]//div[@class="lf__side"][1]//a[starts-with(@href, "/player/")]                
                                            sec_wait=self.SEC_WAIT_MIN, 
                                            print_fail=False
                                        )
                
                    l_urls = [tag.get_attribute('href') for tag in l_tags_players]

                    if not l_tags_players:

                        if self.verbose > 0:
                            logger.warning("SEGUNDO INTENTO")

                        # --- INICIO DEL FALLBACK A LA SEGUNDA FORMA ---
                        # Intento 2 (Fallback): Si no se encontraron jugadores, intenta el XPath para //button
                        l_tags_players = super().extract_tags(
                            tag_inicial=tag_lineup, 
                            xpath=f'.//div[@class="lf__side"][{team}]//button', # //span[text()="Starting Lineups"]//ancestor::div[@class="section"]//div[@class="lf__side"][1]//button
                            sec_wait=self.SEC_WAIT_MIN, 
                            print_fail=False
                        )
                        
                        # Obtengo Urls
                        l_urls = self.obtain_urls_players(l_tags_players)
                    
                    # Si hay listado de urls, extraigo id y nombre del jugador de cada una
                    if l_urls:
                        d_player = self.save_player_data(l_urls, titularidad, team_side)
                        d_row.update(d_player)

                        if self.verbose > 0:
                            print(d_player)
                    else:
                        # Manejar el caso donde no hay URLs para procesar
                        print(f"No se procesarán datos para el equipo {team_side} ya que no se encontraron URLs.")

            else:
                logger.warning(f"No se encontró la seccion {formation} en Lineups.")

        if self.verbose >= 1:
            print(f"Extracting lineups: {d_row}")

        return d_row
    
    def obtain_urls_players(self, l_tags):

        l_urls = []

        for tag in l_tags:

            # Clickeo en boton
            super().click_boton(tag, sec_wait=self.SEC_WAIT_MAX)

            # Obtengo url de ventana emergente (javascript)
            url = super().extract_tag(xpath='//div[@data-testid="wcl-dialogBody"]//a[starts-with(@href, "/player/")]', attribute='href', sec_wait=self.SEC_WAIT_MAX)
            l_urls.append(url)

            # Cerrar ventana actual
            boton_close = super().extract_tag(xpath="//div[@data-testid='wcl-dialogBody']//button[@data-testid='wcl-dialogCloseButton']")
            super().click_boton(boton_close)
            # sleep(0.5)
        
        return l_urls

    def save_player_data(self, l_urls, titularidad, team_side):
        """
        Extrae los datos de los jugadores (ID y nombre) a partir de una lista de etiquetas HTML.

        Args:
            l_tags_player (list): Lista de etiquetas HTML (objetos de Selenium) de los jugadores.
            team_side (str): Lado del equipo, 'home' o 'away'.
            titularidad (str): Tipo de jugador, 'titular' o 'suplente'.

        Returns:
            dict: Un diccionario con los datos de los jugadores.
        """
        d_player_data = {}

        # Obtener ID y nombre de las URLs
        l_ids = [extract_id_from_href(url) for url in l_urls]
        l_names = [extract_name_from_href(url) for url in l_urls]

        # Guardar los datos en el diccionario de retorno
        for i in range(len(l_urls)):
            d_player_data.update({
                f'id_player_{titularidad}_{team_side}_{i + 1}': l_ids[i],
                f'player_name_{titularidad}_{team_side}_{i + 1}': l_names[i]
            })
        return d_player_data

    def extract_disclaimer_data(self):
        """
        Extrar si se juega en estadio neutral y si el partido fue a penales
        """
        d_row = {}

        # Penalties
        d_row['penalties'] = super().extract_tag(xpath='.//div[@class="detailScore__status"]', text=True, sec_wait=0.4, print_fail=False)  # After Penalties o FINISHED  

        # Neutralidad
        tag_info_box = super().extract_tag(xpath='.//div[contains(@class, "infoBoxModule")]', text=True, sec_wait=0.4, print_fail=False) # 

        if tag_info_box is not None:
            # Defino los textos que indican neutralidad
            pos_texts = ['neutral', "playing at", "at a different stadium"]
            
            # Si alguno de los textos está en el tag, entonces es neutral
            texts_in_tag = [text for text in pos_texts if text in tag_info_box.lower()]
            neutralidad = 1 if texts_in_tag else 0
        else:
            neutralidad = 0
        
        d_row.update({"neutral": neutralidad}) #  "url_team": url_team_away

        if self.verbose >= 1:
            print(f"Extracting goals: {d_row}")
        return d_row                

    def extract_coaches(self):
        """
        Extrae entrenadores tanto del equipo local como del equipo visitante.
        :return: Diccionario. Tanto para el equipo local como para el visitante, entrenador y su valor como key y value.
        """        
        d_row = {}

        seccion_entrenadores = super().extract_tag(xpath='//span[text()="Coaches"]/ancestor::div[@class="section"]', sec_wait=self.SEC_WAIT_MIN)

        # Si existe la seccion de entrenadores
        if seccion_entrenadores:

            # Extraigo entrenadores
            coach_home_tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='.//div[@class="lf__side"][1]//a[starts-with(@href, "/player/")]', sec_wait=self.SEC_WAIT_MIN)
            coach_away__tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='.//div[@class="lf__side"][2]//a[starts-with(@href, "/player/")]', sec_wait=self.SEC_WAIT_MIN)

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

        if self.verbose >= 1:
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
        if self.verbose >= 1:
            print("Cantidad de stats a recolectar", len(l_tags_stats))

        # Por stat
        for tag in l_tags_stats:

            # Extraigo el name de la stat
            name_stat = super().extract_tag(tag_inicial=tag, xpath='.//div[@data-testid="wcl-statistics-category"]', text=True, sec_wait=self.SEC_WAIT_MED, print_fail=False)
            try:
                name_stat_form = name_stat.lower().replace(" ", "_").replace('á', 'a').replace('é', 'e').replace('í', 'i').replace("ó", "o").replace('ú', 'u')
            except AttributeError as e: #  'NoneType' object has no attribute 'lower'
                name_stat_form = name_stat
                logger.warning(f"Error {e} al intentar formatear el nombre de una stat que es None.")
            
            if self.verbose >= 1:
                print(f"Stat a recolectar: {name_stat} --> Name formateado: {name_stat_form}")

            # Extraigo valores de la stat para el local y para el visitante
            d_row[f'{name_stat_form}_home'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "homeValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_row[f'{name_stat_form}_away'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "awayValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)

        if self.verbose >= 1:
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
        l_odds_elements = super().extract_tags(xpath='.//div[@class="odds"]//span[@data-testid="wcl-oddsValue"]', sec_wait=self.SEC_WAIT_MAX, print_fail=True)

        if len(l_odds_elements) > 0: 
            l_text_odds_elements = [elem.text for elem in l_odds_elements]

            try: # Fallo para argentina en LPG 2010/11    d_row['odds_draw'] = l_text_odds_elements[1] IndexError: list index out of range
                d_row['odds_home'] = l_text_odds_elements[0]
                d_row['odds_draw'] = l_text_odds_elements[1]
                d_row['odds_away'] = l_text_odds_elements[2]
            except IndexError:
                pass

        if self.verbose >= 1:
            print(f"Extracting odds: {d_row}")
        return d_row

    def extract_bajas_pre_partido(self, days_diff, days_thr: int = 2):
        """
        Extrae listado de jugadores ausentes para el proximo partido.

        # FALTARIA TMB SECCION POSIBLES BAJAS, "Questionable"
        """
        d_row = {}

        # Extraer bajas
        tag_bajas = super().extract_tag(
            xpath=f'//div[contains(@class, "section")]//*[contains(text(), "Will not play")]/ancestor::div[contains(@class, "wcl-headerSection")]/following-sibling::div',
            sec_wait=self.SEC_WAIT_MIN, 
            print_fail=False
            )

        if tag_bajas:

            # Extraigo listado de jugadores
            l_tags_player_home = super().extract_tags(
                tag_inicial=tag_bajas,
                xpath='.//div[@class="lf__side"][1]//a[starts-with(@href, "/player/")]', 
                sec_wait=self.SEC_WAIT_MIN, 
                print_fail=False
                )  # Es lista de tags o None
            
            l_tags_player_away = super().extract_tags(
                tag_inicial=tag_bajas, 
                xpath='.//div[@class="lf__side"][2]//a[starts-with(@href, "/player/")]', 
                sec_wait=self.SEC_WAIT_MIN, 
                print_fail=False
                )  # Es lista de tags o None

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

        else:
            # Tiene que saltar warning solo si no existe el tag_bajas y si faltan menos de 2 dias.
            if days_diff < days_thr:
                logger.warning(f"No existe la seccion 'Will not play' y faltan menos de {days_thr} dias para el partido. Es correcto solo si efectivamente no hay bajas pero sino puede ser por cambio de codigo HTML de la seccion.")
            elif self.verbose >= 1:
                logger.info(f"No existe la seccion 'Will not play' pero faltan mas de {days_thr} dias para el partido.")

        if self.verbose >= 1:
            print(f"Extracting lineups: {d_row}")

        return d_row

    def select_items_by_date(self, l_items, n_days):
        """
        Selecciona los partidos que se juegan entre hoy y los proximos n dias.

        Mejoras:
            - Tiene problemas al extraer partidos cuando cambia de año.
        """
        # Definicion de variables
        l_items_filt = []

        # Determino fecha umbral y año actual
        fecha_umbral = datetime.now() + timedelta(days=n_days)
        anio_actual = datetime.now().year
        anio_siguiente = anio_actual + 1
        if self.verbose >= 1:
            logger.info(f"Fecha umbral: {fecha_umbral}. Año actual: {anio_actual}.")

        # Por partido
        for item in l_items:

            # Obtengo su fecha 
            fecha_str = super().extract_tag(tag_inicial=item, xpath='./div[@class="event__time"]', text=True, sec_wait=1) # (e.g 03.01. 17:00)
            if self.verbose >= 1:
                logger.info(f"Fecha extraida del partido: {fecha_str}")

            # Si el partido tiene div con fecha
            if fecha_str is not None:
                
                # Formateo fecha de str a datetime (agregandole el año pues sino toma 1900)
                mes_str = fecha_str[3:5]
                # anio = anio_siguiente if mes_str == '01' else anio_actual
                anio = anio_actual
                fecha_str = f"{anio}.{fecha_str}"
                
                if self.verbose >= 1:
                    logger.info(f'Mes extraido del partido: {mes_str}')
                    logger.info(f"Fecha del partido con año: {fecha_str}")

                try:
                    # Intento convertir la fecha usando el formato especificado
                    fecha_objeto = datetime.strptime(fecha_str, "%Y.%d.%m. %H:%M")
                    
                    # Si el partido es dentro de los proximos días
                    if fecha_objeto <= fecha_umbral:
                        l_items_filt.append(item)
                        
                        if self.verbose >= 1:
                            logger.critical(f"Partido dentro de umbral de fechas!")

                    else:
                        # Dejo de extraer partidos puesto que luego del primer partido que es posterior a fecha_umbral, todos lo son (estan ordenados por fecha en Flashscore)
                        if self.verbose >=1:
                            logger.warning(f"Partido fuera del umbral de fechas!")
                        break
                except ValueError as e:
                    # Si hay un error en la conversión, imprimo el mensaje de error
                    print(f"Error al convertir la fecha: {e}")

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

def extract_data(id_country, country: str, id_competicion: int, competition: str, is_cup: int, n_seasons_max: int = 0, l_ids_already_collected: list = None, export: bool = True):
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
    crawler = FlashscoreCrawler(headless=False)
    df_match, df_match_player, df_match_odds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Formateo variables para guardado de datos
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(".", "").replace(" ", "-")  # formateo competition para las rutas de archivo y urls
    ruta_base = f"./data/{country_form}/data_understanding/data_seg"
    make_directories(l_directorios=[
        f'{ruta_base}/per_competition/df_match', f'{ruta_base}/per_competition/df_match_player', f'{ruta_base}/per_competition/df_match_odds',
        f'{ruta_base}/per_season/df_match', f'{ruta_base}/per_season/df_match_player', f'{ruta_base}/per_season/df_match_odds'
        ])

    # Ingreso a pagina
    url = f'https://www.flashscore.com/football/{country_form}/{competition_form}/archive/'
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    print(f"URL competition: {url}")

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()

    # Extraigo urls de las distintas seasons (años) de la competition
    l_urls_seasons = crawler.extract_urls_seasons(n_seasons_max)
    print(f'Cantidad de seasons: {len(l_urls_seasons)}: {l_urls_seasons}')

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
                logger.warning(f"De los {len(l_ids)} partidos de la temporada, se recolectan solo los {len(l_ids_filt)} que faltan")
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

def extract_missing_matches(id_country, country: str, id_competicion, competition: str, is_cup: int, n_seasons_max: int = 0, l_ids_already_collected: list = None, export: bool = True):
    
    # DEFINCION DE PARAMETROS & VARIABLES
    crawler = FlashscoreCrawler(headless=False)
    df_match, df_match_player, df_match_odds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Formateo variables para guardado de datos
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(".", "").replace(" ", "-")  # formateo competition para las rutas de archivo y urls
    ruta_base = f"./data/data_understanding/{country_form}/data_seg"  # ⚠️ orden pais/fase invertido vs. el resto del repo (ya así antes del refactor); sin efecto real porque extract_missing_matches() no tiene ningún caller.

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

        df_match_season, df_match_player_season, df_match_odds_season = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Ingreso a pagina de season e imprimo año de la season
        crawler.driver.get(url_season)
        season_year = crawler.extract_season_year()
        print(f" {season_year} ".center(120, "-"))

        # Click en hoja "Results" +  "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la season)
        crawler.click_results_page()
        crawler.click_show_more_matches()

        # Extraigo partidos (items) y sus ids
        l_ids = crawler.extract_id_matches()
        if l_ids_already_collected is not None:
            l_ids_filt = [id_match for id_match in l_ids if id_match not in l_ids_already_collected]
            logger.warning(f"De los {len(l_ids)} partidos de la temporada, se recolectan solo los {len(l_ids_filt)} que faltan")
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

        # Concateno seasons
        df_match = pd.concat([df_match, df_match_season], axis=0)
        df_match_player = pd.concat([df_match_player, df_match_player_season], axis=0)
        df_match_odds = pd.concat([df_match_odds, df_match_odds_season], axis=0)

     # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_match, df_match_player, df_match_odds

def extract_next_matches(id_country, country: str, id_competicion, competition: str, is_cup, n_days: int): # -> tuple[pd.DataFrame, pd.DataFrame]
    """
    Extraccion de los partidos de los proximos <n_days> dias en la competicion <competition> del pais <country>.
    """
    # Definicion de variables
    df_match, df_match_player, df_match_odds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    crawler = FlashscoreCrawler(headless=False)

    # Formateo variables para construir url
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(" ", "-")  # formateo competition para las rutas de archivo y urls
    url = f'https://www.flashscore.com/football/{country_form}/{competition_form}/fixtures/'
    print(f'URL competición: {url}')

    # Ingreso a pagina
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    sleep(random.uniform(2, 4))

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()  
    season_year = crawler.extract_season_year() 
    print(f" {season_year} ".center(120, "-"))
    sleep(random.uniform(2, 4))

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

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Definir paises y competencias a extraer
    l_countries = [1000]
    l_competences = [1001, 1002, 1003]

    d_countries = {-1: "all", 1000: "europe", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    df_comp = pd.read_excel("data/_shared/master_tables/df_competencies.xlsx")
    print(df_comp)
    
    # Por pais
    for id_country in l_countries:

        country = d_countries[id_country]
        print(country)

        # Por competencia
        for id_comp in l_competences:

            row_comp = df_comp[(df_comp['id_country'] == id_country) & (df_comp['id_competition']==id_comp)]
            competition = row_comp['competition_flashscore'].values[0]
            is_cup = row_comp['is_cup'].values[0]
            print(competition, is_cup)
            
            extract_data(id_country, country, id_comp, competition, is_cup=is_cup, n_seasons_max=15)  # df_match, df_match_player, df_match_odds = 

            # Exporto datos
            # df_match.to_excel(f"data/{country}/data_understanding/df_match.xlsx") 
            # df_match_player.to_excel(f"data/{country}/data_understanding/df_match_player.xlsx") 
            # df_match_odds.to_excel(f"data/{country}/data_understanding/df_match_odds.xlsx") 
