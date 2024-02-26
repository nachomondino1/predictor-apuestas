# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p2_data_understanding.collect_initial_data.web_scraping_selenium import Crawler
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta


class FlashscoreCrawler(Crawler):
    """
    Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper
    normal, scraper de fields especificos poro falla y scraper de proximos partidos...
    Contiene todo los xpath.
    """
    def __init__(self, headless, path=None, browser="Chrome"):
        super().__init__(headless, path, browser)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 0.8  # Espera para elementos que muchas veces no estan # con 0.2 fallaba extraccion de campos que si estaban como goals
        self.SEC_WAIT_MED = 2  # Espera para elementos que casi siempre estan
        self.SEC_WAIT_MAX = 5  # Espera para elementos que casi siempre estan

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

        try:
            season_year = season_year.replace("/", "_")
        except:
            print("The season is not a string. Probably it failed the extraction.")
        return season_year

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
        l_items = super().extract_tags(xpath='.//div[@id="live-table"]//div[@class="event__match event__match--static event__match--twoLine" or @title="Click for match detail!"]', sec_wait=self.SEC_WAIT_MAX)
        l_ids = [item.get_attribute('id') for item in l_items]
        l_ids_clean = self.clean_id(l_ids)
        return l_ids_clean
    
    def extract_id_next_matches(self, n_days):
            
        # Extraigo partidos (items) y sus ids --> NO PUDE EXTRAER LOS SVG.. PERO SI EL DIV DE EVENT_TIME... VER 
        l_items = super().extract_tags(xpath='.//div[@id="live-table"]//div[@class="sportName soccer"]//div[contains(@class, "event__match--scheduled")]', sec_wait=self.SEC_WAIT_MAX)

        # Filtro partidos por fecha
        l_items_filt = self.select_items_by_date(l_items, n_days)

        # Obtengo ids
        l_ids = [item.get_attribute('id') for item in l_items_filt]
        l_ids_clean = self.clean_id(l_ids)
        return l_ids_clean

    def clean_id(self, l_ids):
        l_ids_clean = []
        for id_match in l_ids:
            id_match = id_match[id_match.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")
            l_ids_clean.append(id_match)
        return l_ids_clean

    def extract_match_data(self):
        """
        Extrae datos del partido siendo este un partido ya jugado.
        """

        # Reinicio diccionario en el que guardar datos del nuevo match
        d_new_row_df_match = {}
        d_new_row_df_match_player = {}

        # EXTRACCION DE CAMPOS
        ## Extraigo campos de hoja "summary"
        d_new_row_df_match.update(self.extract_basic_data_from_summary())

        ## Si tiene hoja "stats", extraigo campos
        boton_stats = super().extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Stats"]', sec_wait=self.SEC_WAIT_MED, print_fail=True)
        if super().click_boton(boton_stats) is not False:
            d_new_row_df_match.update(self.extract_stats())

        ## Si existe la seccion "odds pre-match", extraigo odds de Bet365
        if super().extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=self.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las oddss pero no hay valores...
            d_new_row_df_match.update(self.extract_odds())

        ## Si tiene hoja "Formations", extraigo campos
        boton_formations = super().extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Lineups"]', sec_wait=self.SEC_WAIT_MED, print_fail=True)
        if super().click_boton(boton_formations) is not False:
            d_new_row_df_match_player.update(self.extract_lineups())
            d_new_row_df_match.update(self.extract_coaches())
        
        return d_new_row_df_match, d_new_row_df_match_player

    def extract_next_match_data(self):
        """
        Extrae datos del partido siendo este un partido aun no jugado.
        """
        # Reinicio diccionario en el que guardar datos del nuevo partido
        d_new_row_df_match = {}
        d_new_row_df_match_player = {}

        # Extraigo campos de hoja "Resumen"
        d_new_row_df_match.update(self.extract_basic_data_from_summary(extract_goals=False))

        #  Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
        if super().extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=self.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...
            d_new_row_df_match.update(self.extract_odds())

        ## Si tiene hoja "Formations", extraigo campos
        boton_formations = super().extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Lineups"]', sec_wait=self.SEC_WAIT_MED, print_fail=True)
        if super().click_boton(boton_formations) is not False:
            d_new_row_df_match_player.update(self.extract_lineups())
            d_new_row_df_match.update(self.extract_coaches())
        else:
            d_new_row_df_match_player.update(self.extract_bajas_pre_partido()) # FALTARIA TMB SECCION POSIBLES BAJAS.
            
        return d_new_row_df_match, d_new_row_df_match_player

    def extract_basic_data_from_summary(self, extract_goals=True):
        """
        Extrae datos basicos de un match como equipos, fecha, cancha, goals, etc.
        :return: Diccionario.
        """
        d_new_row = {}
        d_field_xpath = {
            'team_home': './/div[@class="duelParticipant"]/div[starts-with(@class, "duelParticipant__home")]',
            'team_away': './/div[@class="duelParticipant"]/div[starts-with(@class, "duelParticipant__away")]',
            'referee': './/div[@class="matchInfoData"]//span[contains(text(), "Referee")]/following-sibling::span[@class="matchInfoItem__value"]', # Intente el svg (para evitar el texto "Referee") pero no lo encuentra
            'venue': './/div[@class="matchInfoData"]//span[contains(text(), "Venue")]/following-sibling::span[@class="matchInfoItem__value"]', 
            'capacity': './/div[@class="matchInfoData"]//span[contains(text(), "Capacity")]/following-sibling::span[@class="matchInfoItem__value"]', 
            'attendance': './/div[@class="matchInfoData"]//span[contains(text(), "Attendance")]/following-sibling::span[@class="matchInfoItem__value"]'
        }

        if extract_goals:
            d_field_xpath_goals = {
                'goals_home': './/div[@class="detailScore__wrapper"]/span[1]', # './/div[@class="detailScore__wrapper"]/div[@class="detailScore__divider"]/preceding-sibling::span',
                'goals_away': './/div[@class="detailScore__wrapper"]/span[3]' #  './/div[@class="detailScore__wrapper"]/div[@class="detailScore__wrapper"]/following-sibling::span'
            }
            d_field_xpath.update(d_field_xpath_goals)

        # Extraigo el primer campo con espera para evitar extraer sin que haya cargado la pagina
        d_new_row['date'] = super().extract_tag(xpath='.//div[@class="duelParticipant"]/div[@class="duelParticipant__startTime"]', text=True, sec_wait=self.SEC_WAIT_MED)

        # Por campo a extraer
        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_new_row[field] = value

        # print()
        # print(d_new_row)
        return d_new_row

    def extract_lineups(self):
        """
        Extrae name de jugadores titulares, suplentes y ausentes de cada equipo.
        :return: Diccionario.
        """
        # Definicion de variables
        d_new_row = {}
        d_formations = {"Starting Lineups": "start", 'Substitutes': 'sub', 'Substituted players': 'sub_enter',
                         'Missing Players': 'miss'}

        # Por formation ("Formation inicial", "Suplentes" y  "Ausentes")
        for formation, titularidad in d_formations.items():

            SEC_WAIT = self.SEC_WAIT_MAX if formation=="Starting Lineups" else self.SEC_WAIT_MIN  # Jugadores ausentes muchas veces no esta. Y suplentes en partidos viejos tampocoEsto agiliza la extraccion.

            # Si existe dicha formation
            tag_lineup = super().extract_tag(xpath=f'.//div[@class="lf__lineUp"]/div[@class="section"]/div[text()="{formation}"]', sec_wait=SEC_WAIT, print_fail=True)
            # print(formation, SEC_WAIT)

            if tag_lineup:

                # Extraigo listado de jugadores
                l_tags_player_home = super().extract_tags(tag_inicial=tag_lineup, xpath='.//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None
                l_tags_player_away = super().extract_tags(tag_inicial=tag_lineup, xpath='.//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None

                if l_tags_player_home:
                    l_names_home = [self.extract_name(tag) for tag in l_tags_player_home]
                    for i, name in enumerate(l_names_home):
                        d_new_row[f'player_{titularidad}_home_{i + 1}'] = name

                if l_tags_player_away:
                    l_names_away = [self.extract_name(tag) for tag in l_tags_player_away]
                    for i, name in enumerate(l_names_away):
                        d_new_row[f'player_{titularidad}_away_{i + 1}'] = name

        # print(d_new_row)
        return d_new_row

    def extract_coaches(self):
        """
        Extrae entrenadores tanto del equipo local como del equipo visitante.
        :return: Diccionario. Tanto para el equipo local como para el visitante, entrenador y su valor como key y value.
        """
        d_new_row = {}

        seccion_entrenadores = super().extract_tag(xpath='.//div[@class="lf__lineUp"]/div[@class="section"]/div[text()="Coaches"]', sec_wait=self.SEC_WAIT_MIN)

        # Si existe la seccion de entrenadores
        if seccion_entrenadores:

            # Extraigo entrenadores
            coach_home_tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='./following-sibling::div//div[@class="lf__side"][1]//a[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN)
            coach_away__tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='./following-sibling::div//div[@class="lf__side"][2]//a[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN)

            if coach_home_tag:
                d_new_row['coach_home'] = self.extract_name(coach_home_tag)

            if coach_away__tag:
                d_new_row['coach_away'] = self.extract_name(coach_away__tag)

        # print(d_new_row)
        return d_new_row

    def extract_name(self, tag):  # Funcion auxiliar de extract_lineups() y extract_coaches()
        """
        Intenta extraer name completo del jugador o dt desde su link, o bien, obtiene el name reducido del texto.
        :param tag: Tag <a> con el link del dt o jugador como href y el name reducido del jugador o dt como texto.
        :return: String. Name del jugador o dt.
        """
        try:
            url_with_name = tag.get_attribute('href')
            name = url_with_name.split('/')[4].replace('-', ' ')
        except:
            name = tag.text
        return name

    def extract_stats(self):
        """
        Extrae todas las stats del match que haya
        :return: Diccionario. Tanto para el equipo local como para el visitante, stat y su valor como key y value.
        """
        d_new_row = {}

        # Extraigo tags de stats
        l_tags_stats = super().extract_tags(xpath=f'.//div[@data-testid="wcl-statistics"]', sec_wait=self.SEC_WAIT_MAX, print_fail=True)
        # print("Cantidad de stats a recolectar", len(l_tags_stats))

        # Por stat
        for tag in l_tags_stats:

            # Extraigo el name de la stat
            name_stat = super().extract_tag(tag_inicial=tag, xpath='.//div[@data-testid="wcl-statistics-category"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            name_stat_form = name_stat.lower().replace(" ", "_").replace('á', 'a').replace('é', 'e').replace('í', 'i').replace("ó", "o").replace('ú', 'u')
            # print(f"Stat a recolectar: {name_stat} --> Name formateado: {name_stat_form}")

            # Extraigo valores de la stat para el local y para el visitante
            d_new_row[f'{name_stat_form}_home'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "homeValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_new_row[f'{name_stat_form}_away'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "awayValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)

        # print(d_new_row)
        return d_new_row

    def extract_odds(self):
        """
        Extrae cuotas de casa de apuestas Bet365. 
        Tambien se podria obtener de la hoja "Odds".
        """
        # Definicion de variables
        d_new_row = {}

        # Extraer las oddss en una lista
        l_odds_elements = super().extract_tags(xpath='.//div[@class="oddsRowContent"]//div[@class="cellWrapper"]//span[@class="oddsValueInner"]', sec_wait=self.SEC_WAIT_MAX, print_fail=True)

        if len(l_odds_elements) > 0:
            l_text_odds_elements = [elem.text for elem in l_odds_elements]

            d_new_row['odds_home'] = l_text_odds_elements[0]
            d_new_row['odds_draw'] = l_text_odds_elements[1]
            d_new_row['odds_away'] = l_text_odds_elements[2]
        return d_new_row

    def extract_bajas_pre_partido(self):

        d_new_row = {}

        # Extraer bajas
        tag_bajas = super().extract_tag(xpath=f'.//div[@id="detail"]//div[text()="Will not play"]', sec_wait=self.SEC_WAIT_MIN, print_fail=True)

        if tag_bajas:

            # Extraigo listado de jugadores
            l_tags_player_home = super().extract_tags(tag_inicial=tag_bajas, xpath='.//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None
            l_tags_player_away = super().extract_tags(tag_inicial=tag_bajas, xpath='.//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None

            if l_tags_player_home:
                l_names_home = [self.extract_name(tag) for tag in l_tags_player_home]
                for i, name in enumerate(l_names_home):
                    d_new_row[f'player_miss_home_{i + 1}'] = name

            if l_tags_player_away:
                l_names_away = [self.extract_name(tag) for tag in l_tags_player_away]
                for i, name in enumerate(l_names_away):
                    d_new_row[f'player_miss_away_{i + 1}'] = name

        # print(d_new_row)
        return d_new_row

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


def extract_data(country: str, competition: str, n_seasons_max: int = 0, export: bool = True):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    :param country: Nombre del pais. (str)
    :param competition: Nombre de competicion. (str)
    :param n_season_max: Cantida de temporadas desde la actual para extraer. 0 para extraer todas las temporadas disponibles. (int)
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_match = pd.concat([df_match, pd.DataFrame(d_new_row, index=[0])])
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"  # path_driver_exe = "./p2_data_understanding/collect_initial_data/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")
    df_match, df_match_player = pd.DataFrame(), pd.DataFrame()

    # Formateo variables para guardado de datos
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(" ", "-")  # formateo competition para las rutas de archivo y urls
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
    for url_season in l_urls_seasons:  # De mas reciente a menos reciente

        # Ingreso a pagina de season e imprimo año de la season
        crawler.driver.get(url_season)
        season_year = crawler.extract_season_year()
        print(f" {season_year} ".center(120, "-"))

        # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la season)
        crawler.click_show_more_matches()

        # Extraigo partidos (items) y sus ids
        l_ids = crawler.extract_id_matches()
        print(f"Partidos recolectados de la season {season_year} (e.g. en premier league deberian ser 380): {len(l_ids)}")
        progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

        # POR MATCH (c/u identificado con un id)
        for id_match in l_ids:

            # Ingreso a pagina de informacion del match
            url_match = f'https://www.flashscore.com/match/{id_match}/#/match-summary'
            crawler.driver.get(url_match)

            # Extraigo todos los datos del partido
            d_new_row_df_match, d_new_row_df_match_player = crawler.extract_match_data()
            d_new_row_df_match.update({'season': season_year})

            # Guardo datos del partido
            df_match = pd.concat([df_match, pd.DataFrame(d_new_row_df_match, index=[id_match])])
            df_match_player = pd.concat([df_match_player, pd.DataFrame(d_new_row_df_match_player, index=[id_match])])
            progress_bar.update(1)

        progress_bar.close()

        if export:
            # Guardo partidos de la season (por seguridad)
            df_match.to_excel(f'{ruta_base}/per_season/df_match/{competition_form}_{season_year}.xlsx', index=True)
            df_match_player.to_excel(f'{ruta_base}/per_season/df_match_player/{competition_form}_{season_year}.xlsx', index=True)

    if export:
        # Guardo partidos de la competition
        df_match.to_excel(f'{ruta_base}/per_competition/df_match/{competition_form}.xlsx', index=True)
        df_match_player.to_excel(f'{ruta_base}/per_competition/df_match_player/{competition_form}.xlsx', index=True)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_match, df_match_player

def extract_next_matches(country: str, competition: str, n_days): # -> tuple[pd.DataFrame, pd.DataFrame]
    
    # Definicion de variables
    df_match, df_match_player = pd.DataFrame(), pd.DataFrame()
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")

    # Formateo variables para guardado de datos
    country_form = country.lower().replace(' ', "-")
    competition_form = competition.lower().replace(" ", "-")  # formateo competition para las rutas de archivo y urls

     # Ingreso a pagina
    url = f'https://www.flashscore.com/football/{country_form}/{competition_form}/fixtures/'
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    print(f'URL competición: {url}')

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
        d_new_row_df_match, d_new_row_df_match_player = crawler.extract_next_match_data()
        d_new_row_df_match.update({'season': season_year})

        # GUARDADO DE DATOS EN DATAFRAME
        df_match = pd.concat([df_match, pd.DataFrame(d_new_row_df_match, index=[id_match])])
        df_match_player = pd.concat([df_match_player, pd.DataFrame(d_new_row_df_match_player, index=[id_match])])
        progress_bar.update(1)

    # Cerrar la barra de progreso al finalizar
    progress_bar.close()

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_match, df_match_player

def extract_missing_data(country, competition, l_ids_already_collected, _print: bool = False):  # Se podria usar la misma funcion que extract_normal pero agregando l_ids_already_collected para filtrar partidos... y  tal vez n_seasons_max=1.
    """
    Extrae los partidos aun no extraidos de una competencia de un country.
    """
  # DEFINCION DE PARAMETROS & VARIABLES
    df_match, df_match_player = pd.DataFrame(), pd.DataFrame()
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")

    # Formateo variables para guardado de datos
    country_form = country.lower().replace(' ', "_")
    competicion_form = competition.lower().replace(" ", "-")  # formateo competition para las rutas de archivo y urls

    # Ingreso a pagina de competition
    url = f'https://www.flashscore.com/football/{country_form}/{competicion_form}'
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    if _print:
        print(url)

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()
    season_year = crawler.extract_season_year()
    if _print:
        print(f" {season_year} ".center(120, "-"))

    # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
    crawler.click_show_more_matches()

    # Extraigo partidos (items) y sus ids
    l_ids = crawler.extract_id_matches()
    l_ids_filt = [id_match for id_match in l_ids if id_match not in l_ids_already_collected]
    if _print:
        print(f"Partidos recolectados de la temporada {season_year} (e.g. en premier league deberian ser 380): {len(l_ids_filt)}")
        progress_bar = tqdm(total=len(l_ids_filt), ncols=80)  # Inicializo barra de progreso

    # POR PARTIDO (c/u identificado con un id)
    for id_match in l_ids_filt:

        # Ingreso a pagina de informacion del partido
        url_partido = f'https://www.flashscore.com/match/{id_match}/#/match-summary'
        crawler.driver.get(url_partido)

        # Extraigo todos los datos del partido
        d_new_row_df_match, d_new_row_df_match_player = crawler.extract_match_data()
        d_new_row_df_match.update({'season': season_year})

        # Guardo datos del partido
        df_match = pd.concat([df_match, pd.DataFrame(d_new_row_df_match, index=[id_match])])
        df_match_player = pd.concat([df_match_player, pd.DataFrame(d_new_row_df_match_player, index=[id_match])])
        if _print:
            progress_bar.update(1)

    # Cerrar la barra de progreso al finalizar
    if _print:
        progress_bar.close()

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_match, df_match_player

def prueba():
    # Selecciono country a extraer y obtengo las competencias y su categoria
    country = 'England'  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    competition = 'Fa Cup'
    n_seasons_max = 16

    # Extraigo partidos
    df_match, df_match_player = extract_data(country, competition, n_seasons_max, export=False)
    
    # Exporto datasets
    df_match.to_excel('/Users/nachomondino/Desktop/df_match.xlsx', index=True)
    df_match_player.to_excel('/Users/nachomondino/Desktop/df_match_player.xlsx', index=True)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()