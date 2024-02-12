# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
from tqdm import tqdm
import warnings


class FlashscoreCrawler(Crawler):
    """
    Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper
    normal, scraper de fields especificos poro falla y scraper de proximos partidos...
    """
    def __init__(self, headless, path=None, browser="Chrome"):
        super().__init__(headless, path, browser)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 0.8  # Espera para elementos que muchas veces no estan # con 0.2 fallaba extraccion de campos que si estaban como goles
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

    def extract_urls_temporadas(self):
        """
        Extrae listado de urls de las temporadas de una competicion.
        :return: List. Urls de temporadas de la competicion.
        """
        # Extraigo tags
        l_tag_temporadas = super().extract_tags(xpath='.//section[@id="tournament-page-archiv"]//div[@class="archive__row"]/div[@class="archive__season"]/a')

        # Obtengo href de cada tag y lo hago lista
        l_urls_temporadas = [tag.get_attribute('href') for tag in l_tag_temporadas]
        return l_urls_temporadas

    def extract_basic_data_from_resumen(self, extract_goles=True):
        """
        Extrae datos basicos de un partido como equipos, fecha, cancha, goles, etc.
        :return: Diccionario.
        """
        d_nueva_fila = {}
        d_field_xpath = {
            'equipo_loc': './/div[@class="duelParticipant"]/div[starts-with(@class, "duelParticipant__home")]',
            'equipo_vis': './/div[@class="duelParticipant"]/div[starts-with(@class, "duelParticipant__away")]',
            'arbitro': './/div[@class="matchInfoData"]//span[contains(text(), "Árbitro")]/following-sibling::span',
            'cancha': './/div[@class="matchInfoData"]//span[contains(text(), "Estadio")]/following-sibling::span'
        }

        if extract_goles:
            d_field_xpath_goles = {
                'goles_loc': './/div[@class="duelParticipant__score"]//div[@class="detailScore__wrapper"]/span[1]',
                'goles_vis': './/div[@class="duelParticipant__score"]//div[@class="detailScore__wrapper"]/span[3]'
            }
            d_field_xpath.update(d_field_xpath_goles)

        # Extraigo el primer campo con espera para evitar extraer sin que haya cargado la pagina
        d_nueva_fila['fecha'] = super().extract_tag(xpath='.//div[@class="duelParticipant"]/div[@class="duelParticipant__startTime"]', text=True, sec_wait=self.SEC_WAIT_MED)

        # Por campo a extraer
        for field, xpath in d_field_xpath.items():
            value = super().extract_tag(xpath=xpath, text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_nueva_fila[field] = value

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_formacion(self):
        """
        Extrae nombre de jugadores titulares, suplentes y ausentes de cada equipo.
        :return: Diccionario.
        """
        # Definicion de variables
        d_nueva_fila = {}
        d_formaciones = {"Formaciones iniciales": "tit", 'Suplentes': 'sup', 'Jugadores reemplazados': 'sup_ing',
                         'Jugadores ausentes': 'aus'}  # 'Alineaciones iniciales': 'tit', 'Jugadores sustituidos': 'sup_ing',

        # Por formacion ("Formacion inicial", "Suplentes" y  "Ausentes")
        for formacion, titularidad in d_formaciones.items():

            SEC_WAIT = self.SEC_WAIT_MIN if formacion=='Jugadores ausentes' else self.SEC_WAIT_MAX  # Jugadores ausentes muchas veces no esta. Esto agiliza la extraccion.

            # Si existe dicha formacion
            tag_formacion = super().extract_tag(xpath=f'.//div[@class="lf__lineUp"]//div[text()="{formacion}"]', sec_wait=SEC_WAIT, print_fail=True)

            if tag_formacion:

                # Extraigo listado de jugadores
                l_tags_jug_loc = super().extract_tags(tag_inicial=tag_formacion, xpath='.//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None
                l_tags_jug_vis = super().extract_tags(tag_inicial=tag_formacion, xpath='.//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None

                if l_tags_jug_loc:
                    l_nombres_loc = [self.extract_nombre(tag) for tag in l_tags_jug_loc]
                    for i, nombre in enumerate(l_nombres_loc):
                        d_nueva_fila[f'jug_{titularidad}_loc_{i + 1}'] = nombre

                if l_tags_jug_vis:
                    l_nombres_vis = [self.extract_nombre(tag) for tag in l_tags_jug_vis]
                    for i, nombre in enumerate(l_nombres_vis):
                        d_nueva_fila[f'jug_{titularidad}_vis_{i + 1}'] = nombre

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_dts(self):
        """
        Extrae entrenadores tanto del equipo local como del equipo visitante.
        :return: Diccionario. Tanto para el equipo local como para el visitante, entrenador y su valor como key y value.
        """
        d_nueva_fila = {}

        seccion_entrenadores = super().extract_tag(xpath='.//div[text()="Entrenadores"]', sec_wait=self.SEC_WAIT_MIN)

        # Si existe la seccion de entrenadores
        if seccion_entrenadores:

            # Extraigo entrenadores
            dt_loc_tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='./following-sibling::div//div[@class="lf__side"][1]//a[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN)
            dt_vis_tag = super().extract_tag(tag_inicial= seccion_entrenadores, xpath='./following-sibling::div//div[@class="lf__side"][2]//a[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN)

            if dt_loc_tag:
                d_nueva_fila['dt_loc'] = self.extract_nombre(dt_loc_tag)

            if dt_vis_tag:
                d_nueva_fila['dt_vis'] = self.extract_nombre(dt_vis_tag)

        return d_nueva_fila

    def extract_nombre(self, tag):  # Funcion auxiliar de extract_formacion() y extract_dts()
        """
        Intenta extraer nombre completo del jugador o dt desde su link, o bien, obtiene el nombre reducido del texto.
        :param tag: Tag <a> con el link del dt o jugador como href y el nombre reducido del jugador o dt como texto.
        :return: String. Nombre del jugador o dt.
        """
        try:
            url_with_nombre = tag.get_attribute('href')
            nombre = url_with_nombre.split('/')[4].replace('-', ' ')
        except:
            nombre = tag.text
        return nombre

    def extract_estadisticas(self):
        """
        Extrae todas las estadisticas del partido que haya
        :return: Diccionario. Tanto para el equipo local como para el visitante, estadistica y su valor como key y value.
        """
        d_nueva_fila = {}

        # Extraigo tags de estadisticas
        l_tags_estadisticas = super().extract_tags(xpath=f'.//div[@data-testid="wcl-statistics"]', sec_wait=self.SEC_WAIT_MAX, print_fail=True)
        # print("Cantidad de estadisticas a recolectar", len(l_tags_estadisticas))

        # Por estadistica
        for tag in l_tags_estadisticas:

            # Extraigo el nombre de la estadistica
            nombre_estadistica = super().extract_tag(tag_inicial=tag, xpath='.//div[@data-testid="wcl-statistics-category"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            nombre_estadistica_form = nombre_estadistica.lower().replace(" ", "_").replace('á', 'a').replace('é', 'e').replace('í', 'i').replace("ó", "o").replace('ú', 'u')
            # print(f"Estadistica a recolectar: {nombre_estadistica} --> Nombre formateado: {nombre_estadistica_form}")

            # Extraigo valores de la estadistica para el local y para el visitante
            d_nueva_fila[f'{nombre_estadistica_form}_loc'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "homeValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)
            d_nueva_fila[f'{nombre_estadistica_form}_vis'] = super().extract_tag(tag_inicial=tag, xpath=f'.//div[contains(@class, "awayValue") and @data-testid="wcl-statistics-value"]', text=True, sec_wait=self.SEC_WAIT_MIN, print_fail=False)

        # print(d_nueva_fila)
        return d_nueva_fila

    def extract_cuota(self):

        # Definicion de variables
        d_nueva_fila = {}
        l_odds = ['odds_loc', 'odds_emp', 'odds_vis']

        # Extraer las cuotas en una lista
        odds_elements = super().extract_tags(xpath='.//div[@class="cellWrapper"]', sec_wait=self.SEC_WAIT_MAX, print_fail=False)

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
    
    def extract_bajas_pre_partido(self):

        d_nueva_fila = {}

        # Extraer bajas
        tag_bajas = super().extract_tag(xpath=f'.//div[@id="detail"]//div[text()="Bajas"]', sec_wait=self.SEC_WAIT_MIN, print_fail=True)

        if tag_bajas:

            # Extraigo listado de jugadores
            l_tags_jug_loc = super().extract_tags(tag_inicial=tag_bajas, xpath='.//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None
            l_tags_jug_vis = super().extract_tags(tag_inicial=tag_bajas, xpath='.//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=self.SEC_WAIT_MIN, print_fail=False)  # Es lista de tags o None

            if l_tags_jug_loc:
                l_nombres_loc = [self.extract_nombre(tag) for tag in l_tags_jug_loc]
                for i, nombre in enumerate(l_nombres_loc):
                    d_nueva_fila[f'jug_aus_loc_{i + 1}'] = nombre

            if l_tags_jug_vis:
                l_nombres_vis = [self.extract_nombre(tag) for tag in l_tags_jug_vis]
                for i, nombre in enumerate(l_nombres_vis):
                    d_nueva_fila[f'jug_aus_vis_{i + 1}'] = nombre

        print(d_nueva_fila)
        return d_nueva_fila


def extract_data_flashscore(pais, competicion, is_cup, n_temps_max=None, export=True):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"
    # path_driver_exe = "/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/collect_initial_data/copia_chrome_driver/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")
    df_part, df_part_jug = pd.DataFrame(), pd.DataFrame()
    pais_form = pais.lower().replace(' ', "_")
    competicion_form = competicion.lower().replace(" ", "-")  # formateo competicion para las rutas de archivo y urls
    ruta_base = f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg"

    # Ingreso a pagina
    url = f'https://www.flashscore.com.ar/futbol/{pais.lower()}/{competicion_form}/archivo/'
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
    print(f' Competicion: {competicion} '.center(120, '+'), f"\nURL competición: {url}")

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()

    # Extraigo urls de las distintas temporadas (años) de la competicion
    l_urls_temporadas = crawler.extract_urls_temporadas()
    if n_temps_max is not None:
        l_urls_temporadas = l_urls_temporadas[:n_temps_max]
    print(f'Cantidad de temporadas: {len(l_urls_temporadas)}')

    # POR TEMPORADA
    for url_temp in l_urls_temporadas:  # De mas reciente a menos reciente

        # Ingreso a pagina de temporada e imprimo año de la temporada
        crawler.driver.get(url_temp)
        temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
        print(f" {temp_year} ".center(120, "-"))

        # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
        while True:
            boton_mostrar = crawler.extract_tag(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=crawler.SEC_WAIT_MAX) # boton_mostrar = crawler.extract_tag(xpath='.//div[@id="live-table"]//div[@class="tabs" and text()="Últimos Resultados"]/following_sibling()::div//a[@class="event__more event__more--static"]', sec_wait=crawler.SEC_WAIT_MAX * 15)
            if crawler.click_boton(boton_mostrar) is False:
                break

        # Extraigo partidos (items) y sus ids
        l_items = crawler.extract_tags(xpath='.//div[@id="live-table"]//div[@class="event__match event__match--static event__match--twoLine" or @title="¡Haga click para detalles del partido!"]', sec_wait=crawler.SEC_WAIT_MAX)
        l_ids = [item.get_attribute('id') for item in l_items]
        print(f"Partidos recolectados de la temporada {temp_year} (e.g. en premier league deberian ser 380): {len(l_ids)}")
        progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

        # POR PARTIDO (c/u identificado con un id)
        for id_part in l_ids:

            # Ingreso a pagina de informacion del partido
            id_part = id_part[id_part.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")
            url_partido = f'https://www.flashscore.com.ar/partido/{id_part}/#/resumen-del-partido/resumen-del-partido'
            crawler.driver.get(url_partido)

            # Reinicio diccionario en el que guardar datos del nuevo partido
            d_new_row_df_part = {'id_part': id_part, 'competicion': competicion, 'temporada': temp_year, 'pais': pais, 'es_copa': is_cup}
            d_new_row_df_part_jug = {'id_part': id_part}

            # EXTRACCION DE CAMPOS
            ## Extraigo campos de hoja "Resumen"
            d_new_row_df_part.update(crawler.extract_basic_data_from_resumen())

            ## Si tiene hoja "Estadisticas", extraigo campos
            boton_estadisticas = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Estadísticas"]', sec_wait=crawler.SEC_WAIT_MED, print_fail=True)
            if crawler.click_boton(boton_estadisticas) is not False:
                d_new_row_df_part.update(crawler.extract_estadisticas())

            ## Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
            if crawler.extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=crawler.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...
                d_new_row_df_part.update(crawler.extract_cuota())

            ## Si tiene hoja "Formaciones", extraigo campos
            boton_formaciones = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Formaciones" or text()="Alineaciones"]', sec_wait=crawler.SEC_WAIT_MED, print_fail=True)
            if crawler.click_boton(boton_formaciones) is not False:
                d_new_row_df_part_jug.update(crawler.extract_formacion())
                d_new_row_df_part.update(crawler.extract_dts())

            # GUARDADO DE DATOS EN DATAFRAME
            df_part = pd.concat([df_part, pd.DataFrame(d_new_row_df_part, index=[0])])
            df_part_jug = pd.concat([df_part_jug, pd.DataFrame(d_new_row_df_part_jug, index=[0])])
            progress_bar.update(1)

        # Cerrar la barra de progreso al finalizar
        progress_bar.close()

        if export:
            # Guardo partidos de la temporada (por seguridad)
            df_part.to_excel(f'{ruta_base}/por_temporada/df_part/{competicion_form}_{temp_year.replace("/", "_")}_{pais_form}.xlsx', index=False)
            df_part_jug.to_excel(f'{ruta_base}/por_temporada/df_part_jug/{competicion_form}_{temp_year.replace("/", "_")}_{pais_form}.xlsx', index=False)

    if export:
        # Guardo partidos de la competicion
        df_part.to_excel(f'{ruta_base}/por_competicion/df_part/{competicion_form}_{pais_form}.xlsx', index=False)
        df_part_jug.to_excel(f'{ruta_base}/por_competicion/df_part_jug/{competicion_form}_{pais_form}.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part, df_part_jug


def extract_data_faltante_flashcore(df_part_old, df_part_jug_old, pais, export=True):
    """
    Extraer automaticamente los partidos que faltan en df_part y df_part_jug. Sobretodo es esencial para poder construir estadisticas en 
    partidos nuevos a predecir.
    """
  # DEFINCION DE PARAMETROS & VARIABLES
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")
    df_part, df_part_jug = pd.DataFrame(), pd.DataFrame()
    pais_form = pais.lower().replace(' ', "_")
    ruta_base = f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais_form}/data_seg"

    # Obtengo las competencias a extraer 
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx')
    df_comp_pais = df_comp[df_comp['pais_flashscore'] == pais.capitalize()]
    print(f' PAIS: {pais} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_pais['competicion_flashscore']}")

    # POR COMPETICION
    for i, row in df_comp_pais.iterrows():

        if row['competicion_flashscore'] not in ['Championship']:

            competicion_form = row['competicion_flashscore'].lower().replace(" ", "-")  # formateo competicion para las rutas de archivo y urls
            is_cup = row['is_cup']

            # Ingreso a pagina
            url = f'https://www.flashscore.com.arc/futbol/{pais.lower()}/{competicion_form}'
            crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
            print(f' Competicion: {competicion_form} '.center(120, '+'), f"\nURL competición: {url}")

            # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
            crawler.accept_cookies()

            temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
            print(f" {temp_year} ".center(120, "-"))

            # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
            while True:
                boton_mostrar = crawler.extract_tag(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=crawler.SEC_WAIT_MAX) # boton_mostrar = crawler.extract_tag(xpath='.//div[@id="live-table"]//div[@class="tabs" and text()="Últimos Resultados"]/following_sibling()::div//a[@class="event__more event__more--static"]', sec_wait=crawler.SEC_WAIT_MAX * 15)
                if crawler.click_boton(boton_mostrar) is False:
                    break

            # Extraigo partidos (items) y sus ids
            l_items = crawler.extract_tags(xpath='.//div[@id="live-table"]//div[@class="event__match event__match--static event__match--twoLine" or @title="¡Haga click para detalles del partido!"]', sec_wait=crawler.SEC_WAIT_MAX)
            l_ids = [item.get_attribute('id') for item in l_items]
            print(f"Partidos recolectados de la temporada {temp_year} (e.g. en premier league deberian ser 380): {len(l_ids)}")
            progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

            # POR PARTIDO (c/u identificado con un id)
            for id_part in l_ids:

                id_part = id_part[id_part.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")

                if id_part not in df_part_old['id_part']:

                    # Ingreso a pagina de informacion del partido
                    url_partido = f'https://www.flashscore.com.ar/partido/{id_part}/#/resumen-del-partido/resumen-del-partido'
                    crawler.driver.get(url_partido)

                    # Reinicio diccionario en el que guardar datos del nuevo partido
                    d_new_row_df_part = {'id_part': id_part, 'competicion': row['competicion_flashscore'], 'temporada': temp_year, 'pais': pais, 'es_copa': is_cup}
                    d_new_row_df_part_jug = {'id_part': id_part}

                    # EXTRACCION DE CAMPOS
                    ## Extraigo campos de hoja "Resumen"
                    d_new_row_df_part.update(crawler.extract_basic_data_from_resumen())

                    ## Si tiene hoja "Estadisticas", extraigo campos
                    boton_estadisticas = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Estadísticas"]', sec_wait=crawler.SEC_WAIT_MED, print_fail=True)
                    if crawler.click_boton(boton_estadisticas) is not False:
                        d_new_row_df_part.update(crawler.extract_estadisticas())

                    ## Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                    if crawler.extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=crawler.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...
                        d_new_row_df_part.update(crawler.extract_cuota())

                    ## Si tiene hoja "Formaciones", extraigo campos
                    boton_formaciones = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Formaciones" or text()="Alineaciones"]', sec_wait=crawler.SEC_WAIT_MED, print_fail=True)
                    if crawler.click_boton(boton_formaciones) is not False:
                        d_new_row_df_part_jug.update(crawler.extract_formacion())
                        d_new_row_df_part.update(crawler.extract_dts())

                    # GUARDADO DE DATOS EN DATAFRAME
                    df_part = pd.concat([df_part, pd.DataFrame(d_new_row_df_part, index=[0])])
                    df_part_jug = pd.concat([df_part_jug, pd.DataFrame(d_new_row_df_part_jug, index=[0])])
                    progress_bar.update(1)

            # Cerrar la barra de progreso al finalizar
            progress_bar.close()

    # Concateno dfs
    print(df_part)
    print(f"Cantidad de nuevos partidos: {df_part.shape[0]}")
    df_part_concat = pd.concat([df_part, df_part_old], axis=0)
    df_part_jug_concat = pd.concat([df_part_jug, df_part_jug_old], axis=0)

    if export:
        # Guardo partidos
        df_part.to_excel(f'{ruta_base}/df_part_new_matches_{competicion_form}_{temp_year.replace("/", "_")}_{pais_form}.xlsx', index=False)
        df_part_jug.to_excel(f'{ruta_base}/df_part_jug_new_matches_{competicion_form}_{temp_year.replace("/", "_")}_{pais_form}.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part, df_part_jug


def prueba():
    # Selecciono pais a extraer y obtengo las competencias y su categoria
    pais = 'England'  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    competicion = 'Premier league'
    is_cup = 0

    # Extraigo partidos
    # df_part, df_part_jug = extract_data_flashscore(pais, competicion, is_cup, export=False)
    
    # Extraigo partidos faltantes
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_part.xlsx')
    df_part_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_part_jug.xlsx')
    df_part, df_part_jug = extract_data_faltante_flashcore(df_part, df_part_jug, pais, export=False)

    # Exporto datasets
    df_part.to_excel('/Users/nachomondino/Desktop/df_part.xlsx', index=False)
    df_part_jug.to_excel('/Users/nachomondino/Desktop/df_part_jug.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()