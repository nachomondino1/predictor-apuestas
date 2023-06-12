# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
from selenium.webdriver.common.keys import Keys
import datetime
import time
import random
import warnings
from tqdm import tqdm


def extract_jugadores_sofifa(l_paises):  # Si bien puede extraer varios paises, creo que lo usare para un solo a la vez. De todas maneras dejo la funcion para multiples paises.
    """
    Obtengo datos de jugadores mediante scrapear sofifa.com
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    df_jug = pd.DataFrame(columns=['id_jugador', 'fifa', 'fecha', 'nombre', 'edad', 'altura', 'pie_habil', 'overall_rating', 'potencial', 'equipo_actual', 'valor_mercado', 'sueldo', 'pais'])  # usar d.keys() de headers... asi es automatico.. Ah no, pues extraigo algunos campos mas..
    crawler = Crawler(headless=True, path=None)
    d_pais_a_seleccionar = {'brazil': 'brasil'}  # Por diferencias entre nombres de paises entre Flashscore y Sofifa

    # Ingreso a pagina de sofifa.com seleccionando como filtro los campos buscados
    crawler.driver.get('https://sofifa.com/players?showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg')

    # Obtengo urls de las paginas de la paginacion (c/pagina es un año o fifa)
    crawler.click_boton(xpath='.//h2//div[@class="dropdown"][1]/a')  # Ver si hace click, tal vez ni hace falta
    l_tag_years = crawler.extract_tags(xpath='.//h2//div[@class="dropdown"][1]/div/a')
    l_urls_years = [tag.get_attribute('href') for tag in l_tag_years]

    # POR FIFA (e.g. Fifa 23, fifa 22, fifa 21, ..., fifa 07)
    for url_year in l_urls_years[::-1]:

        # Ingreso a pagina del año o fifa
        crawler.driver.get(url_year)

        # Obtengo urls de las paginas de la paginacion (c/pagina es una actualizacion de un fifa)
        crawler.click_boton(xpath='.//h2//div[@class="dropdown"][2]/a')  # Ver si hace click, tal vez ni hace falta
        l_tags_act_year = crawler.extract_tags(xpath='.//h2//div[@class="dropdown"][2]/div/a[not(contains(text(), "World Cup"))]')  # Ojo con la actualizacion World Cup 2022...
        l_urls_act_year = [tag.get_attribute('href') for tag in l_tags_act_year]
        l_urls_act_year_sel = [l_urls_act_year[0], l_urls_act_year[-1]]  # Selecciono unicamente la primera y la ultima actualizacion

        # Extraigo fifa
        fifa = crawler.extract_tag(xpath='.//h2/div[@class="dropdown"][1]', text=True)  # Fifa 21
        print(f" FIFA: {fifa} ".center(120, "#"))

        # POR ACTUALIZACION EN DICHO FIFA (e.g. Jun 7, 2023;  Apr 17, 2023; etc)
        for url_year_act in l_urls_act_year_sel:

            crawler.driver.get(url_year_act)

            # Extraigo fecha de la actualizacion e.g. May 16, 2023
            fecha_str = crawler.extract_tag(xpath='.//h2/div[@class="dropdown"][2]', text=True)  # e.g. May 16, 2023
            print(f" Fecha de actualizacion: {fecha_str} ".center(120, "+"))

            # POR PAIS
            for pais in l_paises:
                print(f" PAÍS: {pais} ".center(120, "-"))

                # Remuevo busqueda anterior
                crawler.click_boton(xpath='.//form[@class="relative pjax-form"]//input[@aria-label="Leagues"]//preceding-sibling::div//button', sec_wait=3)

                # Cargo pais como filtro
                input_league = crawler.extract_tag(xpath='.//form[@class="relative pjax-form"]//input[@aria-label="Leagues"]')
                input_league.send_keys(pais)  # "[Argentina] Liga Profesional"

                # Extraigo la liga que segun sofifa se asemeja mas a nuestra busqueda
                liga_a_seleccionar = crawler.extract_tag(xpath='.//form[@class="relative pjax-form"]//input[@aria-label="Leagues"]//parent::div//following-sibling::div//div[starts-with(@class, "choices-item")][1]', text=True)  # e.g. [Argentina] Liga profesional

                # Si encontro resultados para nuestro input
                if liga_a_seleccionar.lower() != "no results found":

                    # Selecciono la liga que mas se asemeja a mi input (cuidado: puede no existar la liga para dicho fifa)
                    input_league.send_keys(Keys.ENTER)  # Tengo que dar enter

                    pais_a_seleccionar = liga_a_seleccionar[liga_a_seleccionar.find('[')+1: liga_a_seleccionar.find(']')].lower()  # e.g. [Argentina] Liga profesional
                    if pais_a_seleccionar in d_pais_a_seleccionar.keys():  # Por ejemplo, en flashscore aparece 'Brasil' mientras que en Sofifa aparece Brazil
                        pais_a_seleccionar = d_pais_a_seleccionar[pais_a_seleccionar]
                    print(f"Pais a seleccionar: {pais_a_seleccionar}")

                    # Si la liga a seleccionar corresponde a la del pais
                    if pais in pais_a_seleccionar:

                        # Clickeo en buscar jugadores
                        crawler.click_boton(xpath='.//button[text()="Submit"]')

                        # POR PAGINA CON LISTADO DE JUGADORES
                        while True:

                            # Obtengo tags de jugadores
                            l_tag_jugadores = crawler.extract_tags(xpath='.//table[@class="table table-hover persist-area"]/tbody/tr')
                            print(f"Cantidad de jugadores: {len(l_tag_jugadores)}")

                            # Por jugador
                            for tag in l_tag_jugadores:

                                # Extraigo datos del jugador
                                d_data = {}
                                d_data['id_jugador'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pi"]', text=True)
                                d_data['fifa'] = fifa
                                d_data['fecha'] = fecha_str
                                d_data['nombre'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@class="col-name"]/a', attribute="aria-label") # Nombre corto (e.g. l. gonzalez pirez) d_data['nombre'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@class="col-name"]/a/div[@class="ellipsis"]', text=True)
                                d_data['edad'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="ae"]', text=True)
                                d_data['altura'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="hi"]', text=True)
                                d_data['pie_habil'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pf"]', text=True)
                                d_data['overall_rating'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="oa"]', text=True)
                                d_data['potencial'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pt"]', text=True)
                                d_data['equipo_actual'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@class="col-name"]/div[@class="ellipsis"]/a', text=True)
                                d_data['valor_mercado'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="vl"]', text=True)
                                d_data['sueldo'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="wg"]', text=True)
                                d_data['pais'] = pais

                                # Formateo campo altura # e.g. 193cm / 6'4" --> 193
                                d_data['altura'] = d_data['altura'].split('cm')[0]

                                # Guardo los datos del jugador para dicho año
                                nueva_fila_df = pd.DataFrame([d_data])
                                df_jug = pd.concat([df_jug, nueva_fila_df], ignore_index=True)
                                print(d_data)

                            # Clickeo en boton "Next" para recorrer todas las paginas
                            if crawler.click_boton(xpath='.//div[@class="pagination"]//span[contains(@class, "right")]//parent::a') is False:
                                print('Ya no hay mas boton "Next". Es decir, ya no hay mas jugadores en la liga.')
                                break

                    # Si la liga a seleccionar no corresponde a la del pais
                    else:
                        print(f'Sofifa encontró la liga {liga_a_seleccionar} la cual no corresponde a nuestra busqueda {pais}. Es posible que no exista la liga de {pais} en el fifa del año {fecha_str}')

                # Si no encontro resultados para nuestro input ("No results found")
                else:
                    print(f'Sofifa no encontró resultados a nuestra busqueda. Es posible que no exista la liga de {pais} en el fifa del año {fecha_str}')

                    # Seleccionar el texto cargado en el input y lo borro
                    input_league.send_keys(Keys.SHIFT + Keys.HOME)
                    input_league.send_keys(Keys.DELETE)

                # Exporto datos de la actualizacion (Por seguridad)
                df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/data_seg/entidad_jugadores_{fecha_str}.xlsx')

    # Exporto dataset final y cierro webdriver
    df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/entidad_jugadores.xlsx')
    crawler.driver.close()
    return df_jug

def extract_partidos_flashscore(l_paises):  # Cambiar argumentos...
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    fecha_act = datetime.datetime.now()
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    crawler = FlashscoreCrawler(headless=True, path=None)
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/df_competencias.xlsx')

    # POR PAIS
    for pais in l_paises:

        df_part = pd.DataFrame()  # Reinicio df por pais # No hace falta definir columnas por mas que no haya extraido partidos

        # Selecciono competencias del pais
        df_comp = df_comp[df_comp['pais'] == pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
        print(f' PAIS: {pais} '.center(120, '#'))

        # POR COMPETICION
        for competicion, categoria in zip(df_comp['nombre'], df_comp['categoria']):

            # Obtengo datos de la competicion
            print(f' Competicion: {competicion} '.center(120, '+'))
            competicion_form = competicion.replace(" ", "-")  # formateo competicion para las rutas de archivo y urls

            # Ingreso a pagina
            url = f'https://www.flashscore.es/futbol/{pais}/{competicion_form}/archivo/'
            crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
            print(url)

            # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
            crawler.click_boton(xpath='.//button[@id="onetrust-accept-btn-handler"]')

            # Extraigo links de temporadas (años)
            l_tag_temporadas = crawler.extract_tags(xpath='.//section[@id="tournament-page-archiv"]//div[@class="archive__row"]/div[@class="archive__season"]/a')
            l_urls_temporadas = [tag.get_attribute('href') for tag in l_tag_temporadas]
            print(f'Cantidad de temporadas: {len(l_urls_temporadas)}')

            # POR TEMPORADA (PAGINA DE PAGINACION)
            for url_temp in l_urls_temporadas:

                # Ingreso a pagina de temporada e imprimo año de la temporada
                crawler.driver.get(url_temp)
                temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
                print(f" {temp_year} ".center(120, "-"))

                # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
                crawler.click_boton(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=SEC_WAIT_LONG * 3, repeat_click=True)  # Si hace click, es None. Si falla, es un str  # A veces me tira (y no entiendo por qué): selenium.common.exceptions.StaleElementReferenceException: Message: stale element reference: stale element not found

                # Extraigo partidos (items) y sus ids
                l_items = crawler.extract_tags(xpath='.//div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')
                l_ids = [item.get_attribute('id') for item in l_items]
                print(f"Cantidad de items (partidos): {len(l_items)}")

                # Inicializo barra de progreso
                progress_bar = tqdm(total=len(l_ids), ncols=80)

                # POR PARTIDO (c/u identificado con un id)
                for cont_part, id in enumerate(l_ids, start=1):  #  Evito contador manual en for id in l_ids:

                    # Definicion de variables
                    progress_bar.update(1)
                    id = id[id.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")
                    d_nueva_fila = {'id': id, 'competicion': competicion, 'temporada': temp_year, 'pais': pais,'es_copa': 1 if categoria == "Copa" else 0}  # Reinicio diccionario en el que guardar la nueva fila
                    # print(f" Partido {cont_part} de {len(l_items)}. Recolectado el {cont_part / len(l_items) * 100:.0f}% ".center(120, "."))

                    # Ingreso a pagina de informacion del partido
                    crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

                    # EXTRACCION DE CAMPOS
                    fecha_str = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                    fecha_dt = datetime.datetime.strptime(fecha_str, "%d.%m.%Y %H:%M")

                    # Si el partido aun no se jugo (extraia partidos de la Copa de la Liga profesional 2023 la cual aun no se jugo pero ya esta el fixture... tampoco es tan grave solo esta la jornada 1)
                    if fecha_dt < fecha_act:

                        # Extraigo campos de hoja "Resumen"
                        d_nueva_fila['fecha'] = fecha_dt  # d_nueva_fila['fecha'] = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                        d_nueva_fila.update(crawler.extract_resumen(l_fields=['equipo_loc', 'equipo_vis', 'goles_loc', 'goles_vis', 'arbitro', 'cancha']))

                        # Extraigo campos de hoja "Estadisticas"
                        d_nueva_fila.update(crawler.extract_estadisticas())

                        # Extraigo campos de hoja "Formaciones"
                        d_nueva_fila.update(crawler.extract_formacion())

                        # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                        d_nueva_fila.update(crawler.extract_cuota())

                        # GUARDADO DE DATOS EN DATAFRAME
                        df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
                        # print(df_part.iloc[-1])

                # Cerrar la barra de progreso al finalizar
                progress_bar.close()

                # Guardo partidos de la temporada (por seguridad)
                df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/data_seg/{competicion_form}_{temp_year.replace("/", "_")}_{pais}.xlsx',index=False)

            # Guardo partidos de la competicion (por seguridad)
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/data_seg/{competicion_form}_{pais}.xlsx',index=False)

        # Guardado datos a nivel pais
        df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_partido.xlsx',index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

def extract_proximos_partidos_flashcore(l_paises, n_dias_max):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    fecha_act = datetime.datetime.now()
    crawler = FlashscoreCrawler(headless=True, path=None)  # Creo objeto de clase CrawlerActions()
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/df_competencias.xlsx')

    # POR PAIS
    for pais in l_paises:

        df = pd.DataFrame()  # Reinicio df por pais # No hace falta definir columnas por mas que no haya extraido partidos

        # Selecciono competencias del pais
        df_comp = df_comp[df_comp['pais'] == pais]
        print(f' PAIS: {pais} '.center(120, '#'))

        # POR COMPETICION
        for competicion, categoria in zip(df_comp['nombre'], df_comp['categoria']):

            # Obtengo datos de la competicion
            print(f' Competicion: {competicion} '.center(120, '+'))
            competicion_form = competicion.replace(" ", "-")  # formateo competicion para las rutas de archivo y urls

            # Ingreso a pagina
            url = f'https://www.flashscore.es/futbol/{pais}/{competicion_form}/resultados/'
            crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
            print(url)

            # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
            crawler.click_boton(xpath='.//button[@id="onetrust-accept-btn-handler"]')

            # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
            # crawler.click_boton(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=SEC_WAIT_LONG*3, repeat_click=True)  # Si hace click, es None. Si falla, es un str  # A veces me tira (y no entiendo por qué): selenium.common.exceptions.StaleElementReferenceException: Message: stale element reference: stale element not found

            # Obtengo temporada
            # No tiene sentido extraer todas las temporadas puesto que solo necesito la ultima...
            temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)

            # Extraigo partidos (items) y sus ids
            l_items = crawler.extract_tags(xpath='.//div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')
            l_ids = [item.get_attribute('id') for item in l_items]
            print(f"Cantidad de items (partidos): {len(l_items)}")

            # POR PARTIDO (c/u identificado con un id)
            cont_part = 0
            for id in l_ids:

                start = time.time()
                # Definicion de variables
                id = id[id.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")
                d_nueva_fila = {'id': id, 'competicion': competicion, 'temporada': temp_year, 'pais': pais, 'es_copa': 1 if categoria == "Copa" else 0}  # Reinicio diccionario en el que guardar la nueva fila
                cont_part += 1
                print(f" Partido {cont_part} de {len(l_items)}. Recolectado el {cont_part / len(l_items) * 100:.0f}% ".center(120, "."))

                # Ingreso a pagina de informacion del partido
                crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

                # EXTRACCION DE CAMPOS
                fecha_str = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                fecha_dt = datetime.datetime.strptime(fecha_str, "%d.%m.%Y %H:%M")
                print(fecha_dt, fecha_act)
                dif_fecha = (fecha_dt - fecha_act).days  # Ojo que si falta 1 dia y 23 hs, lo toma como 1...
                print(dif_fecha)

                # Si el partido aun no se jugo (extraia partidos de la Copa de la Liga profesional 2023 la cual aun no se jugo pero ya esta el fixture... tampoco es tan grave solo esta la jornada 1)
                if dif_fecha < n_dias_max:

                    # Extraigo campos de hoja "Resumen"
                    d_nueva_fila['fecha'] = fecha_dt
                    d_nueva_fila.update(crawler.extract_resumen(l_fields=['equipo_loc', 'equipo_vis', 'arbitro', 'cancha']))

                    # No tiene sentido extraer estadisticas puesto que nunca abrá de un partido que aun no se jugo...

                    # Extraigo formacion
                    d_nueva_fila.update(crawler.extract_formacion())  # --> Extriago de algun diario? https://www.ole.com.ar/futbol-primera/estudiantes-vs-barracas-central-hora-tv-posibles-formaciones-liga-profesional-2023_0_QdU0rRnj59.html

                    # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                    d_nueva_fila.update(crawler.extract_cuota())

                    # GUARDADO DE DATOS EN DATAFRAME
                    df = pd.concat([df, pd.DataFrame(d_nueva_fila, index=[0])])
                    print(df.iloc[-1])

                    end = time.time()
                    print(f"Partido recolectado en {(end - start):.1f} segundos")

        # Guardado datos (a nivel pais)
        df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_next_partidos.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

def reextract_campos_especificos_flashscore(pais, l_var):  # Re hacer para poder elegir que field recolectar nuevamente...

    # Solucionar el tema de que cuando falla un campo, tengo que volver a extraer tod@... Dar la posibildiad de recorrer los ids ya extraidos y extraer de nuevo el campo que falló

    # DEFINCION DE PARAMETROS & VARIABLES
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    crawler = Crawler(headless=True, path=None) # Creo objeto de clase CrawlerActions()
    l_var = ['odds_loc', 'odds_emp', 'odds_vis']

    # Levanto dataset de partidos ya extraidos
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_partido.xlsx')

    # Volver a extraer los campos para los ids seleccionados
    for id in df_part['id']:   # Seleccionar solo los ids que tienen None en los campos que se quieren re-extraer?

        # Ingreso a pagina de informacion del partido
        crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

        d_nueva_fila = extract_cuota(crawler)

        # Reemplazo valores

    '''
    # Por partido
    for i in range(len(df_part)):

        print(f" Partido Nº{i + 1} ".center(120, "#"))

        # Por variable
        for var in l_var:

            # Convierto str a float
            try:
                float(df_part.loc[i, var])

            # Si falla la conversion a float (es un str)
            except ValueError:

                # Vuelvo a buscar la cuota
                id = df_part.loc[i, 'id']

                # Reinicio diccionario en el que guardar la nueva fila
                d_nueva_fila = {'odds_loc': None, 'odds_emp': None, 'odds_vis': None}

                # Ingreso a pagina de informacion del partido
                crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

                # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                if crawler.extract_tag(xpath='.//div[@class="oddsRow"]') is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...

                    d_nueva_fila['odds_loc'] = extract_cuota(crawler, SEC_WAIT, i=1)
                    d_nueva_fila['odds_emp'] = extract_cuota(crawler, SEC_WAIT, i=2)
                    d_nueva_fila['odds_vis'] = extract_cuota(crawler, SEC_WAIT, i=3)

                print(f" Partido Nº{i + 1} ".center(120, "#"))
                print(f'{id} \nOdds_loc: {df_part.loc[i, "odds_loc"]} -> {d_nueva_fila["odds_loc"]} '
                      f'\nOdds emp: {df_part.loc[i, "odds_emp"]} -> {d_nueva_fila["odds_emp"]}'
                      f'\nOdds vis: {df_part.loc[i, "odds_vis"]} -> {d_nueva_fila["odds_vis"]}')

                # GUARDADO DE DATOS EN DATAFRAME
                df_part.loc[i, 'odds_loc'] = d_nueva_fila['odds_loc']
                df_part.loc[i, 'odds_emp'] = d_nueva_fila['odds_emp']
                df_part.loc[i, 'odds_vis'] = d_nueva_fila['odds_vis']
                break

    # Guardado de archivo excel en computadora
    df_part['odds_loc'] = df_part['odds_loc'].apply(lambda x: float(x))
    df_part['odds_emp'] = df_part['odds_emp'].apply(lambda x: float(x))
    df_part['odds_vis'] = df_part['odds_vis'].apply(lambda x: float(x))

    df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_partido_argentina_prueba.xlsx', index=False)
    '''
    df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_partido_argentina_prueba.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

class FlashscoreCrawler(Crawler):
    # Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper normal, scraper de fields especificos poro falla y scraper de proximos partidos...

    def __init__(self, headless, path):
        super().__init__(headless, path)
        self.child_driver = self.driver
        # Definir SEC_WAIT, SEC_WAIT_LONG como atrib..

    def extract_resumen(self, l_fields):

        SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
        d_nueva_fila = {}
        fields_to_extract = {
            'fecha': './/div[@class="duelParticipant__startTime"]',
            'equipo_loc': './/div[starts-with(@class, "duelParticipant__home")]',
            'equipo_vis': './/div[starts-with(@class, "duelParticipant__away")]',
            'goles_loc': './/div[@class="detailScore__wrapper"]/span[1]',
            'goles_vis': './/div[@class="detailScore__wrapper"]/span[3]',
            'arbitro': './/div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span',
            'cancha': './/div[@class="mi__data"]//span[contains(text(), "Estadio")]/following-sibling::span'
        }

        for field in l_fields:
            if field in fields_to_extract:
                value = super().extract_tag(xpath=fields_to_extract[field], text=True, sec_wait=SEC_WAIT)
                d_nueva_fila[field] = value

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
        SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
        d_nueva_fila = {}
        d_formaciones = {'Formaciones iniciales': 'tit', 'Suplentes': 'sup', 'Jugadores reemplazados': 'sup_ing',
                         'Jugadores ausentes': 'aus'}

        # Hago click en hoja "Formaciones"
        if super().click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Formaciones"]', sec_wait=SEC_WAIT) is not False:

            time.sleep(random.uniform(SEC_WAIT + 3, SEC_WAIT_LONG + 3))  # Por posible falla en el primer campo a extraer  # WebDriverWait(crawler.driver, SEC_WAIT_LONG + 3).until(EC.presence_of_element_located((By.XPATH, './/div[@class="preMatchTabCnt preMatchTabCnt1"]')))

            # Por formacion ("Formacion inicial", "Suplentes" y  "Ausentes")
            for formacion, titularidad in d_formaciones.items():

                # Si exista dicha formacion
                if super().extract_tag(xpath=f'.//div[text()="{formacion}"]', sec_wait=SEC_WAIT_LONG):

                    # Extraigo listado de jugadores
                    l_tags_jug_loc = super().extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]',sec_wait=SEC_WAIT * 2)  # Es lista de tags o None
                    l_tags_jug_vis = super().extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]',sec_wait=SEC_WAIT * 2)  # Es lista de tags o None

                    if l_tags_jug_loc:
                        l_nombres_loc = [self.extract_nombre(tag) for tag in l_tags_jug_loc]
                        for i, nombre in enumerate(l_nombres_loc):
                            d_nueva_fila[f'jug_{titularidad}_loc_{i + 1}'] = nombre

                    if l_tags_jug_vis:
                        l_nombres_vis = [self.extract_nombre(tag) for tag in l_tags_jug_vis]
                        for i, nombre in enumerate(l_nombres_vis):
                            d_nueva_fila[f'jug_{titularidad}_vis_{i + 1}'] = nombre

            # Extraigo entrenadores
            dt_loc_tag = super().extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]//a[@class="lf__participantName"]', sec_wait=SEC_WAIT)
            dt_vis_tag = super().extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]//a[@class="lf__participantName"]', sec_wait=SEC_WAIT)

            if dt_loc_tag:
                d_nueva_fila['dt_loc'] = self.extract_nombre(dt_loc_tag)

            if dt_vis_tag:
                d_nueva_fila['dt_vis'] = self.extract_nombre(dt_vis_tag)

        return d_nueva_fila

    def extract_estadisticas(self):

        SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
        d_nueva_fila = {}
        d_estadisticas = {'posesion': 'Posesión de balón', 'remates': 'Remates', 'remates_a_puerta': 'Remates a puerta',
                          'tarjetas_amarillas': 'Tarjetas amarillas', 'faltas': 'Faltas', 'pases': 'Pases totales',
                          'pases_comp': 'Pases completados', 'offsides': 'Fueras de juego', 'ataques': 'Ataques',
                          'ataques_pelig': 'Ataques peligrosos'}

        if super().click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Estadísticas"]', sec_wait=SEC_WAIT_LONG) is not False:

            time.sleep(random.uniform(SEC_WAIT + 3,SEC_WAIT_LONG + 3))  # Falla el campo posesion_loc puesto que es el primero en ser extraido y aun no cargo...

            # Por estadistica (posesion, remates, etc)
            for estadistica in d_estadisticas.keys():

                # Extraigo dicha estadistica tanto para el equipo local como para el visitante
                d_nueva_fila[f'{estadistica}_loc'] = super().extract_tag(xpath=f'.//div[text()="{d_estadisticas[estadistica]}"]//preceding-sibling::div', text=True,sec_wait=SEC_WAIT)  # Falla el campo posesion_loc puesto que es el primero en ser extraido y aun no cargo...
                d_nueva_fila[f'{estadistica}_vis'] = super().extract_tag(xpath=f'.//div[text()="{d_estadisticas[estadistica]}"]//following-sibling::div', text=True,sec_wait=SEC_WAIT)

        return d_nueva_fila

    def extract_cuota(self):  # Puedo volver a la anterior, solo fallaron 41 cuotas por "Cuotas retiradas por la casa de apuestas."

        # Definicion de variables
        SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
        d_nueva_fila = {}
        l_odds = ['odds_loc', 'odds_emp', 'odds_vis']

        if super().extract_tag(xpath='.//div[@class="oddsRow"]') is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...

            # Por cuota (local, emp y vis)
            for i, columna in enumerate(l_odds, start=1):

                # Busco odd suponiendo que cambio durante el partido
                try:
                    odds_str = super().extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]', attribute='title', sec_wait=SEC_WAIT)  # 3.00 » 2.25
                    cuota = float(odds_str.split('»')[0].strip())  # 3.00

                # Si la odd no cambio durante el partido, o bien, aparece "Cuotas retiradas por la casa de apuestas."
                except (ValueError, AttributeError):  # could not convert string to float:  # AttributeError: 'NoneType' object has no attribute 'split'
                    try:
                        odds_str = super().extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]//span[@class="oddsValueInner"]', text=True,sec_wait=SEC_WAIT)
                        cuota = float(odds_str)

                    except (ValueError, TypeError):
                        print("Fallo extraccion de la cuota")
                        cuota = None

                # Guardo cuota
                d_nueva_fila[columna] = cuota
        return d_nueva_fila

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Selecciono pais a extraer y obtengo las competencias y su categoria
    # pais = "argentina"  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    l_paises = ['argentina']

    # Extraigo partidos
    extract_partidos_flashscore(l_paises)

    # Extraigo jugadores
    # extract_jugadores_sofifa(l_paises)
