# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
import time
import random

import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
import time

def extract_cuota(crawler, SEC_WAIT_LONG, i):

    # Busco odd suponiendo que cambio durante el partido
    odds = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]', attribute='title', sec_wait=SEC_WAIT_LONG + 1)

    # Si la odd cambio (el title no es un string vacio)
    if len(odds) > 1:
        # Extraigo la odd antes de iniciar el partido
        return odds.split('»')[0].strip()

    # Si la odd no cambio (el title es un string vacio)
    else:
        # Busco el texto de la odd inicial
        return crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]//span[@class="oddsValueInner"]', text=True, sec_wait=SEC_WAIT_LONG + 1)


def extract_flashscore():
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    crawler = Crawler(headless=True, path=None) # Creo objeto de clase CrawlerActions()
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_competicion.xlsx')
    df_part = pd.DataFrame( # Debo agregar una columna por cada jugador en vez de las listas....
        columns=['id', 'competicion', 'temporada', 'pais', 'es_copa', 'fecha', 'equipo_loc', 'equipo_vis', 'arbitro',
                 'cancha', 'dt_loc', 'dt_vis', 'goles_loc', 'goles_vis', 'posesion_loc', 'posesion_vis', 'remates_loc',
                 'remates_vis', 'remates_a_puerta_loc', 'remates_a_puerta_vis', 'tarjetas_amarillas_loc',
                 'tarjetas_amarillas_vis', 'faltas_loc', 'faltas_vis', 'pases_loc', 'pases_vis', 'pases_comp_loc',
                 'pases_comp_vis', 'offsides_loc', 'offsides_vis', 'ataques_loc', 'ataques_vis', 'ataques_pelig_loc',
                 'ataques_pelig_vis', 'l_jug_ausentes_loc', 'l_jug_ausentes_vis', 'l_jug_tit_loc', 'l_jug_tit_vis',
                 'l_jug_sup_loc', 'l_jug_sup_vis', 'odds_loc', 'odds_emp', 'odds_vis'])

    # POR PAIS
    for pais in df_comp['pais'].unique():

        # Filtro competiciones por pais
        df_pais = df_comp[df_comp['pais'] == pais]
        print(f' PAIS: {pais} '.center(120, '#'))
        pais = pais.lower()

        # Obtengo datos ya extraidos del pais  (TENER EN CUENTA SI FALLA...)
        try:
            df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_partido_{pais}.xlsx')
        except FileNotFoundError:
            df_part = pd.DataFrame(columns=['id'])  # inicializo con id para evitar error en if id not in df_part

        # POR COMPETICION
        for competicion, categoria in zip(df_pais['nombre'], df_pais['categoria']):

            # Obtengo datos de la competicion
            print(f' Competicion: {competicion} '.center(120, '+'))
            competicion_form = competicion.lower().replace(" ", "-")  # formateo competicion para las rutas de archivo y urls

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

            # POR PAGINA (TEMPORADA) DE PAGINACION
            for url_temp in l_urls_temporadas:

                # Ingreso a pagina de temporada e imprimo año de la temporada
                crawler.driver.get(url_temp)
                temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
                print(f" {temp_year} ".center(120, "-"))

                # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
                crawler.click_boton(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=5, repeat_click=True)  # Si hace click, es None. Si falla, es un str

                # Extraigo partidos (items) y sus ids
                l_items = crawler.extract_tags(xpath='.//div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')
                l_ids = [item.get_attribute('id') for item in l_items]
                print(f"Cantidad de items (partidos): {len(l_items)}")

                # POR PARTIDO (c/u identificado con un id)
                cont_part = 0
                for id in l_ids:

                    start = time.time()

                    # Quito lo que no es del id
                    id = id[id.rfind('_') + 1:]  # paso de "g_1_fshvzbls" a "fshvzbls"

                    # Reinicio diccionario en el que guardar la nueva fila
                    d_nueva_fila = {'id': id, 'competicion': competicion, 'temporada': temp_year, 'pais': pais, 'es_copa': True if categoria == "Copa" else False}
                    cont_part += 1
                    print(f" Partido {cont_part} de {len(l_items)}. Recolectado el {cont_part / len(l_items) * 100:.0f}% ".center(120, "."))

                    # Si aun no extraje dicho id
                    if id not in list(df_part['id']):

                        # Ingreso a pagina de informacion del partido
                        crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')  # Cambie porque fallo en un partido de 2005 de Peru. Saque el ultimo /resumen-del-partido... selenium.common.exceptions.WebDriverException: Message: unknown error: net::ERR_NAME_NOT_RESOLVED f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido/resumen-del-partido'

                        # EXTRACCION DE CAMPOS
                        # Extraigo campos de hoja "Resumen"
                        d_nueva_fila['fecha'] = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                        d_nueva_fila['equipo_loc'] = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__home")]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['equipo_vis'] = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__away")]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['goles_loc'] = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[1]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['goles_vis'] = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[3]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['arbitro'] = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['cancha'] = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Estadio")]/following-sibling::span', text=True, sec_wait=SEC_WAIT)

                        # Extraigo campos de hoja "Estadísticas"
                        if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Estadísticas"]', sec_wait=SEC_WAIT_LONG) is not False:
                            d_nueva_fila['posesion_loc'] = crawler.extract_tag(xpath='.//div[text()="Posesión de balón"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT_LONG)
                            d_nueva_fila['posesion_vis'] = crawler.extract_tag(xpath='.//div[text()="Posesión de balón"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['remates_loc'] = crawler.extract_tag(xpath='.//div[text()="Remates"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['remates_vis'] = crawler.extract_tag(xpath='.//div[text()="Remates"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['remates_a_puerta_loc'] = crawler.extract_tag(xpath='.//div[text()="Remates a puerta"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['remates_a_puerta_vis'] = crawler.extract_tag(xpath='.//div[text()="Remates a puerta"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['tarjetas_amarillas_loc'] = crawler.extract_tag(xpath='.//div[text()="Tarjetas amarillas"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['tarjetas_amarillas_vis'] = crawler.extract_tag(xpath='.//div[text()="Tarjetas amarillas"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['faltas_loc'] = crawler.extract_tag(xpath='.//div[text()="Faltas"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['faltas_vis'] = crawler.extract_tag(xpath='.//div[text()="Faltas"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['pases_loc'] = crawler.extract_tag(xpath='.//div[text()="Pases totales"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['pases_vis'] = crawler.extract_tag(xpath='.//div[text()="Pases totales"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['pases_comp_loc'] = crawler.extract_tag(xpath='.//div[text()="Pases completados"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['pases_comp_vis'] = crawler.extract_tag(xpath='.//div[text()="Pases completados"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['offsides_loc'] = crawler.extract_tag(xpath='.//div[text()="Fueras de juego"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['offsides_vis'] = crawler.extract_tag(xpath='.//div[text()="Fueras de juego"]//following-sibling::div', text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['ataques_loc'] = crawler.extract_tag(xpath='.//div[text()="Ataques"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['ataques_vis'] = crawler.extract_tag(xpath='.//div[text()="Ataques"]//following-sibling::div', text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['ataques_pelig_loc'] = crawler.extract_tag(xpath='.//div[text()="Ataques peligrosos"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['ataques_pelig_vis'] = crawler.extract_tag(xpath='.//div[text()="Ataques peligrosos"]//following-sibling::div', text=True, sec_wait=SEC_WAIT)

                        # Extraigo campos de hoja "Formaciones"
                        if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Formaciones"]', sec_wait=SEC_WAIT) is not False:

                            # Si existe la seccion de Titulares
                            if crawler.extract_tag(xpath='.//div[text()="Formaciones iniciales"]', sec_wait=SEC_WAIT_LONG) is not None:
                                # NO DEBO EXTRAER EL TAG A, HAY VECES QUE FALLA PORQUE LOS JUGADORES NO TIENEN ASOCIADO UN LINK
                                l_tags_jug_tit_loc = crawler.extract_tags(xpath='.//div[text()="Formaciones iniciales"]//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT)  # Es lista de tags o None
                                l_tags_jug_tit_vis = crawler.extract_tags(xpath='.//div[text()="Formaciones iniciales"]//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT)  # Es lista de tags o None

                                # Creo una columna por cada jugador titular del equipo local
                                if l_tags_jug_tit_loc is not None:
                                    for i in range(len(l_tags_jug_tit_loc)):
                                        d_nueva_fila[f'jug_tit_loc_{i}'] = l_tags_jug_tit_loc[i].text

                                # Creo una columna por cada jugador titular del equipo visitante
                                if l_tags_jug_tit_vis is not None:
                                    for i in range(len(l_tags_jug_tit_vis)):
                                        d_nueva_fila[f'jug_tit_vis_{i}'] = l_tags_jug_tit_vis[i].text

                            # Si existe la seccion de Suplentes
                            if crawler.extract_tag(xpath='.//div[text()="Suplentes"]', sec_wait=SEC_WAIT) is not None:
                                # NO DEBO EXTRAER EL TAG A, HAY VECES QUE FALLA PORQUE LOS JUGADORES NO TIENEN ASOCIADO UN LINK
                                l_tags_jug_sup_loc = crawler.extract_tags(xpath='.//div[text()="Suplentes"]//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT)  # Es lista de tags o None
                                l_tags_jug_sup_vis = crawler.extract_tags(xpath='.//div[text()="Suplentes"]//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT)  # Es lista de tags o None

                                # Creo una columna por cada jugador suplente del equipo local
                                if l_tags_jug_sup_loc is not None:
                                    for i in range(len(l_tags_jug_sup_loc)):
                                        d_nueva_fila[f'jug_sup_loc_{i}'] = l_tags_jug_sup_loc[i].text

                                # Creo una columna por cada jugador suplente del equipo visitante
                                if l_tags_jug_sup_vis is not None:
                                    for i in range(len(l_tags_jug_sup_vis)):
                                        d_nueva_fila[f'jug_sup_vis_{i}'] = l_tags_jug_sup_vis[i].text

                            # Si existe la seccion de jugadores ausentes
                            if crawler.extract_tag(xpath='.//div[text()="Jugadores ausentes"]', sec_wait=SEC_WAIT) is not None:
                                # NO DEBO EXTRAER EL TAG A, HAY VECES QUE FALLA PORQUE LOS JUGADORES NO TIENEN ASOCIADO UN LINK
                                l_tags_jug_ausentes_loc = crawler.extract_tags(xpath='.//div[text()="Jugadores ausentes"]//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT)  # Es lista de tags o None
                                l_tags_jug_ausentes_vis = crawler.extract_tags(xpath='.//div[text()="Jugadores ausentes"]//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT)

                                # Creo una columna por cada jugador ausente del equipo local
                                if l_tags_jug_ausentes_loc is not None:
                                    for i in range(len(l_tags_jug_ausentes_loc)):
                                        d_nueva_fila[f'jug_aus_loc_{i}'] = l_tags_jug_ausentes_loc[i].text

                                # Creo una columna por cada jugador ausente del equipo visitante
                                if l_tags_jug_ausentes_vis is not None:
                                    for i in range(len(l_tags_jug_ausentes_vis)):
                                        d_nueva_fila[f'jug_aus_vis_{i}'] = l_tags_jug_ausentes_vis[i].text

                            d_nueva_fila['dt_loc'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]', text=True, sec_wait=SEC_WAIT)
                            d_nueva_fila['dt_vis'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]', text=True, sec_wait=SEC_WAIT)

                        # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                        if crawler.extract_tag(xpath='.//div[@class="oddsRow"]') is not None:
                            d_nueva_fila['odds_loc'] = extract_cuota(crawler, SEC_WAIT_LONG, i=1)
                            d_nueva_fila['odds_emp'] = extract_cuota(crawler, SEC_WAIT_LONG, i=2)
                            d_nueva_fila['odds_vis'] = extract_cuota(crawler, SEC_WAIT_LONG, i=3)

                        # GUARDADO DE DATOS EN DATAFRAME
                        df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
                        print(df_part.shape)
                        print(df_part.iloc[-1])

                        end = time.time()
                        print(f"Partido recolectado en {(end - start):.1f} segundos")

                # Guardo partidos no extraidos de la temporada (por seguridad)
                df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/{competicion_form}_{temp_year.replace("/","_")}_{pais}.xlsx', index=False)

        # Guardado de archivo excel en computadora
        df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/entidad_partido_{pais}.xlsx',index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part

extract_flashscore()