# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
import time
import random


def main():
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    SLEEP_MIN, SLEEP_MAX = 1, 3  # Tiempos de espera luego de clicks para humanizar programa
    SEC_WAIT, SEC_WAIT_LONG = 0.1, 1.5
    crawler = Crawler(headless=True, path=None) # Creo objeto de clase CrawlerActions()
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/entidad_competicion.xlsx')
    df = pd.DataFrame(
        columns=['id', 'competicion', 'temporada', 'pais', 'es_copa', 'fecha', 'equipo_loc', 'equipo_vis',
                 'arbitro', 'cancha', 'dt_loc', 'dt_vis', 'goles_loc', 'goles_vis', 'posesion_loc', 'posesion_vis',
                 'remates_loc', 'remates_vis', 'remates_a_puerta_loc', 'remates_a_puerta_vis',
                 'tarjetas_amarillas_loc', 'tarjetas_amarillas_vis', 'faltas_loc', 'faltas_vis', 'pases_loc',
                 'pases_vis', 'pases_comp_loc', 'pases_comp_vis', 'offsides_loc', 'offsides_vis', 'ataques_loc',
                 'ataques_vis', 'ataques_pelig_loc', 'ataques_pelig_vis', 'l_jug_ausentes_loc',
                 'l_jug_ausentes_vis', 'l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc', 'l_jug_sup_vis',
                 'odds_loc', 'odds_emp', 'odds_vis'])

    # Por competicion
    for i in range(28, 29):  # len(df_comp)  # Uruguay: 28

        # Obtengo datos de la competicion
        competicion, pais = df_comp.loc[i, 'nombre'], df_comp.loc[i, 'pais']
        es_copa = True if df_comp.loc[i, 'categoria'] == "Copa" else False
        print(f' Competicion: {competicion} ({pais}) '.center(120, '+'))

        # Ingreso a pagina
        url = f'https://www.flashscore.es/futbol/{pais.lower()}/{competicion.lower().replace(" ", "-")}/archivo/'
        crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
        print(url)
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        # Accept cookies  OJO! Que a veces no llega a cargar... Igual creo que no afecta..
        crawler.click_boton(xpath='.//button[@id="onetrust-accept-btn-handler"]')

        # Extraigo links de temporadas (años)
        l_tag_temporadas = crawler.extract_tags(xpath='.//section[@id="tournament-page-archiv"]//div[@class="archive__row"]/div[@class="archive__season"]/a')
        l_urls_temporadas = [tag.get_attribute('href') for tag in l_tag_temporadas]
        print(f'Cantidad de temporadas: {len(l_urls_temporadas)}')

        # POR PAGINA (TEMPORADA) DE PAGINACION
        for url_temp in l_urls_temporadas[11:]:  #  Uruguay: 11

            # Ingreso a pagina de temporada
            crawler.driver.get(url_temp)

            # Obtengo nombre de temporada y lo imprimo
            temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
            print(f" {temp_year} ".center(120, "-"))

            # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
            n_clicks = 0
            while True:
                click = crawler.click_boton(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=SEC_WAIT_LONG*5)  # Si hace click, es None. Si falla, es un str

                if click is not False:
                    n_clicks += 1
                # Si falla la extraccion del boton o falla el click
                else:
                    print(f"Hizo {n_clicks} clicks")
                    break

            # Extraigo items (partidos)
            l_items = crawler.extract_tags(xpath='.//div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')
            print(f"Cantidad de items (partidos): {len(l_items)}")

            # Obtengo el id de cada item para poder extraer los campos luego
            l_ids = [item.get_attribute('id') for item in l_items]

            # POR ITEM (PARTIDO, c/u identificado con un id)
            cont_part = 0
            for id in l_ids:

                start = time.time()

                # Quito lo que no es del id
                id = id[id.rfind('_') + 1:]  # paso de "g_1_fshvzbls" a "fshvzbls"
                cont_part += 1
                print(f" Partido {cont_part} de {len(l_items)}. Recolectado el {cont_part / len(l_items) * 100:.0f}% ".center(120, "."))

                # Si aun no extraje dicho id
                # if id not in list(df_arg['id']):

                # Ingreso a pagina de informacion del partido
                crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')  # Cambie porque fallo en un partido de 2005 de Peru. Saque el ultimo /resumen-del-partido... selenium.common.exceptions.WebDriverException: Message: unknown error: net::ERR_NAME_NOT_RESOLVED f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido/resumen-del-partido'

                # EXTRACCION DE CAMPOS
                # Extraigo campos de hoja "Resumen"
                fecha = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                equipo1 = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__home")]', text=True, sec_wait=SEC_WAIT)
                equipo2 = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__away")]', text=True, sec_wait=SEC_WAIT)
                goles_loc = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[1]', text=True, sec_wait=SEC_WAIT)
                goles_vis = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[3]', text=True, sec_wait=SEC_WAIT)
                arbitro = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span', text=True, sec_wait=SEC_WAIT)
                cancha = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Estadio")]/following-sibling::span', text=True, sec_wait=SEC_WAIT)

                # Extraigo campos de hoja "Estadísticas"
                if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Estadísticas"]', sec_wait=SEC_WAIT) is not False:
                    posesion_loc = crawler.extract_tag(xpath='.//div[text()="Posesión de balón"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT_LONG)
                    posesion_vis = crawler.extract_tag(xpath='.//div[text()="Posesión de balón"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                    remates_loc = crawler.extract_tag(xpath='.//div[text()="Remates"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                    remates_vis = crawler.extract_tag(xpath='.//div[text()="Remates"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                    remates_a_puerta_loc = crawler.extract_tag(xpath='.//div[text()="Remates a puerta"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                    remates_a_puerta_vis = crawler.extract_tag(xpath='.//div[text()="Remates a puerta"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                    tarjetas_amarillas_loc = crawler.extract_tag(xpath='.//div[text()="Tarjetas amarillas"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                    tarjetas_amarillas_vis = crawler.extract_tag(xpath='.//div[text()="Tarjetas amarillas"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                    faltas_loc = crawler.extract_tag(xpath='.//div[text()="Faltas"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                    faltas_vis = crawler.extract_tag(xpath='.//div[text()="Faltas"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                    pases_loc = crawler.extract_tag(xpath='.//div[text()="Pases totales"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                    pases_vis = crawler.extract_tag(xpath='.//div[text()="Pases totales"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                    pases_comp_loc = crawler.extract_tag(xpath='.//div[text()="Pases completados"]//preceding-sibling::div',text=True, sec_wait=SEC_WAIT)
                    pases_comp_vis = crawler.extract_tag(xpath='.//div[text()="Pases completados"]//following-sibling::div',text=True, sec_wait=SEC_WAIT)
                    offsides_loc = crawler.extract_tag(xpath='.//div[text()="Fueras de juego"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)
                    offsides_vis = crawler.extract_tag(xpath='.//div[text()="Fueras de juego"]//following-sibling::div', text=True, sec_wait=SEC_WAIT)
                    ataques_loc = crawler.extract_tag(xpath='.//div[text()="Ataques"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)
                    ataques_vis = crawler.extract_tag(xpath='.//div[text()="Ataques"]//following-sibling::div', text=True, sec_wait=SEC_WAIT)
                    ataques_pelig_loc = crawler.extract_tag(xpath='.//div[text()="Ataques peligrosos"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)
                    ataques_pelig_vis = crawler.extract_tag(xpath='.//div[text()="Ataques peligrosos"]//following-sibling::div', text=True, sec_wait=SEC_WAIT)
                # Si no hay hoja "Estadisticas"
                else:
                    posesion_loc, posesion_vis = None, None
                    remates_loc, remates_vis = None, None
                    remates_a_puerta_loc, remates_a_puerta_vis = None, None
                    tarjetas_amarillas_loc, tarjetas_amarillas_vis = None, None
                    faltas_loc, faltas_vis = None, None
                    pases_loc, pases_vis = None, None
                    pases_comp_loc, pases_comp_vis = None, None
                    offsides_loc, offsides_vis = None, None
                    ataques_loc, ataques_vis = None, None
                    ataques_pelig_loc, ataques_pelig_vis = None, None

                # Extraigo campos de hoja "Formaciones"
                if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Formaciones"]', sec_wait=SEC_WAIT) is not False:

                    # Si existe la seccion de Titulares
                    if crawler.extract_tag(xpath='.//div[text()="Formaciones iniciales"]', sec_wait=SEC_WAIT_LONG) is not None:
                        l_tags_jug_tit_loc = crawler.extract_tags(xpath='.//div[text()="Formaciones iniciales"]//following-sibling::div//div[@class="lf__side"][1]//a', sec_wait=SEC_WAIT)  # Es lista de tags o None
                        l_jug_tit_loc = [tag.text for tag in l_tags_jug_tit_loc] if l_tags_jug_tit_loc is not None else []
                        l_tags_jug_tit_vis = crawler.extract_tags(xpath='.//div[text()="Formaciones iniciales"]//following-sibling::div//div[@class="lf__side"][2]//a', sec_wait=SEC_WAIT)  # Es lista de tags o None
                        l_jug_tit_vis = [tag.text for tag in l_tags_jug_tit_vis] if l_tags_jug_tit_vis is not None else []
                    # Si no existe la seccion de Formacion inicial
                    else:
                        l_jug_tit_loc, l_jug_tit_vis = None, None

                    # Si existe la seccion de Suplentes
                    if crawler.extract_tag(xpath='.//div[text()="Suplentes"]', sec_wait=SEC_WAIT) is not None:
                        l_tags_jug_sup_loc = crawler.extract_tags(xpath='.//div[text()="Suplentes"]//following-sibling::div//div[@class="lf__side"][1]//a', sec_wait=SEC_WAIT)  # Es lista de tags o None
                        l_jug_sup_loc = [tag.text for tag in l_tags_jug_sup_loc] if l_tags_jug_sup_loc is not None else []
                        l_tags_jug_sup_vis = crawler.extract_tags(xpath='.//div[text()="Suplentes"]//following-sibling::div//div[@class="lf__side"][2]//a', sec_wait=SEC_WAIT)  # Es lista de tags o None
                        l_jug_sup_vis = [tag.text for tag in l_tags_jug_sup_vis] if l_tags_jug_sup_vis is not None else []
                    # Si no existe la seccion de Suplentes
                    else:
                        l_jug_sup_loc, l_jug_sup_vis = None, None

                    # Si existe la seccion de jugadores ausentes
                    if crawler.extract_tag(xpath='.//div[text()="Jugadores ausentes"]', sec_wait=SEC_WAIT) is not None:
                        l_tags_jug_ausentes_loc = crawler.extract_tags(xpath='.//div[text()="Jugadores ausentes"]//following-sibling::div//div[@class="lf__side"][1]//a', sec_wait=SEC_WAIT)  # Es lista de tags o None
                        l_jug_ausentes_loc = [tag.text for tag in l_tags_jug_ausentes_loc] if l_tags_jug_ausentes_loc is not None else []
                        l_tags_jug_ausentes_vis = crawler.extract_tags(xpath='.//div[text()="Jugadores ausentes"]//following-sibling::div//div[@class="lf__side"][2]//a', sec_wait=SEC_WAIT)
                        l_jug_ausentes_vis = [tag.text for tag in l_tags_jug_ausentes_vis] if l_tags_jug_ausentes_vis is not None else []
                    # Si no existe la seccion de jugadores lesionados
                    else:
                        l_jug_ausentes_loc, l_jug_ausentes_vis = None, None

                    dt_loc = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]', text=True, sec_wait=SEC_WAIT)
                    dt_vis = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]', text=True, sec_wait=SEC_WAIT)

                # Si no hay hoja "Formaciones"
                else:
                    l_jug_tit_loc, l_jug_tit_vis = None, None
                    l_jug_sup_loc, l_jug_sup_vis = None, None
                    l_jug_ausentes_loc, l_jug_ausentes_vis = None, None
                    dt_loc, dt_vis = None, None

                # Extrae probabilidades de sitio de apuestas
                if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail"]//a[text()="Cuotas"]', sec_wait=SEC_WAIT) is not False:
                    odds_loc = crawler.extract_tag(xpath='.//div[@class="ui-table__row"]/a[1]', text=True, sec_wait=SEC_WAIT_LONG+1)
                    odds_emp = crawler.extract_tag(xpath='.//div[@class="ui-table__row"]/a[2]', text=True, sec_wait=SEC_WAIT)
                    odds_vis = crawler.extract_tag(xpath='.//div[@class="ui-table__row"]/a[3]', text=True, sec_wait=SEC_WAIT)

                # Si no existe la hoja "Cuotas" pero si la seccion "Cuotas pre-partido"
                elif crawler.extract_tag(xpath='.//div[@class="oddsRow"]'):
                    odds_loc = crawler.extract_tag(xpath='.//div[@class="cellWrapper"][1]', attribute='title', sec_wait=SEC_WAIT_LONG+1)
                    odds_emp = crawler.extract_tag(xpath='.//div[@class="cellWrapper"][2]', attribute='title', sec_wait=SEC_WAIT)
                    odds_vis = crawler.extract_tag(xpath='.//div[@class="cellWrapper"][3]', attribute='title', sec_wait=SEC_WAIT)

                # Si no hay hoja "Cuotas" ni seccion "Cuotas prepartido"
                else:
                    odds_loc, odds_emp, odds_vis = None, None, None

                # GUARDADO DE DATOS EN DATAFRAME
                l_data = [id, competicion, temp_year, pais, es_copa, fecha, equipo1, equipo2, arbitro, cancha, dt_loc, dt_vis, goles_loc, goles_vis, posesion_loc, posesion_vis, remates_loc, remates_vis, remates_a_puerta_loc, remates_a_puerta_vis, tarjetas_amarillas_loc, tarjetas_amarillas_vis, faltas_loc, faltas_vis, pases_loc, pases_vis, pases_comp_loc, pases_comp_vis, offsides_loc, offsides_vis, ataques_loc, ataques_vis, ataques_pelig_loc, ataques_pelig_vis, l_jug_ausentes_loc, l_jug_ausentes_vis, l_jug_tit_loc, l_jug_tit_vis, l_jug_sup_loc, l_jug_sup_vis, odds_loc, odds_emp, odds_vis]
                df.loc[len(df)] = l_data
                print(df.iloc[-1])

                end = time.time()
                print(f"Partido recolectado en {(end - start):.1f} segundos")

            # Guardado de archivo excel en computadora
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/{competicion.lower().replace(" ","_")}_{temp_year.replace("/","_")}_{pais.lower()}.xlsx', index=False)  # Cambiar la ruta del archivo

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

main()