# Importo librerias
import pandas as pd
from dspy.data_understanding.collect_data.web_scraping.selenium import Crawler
from time import sleep
import random


def main():
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    SLEEP_MIN, SLEEP_MAX = 1, 3  # Tiempos de espera luego de clicks para humanizar programa
    N_TEMPS_MAX = 10
    crawler = Crawler(headless=True, path=None) # Creo objeto de clase CrawlerActions()
    df = pd.DataFrame(columns=['id', 'fecha','equipo_loc', 'equipo_vis', 'arbitro', 'cancha', 'dt_loc', 'dt_vis',
                               'goles_loc', 'goles_vis', 'posesion_loc', 'posesion_vis', 'remates_loc', 'remates_vis',
                               'remates_a_puerta_loc', 'remates_a_puerta_vis', 'tarjetas_amarillas_loc',
                               'tarjetas_amarillas_vis', 'faltas_loc', 'faltas_vis', 'pases_loc', 'pases_vis',
                               'pases_comp_loc', 'pases_comp_vis', 'offsides_loc', 'offsides_vis', 'ataques_loc',
                               'ataques_vis', 'ataques_pelig_loc', 'ataques_pelig_vis', 'l_jug_lesionados_loc',
                               'l_jug_lesionados_vis'])

    # Ingreso a pagina
    crawler.driver.get('https://www.flashscore.es/futbol/argentina/liga-profesional/archivo/')  # hasta que no se carga toda la pagina, no sigue...
    sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # Accept cookies
    crawler.click_boton(xpath='.//button[@id="onetrust-accept-btn-handler"]')

    # Extraigo links de temporadas (años)
    l_tag_temporadas = crawler.extract_tags(xpath='.//section[@id="tournament-page-archiv"]//div[@class="archive__row"]/div[@class="archive__season"]/a')
    l_urls_temporadas = [tag.get_attribute('href') for tag in l_tag_temporadas]

    # POR PAGINA (TEMPORADA) DE PAGINACION
    for url_temp in l_urls_temporadas[:N_TEMPS_MAX]:

        # Ingreso a pagina de temporada
        crawler.driver.get(url_temp)

        # Obtengo nombre de temporada y lo imprimo
        nombre_temp = crawler.extract_tag(xpath='.//div[@class="heading__title"]', text=True)
        print(f" Temporada: {nombre_temp} ".center(120, "#"))

        # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
        while True:
            try:
                crawler.click_boton(xpath='.//a[text()="Mostrar más partidos"]')
            except :
                break

        # Extraigo items (partidos)
        l_items = crawler.extract_tags(xpath='.//div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')
        print(f"Cantidad de items (partidos): {len(l_items)}")

        # Obtengo el id de cada item para poder extraer los campos luego
        l_ids = [item.get_attribute('id') for item in l_items]

        # EXTRACCION DE CAMPOS
        # Por item (c/u identificado con un id)
        for id in l_ids:

            # Quito lo que no es del id
            id = id[id.rfind('_') + 1:]  # paso de "g_1_fshvzbls" a "fshvzbls"

            # Ingreso a pagina de informacion del partido
            url = 'https://www.flashscore.es/partido/{}/#/resumen-del-partido/resumen-del-partido'.format(id)
            crawler.driver.get(url)

            # Extraigo campos de hoja "Resumen"
            fecha = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True)
            equipo1 = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__home")]', text=True)
            equipo2 = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__away")]', text=True)
            goles_loc = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[1]', text=True)
            goles_vis = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[3]', text=True)
            arbitro = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span', text=True)
            cancha = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Estadio")]/following-sibling::span', text=True)

            # Extraigo campos de hoja "Estadísticas"
            crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Estadísticas"]')
            posesion_loc = crawler.extract_tag(xpath='.//div[text()="Posesión de balón"]//preceding-sibling::div',text=True)
            posesion_vis = crawler.extract_tag(xpath='.//div[text()="Posesión de balón"]//following-sibling::div',text=True, sec_wait=0.5)
            remates_loc = crawler.extract_tag(xpath='.//div[text()="Remates"]//preceding-sibling::div',text=True, sec_wait=0.5)
            remates_vis = crawler.extract_tag(xpath='.//div[text()="Remates"]//following-sibling::div',text=True, sec_wait=0.5)
            remates_a_puerta_loc = crawler.extract_tag(xpath='.//div[text()="Remates a puerta"]//preceding-sibling::div',text=True, sec_wait=0.5)
            remates_a_puerta_vis = crawler.extract_tag(xpath='.//div[text()="Remates a puerta"]//following-sibling::div',text=True, sec_wait=0.5)
            tarjetas_amarillas_loc = crawler.extract_tag(xpath='.//div[text()="Tarjetas amarillas"]//preceding-sibling::div',text=True, sec_wait=0.5)
            tarjetas_amarillas_vis = crawler.extract_tag(xpath='.//div[text()="Tarjetas amarillas"]//following-sibling::div',text=True, sec_wait=0.5)
            faltas_loc = crawler.extract_tag(xpath='.//div[text()="Faltas"]//preceding-sibling::div',text=True, sec_wait=0.5)
            faltas_vis = crawler.extract_tag(xpath='.//div[text()="Faltas"]//following-sibling::div',text=True, sec_wait=0.5)
            pases_loc = crawler.extract_tag(xpath='.//div[text()="Pases totales"]//preceding-sibling::div',text=True, sec_wait=0.5)
            pases_vis = crawler.extract_tag(xpath='.//div[text()="Pases totales"]//following-sibling::div',text=True, sec_wait=0.5)
            pases_comp_loc = crawler.extract_tag(xpath='.//div[text()="Pases completados"]//preceding-sibling::div',text=True, sec_wait=0.5)
            pases_comp_vis = crawler.extract_tag(xpath='.//div[text()="Pases completados"]//following-sibling::div',text=True, sec_wait=0.5)
            offsides_loc = crawler.extract_tag(xpath='.//div[text()="Fueras de juego"]//preceding-sibling::div', text=True, sec_wait=0.5)
            offsides_vis = crawler.extract_tag(xpath='.//div[text()="Fueras de juego"]//following-sibling::div', text=True, sec_wait=0.5)
            ataques_loc = crawler.extract_tag(xpath='.//div[text()="Ataques"]//preceding-sibling::div', text=True, sec_wait=0.5)
            ataques_vis = crawler.extract_tag(xpath='.//div[text()="Ataques"]//following-sibling::div', text=True, sec_wait=0.5)
            ataques_pelig_loc = crawler.extract_tag(xpath='.//div[text()="Ataques peligrosos"]//preceding-sibling::div', text=True, sec_wait=0.5)
            ataques_pelig_vis = crawler.extract_tag(xpath='.//div[text()="Ataques peligrosos"]//following-sibling::div', text=True, sec_wait=0.5)

            # Extraigo campos de hoja "Formaciones"
            crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Formaciones"]')
            l_tags_jug_lesionados_loc = crawler.extract_tags(xpath='.//div[text()="Jugadores ausentes"]//following-sibling::div//div[@class="lf__side"][1]//a', sec_wait=3)
            l_jug_lesionados_loc = [tag.text for tag in l_tags_jug_lesionados_loc] if l_tags_jug_lesionados_loc is not None else []
            l_tags_jug_lesionados_vis = crawler.extract_tags(xpath='.//div[text()="Jugadores ausentes"]//following-sibling::div//div[@class="lf__side"][2]//a', sec_wait=0.5)
            l_jug_lesionados_vis = [tag.text for tag in l_tags_jug_lesionados_vis] if l_tags_jug_lesionados_vis is not None else []
            dt_loc = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]', text=True, sec_wait=0.5)
            dt_vis = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]', text=True, sec_wait=0.5)

            # GUARDADO DE DATOS EN DATAFRAME
            l_data = [id, fecha, equipo1, equipo2, arbitro, cancha, dt_loc, dt_vis, goles_loc, goles_vis, posesion_loc,
                      posesion_vis, remates_loc, remates_vis, remates_a_puerta_loc, remates_a_puerta_vis,
                      tarjetas_amarillas_loc, tarjetas_amarillas_vis, faltas_loc, faltas_vis, pases_loc, pases_vis,
                      pases_comp_loc, pases_comp_vis, offsides_loc, offsides_vis, ataques_loc, ataques_vis,
                      ataques_pelig_loc, ataques_pelig_vis, l_jug_lesionados_loc, l_jug_lesionados_vis]
            df.loc[len(df)] = l_data
            print(l_data)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

    # Guardado de archivo excel en computadora
    df.to_excel('./liga_argentina_historico.xlsx', index=False)  # Cambiar la ruta del archivo

main()