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
    fecha_act = datetime.datetime.now()
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    crawler = FlashscoreCrawler(headless=False, path=None)
    df = pd.DataFrame()  # Reinicio df por pais # No hace falta definir columnas por mas que no haya extraido partidos
    df_comp = pd.read_excel('/Users/nachomondino/Desktop/df_competencias_who_scored.xlsx')
    print(df_comp)
    print(f' PAIS: {pais} '.center(120, '#'))

    # Selecciono competencias del pais
    df_comp = df_comp[df_comp['pais'] == pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]

    # POR COMPETICION
    for cod_pais, cod_comp, competicion, tipo_comp in zip(df_comp['cod_pais'], df_comp['cod_competicion'], df_comp['competicion'], df_comp['tipo_comp']):

        # Ingreso a pagina
        competicion_form = competicion.replace(" ", "-")  # formateo competicion para las rutas de archivo y urls
        url = f'https://www.whoscored.com/Regions/{cod_pais}/Tournaments/{cod_comp}/{pais}-{competicion_form}'
        crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
        print(f' Competicion: {competicion} '.center(120, '+'))
        print(url)
        time.sleep(random.uniform(4,10))

        # Extraigo temporada inicial
        actual_temp = crawler.extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="seasons"]/option[@selected="selected"]', text=True)

        # POR TEMPORADA
        while True:

            print(f' Temporada: {actual_temp} '.center(120, '+'))

            # Extriago urls de items
            l_urls_items = crawler.extract_urls_items()

            # POR PARTIDO (c/u identificado con un id)
            for url in l_urls_items:  # Si no uso cont_part: for id in l_ids:

                # Intrego a pagina de partido
                crawler.driver.get(url)

                # Extraigo campos
                id = extract_id_from_url(url)
                print(id, url)

                hora_str = crawler.extract_tag(xpath='.//div[@id="match-header"]//tr[2]/td[2]/div[3]//dd[1]', text=True, sec_wait=SEC_WAIT_LONG)
                print(hora_str)
                fecha_str = crawler.extract_tag(xpath='.//div[@id="match-header"]//tr[2]/td[2]/div[3]//dd[2]', text=True, sec_wait=SEC_WAIT_LONG)
                print(fecha_str)
                # fecha_dt = datetime.datetime.strptime(fecha_str, "%d.%m.%Y %H:%M")

                equipo_loc = crawler.extract_tag(xpath='.//div[@id="match-header"]//td[@class="team"][1]', text=True, sec_wait=SEC_WAIT_LONG)
                equipo_vis = crawler.extract_tag(xpath='.//div[@id="match-header"]//td[@class="team"][2]', text=True, sec_wait=SEC_WAIT_LONG)
                print(equipo_loc)
                print(equipo_vis)

                ft_result = crawler.extract_tag(xpath='.//div[@id="match-header"]//td[@class="result"]', text=True, sec_wait=SEC_WAIT_LONG)
                goles_loc, goles_vis = int(ft_result.split(":")[0].strip()), int(ft_result.split(":")[1].strip())
                print(ft_result)
                print(goles_loc, goles_vis)

                ht_result = crawler.extract_tag(xpath='.//div[@id="match-header"]//tr[2]/td[2]/div[2]//dd[1]', text=True, sec_wait=SEC_WAIT_LONG)
                ht_goles_loc, ht_goles_vis = int(ht_result.split(":")[0].strip()), int(ht_result.split(":")[1].strip())
                print(ht_result)
                print(ht_goles_loc, ht_goles_vis)

                arbitro = crawler.extract_tag(xpath='.//span[@class="referee"]', attribute='title')
                cancha = crawler.extract_tag(xpath='.//span[@class="venue"]', attribute='title')
                print(arbitro)
                print(cancha)

                dt_loc = crawler.extract_tag(xpath='.//span[@class="manager-name"][1]', text=True)
                dt_vis = crawler.extract_tag(xpath='.//span[@class="manager-name"][2]', text=True)
                print(dt_loc)
                print(dt_vis)

                # Estadisticas del partido
                #  'tarjetas_amarillas': 'Tarjetas amarillas', --> no hace falta
                #  'ataques': 'Ataques', 'ataques_pelig': 'Ataques peligrosos'

                d = crawler.extract_estadisticas()
                print(d)

                d2 = crawler.estadisticas_detail()
                print(d2)

                '''
                rating_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="ratings"]//span[@data-field="home"]', text=True)
                rating_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="ratings"]//span[@data-field="away"]', text=True)
                print(rating_loc)
                print(rating_vis)


                # 'remates': 'Remates', 'remates_a_puerta': 'Remates a puerta',
                # Click en more y sacar de seccion no visible
                boton_more = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="shotsTotal"]/div[2]')
                crawler.click_boton(boton_more)

                remates_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsTotal"]//span[@data-field="home"]', text=True)
                remates_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsTotal"]//span[@data-field="away"]', text=True)
                remates_a_puerta_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsOnTarget"]//span[@data-field="home"]',text=True)
                remates_a_puerta_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsOnTarget"]//span[@data-field="away"]',text=True)
                remates_palos_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsOnPost"]//span[@data-field="home"]',text=True)
                remates_palos_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsOnPost"]//span[@data-field="away"]',text=True)
                remates_fuera_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsOffTarget"]//span[@data-field="home"]', text=True)
                remates_fuera_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsOffTarget"]//span[@data-field="away"]', text=True)
                remates_block_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsBlocked"]//span[@data-field="home"]',text=True)
                remates_block_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="shotsBlocked"]//span[@data-field="away"]',text=True)
                print(remates_loc)
                print(remates_vis)
                print(remates_a_puerta_loc)
                print(remates_a_puerta_vis)

                # posesion': 'Posesión de balón'
                posesion_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="possession"]//span[@data-field="home"]', text=True)
                posesion_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="possession"]//span[@data-field="away"]', text=True)
                print(posesion_loc, posesion_vis)

                # 'pases': 'Pases totales', 'pases_comp': 'Pases completados',
                # Click en more y sacar de seccion no visible
                boton_more = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="passSuccess"]/div[2]')
                crawler.click_boton(boton_more)

                porc_pases_comp_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passSuccess"]//span[@data-field="home"]', text=True)
                porc_pases_comp_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passSuccess"]//span[@data-field="away"]', text=True)
                total_pases_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passesTotal"]//span[@data-field="home"]', text=True)
                total_pases_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passesTotal"]//span[@data-field="away"]', text=True)
                pases_acer_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passesAccurate"]//span[@data-field="home"]',text=True)
                pases_acer_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passesAccurate"]//span[@data-field="away"]',text=True)
                pases_clave_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passesKey"]//span[@data-field="home"]', text=True)
                pases_clave_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="passesKey"]//span[@data-field="away"]', text=True)
                print(porc_pases_comp_loc, porc_pases_comp_vis)
                print(total_pases_loc, total_pases_vis)
                print(pases_acer_loc, pases_acer_vis)

                # Dribbles, aerials won, tackles?


                # 'faltas': 'Faltas', 'offsides': 'Fueras de juego',
                # Click en more y sacar de seccion no visible
                boton_more = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="dispossessed"]/div[2]')
                crawler.click_boton(boton_more)
                faltas_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="foulsCommited"]//span[@data-field="home"]', text=True)
                faltas_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="foulsCommited"]//span[@data-field="away"]', text=True)
                offsides_loc = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="offsidesCaught"]//span[@data-field="home"]', text=True)
                offsides_vis = crawler.extract_tag(xpath='.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="offsidesCaught"]//span[@data-field="away"]', text=True)
                '''

                # Datos de jugadores --> hoja "Player statistics"
                # Alineaciones + ratings


                # odds --> hoja "Betting"


                time.sleep(5)

                # l_data = [id, competicion, actual_temp, pais, tipo_comp, hora_str, fecha_str, equipo_loc, equipo_vis, goles_loc, goles_vis, ht_goles_loc, ht_goles_vis, arbitro, cancha, dt_loc, dt_vis]
                # print(l_data)
                # # Guardo datos
                # df.loc[len(df)] = l_data
                # df = pd.concat([df, pd.DataFrame(data)])


            '''
            # Si hay siguiente temporada
            tag_next_temp = crawler.extract_tag(xpath='.//div[@id="breadcrumb-nav"]//select[@id="seasons"]/option[@selected="selected"]//following-sibling::option')
            if tag_next_temp is not None:
                actual_temp = tag_next_temp.text
                crawler.click_boton(tag_next_temp)
            else:
                print("Ya no hay mas temporadas para esta competicion")
                break
            '''
            break


    '''
    l_comp = list(itertools.chain.from_iterable([[competicion]] * len(l_ids)))
    l_temps = list(itertools.chain.from_iterable([[actual_temp]] * len(l_ids)))
    l_pais = list(itertools.chain.from_iterable([[pais]] * len(l_ids)))
    l_copa = list(itertools.chain.from_iterable([[tipo_comp]] * len(l_ids)))

    data = {'id': l_ids, 'competicion': l_comp, 'temporada': l_temps, 'pais': l_pais, 'es_copa': l_copa,
            'url': l_url_items}
    df = pd.concat([df, pd.DataFrame(data)], axis=0)
    '''
    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()




class FlashscoreCrawler(Crawler):
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
            print(f"Cantidad de partidos: {len(l_url_items)}")
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

    def extract_estadisticas(self):

        SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
        d_nueva_fila = {}
        d_est = {'rating': 'ratings', 'posesion': 'possession'}

        # d_estadisticas = {'rating': 'ratings', 'remates': 'shotsTotal', 'remates_a_puerta': 'shotsOnTarget',
        #                   'tarjetas_amarillas': 'Tarjetas amarillas', 'faltas': 'Faltas', 'pases': 'Pases totales',
        #                   'pases_comp': 'Pases completados', 'offsides': 'Fueras de juego', 'ataques': 'Ataques',
        #                   'ataques_pelig': 'Ataques peligrosos'}

        # Por estadistica (posesion, remates, etc)
        for estadistica in d_est.keys():

            # Si no requiero hacer click en "More"
            # if d_est[estadistica] not in d_est_detail.keys():

            # Extraigo dicha estadistica tanto para el equipo local como para el visitante
            d_nueva_fila[f'{estadistica}_loc'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{d_est[estadistica]}"]//span[@data-field="home"]', text=True, sec_wait=SEC_WAIT)  #                 d_nueva_fila[f'{estadistica}_loc'] = super().extract_tag(xpath=f'.//div[text()="{d_estadisticas[estadistica]}"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)  # Falla el campo posesion_loc puesto que es el primero en ser extraido y aun no cargo...
            d_nueva_fila[f'{estadistica}_vis'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{d_est[estadistica]}"]//span[@data-field="away"]', text=True, sec_wait=SEC_WAIT)

        return d_nueva_fila

    def estadisticas_detail(self):

        SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
        d_nueva_fila = {}
        d_est = {'shotsTotal': ['shotsTotal', 'shotsOnTarget', 'shotsOnPost', 'shotsOffTarget', 'shotsBlocked'],
                 'passSuccess': ['passSuccess', 'passesTotal', 'passesAccurate', 'passesKey'],
                 'dispossessed': ['foulsCommited', 'offsidesCaught']}
        d_est_detail = {'shotsTotal': 'remates', 'shotsOnTarget': 'remates_a_puerta', 'shotsOnPost': 'remates_palos',
                        'shotsOffTarget': 'remates_fuera', 'shotsBlocked': 'remates_block', 'passSuccess': 'porc_pases_comp',
                        'passesTotal': 'total_pases', 'passesAccurate': 'pases_acer', 'passesKey': 'pases_clave', 'foulsCommited': 'faltas', 'offsidesCaught': 'offsides'}

        for campo in d_est.keys():
            print("Campo: ", campo)

            # Click en more y sacar de seccion no visible (es necesario hacerlo)
            boton_more = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]//li[@data-for="{campo}"]/div[2]')
            super().click_boton(boton_more)

            for sub_campo in d_est[campo]:
                print("\tSubcampo: ", sub_campo)

                d_nueva_fila[f'{d_est_detail[sub_campo]}_loc'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="{sub_campo}"]//span[@data-field="home"]',text=True,sec_wait=SEC_WAIT)  # d_nueva_fila[f'{estadistica}_loc'] = super().extract_tag(xpath=f'.//div[text()="{d_estadisticas[estadistica]}"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)  # Falla el campo posesion_loc puesto que es el primero en ser extraido y aun no cargo...
                d_nueva_fila[f'{d_est_detail[sub_campo]}_vis'] = super().extract_tag(xpath=f'.//div[@class="match-centre-stats" and @data-mode="team"]/ul[2]//li[@data-for="{sub_campo}"]//span[@data-field="away"]',text=True, sec_wait=SEC_WAIT)

                c1 = f'{d_est_detail[sub_campo]}_loc'
                c2 = f'{d_est_detail[sub_campo]}_vis'
                print(f'{c1}: {d_nueva_fila[c1]}')
                print(f'{c2}: {d_nueva_fila[c2]}')

        return d_nueva_fila

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Selecciono pais a extraer y obtengo las competencias y su categoria
    # pais = "argentina"  # Ver si creo un df y hago un ciclo para recorrer ≠ paises o que
    pais = 'Argentina'

    # Extraigo partidos
    df = extract_partidos_whoscored(pais)
