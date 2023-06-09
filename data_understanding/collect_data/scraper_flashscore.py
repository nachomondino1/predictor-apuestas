# Importo librerias
import pandas as pd
import datetime
import time
import random
import warnings
from dspy.data_understanding.web_scraping.selenium import Crawler


def extract_flashscore():
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings("ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    fecha_act = datetime.datetime.now()
    crawler = Crawler(headless=True, path=None) # Creo objeto de clase CrawlerActions()
    d_formaciones = {'Formaciones iniciales': 'tit', 'Suplentes': 'sup', 'Jugadores reemplazados': 'sup_ing','Jugadores ausentes': 'aus'}
    d_estadisticas = {'posesion': 'Posesión de balón', 'remates': 'Remates', 'remates_a_puerta': 'Remates a puerta',
                      'tarjetas_amarillas': 'Tarjetas amarillas', 'faltas': 'Faltas', 'pases': 'Pases totales',
                      'pases_comp': 'Pases completados', 'offsides': 'Fueras de juego', 'ataques': 'Ataques',
                      'ataques_pelig': 'Ataques peligrosos'}
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_competicion.xlsx')
    df_part = pd.DataFrame()  # No hace falta definir columnas por mas que no haya extraido partidos

    # POR PAIS
    for pais in df_comp['pais'].unique()[:1]:
        pais = "inglaterra"

        # Filtro competiciones por pais
        # df_comp_pais = df_comp[df_comp['pais'] == pais]
        l_comp = ['premier league']
        l_cat = ['Liga']
        print(f' PAIS: {pais} '.center(120, '#'))

        '''  # Para cuando intente no recolectar los partidos ya extraidos
        # Obtengo datos ya extraidos del pais  (TENER EN CUENTA SI FALLA...)
        try:
            df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_partido_{pais}.xlsx')
        except FileNotFoundError:
            df_part = pd.DataFrame(columns=['id'])  # inicializo con id para evitar error en if id not in df_part
        '''

        # POR COMPETICION
        # for competicion, categoria in zip(df_comp_pais['nombre'], df_comp_pais['categoria']):
        for competicion, categoria in zip(l_comp, l_cat):

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

            # POR PAGINA (TEMPORADA) DE PAGINACION
            for url_temp in l_urls_temporadas[4:5]:  # solo 2018/19

                # Ingreso a pagina de temporada e imprimo año de la temporada
                crawler.driver.get(url_temp)
                temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
                print(f" {temp_year} ".center(120, "-"))

                # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
                crawler.click_boton(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=SEC_WAIT_LONG*3, repeat_click=True)  # Si hace click, es None. Si falla, es un str  # A veces me tira (y no entiendo por qué): selenium.common.exceptions.StaleElementReferenceException: Message: stale element reference: stale element not found

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
                    d_nueva_fila = {'id': id, 'competicion': competicion, 'temporada': temp_year, 'pais': pais, 'es_copa': 1 if categoria == "Copa" else 0}
                    cont_part += 1
                    print(f" Partido {cont_part} de {len(l_items)}. Recolectado el {cont_part / len(l_items) * 100:.0f}% ".center(120, "."))

                    # Si aun no extraje dicho id
                    # if id not in list(df_part['id']):

                    # Ingreso a pagina de informacion del partido
                    crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

                    # EXTRACCION DE CAMPOS
                    fecha_str = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                    fecha_dt = datetime.datetime.strptime(fecha_str, "%d.%m.%Y %H:%M")

                    # Si el partido aun no se jugo (extraia partidos de la Copa de la Liga profesional 2023 la cual aun no se jugo pero ya esta el fixture... tampoco es tan grave solo esta la jornada 1)
                    if fecha_dt < fecha_act:

                        # Extraigo campos de hoja "Resumen"
                        d_nueva_fila['fecha'] = fecha_dt  # d_nueva_fila['fecha'] = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                        d_nueva_fila['equipo_loc'] = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__home")]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['equipo_vis'] = crawler.extract_tag(xpath='.//div[starts-with(@class, "duelParticipant__away")]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['goles_loc'] = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[1]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['goles_vis'] = crawler.extract_tag(xpath='.//div[@class="detailScore__wrapper"]/span[3]', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['arbitro'] = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span', text=True, sec_wait=SEC_WAIT)
                        d_nueva_fila['cancha'] = crawler.extract_tag(xpath='.//div[@class="mi__data"]//span[contains(text(), "Estadio")]/following-sibling::span', text=True, sec_wait=SEC_WAIT)

                        # Extraigo campos de hoja "Estadísticas"
                        if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Estadísticas"]', sec_wait=SEC_WAIT_LONG) is not False:

                            time.sleep(random.uniform(SEC_WAIT+3, SEC_WAIT_LONG+3)) # Falla el campo posesion_loc puesto que es el primero en ser extraido y aun no cargo...

                            # Por estadistica (posesion, remates, etc)
                            for estadistica in d_estadisticas.keys():

                                # Extraigo dicha estadistica tanto para el equipo local como para el visitante
                                d_nueva_fila[f'{estadistica}_loc'] = crawler.extract_tag(xpath=f'.//div[text()="{d_estadisticas[estadistica]}"]//preceding-sibling::div', text=True, sec_wait=SEC_WAIT)  # Falla el campo posesion_loc puesto que es el primero en ser extraido y aun no cargo...
                                d_nueva_fila[f'{estadistica}_vis'] = crawler.extract_tag(xpath=f'.//div[text()="{d_estadisticas[estadistica]}"]//following-sibling::div', text=True, sec_wait=SEC_WAIT)

                        # Extraigo campos de hoja "Formaciones"
                        if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Formaciones"]', sec_wait=SEC_WAIT) is not False:

                            time.sleep(random.uniform(SEC_WAIT+3, SEC_WAIT_LONG+3)) # Por posible falla en el primer campo a extraer

                            # Por seccion ("Formacion inicial", "Suplentes" y  "Ausentes")
                            for formacion in d_formaciones.keys():

                                # Si existe la seccion "Formaciones"
                                if crawler.extract_tag(xpath=f'.//div[text()="{formacion}"]', sec_wait=SEC_WAIT_LONG) is not None:

                                    # Extraigo listado de jugadores
                                    l_tags_jug_loc = crawler.extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT*2)  # Es lista de tags o None
                                    l_tags_jug_vis = crawler.extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT*2)  # Es lista de tags o None

                                    # Creo una columna por cada jugador local
                                    if l_tags_jug_loc is not None:
                                        for i in range(len(l_tags_jug_loc)):
                                            # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "Genez Nahuel")
                                            try:
                                                url_with_nombre = l_tags_jug_loc[i].get_attribute('href')  # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                                                nombre = url_with_nombre.split('/')[4].replace('-', " ")
                                            # Extraigo nombre corto (e.g. "N.genez")
                                            except:
                                                nombre = l_tags_jug_loc[i].text

                                            # Extraigo nombre
                                            d_nueva_fila[f'jug_{d_formaciones[formacion]}_loc_{i+1}'] = nombre  # l_tags_jug_loc[i].text

                                    # Creo una columna por cada jugador visitante
                                    if l_tags_jug_vis is not None:
                                        for i in range(len(l_tags_jug_vis)):
                                            # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "genez nahuel")
                                            try:
                                                url_with_nombre = l_tags_jug_vis[i].get_attribute('href')  # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                                                nombre = url_with_nombre.split('/')[4].replace('-', " ")
                                            # Extraigo nombre corto (e.g. "N.genez")
                                            except:
                                                nombre = l_tags_jug_vis[i].text

                                            d_nueva_fila[f'jug_{d_formaciones[formacion]}_vis_{i+1}'] = nombre  # l_tags_jug_vis[i].text

                            # Extraigo entrenadores  # Sacar los try - except horribles
                            # d_nueva_fila['dt_loc'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]', text=True, sec_wait=SEC_WAIT)
                            # d_nueva_fila['dt_vis'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]', text=True, sec_wait=SEC_WAIT)
                            # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "genez nahuel")
                            try:
                                url_with_nombre = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]//a[@class="lf__participantName"]',  attribute='href', sec_wait=SEC_WAIT) # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                                d_nueva_fila['dt_loc'] = url_with_nombre.split('/')[4].replace('-', " ")
                            # Extraigo nombre corto (e.g. "N.genez")
                            except:
                                d_nueva_fila['dt_loc'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]', text=True, sec_wait=SEC_WAIT)

                            # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "genez nahuel")
                            try:
                                url_with_nombre = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]//a[@class="lf__participantName"]', attribute='href', sec_wait=SEC_WAIT)  # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                                d_nueva_fila['dt_vis'] = url_with_nombre.split('/')[4].replace('-', " ")
                            # Extraigo nombre corto (e.g. "N.genez")
                            except:
                                d_nueva_fila['dt_vis'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]',text=True, sec_wait=SEC_WAIT)

                        # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                        if crawler.extract_tag(xpath='.//div[@class="oddsRow"]') is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...
                            
                            d_nueva_fila['odds_loc'] = extract_cuota(crawler, SEC_WAIT, i=1)
                            d_nueva_fila['odds_emp'] = extract_cuota(crawler, SEC_WAIT, i=2)
                            d_nueva_fila['odds_vis'] = extract_cuota(crawler, SEC_WAIT, i=3)
                            print(d_nueva_fila['odds_loc'], d_nueva_fila['odds_emp'], d_nueva_fila['odds_vis'])

                        # GUARDADO DE DATOS EN DATAFRAME
                        df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
                        print(df_part.iloc[-1])

                        end = time.time()
                        print(f"Partido recolectado en {(end - start):.1f} segundos")

                # Guardo partidos no extraidos de la temporada (por seguridad)
                df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/data_seg/{competicion_form}_{temp_year.replace("/","_")}_{pais}.xlsx', index=False)

        # Guardado de archivo excel en computadora
        df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/{pais}/entidad_partido_prueba.xlsx',index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part

def extract_cuota(crawler, SEC_WAIT, i):  # Puedo volver a la anterior, solo fallaron 41 cuotas por "Cuotas retiradas por la casa de apuestas."

    # Busco odd suponiendo que cambio durante el partido
    try:
        odds_str = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]', attribute='title', sec_wait=SEC_WAIT)  # 3.00 » 2.25
        cuota = float(odds_str.split('»')[0].strip())  # 3.00

    # Si la odd no cambio durante el partido, o bien, aparece "Cuotas retiradas por la casa de apuestas."
    except ValueError: # could not convert string to float:

        try:
            odds_str = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]//span[@class="oddsValueInner"]', text=True, sec_wait=SEC_WAIT)
            cuota = float(odds_str)

        except (ValueError, TypeError):
            print("Fallo extraccion de la cuota")
            cuota = None
    return cuota

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    extract_flashscore()


# Puedo eficientizar el codigo (en entrenadores y demas) agregando la posibilidad de un xpath alternativo en extract_tag...
# Solucionar el tema de que cuando falla un campo, tengo que volver a extraer tod@... Dar la posibildiad de recorrer los ids ya extraidos y extraer de nuevo el campo que falló


# Podria poner la extraccion de datos, campos mal extraidos y proximos partidos junto en el mismo archivo...
# Poner argumentos en extract_flashscore...