# Importo librerias
import time

import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler


def extract_jugadores_sofifa(pais, liga):
    """
    Obtengo datos de jugadores mediante scrapear sofifa.com
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    # Definicion de variables
    crawler = Crawler(headless=False, path=None)  # Usar False (con True no funciona)
    df_jug = pd.DataFrame(columns=['id_jugador', 'fifa', 'fecha', 'nombre', 'edad', 'altura', 'pie_habil', 'overall_rating', 'potencial', 'equipo_actual', 'valor_mercado', 'sueldo', 'pais'])  # usar d.keys() de headers... asi es automatico.. Ah no, pues extraigo algunos campos mas..
    print(f" PAÍS: {pais} ".center(120, "#"))

    # Ingreso a pagina de sofifa.com seleccionando como filtro los campos buscados (edad, altura, or, pot, valor_merc, etc)
    crawler.driver.get('https://sofifa.com/players?showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg')
    # crawler.driver.save_screenshot("1_pagina_inicial_sofifa.png")  # Tomar un screenshot y guardarlo en un archivo (sale cortada pero maximizé y no funcionó)

    # Filtro listado de jugadores segun la liga del pais que busco
    select_liga_as_filter(crawler, pais, liga)  # crawler.driver.save_screenshot("2_pagina_antes_de_seleccionar_liga.png")
    boton_sumbit = crawler.extract_tag(xpath='.//button[text()="Submit"]')  # Clickeo en "buscar"
    crawler.click_boton(boton_sumbit)

    # Obtengo urls de las paginas de la paginacion (c/pagina es un año o fifa)
    l_tag_years = crawler.extract_tags(xpath='.//select[@name="version"]/option')
    l_urls_years = ["https://sofifa.com/" + tag.get_attribute('value') for tag in l_tag_years]

    # POR FIFA (e.g. Fifa 23, fifa 22, fifa 21, ..., fifa 07)
    for url_year in l_urls_years:

        # Ingreso a pagina del año o fifa
        crawler.driver.get(url_year)
        # crawler.driver.save_screenshot("3_seleccion_de_fifa.png")

        # Obtengo urls de las paginas de la paginacion (c/pagina es una actualizacion de un fifa)
        l_tags_act_year = crawler.extract_tags(xpath='.//select[@name="roster"]/option') # .//h2//div[@class="dropdown"][2]/div/a[not(contains(text(), "World Cup"))] # Ojo con la actualizacion World Cup 2022...  # NO HACE FALTA HACER CLICK EN FLECHITA ANTES -->  #   boton_selec_act_fifa =  crawler.extract_tag(xpath='.//h2//div[@class="dropdown"][2]/a')  # Ver si hace click, tal vez ni hace falta  # crawler.click_boton(boton_selec_act_fifa)
        l_urls_act_year = ["https://sofifa.com/" + tag.get_attribute('value') for tag in l_tags_act_year]
        l_urls_act_year_sel = [l_urls_act_year[0], l_urls_act_year[-1]]  # Selecciono unicamente la primera y la ultima actualizacion
        fifa = crawler.extract_tag(xpath='.//select[@name="version"]/option[@selected]', text=True)  # Fifa 21
        print(f" FIFA: {fifa} ".center(120, "+"))

        # POR ACTUALIZACION EN DICHO FIFA (e.g. Jun 7, 2023;  Apr 17, 2023; etc)
        for url_year_act in l_urls_act_year_sel:

            # Ingreso a paging de la actualizacion
            crawler.driver.get(url_year_act)

            # Obtengo la fecha de actualizacion
            fecha_str = crawler.extract_tag(xpath='.//select[@name="roster"]/option[@selected]', text=True)  # e.g. May 16, 2023
            print(f" Fecha de actualizacion: {fecha_str} ".center(120, "-"))
            n_jug_encontrados = 0

            # POR PAGINA CON LISTADO DE JUGADORES
            while True:

                # Obtengo tags de jugadores
                l_tag_jugadores = crawler.extract_tags(xpath='.//main/article/table/tbody/tr')
                n_jug_encontrados += len(l_tag_jugadores)
                print(f"Cantidad de jugadores: {len(l_tag_jugadores)}")

                # Por jugador
                for tag in l_tag_jugadores:
                    # Extraigo datos del jugador
                    d_data = {}
                    d_data['id_jugador'] = crawler.extract_tag(tag_inicial=tag,xpath='.//td[@data-col="pi"]', text=True)
                    d_data['fifa'] = fifa
                    d_data['fecha'] = fecha_str
                    d_data['nombre'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[not(@class)]/a', attribute="data-tippy-content")
                    d_data['edad'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="ae"]', text=True)
                    d_data['altura'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="hi"]', text=True)
                    d_data['pie_habil'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pf"]', text=True)
                    d_data['overall_rating'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="oa"]/em', text=True)
                    d_data['potencial'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pt"]/em', text=True)
                    d_data['equipo_actual'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td/a[starts-with(@href, "/team")]', text=True)
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
                boton_next = crawler.extract_tag(xpath='.//div[@class="pagination"]/a[text()="Next "]')
                if crawler.click_boton(boton_next) is False:
                    print('Ya no hay mas boton "Next". Es decir, ya no hay mas jugadores en la liga.')
                    break

            print(f"Cantidad de jugadores encontrados: {n_jug_encontrados}")

        # Exporto datos del fifa (Por seguridad)
        df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/data_seg/por_temporada/df_jug/{fifa}.xlsx')

    # Exporto dataset final
    df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_jug.xlsx')

    # Cierro webdriver
    crawler.driver.close()
    return df_jug

def select_liga_as_filter(crawler, pais, liga=None):
    """
    Poner el pais como filtro para obtener los jugadores solo de la liga de dicho pais
    :param crawler:
    :param pais: String. Nombre de pais al que pertenece la liga.
    :param liga: String. Nombre de la liga de la cual extraer los jugadores.
    :return:
    """
    print("Seleccionando liga del pais como filtro...")

    # Si el usuario no pasa una liga
    if liga is None:
        # Selecciono competicion mas importante del pais
        df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx')
        liga = df_comp[(df_comp['pais'] == pais) and (df_comp['is_cup'] == 0)]['competicion'].values[0]  # La primera liga (evito si esta la B de esa liga) # df_comp[df_comp['pais'] == pais]['competicion'].values[0]  # La primera competicion de todas las competiciones del pais

    # Cargo competicion en el buscador de ligas
    input_league = crawler.extract_tag(xpath='.//form[@class="pjax-form" and @action="/players"]//input[@placeholder="Leagues"]')
    input_league.send_keys(liga)

    # Posibles ligas segun nuestra busqueda
    l_posibles_ligas = crawler.extract_tags(xpath='.//form[@class="pjax-form" and @action="/players"]//input[@placeholder="Leagues"]//parent::div//following-sibling::div//div[starts-with(@class, "choices-item")]') # a veces crashea el click pero funciona
    print("\tNº de posibles ligas:", len(l_posibles_ligas))

    # Por posible liga
    for liga in l_posibles_ligas:

        # Extraigo el pais
        pais_posible_liga = crawler.extract_tag(tag_inicial=liga, xpath='./img', attribute='title')  # Selecciono el div antes que la img.
        print("\tPais de posible liga: ", pais_posible_liga)

        # Si es el pais que estoy buscando
        if pais_posible_liga.lower() == pais.lower():  # Podria agregarle coincidencia del 90% por si cambia algun caracter. O bien el tema idioma.
            print("\tEncontró la liga y el pais deseado")

            # Hago click en la liga
            time.sleep(5) # Espero a que se cargen las opciones antes de hacer click (puede que el div no sea clickable o algo)
            if crawler.click_boton(boton=liga) is False:
                print(f'Sofifa no encontró resultados a nuestra busqueda. Es posible que no exista la liga de {pais}.')
            break

def prueba():
    # Agregar: Definir que competicion y que pais queres extraer aquí segun df_comp...
    pais = "England"
    liga = "Championship"
    extract_jugadores_sofifa(pais, liga)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()
