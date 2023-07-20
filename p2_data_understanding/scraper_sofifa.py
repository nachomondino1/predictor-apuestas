# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler


def extract_jugadores_sofifa(pais):
    """
    Obtengo datos de jugadores mediante scrapear sofifa.com
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    crawler = Crawler(headless=True, path=None)
    df_jug = pd.DataFrame(columns=['id_jugador', 'fifa', 'fecha', 'nombre', 'edad', 'altura', 'pie_habil', 'overall_rating', 'potencial', 'equipo_actual', 'valor_mercado', 'sueldo', 'pais'])  # usar d.keys() de headers... asi es automatico.. Ah no, pues extraigo algunos campos mas..
    print(f" PAÍS: {pais} ".center(120, "#"))

    # Ingreso a pagina de sofifa.com seleccionando como filtro los campos buscados (edad, altura, or, pot, valor_merc, etc)
    crawler.driver.get('https://sofifa.com/players?showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg')

    # Obtengo urls de las paginas de la paginacion (c/pagina es un año o fifa)
    l_tag_years = crawler.extract_tags(xpath='.//h2//div[@class="dropdown"][1]/div/a')   # NO HACE FALTE HACER CLICK PREVIO EN LA FLECHITA -->  # boton_selec_fifa = crawler.extract_tag(xpath='.//h2//div[@class="dropdown"][1]/a')   # crawler.click_boton(boton_selec_fifa)  # Hace falta?
    l_urls_years = [tag.get_attribute('href') for tag in l_tag_years]

    # POR FIFA (e.g. Fifa 23, fifa 22, fifa 21, ..., fifa 07)
    for url_year in l_urls_years:

        # Ingreso a pagina del año o fifa
        crawler.driver.get(url_year)

        # Obtengo urls de las paginas de la paginacion (c/pagina es una actualizacion de un fifa)
        l_tags_act_year = crawler.extract_tags(xpath='.//h2//div[@class="dropdown"][2]/div/a[not(contains(text(), "World Cup"))]')  # Ojo con la actualizacion World Cup 2022...  # NO HACE FALTA HACER CLICK EN FLECHITA ANTES -->  #   boton_selec_act_fifa =  crawler.extract_tag(xpath='.//h2//div[@class="dropdown"][2]/a')  # Ver si hace click, tal vez ni hace falta  # crawler.click_boton(boton_selec_act_fifa)
        l_urls_act_year = [tag.get_attribute('href') for tag in l_tags_act_year]
        l_urls_act_year_sel = [l_urls_act_year[0], l_urls_act_year[-1]]  # Selecciono unicamente la primera y la ultima actualizacion
        fifa = crawler.extract_tag(xpath='.//h2/div[@class="dropdown"][1]', text=True)  # Fifa 21
        print(f" FIFA: {fifa} ".center(120, "+"))

        # POR ACTUALIZACION EN DICHO FIFA (e.g. Jun 7, 2023;  Apr 17, 2023; etc)
        for url_year_act in l_urls_act_year_sel:

            # Ingreso a pagina de la actualizacion y obtengo la fecha
            crawler.driver.get(url_year_act)
            fecha_str = crawler.extract_tag(xpath='.//h2/div[@class="dropdown"][2]', text=True)  # e.g. May 16, 2023
            print(f" Fecha de actualizacion: {fecha_str} ".center(120, "-"))

            # Selecciono la liga del pais como filtro
            select_liga_as_filter(crawler, pais)

            # Clickeo en "buscar"
            boton_sumbit = crawler.extract_tag(xpath='.//button[text()="Submit"]')
            crawler.click_boton(boton_sumbit)

            # POR PAGINA CON LISTADO DE JUGADORES
            while True:

                # Obtengo tags de jugadores
                l_tag_jugadores = crawler.extract_tags(xpath='.//table[@class="table table-hover persist-area"]/tbody/tr')
                print(f"Cantidad de jugadores: {len(l_tag_jugadores)}")

                # Por jugador
                for tag in l_tag_jugadores:
                    # Extraigo datos del jugador
                    d_data = {}
                    d_data['id_jugador'] = crawler.extract_tag(tag_inicial=tag,xpath='.//td[@data-col="pi"]', text=True)
                    d_data['fifa'] = fifa
                    d_data['fecha'] = fecha_str
                    d_data['nombre'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@class="col-name"]/a', attribute="aria-label")  # Nombre corto (e.g. l. gonzalez pirez) d_data['nombre'] = crawler.extract_tag(tag_inicial=tag, xpath='.//td[@class="col-name"]/a/div[@class="ellipsis"]', text=True)
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
                boton_next = crawler.extract_tag(xpath='.//div[@class="pagination"]//span[contains(@class, "right")]//parent::a')
                if crawler.click_boton(boton_next) is False:
                    print('Ya no hay mas boton "Next". Es decir, ya no hay mas jugadores en la liga.')
                    break

        # Exporto datos del fifa (Por seguridad)
        df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/data_seg/entidad_jugadores_{fifa}.xlsx')

    # Exporto dataset final
    df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/entidad_jugadores.xlsx')

    # Cierro webdriver
    crawler.driver.close()
    return df_jug

def select_liga_as_filter(crawler, pais):
    """
    Poner el pais como filtro para obtener los jugadores solo de la liga de dicho pais
    :param crawler:
    :param pais:
    :return:
    """
    # Definicion de variables
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx')

    # Selecciono competicion mas importante del pais
    pais_a_buscar = df_comp[df_comp['pais'] == pais]['pais'].values[0]  # pais_a_buscar = df_comp[df_comp['pais'] == pais]['pais_sofifa'].values[0]
    comp_a_buscar = df_comp[df_comp['pais'] == pais]['competicion'].values[0]  # comp_a_buscar = df_comp[df_comp['pais'] == pais]['nombre_sofifa'].values[0]

    # Busco competicion
    input_league = crawler.extract_tag(xpath='.//form[@class="relative pjax-form"]//input[@aria-label="Leagues"]')
    input_league.send_keys(comp_a_buscar)  # AHORA: "Liga Profesional" ya no puedo usar el pais..  ANTES:  "[Argentina] Liga Profesional"

    # Si encontró resultados a la busqueda
    if len(crawler.extract_tags(xpath=f'.//form[@class="relative pjax-form"]//input[@aria-label="Leagues"]//parent::div//following-sibling::div//div[starts-with(@class, "choices-item")]')) > 0:  # AHORA: e.g. Liga profesional  ANTES: e.g. [Argentina] Liga profesional

        # Busco el boton para la liga requerida segun el pais buscado
        boton_liga_a_selec = crawler.extract_tag(xpath=f'.//form[@class="relative pjax-form"]//input[@aria-label="Leagues"]//parent::div//following-sibling::div//div[starts-with(@class, "choices-item")]/img[@title="{pais_a_buscar.capitalize()}"]')  # AHORA: e.g. Liga profesional  ANTES: e.g. [Argentina] Liga profesional

        # Hago click en la liga
        if crawler.click_boton(boton_liga_a_selec) is False:
            print(f'Sofifa no encontró resultados a nuestra busqueda. Es posible que no exista la liga de {pais}.')

def prueba():
    pais = "Argentina"
    extract_jugadores_sofifa(pais)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()
