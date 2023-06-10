import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
from selenium.webdriver.common.keys import Keys


def extract_sofifa():
    """
    Obtengo datos de jugadores mediante scrapear sofifa.com
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_competicion.xlsx')
    df_jug = pd.DataFrame(columns=['id_jugador', 'fecha', 'nombre', 'edad', 'altura', 'pie_habil', 'overall_rating', 'potencial', 'equipo_actual', 'valor_mercado', 'sueldo', 'pais'])  # usar d.keys() de headers... asi es automatico.. Ah no, pues extraigo algunos campos mas..
    crawler = Crawler(headless=True, path=None)
    d_pais_a_seleccionar = {'brazil': 'brasil'}  # Por diferencias entre nombres de paises entre Flashscore y Sofifa
    l_paises = ['argentina', 'uruguay']

    # Ingreso a pagina de sofifa.com seleccionando como filtro los campos buscados
    crawler.driver.get('https://sofifa.com/players?showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg')

    # Obtengo urls de las paginas de la paginacion (c/pagina es un año o fifa)
    crawler.click_boton(xpath='.//h2//div[@class="dropdown"][1]/a')  # Ver si hace click, tal vez ni hace falta
    l_tag_years = crawler.extract_tags(xpath='.//h2//div[@class="dropdown"][1]/div/a')
    l_urls_years = [tag.get_attribute('href') for tag in l_tag_years]

    # POR AÑO O FIFA (e.g. Fifa 23, fifa 22, fifa 21, ..., fifa 07)
    for url_year in l_urls_years[::-1]:

        # Ingreso a pagina del año o fifa
        crawler.driver.get(url_year)
        print(f" Url pagina: {url_year} ")

        # Obtengo urls de las paginas de la paginacion (c/pagina es una actualizacion de un fifa)
        crawler.click_boton(xpath='.//h2//div[@class="dropdown"][2]/a')  # Ver si hace click, tal vez ni hace falta
        l_tags_act_year = crawler.extract_tags(xpath='.//h2//div[@class="dropdown"][2]/div/a[not(contains(text(), "World Cup"))]')  # Ojo con la actualizacion World Cup 2022...
        l_urls_act_year = [tag.get_attribute('href') for tag in l_tags_act_year]
        l_urls_act_year_sel = [l_urls_act_year[0], l_urls_act_year[-1]]  # Selecciono unicamente la primera y la ultima actualizacion
        print("Lista de urls de actualizaciones fifa seleccionadas: ", l_urls_act_year)

        # POR ACTUALIZACION EN DICHO FIFA (e.g. Jun 7, 2023;  Apr 17, 2023; etc)
        for url_year_act in l_urls_act_year_sel:

            crawler.driver.get(url_year_act)
            print(f" Url actualizacion: {url_year_act} ")

            # Extraigo fecha de la actualizacion e.g. May 16, 2023
            fecha_str = crawler.extract_tag(xpath='.//h2/div[@class="dropdown"][2]', text=True)  # e.g. May 16, 2023
            print(f" Fecha de actualizacion del fifa: {fecha_str} ".center(120, "#"))

            # POR PAIS
            # for pais in df_comp['pais'].unique()[:-1]:  # Evito "Sudamerica"
            for pais in l_paises:  # CREO QUE LO VOY A HACER PARA UN SOLO PAIS. DE ULTIMA SI DESPUES LOS QUIERO JUNTAR, LOS CONCATENO... Y EL PAIS LO PASO POR PARAMETRO.
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

            # Por seguridad, exporto datasets
            df_jug.to_excel(f'./data/data_seg/entidad_jugadores_{fecha_str}.xlsx')

    # Exporto dataset final y cierro webdriver
    df_jug.to_excel('./enticed_jugadores.xlsx')
    crawler.driver.close()

    return df_jug

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    extract_sofifa()