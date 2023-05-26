import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler
from selenium.webdriver.common.keys import Keys
from time import sleep
import random


def main():
    """
    Obtengo datos de jugadores mediante scrapear sofifa.com
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_competicion.xlsx')
    df_jug = pd.DataFrame(columns=['id_jugador', 'fecha', 'nombre', 'edad', 'altura', 'pie_habil', 'overall_rating', 'potencial', 'equipo_actual', 'valor_mercado', 'sueldo', 'pais'])  # usar d.keys() de headers... asi es automatico.. Ah no, pues extraigo algunos campos mas..
    crawler = Crawler(headless=False, path=None)  # por algun motivo tarda 10 años el headless=True. A su vez el headless=False solo funciona si efectivamente miro el driver que se abre (sino no)
    d = {}

    # Ingreso a pagina de sofifa.com con los filtros correctos
    crawler.driver.get('https://sofifa.com/players?showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg')

    # Extraigo headers para poder saber que indice tiene cada campo a extraer (la estructura html es complicada)
    l_tags_headers = crawler.extract_tags(xpath='.//tr[@class="persist-header"]/th')
    l_headers = [tag.text for tag in l_tags_headers]
    for i, header in enumerate(l_headers):
        d[header] = i + 1
    print(f"Campos a extraer y su posicion: \n{d}")

    # Obtengo paginas de la paginacion (c/pagina es un año)
    crawler.click_boton(xpath='.//h2//div[@class="dropdown"][1]/a')  # Ver si hace click, tal vez ni hace falta
    l_tag_years = crawler.extract_tags(xpath='.//h2//div[@class="dropdown"][1]/div/a')
    l_urls_years = [tag.get_attribute('href') for tag in l_tag_years]

    # POR AÑO (recorro todos las paginas de sofifa.com)
    for url_year in l_urls_years:

        # Ingreso a pagina del año
        crawler.driver.get(url_year)
        print(f" Url pagina: {url_year} ")

        # Extraigo fecha
        # crawler.click_boton(xpath='.//h2/div[@class="dropdown"][2]')  # e.g. May 16, 2023
        fecha = crawler.extract_tag(xpath='.//h2/div[@class="dropdown"][2]', text=True)  # e.g. May 16, 2023
        print(f" Fecha de actualizacion del fifa: {fecha} ".center(120, "#"))

        # POR PAIS
        for pais in df_comp['pais'].unique()[:-1]:  # Evito "Sudamerica"

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

                pais_a_seleccionar = liga_a_seleccionar[liga_a_seleccionar.find('[')+1: liga_a_seleccionar.find(']')] # e.g. [Argentina] Liga profesional
                print(f"Pais a seleccionar: {pais_a_seleccionar}")

                # Si la liga a seleccionar corresponde a la del pais
                if pais.lower()[:3] in pais_a_seleccionar.lower()[:3]:  # Comparo las 3 primeras letras puesto que falla 'Brasil' in '[Brazil] ...' y 'Perú' in '[Peru] ...'

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
                            # Por campo a extraer
                            for header, pos_tag in d.items():

                                # Si no debo extraer el nombre
                                if header != "NAME":
                                    d_data[header] = crawler.extract_tag(tag_inicial=tag, xpath=f'.//td[{pos_tag}]', text=True)
                                else:
                                    d_data[header] = crawler.extract_tag(tag_inicial=tag, xpath=f'.//td[{pos_tag}]/a',attribute="aria-label")

                            # Formateo campos
                            print(d_data['NAME'])
                            d_data['NAME'] = d_data['NAME'].split('\n')[0]
                            d_data['HEIGHT'] = d_data['HEIGHT'].split('cm')[0]  # e.g. 193cm / 6'4" --> 193
                            d_data['TEAM & CONTRACT'] = d_data['TEAM & CONTRACT'].split('\n')[0]  # e.g. Salzburg 2021 ~ 2026 --> Salzburg

                            # Guardo los datos del jugador para dicho año
                            l_data = [d_data['ID'], fecha, d_data['NAME'], d_data['AGE'], d_data['HEIGHT'], d_data['FOOT'], d_data['OVERALL RATING'], d_data['POTENTIAL'], d_data['TEAM & CONTRACT'], d_data['VALUE'], d_data['WAGE'], pais]
                            df_jug.loc[len(df_jug)] = l_data
                            print(l_data)

                        # Clickeo en boton "Next" para recorrer todas las paginas
                        if crawler.click_boton(xpath='.//div[@class="pagination"]//span[contains(@class, "right")]//parent::a') is False:
                            print('Ya no hay mas boton "Next". Es decir, ya no hay mas jugadores en la liga.')
                            break

                # Si la liga a seleccionar no corresponde a la del pais
                else:
                    print(f'Sofifa encontró la liga {liga_a_seleccionar} la cual no corresponde a nuestra busqueda {pais}. Es posible que no exista la liga de {pais} en el fifa del año {fecha}')

            # Si no encontro resultados para nuestro input ("No results found")
            else:
                print(f'Sofifa no encontró resultados a nuestra busqueda. Es posible que no exista la liga de {pais} en el fifa del año {fecha}')

                # Seleccionar el texto cargado en el input y lo borro
                input_league.send_keys(Keys.SHIFT + Keys.HOME)
                input_league.send_keys(Keys.DELETE)

        # Exporto datasets por seguridad
        df_jug.to_excel(f'./entidad_jugadores_{fecha}.xlsx')

    # Exporto dataset final
    df_jug.to_excel(f'./entidad_jugadores.xlsx')
    crawler.driver.close()

main()