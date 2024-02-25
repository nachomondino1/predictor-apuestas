# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p2_data_understanding.collect_initial_data.web_scraping_selenium import Crawler
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm


class SofifaCrawler(Crawler):
    """
    Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper
    normal, scraper de fields especificos poro falla y scraper de proximos partidos...
    Contiene todo los xpath.
    """
     
    def __init__(self, headless, path=None, browser="Chrome"):
        super().__init__(headless, path, browser)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 0.8  # Espera para elementos que muchas veces no estan # con 0.2 fallaba extraccion de campos que si estaban como goals
        self.SEC_WAIT_MAX = 5  # Espera para elementos que casi siempre estan

    def select_league_as_filter(self, country, league):
        """
        Poner el country como filtro para obtener los players solo de la league de dicho country
        :param crawler:
        :param country: String. Name de country al que pertenece la league.
        :param league: String. Name de la league de la cual extraer los players.
        :return:
        """
        print("Seleccionando league del country como filtro...")
        # Cargo competition en el buscador de leagues
        input_league = super().extract_tag(xpath='.//form[@class="pjax-form" and @action="/players"]//input[@placeholder="Leagues"]')
        input_league.send_keys(league)

        # Selecciona la opcion segun name del country con send_keys
        ## Posibles leagues segun nuestra busqueda
        l_posibles_leagues = super().extract_tags(xpath='.//form[@class="pjax-form" and @action="/players"]//input[@placeholder="Leagues"]//parent::div//following-sibling::div//div[starts-with(@class, "choices-item")]', sec_wait=self.SEC_WAIT_MAX)  # a veces crashea el click pero funciona
        print("\tNº de posibles leagues:", len(l_posibles_leagues))

        ## Por posible league
        for tag_league in l_posibles_leagues:
            
            # Extraigo el country
            country_posible_league = super().extract_tag(tag_inicial=tag_league, xpath='./img', attribute='title', sec_wait=self.SEC_WAIT_MAX)  # Selecciono el div antes que la img.
            print("\tCountry de posible league: ", country_posible_league)

            # Si es el country que estoy buscando
            if country_posible_league.lower() == country.lower():  # Podria agregarle coincidencia del 90% por si cambia algun caracter. O bien el tema idioma.
                input_league.send_keys(Keys.RETURN)
                print("Se seleccionó una league.")
                break
            else:
                input_league.send_keys(Keys.ARROW_DOWN) # con tab no funciona
    
    def click_submit(self):
        boton_sumbit = super().extract_tag(xpath='.//button[text()="Submit"]', sec_wait=self.SEC_WAIT_MAX)  # Clickeo en "buscar"
        super().click_boton(boton_sumbit)

    def extract_pages_pagination(self):
          
        l_tag_years = super().extract_tags(xpath='.//select[@name="version"]/option')
        l_urls_years = ["https://sofifa.com/" + tag.get_attribute('value') for tag in l_tag_years]
        return l_urls_years
    
    def extract_pages_sub_pagination(self):

        l_tags_act_year = super().extract_tags(xpath='.//select[@name="roster"]/option') # .//h2//div[@class="dropdown"][2]/div/a[not(contains(text(), "World Cup"))] # Ojo con la actualizacion World Cup 2022...  # NO HACE FALTA HACER CLICK EN FLECHITA ANTES -->  #   boton_selec_act_fifa =  crawler.extract_tag(xpath='.//h2//div[@class="dropdown"][2]/a')  # Ver si hace click, tal vez ni hace falta  # crawler.click_boton(boton_selec_act_fifa)
        l_urls_act_year = ["https://sofifa.com/" + tag.get_attribute('value') for tag in l_tags_act_year]
        l_urls_act_year_sel = [l_urls_act_year[0], l_urls_act_year[-1]]  # Selecciono unicamente la primera y la ultima actualizacion
        return l_urls_act_year_sel
    
    def extract_fifa_name(self):
        fifa = super().extract_tag(xpath='.//select[@name="version"]/option[@selected]', text=True)  # Fifa 21
        return fifa
    
    def extract_player_data(self, tag):
        # Extraigo datos del jugador
        d_data = {}
        d_data['id_player'] = super().extract_tag(tag_inicial=tag,xpath='.//td[@data-col="pi"]', text=True)
        d_data['name'] = super().extract_tag(tag_inicial=tag, xpath='.//td[not(@class)]/a', attribute="data-tippy-content")
        d_data['age'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="ae"]', text=True)
        d_data['height'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="hi"]', text=True)
        d_data['preferred_foot'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pf"]', text=True)
        d_data['overall_rating'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="oa"]/em', text=True)
        d_data['potential'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pt"]/em', text=True)
        d_data['actual_team'] = super().extract_tag(tag_inicial=tag, xpath='.//td/a[starts-with(@href, "/team")]', text=True)
        d_data['value'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="vl"]', text=True)
        d_data['wage'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="wg"]', text=True)

        # Formateo campo height # e.g. 193cm / 6'4" --> 193
        d_data['height'] = d_data['height'].split('cm')[0]
        return d_data
    
    def click_next_page(self):

        # Clickeo en boton "Next" para recorrer todas las paginas
        boton_next = super().extract_tag(xpath='.//div[@class="pagination"]/a[text()="Next "]', sec_wait=self.SEC_WAIT_MAX)
        if super().click_boton(boton_next) is False:
            print('Ya no hay mas boton "Next". Es decir, ya no hay mas players en la league.')
            return False
        return True

    def extract_fifa_update_date(self):
        date_str = super().extract_tag(xpath='.//select[@name="roster"]/option[@selected]', text=True)  # e.g. May 16, 2023
        return date_str
        

def extract_players_sofifa(country, league, export=True):
    """
    Obtengo datos de players mediante scrapear sofifa.com

    # Cosas a mejorar:
    - Que no haga un get() a la fecha de actualizacion actual. Solo necesito la primera fecha de actualizacion y hacerle un get al final y listo.
    """
    print("\nCollecting data from sofifa.com...")
    # DEFINCION DE PARAMETROS & VARIABLES   
    df_player = pd.DataFrame(columns=['id_player', 'fifa', 'date', 'name', 'age', 'height', 'preferred_foot', 'overall_rating', 'potential', 'actual_team', 'value', 'wage'])  # usar d.keys() de headers... asi es automatico.. Ah no, pues extraigo algunos campos mas..
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver" # path_driver_exe = "./p2_data_understanding/collect_initial_data/chromedriver"
    crawler = SofifaCrawler(headless=False, path=path_driver_exe)  # Usar False (con True no funciona)

    # Ingreso a pagina de sofifa.com seleccionando como filtro los campos buscados (age, height, or, pot, valor_merc, etc)
    url_pagina = 'https://sofifa.com/players?showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg'
    crawler.driver.get(url_pagina)

    # Filtro listado de players segun la league del country que busco
    crawler.select_league_as_filter(country, league)  # El codigo funciona pero a veces falla en hacer click en la league que quiero elegir.
    crawler.click_submit()

    # Obtengo urls de las paginas de la paginacion (c/pagina es un año o fifa)
    l_urls_years = crawler.extract_pages_pagination()

    # POR FIFA (e.g. Fifa 23, fifa 22, fifa 21, ..., fifa 07)
    for url_year in l_urls_years:

        # Ingreso a pagina del año o fifa
        crawler.driver.get(url_year)

        fifa = crawler.extract_fifa_name()
        print(f" FIFA: {fifa} ".center(120, "+"))

        # Obtengo urls de las paginas de la paginacion (c/pagina es una actualizacion de un fifa)
        l_urls_act_year_sel = crawler.extract_pages_sub_pagination()
       
        # POR ACTUALIZACION EN DICHO FIFA (e.g. Jun 7, 2023;  Apr 17, 2023; etc)
        for url_year_act in l_urls_act_year_sel:

            # Ingreso a paging de la actualizacion
            crawler.driver.get(url_year_act)

            # Obtengo la date de actualizacion
            date_str = crawler.extract_fifa_update_date()
            print(f" Date de actualizacion: {date_str} ".center(120, "-"))
            
            n_jug_encontrados = 1
            is_boton_next_page = True

            # POR PAGINA CON LISTADO DE PLAYERS
            while is_boton_next_page:

                # Obtengo tags de players
                l_tag_players = crawler.extract_tags(xpath='.//main/article/table/tbody/tr')
                n_jug_encontrados += len(l_tag_players)
                print(f"Cantidad de players: {len(l_tag_players)}")
                progress_bar = tqdm(total=len(l_tag_players), ncols=80)  # Inicializo barra de progreso


                # Por jugador
                for tag in l_tag_players:

                    # Extraigo data
                    d_data = crawler.extract_player_data(tag)

                    # Agrego columnas ya extraidas
                    d_data['fifa'] = fifa
                    d_data['date'] = date_str
                    # print(d_data)

                    # Guardo los datos del jugador para dicho año
                    df_player = pd.concat([df_player, pd.DataFrame([d_data])], ignore_index=True)
                    progress_bar.update(1)

                print("Intento click en boton 'Next'")
                is_boton_next_page = crawler.click_next_page()
                progress_bar.close()
            print(f"Cantidad de players encontrados: {n_jug_encontrados}")

        # Exporto datos del fifa (Por seguridad)
        if export:
            df_player.to_excel(f'./p2_data_understanding/data/{country}/data_seg/per_season/df_player/{fifa}.xlsx', index=False)

    # Exporto dataset final
    if export:
        df_player.to_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx', index=False)

    # Cierro webdriver
    crawler.driver.close()
    return df_player

def prueba():
    # Seleccionar pais  
    country = "Argentina"
    liga = "Liga Profesional de Fútbol"

    df_player = extract_players_sofifa(country, liga, export=True)
    df_player.to_excel(f'/Users/nachomondino/Desktop/df_player_{country}.xlsx', index=False)

    """ A y B
    # Levanto datasets
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')

    # Determino competencias a extraer
    id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]
    l_comp = df_comp[(df_comp['id_country']==id_country) & (df_comp['is_cup']==0)]['competition_sofifa']

    df_player_concat = pd.DataFrame()

    # Por liga
    for liga in l_comp:

        # Extraigo competicion
        df_player = extract_players_sofifa(country, liga, export=True)

        # Guardo datos
        df_player_concat = pd.concat([df_player_concat, df_player], axis=0)
    
    df_player_concat.to_excel(f'/Users/nachomondino/Desktop/df_player_{country}.xlsx', index=False)
    """
    
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()