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
        input_league = super().extract_tag(xpath='.//form[@class="pjax-form" and (@action="/players" or @action="/teams")]//input[@placeholder="Leagues"]')
        input_league.send_keys(league)

        # Selecciona la opcion segun name del country con send_keys
        ## Posibles leagues segun nuestra busqueda
        l_posibles_leagues = super().extract_tags(xpath='.//form[@class="pjax-form" and (@action="/players" or @action="/teams")]//input[@placeholder="Leagues"]//parent::div//following-sibling::div//div[starts-with(@class, "choices-item")]', sec_wait=self.SEC_WAIT_MAX)  # a veces crashea el click pero funciona
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
    
    def extract_temporary_player_data(self, tag):
        # Extraigo datos del jugador
        d_data = {}
        d_data['id_player'] = super().extract_tag(tag_inicial=tag,xpath='.//td[@data-col="pi"]', text=True)
        d_data['age'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="ae"]', text=True)
        d_data['overall_rating'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="oa"]/em', text=True)
        d_data['potential'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pt"]/em', text=True)
        url_team = super().extract_tag(tag_inicial=tag, xpath='.//td/a[starts-with(@href, "/team/")]', attribute='href')
        d_data['value'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="vl"]', text=True)
        d_data['wage'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="wg"]', text=True)
        int_reputation_str = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="ir"]', text=True)
        d_data['int_reputation'] = int(int_reputation_str.strip()) if int_reputation_str.strip().isdigit() else int_reputation_str
        d_data['id_team'] = extract_id_from_url_team(url_team)
        return d_data
    
    def extract_timeless_player_data(self, tag):
        # Extraigo datos del jugador
        d_data = {}
        d_data['player_name'] = super().extract_tag(tag_inicial=tag, xpath='.//td[not(@class)]/a', attribute="data-tippy-content")
        d_data['height'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="hi"]', text=True)
        d_data['preferred_foot'] = super().extract_tag(tag_inicial=tag, xpath='.//td[@data-col="pf"]', text=True)
        d_data['url_player'] = super().extract_tag(tag_inicial=tag, xpath='.//td/a[starts-with(@href, "/player/")]', attribute='href')

        # Formateo campo height # e.g. 193cm / 6'4" --> 193
        d_data['height'] = d_data['height'].split('cm')[0]
        return d_data
    
    def extract_team_data(self):

        # team_name
        team_name = super().extract_tag(xpath='.//h1[@class="ellipsis"]', text=True)

        # rival_team
        rival_team = super().extract_tag(xpath='.//label[text()="Rival team"]/following-sibling::a')
        rival_team_name = rival_team.text
        rival_team_url = rival_team.get_attribute('href')
        id_rival_team =  extract_id_from_url_team(rival_team_url)

        # home_stadium
        home_stadium = super().extract_tag(xpath='.//label[text()="Home stadium"]/parent::li', text=True)  # attribute="innerText" no funcionó
        if isinstance(home_stadium, str):
            home_stadium = home_stadium.replace("Home stadium ", "")

        # international_prestige
        international_prestige = super().extract_tag(xpath='.//label[text()="International prestige"]/following-sibling::em', text=True)

        # domestic_prestige
        domestic_prestige = super().extract_tag(xpath='.//label[text()="Domestic prestige"]/following-sibling::em', text=True)

        d_new_row = {'team_name': team_name, 'home_stadium': home_stadium, 'international_prestige': international_prestige, 'domestic_prestige': domestic_prestige, 'id_rival_team': id_rival_team, 'rival_team_name': rival_team_name}
        return d_new_row

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
    

def extract_players(id_country, country, id_competition, league, export=True):

    # Definicion de variables
    df_player, df_player_fifa = pd.DataFrame(), pd.DataFrame()
    # path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver" # path_driver_exe = "./p2_data_understanding/collect_initial_data/chromedriver"
    crawler = SofifaCrawler(headless=False)  # Usar False (con True no funciona)
    league_form = league.lower().replace(" ", "-")

    # Ingreso a pagina de sofifa.com seleccionando como filtro los campos buscados (age, height, or, pot, valor_merc, etc)
    url_pagina = 'https://sofifa.com/players?&showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg&showCol%5B%5D=ir' # 'https://sofifa.com/players?showCol%5B%5D=pi&showCol%5B%5D=ae&showCol%5B%5D=hi&showCol%5B%5D=pf&showCol%5B%5D=oa&showCol%5B%5D=pt&showCol%5B%5D=vl&showCol%5B%5D=wg'
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
            
            is_boton_next_page = True

            # POR PAGINA CON LISTADO DE PLAYERS
            while is_boton_next_page:

                # Obtengo tags de players
                l_tag_players = crawler.extract_tags(xpath='.//main/article/table/tbody/tr')
                progress_bar = tqdm(total=len(l_tag_players), ncols=80)  # Inicializo barra de progreso

                # Por jugador
                for tag in l_tag_players:

                    # Extraigo data temporal
                    d_row_player_temp = crawler.extract_temporary_player_data(tag)
                    id_player = d_row_player_temp['id_player']

                    # Si el id_player es nuevo:
                    if id_player not in df_player.index:
                        # Extraigo data atemporal
                        d_row_player = crawler.extract_timeless_player_data(tag)
                        df_player = pd.concat([df_player, pd.DataFrame(d_row_player, index=[id_player])])
                        
                    # Agrego columnas ya extraidas
                    d_row_player_temp['fifa'] = fifa
                    d_row_player_temp['date'] = date_str
                    d_row_player_temp['id_country'] = id_country
                    d_row_player_temp['id_competition'] = id_competition

                    # Guardo los datos del jugador para dicho año
                    df_player_fifa = pd.concat([df_player_fifa, pd.DataFrame(d_row_player_temp, index=[len(df_player_fifa)])])
                    progress_bar.update(1)

                progress_bar.close()
                print("Intento click en boton 'Next'")
                is_boton_next_page = crawler.click_next_page()

        # Exporto datos del fifa (Por seguridad)
        if export:
            df_player.to_excel(f'./p2_data_understanding/data/{country}/data_seg/per_season/df_player_sofifa/{fifa}_{league_form}.xlsx', index=True)
            df_player_fifa.to_excel(f'./p2_data_understanding/data/{country}/data_seg/per_season/df_player_fifa_sofifa/{fifa}_{league_form}.xlsx', index=False)

    # Exporto dataset final
    if export:
        df_player.to_excel(f'./p2_data_understanding/data/{country}/data_seg/per_competition/df_player_sofifa/{league_form}.xlsx', index=True)
        df_player_fifa.to_excel(f'./p2_data_understanding/data/{country}/data_seg/per_competition/df_player_fifa_sofifa/{league_form}.xlsx', index=False)

    # Cierro webdriver
    crawler.driver.close()
    return df_player, df_player_fifa

def extract_teams(id_country, country, league):
    """
    Extraccion de equipos de la liga de un pais.
    """
    # Definicion de variables
    df_teams = pd.DataFrame()
    crawler = SofifaCrawler(headless=False)  # Usar False (con True no funciona)

    # Ingreso a pagina de sofifa.com seleccionando como filtro los campos buscados (age, height, or, pot, valor_merc, etc)
    url_pagina = "https://sofifa.com/teams"
    crawler.driver.get(url_pagina)

    # Filtro listado de players segun la league del country que busco
    crawler.select_league_as_filter(country, league)  # El codigo funciona pero a veces falla en hacer click en la league que quiero elegir.
    crawler.click_submit()

    # Obtengo urls de las paginas de la paginacion (c/pagina es un equipo)
    l_tags_a = crawler.extract_tags(xpath='.//main/article/table//td[@class="s20"]/a[starts-with(@href, "/team/")]')
    # print(f'Cantidad de equipos: {len(l_tags_a)}')
    l_url_teams = [tag.get_attribute('href') for tag in l_tags_a]
    print(f'Cantidad de equipos: {len(l_url_teams)}')

    # Por equipo
    for url_team in l_url_teams:

        # Ingreso a pagina del equipo
        crawler.driver.get(url_team)

        # Extraccion de campos
        id_team = extract_id_from_url_team(url_team)
        d_new_row = crawler.extract_team_data()
        d_new_row.update({'url_team': str(url_team), 'id_country': id_country})
        print(d_new_row)

        # Guardo datos
        df_teams = pd.concat([df_teams, pd.DataFrame(d_new_row, index=[id_team])])

        # Salgo de la pagina del equipo (podria prescindir de esto)
        crawler.driver.back()

    crawler.driver.close()
    return df_teams

def extract_id_from_url_team(url_team):
    """
    Extra el id del equipo de la url de dicho equipo.
    """
    # Convierto url a string para poder usar replace()
    str_url_team = str(url_team)

    # Obtengo posicion inidial y final
    pos_ini = str_url_team.find("/team/") + len("/team/")
    resto_url = str_url_team[pos_ini:]
    pos_fin = pos_ini + resto_url.find('/')

    # Extraigo id con pos_ini y pos_fin
    id_team = str_url_team[pos_ini: pos_fin]
    return id_team

def extract_all_teams():
       # Se extrae una vez y listo o no?
    df_teams_concat = pd.DataFrame()

    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx', index_col=0)
    df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx', index_col="id_competition")
    # print(df_comp.head(1))
    l_id_countries = [6]  # l_countries = df_comp['id_country'].unique()

    # Por pais  
    for id_country in l_id_countries:

        # Obtengo ligas del pais
        country = df_countries[df_countries.index == id_country]['country_name'].values[0]
        l_leagues = df_comp[(df_comp['id_country']==id_country) & (df_comp['is_cup']==0)]['competition_sofifa'].values
        print(country)
        print(l_leagues)

        # Por competencia
        for league in l_leagues :
            print(league)
        
            # Obtengo equipos del pais
            df_teams = extract_teams(id_country, country, league)
            df_teams.to_excel(f'/Users/nachomondino/Desktop/df_teams_{country}_{league}.xlsx', index=True)

            # Guardo datos 
            df_teams_concat = pd.concat([df_teams_concat, df_teams], axis=0)
            print(df_teams_concat.shape)
    
    df_teams_concat.to_excel('/Users/nachomondino/Desktop/df_teams_prueba.xlsx', index=True)

def prueba():
    # Seleccionar pais  
    id_country = 48 # 6
    id_competition = 101 # 106
    country = "England" # 'England'
    liga = 'Championship' # "Liga Profesional de Fútbol" #'Championship'
    # l_leagues = ['Premier League', 'Championship']

    # for liga in l_leagues:
    df_player, df_player_fifa = extract_players(id_country, country, id_competition, liga, export=True)
    df_player.to_excel(f'/Users/nachomondino/Desktop/df_player_{country}_{liga}.xlsx', index=True)
    df_player_fifa.to_excel(f'/Users/nachomondino/Desktop/df_player_fifa_{country}_{liga}.xlsx', index=False)
    
    # extract_all_teams()

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()