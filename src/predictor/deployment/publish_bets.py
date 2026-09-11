# Importo librerias
import pandas as pd
from predictor.data_understanding.web_scraping_selenium import Crawler
from tqdm import tqdm
from datetime import datetime, timedelta
import re
from predictor.utils.set_up_logging import logger
from time import sleep
import random
import os
from dotenv import load_dotenv


class StakeHunterCrawler(Crawler):
    """
    Desarrollo extracciones de distintos campos en funciones de manera de poder usar estas funciones para scraper
    normal, scraper de fields especificos poro falla y scraper de proximos partidos...
    Contiene todo los xpath.
    """
    def __init__(self, headless: bool = True, browser: str = "Chrome", verbose: int = 0):
        
        super().__init__(headless, browser)
        self.child_driver = self.driver
        self.SEC_WAIT_MIN = 2 # Espera para elementos que muchas veces no estan # con 0.2 fallaba extraccion de campos que si estaban como goals
        self.SEC_WAIT_MED = 4 # Espera para elementos que casi siempre estan
        self.SEC_WAIT_MAX = 6 # Espera para elementos que casi siempre estan
        self.verbose =  verbose # Para imprimir el funcionamiento de cada funcion y poder hacer pruebas...

    def league_to_search(self, bet):
        """
        Defino nombre de la liga a buscar en Stake Hunter tal y como aparece en este.
        Mi df puede tener ≠ nombre para una competition que StakeHunter
        """
        d = {
            "laliga": "la liga"
        }

        if bet['competition'] in d.keys():
            comp = d[bet['competition']]
        else:
            comp = bet['competition']

        return f"{bet['country']} - {comp}"

    def format_team_name(self, team):
        
        d_teams = {'wolves': 'wolverhampton'}

        list_names = team.split(" ") # Evito nombres compuestos que hacen fallar como "manchester utd", "r oviedo"
        
        max_lenght = 0
        for elem in list_names:
            elem_lenght = len(elem)
            if elem_lenght > max_lenght:
                search = elem
                max_lenght = elem_lenght
        
        if search in d_teams.keys():
            search = d_teams[search]

        return search
    
    def load_bet(self, bet):
        
        xpath_input = "//span[contains(@class, 'search--dropdown')]/input"

        # Click en Sport Football ("Pinnacle Tip")
        tag_football = super().extract_tag(xpath='.//div[@class="select-sport"]/div[contains(@class, "soccer")]')
        super().click_boton(tag_boton=tag_football)
        sleep(2)

        # Seleccionar Liga desde input multiple choices.
        super().select_option(
            option = self.league_to_search(bet),
            xpath_flechita="//span[@aria-labelledby='select2-chosen-league-container']/span[contains(@class, 'arrow')]",
            xpath_input=xpath_input
        )

        # Seleccionar event (partido)
        home_team = self.format_team_name(bet['id_team_home'])
        away_team = self.format_team_name(bet['id_team_away'])

        super().select_option(
            option=home_team,
            xpath_flechita="//span[@aria-labelledby='select2-chosen-event-container']/span[contains(@class, 'arrow')]",
            xpath_input = xpath_input
        )

        # Definir stake
        stake_to_bet = round(bet['stake_to_bet'], 0)
        stake_to_bet = max(0, min(10, stake_to_bet))
        print(stake_to_bet)

        tag_stake = super().extract_tag(xpath=f'.//div[@class="stake-list"]/*[text()={stake_to_bet}]')
        super().click_boton(tag_boton=tag_stake)

        # Definir bet type --> "Money line - Game"
        super().select_option(
            option = "Money line - Game",
            xpath_flechita = "//span[@aria-labelledby='select2-chosen-type-container']/span[contains(@class, 'arrow')]",
            xpath_input=xpath_input
        )

        # Definir ganador
        res = bet['result_to_bet']
        ganador = home_team if res == 1 else ("draw" if res==0 else away_team)
        print(res, ganador)

        super().select_option(
            option = ganador,
            xpath_flechita="//span[@aria-labelledby='select2-chosen-selection-container']/span[contains(@class, 'arrow')]",
            xpath_input=xpath_input
        )

        # Escribir descripcion (podria hacerlo con chatGPT...)
        '''
        des = f"The result of the match between {bet['id_team_home']} and {bet['id_team_away']} will be {ganador} with a {bet['prob_result_to_bet']*100}% of probability."
        super().accept_cookies_in_document_tag(
            xpath_document_parent=".//iframe[contains(@class, 'cke_wysiwyg_frame')]",
            xpath_boton=".//body[contains(@class, 'cke_editable')]"
            )
        super.fill_form(xpath_input=".//body[contains(@class, 'cke_editable')]", text=des) # Esta en un #document
        '''

        # Click en "Publish"
        tag_pub = super().extract_tag(xpath=".//input[@value='Publish']")
        super().click_boton(tag_boton=tag_pub)
        sleep(random.randint(5, 7))

def main(df):
    
    # Defino variables
    crawler = StakeHunterCrawler(headless=False)
    load_dotenv()

    # Ingresar a sitio web
    url = "https://stakehunters.com/"
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...

    # Click en "Sing in"
    tag_boton = crawler.extract_tag(xpath="//a[@href='/auth']") # attribute="href"
    crawler.click_boton(tag_boton)
    sleep(random.randint(3,5))
    
    # Ingreso con credenciales    
    crawler.login_website(
        user=os.getenv('USER_SH'),
        password=os.getenv('PASS_SH'),
        xpath_user="//form[@class='auth-form']//input[@name='username']",
        xpath_pass="//form[@class='auth-form']//input[@name='password']",
        xpath_boton_login="//form[@class='auth-form']//input[@name='login']"
    )
    
    # Click en "Create Tips" 
    tag_boton_ct = crawler.extract_tag(xpath=".//a[@href='/create-tip']")
    crawler.click_boton(tag_boton=tag_boton_ct)

    # Cargar bets
    for idx, match in df.iterrows():
        
        logger.info(match)

        crawler.load_bet(match)    
        
        # Recarga la página
        crawler.driver.refresh()
        sleep(2)

    crawler.driver.close()


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    l_countries = [148]
    hoy = datetime.now()

    # Load bets to publish
    df = pd.read_excel("./data/_shared/predictions/predicciones.xlsx", index_col=0)
    logger.info(df)
    print(df.shape)

    # Filtro por pais
    df = df[df['id_country'].isin(l_countries)]

    # Filtro por ids a evitar
    # l_ids_to_avoid = ['4EQGmZBs', '0fab0Cem']
    l_ids_to_sel = ['IJLdrvz2']
    # df = df[~df.index.isin(l_ids_to_avoid)]
    df = df[df.index.isin(l_ids_to_sel)]

    # Me quedo con aquellos en los que el stake > 0 y que aun no se hayan jugado
    df_filt = df[(df['stake_to_bet'] > 0.5) & (df['date'] > hoy)]
    print(df_filt.shape)

    if len(df_filt) > 0:
        main(df_filt)
    else:
        logger.info("Se evito cargar apuestas porque no hay proximos partidos con stake > 0.")