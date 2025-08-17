# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p2_data_understanding.collect_initial_data.web_scraping_selenium import Crawler
from tqdm import tqdm
from datetime import datetime, timedelta
import re
from utils.set_up_logging import logger
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
        self.SEC_WAIT_MIN = 0.8 * 5 # Espera para elementos que muchas veces no estan # con 0.2 fallaba extraccion de campos que si estaban como goals
        self.SEC_WAIT_MED = 1.5 * 5 # Espera para elementos que casi siempre estan
        self.SEC_WAIT_MAX = 5 # Espera para elementos que casi siempre estan
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
        super().select_option(
            option = f"{bet['id_team_home']}", # {bet['id_team_away']}
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
        ganador = bet['id_team_home'] if res == 1 else ("draw" if res==0 else bet['id_team_away'])
        print(res, ganador)

        super().select_option(
            option = ganador,
            xpath_flechita="//span[@aria-labelledby='select2-chosen-selection-container']/span[contains(@class, 'arrow')]",
            xpath_input=xpath_input
        )

        # Escribir descripcion (podria hacerlo con chatGPT...)
        des = f"The result of the match between {bet['id_team_home']} and {bet['id_team_away']} will be {ganador} with a {bet['prob_result_to_bet']*100}% of probability."
        super().accept_cookies_in_document_tag(
            xpath_document_parent=".//iframe[contains(@class, 'cke_wysiwyg_frame')]",
            xpath_boton=".//body[contains(@class, 'cke_editable')]"
            )
        super.fill_form(xpath_input=".//body[contains(@class, 'cke_editable')]", text=des) # Esta en un #document

        # Click en "Publish"
        tag_pub = super().extract_tag(xpath=".//input[@value='Publish']")
        super().click_boton(tag_boton=tag_pub)
        sleep(10)

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
    sleep(5)
    
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

        sleep(random.randint(2,5))

    crawler.driver.close()


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    hoy = datetime.now()

    # Load bets to publish
    df = pd.read_excel("./data/predicciones.xlsx")
    
    # Me quedo con aquellos en los que el stake > 0 y que aun no se hayan jugado
    df_filt = df[(df['stake_to_bet'] > 0) & (df['date'] > hoy)]

    print(df.shape)
    print(df_filt.shape)

    main(df_filt)