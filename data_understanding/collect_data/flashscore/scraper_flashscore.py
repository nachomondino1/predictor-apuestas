# Importo librerias
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import random
from time import sleep
import pandas as pd
import re

class CrawlerActions():
    """ It contains all the actions that the bot can perform from accepting cookies to clicking on the next one. """

    def __init__(self):
        """Initialize attributes of the parent class."""
        self.driver = self.inicialize_driver()

    def inicialize_driver(self):
        """
        Inicializa un chrome driver automatico
        :return: Chrome driver automatico
        """
        # Defino opciones del webdriver
        options = webdriver.ChromeOptions()
        options.add_argument("start-maximized")
        options.add_argument("enable-automation")
        # options.add_argument("--headless")  # Hace que no se abra un web browser en tu compu. Falla en este caso...
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-browser-side-navigation")
        options.add_argument("--disable-gpu")
        options.add_argument("--incognito")
        options.add_argument("--disable-popup-blocking")

        # Inicializo el webdriver (Defino a Chrome como Web Browser)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        return driver

    def reject_cookies(self):
        """
        Click en rechazar cookies
        """
        # Ubico el botón "Rechazar cookies" y lo clikeo
        try:
            WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, './/button[@id="onetrust-reject-all-handler"]'))).click()
            print("Rechace las cookies correctamente")
        except:
            print("Fallo click en boton rechazar cookies")

    def extract_fecha(self):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo campo
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, './/div[@class="duelParticipant__startTime"]')))
            fecha = self.driver.find_element(By.XPATH, './/div[@class="duelParticipant__startTime"]').text
            return fecha

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion de la fecha")
            return None

    def extract_teams(self):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo campo
            equipo1 = self.driver.find_element(By.XPATH, './/div[starts-with(@class, "duelParticipant__home")]').text
            equipo2 = self.driver.find_element(By.XPATH, './/div[starts-with(@class, "duelParticipant__away")]').text
            return equipo1, equipo2

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion de los equipos")
            return None, None

    def extract_arbitro(self):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo campo
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, './/div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span')))  # Hay dos botones "Mostrar mas partidos" pero selecciona el primero
            arbitro = self.driver.find_element(By.XPATH, './/div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span').text
            return arbitro

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion del arbitro")
            return None

    def extract_dts(self):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Dentro de la pagina de informacion del partido, entro a seccion "Alineaciones"
            boton = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, './/div[@class="tabs tabs__detail--nav"]//a[text()="Alineaciones"]')))  # Hay dos botones "Mostrar mas partidos" pero selecciona el primero
            boton.click()

            # Extraigo dts
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH,'.//div[@class="lf__lineUp"]/div[@class="section"][last()]')))  # Hay dos botones "Mostrar mas partidos" pero selecciona el primero
            dt_loc = self.driver.find_element(By.XPATH, './/div[@class="lf__lineUp"]/div[@class="section"][last()]//div[@class="lf__participant "]').text
            dt_vis = self.driver.find_element(By.XPATH, './/div[@class="lf__lineUp"]/div[@class="section"][last()]//div[@class="lf__participant lf__isReversed"]').text
            return dt_loc, dt_vis

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo click en seccion 'Alineaciones' para extraer los entrenadores de cada equipo")
            return None, None

    def extract_result(self):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo goles de cada equipo para ver quien gano
            n_goles = self.driver.find_element(By.XPATH, './/div[@class="detailScore__wrapper"]').text  # e.g. 1 \n - \n 2
            n_goles_equipo1, sep, n_goles_equipo2 = n_goles.split("\n")
            return int(n_goles_equipo1), int(n_goles_equipo2)

        # Si falla la extraccion del resultado, retorno None
        except:
            print("Fallo la extraccion del resultado")
            return None, None

    def get_urls_temporadas(self, n_temp):
        """
        Extrar URLs de las ultimas temporadas del futbol argentino
        :param n_temp: Integer. Cantidad de temporadas a las cuales extraer su url.
        :return: Lista de urls de las ultimas <n_temp> temporadas
        """
        # Definicion de variables
        l_urls_temporadas = []

        try:
            # Obtengo los tags que contienen urls de temporadas
            l_tag_temporadas = self.driver.find_elements(By.XPATH,'.//section[@id="tournament-page-archiv"]//div[@class="archive__row"]/div[@class="archive__season"]/a')

            # Por tag (c/u contiene la url de una temporada)
            for i in range(n_temp):
                # Guardo url de temporada
                l_urls_temporadas.append(l_tag_temporadas[i].get_attribute('href'))

        except:
            print("Fallo extraccion de las urls de las temporada")
        return l_urls_temporadas

    def is_button_mas_part(self):
        """
        Click en siguiente jornada
        :return:
        """
        try:
            # Busco boton "Mostrar mas partidos"
            WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, './/a[text()="Mostrar más partidos"]')))
            return True
        except:
            return False

    def click_button_mas_part(self):
        """
        Click en siguiente jornada
        :return:
        """
        try:
            # Click en boton "Mostrar mas partidos"
            WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, './/a[text()="Mostrar más partidos"]'))).click()  # no se por que a veces falla el click. De todos modos hace los necesarios gracias a is_button_mas_part()
            print("Click en 'Mostrar mas partidos'")
        except:
            print("No hay mas boton 'Mostrar mas partidos' o bien fallo el click")

    def click_info_part(self, id_part):
        """
        Click en informacion del partido
        :return:
        """
        try:
            # Construyo url de info del partido a partir del id
            url = 'https://www.flashscore.es/partido/{}/#/resumen-del-partido/resumen-del-partido'.format(id_part)
            print(url)

            # Ingreso a info_part
            self.driver.get(url)
        except:
            print("Fallo click en pagina de informacion del partido")

def main():
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    SLEEP_MIN, SLEEP_MAX = 1, 3  # Tiempos de espera luego de clicks para humanizar programa
    N_TEMPS = 8
    df = pd.DataFrame(columns=['id', 'fecha','equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis', 'goles_loc', 'goles_vis'])
    crawler = CrawlerActions()  # Creo objeto de clase CrawlerActions()
    l_ids = []

    # Ingreso a pagina
    crawler.driver.get('https://www.flashscore.es/futbol/argentina/liga-profesional/archivo/')  # hasta que no se carga toda la pagina, no sigue...
    sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # Reject cookies
    crawler.reject_cookies()

    # POR PAGINA (TEMPORADA) DE PAGINACION
    for url_temp in crawler.get_urls_temporadas(N_TEMPS):

        # Ingreso a pagina de temporada
        crawler.driver.get(url_temp)
        print(" Temporada: ".center(120, "#"))

        # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
        crawler.click_button_mas_part()

        # Cargo todas las jornadas (al pprio solo aparecen algunas jornadas)
        i=1
        while crawler.is_button_mas_part():
            crawler.click_button_mas_part()
            print(i)
            i+=1

        # Extraigo items (partidos)
        l_items = crawler.driver.find_elements(By.XPATH, './/div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')

        # Por item (partido)
        for item in l_items:

            # Extraigo id del item para poder extraer los campos luego
            id = item.get_attribute('id')  # (e.g. "g_1_fshvzbls")
            l_ids.append(id[id.rfind('_')+1:])  # (e.g. "fshvzbls")

    # EXTRACCION DE CAMPOS
    # Por item (c/u identificado con un id)
    for i in range(len(l_ids)):

        # Ingreso a pagina de informacion del partido
        crawler.click_info_part(l_ids[i])

        # Extraigo campos
        fecha = crawler.extract_fecha()
        equipo1, equipo2 = crawler.extract_teams()
        goles_loc, goles_vis = crawler.extract_result()
        arbitro = crawler.extract_arbitro()
        dt_loc, dt_vis = crawler.extract_dts()

        # GUARDADO DE DATOS EN DATAFRAME
        l_data = [l_ids[i], fecha, equipo1, equipo2, arbitro, dt_loc, dt_vis, goles_loc, goles_vis]
        df.loc[len(df)] = l_data
        print(l_data)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

    # Guardado de archivo excel en computadora
    df.to_excel('./liga_argentina_historico.xlsx', index=False)  # Cambiar la ruta del archivo


main()


