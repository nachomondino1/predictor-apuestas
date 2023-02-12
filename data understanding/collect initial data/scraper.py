# Importo librerias
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re
import random
from time import sleep
import pandas as pd
import datetime


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
        options.add_argument("--headless")  # Hace que no se abra un web browser en tu compu
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

    def accept_cookies(self):
        """
        Click en aceptar cookies
        """
        # Ubico el botón "Aceptar cookies" y lo clikeo
        try:
            WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, './/button/span[text()="ACEPTO"]'))).click()
            print("Acepte cookies correctamente")
        except:
            print("Fallo click en boton aceptar cookies")

    def extract_items(self):
        try:
            l_items = WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, './/div[@id="col-resultados"]//table//tr[@class="vevent " or @class="vevent impar"]')))
            print("Cantidad de items: {}".format(len(l_items)))
            return l_items
        except:
            print("Fallo extraccion de items")

    def extract_fecha(self, item):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo campo
            tag_fecha = WebDriverWait(item, 10).until(EC.presence_of_element_located((By.XPATH, './/td[@class="fecha"]')))
            fecha = tag_fecha.text[:tag_fecha.text.find("\n")]
            return fecha

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion del campo")
            return None

    def extract_teams(self, item):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo campo
            equipo1 = item.find_element(By.XPATH, './/td[@class="equipo1"]').text
            equipo2 = item.find_element(By.XPATH, './/td[@class="equipo2"]').text
            return equipo1, equipo2

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion del campo")
            return None, None

    def extract_result(self, item, equipo1, equipo2):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo resultado
            resultado = item.find_element(By.XPATH, './/td[@class="rstd"]/a').text  # (e.g. 4-0)

            # Seeparo goles de cada equipo para ver quien gano
            n_goles_equipo1, n_goles_equipo2 = resultado.split("-")

            # Si gano el equipo1
            if n_goles_equipo1 > n_goles_equipo2:
                return "Local"

            # Si emparaton
            elif n_goles_equipo1 == n_goles_equipo2:
                return "Empate"

            # Si gano el equipo2
            else:
                return "Visitante"

        # Si falla la extraccion del resultado, retorno None
        except:
            print("Fallo la extraccion del resultado")
            return None

    def get_pagination_temporada(self):
        # Definicion de variables
        l_temporadas_new = []

        try:
            l_tag_temporadas = self.driver.find_elements(By.XPATH, './/div[@id="titular"]//div[@class="bar_jornada"]//li/a')

            # Por tag (c/u contiene la url de una jornada)
            for tag in l_tag_temporadas:
                l_temporadas_new.append(tag.get_attribute('href'))
        except:
            print("Fallo extraccion de links de las paginas de las jornadas")

        print("Nº de Temporadas", len(l_temporadas_new))
        return l_temporadas_new

    def get_pagination_jornada(self):
        # Definicion de variables
        l_jornadas_new = []

        try:
            l_tag_jornadas = self.driver.find_elements(By.XPATH, './/div[@id="col-resultados"]//div[@class="bar_jornada"]//li/a')

            # Por tag (c/u contiene la url de una jornada)
            for tag in l_tag_jornadas:
                l_jornadas_new.append(tag.get_attribute('href'))
        except:
            print("Fallo extraccion de links de las paginas de las jornadas")

        print("Nº de jornadas", len(l_jornadas_new))
        return l_jornadas_new

def format_date(fecha_string):
    """
    Convierte fecha del formato "Jan 14" y hora "8:00 PM" a formato "dd/mm/yyyy hh:mm", es decir,
    "14/01/2023 19:30"
    :param fecha_string: String. Fecha en formato "Jan 14"
    :return: String. Fecha y hora con formato "dd/mm/yyyy hh:mm", por ejemplo, "14/01/2023 20:00"
    """
    # PASO 1: SPLIT FECHA (STRING) EN ELEMENTOS (DIA, MES, AÑO, HORA Y MIN)
    n_año, n_mes, n_dia = split_date_string(fecha_string)  # Paso de string a un integer para el mes y otro para el dia

    # PASO 2: FORMATEO FECHA
    # Si la fecha tiene hora o minutos
    try:
        fecha_datetime = datetime.datetime(n_año, n_mes, n_dia) # Convierto variable de clase 'str' a clase 'datetime.datetime'
        return fecha_datetime.strftime("%d/%m/%Y")  # Convierto el formato de datetime del default al formato deseado
    # Si la fecha no tiene hora o minutos
    except:
        print("Fallo formateo de fecha")

def split_date_string(fecha_string):
    """
    A partir de fecha en formato string, se obtiene los elementos (dia, mes y año) de la fecha por separado en formato
    entero
    :param fecha_string: String. Fecha en formato "22 Oct 19"
    :return: Integer & Integer & Integer. Dia, mes y año.
    """
    # Definicion de variables
    d = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, 'nov': 11, 'dic': 12}

    # intento extraer dia, mes y año de string
    try:
        n_dia, mes, n_año = fecha_string.split()  # Separo string original segun espacios para obtener elementos (mes y dia)
        n_año, n_mes, n_dia = int("20"+n_año), d[mes.lower()], int(n_dia)  # Convierto strings a integer
        return n_año, n_mes, n_dia
    # Si falla extraccion de dia, mes y año de string
    except:
        print("Fallo la conversion de la fecha {} de string a integer".format(fecha_string))  # Mensaje de aviso
        return None, None, None


def main():
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    SLEEP_MIN, SLEEP_MAX = 1, 3  # Tiempos de espera luego de clicks para humanizar programa
    N_TEMPS = 5
    df = pd.DataFrame(columns=['fecha', 'equipo_loc', 'equipo_vis', 'equipo_ganador'])
    crawler = CrawlerActions()  # Creo objeto de clase CrawlerActions()

    # Ingreso a pagina
    crawler.driver.get('https://www.resultados-futbol.com/primera_division_argentina2020')  # hasta que no se carga toda la pagina, no sigue...
    sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # Aceptar cookies en pop up
    crawler.accept_cookies()

    # POR PAGINA (TEMPORADA) DE PAGINACION
    for temporada in crawler.get_pagination_temporada()[1:N_TEMPS]:  # ver que hacer con la temporada actual

        # Ingreso a pagina de temporada
        crawler.driver.get(temporada)
        print(" Temporada ".center(120, "#"))
        i = 1

        # POR SUBPAGINA (JORNADA)
        for jornada in crawler.get_pagination_jornada():

            # Ingreso a pagina de jornada
            crawler.driver.get(jornada)
            print(" Jornada Nº {}".format(i).center(120, "-"))
            i += 1

            # Obtengo lista de items de pagina
            l_items = crawler.extract_items()

            # POR ITEM (Partido)
            for item in l_items:

                # Extraigo campos
                fecha = crawler.extract_fecha(item)
                equipo1, equipo2 = crawler.extract_teams(item)
                resultado = crawler.extract_result(item, equipo1, equipo2)

                # Formateo fecha
                fecha_form = format_date(fecha)

                # GUARDADO DE DATOS EN DATAFRAME
                l_data = [fecha_form, equipo1, equipo2, resultado]
                df.loc[len(df)] = l_data
                print(l_data)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    
    # Guardado de archivo excel en computadora
    df.to_excel('./liga_argentina_historico.xlsx', index=False)  # Cambiar la ruta del archivo

main()