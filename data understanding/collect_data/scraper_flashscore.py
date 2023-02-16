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
        # options.add_argument("--headless")  # Hace que no se abra un web browser en tu compu
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

    def extract_fecha(self, item):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo

        try:
            # Extraigo campo
            fecha = item.find_element(By.XPATH, './div[@class="event__time"]').text
            return fecha

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion de la fecha")
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
            equipo1 = item.find_element(By.XPATH, './/div[starts-with(@class, "event__participant event__participant--home")]').text
            equipo2 = item.find_element(By.XPATH, './/div[starts-with(@class, "event__participant event__participant--away")]').text
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
            arbitro = self.driver.find_element(By.XPATH, './/div[@id="ficha-horario"]/span/span').text
            return arbitro

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion del campo")
            return None

    def extract_dts(self):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo campo
            entrenador_loc = self.driver.find_element(By.XPATH,
                                                      './/table[@id="formacion1"]//tr[@class="dttr"]/td[@colspan="2"]').text
            entrenador_vis = self.driver.find_element(By.XPATH,
                                                      './/table[@id="formacion2"]//tr[@class="dttr"]/td[@colspan="2"]').text
            return entrenador_loc, entrenador_vis

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion del campo")
            return None, None

    def extract_result(self, item):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Defino funcion que determina equipo ganador segun los goles que convirtio cada equipo
        equipo_gan = lambda ng1, ng2: "Local" if ng1 > ng2 else ("Empate" if ng1 == ng2 else "Visitante")

        # Intento extraer el campo
        try:
            # Extraigo goles de cada equipo para ver quien gano
            n_goles_equipo1 = item.find_element(By.XPATH, './div[@class="event__score event__score--home"]').text  # e.g. 1
            n_goles_equipo2 = item.find_element(By.XPATH, './div[@class="event__score event__score--away"]').text  # e.g. 2
            # return equipo_gan(ng1=n_goles_equipo1, ng2=n_goles_equipo2)
            return int(n_goles_equipo1), int(n_goles_equipo2)

        # Si falla la extraccion del resultado, retorno None
        except:
            print("Fallo la extraccion del resultado")
            return None

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

    def click_button_mas_part(self):
        """
        Click en siguiente jornada
        :return:
        """
        # Mientras exista el boton "Mosotrar mas partidos"
        try:
            # Click en boton "Mostrar mas partidos"
            tag_url = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, './/a[text()="Mostrar más partidos"]')))  # Hay dos botones "Mostrar mas partidos" pero selecciona el primero
            self.driver.get(tag_url.get_attribute('href'))
            print("Click en 'Mostrar mas partidos'")
        except:
            print("No hay mas boton 'Mostrar mas partidos' o bien fallo el click")

    def click_button_mas_part_2(self):  # ES NECESARIA LA FUNCION. ES DISTINTA A LA OTRA.
        """
        Click en siguiente jornada
        :return:
        """
        # Mientras exista el boton "Mosotrar mas partidos"
        while True:
            try:
                # Click en boton "Mostrar mas partidos"
                WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, './/a[text()="Mostrar más partidos"]'))).click()  # Hay dos botones "Mostrar mas partidos" pero selecciona el primero
                # sleep(3)  # No llega a cargarse la nueva pagina para buscar el boton?
                print("Click en 'Mostrar mas partidos'")
            except:
                print("No hay mas boton 'Mostrar mas partidos' o bien fallo el click")
                break

def format_date(fecha_string):
    """
    Convierte fecha del formato "Sábado 22 de Octubre 2022" a formato "dd/mm/yyyy", es decir, "14/01/2023"
    :param fecha_string: String. Fecha en formato "Sábado 22 de Octubre 2022"
    :return: String. Fecha y hora con formato "dd/mm/yyyy", por ejemplo, "Sábado 22 de Octubre 2022"
    """
    # PASO 1: SPLIT FECHA (STRING) EN ELEMENTOS (DIA, MES, AÑO, HORA Y MIN)
    n_año, n_mes, n_dia = split_date_string(fecha_string)  # Paso de string a un integer para el mes y otro para el dia

    # PASO 2: FORMATEO FECHA
    # Si la fecha tiene hora o minutos
    try:
        fecha_datetime = datetime.datetime(n_año, n_mes,
                                           n_dia)  # Convierto variable de clase 'str' a clase 'datetime.datetime'
        return fecha_datetime.strftime("%d/%m/%Y")  # Convierto el formato de datetime del default al formato deseado
    # Si la fecha no tiene hora o minutos
    except:
        print("Fallo formateo de fecha")
        return False

def split_date_string(fecha_string):
    """
    A partir de fecha en formato string, se obtiene los elementos (dia, mes y año) de la fecha por separado en formato
    entero
    :param fecha_string: String. Fecha en formato "Sábado 22 de Octubre 2022"
    :return: Integer & Integer & Integer. Dia, mes y año.
    """
    # Definicion de variables
    d = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, 'nov': 11,
         'dic': 12}

    # intento extraer dia, mes y año de string
    try:
        dia, n_dia, de, mes, n_año = fecha_string.split()  # Separo string original segun espacios para obtener elementos (mes y dia)
        n_año, n_mes, n_dia = int(n_año), d[mes[:3].lower()], int(n_dia)  # Convierto strings a integer
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
    n_temp = 1
    N_TEMPS = 8
    # df = pd.DataFrame(columns=['fecha', 'equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis', 'equipo_ganador'])
    df = pd.DataFrame(columns=['fecha', 'equipo_loc', 'equipo_vis', 'goles_loc', 'goles_vis'])
    crawler = CrawlerActions()  # Creo objeto de clase CrawlerActions()

    # Ingreso a pagina
    crawler.driver.get('https://www.flashscore.es/futbol/argentina/liga-profesional/archivo/')  # hasta que no se carga toda la pagina, no sigue...
    sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # Reject cookies
    crawler.reject_cookies()

    # POR PAGINA (TEMPORADA) DE PAGINACION
    for url_temp in crawler.get_urls_temporadas(N_TEMPS):

        # Ingreso a pagina de temporada
        crawler.driver.get(url_temp)
        print(" Nº Temporada: {} ".format(n_temp).center(120, "#"))

        # Click en boton "Mostrar mas partidos" (para ver todas las jornadas de la temporada)
        crawler.click_button_mas_part()
        sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))  # no deberia ser necesaria pues el click_button que sigue tiene un WebDriverWait...

        crawler.click_button_mas_part_2()

        # Extraigo items (partidos)
        l_items = crawler.driver.find_elements(By.XPATH, './/div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')

        # Por item (partido)
        for item in l_items:

            # Extraigo campos
            fecha = crawler.extract_fecha(item)
            equipo1, equipo2 = crawler.extract_teams(item)
            goles_loc, goles_vis = crawler.extract_result(item)

            '''
            # Ingreso a pagina de informacion del partido
            crawler.driver.get(url_info_part)

            arbitro = crawler.extract_arbitro()
            entrenador_loc, entrenador_vis = crawler.extract_dts()

            crawler.driver.back()
            '''

            # Formateo fecha
            # fecha_form = format_date(fecha)

            # GUARDADO DE DATOS EN DATAFRAME
            # l_data = [fecha_form, equipo1, equipo2, arbitro, entrenador_loc, entrenador_vis, equipo_ganador]
            l_data = [fecha, equipo1, equipo2, goles_loc, goles_vis]
            df.loc[len(df)] = l_data
            print(l_data)

        # Vuelvo a pagina donde se listan las temporadas
        crawler.driver.back()  # Salgo de pagina donde se desplegan las jornadas de la temporada
        crawler.driver.back()  # Salgo de pagina donde solo se muestra la ultima jornada de la temporada

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

    # Guardado de archivo excel en computadora
    df.to_excel('./liga_argentina_historico_2.xlsx', index=False)  # Cambiar la ruta del archivo


main()


