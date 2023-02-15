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

    def extract_fecha_2(self, n_part):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        try:
            # Extraigo campo
            l_items = WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, './/div[@id="fixturein"]//td[@class="game-info"]/a/ancestor::tr')))  # Uso parent para evitar los tr que no corresponden a items
            item = l_items[n_part]
            fecha = item.find_element(By.XPATH, './preceding-sibling::tr[@class="diapart"][1]').text
            return fecha

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion de la fecha")
            return None

    def extract_fecha(self):
        """
        Extrae field
        :param <param_name>: <param description>
        :return: String con field, en caso contrario, None
        """
        # Intento extraer el campo
        l_dias = ['Viernes', 'Sábado', 'Domingo', "Lunes", 'Martes', 'Miércoles', 'Jueves', ]

        try:
            '''
            # Alternativa (PROBAR!)
            for dia in l_dias:
                xpath = './/*[starts-with(text(), "{}")]'.format(dia)
                WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath)))
                fecha = self.driver.find_element(By.XPATH, xpath).text
            '''
            # Extraigo campo
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, './/div[@id="ficha-horario"]')))
            fecha = self.driver.find_element(By.XPATH, './/div[@id="ficha-horario"]').text
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
            equipo1 = self.driver.find_element(By.XPATH, './/table[@id="formacion1"]//tr[1]').text
            equipo2 = self.driver.find_element(By.XPATH, './/table[@id="formacion2"]//tr[1]').text
            return equipo1, equipo2

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion del campo")
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
            entrenador_loc = self.driver.find_element(By.XPATH, './/table[@id="formacion1"]//tr[@class="dttr"]/td[@colspan="2"]').text
            entrenador_vis = self.driver.find_element(By.XPATH, './/table[@id="formacion2"]//tr[@class="dttr"]/td[@colspan="2"]').text
            return entrenador_loc, entrenador_vis

        # Si falla la extraccion del campo, retorno None
        except:
            print("Fallo la extraccion del campo")
            return None, None

    def extract_result(self):
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
            n_goles_equipo1 = self.driver.find_element(By.XPATH, './/div[@id="ficha-resultado1"]').text  # e.g. 1
            n_goles_equipo2 = self.driver.find_element(By.XPATH, './/div[@id="ficha-resultado2"]').text  # e.g. 2

            return equipo_gan(ng1=n_goles_equipo1, ng2=n_goles_equipo2)

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
            l_tag_temporadas = self.driver.find_elements(By.XPATH, './/div[@id="historneos"]//td[@style="background:green"]/a')

            # Por tag (c/u contiene la url de una temporada)
            for i in range(n_temp):

                # Guardo url de temporada
                tag = l_tag_temporadas[i]
                l_urls_temporadas.append(tag.get_attribute('href'))

        except:
            print("Fallo extraccion de las urls de las temporada")
        return l_urls_temporadas

    def get_button_next_jornada(self):
        """
        Click en siguiente jornada
        :return:
        """
        try:
            # Obtengo la jornada actual, obtengo la jornada anterior (tengo que ir hacia atras pues comienza en la ult jornada)
            jornada = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, './/div[@id="flechaatr"]/img')))
            return jornada

        except:
            print("Fallo el click en la jornada anterior")
            return None

    def get_urls_partidos(self):
        """
        Extraer URLs de la informacion de cada partido
        :return:
        """
        # Definicion de variables
        l_urls_partidos = []

        try:
            # Extraigo los tags que contienen las URLs a la informacion de cada partido de la jornada
            l_tag_partidos = WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, './/div[@id="fixturein"]//td[@class="game-info"]/a')))

            # Por tag (c/u contiene la url de un partido)
            for tag in l_tag_partidos:
                l_urls_partidos.append(tag.get_attribute('href'))
            return l_urls_partidos
        except:
            print("Fallo extraccion de las URLs de la informacion de los partidos")

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
        fecha_datetime = datetime.datetime(n_año, n_mes, n_dia) # Convierto variable de clase 'str' a clase 'datetime.datetime'
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
    d = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, 'nov': 11, 'dic': 12}

    # intento extraer dia, mes y año de string
    try:
        dia, n_dia, de ,mes, n_año = fecha_string.split()  # Separo string original segun espacios para obtener elementos (mes y dia)
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
    df = pd.DataFrame(columns=['fecha', 'equipo_loc', 'equipo_vis', 'arbitro', 'entrenador_loc', 'entrenador_vis', 'equipo_ganador'])
    crawler = CrawlerActions()  # Creo objeto de clase CrawlerActions()

    # Ingreso a pagina
    crawler.driver.get('https://www.promiedos.com.ar/primera=historialtorneos')  # hasta que no se carga toda la pagina, no sigue...
    sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # POR PAGINA (TEMPORADA) DE PAGINACION
    for url_temp in crawler.get_urls_temporadas(N_TEMPS):

        # Ingreso a pagina de temporada
        crawler.driver.get(url_temp)
        n_jorn = 1  # Numero de jornada
        print(" Nº Temporada: {} ".format(n_temp).center(120, "#"))

        # Por fecha o jornada (hasta que no exista flecha de ir a siguiente fecha)
        while True:

            # Definicion de variables
            print(" Nº Jornada: {} ".format(n_jorn).center(120, "-"))
            n_part = 0

            # Por partido
            for url_partido in crawler.get_urls_partidos():

                # Ingreso a pagina de informacion del partido
                crawler.driver.get(url_partido)

                # Extraigo campos
                # fecha = crawler.extract_fecha()
                # print("aksndgfnjadfia", fecha)
                # equipo1, equipo2 = crawler.extract_teams()
                # arbitro = crawler.extract_arbitro()
                # entrenador_loc, entrenador_vis = crawler.extract_dts()
                # equipo_ganador = crawler.extract_result()

                # Salgo de pagina de info del partido recien extraido
                crawler.driver.back()  # No puedo quitarla

                # Si el partido no tiene fecha adentro de info partido... 1) Busco en pagina de afuera? 2) Asigno al azar una fecha de otro partido de la jornada
                # if fecha is None:
                #     fecha = crawler.extract_fecha_2(n_part)

                # Formateo fecha
                # fecha_form = format_date(fecha)

                # # GUARDADO DE DATOS EN DATAFRAME
                # l_data = [fecha_form, equipo1, equipo2, arbitro, entrenador_loc, entrenador_vis, equipo_ganador]
                # df.loc[len(df)] = l_data
                # print(l_data)
                # n_part += 1


            # CLICK EN SIGUIENTE FECHA/JORNADA
            sleep(5)  # Falla la paginacion de la jornada. Lo raro es que falla solo cuando extrae los campos de los partidos sino no, especificamente cuando hace el driver.get() y el driver.back()
            button_next_jornada = crawler.get_button_next_jornada()

            # Si hay aun jornadas sin extraer
            if button_next_jornada is not None:
                # Clickeo en sigueinte jornada
                button_next_jornada.click()
                n_jorn += 1
            # Si ya extraje todas las jornadas de la temporada
            else:
                # Salgo del while tal que cambio de temporada
                n_temp += 1
                break

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    
    # Guardado de archivo excel en computadora
    df.to_excel('./liga_argentina_historico_2.xlsx', index=False)  # Cambiar la ruta del archivo


main()