from predictor.utils.set_up_logging import logger
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException, StaleElementReferenceException, NoSuchWindowException
from time import sleep
import os
import platform
import subprocess
from dotenv import load_dotenv

load_dotenv()  # Permite leer CHROMEDRIVER_PATH (y demás env vars) aunque se use el módulo directamente


class Crawler:
    """ It contains all the actions that the bot can perform from accepting cookies to clicking on the next one. """

    def __init__(self, headless: bool = True, browser: str = "Chrome"):
        """Initialize attributes of the parent class."""
        if browser == "Chrome":
            self.driver = self.inicialize_chrome_driver(headless)
        elif browser == "Firefox":
            self.driver = self.initialize_firefox_driver(headless)
        elif browser == "Safari":
            self.driver = self.initialize_safari_driver()
        else:
            logger.error("La libreria no posee ese browser")

    def get_chrome_version(self):
        """
        Detecta la version de Google Chrome en mi compu local. Esto es para poder crear el chrome driver con la misma version para evitar el problema de incompatibilidad de versiones.
        """
        system = platform.system()
        try:
            if system == "Windows":
                import winreg
                reg_path = r"SOFTWARE\Google\Chrome\BLBeacon"
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path)
                version, _ = winreg.QueryValueEx(key, "version")
                return version
            elif system == "Darwin":
                process = subprocess.run(
                    ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                version = process.stdout.decode().strip().split()[-1]
                return version
            elif system == "Linux":
                process = subprocess.run(
                    ["google-chrome", "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                version = process.stdout.decode().strip().split()[-1]
                return version
            else:
                raise Exception("Unsupported OS")
        except Exception as e:
            logger.error(f"Error obtaining Chrome version: {e}")
            return None

    def inicialize_chrome_driver(self, headless: bool):
        """
        Initialize a Chrome WebDriver.

        Intenta 3 estrategias en orden y devuelve el primer driver que se cree:
          1) ChromeDriver con la última versión que ofrece webdriver-manager.
          2) ChromeDriver con la misma versión que el Google Chrome instalado localmente.
          3) ChromeDriver desde un ejecutable local (ruta en la env var CHROMEDRIVER_PATH).

        Args:
            headless (bool): True to prevent the web browser from opening, False otherwise.

        Returns:
            WebDriver: Chrome WebDriver instance.
        """
        # Defino opciones del webdriver
        options = webdriver.ChromeOptions()
        # options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # google_chrome_path cl
        options.add_argument("--window-size=1920,1080")
        options.add_argument("start-maximized")
        options.add_argument("enable-automation")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-browser-side-navigation")
        options.add_argument("--disable-gpu")
        options.add_argument("--incognito")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--remote-debugging-port=9222")

        if headless:
            options.add_argument("--headless")

        # Estrategias de inicialización en orden de preferencia. Cada una devuelve un
        # driver o lanza una excepción; se usa la primera que tenga éxito.
        def _driver_latest_version():
            # 1) ChromeDriver con la última versión disponible
            chrome_driver = ChromeDriverManager().install()
            return webdriver.Chrome(service=Service(chrome_driver), options=options)

        def _driver_local_chrome_version():
            # 2) ChromeDriver con la misma versión que el Google Chrome instalado
            chrome_version = self.get_chrome_version()
            if not chrome_version:
                raise RuntimeError("No se pudo detectar la versión de Google Chrome instalada")
            logger.info(f"Detected Google Chrome version: {chrome_version}")
            chrome_driver = ChromeDriverManager(driver_version=chrome_version).install()
            return webdriver.Chrome(service=Service(chrome_driver), options=options)

        def _driver_from_executable():
            # 3) ChromeDriver desde ejecutable local (descargar desde https://googlechromelabs.github.io/chrome-for-testing/)
            chrome_driver_path = os.getenv("CHROMEDRIVER_PATH")
            if not chrome_driver_path:
                raise RuntimeError("La env var CHROMEDRIVER_PATH no está definida")
            return webdriver.Chrome(service=Service(chrome_driver_path), options=options)

        strategies = [
            ("última versión de webdriver-manager", _driver_latest_version),
            ("versión del Chrome local", _driver_local_chrome_version),
            ("ejecutable local (CHROMEDRIVER_PATH)", _driver_from_executable),
        ]
        hint = (
            "Posibles soluciones: "
            "\n1) Tenes una actualizacion de software pendiente en tu compu. "
            "\n2) Tenes que eliminar caché de webdriver-manager (abri terminal y ejecutá 'rm -rf ~/.wdm'). "
            "\n3) Cerrar Google Chrome y volver a ejecutar el script."
        )

        for name, build_driver in strategies:
            try:
                driver = build_driver()
                logger.critical(f"ChromeDriver inicializado ({name})")
                return driver
            except Exception as e:
                logger.error(f"Falló la inicialización de ChromeDriver por '{name}': {e}")

        logger.error(f"Fallaron todas las vías de inicialización del webdriver. \n{hint}")
        raise RuntimeError("No se pudo inicializar el ChromeDriver por ninguna vía")

    def initialize_safari_driver(self):
        driver = webdriver.Safari()
        return driver

    def initialize_firefox_driver(self, headless=True):

        options = webdriver.FirefoxOptions()
        options.add_argument("--width=1920")  # Ancho de la ventana (equivalente a --window-size en Chrome)
        options.add_argument("--height=1080")  # Altura de la ventana (equivalente a --window-size en Chrome)
        options.add_argument("--start-maximized")  # Iniciar maximizado
        options.add_argument("--disable-infobars")  # Deshabilitar las barras de información
        options.add_argument("--disable-extensions")  # Deshabilitar extensiones
        options.add_argument("--private")  # Modo incógnito (equivalente a --incognito en Chrome)
        options.add_argument("--disable-popup-blocking")  # Deshabilitar el bloqueo de ventanas emergentes
        options.headless = headless  # Modo sin cabeza (equivalente a --headless en Chrome)

        driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
        return driver

    def extract_tag(self, xpath: str, xpath_alt: str = None, tag_inicial=None, attribute: str = None, text:bool = False, sec_wait: float = 10, print_fail: bool = True):
        """
        Extrae texto de un tag
        
        # Parameters
            xpath: XPATH del tag del cual extraer datos
            tag_inicial: Selenium Web Element desde el cual se busca el xpath. 
                Si usas tag_inicial el xpath debe empezar con un punto (".//") para buscar dicho xpath desde el tag (y desde todo el html)
            attribute: String con el nombre del atributo a extraer del tag (e.g. "href")
            text: True para extraer texto del tag
        
        # Return
            String. En caso que falle la extraccion, None
        """
        tag_inicial = self.driver if tag_inicial is None else tag_inicial  # Tag desde el que buscar el xpath
        xpaths = [xpath] if xpath_alt is None else [xpath, xpath_alt]

        for xpath_expr in xpaths:
            attempts = 0
            max_attempts = 3

            while attempts < max_attempts:
                try:
                    tag_res = WebDriverWait(tag_inicial, sec_wait).until(EC.presence_of_element_located((By.XPATH, xpath_expr)))
                    return tag_res.text if text else tag_res.get_attribute(attribute) if attribute is not None else tag_res
                except StaleElementReferenceException:
                    attempts += 1
                    logger.warning(f"Se produjo una excepción StaleElementReferenceException. Intento {attempts}/{max_attempts}")
                    sleep(1)
                except NoSuchWindowException:
                    logger.warning(f'Selenium está intentando interactuar con una ventana del navegador que ya se ha cerrado')
                    break
                except TimeoutException:
                    break  # Salir del bucle si se alcanza el tiempo de espera máximo

        if print_fail:
            logger.error(f"Fallo la extraccion del campo. Probablemente no exista el xpath {xpath}")
        return None

    def extract_tags(self, xpath, tag_inicial=None, sec_wait=10, print_fail=True):
        """
        Encuentra todos los tags segun el xpath
        :param xpath: XPATH de los tags
        :return: Lista de tags, en caso contrario, lista vacia

        """
        tag_inic = self.driver if tag_inicial is None else tag_inicial  # Tag desde el que buscar el xpath

        try:
            return WebDriverWait(tag_inic, sec_wait).until(EC.presence_of_all_elements_located((By.XPATH, xpath)))

        except TimeoutException:
            if print_fail:
                logger.error(f"Fallo la extraccion de tags. Probablemente no exista el xpath {xpath}")
            return []  # Si devuelvo None y el usuario itera sobre el return, dara el error: TypeError: 'NoneType' object is not iterable

    def click_boton(self, tag_boton, sec_wait: float = 10):
        """
        Click en boton

        # Parameters:
            tag_boton: Selenium Web Element. Tag HTML (y no xpath) sobre el cual hacer click.
            sec_wait: Integer. Espera maxima para encontrar el boton y hacer click.            
        """
        if tag_boton is not None:
            try:
                WebDriverWait(self.driver, sec_wait).until(EC.element_to_be_clickable(tag_boton))
                tag_boton.click()
            except (StaleElementReferenceException, TimeoutException, ElementClickInterceptedException):
                try:
                    self.driver.execute_script("arguments[0].click()", tag_boton)
                except:
                    return False
        else:
            return False

    # Forma 1: Boton aceptar cookies OCULT0 en tag #shadow-root
    def accept_cookies_in_shadow_tag(self, xpath_shadow_parent, xpath_boton):
        """
        Click en aceptar cookies
        """
        # Ubico el padre del "shadow root"
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath_shadow_parent)))
        shadow_parent = self.driver.find_element(By.XPATH, xpath_shadow_parent)

        # Obtengo el shadow root
        shadow_root = self.driver.execute_script('return arguments[0].shadowRoot', shadow_parent)

        # Ubico el botón "Aceptar cookies" y lo clikeo
        tag_boton = self.extract_tag(tag_inicial=shadow_root, xpath=xpath_boton)
        self.click_boton(tag_boton)

    # Forma 2: Boton aceptar cookies OCULT0 en tag #document
    def accept_cookies_in_document_tag(self, xpath_document_parent, xpath_boton):
        """
        Click en aceptar cookies
        """
        # Obtengo tag padre de #document
        iframe = self.driver.find_element(By.XPATH, xpath_document_parent)

        # Switcheo frame al padre de #document
        self.driver.switch_to.frame(iframe)

        # Ubico el botón "Aceptar cookies" y lo clikeo
        tag_boton = self.extract_tag(xpath=xpath_boton)
        self.click_boton(tag_boton)

    # Login website
    def fill_form(self, xpath_input, text, enter=True):
        """
        Carga de texto en input.
        """
        # Busco tag input para el usuario y escribo el usuario
        input_tag = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath_input)))
        input_tag.send_keys(text)

        # Si no hay opciones que elegir
        if enter:
            input_tag.send_keys(Keys.ENTER)

        # Si hay opciones que elegir
        # ...

    def login_website(self, user, password, xpath_user, xpath_pass, xpath_boton_login, xpath_boton_validate_user=None):
        """
        Login website

        # Parameters:
            user: Tu usuario (string).
            password: Tu password (string).
            xpath_user: Xpath del input donde cargar usuario (string)
            xpath_pass: Xpath del input donde cargar password (string)
            xpath_boton_login: Xpath del boton donde clickear para hacer login (string)
        """
        # Busco tag input para el usuario y escribo el usuario
        self.fill_form(xpath_user, user, enter=False)

        # Si hay que validar el usuario
        if xpath_boton_validate_user is not None:

            # Localizo el boton que valida el usuario y lo clickeo
            tag_boton = self.extract_tag(xpath=xpath_boton_validate_user)
            self.click_boton(tag_boton)

        # Busco tag input para la pass y escribo la pass
        self.fill_form(xpath_pass, password, enter=False)

        # Localizo el boton "Iniciar sesion" y lo clickeo
        tag_boton = self.extract_tag(xpath=xpath_boton_login)
        self.click_boton(tag_boton)

    # Select option in multiple choice input
    def select_option(self, option, xpath_flechita, xpath_input):
        """
        Seleccionar una opcion de un input multiple choice

        # Parameters
            option: Opcion a seleccionar (string)
            xpath_input: Xpath del input multiple choice (string)
            xpath_options: Xpath del listado de opciones (string)
        """
        # Click en flechita para desplegar opciones
        tag_flechita = self.extract_tag(xpath=xpath_flechita)
        self.click_boton(tag_boton=tag_flechita)
        sleep(1) # Tiempo para que despliegue las opciones y cargen

        # Ubicar tag input donde escribir
        tag_input = self.extract_tag(xpath=xpath_input)

        # Cargo opcion
        tag_input.send_keys(option)

        # Click en primera opcion
        tag_input.send_keys(Keys.RETURN)
        sleep(1) # Tiempo para que carue la pagina luego del enter