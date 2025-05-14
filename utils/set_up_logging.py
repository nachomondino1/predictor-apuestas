import logging
import datetime
import os
import sys

# Obtener la fecha y hora actual para distinguir corridas en el mismo día
fecha_hoy = datetime.datetime.now().date()
# hora_actual = datetime.datetime.now().strftime("%H-%M-%S")  # Formato HH-MM-SS

# Obtener el nombre del script que ejecuta el proceso
script_name = os.path.basename(sys.argv[0]).replace(".py", "")  # Elimina la extensión .py

# Definir el archivo de log con fecha, hora y nombre del script principal
log_file = f'data/_logs/{fecha_hoy}_{script_name}.csv' # {hora_actual}

# Configurar logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configurar el handler para la consola de depuración
debug_console_handler = logging.StreamHandler()
debug_console_handler.setLevel(logging.INFO)

# Configurar el handler para escribir en el archivo CSV
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.INFO)

# Formateador de colores para consola
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\x1b[34m",  # Azul
        logging.ERROR: "\x1b[31m",  # Rojo
        logging.WARNING: "\x1b[33m",  # Amarillo
        logging.CRITICAL: "\x1b[32m"  # Verde (éxito)
    }
    RESET = "\x1b[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{super().format(record)}{self.RESET}"

console_formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
debug_console_handler.setFormatter(console_formatter)

# Formato para archivo CSV (sin colores)
file_formatter = logging.Formatter("%(asctime)s,%(name)s,%(levelname)s,%(message)s")
file_handler.setFormatter(file_formatter)

# Agregar handlers al logger
logger.addHandler(debug_console_handler)
logger.addHandler(file_handler)

# Ejemplo de logs
logger.info("Este es un mensaje de información")
logger.warning("Este es un mensaje de advertencia")
logger.error("Este es un mensaje de error")
logger.critical("Este es un mensaje de éxito")
