import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Establecer el nivel de registro a INFO

# Crear un controlador para la consola de depuración de Visual Studio Code
debug_console_handler = logging.StreamHandler()
debug_console_handler.setLevel(
    logging.INFO
)  # Establecer el nivel de registro para la consola de depuración


# Crear un formateador personalizado para agregar colores
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\x1b[34m",  # Azul
        logging.ERROR: "\x1b[31m",  # Rojo
        logging.WARNING: "\x1b[33m",  # Amarillo (si deseas agregar color para los mensajes de advertencia)
        logging.CRITICAL: "\x1b[32m"  # Verde. Lo uso para mensajes de exito!
    }
    RESET = "\x1b[0m"

    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{message}{self.RESET}"

formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
debug_console_handler.setFormatter(formatter)

# Agregar el controlador a logger
logger.addHandler(debug_console_handler)