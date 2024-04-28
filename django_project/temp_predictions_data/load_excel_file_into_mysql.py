from datetime import datetime
import logging
import mysql.connector
import pandas as pd

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

# Ejemplo de uso del logger
logger.info("Este es un mensaje de información")
logger.error("Este es un mensaje de error")


def load_excel_file_into_mysql(
    file_name,
    table_name,
    password="MiComidaFavoritaMilanesa_2024",
    host="localhost",
    user="root",
    db="predictor_apuestas",
) -> None:
    """This function loads excel files into MySQL database locally.
    We need to authenticate with that db first.

    Args:
        file_name (str): name of the excel file to load into the database
        table_name (str): name of the table in the database that will be populated with data
        password (str): MySQL password of the database.
        host (str): MySQL host name of the database. Default = localhost
        user (str): MySQL user name of the database. Default = root
        db (str): MySQL name of the database. Default = predictor_apuestas

    Returns:
        None
    """
    excel_data = pd.read_excel(file_name)
    excel_data["date"] = pd.to_datetime(excel_data["date"], format="%d.%m.%Y %H:%M")
    excel_data = excel_data.fillna(0)
    logger.info(excel_data.info())

    # Conexión a la base de datos MySQL
    try:
        conexion_mysql = mysql.connector.connect(
            host=host, user=user, password=password, database=db
        )
    except (mysql.connector.Error, IOError) as err:
        logger.info("Failed to connect, exiting without a connection: %s", err)
        return None

    # Inserta los datos en la tabla MySQL
    cursor = conexion_mysql.cursor()

    for index, row in excel_data.iterrows():
        values = tuple(row)
        placeholders = ",".join(["%s"] * len(row))
        query = f"INSERT INTO {table_name} ({','.join(excel_data.columns.tolist())}) VALUES ({placeholders})"
        try:
            cursor.execute(query, values)
            logger.info("Row inserted in MySQL table")
        except (mysql.connector.Error, IOError) as err:
            logger.info("Failed to insert row, duplicate entry: %s", err)

    # Save changes in the db
    conexion_mysql.commit()
    # Close conextion
    cursor.close()

    return None


# file_name = "django_project/data/predicciones.xlsx"
file_name = "django_project/data/predicciones_v02.xlsx"
load_excel_file_into_mysql(file_name, "home_prediction")