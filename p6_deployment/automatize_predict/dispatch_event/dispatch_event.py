import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import requests
import json
import os

# Configura tus variables
TARGET_OWNER = "mondineta"  # Cambia esto al nombre de tu usuario u organización en GitHub
TARGET_REPO = "landing"  # Cambia esto al nombre del segundo repositorio
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Obtener el token de acceso personal de las variables de entorno


# URL de la API de GitHub para despachar el evento
url = f"https://api.github.com/repos/{TARGET_OWNER}/{TARGET_REPO}/dispatches"


# Datos del evento
payload = {
    "event_type": "transfer_xlsx"
}

# Cabeceras para la autenticación
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Despachar el evento
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Comprobar la respuesta
if response.status_code == 204:
    logger.critical("Evento despachado con éxito")
else:
    logger.error(f"Error al despachar el evento: {response.status_code}")
    logger.error(response.json())
