import requests
import json

# Configura tus variables
TARGET_OWNER = "mondineta"  # Cambia esto al nombre de tu usuario u organización en GitHub
TARGET_REPO = "landing"  # Cambia esto al nombre del segundo repositorio
GITHUB_TOKEN = "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"  # Tu token de acceso personal de GitHub

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
    print("Evento despachado con éxito")
else:
    print(f"Error al despachar el evento: {response.status_code}")
    print(response.json())
