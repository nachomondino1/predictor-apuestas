import requests
import json
import os

# Configura tus variables
OWNER = "nachomondino1"  # Cambia esto al nombre de tu usuario u organización en GitHub
TARGET_OWNER = "mondineta"
REPO = "predictor-apuestas"    # Cambia esto al nombre del primer repositorio
TARGET_REPO = "landing"  # Cambia esto al nombre del segundo repositorio
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Tu token de acceso personal de GitHub (definir por terminal con 'export GITHUB_TOKEN=your_personal_access_token')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')  # El secreto que usaste para el webhook (definir por terminal con 'export WEBHOOK_SECRET=your_personal_access_token')

# URL de la API de GitHub para configurar el webhook
url = f"https://api.github.com/repos/{OWNER}/{REPO}/hooks"

# Datos del webhook
payload = {
    "name": "web",
    "active": True,
    "events": ["push"],
    "config": {
        "url": f"https://api.github.com/repos/{TARGET_OWNER}/{TARGET_REPO}/dispatches",
        "content_type": "json",
        "secret": WEBHOOK_SECRET,
    }
}

# Cabeceras para la autenticación
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Crear el webhook
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Comprobar la respuesta
if response.status_code == 201:
    print("Webhook creado con éxito")
else:
    print(f"Error al crear el webhook: {response.status_code}")
    print(response.json())
