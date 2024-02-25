# Predictor de apuestas deportivas

## Instalación del Virtual Environment

- 1. Instalar virtualenv: `pip install virtualenv`
- 2. Crear un nuevo virtual environment: `virtualenv <nombre_virtual_env>`. Ejemplo: `virtualenv backEnv`
- 3. Activar el virtual env: `source <nombre_virtual_env>/bin/activate`. Ejemplo: `source backEnv/bin/activate`
- 4. Instalar las dependencias: `pip install -r requirements.txt`

## Actualizar el Virtual Environment luego de agregar una libreria: 

- Cada vez que agreguemos librerias nuevas al virtual environment, debemos 
actualizar el archivo de reuirements.txt corriendo el comando: 
`pip freeze > requirements.txt`