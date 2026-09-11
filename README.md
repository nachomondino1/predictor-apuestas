# Predictor de apuestas deportivas

> Documentación del código y del refactor en curso: [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md)
> y [`docs/REFACTOR.md`](docs/REFACTOR.md).

## Instalación

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # dependencias (para entrenar)
# o, más liviano, solo para predecir/scrapear:  pip install -r src/predictor/deployment/requirements_mnm.txt
pip install -e . --no-deps           # instala el repo como paquete -> imports absolutos sin sys.path hacks
```

El código de librería vive en `src/predictor/` (`data_understanding/`,
`data_preparation/`, `modeling/`, `deployment/`, `utils/`, `stages.py`). El
`pip install -e .` reemplaza el viejo `sys.path.append('.')` que había al inicio
de cada script: con el paquete instalado, `from predictor.utils...`,
`from predictor.data_preparation...` funcionan desde cualquier directorio, sin
necesitar `PYTHONPATH`. Ver [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) para
el detalle completo de la estructura.

## Actualizar el Virtual Environment luego de agregar una libreria: 

- Si otra persona actualizó el requirements.txt y queres actualizar tu virtualenv local:
    1) Activa tu entorno virtual si aún no lo has hecho. Puedes hacerlo con el siguiente comando: <br>
    `source /ruta/a/tu/entorno_virtual/bin/activate` <br>
    2) Una vez activado el entorno virtual, puedes usar el comando pip para instalar las dependencias del archivo requirements.txt. Para instalar todas las dependencias listadas en requirements.txt, simplemente ejecuta: <br>
    `pip install -r requirements.txt` <br>
    Después de ejecutar este comando, tu entorno virtual estará actualizado con las dependencias especificadas en el archivo requirements.txt. <br>

- Cada vez que agreguemos librerias nuevas al virtual environment, debemos 
actualizar el archivo de requirements.txt corriendo el comando: <br>
`pip freeze > requirements.txt`

## Como crear una nueva rama: 
- 1. Crear nueva rama en Github 
- 2. Ir a VSC, abrir la terminal y para ver todas las ramas creadas remotas correr: `git fetch`
- 3. Elegis tu rama y corres: `git checkout <nombre_de_rama>`

## Como correr django para ver la pagina web:
- 1. Cambiar directorio a django_project con `cd django_project`
- 2. Correr el siguiente comando para activar la pagina web de django `python manage.py runserver`
- 3. Ir a la url para obtener: 
    - All predictions: http://127.0.0.1:8000/home/predictions/ . Corre: `GET /home/predictions/ HTTP/1.1`
    - Filtered predictions: http://127.0.0.1:8000/home/predictions/?date=2024-02-24&id_match=&ordering=time . Corre: `GET /home/predictions/?date=2024-02-24&id_match=&ordering=time HTTP/1.1`
    - Predictions by match_id: http://127.0.0.1:8000/home/predictions/MNMbQMK1/ . Corre: `GET /home/predictions/MNMbQMK1/ HTTP/1.1`
    - Home: http://127.0.0.1:8000/home/hello/ . Corre: `GET /home/hello/ HTTP/1.1`
