import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import os
import shutil

def make_directories(l_directorios):
    
    # Verificar si el argumento es un string
    if isinstance(l_directorios, str):
        # Convertir el string en una lista con un solo elemento
        l_directorios = [l_directorios]
    
    # Iterar sobre la lista (ya sea original o convertida)
    for directorio in l_directorios:
        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

def mover_archivo(origen, destino):
    """
    Mueve archivo o directorio de origen a destino.
    """
    try:
        # Mover el archivo desde el origen al destino
        shutil.move(origen, destino)
        logger.critical(f"Archivo movido de {origen} a {destino} correctamente.")
    except FileNotFoundError:
        logger.error(f"No se pudo encontrar el archivo {origen}.")
    except PermissionError:
        logger.error(f"No tienes permisos para acceder o mover el archivo {origen}.")
    except Exception as e:
        logger.error(f"Ocurrió un error al intentar mover el archivo: {e}")

def duplicate_archivo(source_path, destination_path):

    # Copy the file directly
    try:
        shutil.copy(source_path, destination_path)
        print(f"File copied successfully from {source_path} to {destination_path}")
    except FileNotFoundError:
        print(f"Error: The source file {source_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def remove_directories(directories):
    """
    Elimina directorios o archivos especificados en la lista.
    
    Parameters:
        directories: list
            Lista de rutas de directorios o archivos a eliminar.
    """
    for directory in directories:
        try:
            if os.path.isdir(directory):  # Verifica si es un directorio
                shutil.rmtree(directory)
                print(f"Directorio eliminado: {directory}")
            elif os.path.isfile(directory):  # Verifica si es un archivo
                os.remove(directory)
                print(f"Archivo eliminado: {directory}")
            else:
                print(f"No se encontró la ruta: {directory}")
        except Exception as e:
            print(f"Error al eliminar {directory}: {e}")