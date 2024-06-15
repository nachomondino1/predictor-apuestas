
import pandas as pd

def main(minutos_a_restar, df_next_matches):
    """
    Obtengo fecha y hora y pais para el cual correr collect_data.py
    """
    # Obtener fechas unicas y paises 
    df = df_next_matches.loc[:, ['date', 'id_country']]  # Selecciono algunas columnas 
    df = df.drop_duplicates()   # Obtener los registros únicos

    # Restar x minutos a cada hora
    delta = pd.to_timedelta(minutos_a_restar, unit='m')
    df['date_mod'] = df['date'] - delta    # Restar el timedelta a cada valor de la columna 'Hora'

    # Exportar schedules.xlsx
    df.to_excel('github_actions/update_workflow/schedules.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    # Definir la cantidad de minutos a restar
    minutos_a_restar = 15

    # Levantar proximos partidos
    df_next_matches = pd.read_excel('p6_deployment/data/predicciones.xlsx')
    
    main(minutos_a_restar, df_next_matches)



'''
def si_separo_antes_en_date_y_time(minutos_a_restar, df_next_matches):
    """
    Obtengo fecha y hora y pais para el cual correr collect_data.py
    """
    # Obtener fechas unicas y paises 
    df = df_next_matches.loc[:, ['date', 'time', 'id_country']]  # Selecciono algunas columnas 
    df = df.drop_duplicates()   # Obtener los registros únicos

    # Restar x minutos a cada hora
    df['time_mod'] = pd.to_datetime(df['time'], format='%H:%M:%S')  # Convertir la columna 'Hora' a tipo datetime, asumiendo una fecha arbitraria porque solo nos interesa el tiempo
    delta = pd.to_timedelta(minutos_a_restar, unit='m')  # Crear un objeto timedelta con la cantidad de minutos a restar
    df['time_cron'] = df['time_mod'] - delta  # Restar el timedelta a cada valor de la columna 'Hora'
    df['time_cron'] = df['time_cron'].dt.time

    # Exportar schedules.xlsx
    df.to_excel('github_actions/update_workflow/schedules.xlsx', index=False)

'''