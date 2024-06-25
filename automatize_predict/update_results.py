import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import datetime
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_matches_result
from p3_data_preparation.construct_data import determine_result
from p4_modeling.asses_model import determine_winning_bets

def main():
    """
    Actualizo los resultados de los partidos que se jugaron recientemente.
    """
    df_results = pd.DataFrame()

    # Levanto datasets
    df_countries = pd.read_excel('p2_data_understanding/data/df_countries.xlsx')  # Para garantizar que tengo todas las predicciones
    df_comp = pd.read_excel('p2_data_understanding/data/df_competencies.xlsx')  # Para garantizar que tengo todas las predicciones
    df = pd.read_excel('p6_deployment/data/historial_predicciones.xlsx', index_col=0)  # Para garantizar que tengo todas las predicciones

    # Determino competencias a extraer
    df_comp_public = df_comp[df_comp['is_public'] == 1]
    print(df_comp_public)

    # Seleccionar los partidos de ultimo/s dia/s
    fecha_hoy = datetime.datetime.now()
    n_days_to_extract = 7  # 7 por pruebas sino es 1
    fecha_limite = fecha_hoy - datetime.timedelta(days=n_days_to_extract)
    df_filt = df[(df['date'] > fecha_limite) & (df['date'] <= fecha_hoy)]
    l_ids = df_filt.index
    l_countries = df_filt['id_country'].unique()
    print(f"Fechas a filtrar: {fecha_limite} --> {fecha_hoy}")
    print(df_filt.shape)
    print(f"Lista de ids a los que extraer resultado: {l_ids}")

    # Por competicion
    for idx, row in df_comp_public.iterrows():

        # Si hay partidos del pais al cual obtener resultados
        if row['id_country'] in l_countries:

            # Definicion de variables
            country = df_countries[df_countries['id_country']==row['id_country']]['country_name'].values[0]
            competition = row['competition_flashscore']
            print(f"id_country: {row['id_country']} Country: {country} Competition: {competition}")
            
            # Extraer goles home y away en los partidos
            df_results_comp = extract_matches_result(country, competition, l_ids)  # df_results = pd.read_excel('p6_deployment/data/results.xlsx', index_col=0)

            # Guardo results de competencia
            df_results = pd.concat([df_results, df_results_comp], axis=0)
            print(df_results)

    # Agrego columnas 'goals_home' y 'goals_away' a predicciones.xlsx
    df_concat = pd.concat([df_filt, df_results], axis=1)

    # Determino ganador y si acerté
    df_concat = determine_result(df_concat, var_resp='result')
    df_concat = determine_winning_bets(df_concat)     # Determino acierto o fallo
    df_concat.index.name = 'id_match'  # Es importante para la base de datos MySQL
    print(df_concat)

    # Exportar dataset results.xlsx
    df_concat.to_excel('p6_deployment/data/predicciones.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()