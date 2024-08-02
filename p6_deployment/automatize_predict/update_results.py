import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import datetime
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_matches_result
from p3_data_preparation.construct_data import determine_result
from p4_modeling.asses_model import determine_winning_bets

def determine_last_matches(df: pd.DataFrame, n_days: int = 1):
    """
    Selecciono los partidos jugados en los ultimos <n_days>.
    """
    # Seleccionar los partidos de ultimo/s dia/s
    fecha_hoy = datetime.datetime.now()
    fecha_limite = fecha_hoy - datetime.timedelta(days=n_days)
    print(f"Fechas a filtrar: {fecha_limite} --> {fecha_hoy}")

    df_filt = df[(df['date'] > fecha_limite) & (df['date'] <= fecha_hoy)]
    print(df_filt.shape)
    return df_filt

def collect_results(df: pd.DataFrame, df_countries: pd.DataFrame, df_comp_public: pd.DataFrame):
    """
    Extraigo de Flashscore el resultado de los partidos pasados como parametro (df) y lo agrego como columna.
    """
    # Definicion de variables
    df_results = pd.DataFrame()
    l_ids = df.index
    l_countries = df['id_country'].unique()
    print(f"Lista de ids a los que extraer resultado: {l_ids}")

    # Por competicion
    for idx, row in df_comp_public.iterrows():
        competition = row['competition_flashscore']

        # Si hay partidos del pais al cual obtener resultados
        if row['id_country'] in l_countries:

            # Definicion de variables
            country = df_countries[df_countries['id_country']==row['id_country']]['country_name'].values[0]
            print(f"id_country: {row['id_country']} Country: {country} Competition: {competition}")
            
            # Extraer goles home y away en los partidos desde Flashscore
            df_results_competition = extract_matches_result(country, competition, l_ids)

            # Guardo results de competencia
            df_results = pd.concat([df_results, df_results_competition], axis=0)
            print(df_results)

    # Agrego columnas 'goals_home' y 'goals_away' a predicciones.xlsx
    df_pred_with_goals = pd.concat([df, df_results], axis=1)

    # Determino ganador y si acerté
    df_pred_with_result = determine_result(df_pred_with_goals, var_resp='result')
    df_pred_with_result = determine_winning_bets(df_pred_with_result)     # Determino acierto o fallo
    df_pred_with_result.index.name = 'id_match'  # Es importante para la base de datos MySQL
    print(df_pred_with_result)
    return df_pred_with_result

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Parametros de ejecución
    n_days = int(sys.argv[1])  # Numero de dias maximo desde hoy para extraer partidos (e.g. 7)

    # Levanto datasets
    df_historial_predicciones = pd.read_excel('data/historial_predicciones.xlsx', index_col=0)  # Para garantizar que tengo todas las predicciones
    df_countries = pd.read_excel('p2_data_understanding/data/df_countries.xlsx')  # Para garantizar que tengo todas las predicciones

    # Determino competencias a extraer
    df_comp = pd.read_excel('data/df_competencies.xlsx')  # Para garantizar que tengo todas las predicciones
    df_comp_public = df_comp[df_comp['is_public'] == 1]
    print(df_comp_public)

    # Selecciono los partidos de los ultimos <n_days>
    df_last_matches = determine_last_matches(df_historial_predicciones, n_days)

    # Agrego columnas 'goals_home', 'goals_away', 'result' y 'acerte'
    df = collect_results(df_last_matches, df_countries, df_comp_public)

    # Exportar dataset
    df.to_excel('data/predicciones.xlsx')