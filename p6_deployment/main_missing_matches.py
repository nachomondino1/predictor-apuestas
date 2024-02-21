# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from main import DataUnderstanding, DataPreparation
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_missing_data


def collect_missing_data(id_country, country, export=True):
    """
    Extraccion de varias competencias de un mismo country.
    """
    print(" Collecting data... ")
    # Definicion de variables
    df_match_miss, df_match_player_miss = pd.DataFrame(), pd.DataFrame()
    df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx') # /Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx'
    df_comp_country = df_comp[df_comp['id_country'] == id_country]
    print(f' COUNTRY: {country} '.center(120, '#'))

    # Levanto datos viejos 
    df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
    l_competencies = df_match['id_competition'].unique()

    # POR COMPETITION (solo las que hay en df_match)
    for id_competition in l_competencies:
        
        # Obtengo nombre de competicion y is_cup
        df_comp_filt = df_comp_country[df_comp_country['id_competition'] == id_competition]  # Para extrar varios countryes?: df = df_comp[df_comp['country'].isin(l_countryes)]
        competition, is_cup = df_comp_filt['competition_flashscore'].values[0], df_comp_filt['is_cup'].values[0]
        print(f" Competition: {competition} ".center(120, '+'))

        # Actualizo df_match y df_match_player con los partidos faltantes
        df_match_miss_matches, df_match_player_miss_matches = extract_missing_data(country, competition, list(df_match.index))
        print(f"Cantidad de partidos faltantes en df_match: {df_match_miss_matches.shape[0]}")

        # Add columns: id_country, is_cup and id_competition
        df_match_miss_matches['id_country'] = id_country
        df_match_miss_matches['id_competition'] = id_competition
        df_match_miss_matches['is_cup'] = is_cup

        # Concateno dfs
        df_match_miss = pd.concat([df_match_miss, df_match_miss_matches], axis=0)
        df_match_player_miss = pd.concat([df_match_player_miss, df_match_player_miss_matches], axis=0)
   
    # Supongamos que df es tu DataFrame original 
    df_match_odds = df_match_miss.loc[:, ['odds_home', 'odds_draw', 'odds_away']]
    df_match_miss = df_match_miss.drop(['odds_home', 'odds_draw', 'odds_away'], axis=1)

    # Exporto datasets
    if export:
        df_match_miss.to_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_miss.xlsx', index=True)
        df_match_player_miss.to_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_player_miss.xlsx', index=True)
        df_match_odds.to_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_odds_miss.xlsx', index=True)

    return df_match_miss, df_match_player_miss

def main(id_country):
    """
    Extraer automaticamente los partidos que faltan en df_match y df_match_player. Sobretodo es esencial para poder construir estadisticas en 
    partidos nuevos a predecir.

    Cosas a agregar:
    - Actualizar df_player tambien
    """
    # Obtengo el nombre del country mediante el id_country
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0]

    # Creo instancia de clase de main.py (el procesamiento es el mismo)
    # du = DataUnderstanding(id_country=id_country, country=country)
    dp = DataPreparation(country=country)

    # DATA UNDERSTANDING
    # Collect missing data
    df_match_missing, df_match_player_missing = collect_missing_data(id_country, country)
    df_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx') # Podria recolectar nueva version del ultimo fifa.

    # Describe data
    # du.describe_data(df_match_missing, df_match_player_missing, df_player)

    # DATA PREPARATION
    df_match_missing, df_match_player_missing, df_player = dp.format_data(df_match_missing, df_match_player_missing, df_player, export=False)
    df_match_missing, df_match_player_missing, df_player = dp.clean_data(df_match_missing, df_match_player_missing, df_player, export=False)  # thr_nan_col = 1  -> No quiero que elimine nada
    df_integrated_missing = dp.integrate_data(df_match_missing, df_match_player_missing, df_player, export=False)
    return df_integrated_missing

def prueba():
    """
    Extraer automaticamente los partidos que faltan en df_match y df_match_player. Sobretodo es esencial para poder construir estadisticas en 
    partidos nuevos a predecir.

    Cosas a agregar:
    - Actualizar df_player tambien
    """
     # Definicion de variables
    var_resp = 'result'
    country = "England"  # Ponelo en miniscula
    id_country = 48
    data_unders, data_prep = False, True
    export = True

    # Actualizo datasets
    if data_unders:
        # Creo instancia de clase de main.py (el procesamiento es el mismo)
        du = DataUnderstanding(id_country=id_country, country=country)

        # Collect missing data
        df_match_missing, df_match_player_missing = collect_missing_data(country, export=export)
        df_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx') # Podria recolectar nueva version del ultimo fifa.

        # Describe data
        du.describe_data(df_match_missing, df_match_player_missing, df_player)
    else:
        df_match_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_missing.xlsx')
        df_match_player_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_player_missing.xlsx')
        df_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx')

    if data_prep:
        # Creo instancia de clase de main.py (el procesamiento es el mismo)
        dp = DataPreparation(var_resp=var_resp, country=country)

        ## Data prep
        thr_nan_col = 1  # No quiero que elimine nada
        df_match_missing, df_match_player_missing, df_player = dp.format_data(df_match_missing, df_match_player_missing, df_player, export=False)
        df_match_missing, df_match_player_missing, df_player = dp.clean_data(df_match_missing, df_match_player_missing, df_player, thr_nan_col, export=False)
        df_integrated_missing = dp.integrate_data(df_match_missing, df_match_player_missing, df_player, export=False)
        df_integrated_missing.to_excel(f'./p6_deployment/data_next_matches/{country}/data_preparation/df_integrated_missing.xlsx', index=True)

        # Concateno con datos viejos
        ## Levanto datos viejos (para rellenarlos)
        df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx')
        df_integrated_updated = pd.concat([df, df_integrated_missing], axis=0)
        df_integrated_updated.to_excel(f'./p6_deployment/data_next_matches/{country}/data_preparation/df_integrated_updated.xlsx', index=True)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()
