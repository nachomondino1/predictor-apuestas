# Importo librerias
import pandas as pd
import numpy as np
import os
## Data understanding
from p2_data_understanding.collect_initial_data import scraper_flashscore, scraper_sofifa
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
from sklearn.preprocessing import StandardScaler
## Modeling
from p4_modeling import generate_test_design, build_model, asses_model
### Generate test design
from random import randint
from sklearn.model_selection import train_test_split
### Build model
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
### Assess model
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pickle


class DataUnderstanding:

    def __init__(self, id_country: int, country: str):
        self.id_country = id_country
        self.country = country
        self.make_directories()

    def make_directories(self):
        ruta_base = f'./p2_data_understanding/data/{self.country.lower()}/data_seg'
        l_directorios = [f'{ruta_base}/per_season/df_match/',
                         f'{ruta_base}/per_season/df_match_player/',
                         f'{ruta_base}/per_season/df_match_odds/',                    
                         f'{ruta_base}/per_season/df_player/',
                        f'{ruta_base}/per_season/df_player_sofifa/',
                         f'{ruta_base}/per_season/df_player_fifa_sofifa/',

                         f'{ruta_base}/per_competition/df_match/',
                         f'{ruta_base}/per_competition/df_match_player/',
                        f'{ruta_base}/per_competition/df_match_odds/',
                         f'{ruta_base}/per_competition/df_player/',
                        f'{ruta_base}/per_competition/df_teams/',
                        f'{ruta_base}/per_competition/df_coaches/',
                        f'{ruta_base}/per_competition/df_player_sofifa/',
                        f'{ruta_base}/per_competition/df_player_fifa_sofifa/'
                         ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def collect_initial_data(self, export: bool = True):
        """
        Collecting data from Flashscore and Sofifa
        """
        print(" Collecting data... ")
        # Levanto partidos ya extraidos (para no volver a extraer la/s competicione/s ya extraidas)
        try:
            df_match_concat = pd.read_excel(f'p2_data_understanding/data/{self.country}/df_match.xlsx', index_col=0)
            df_match_player_concat = pd.read_excel(f'p2_data_understanding/data/{self.country}/df_match_player.xlsx', index_col=0)
            df_match_odds_concat = pd.read_excel(f'p2_data_understanding/data/{self.country}/df_match_odds.xlsx', index_col=0)
            df_player_concat = pd.read_excel(f'p2_data_understanding/data/{self.country}/df_player.xlsx', index_col=0)
            df_teams_concat = pd.read_excel(f'p2_data_understanding/data/{self.country}/df_teams.xlsx', index_col=0)
            df_coaches_concat = pd.read_excel(f'p2_data_understanding/data/{self.country}/df_coaches.xlsx', index_col=0)
            df_player_sofifa_concat, df_player_fifa_sofifa_concat = pd.DataFrame(), pd.DataFrame()

            l_competition_already_extracted = df_match_concat['id_competition'].unique()
            print(df_match_concat.shape, df_match_player_concat.shape, df_match_odds_concat.shape)
            print(df_player_concat.shape, df_teams_concat.shape, df_coaches_concat.shape)
            print(f"Competiciones ya extraidas: {l_competition_already_extracted}")
        except:
            df_match_concat, df_match_player_concat,df_match_odds_concat, df_player_sofifa_concat, df_player_fifa_sofifa_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            df_teams_concat, df_coaches_concat, df_player_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            l_competition_already_extracted = []

        # Selecciono competencias del country
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
        df_comp_country = df_comp[(df_comp['id_country'] == self.id_country) & ~(df_comp['id_competition'].isin(l_competition_already_extracted))]
        print(f' COUNTRY: {self.country} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_country['competition_flashscore']}")

        # POR COMPETITION
        for i, row in df_comp_country.iterrows():
            print(f' Competition: {row["competition_flashscore"]} '.center(120, '+'))

            # Extraigo partidos de Flashscore (df_match y df_match_player)
            df_match, df_match_player, df_match_odds, df_teams, df_coaches, df_player = scraper_flashscore.extract_data(self.id_country, self.country, row['id_competition'], row['competition_flashscore'], row['is_cup'], n_seasons_max=16, export=export)
            
            print("A", df_teams.shape, df_coaches.shape, df_player.shape)
            df_teams = df_teams[~df_teams.index.isin(df_teams_concat.index)]
            df_coaches = df_coaches[~df_coaches.index.isin(df_coaches_concat.index)]
            df_player = df_player[~df_player.index.isin(df_player_concat.index)]
            print("B" ,df_teams.shape, df_coaches.shape, df_player.shape)

            # Guardo datos de competition
            df_match_concat = pd.concat([df_match_concat, df_match], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player], axis=0)
            df_match_odds_concat = pd.concat([df_match_odds_concat, df_match_odds], axis=0) 
            df_teams_concat = pd.concat([df_teams_concat, df_teams], axis=0)
            df_coaches_concat = pd.concat([df_coaches_concat, df_coaches], axis=0)
            df_player_concat = pd.concat([df_player_concat, df_player], axis=0)
            # Exporto por seguridad
            if export:
                df_match_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_match.xlsx', index=True)
                df_match_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_match_player.xlsx', index=True)
                df_match_odds_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_match_odds.xlsx', index=True)
                df_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_player.xlsx', index=True)
                df_teams_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_teams.xlsx', index=True)
                df_coaches_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_coaches.xlsx', index=True)

            # Si la competition es una liga
            if row['is_cup'] == 0:

                # Extraigo datos de players de Sofifa (df_player)
                df_player_sofifa, df_player_fifa_sofifa_sofifa = scraper_sofifa.extract_players(self.id_country, self.country, row['id_competition'], row['competition_sofifa'], export=export)

                # Save data
                df_player_sofifa_concat = pd.concat([df_player_sofifa_concat, df_player_sofifa], axis=0)
                df_player_fifa_sofifa_concat = pd.concat([df_player_fifa_sofifa_concat, df_player_fifa_sofifa_sofifa], axis=0)
                if export:
                    df_player_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_player_sofifa.xlsx', index=True)
                    df_player_fifa_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_player_fifa_sofifa_sofifa.xlsx', index=True)

        # Exporto datasets con competiciones del country
        if export:
            df_match_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match_player.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match_odds.xlsx', index=True)
            df_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_player.xlsx', index=True)
            df_teams_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_teams.xlsx', index=True)
            df_coaches_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_coaches.xlsx', index=True)
            df_player_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_player_fifa_sofifa.xlsx', index=True)
            df_player_fifa_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_player_fifa_sofifa.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat, df_player_concat, df_coaches_concat, df_player_sofifa_concat, df_player_fifa_sofifa_concat

    def describe_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame,  df_player:  pd.DataFrame, df_teams:  pd.DataFrame, df_coaches: pd.DataFrame, df_player_sofifa:  pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame):

        print(" Describing data... ")

        print("\n DF_MATCH \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match)
        describe_data.verificar_unicidad_registros(df_match) # Verifico unicidad de registros segun campos id

        print("\n DF_MATCH_PLAYER \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_player)

        print("\n DF_MATCH_ODDS \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_odds)

        print("\n DF_PLAYER \n".center(240, "-"))
        describe_data.getting_to_know_data(df_player)
        describe_data.verificar_unicidad_registros(df_player) # Verifico unicidad de registros segun campos id

        print("\n DF_COACHES \n".center(240, "-"))
        describe_data.getting_to_know_data(df_coaches)
        describe_data.verificar_unicidad_registros(df_coaches) # Verifico unicidad de registros segun campos id

        print("\n DF_TEAMS \n".center(240, "-"))
        describe_data.getting_to_know_data(df_teams)
        describe_data.verificar_unicidad_registros(df_teams) # Verifico unicidad de registros segun campos id

        print("\n DF_PLAYER_SOFIFA \n".center(240, "-"))
        describe_data.getting_to_know_data(df_player_sofifa)
        describe_data.verificar_unicidad_registros(df_player_sofifa)

        print("\n DF_PLAYER_FIFA_SOFIFA \n".center(240, "-"))
        describe_data.getting_to_know_data(df_player_fifa_sofifa)
        # describe_data.check_ids_in_both_dataframes(df_player_sofifa, df_player_fifa_sofifa, column='id_player')  # Verifico consistencia en campos que relacionan entidades


class DataPreparation:

    def __init__(self, country: str, var_resp: str = 'result'):
        self.country = country.lower()
        self.var_resp = var_resp
        self.make_directories()

    def make_directories(self):
        l_directorios = [
            f'./p3_data_preparation/data/{self.country.lower()}',
            f'./p3_data_preparation/data/{self.country.lower()}/clean_data',
            f'./p3_data_preparation/data/{self.country.lower()}/integrate_data',
        ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def format_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, export: bool = True):
        """
        Arreglo el data data_type de algunas variables.

        :param df_match: Dataframe de los datos de los partidos. (DataFrame)
        :param df_player_fifa_sofifa: Dataframe de los datos de los players por fifa. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormatting data...")

        # Dataframe match
        ## Date
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        ## Capacity & Attendance
        df_match = format_data.convert_capacity_to_int(df_match)
        ## Ball posession
        df_match = format_data.convert_posesion_to_int(df_match)
        ## Goals
        df_match = format_data.convert_goles_to_int(df_match)
        df_match_player = df_match_player[df_match_player.index.isin(df_match.index)]
        
        # Dataframe player_fifa_sofifa
        ## Fecha
        df_player_fifa_sofifa['date'] = pd.to_datetime(df_player_fifa_sofifa['date'], format='%b %d, %Y')
        ## Market value
        df_player_fifa_sofifa = format_data.convert_value_to_int(df_player_fifa_sofifa)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_match.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_formated.xlsx', index=True)
            df_match_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_player_formated.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'./p3_data_preparation/data/{self.country}/df_player_fifa_sofifa_formated.xlsx', index=True)

        return df_match, df_match_player, df_player_fifa_sofifa

    def clean_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame, df_teams: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, df_teams_sofifa: pd.DataFrame, export: bool = True):
        """
        Limpieza inicial de los dataframes
        """
        start = time.time()
        print("\nCleanning data...")
        warnings.filterwarnings('ignore')
        scaler = StandardScaler()  # Crea un objeto StandardScaler

        ## Elimino filas con alto porcentaje de NaN values
        n_filas = len(df_match)
        df_match_player = df_match_player.dropna(subset=['id_player_start_home_11', 'id_player_start_away_11'], how='any')
        df_match = df_match[df_match.index.isin(df_match_player.index)]
        print(f"De las {n_filas} filas, se eliminan {(n_filas - len(df_match))} por no tener formaciones del partido, quedan {len(df_match)} filas.")

        # Dataframe player
        df_player = clean_data.prepare_text_columns(df_player, l_cols_to_process=['player_name'])

        # Dataframe teams
        df_teams = clean_data.prepare_text_columns(df_teams, l_cols_to_process=['team_name'])
        df_teams = clean_data.clean_teams_names(df_teams)  # Eliminar strings adicionales en names de equipos

        # Dataframe player_sofifa:
        df_player_sofifa = clean_data.prepare_text_columns(df_player_sofifa, l_cols_to_process=['player_name'])  # Preaparo texto para integrar
   
        # Dataframe player_fifa_sofifa
        ## Market Value
        df_player_fifa_sofifa['value'] = scaler.fit_transform(df_player_fifa_sofifa['value'].values.reshape(-1, 1))
        df_player_fifa_sofifa['fifa_year'] = df_player_fifa_sofifa['fifa'].str.split(' ').str[-1]  # Agrego columna "fifa_year" quedandome solo con el año del fifa (e.g. "22" en vez de "FIFA 22")
    
        # df_teams_sofifa
        df_teams_sofifa = clean_data.prepare_text_columns(df_teams_sofifa, l_cols_to_process=['team_name'])

        # Verificar que no haya outliers
        # algo (sacar de mi tesis)

        end = time.time()
        print(f"Clean data in {(end - start) / 60:.1f} minutes")

        if export:
            # df_etiquetas.to_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx', index=False)
            df_match.to_excel(f'./p3_data_preparation/data/{self.country}/clean_data/df_match_cleaned.xlsx', index=True)
            df_match_player.to_excel(f'./p3_data_preparation/data/{self.country}/clean_data/df_match_player_cleaned.xlsx', index=True)
            df_player.to_excel(f'./p3_data_preparation/data/{self.country}/clean_data/df_player_cleaned.xlsx', index=True)
            df_player_sofifa.to_excel(f'./p3_data_preparation/data/{self.country}/clean_data/df_player_sofifa_cleaned.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'./p3_data_preparation/data/{self.country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx', index=True)
            df_teams_sofifa.to_excel(f'./p3_data_preparation/data/{self.country}/clean_data/df_teams_sofifa_cleaned.xlsx', index=True)

        return df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa

    def integrate_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame, df_teams: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, df_teams_sofifa: pd.DataFrame, export: bool = True):
        """
        Integra los datos de partidos y players en un solo dataframe.

        :param df_match: Dataframe de los datos de los partidos.
        :param df_match_player: Dataframe de los datos de los players en cada partido.
        :param df_player: Dataframe de los datos de los players.
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de
        lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        start = time.time()
        print("\nIntegrating data...")

        # TEAMS --> MATCH  (Mapeo df_teams_sofifa con df_teams e integro a df_match)
        print("\nIntegrating team's data to df_match...")
        df_map_teams_fs_so = match_dataframes_by_str_column(df_teams, df_teams_sofifa, column_to_relation='team_name', column_to_integrate='id_team', thr_coincidence_min=90)
        df_match = integrate_team_data_in_match(df_match, df_map_teams_fs_so, df_teams_sofifa)

        # PLAYERS --> MATCH (Mapeo df_player_sofifa con df_player e integro a df_match)
        print("\nIntegrating player's data to df_match...")
        df_map_players_fs_so = match_dataframes_by_str_column(df_player, df_player_sofifa, column_to_relation="player_name", column_to_integrate='id_player', thr_coincidence_min=90)
        df = integrate_player_data_in_match(df_match, df_match_player, df_map_players_fs_so, df_player_sofifa, df_player_fifa_sofifa)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")
        
        if export:
            df_map_teams_fs_so.to_excel(f"./p3_data_preparation/data/{self.country}/integrate_data/df_map_teams_fs_so.xlsx")
            df_map_players_fs_so.to_excel(f"./p3_data_preparation/data/{self.country}/integrate_data/df_map_players_fs_so.xlsx")
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_integrated.xlsx', index=True)

        return df

    def construct_data(self, df: pd.DataFrame, n_days: int, n_years_h2h: int, segun_localia: bool, export: bool = True):
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de players. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        start = time.time()
        print("\nConstructing data...")

        # Construyo variables: "result" y points obtenidos
        df = construct_data.determine_result(df, self.var_resp)
        df = construct_data.determine_points(df)
        # Diferencia en cantidad de ultimos partidos
        df = construct_data.determine_number_matches_last_days(df, n_days=n_days) # numero de partidos jugados en ultimos n days
        # df['dif_n_matches_last_days'] = df['n_matches_last_days_home'] - df['n_matches_last_days_away']
        # df = df.drop(['n_matches_last_days_home', 'n_matches_last_days_away'], axis=1)  

        # Construyo variables porcentajes (funciona ok!) No tira error de division ni nada. Es nan solo cuando es 0/0 (sin tiirar error).
        # df['perc_attendance'] = df["attendance"] / df["capacity"]
        df = df.drop(['attendance', 'capacity'], axis=1)  # --> generan problemas de convergencia por ser nros altos y ademas su info puede ser importante junta y no separada.        
     
        # STATS
         # Elimino estadisticas que no quiero promediar porque no sirven y solo introducen ruido en el analisis
        print(df.shape)
        df = df.drop(['clearances_completed_home', 'clearances_completed_away', 'red_cards_home', 'red_cards_away', 'yellow_cards_home', 'yellow_cards_away', 'offsides_home', 'offsides_away', 'attacks_home', 'attacks_away', 'dangerous_attacks_home', 'dangerous_attacks_away','corner_kicks_home', 'corner_kicks_away', 'blocked_shots_home', 'blocked_shots_away', 'throw-ins_home', 'throw-ins_away', 'goalkeeper_saves_home', 'goalkeeper_saves_away', 'goal_kicks_home', 'goal_kicks_away', 'pass_success_%_home', 'pass_success_%_away', 'free_kicks_home', 'free_kicks_away', 'crosses_completed_home', 'crosses_completed_away', 'shots_off_goal_home', 'shots_off_goal_away'], axis=1)  # --> para mi meten ruido en el analisis...
        df = df.drop(['tackles_home', 'tackles_away', 'completed_passes_home', 'completed_passes_away', 'total_passes_home', 'total_passes_away'], axis=1)
        print(df.shape)

        # Construyo variables porcentajes (funciona ok!) No tira error de division ni nada. Es nan solo cuando es 0/0 (sin tiirar error).
        df = construct_data.construct_percentaje_column(df, col_num="shots_on_goal", col_den="goal_attempts")
        df = construct_data.construct_percentaje_column(df, col_num="goals", col_den="goal_attempts")

        # Determino cuales son las variables stats automaticamente
        l_stats = construct_data.determine_stats_columns(df)
        print(f"Stats a promediar en ultimos partidos: {l_stats}")

        # Por estadistica del partido
        for var in l_stats: # e.g. shots_on_goal
            print(f"\tEstadistica a promediar: {var}", df[f"{var}_home"].dtype, df[f"{var}_away"].dtype)
            
            # Determine la diferencia de la estadistica entre equipo local y visitante de cada partido
            df[f'dif_{var}'] = df[f'{var}_home'] - df[f'{var}_away']  # (e.g. dif_goles = goles_home - goles_away)
            df = df.drop([f'{var}_home', f'{var}_away'], axis=1)  # (e.g. borro goles_home y goles_away)

            # Determine para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
            df = construct_data.determine_mean_in_last_matches(df, n_days=n_days, variable=f'dif_{var}', segun_localia=segun_localia, tipo='mean')  # mean_last_match_dif_points_home
            df = df.drop([f'dif_{var}'], axis=1)

            # Determine la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            df[f'dif_mean_last_match_dif_{var}'] = df[f'mean_last_match_dif_{var}_home'] - df[f'mean_last_match_dif_{var}_away']  # KeyError: 'mean_last_match_dif_points_home'
            df = df.drop(columns=[f'mean_last_match_dif_{var}_home', f'mean_last_match_dif_{var}_away'], axis=1)

        # PLAYER
        # Construyo variables de diferencias para las variables promedio de los players
        df = clean_data.replace_nan_with_zero(df, 'n_player_miss_home', 'n_player_miss_away')  # Funciona perfecto
        df = construct_data.suma_rat_player_missing(df)  # Funciona perfecto
        df = construct_data.calculate_dif_col_players(df)
    
        # TEAM
        ## Historial entre si
        df = construct_data.h2h_by_date(df, n_years=16, segun_localia=True)
        df = construct_data.h2h_by_date(df, n_years=16, segun_localia=False)
        df = construct_data.h2h_by_date(df, n_years=n_years_h2h, segun_localia=True)
        df = construct_data.h2h_by_date(df, n_years=n_years_h2h, segun_localia=False)
        ## Rival team de Sofifa
        func = lambda row: 1 if (row['id_team_home_rival_team'] == row['id_team_away']) or (row['id_team_away_rival_team'] == row['id_team_home']) else 0
        df['is_rival_match'] = df.apply(func, axis=1)
        df = df.drop(columns=['id_team_home_rival_team', 'id_team_away_rival_team'], axis=1)

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")
        
        if export:
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_constructed.xlsx', index=True)

        return df

    def etiquetado(self, df: pd.DataFrame, export: bool = True):
        """
        """
        # Elimino columna 'season'
        df = df.drop(['season'], axis=1)  # Arrooja error TypeError porque tiene tanto str como int en los valores originales y el label solo puede recibir un tipo (str o int). Season tiene valores como "2021" y "2020_2021", los primeros los entiende como int y los segundos como str.
        
        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df)

        if export:
            df_etiquetas.to_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx', index=False)
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_constructed_etiquetado.xlsx', index=True)
        return df
    
    def select_data(self, df: pd.DataFrame, thr_corr=None, thr_fs= None, export: bool = True):
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        warnings.filterwarnings('ignore')
        start = time.time()
        n_reg_min = int(0.2*len(df))
        print("\nSelecting data...")

        # Elimino variables que no usare en el modelo fecha (la idea es usar todas las posibles)
        df = df.drop(['date', 'venue'], axis=1)  

        # Elimino columnas constantes
        constant_cols = df.columns[df.nunique() == 1]
        df.drop(columns=constant_cols, inplace=True)
        print(f"Columnas constantes eliminadas: {constant_cols}")

        # Elimino columnas con alto porcentaje de NaN values (de manera que tras el dropna quedarian menos de n_reg_min)
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp] # Separo en X e y
        X_sin_col_mucho_nan = clean_data.drop_columns_until_drop_na_min_rows(X, n_reg_min=n_reg_min) # elimina las columnas hasta que pueda hacer dropna()
        print(f"Shape X_sin_col_mucho_nan: {X_sin_col_mucho_nan.shape}")
        if len(X.columns) != len(X_sin_col_mucho_nan.columns):
            l_col_eliminated = list(X.columns.difference(X_sin_col_mucho_nan.columns))
            text = f"Se han tenido que eliminar {len(X.columns) - len(X_sin_col_mucho_nan.columns)} columnas de {len(X.columns)} porque no se alcanzaba el minimo de {n_reg_min} registros para entrenar el modelo. Columnas eliminadas: {l_col_eliminated}"
            warnings.warn(text)
        df = pd.concat([X_sin_col_mucho_nan, y], axis=1)

        # Elimino variables altamente correlacionadas
        if thr_corr is not None:
            l_columnas_a_eliminar = select_data.delete_correlated_columns(df, self.var_resp, thr_corr)
            df = df.drop(l_columnas_a_eliminar, axis=1)
            print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            n_cols = len(df.columns)-1  # -1 por variable respuesta
            l_important_features = select_data.select_best_features(df, self.var_resp, thr_fs, graf=export)
            l_col_eliminated = list(df.columns.difference(l_important_features))
            df = df.loc[:, l_important_features + [self.var_resp]]
            print(f"\tSe eliminaron {n_cols-len(l_important_features)} de {n_cols} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%. Columnas eliminadas: {l_col_eliminated}")

        print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")
        
        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_selected.xlsx', index=True)
        return df
    
    def treat_nan_values(self, df , fill_na, percentil_nan: int = 75, export: bool = True, _print: bool = True):
        """
        Tratamiento de nan values

        # Parameters
        df: Dataframe a tratar nan values. (DataFrame)
        fill_na: Tipo de rellenado de NaN values.
        percentil_nan: Percentil para definir que columnas son consideradas con mucho nan y cuales con poco nan. Solo cuando haces fillna.
        export: 
        _print:

        # Returns
        Dataframe sin NaN values
        """ 
        print("\nTreating NaN values to avoid input=NaN in Modeling...")
        # Separo en X e y
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

        # Determino las columns con mucho NaN (mas de nan_threshold%)
        if fill_na is not None:
            l_columns_poco_nan, l_columns_mucho_nan = clean_data.determine_columns_to_fill(X, percentil_nan=percentil_nan)

        # Elimino registros con al menos un NaN 
        X = X.dropna(subset=X.columns if fill_na is None else l_columns_poco_nan)  # df = clean_data.delete_rows_nan(X_sin_col_mucho_nan, porc_nan_max=0)
        if _print:
            print(f"De las {len(df)} filas, se han eliminado {len(df)-len(X)} por tener al menos un Nan value. Quedan {len(X)} filas. Shape final: {X.shape}") 

        # Si hay que rellenar, hago el rellenado de columnas con mucho NaN
        if fill_na is not None:
            # Determino que filas relleno y cuales no (antes de fill porque despues de rellenar no puedo diferenciar que filas rellene y cuales no)
            df_rellenado = pd.DataFrame(index=X.index)
            df_rellenado['rellenado'] = X[l_columns_mucho_nan].isnull().any(axis=1)
            if export:
                df_rellenado.to_excel(f'./p3_data_preparation/data/{self.country}/df_rellenado.xlsx', index=True)

            # Relleno nan de las columnas con mucho NaN
            X = clean_data.fill_nan_values(X, l_columns_mucho_nan, fill_type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las accuracyes casi siempre seran mayores que dropna() en train y test, lo que cuenta es la accuracy en next_matches o en un dataset que no haya sido filleado...
            if _print:
                print(f"Columnas consideradas con mucho nan (a las cuales rellenar): {l_columns_mucho_nan}")
                print(f"\tSe realizó el rellenado de NaN values. Shape X luego de rellenado: {X.shape}")

            # Agrego columna rellenado a X (post fill puesto que no quiero limpiar la columna "rellenado")
            X['rellenado'] = df_rellenado['rellenado']

        # Concateno X e y
        y = y[y.index.isin(X.index)]
        df = pd.concat([X, y], axis=1)

        if export:
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_selected_nan.xlsx', index=True)
      
        return df


class Modeling:

    def __init__(self, var_resp: str, var_pred: str, country: str):
        if not isinstance(var_resp, str) or not isinstance(var_pred, str):
            raise TypeError("Los parámetros var_resp y var_pred deben ser cadenas de texto.")
        if not isinstance(country, str):
            raise TypeError("El parámetro country debe ser una cadena de texto.")

        self.var_resp = var_resp
        self.var_pred = var_pred
        self.country = country
        self.make_directories()

    def make_directories(self):

        l_directorios = [
            f'./p4_modeling/data/{self.country.lower()}',
            f'./p4_modeling/data/{self.country.lower()}/generate_test_design',
        ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)
        
    def generate_test_design(self, df: pd.DataFrame, bal_type, val_size: float = 0.15, test_size: float = 0.15, export: bool = True):
        """
        Separa conjuntos de datos en train, validacion y test, balancea las clases del dataset y elimina los NaN values.

        # Parameters
        df: Dataframe a dividir en test, validation y train. (Dataframe)
        bal_type: Tipo de balanceo de clases a realizar. (String)
        val_size: Porcentaje del total de datos destinado a validacion. (Float)
        test_size:  Porcentaje del total de datos destinado a test. (Float)
        export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (Bool)
        
        # Returns
        Dataframe de entrenamiento y de testeo balanceados (DataFrame)
        """
        warnings.filterwarnings('ignore') # no son mias, son de openpyxl
        print("\nSeparating data in train, val and test...")

        # Separo df_test (si rellené, dejo registros sin rellenar)
        if 'rellenado' in df.columns:
            print("\tDejo registros no rellenados en df_test.")

            # Obtengo indice de filas no rellenadas
            index_no_rellenado = df[~df['rellenado']].index
            df = df.drop('rellenado', axis=1)

            # Todos los registros con al menos un NaN value los guardo en el conjunto de entrenamiento
            n_reg_test = int(len(df) * test_size)
            n_reg_test_max = len(index_no_rellenado)
            if n_reg_test > n_reg_test_max: # Si no hay suficientes filas no rellenadas disponibles
                # Ajusta n para tomar todas las filas no rellenadas disponibles
                print(f"Tamaño que deberia tener df_test: {n_reg_test} pero hay solo {n_reg_test_max} registros disponibles (pues son solo los registros que no han sido rellenados)")
                n_reg_test = n_reg_test_max

            # Construyo el dataset de prueba a partir de registros que no han sido rellenados
            df_test = df.loc[index_no_rellenado].sample(n_reg_test, random_state=42) # df_test = df[~df_rellenado['rellenado']].sample(n, random_state=42)
            X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]

            # Construyo train y val a partir de las filas que quedan
            df_train_val = df[~df.index.isin(df_test.index)]
            X_train_val, y_train_val = df_train_val.drop(self.var_resp, axis=1), df_train_val[self.var_resp]

        else:
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=randint(1, 1000), shuffle=True)

        # Calcula el tamaño relativo del conjunto de prueba
        test_size_ratio = len(X_test) / len(df)
        # Calcula el tamaño relativo del conjunto de validación
        val_size_ratio = val_size / (1 - test_size_ratio)

        # Balanceo el dataset de entrenamiento y validacion
        if bal_type is not None:
            X_train_val, y_train_val = generate_test_design.balance_dataset(X_train_val, y_train_val, bal_type=bal_type)

        # Separo train y validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_ratio, random_state=randint(1, 1000), shuffle=True)
        print(f'Train: {X_train.shape} {y_train.shape}', f'\nVal: {X_val.shape} {y_val.shape}', f'\nTest: {X_test.shape} {y_test.shape}')

        if export:
            X_train.to_excel(f'./p4_modeling/data/{self.country}/generate_test_design/X_train.xlsx', index=True)
            X_val.to_excel(f'./p4_modeling/data/{self.country}/generate_test_design/X_val.xlsx', index=True)
            X_test.to_excel(f'./p4_modeling/data/{self.country}/generate_test_design/X_test.xlsx', index=True)
            y_train.to_excel(f'./p4_modeling/data/{self.country}/generate_test_design/y_train.xlsx', index=True)
            y_val.to_excel(f'./p4_modeling/data/{self.country}/generate_test_design/y_val.xlsx', index=True)
            y_test.to_excel(f'./p4_modeling/data/{self.country}/generate_test_design/y_test.xlsx', index=True)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, model, X_val: pd.DataFrame, y_val: pd.DataFrame, X_train: pd.DataFrame, y_train, k: int, params: dict = None, export: bool = True):
        """
        Selecciona el mejor modelo a partir de la accuracy.
        :param model: Modelo de Machine Learning. (sklearn.ensemble)
        :param X_val: Dataframe de validacion con variables predictoras. (DataFrame)
        :param y_val: Dataframe de validacion solo con variable respuesta. (DataFrame)
        :param X_train: Dataframe de entrenamiento con variables predictoras.  (DataFrame)
        :param y_train: Dataframe de entrenamiento solo con variable respuesta. (DataFrame)
        :param k: Numero de folds. (int)
        :param timeout: Cantidad de segundos de espera maxima para entrenar un modelo. (int)
        :return: Mejor modelo. (sklearn.ensemble?)
        """
        warnings.filterwarnings("ignore")
        print("\nTraining model...")
        
        # Find best hiperparameters
        if params is None:
            model_best_params = build_model.select_best_hiperparameters(model, X_val, y_val, k=10, _print=True)
        else:
            model_best_params = model.set_params(**params)
            # DEBERIA CONCATENAR X_VAL E Y_VAL A X_TRAIN E Y_TRAIN PUESTO QUE SINO ESTOY TIRANDO DATOS AL TACHO.

        d_best_hiper = model_best_params.get_params()
        print("Hiperparametros:", d_best_hiper)

        # Fit model
        model_best_params.fit(X_train, y_train)

        # Evaluo el modelo con Cross Validation
        cv_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
        print(f"\nAccuracy promedio de validación cruzada: {cv_accuracy:.1f}%")

        if export:
            pickle.dump(model_best_params, open(f"./p4_modeling/data/{self.country}/modelo.pkl", "wb"))
            df_hiperparametros = pd.DataFrame.from_dict(d_best_hiper, orient='index', columns=['Valor'])
            df_hiperparametros.to_csv(f"./p4_modeling/data/{self.country}/modeling/hiperparametros.csv")

        return d_best_hiper, model_best_params, cv_accuracy

    def assess_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, export: bool = False, _print: bool = True):
        """
        Evalúa un modelo de machine learning utilizando datos de prueba y calcula métricas de desempeño.

        :param model: Modelo de Machine Learning entrenado. (sklearn.ensemble)
        :param X_test: Dataframe de prueba con variables predictoras. (DataFrame)
        :param y_test: Dataframe de prueba solo con variable respuesta. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el DataFrame seleccionado. True para exportar, False
        de lo contrario. (bool)
        :return: Precisión del modelo y ROI en el conjunto de prueba. (int) y (float)
        """
        print("\nEvaluating trained model with test sets...")
        # Levanto df_match_odds (solo los partidos en X_test)
        df_match_odds = pd.read_excel(f'./p2_data_understanding/data/{self.country}/df_match_odds.xlsx', index_col=0)
        df_match_odds = df_match_odds[df_match_odds.index.isin(X_test.index)]  # Selecciono los partidos que estan en df_test
        df_match_odds = df_match_odds.reindex(X_test.index)  # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test
        self.var_pred_bm = 'bookmaker_result'  

        # Predecir las etiquetas para los datos de prueba
        y_pred_prob = model.predict_proba(X_test) # Te da las probabilidad de cada clase. Funciona para todos los modelos? # AttributeError: predict_proba is not available when probability=False
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad  # y_pred = model.predict(X_test)  # es un numpy array
        df_pred_proba = pd.DataFrame({self.var_resp: y_test, self.var_pred: y_pred, f'prob_class_{model.classes_[0]}': y_pred_prob[:, 0], f'prob_class_{model.classes_[1]}': y_pred_prob[:, 1], f'prob_class_{model.classes_[2]}': y_pred_prob[:, 2]}, index=X_test.index)
   
        # Calculo metricas
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100

        # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
        df_conf_mat = asses_model.confusion_matrix(y_test, y_pred)

        # Agrego predicciones de bookmaker
        df_match_odds = asses_model.calculate_probas_bookmarker(df_match_odds) # Caculo probabilidades segun casa de apuesta
        df_match_odds = asses_model.determine_bookmaker_result(df_match_odds, self.var_pred_bm)  # Determino resultado predicho segun cuota minima (e.g. "Home")
        y_pred_bm = df_match_odds[self.var_pred_bm].values

        # Calculo precision de casa de apuesta
        test_precision_bookmaker = accuracy_score(y_test, y_pred_bm) * 100  # Calcula bien tras el reindex()
        dif_prec = test_accuracy - test_precision_bookmaker
        d_metrics = {'test_accuracy': test_accuracy, 'recall': recall, 'f1_score': f1, 'test_accuracy_bm': test_precision_bookmaker, 'dif_prec_bm': dif_prec}
 
        # Concateno dfs
        df_predicciones = pd.concat([df_pred_proba, df_match_odds], axis=1)
        
        # Calculo ROI
        d_roi = asses_model.calculate_roi_by_betting_strategy(df_predicciones)
        d_metrics.update(d_roi)
     
        if _print:
            print(f"\n\nMatriz de confusion:\n {df_conf_mat}")
            print(d_metrics)

        if export:
            df_conf_mat.to_excel(f'./p4_modeling/data/{self.country}/modeling/df_conf_matrix.xlsx')
            df_predicciones.to_excel(f'./p4_modeling/data/{self.country}/modeling/df_predicciones.xlsx')

        return d_metrics
    
    def select_best_model(self, l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, export=True):
        """
        Pruebo varios modelos 
        Me gusta que este en Modeling() (y no en find_best_hyper) puesto que usa build_model y asses_model.
        """
        # Definicion de variables
        d = {}
        best_roi_max = -100000
        
        # Por modelo
        for modelo in l_modelos:
            
            model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")
            print(f" Modelo: {model_name} ".center(120, '-'))

            # Entreno modelo y evaluo su rendimiento     
            try:
                d_best_hiper, model_best_params, cv_accuracy = self.build_model(modelo, X_val, y_val, X_train, y_train, k, export=False)
                d_metrics = self.assess_model(model_best_params, X_test, y_test)

                # Si la precision_test_es mayor, guardar datos...
                if d_metrics['best_roi'] > best_roi_max:
                    best_roi_max = d_metrics['best_roi']
                 
                    # Guardo datos del mejor modelo
                    best_hiper = d_best_hiper
                    best_model = model_best_params
                    d = {'model_name': model_name, 'model_trained': model_best_params, 'train_cv_accuracy': cv_accuracy}
                    d.update(d_metrics)

            except KeyboardInterrupt:
                print("Se evitó entrenar este modelo")
        
        return best_hiper, best_model, d

##################################################### MAIN #####################################################
def main():
    """
    Extraction, processing and analysis of matches to predict match results.
    """
    # Definicion de variables
    country = 'England'  # country = str(input("Choose country to extract (e.g. England, Germany, etc): "))
    var_resp, var_pred = 'result', 'predicted_result'
    data_unders, data_prep, modeling = False, True, True
    export = True
     
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]

    # Creo instancias de clases
    du = DataUnderstanding(id_country, country) # Creo objeto de clase DataPreparation
    dp = DataPreparation(country) # Creo objeto de clase DataPreparation
    mo = Modeling(var_resp=var_resp, var_pred=var_pred, country=country)  # Creo objeto de clase Modeling

    #------------------------------------------- DATA UNDERSTANDING -------------------------------------------#
    if data_unders:
        print(" Data understanding ".center(120, "#"))
        # Extriago datos o los levanto
        df_match, df_match_player, df_match_odds, df_player, df_teams, df_coaches, df_player_sofifa, df_player_fifa_sofifa = du.collect_initial_data(export=export)
        # df_teams_sofifa = pd.read_excel('./p2_data_understanding/data/{country}/df_teams_sofifa.xlsx', index_col=0)

        # Describo datos
        du.describe_data(df_match, df_match_player, df_match_odds, df_player, df_teams, df_coaches, df_player_sofifa, df_player_fifa_sofifa)

    elif data_prep:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_odds.xlsx', index_col=0)
        df_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx', index_col=0)
        df_teams = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams.xlsx', index_col=0)
        df_coaches = pd.read_excel(f'./p2_data_understanding/data/{country}/df_coaches.xlsx', index_col=0)

        df_player_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_sofifa.xlsx', index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_fifa_sofifa.xlsx')
        df_teams_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams_sofifa.xlsx', index_col=0)

        # du.describe_data(df_match, df_match_player, df_match_odds, df_player, df_teams, df_coaches, df_player_sofifa, df_player_fifa_sofifa)

    #------------------------------------------- DATA PREPARATION -------------------------------------------#
    if data_prep:
        print(" Data preparation ".center(120, "#"))
        # Hiperparametros # PODRIA PONERLOS EN UN DICT Y HACER EL DATAFRAME MAS AUTOMATICO
        n_days, n_years_h2h, segun_localia = 30, 3, False
        thr_corr, thr_fs = 0.7, 0.1
        fill_na = None
        df_hiper_prep = pd.DataFrame(data={'n_days': [n_days], 'n_years_h2h': [n_years_h2h], 'segun_localia': [segun_localia], 'thr_corr': [thr_corr], 'thr_fs': [thr_fs], 'fill_na': [fill_na]}, index=[0])
        
        # df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_selected.xlsx', index_col=0)
        # print(df.head(2))

        # Preparo el dataset para el analisis
        df_match, df_match_player, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_player_fifa_sofifa, export=False)
        df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export)
        df = dp.integrate_data(df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export) 
        df = dp.construct_data(df, n_days=n_days, n_years_h2h=n_years_h2h, segun_localia=segun_localia, export=export)
        df = dp.etiquetado(df)
        df = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, export=export)
        df = dp.treat_nan_values(df, fill_na=fill_na, export=export)
        
        if export:
            df_hiper_prep.to_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx', index=False)

    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_selected_nan.xlsx', index_col=0)
        print(df.head(3), df.shape)

    #------------------------------------------- MODELING -------------------------------------------#
    if modeling:
        print(" Modeling ".center(120, "#"))
        # Hiperparametros
        val_size, test_size = 0.125, 0.125
        bal_type = None
        k = 10
        df_hiper_mod = pd.DataFrame(data={'val_size': [val_size], 'test_size': [test_size], 'bal_type': [bal_type], 'k': [k]}, index=[0])        

        l_modelos = [LogisticRegression(), RandomForestClassifier()]
        modelo = RandomForestClassifier()  # LogisticRegression(), RandomForestClassifier()
        model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")

        build_specific_model = False
        if build_specific_model:

            # LogisticRegression(C=0.1, fit_intercept=False, penalty='l1', solver='saga')
            d_params = {
                'RandomForestClassifier': {
                    'n_estimators': 500,
                    'criterion': 'entropy',
                    'max_depth': 3,
                    'min_samples_split': 2, 
                    'min_samples_leaf': 4, 
                    'max_features': 'sqrt',
                    'bootstrap': True,
                },
                'LogisticRegression': {
                    'penalty': 'l1',
                    'C': 10,
                    'solver': 'liblinear',
                    'fit_intercept': True,
                    'max_iter': 10000, 
                    'multi_class': 'auto',
                },
            }
            hiperparametros = d_params[model_name]
        else:
            hiperparametros = None

        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df, bal_type, val_size, test_size, export=export)

        # Analizo datos con un modelo
        d_best_hiper, model_best_params, cv_accuracy = mo.build_model(modelo, X_val=X_val, y_val=y_val, X_train=X_train, y_train=y_train, k=k, params=hiperparametros, export=export)
        d_metrics = mo.assess_model(model_best_params, X_test, y_test, export=export)

        # Analizo mas de un modelo
        # d_best_hiper, model_best_params, d_best_model = mo.select_best_model(l_modelos, X_val=X_val, y_val=y_val, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, k=k, export=False)

        if export:
            df_hiper_mod.to_excel(f'./p4_modeling/data/{country}/modeling/df_hiper_mod.xlsx', index=True)
            pickle.dump(model_best_params, open(f"./p4_modeling/data/{country}/modeling/modelo.pkl", "wb"))

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()