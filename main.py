# Importo librerias
import pandas as pd
import numpy as np
import os
import datetime
from set_up_logging import logger
## Data understanding
from p2_data_understanding.collect_initial_data import scraper_flashscore, scraper_sofifa
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
from sklearn.preprocessing import StandardScaler
import joblib
## Modeling
from p4_modeling import generate_test_design, build_model, asses_model
### Generate test design
from random import randint
from sklearn.model_selection import train_test_split
### Build model
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
# import xgboost as xgb  # XGBoost
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
        self.country = country.lower()
        self.make_directories()

    def make_directories(self):
        ruta_base = f'./data/{self.country}/p2_data_understanding/data_seg'
        l_directorios = [f'{ruta_base}/per_season/df_match/',
                        f'{ruta_base}/per_season/df_match_player/',
                        f'{ruta_base}/per_season/df_match_odds/',                    
                        f'{ruta_base}/per_season/df_player_sofifa/',
                        f'{ruta_base}/per_season/df_player_fifa_sofifa/',
                        f'{ruta_base}/per_competition/df_match/',
                        f'{ruta_base}/per_competition/df_match_player/',
                        f'{ruta_base}/per_competition/df_match_odds/',
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
        logger.info(" Collecting data... ")
        # Defino variables
        df_match_concat, df_match_player_concat,df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Flashscore
        df_player_sofifa_concat, df_player_fifa_sofifa_concat, df_teams_sofifa_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Sofifa

        # Selecciono competencias del country
        df_comp = pd.read_excel('./data/df_competencies.xlsx')
        df_comp_country = df_comp[(df_comp['id_country'] == self.id_country)]
        print(f' COUNTRY: {self.country} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_country['competition_flashscore']}")

        # POR COMPETITION
        for i, row in df_comp_country.iterrows():
            print(f' Competition: {row["competition_flashscore"]} '.center(120, '+'))

            # Extraigo partidos de Flashscore (df_match y df_match_player)
            df_match, df_match_player, df_match_odds = scraper_flashscore.extract_data(self.id_country, self.country, row['id_competition'], row['competition_flashscore'], row['is_cup'], export=export)
            
            # Guardo datos
            df_match_concat = pd.concat([df_match_concat, df_match], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player], axis=0)
            df_match_odds_concat = pd.concat([df_match_odds_concat, df_match_odds], axis=0) 
            if export:
                df_match_concat.to_excel(f'./data/{self.country}/p2_data_understanding/data_seg/df_match.xlsx', index=True)
                df_match_player_concat.to_excel(f'./data/{self.country}/p2_data_understanding/data_seg/df_match_player.xlsx', index=True)
                df_match_odds_concat.to_excel(f'./data/{self.country}/p2_data_understanding/data_seg/df_match_odds.xlsx', index=True)
                
            # Si la competition es una liga
            if row['is_cup'] == 0:

                # Extraigo datos de players de Sofifa 
                ## Player
                df_player_sofifa, df_player_fifa_sofifa = scraper_sofifa.extract_players(self.id_country, self.country, row['id_competition'], row['competition_sofifa'], export=export)
                df_player_sofifa_concat = pd.concat([df_player_sofifa_concat, df_player_sofifa], axis=0)
                df_player_fifa_sofifa_concat = pd.concat([df_player_fifa_sofifa_concat, df_player_fifa_sofifa], axis=0)
                if export:
                    df_player_sofifa_concat.to_excel(f'./data/{self.country}/p2_data_understanding/data_seg/df_player_sofifa.xlsx', index=True)
                    df_player_fifa_sofifa_concat.to_excel(f'./data/{self.country}/p2_data_understanding/data_seg/df_player_fifa_sofifa.xlsx', index=True)
               
                ## Teams
                df_teams = scraper_sofifa.extract_teams(self.id_country, self.country, row['competition_sofifa'])
                df_teams_sofifa_concat = pd.concat([df_teams_sofifa_concat, df_teams], axis=0)
                if export:
                    df_teams_sofifa_concat.to_excel(f'./data/{self.country}/p2_data_understanding/data_seg/df_teams_sofifa.xlsx', index=True)
       
        # Exporto datasets con competiciones del country
        if export:
            df_match_concat.to_excel(f'./data/{self.country}/p2_data_understanding/df_match.xlsx', index=True)
            df_match_player_concat.to_excel(f'./data/{self.country}/p2_data_understanding/df_match_player.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./data/{self.country}/p2_data_understanding/df_match_odds.xlsx', index=True)
            df_player_sofifa_concat.to_excel(f'./data/{self.country}/p2_data_understanding/df_player_sofifa.xlsx', index=True)
            df_player_fifa_sofifa_concat.to_excel(f'./data/{self.country}/p2_data_understanding/df_player_fifa_sofifa.xlsx', index=True)
            df_teams_sofifa_concat.to_excel(f'./data/{self.country}/p2_data_understanding/df_teams_sofifa.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat, df_player_sofifa_concat, df_player_fifa_sofifa_concat, df_teams_sofifa_concat

    def describe_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame, df_player_sofifa:  pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame):
        """
        Descripcion basica de los datos recolectados como shape, datatypes, cantidad de NaN por columna, etcetera.
        """
        logger.info(" Describing data... ")

        print("\n DF_MATCH \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match)
        describe_data.verificar_unicidad_registros(df_match) # Verifico unicidad de registros segun campos id

        print("\n DF_MATCH_PLAYER \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_player)

        print("\n DF_MATCH_ODDS \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_odds)

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
            f'./data/{self.country}/p3_data_preparation/clean_data',
            f'./data/{self.country}/p3_data_preparation/integrate_data',
            f'./data/{self.country}/p3_data_preparation/select_data',
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
        logger.info("\nFormatting data...")

        # Dataframe match
        ## Date
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        ## Capacity & Attendance
        df_match = format_data.convert_capacity_to_int(df_match)
        ## Ball posession
        df_match = format_data.convert_ball_possession_to_int(df_match)
        ## Goals
        df_match = format_data.convert_goals_to_int(df_match)
        df_match_player = df_match_player[df_match_player.index.isin(df_match.index)]
        ## Todas las columnas
        df_match = format_data.convert_columns_to_float(df_match)  # Formateo estadisticas a float (no se por que son object)
        # df_match_odds = convert_columns_to_float(df_match_odds)  # Formateo estadisticas a float (no se por que son object)

        # Dataframe player_fifa_sofifa
        ## Fecha
        df_player_fifa_sofifa['date'] = pd.to_datetime(df_player_fifa_sofifa['date'], format='%b %d, %Y')
        ## Market value
        df_player_fifa_sofifa = format_data.convert_value_to_int(df_player_fifa_sofifa)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_match.to_excel(f'./data/{self.country}/p3_data_preparation/df_match_formated.xlsx', index=True)
            df_match_player.to_excel(f'./data/{self.country}/p3_data_preparation/df_match_player_formated.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'./data/{self.country}/p3_data_preparation/df_player_fifa_sofifa_formated.xlsx', index=True)

        return df_match, df_match_player, df_player_fifa_sofifa

    def clean_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, df_teams_sofifa: pd.DataFrame, export: bool = True):
        """
        Limpieza inicial de los dataframes
        """
        start = time.time()
        logger.info("\nCleanning data...")

        # Elimino partidos viejos sin estadisticas y sin datos de jugadores --> PROBAR!
        n_rows_inic = len(df_match)
        df_match['date'] = pd.to_datetime(df_match['date'])  # Asegurarte de que la columna 'date' sea de tipo datetime (si no lo es ya)
        start_date = '2012-01-01' # Filtrar por fecha (por ejemplo, para filtrar datos desde una fecha específica)
        df_match = df_match[df_match['date'] >= start_date]
        df_match_player = df_match_player[df_match_player.index.isin(df_match.index)] # Es clave para eliminar jugadores y hacer una mejor integracion (tener menos falsos positivos)
        logger.info(f"Partidos jugados antes de {start_date} eliminados. {n_rows_inic} --> {len(df_match)}. {len(df_match_player)}")

        # Elimino columnas de jugadores que son todo NaN (se ve que hay porque las creo y no les guardo nada eso debe ser porque obtengo nombres solo si tiene url)
        non_object_columns = df_match_player.select_dtypes(exclude=['object']).columns
        df_match_player.drop(columns=non_object_columns, inplace=True)
        print("Shape df_match_player: ", df_match_player.shape)

        # Preparacion de texto
        print("\nPreparacion de columnas string")
        ## FLASHSCORE
        columns_to_keep = [col for col in df_match.columns if df_match[col].dtype == 'object' and 'id_' not in col]
        df_match = clean_data.prepare_text_columns(df_match, l_cols_to_process=columns_to_keep) # Ver si selecciona bien.. # ['team_home', 'team_away', 'coach_home', 'coach_away', 'venue', 'referee'])
        columns_player_names = list(df_match_player.filter(like='player_name').columns)
        df_match_player = clean_data.prepare_text_columns(df_match_player, l_cols_to_process=columns_player_names)
        df_match = clean_data.clean_teams_names(df_match)  # una vez que ya aplique el lower()

        ## SOFIFA
        ### Elimino jugadores duplicados por haber jugado mas de una competicion del pais
        len_inic = len(df_player_sofifa)
        df_player_sofifa = df_player_sofifa[~df_player_sofifa.index.duplicated(keep='first')]
        df_player_fifa_sofifa = df_player_fifa_sofifa.drop_duplicates() # No por id_player porque no es unico (hay 2 por fifa)
        n_players_eliminated = len_inic - len(df_player_sofifa)
        if n_players_eliminated > 0:
            logger.warning(f"Se eliminaron {n_players_eliminated} jugadores de los {len_inic} de Sofifa que habia.")
        ### Dataframe player sofifa (df)
        df_player_sofifa = clean_data.prepare_text_columns(df_player_sofifa, l_cols_to_process=['player_name', 'player_name_short'])  # Preaparo texto para integrar
        ### Dataframe teams sofifa (df_teams_sofifa)
        df_teams_sofifa = clean_data.prepare_text_columns(df_teams_sofifa, l_cols_to_process=['team_name'])

        # Correcion de valores
        print("\nCorrecion de valores")
        df_player_fifa_sofifa['fifa_year'] = df_player_fifa_sofifa['fifa'].str.split(' ').str[-1]  # Agrego columna "fifa_year" quedandome solo con el año del fifa (e.g. "22" en vez de "FIFA 22")
    
        # Verificar que no haya outliers
        # algo (sacar de mi tesis)

        end = time.time()
        print(f"Clean data in {(end - start) / 60:.1f} minutes")

        if export:
            df_match.to_excel(f'./data/{self.country}/p3_data_preparation/clean_data/df_match_cleaned.xlsx', index=True)
            df_match_player.to_excel(f'./data/{self.country}/p3_data_preparation/clean_data/df_match_player_cleaned.xlsx', index=True)
            df_player_sofifa.to_excel(f'./data/{self.country}/p3_data_preparation/clean_data/df_player_sofifa_cleaned.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'./data/{self.country}/p3_data_preparation/clean_data/df_player_fifa_sofifa_cleaned.xlsx', index=True)
            df_teams_sofifa.to_excel(f'./data/{self.country}/p3_data_preparation/clean_data/df_teams_sofifa_cleaned.xlsx', index=True)
    
        return df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa

    def integrate_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, df_teams_sofifa: pd.DataFrame, export: bool = True):
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        Parameters:
            df_match (pd.DataFrame): Dataframe de los datos de los partidos.
            df_match_player (pd.DataFrame): Dataframe de los datos de los jugadores en cada partido.
            df_player_sofifa (pd.DataFrame): Dataframe de los datos de los jugadores en Sofifa.
            df_player_fifa_sofifa (pd.DataFrame): Dataframe de los datos de los jugadores en Fifa-Sofifa.
            df_teams_sofifa (pd.DataFrame): Dataframe de los datos de los equipos en Sofifa.
            export (bool): Booleano para indicar si se debe exportar el dataframe integrado. True para exportar y False para no exportar.
        
        Returns:
            pd.DataFrame: Dataframe integrado.
        """
        start = time.time()
        logger.info("\nIntegrating data...")

        '''
        # TEAMS --> MATCH  (Mapeo df_teams_sofifa con df_teams e integro a df_match)
        print("\nIntegrating team's data to df_match...")
        # Si ya hice el mapeo
        try:
            df_map_teams_fs_so = pd.read_excel(f'data/{self.country}/p3_data_preparation/integrate_data/df_map_teams_fs_so.xlsx')
            print("No vuelvo a mapear sino que levanto df_map ")
        # Si aun no hice el mapeo
        except FileNotFoundError:
            # Matcheo equipos de Sofifa y Flashscore
            print("Mapeo equipos de Sofifa y Flashscore")
            df_teams = create_df_teams(df_match)
            df_map_teams_fs_so = match_dataframes_by_str_column(df1=df_teams, df2=df_teams_sofifa, column_to_match1='team_name', column_to_match2='team_name', column_to_integrate='id_team', thr_coincidence_min=90)

            if export:
                df_teams.to_excel(f"./data/{self.country}/p3_data_preparation/integrate_data/df_teams.xlsx", index=True)
                df_map_teams_fs_so.to_excel(f"./data/{self.country}/p3_data_preparation/integrate_data/df_map_teams_fs_so.xlsx")
        
        # Integro datos de equipos a df_match usando el mapeo       
        df = integrate_team_data_in_match(df_match, df_map_teams_fs_so, df_teams_sofifa)
        '''

        # PLAYERS --> MATCH (Mapeo df_player_sofifa con df_player e integro a df_match)
        print("\nIntegrating player's data to df_match...")
        # Si ya hice el mapeo
        try:
            df_map_players_fs_so = pd.read_excel(f'data/df_map_players_fs_so.xlsx', index_col=0)
            logger.info("No vuelvo a mapear sino que levanto df_map general ")

            # Reemplazo los df_player de Sofifa del pais por los completos
            df_player_sofifa = pd.read_excel(f'data/df_player_sofifa.xlsx', index_col=0)
            df_player_fifa_sofifa = pd.read_excel(f'data/df_player_fifa_sofifa.xlsx', index_col=0)
            
            logger.critical("Integración usando el df_map completo!")

        # Si aun no hice el mapeo
        except FileNotFoundError:

            try:
                df_map_players_fs_so = pd.read_excel(f'data/{self.country}/p3_data_preparation/integrate_data/df_map_players_fs_so.xlsx')
                logger.info("No vuelvo a mapear sino que levanto df_map del pais ")

            except FileNotFoundError:
                # Matcheo jugadores de Sofifa y Flashscore
                logger.info("Mapeo jugadores de Sofifa y Flashscore")
                df_player = create_df_player(df_match_player)
                df_map_players_fs_so = match_dataframes_by_str_column(df1=df_player, df2=df_player_sofifa, column_to_match1="player_name", column_to_match2="player_name", column_to_match2_aux='player_name_short', column_to_integrate='id_player', thr_coincidence_min=90)

                if export:
                    df_player.to_excel(f"./data/{self.country}/p3_data_preparation/integrate_data/df_player.xlsx", index=True)
                    df_map_players_fs_so.to_excel(f"./data/{self.country}/p3_data_preparation/integrate_data/df_map_players_fs_so.xlsx")

        # Integro datos de jugadores a df_match usando el mapeo
        df, df_aux = integrate_player_data_in_match(df_match, df_match_player, df_map_players_fs_so, df_player_sofifa, df_player_fifa_sofifa)

        # Drop de columnas que use para df_teams, df_player, df_coaches, etc..
        cols_to_drop = ['team_home', 'team_away', 'coach_home', 'coach_away'] # main_next a veces no tiene coaches.. deeberia copiar antes...
        cols_to_drop_filt = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(cols_to_drop_filt, axis=1)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")
        
        if export:
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_integrated.xlsx', index=True)
            df_aux.to_excel(f'./data/{self.country}/p3_data_preparation/integrate_data/n_players_integrated.xlsx', index=True)

        return df

    def construct_data(self, df: pd.DataFrame, l_days: list , n_years_h2h: int, segun_localia: bool, with_h2h: bool = False, with_historic: bool = True, dif_con_against: bool = True, export: bool = True):
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de players. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        start = time.time()
        logger.info("Constructing data...")

        # Elimino columnas "Ruido"
        df = df.drop(['attendance', 'capacity'], axis=1)  # --> generan problemas de convergencia por ser nros altos y ademas su info puede ser importante junta y no separada.        

        # Si quiero construir variables historicas
        if with_historic:

            # VARIABLE RESPUESTA (no son historicas estan filtradas)
            df = construct_data.determine_result(df, self.var_resp)
            df = construct_data.determine_points(df)

            # VARIABLES DERIVADAS
            # Expected Result and Expected Points (xPts) (from Expected Goals)
            df = construct_data.determine_expected_result(df)
            df = construct_data.determine_expected_points(df)
            df = df.drop(['expected_result'], axis=1) 

            ## OFENSIVE
            ## Goal ratio
            df = construct_data.construct_percentaje_column(df, col_num='goals', col_den="goal_attempts", laplace=True,  column_name="goal_ratio")  # Similar a G2A

            # Traduccion de posesion a tiros
            df = construct_data.construct_percentaje_column(df, col_num='goal_attempts', col_den="total_passes", laplace=True,  column_name="PPS")  # home = home / home

            # Dead balls
            df = construct_data.construct_sum_columns(df, l_columns=['throw-ins', 'corner_kicks', 'free_kicks'], column_name="dead_balls") #  # home = home + home

            # Attacking efficiency --> (lo evito por cantidad de NaN)
            # df['attacking_efficiency_home'] = np.where(df['expected_goals_(xg)_home'].notna(), df['goals_home'] - df['expected_goals_(xg)_home'], None)
            # df['attacking_efficiency_away'] = np.where(df['expected_goals_(xg)_away'].notna(), df['goals_away'] - df['expected_goals_(xg)_away'],  None)

            ## DEFENSIVE 
            ## Passess per defensive action (PPDA) --> (no es solamente en el 60% de la cancha pues no tengo ese dato)
            df = construct_data.construct_sum_columns(df, l_columns=['fouls', 'tackles', 'interceptions', 'clearances', 'blocked_shots'], column_name="defensive_actions") # Calculo defesive actions  # home = home + home
            df['PPDA_home'] = np.where(df['defensive_actions_home'].notna(),  df['total_passes_away'] / df['defensive_actions_home'], None)
            df['PPDA_away'] = np.where(df['defensive_actions_away'].notna(), df['total_passes_home'] / df['defensive_actions_away'],  None)

            # Clean Sheets
            df['clean_sheet_home'] = (df['goals_away'] == 0).astype(int)
            df['clean_sheet_away'] = (df['goals_home'] == 0).astype(int)

            # Keeping Goals Prevented (KGP)
            df['KGP_home'] = np.where(df['dangerous_attacks_away'].notna(),  (df['goals_away'] + 1) / df['dangerous_attacks_away'], None)
            df['KGP_away'] = np.where(df['dangerous_attacks_home'].notna(),  (df['goals_home'] + 1) / df['dangerous_attacks_home'],  None)

            # Defensive efficiency (en la teoria esto es KGP) --> (lo evito por cantidad de NaN)
            # df['defensive_efficiency_home'] = np.where(df['expected_goals_(xg)_away'].notna(),  df['expected_goals_(xg)_away'] - df['goals_away'], None)
            # df['defensive_efficiency_away'] = np.where(df['expected_goals_(xg)_home'].notna(), df['expected_goals_(xg)_home'] - df['goals_home'],  None)

            # Efficiency --> (lo evito por cantidad de NaN)
            # df['efficiency_home'] = df['attacking_efficiency_home'] + df['defensive_efficiency_home']
            # df['efficiency_away'] =df['attacking_efficiency_away'] + df['defensive_efficiency_away']

            # VARIABLES HISTORICAS
            # Numero de partidos jugados en ultimos n days
            for n_days in l_days:
                df = construct_data.determine_number_matches_last_days(df, n_days=n_days) 
  
            # Historiales
            if with_h2h:
                df = construct_data.h2h_by_date(df, n_years=-1)
                df = construct_data.h2h_by_date(df, n_years=n_years_h2h)
                df = construct_data.h2h_by_date_by_localia(df, n_years=-1)  # TENEMOOS QUE DARLE EL DF ADICIONAL CON EL CUAL CALCULAR EL HISTORIAL SOLO PARA EL DF ORIGINAL
                df = construct_data.h2h_by_date_by_localia(df, n_years=n_years_h2h)

            # Determino cuales son las variables stats automaticamente
            stats_columns = construct_data.determine_stats_columns(df)
            relevant_stats_columns = [
                # Ofensive
                'expected_goals_(xg)', 'expected_points', # 'expected_result', --> la tengo que eliminar? si no la uso, si. Es medio dificil calcular el promedio en ultimos partidos... es como el historial...
                'shots_on_goal', 'goal_attempts', 'goals', 'points', 'goal_ratio', 'PPS', # 'shots_off_goal'
                'attacks', 'dangerous_attacks', 'dead_balls', # 'goal_ratio_dead_balls',
                'ball_possession', 'total_passes', # 'attacking_efficiency',
                # Defensive
                'yellow_cards', 'red_cards', 'defensive_actions', # 'fouls',  'interceptions'
                'PPDA', 'KGP', 'clean_sheet' # 'defensive_efficiency', "efficiency"
            ] 
            df = clean_data.delete_not_relevant_stats(df, stats_columns, relevant_stats_columns)
            logger.info(f"Stats a promediar en ultimos partidos: {relevant_stats_columns}")

            # Por stat (e.g. shots_on_goal)
            for var in relevant_stats_columns: 
                logger.info(f"Estadistica a promediar: {var}")
                
                # Por periodo de tiempo en el que calcular promedio
                for n_days in l_days:
                
                    # Calculo promedio de stats en ultimos partidos y la diferencia entre local y visitante
                    try:
                        # Determine para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
                        df = construct_data.determine_mean_in_last_matches(df, n_days=n_days, variable=var, segun_localia=segun_localia, dif_con_against=dif_con_against)  # mean_last_match_dif_points_home

                    except KeyError as e:
                        logger.warning(f"Fallo la construccion de {var} por error {e}. Posibles causas: \n 1) Deberia ser porque hay muy pocos ultimos partidos. \n 2) En algun caso particular, si es una sola variable, puede que realmente no tenga valor en los ultimos partidos (En USA, no miedieron expected goals durante 1 mes y era NaN en todos los ultimos partidos)")
                        
                        # Construyo las variables para evitar KeyError mas adelante
                        df[f'dif_mean_last_{n_days}_matches_{var}'] = np.nan  # relleno con nan y no con 0
                        df[f'dif_mean_last_{n_days}_matches_{var}_against'] = np.nan # relleno con nan y no con 0
                
                # Elimino variables stat
                df = df.drop([f'{var}_home', f'{var}_away'], axis=1)  # (e.g. borro goles_home y goles_away)
                        
            # Historica de jugadores
            try:
                df = construct_data.determine_mean_in_last_matches(df, n_days, variable='mean_rat_player_start', segun_localia=segun_localia, calculate_dif=True, dif_con_against=dif_con_against) # Variable para ponderar estadisticas
                n_days_final = n_days * 2 if segun_localia else n_days
                df = df.drop([f'dif_mean_last_{n_days_final}_matches_mean_rat_player_start'], axis=1)  # Solo dejo against. Es para tener medida de los rivales

            except KeyError:
                # Construyo las variables para evitar KeyError mas adelante
                df[f'mean_last_{n_days}_matches_mean_rat_player_start_home_against'] = np.nan # relleno con nan y no con 0
                df[f'mean_last_{n_days}_matches_mean_rat_player_start_away_against'] = np.nan # relleno con nan y no con 0

        # VARIABLE DE JUGADORES
        df = construct_data.calculate_dif_col_players(df)  # Construyo variables de diferencias para las variables promedio de los players

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")
        
        if export:
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_constructed.xlsx', index=True)

        return df

    def tag_string_data_to_integer(self, df: pd.DataFrame, export: bool = True):
        """
        Conversion de columnas tipo "object" a "integer"
        """
        # Elimino columna 'season'
        df = df.drop(['season'], axis=1)  # Arroja error TypeError porque tiene tanto str como int en los valores originales y el label solo puede recibir un tipo (str o int). Season tiene valores como "2021" y "2020_2021", los primeros los entiende como int y los segundos como str.
        
        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df)

        if export:
            df_etiquetas.to_excel(f'./data/{self.country}/p3_data_preparation/df_etiquetas.xlsx', index=False)
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_constructed_etiquetado.xlsx', index=True)
        return df, df_etiquetas
    
    def clean_data_2(self, df: pd.DataFrame, n_years_to_select: int = None, competencies_to_select: list = None, _print: bool = True, export: bool = True):
        """
        Eliminacion de filas y columnas con mucho NaN y escalado de datos

        # Parameters:
            df: Dataframe a tratar NaN y escalar.

        # Returns:
            df: Dataframe pasado como parametro sin filas y columnas con mucho NaN y con datos escalados.
        """
        # Reemplazo infinitos
        df = clean_data.replace_infinite(df)

        # Ordeno valores por fecha y separo X e y
        df = df.sort_values(by='date', ascending=False)
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]  # Separo X e y
        
        # (1) Eliminacion de filas 
        ## Para evitar partidos muy viejos
        print("Eliminacion de filas...")
        n_reg_inic = len(X)
        if n_years_to_select is not None:
            fecha_limite = X.iloc[0]['date'] - datetime.timedelta(days=n_years_to_select*365)
            X = X[X['date'] >= fecha_limite] 
            print(f"Eliminacion por fecha. Cantidad de filas: {n_reg_inic} --> {len(X)}")
        ## Para evitar ciertas competencias
        if competencies_to_select is not None:
            n_reg_inic_2 = len(X)
            X = X[X['id_competition'].isin(competencies_to_select)]
            print(f"Eliminacion por competencias. Cantidad de filas: {n_reg_inic_2} --> {len(X)}")
        ## con mucho NaN (filas sin estadisticas ni formaciones)
        n_reg_inic_3 = len(X)
        X = clean_data.delete_rows_nan(X, 0.6)
        if _print:
            print(f"Eliminaccion por mucho NaN. Cantidad de filas: {n_reg_inic_3} --> {len(X)}")
            logger.warning(f"Cantidad de filas: {n_reg_inic} --> {len(X)}")

        # (2) Eliminacion de columnas   
        print("Eliminación de columnas...")
        ## usadas solo para construir y constantes
        cols_for_construct = ['date', 'venue', 'id_competition', 'id_team_home', 'id_team_away']  # Elimino variables que no usare en el modelo fecha (la idea es usar todas las posibles)
        cols_constants = list(X.columns[X.nunique() == 1])  # Elimino columnas constantes
        X.drop(columns=cols_for_construct+cols_constants, inplace=True)
        ## con mucho NaN --> Elimino columnas con alto porcentaje de NaN values (de manera que tras el dropna quedarian menos de n_reg_min)
        n_reg_min = int(0.15*len(X)) # no uso n_features_min porque hay tengo un millon de columnas extra que eliminare en select...
        X_sin_col_mucho_nan = clean_data.drop_columns_until_drop_na_min_rows(X, n_reg_min=n_reg_min) # elimina las columnas hasta que pueda hacer dropna()

        if _print and len(X.columns) != len(X_sin_col_mucho_nan.columns):
            print(f"Columnas constantes eliminadas: {cols_constants}")
            print(f"Shape X_sin_col_mucho_nan: {X_sin_col_mucho_nan.shape}")
            l_col_eliminated = list(X.columns.difference(X_sin_col_mucho_nan.columns))
            logger.warning(f"Se han tenido que eliminar {len(X.columns) - len(X_sin_col_mucho_nan.columns)} columnas de {len(X.columns)} porque no se alcanzaba el minimo de {n_reg_min} registros para entrenar el modelo. Columnas eliminadas: {l_col_eliminated}")

        # (3) Escalado de datos
        print("\nEscalado de datos...")
        scaler = StandardScaler()
        scaler.fit(X_sin_col_mucho_nan) # Paso 1: Ajusta el StandardScaler a tus datos
        X_scaled = scaler.transform(X_sin_col_mucho_nan) # Paso 2: Transforma tus datos utilizando el StandardScaler ajustado
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_sin_col_mucho_nan.columns, index=X_sin_col_mucho_nan.index)

        # Concateno X e y
        y_sin_nan = y[y.index.isin(X.index)] # Dado que elimine filas de X
        df = pd.concat([X_scaled_df, y_sin_nan], axis=1)

        if export: 
            joblib.dump((scaler, X_sin_col_mucho_nan.columns), f"./data/{self.country}/p3_data_preparation/scaler_model.pkl")       
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_constructed_clean.xlsx', index=True)

        return df, scaler, X_sin_col_mucho_nan.columns
    
    def select_data(self, df: pd.DataFrame, thr_corr: float = None, thr_fs: float = None, export: bool = True):
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        start = time.time()
        logger.info("\nSelecting data...")

        # Elimino variables altamente correlacionadas
        if thr_corr is not None:
            l_columnas_a_eliminar, df_corr_tri_X = select_data.delete_correlated_columns(df, self.var_resp, thr_corr)
            df = df.drop(l_columnas_a_eliminar, axis=1)
            print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            n_cols = len(df.columns)-1  # -1 por variable respuesta
            l_important_features, df_normalized = select_data.select_best_features(df, self.var_resp, thr_fs, graf=export)
            l_col_eliminated = list(df.columns.difference(l_important_features))
            df = df.loc[:, l_important_features + [self.var_resp]]
            print(f"\tSe eliminaron {n_cols-len(l_important_features)} de {n_cols} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%. Columnas eliminadas: {l_col_eliminated}")

        print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")
        
        end = time.time()
        logger.info(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_corr_tri_X.to_excel(f'./data/{self.country}/p3_data_preparation/select_data/df_correlation.xlsx', index=True)
            df_normalized.to_excel(f'./data/{self.country}/p3_data_preparation/select_data/df_fs.xlsx', index=True)
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_selected.xlsx', index=True)

        return df
    
    def treat_nan_values(self, df: pd.DataFrame , fill_na: str = None, percentil_nan: int = 75, export: bool = True, _print: bool = True):
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
        start = time.time()

        # Separo en X e y
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

        # Determino las columns con mucho NaN (mas de nan_threshold%)
        if fill_na is not None:
            l_columns_poco_nan, l_columns_mucho_nan = clean_data.determine_columns_to_fill(X, percentil_nan=percentil_nan)

            # Elimino registros con al menos un NaN 
            X = X.dropna(subset=l_columns_poco_nan)
            if _print:
                print(f"De las {len(df)} filas, se han eliminado {len(df)-len(X)} por tener al menos un Nan value. Quedan {len(X)} filas. Shape final: {X.shape}") 

            # Determino que filas relleno y cuales no (antes de fill porque despues de rellenar no puedo diferenciar que filas rellene y cuales no)
            df_rellenado = pd.DataFrame(index=X.index)
            df_rellenado['rellenado'] = X[l_columns_mucho_nan].isnull().any(axis=1)
            if export:
                df_rellenado.to_excel(f'./data/{self.country}/p3_data_preparation/df_rellenado.xlsx', index=True)

            # Relleno nan de las columnas con mucho NaN
            X = clean_data.fill_nan_values(X, l_columns_mucho_nan, fill_type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las accuracyes casi siempre seran mayores que dropna() en train y test, lo que cuenta es la accuracy en next_matches o en un dataset que no haya sido filleado...
            if _print:
                print(f"Columnas consideradas con mucho nan (a las cuales rellenar): {l_columns_mucho_nan}")
                print(f"\tSe realizó el rellenado de NaN values. Shape X luego de rellenado: {X.shape}")

            # Agrego columna rellenado a X (post fill puesto que no quiero limpiar la columna "rellenado")
            X['rellenado'] = df_rellenado['rellenado']

        else:
            # Elimino registros con al menos un NaN 
            X = X.dropna(subset=X.columns)

        # Concateno X e y
        y = y[y.index.isin(X.index)]
        df = pd.concat([X, y], axis=1)

        end = time.time()
        logger.info(f"Tratamiento de NaN values en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_selected_nan.xlsx', index=True)
        
        return df


class Modeling:

    def __init__(self, var_resp: str, var_pred: str, country: str):
        if not isinstance(var_resp, str) or not isinstance(var_pred, str):
            raise TypeError("Los parámetros var_resp y var_pred deben ser cadenas de texto.")
        if not isinstance(country, str):
            raise TypeError("El parámetro country debe ser una cadena de texto.")

        self.var_resp = var_resp
        self.var_pred = var_pred
        self.country = country.lower()
        # self.make_directories()

    def make_directories(self):

        l_directorios = [
            f'./data/{self.country}/p4_modeling/generate_test_design',
            f'./data/{self.country}/p4_modeling/modeling',
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
        # warnings.filterwarnings('ignore') # no son mias, son de openpyxl
        print("\nSeparating data in train, val and test...")

        # Si rellené NaN values
        if 'rellenado' in df.columns:
            print("\tDejo registros no rellenados en df_test y df_val")

            # Obtengo indice de filas no rellenadas
            index_no_rellenado = df[~df['rellenado']].index
            df = df.drop('rellenado', axis=1)
            print(f"Cantidad de registros no rellenados: {len(index_no_rellenado)}")

            # Determino si hay suficientes registros no rellenados para poner en el dataframe de testeo
            n_reg_test = int(len(df) * test_size)
            n_reg_test_max = len(index_no_rellenado)
            print(f"Numero de registros para df_test: {n_reg_test}")
            if n_reg_test > n_reg_test_max: # Si no hay suficientes filas no rellenadas disponibles
                # Ajusta n para tomar todas las filas no rellenadas disponibles
                print(f"Tamaño que deberia tener df_test: {n_reg_test} pero hay solo {n_reg_test_max} registros disponibles (pues son solo los registros que no han sido rellenados)")
                n_reg_test = n_reg_test_max

            # Construyo el dataset de testeo a partir de registros que no han sido rellenados
            df_test = df.loc[index_no_rellenado].sample(n_reg_test, random_state=42) # df_test = df[~df_rellenado['rellenado']].sample(n, random_state=42)
            X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]
            print(f"Shape df_test: {df_test.shape}")

            # Eliminar los índices de df_test de index_no_rellenado
            indices_a_eliminar = df_test.index
            index_no_rellenado_sin_test = index_no_rellenado.drop(indices_a_eliminar)
            print(f"Cantidad de registros no rellenados disponibles para validacion: {len(index_no_rellenado_sin_test)}")

            # Determino si hay suficientes registros no rellenados para poner en el dataframe de validacion
            n_reg_val = int(len(df) * val_size)
            n_reg_val_max = len(index_no_rellenado_sin_test)
            print(f"Numero de registros para df_val: {n_reg_val}")
            if n_reg_val > n_reg_val_max: # Si no hay suficientes filas no rellenadas disponibles
                # Ajusta n para tomar todas las filas no rellenadas disponibles
                print(f"Tamaño que deberia tener df_val: {n_reg_val} pero hay solo {n_reg_val_max} registros disponibles (pues son solo los registros que no han sido rellenados)")
                n_reg_val = n_reg_val_max

            # Construyo train y val a partir de las filas que quedan
            df_train_val = df[~df.index.isin(df_test.index)]
            df_val = df_train_val.loc[index_no_rellenado_sin_test].sample(n_reg_val, random_state=42) # df_test = df[~df_rellenado['rellenado']].sample(n, random_state=42)
            X_val, y_val = df_val.drop(self.var_resp, axis=1), df_val[self.var_resp]
            print(f"Shape df_train_val: {df_train_val.shape}")
            print(f"Shape df_val: {df_val.shape}")

            # Construyo train con los registros que quedan
            df_train = df_train_val[~df_train_val.index.isin(df_val.index)]
            X_train, y_train = df_train.drop(self.var_resp, axis=1), df_train[self.var_resp]
            print(f"Shape df_train: {df_train.shape}")

        # Si no rellene nan values
        else:
            # Separo test y train_val
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=randint(1, 1000), shuffle=True)

            # Calcula el tamaño relativo del conjunto de validación
            test_size_ratio = len(X_test) / len(df)  # Calcula el tamaño relativo del conjunto de prueba
            val_size_ratio = val_size / (1 - test_size_ratio) 

            # Separo en train y validation
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_ratio, random_state=randint(1, 1000), shuffle=True)

        # Balanceo el dataset de entrenamiento (No se debe balancear el de validacion)
        if bal_type is not None:
            X_train, y_train = generate_test_design.balance_dataset(X_train, y_train, bal_type=bal_type)

        print(f'Train: {X_train.shape} {y_train.shape}', f'\nVal: {X_val.shape} {y_val.shape}', f'\nTest: {X_test.shape} {y_test.shape}')
        if export:
            X_train.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/X_train.xlsx', index=True)
            X_val.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/X_val.xlsx', index=True)
            X_test.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/X_test.xlsx', index=True)
            y_train.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/y_train.xlsx', index=True)
            y_val.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/y_val.xlsx', index=True)
            y_test.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/y_test.xlsx', index=True)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, model, X_val: pd.DataFrame, y_val: pd.DataFrame, X_train: pd.DataFrame, y_train, k: int, params: dict = None, bayes:bool = True, export: bool = True):
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
        # warnings.filterwarnings("ignore")
        print("\nTraining model...")
        self.classes = np.unique(y_train)

        if model == "neural_network": # A diferencia de los otros modelos, la tengo que crear                
            logger.info("Entrenando red neuronal")

            # Creo instancia de clase NeuralNetwork()
            red = build_model.NeuralNetwork()

            # Seleccion mejor arquitectura con la validacion y entreno el modelo
            model_best_params, d_hiper_model, train_accuracy = red.select_best_arquitecture(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        else:
            # Find best hiperparameters
            if params is None:
                model_best_params = build_model.select_best_hiperparameters(model, X_val, y_val, k=5, bayes=bayes,_print=True)
            else:
                model_best_params = model.set_params(**params)
                # DEBERIA CONCATENAR X_VAL E Y_VAL A X_TRAIN E Y_TRAIN PUESTO QUE SINO ESTOY TIRANDO DATOS AL TACHO.

            d_hiper_model = model_best_params.get_params()
            # print("Hiperparametros:", d_hiper_model)

            # Fit model
            model_best_params.fit(X_train, y_train)
            self.classes = model_best_params.classes_

            # Evaluo el modelo con Cross Validation
            train_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
            print(f"\nAccuracy promedio de validación cruzada: {train_accuracy:.1f}%")

        if export:
            pickle.dump(model_best_params, open(f"./data/{self.country}/p4_modeling/modelo.pkl", "wb"))
            df_hiperparametros = pd.DataFrame.from_dict(d_hiper_model, orient='index', columns=['Valor'])
            df_hiperparametros.to_csv(f"./data/{self.country}/p4_modeling/modeling/hiperparametros.csv")

        return model_best_params, d_hiper_model, train_accuracy

    def assess_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, export: bool = False, _print: bool = True):
        """
        Evalúa un modelo de machine learning utilizando datos de prueba y calcula métricas de desempeño.

        # Parameters:
            model: Modelo de Machine Learning entrenado. (sklearn.ensemble)
            X_test: Dataframe de prueba con variables predictoras. (DataFrame)
            y_test: Dataframe de prueba solo con variable respuesta. (DataFrame)
            export: Booleano para indicar si se debe exportar el DataFrame seleccionado. True para exportar, de lo contrario, False.  (bool)

        # Returns:
            Precisión del modelo y ROI en el conjunto de prueba. (int) y (float)
        """
        print("\nEvaluating trained model with test sets...")
        # Levanto df_match_odds (solo los partidos en X_test)
        df_match_odds = pd.read_excel(f'./data/{self.country}/p2_data_understanding/df_match_odds.xlsx', index_col=0)
        df_match_odds = df_match_odds[df_match_odds.index.isin(X_test.index)]  # Selecciono los partidos que estan en df_test
        df_match_odds = df_match_odds.reindex(X_test.index)  # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test
        self.var_pred_bm = 'bookmaker_result'  

        # Predecir las etiquetas para los datos de prueba
        try:
            y_pred_prob = model.predict_proba(X_test) # Te da las probabilidad de cada clase. Funciona para todos los modelos? # AttributeError: predict_proba is not available when probability=False
        except AttributeError: # AttributeError: 'Sequential' object has no attribute 'predict_proba'
            y_pred_prob = model.predict(X_test)
            
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad  # y_pred = model.predict(X_test)  # es un numpy array

        # Crear el DataFrame con las probabilidades
        df_pred_proba = pd.DataFrame({
                self.var_resp: y_test,
                self.var_pred: y_pred,
                f'prob_class_{self.classes[0]}': y_pred_prob[:, 0],  # Probabilidad de la clase 0
                f'prob_class_{self.classes[1]}': y_pred_prob[:, 1],  # Probabilidad de la clase 1
                f'prob_class_{self.classes[2]}': y_pred_prob[:, 2]   # Probabilidad de la clase 2 (si hay 3 clases)
            }, index=X_test.index)
    
        # Calculo metricas
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100

        # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
        df_conf_mat = asses_model.confusion_matrix(y_test, y_pred)

        # Agrego predicciones de bookmaker
        df_match_odds = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta
        df_match_odds = asses_model.determine_result_by_bookmaker(df_match_odds, self.var_pred_bm)  # Determino resultado predicho segun cuota minima (e.g. "Home")
        y_pred_bm = df_match_odds[self.var_pred_bm].values

        # Calculo precision de casa de apuesta
        test_precision_bookmaker = accuracy_score(y_test, y_pred_bm) * 100  # Calcula bien tras el reindex()
        dif_prec = test_accuracy - test_precision_bookmaker
        d_metrics = {'test_accuracy': test_accuracy, 'recall': recall, 'f1_score': f1, 'test_accuracy_bm': test_precision_bookmaker, 'dif_prec_bm': dif_prec}
        
        # Concateno dfs
        df_predicciones = pd.concat([df_pred_proba, df_match_odds], axis=1)
        
        # Calculo ROI
        df_predicciones, d_roi = asses_model.calculate_roi_by_betting_strategy(df_predicciones)
        d_metrics.update(d_roi)
        if _print:
            print(f"\n\nMatriz de confusion:\n {df_conf_mat}")
            print(d_metrics)

        if export:
            df_conf_mat.to_excel(f'./data/{self.country}/p4_modeling/modeling/df_conf_matrix.xlsx')
            df_predicciones.to_excel(f'./data/{self.country}/p4_modeling/modeling/df_predicciones.xlsx')

        return df_predicciones, d_metrics
    
    def train_models(self, l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, ruta_base_mod_seg, cont_iter, export=True):
        """
        Pruebo varios modelos 
        Me gusta que este en Modeling() (y no en find_best_hyper) puesto que usa build_model y asses_model.
        """
        df_metrics = pd.DataFrame()

        # Por modelo
        for modelo in l_modelos:
            
            model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")
            print(f" Modelo: {model_name} ".center(120, '-'))

            # Entreno modelo y evaluo su rendimiento 
            try:
                model, d_hiper_model, cv_accuracy = self.build_model(modelo, X_val, y_val, X_train, y_train, k, bayes=True, export=False)
                df_predicciones, d_metrics = self.assess_model(model, X_test, y_test)

                # Hiperparametros del modelo y Metricas en testeo y train
                new_row = {'n_iteration': cont_iter, 'model_name': model_name, 'cv_accurracy': cv_accuracy, 'model_hiper': d_hiper_model}
                new_row.update(d_metrics)
                df_metrics_new = pd.DataFrame([new_row])  # 1. Convertir el diccionario d_metrics en un DataFrame de una fila
                df_metrics = pd.concat([df_metrics, df_metrics_new], ignore_index=True)  # 2. Concatenar este nuevo DataFrame con df_metrics existente

                # Exporto datos del modelo
                pickle.dump(model, open(f"{ruta_base_mod_seg}/{cont_iter}_{model_name}.pkl", "wb"))
                df_predicciones.to_excel(f'{ruta_base_mod_seg}/{cont_iter}__{model_name}_predicciones.xlsx', index=True)

            except KeyboardInterrupt as e:
                logger.warning(f"Se evitó entrenar este modelo mediante {e}")
                
        return df_metrics

##################################################### MAIN #####################################################
def main(id_country, d_run, export: bool = True):
    """
    Extraction, processing and analysis of matches to predict match results.
    """
    # Definicion de variables
    data_unders, data_prep, modeling = d_run['data_unders'], d_run['data_prep'], d_run['modeling']
    var_resp, var_pred = 'result', 'predicted_result'
    df_countries = pd.read_excel('./data/df_countries.xlsx')
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0].lower()
    
    # Creo instancias de clases
    du = DataUnderstanding(id_country, country) # Creo objeto de clase DataPreparation
    dp = DataPreparation(country) # Creo objeto de clase DataPreparation
    mo = Modeling(var_resp=var_resp, var_pred=var_pred, country=country)  # Creo objeto de clase Modeling

    #------------------------------------------- DATA UNDERSTANDING -------------------------------------------#
    if data_unders:
        print(" Data understanding ".center(120, "#"))
        # Extriago datos o los levanto
        df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = du.collect_initial_data(export=export)

        # Describo datos
        du.describe_data(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa)

    elif data_prep:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match_odds.xlsx', index_col=0)
        df_player_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_player_sofifa.xlsx', index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_player_fifa_sofifa.xlsx') #  index_col=0 --> si lo uso falla la integracion porque pone 'id_player' como index
        df_teams_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_teams_sofifa.xlsx', index_col=0)

        du.describe_data(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa)

    #------------------------------------------- DATA PREPARATION -------------------------------------------#
    if data_prep:
        print(" Data preparation ".center(120, "#"))
        # Hiperparametros # PODRIA PONERLOS EN UN DICT Y HACER EL DATAFRAME MAS AUTOMATICO
        d_comps = select_data.determine_country_competitions(id_country)
        l_days, n_years_h2h, segun_localia, dif_con_against = [30, 180], 3, False, False
        thr_corr, thr_fs = 0.85, 0.5
        n_years_to_select, comp_to_select = 3, d_comps['comp_sin_b']
        fill_na = None
        df_hiper_prep = pd.DataFrame(data={'n_dias_ult_part': [l_days], 'n_anios_hist': [n_years_h2h], 'segun_localia': [segun_localia], 'dif_con_against': [dif_con_against], 'thr_corr': [thr_corr], 'thr_fs': [thr_fs], 'fill_na': [fill_na], 'n_years_to_select': [n_years_to_select], 'comp_to_select': [comp_to_select]}, index=[0])
        
        # df = pd.read_excel(f'./data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)
        # print(df.head(2))

        # Preparo el dataset para el analisis
        df_match, df_match_player, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_player_fifa_sofifa, export=False)
        df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export)
        df = dp.integrate_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export) 
        df = dp.construct_data(df, l_days=l_days, n_years_h2h=n_years_h2h, segun_localia=segun_localia, dif_con_against=dif_con_against, export=export)
        df, df_etiquetas = dp.tag_string_data_to_integer(df, export=export)
        df, scaler, columns_used = dp.clean_data_2(df, n_years_to_select, comp_to_select, export=export)
        df = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, export=export)
        df = dp.treat_nan_values(df, fill_na=fill_na, export=export)
        
        if export:
            df_hiper_prep.to_excel(f'./data/{country}/p3_data_preparation/df_hiper_prep.xlsx', index=False)

    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./data/{country}/p3_data_preparation/df_selected_nan.xlsx', index_col=0)
        print(df.head(3), df.shape)

    #------------------------------------------- MODELING -------------------------------------------#
    if modeling:
        print(" Modeling ".center(120, "#"))
        # Hiperparametros
        val_size, test_size = 0.125, 0.125
        bal_type = None
        k = 10
        d_hiper_mod = {'val_size': [val_size], 'test_size': [test_size], 'bal_type': [bal_type], 'k': [k]}

        modelo = LogisticRegression()  # LogisticRegression(), RandomForestClassifier()
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
        model, d_hiper_model, cv_accuracy = mo.build_model(modelo, X_val=X_val, y_val=y_val, X_train=X_train, y_train=y_train, k=k, params=hiperparametros, export=export)
        df_pred, d_metrics = mo.assess_model(model, X_test, y_test, export=export)

        # Construyo dataframe con hiperparametros de Modeling() (incluyendo los de la estrategia de apuesta)
        d_hiper_mod.update(d_metrics)
        df_hiper_mod = pd.DataFrame(data=d_hiper_mod, index=[0])        

        if export:
            df_hiper_mod.to_excel(f'./data/{country}/p4_modeling/modeling/df_hiper_mod.xlsx', index=True)
            pickle.dump(model, open(f"./data/{country}/p4_modeling/modeling/modelo.pkl", "wb"))

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Definicion declea variables
    id_country = 167 # 55, 59, 77, 148
    d_params = {'data_unders': False, 'data_prep': True, 'modeling': False}

    main(id_country, d_params, export=True)