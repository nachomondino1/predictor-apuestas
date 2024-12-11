# Importo librerias
import pandas as pd
import numpy as np
import os
import datetime
from utils.set_up_logging import logger
## Data understanding
from p2_data_understanding.collect_initial_data import scraper_flashscore, scraper_sofifa
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
from sklearn.preprocessing import StandardScaler
import joblib
## Modeling
from p4_modeling import generate_test_design, build_model, asses_model, betting_strategy
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
from types import SimpleNamespace


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
            f'./data/{self.country}/p3_data_preparation/treat_nan',
        ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def format_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, verbose : int = 0, export: bool = True):
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

    def clean_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, 
                   df_teams_sofifa: pd.DataFrame, export: bool = True, verbose : int = 0):
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
        
        if verbose >= 1:
            logger.info(f"Partidos jugados antes de {start_date} eliminados. {n_rows_inic} --> {len(df_match)}. {len(df_match_player)}")

        # Elimino columnas de jugadores que son todo NaN (se ve que hay porque las creo y no les guardo nada eso debe ser porque obtengo nombres solo si tiene url)
        non_object_columns = df_match_player.select_dtypes(exclude=['object']).columns
        df_match_player.drop(columns=non_object_columns, inplace=True)
        
        if verbose >= 1:
            print("Shape df_match_player: ", df_match_player.shape)
            print("\nPreparacion de columnas string...")

        # Preparacion de texto
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
        if verbose >= 0 and n_players_eliminated > 0:
            logger.warning(f"Se eliminaron {n_players_eliminated} jugadores de los {len_inic} de Sofifa que habia.")
        ### Dataframe player sofifa (df)
        df_player_sofifa = clean_data.prepare_text_columns(df_player_sofifa, l_cols_to_process=['player_name', 'player_name_short'])  # Preaparo texto para integrar
        ### Dataframe teams sofifa (df_teams_sofifa)
        df_teams_sofifa = clean_data.prepare_text_columns(df_teams_sofifa, l_cols_to_process=['team_name'])

        # Correcion de valores
        if verbose >= 1:
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

    def integrate_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, 
                       df_teams_sofifa: pd.DataFrame, export: bool = True, verbose : int = 0):
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

        # (Temporalmente) Obtengo el listado de equipos unicos de Flashscore
        df_teams = create_df_teams(df_match)
        if export:
            df_teams.to_excel(f"./data/{self.country}/p3_data_preparation/integrate_data/df_teams.xlsx", index=True)

        # PLAYERS --> MATCH (Mapeo df_player_sofifa con df_player e integro a df_match)
        print("\nIntegrating player's data to df_match...")
        # Si ya hice el mapeo
        try:
            path_map = 'data/all/p3_data_preparation/integrate_data/df_map_players_fs_so.xlsx' # data/df_map_players_fs_so.xlsx
            df_map_players_fs_so = pd.read_excel(path_map, index_col=0)
            logger.info("No vuelvo a mapear sino que levanto df_map general ")

            # Reemplazo los df_player de Sofifa del pais por los completos
            df_player_sofifa = pd.read_excel('data/all/p3_data_preparation/clean_data/df_player_sofifa_cleaned.xlsx', index_col=0) #'data/df_player_sofifa.xlsx'
            df_player_fifa_sofifa = pd.read_excel('data/all/p3_data_preparation/clean_data/df_player_fifa_sofifa_cleaned.xlsx', index_col=0) # 'data/df_player_fifa_sofifa.xlsx'
            
            logger.critical("Integración usando el df_map completo!")

        # Si aun no hice el mapeo
        except FileNotFoundError:

            try:
                df_map_players_fs_so = pd.read_excel(f'data/{self.country}/p3_data_preparation/integrate_data/df_map_players_fs_so.xlsx')
                logger.info("No vuelvo a mapear sino que levanto df_map del pais ")

            except FileNotFoundError:
                # Matcheo jugadores de Sofifa y Flashscore
                logger.info("Mapeo jugadores de Sofifa y Flashscore")

                df_map_players_fs_so, df_player = pd.DataFrame(), pd.DataFrame()

                # Seleccionar ligas
                d_comps = select_data.determine_country_competitions(id_country)            
                print(f"Competiciones: {d_comps['comp_sin_cups']}")

                # Por Liga:
                for id_comp in d_comps['comp_sin_cups']:
                    print(f"Competicion: {id_comp}")

                    # filtrar df_match y df_match_player y df_player_sofifa
                    ## Flashscore
                    print(f"AA: {len(df_match)} {len(df_match_player)}")
                    df_match_league = df_match[df_match['id_competition'] == id_comp]
                    df_match_player_league = df_match_player[df_match_player.index.isin(df_match_league.index)]
                    df_player_league = create_df_player(df_match_player_league)
                    print(f"BB: {len(df_match_league)} {len(df_match_player_league)} {len(df_player_league)}")

                    ## Sofifa
                    print(f"FF: {len(df_player_sofifa)} {len(df_player_fifa_sofifa)}")
                    df_player_fifa_sofifa_league = df_player_fifa_sofifa[df_player_fifa_sofifa['id_competition'] == id_comp]
                    ids_players = df_player_fifa_sofifa_league['id_player'].unique()
                    df_player_sofifa_league = df_player_sofifa[df_player_sofifa.index.isin(ids_players)]
                    print(f"GG: {len(df_player_sofifa_league)} {len(df_player_fifa_sofifa_league)}")

                    # Mapeo jugadores...
                    df_map_players_fs_so_league = match_dataframes_by_str_column(df1=df_player_league, df2=df_player_sofifa_league, column_to_match1="player_name", column_to_match2="player_name", column_to_match2_aux='player_name_short', column_to_integrate='id_player', thr_coincidence_min=90)

                    # Concateno mapeos de ligas
                    df_map_players_fs_so = pd.concat([df_map_players_fs_so, df_map_players_fs_so_league], axis=0)
                    df_player = pd.concat([df_player, df_player_league], axis=0)
                    print(f"ZZ: {len(df_map_players_fs_so)}")

                    # df_map_players_fs_so.drop_duplicates()
                    if export:
                        df_player_league.to_excel(f"./data/{self.country}/p3_data_preparation/integrate_data/df_player_{id_comp}.xlsx", index=True)
                        df_map_players_fs_so_league.to_excel(f"./data/{self.country}/p3_data_preparation/integrate_data/df_map_players_fs_so_{id_comp}.xlsx")
                
                print(f"Final: {len(df_map_players_fs_so)}")
                 # Eliminar jugadores duplicados (x jugar en ambas competicioens)
                df_map_players_fs_so = df_map_players_fs_so.drop_duplicates(subset=['id_player_fs'], keep='first')
                print(f"Final sin dup: {len(df_map_players_fs_so)}")

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

    def construct_data(self, df: pd.DataFrame, l_days: list , n_years_h2h: int, segun_localia: bool, with_h2h: bool = False, with_historic: bool = True, 
                       dif_con_against: bool = True, export: bool = True, verbose : int = 0):
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
            thr_ajustado = construct_data.adjust_thr_to_match_distributions(df, initial_thr=0.3, tolerance=0.04)
            df = construct_data.determine_expected_result(df, thr_expected=thr_ajustado)
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

    def tag_string_data_to_integer(self, df: pd.DataFrame, verbose : int = 0, export: bool = True):
        """
        Conversion de columnas tipo "object" a "integer"
        """
        # Elimino columna 'season'
        df = df.drop(['season'], axis=1)  # Arroja error TypeError porque tiene tanto str como int en los valores originales y el label solo puede recibir un tipo (str o int). Season tiene valores como "2021" y "2020_2021", los primeros los entiende como int y los segundos como str.
        
        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df, verbose=verbose)

        if export:
            df_etiquetas.to_excel(f'./data/{self.country}/p3_data_preparation/df_etiquetas.xlsx', index=False)
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_constructed_etiquetado.xlsx', index=True)
        return df, df_etiquetas
    
    def clean_data_2(self, df: pd.DataFrame, n_years_to_select: int = None, competencies_to_select: list = None, fill_na: str = None, verbose : int = 0, 
                     export: bool = True):
        """
        Eliminacion de filas y columnas con mucho NaN y escalado de datos

        # Parameters:
            df: Dataframe a tratar NaN y escalar.

        # Returns:
            df: Dataframe pasado como parametro sin filas y columnas con mucho NaN y con datos escalados.
        """
        start = time.time()
        logger.info("\nSelecting data...")

        # Reemplazo infinitos
        df = clean_data.replace_infinite(df)

        # Ordeno valores por fecha y separo X e y
        df = df.sort_values(by='date', ascending=False)
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]  # Separo X e y
        
        # (1) Eliminacion de filas 
        ## Para evitar partidos muy viejos
        if verbose >= 1:
            print("Eliminacion de filas...")
            n_reg_inic = len(X)

        if n_years_to_select is not None:
            fecha_limite = X.iloc[0]['date'] - datetime.timedelta(days=n_years_to_select*365)
            X = X[X['date'] >= fecha_limite] 

            if verbose >= 1:
                print(f"Eliminacion por fecha. Cantidad de filas: {n_reg_inic} --> {len(X)}")

        ## Para evitar ciertas competencias
        if competencies_to_select is not None:
            n_reg_inic_2 = len(X)
            X = X[X['id_competition'].isin(competencies_to_select)]

            if verbose >= 1:
                print(f"Eliminacion por competencias. Cantidad de filas: {n_reg_inic_2} --> {len(X)}")

        # (2) Eliminacion de columnas   
        ## usadas solo para construir y constantes
        cols_for_construct = ['date', 'venue'] # 'id_team_home', 'id_team_away' # Elimino variables que no usare en el modelo fecha (la idea es usar todas las posibles) # ['date', 'venue', 'id_competition', 'id_team_home', 'id_team_away']  
        cols_constants = list(X.columns[X.nunique() == 1])  # Elimino columnas constantes
        X.drop(columns=cols_for_construct+cols_constants, inplace=True)
        if verbose >= 1:
            print("Eliminación de columnas...")
            print(f"Columnas constantes eliminadas: {cols_constants}")

        # (3) Tratamiento de NaN values
        shape_inicial = X.shape
        X = self.treat_nan_values(X=X, fill_na=fill_na)
        if verbose >= 1:
            print(f"Tras fill_na={fill_na}. Shape X_sin_col_mucho_nan: {shape_inicial} --> {X.shape}")

        # (4) Escalado de datos
        if verbose >= 1:
            print("\nEscalado de datos...")

        scaler = StandardScaler()
        # X_sin_col_mucho_nan = X.select_dtypes(exclude=['datetime64[ns]'])  # Excluy escalado de columnas datetime -->  The DType <class 'numpy.dtypes.DateTime64DType'> could not be promoted by <class 'numpy.dtypes.Float64DType'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is object.
        scaler.fit(X) # Paso 1: Ajusta el StandardScaler a tus datos
        X_scaled = scaler.transform(X) # Paso 2: Transforma tus datos utilizando el StandardScaler ajustado
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Concateno X e y
        y_sin_nan = y[y.index.isin(X.index)] # Dado que elimine filas de X
        df = pd.concat([X_scaled_df, y_sin_nan], axis=1)

        end = time.time()
        logger.info(f"Clean data 2 en {(end - start)/60:.1f} minutos")

        if export: 
            joblib.dump((scaler, X.columns), f"./data/{self.country}/p3_data_preparation/scaler_model.pkl")       
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_constructed_clean.xlsx', index=True)

        return df, scaler, X.columns
    
    def treat_nan_values(self, X: pd.DataFrame , fill_na: str = None, porc_min_no_nan: float = 0.7, percentil_nan: int = 75, export: bool = True, verbose: int = 0):
        """
        Tratamiento de nan values

        # Para ENG y SPA usé porc_min_no_nan = 0.5. Pero para FRA tengo que usar 0.8 porque sino X_test queda vacio.
        Tan alto puede tirar error porque elimina todas las columnas.

        # Parameters
            df: Dataframe a tratar nan values. (DataFrame)
            fill_na: Tipo de rellenado de NaN values.
            porc_min_no_nan: Porcentaje minimo de no NaN value que debe tener una columna para evitar ser eliminada. (float)
            percentil_nan: Percentil para definir que columnas son consideradas con mucho nan y cuales con poco nan. Solo cuando haces fillna.
            export: 
            _print:

        # Returns
            Dataframe sin NaN values
        """ 
        start = time.time()
        if verbose >= 1:
            print("\nTreating NaN values to avoid input=NaN in Modeling...")
            logger.info(f"1. Datos de entrada a treat_nan: {self.n_rows_to_test(X)}")

        # (1) Eliminacion de filas con mucho NaN (filas sin estadisticas ni formaciones) --> Elimina "ultimos partidos" en SPA probablemente por falta de estadistica "total_passes". No sirve si fill_na=None pero si cuando fill_na=ml.
        n_reg_inic_3 = len(X)
        X = clean_data.delete_rows_nan(X, porc_min_no_nan)

        if verbose >= 1:
            print(f"Eliminaccion por mucho NaN. Cantidad de filas: {n_reg_inic_3} --> {len(X)}")
            logger.warning(f"Cantidad de filas: {n_reg_inic_3} --> {len(X)}")
            logger.critical(f"2. Luego de eliminar FILAS con mucho NaN: {self.n_rows_to_test(X)}")

        # (2) Eliminacion de columnas con mucho NaN --> Elimino columnas con alto porcentaje de NaN values (de manera que tras el dropna quedarian menos de n_reg_min)
        n_reg_min = int(porc_min_no_nan*len(X)) # no uso n_features_min porque hay tengo un millon de columnas extra que eliminare en select...
        X_sin_col_mucho_nan = X.copy()
        X = clean_data.drop_columns_until_drop_na_min_rows(X, n_reg_min=n_reg_min) # elimina las columnas hasta que pueda hacer dropna()

        if verbose >= 0 and len(X.columns) != len(X_sin_col_mucho_nan.columns):
            l_col_eliminated = list(X_sin_col_mucho_nan.columns.difference(X.columns))
            logger.warning(f"Se han tenido que eliminar {len(X_sin_col_mucho_nan.columns) - len(X.columns) } columnas de {len(X_sin_col_mucho_nan.columns)} porque no se alcanzaba el minimo de {n_reg_min} registros para entrenar el modelo. Columnas eliminadas: {l_col_eliminated}")
        
        if verbose >= 1:
            print(f"Tras eliminar columnas con mas de {(1-porc_min_no_nan)*100:.0f}% de NaN values. Shape X_sin_col_mucho_nan: {X_sin_col_mucho_nan.shape} --> {X.shape} ")
            logger.info(f"3. Luego de eliminar COLUMNAS con mucho NaN: {self.n_rows_to_test(X_sin_col_mucho_nan)}")

        # (3) Eliminacion de todo NaN ya sea drop o fill_na
        # Determino las columns con mucho NaN (mas de nan_threshold%)
        if fill_na is not None:
            l_columns_poco_nan, l_columns_mucho_nan = clean_data.determine_columns_to_fill(X, percentil_nan=percentil_nan)

            # Elimino registros con al menos un NaN 
            X = X.dropna(subset=l_columns_poco_nan)
            if verbose >= 1:
                print(f"De las {len(df)} filas, se han eliminado {len(df)-len(X)} por tener al menos un Nan value. Quedan {len(X)} filas. Shape final: {X.shape}") 
                logger.info(f"4. Luego de dropna de columnas con 'poco' nan: {self.n_rows_to_test(X)}")

            # Determino que filas relleno y cuales no (antes de fill porque despues de rellenar no puedo diferenciar que filas rellene y cuales no)
            df_rellenado = pd.DataFrame(index=X.index)
            df_rellenado['rellenado'] = X[l_columns_mucho_nan].isnull().any(axis=1)
            df_rellenado.to_excel(f'./data/{self.country}/p3_data_preparation/df_rellenado.xlsx', index=True)

            # Relleno nan de las columnas con mucho NaN
            X = clean_data.fill_nan_values(X, l_columns_mucho_nan, fill_type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las accuracyes casi siempre seran mayores que dropna() en train y test, lo que cuenta es la accuracy en next_matches o en un dataset que no haya sido filleado...
            if verbose >= 1:
                print(f"Columnas consideradas con mucho nan (a las cuales rellenar): {l_columns_mucho_nan}")
                print(f"\tSe realizó el rellenado de NaN values. Shape X luego de rellenado: {X.shape}")

        else:
            # Elimino registros con al menos un NaN 
            X = X.dropna(subset=X.columns)

        if verbose >= 0:
            logger.info(f"Luego de eliminar todo NaN: {self.n_rows_to_test(X)}")

        end = time.time()
        print(f"Tratamiento de NaN values en {(end - start)/60:.1f} minutos")

        if export:
            X.to_excel(f'./data/{self.country}/p3_data_preparation/df_selected_nan.xlsx', index=True)
        
        return X
    
    def n_rows_to_test(self, X, verbose: int = 0):
        """
        Imprime por pantalla cuantos registros quedarian en df_test segun los requisitos exigidos.
        """
        # Para ver donde se eliminan los ultimos partidos.
        df_match = pd.read_excel(f'data/{self.country}/p6_deployment/missing/old_updated/df_match.xlsx', index_col=0)
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        df_match = df_match.sort_values(by='date', ascending=False)

        # Requisito 1: id competition.   # En df_match obtengo id_competition por match y determino posibles id_matches
        from p3_data_preparation.select_data import select_league_matches
        df1 = select_league_matches(df_match, verbose=-1)
        index_comp = df1.index

        # Requisito 2: Last matches (6 meses?)
        fecha_last_match = df_match.iloc[0]['date']
        fecha_limite = fecha_last_match - datetime.timedelta(days=0.25*365)
        df2 = df_match[df_match['date'] >= fecha_limite] 
        index_last_matches = df2.index

        # Cantidad de registros que pasan 1 y 2 en df_match
        df_match_filt = df_match[df_match.index.isin(index_comp) & df_match.index.isin(index_last_matches)] # "ultimos partidos de id_competition en df_match"
        n_part_expected = len(df_match_filt)

        filters = X.index.isin(index_comp) & X.index.isin(index_last_matches)
        df_filt_2 = X[filters]
        n_part_final = len(df_filt_2)

        if verbose >= 0:
            logger.info(f"Nº partidos expected: {n_part_expected}. Nº partidos que pasaron: {n_part_final}")

        return n_part_final

    def select_data(self, df: pd.DataFrame, thr_corr: float = None, thr_fs: float = None, verbose: int = 0, export: bool = True):
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        start = time.time()
        logger.info("Selecting data...")

        # Elimino variables altamente correlacionadas
        if thr_corr is not None:
            l_columnas_a_eliminar, df_corr_tri_X = select_data.delete_correlated_columns(df, self.var_resp, thr_corr)
            df = df.drop(l_columnas_a_eliminar, axis=1)

            if verbose >= 1:
                print('\n Eliminando columnas correlacionadas...')
                print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            n_cols = len(df.columns)-1  # -1 por variable respuesta
            l_important_features, df_normalized = select_data.select_best_features(df, self.var_resp, thr_fs, graf=export)
            l_col_eliminated = list(df.columns.difference(l_important_features))
            df = df.loc[:, l_important_features + [self.var_resp]]

            if verbose >= 1:
                print('\n Feature Selection...')
                print(f"\tSe eliminaron {n_cols-len(l_important_features)} de {n_cols} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%. Columnas eliminadas: {l_col_eliminated}")

        if verbose >= 1:
            print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")
        
        end = time.time()
        logger.info(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_corr_tri_X.to_excel(f'./data/{self.country}/p3_data_preparation/select_data/df_correlation.xlsx', index=True)
            df_normalized.to_excel(f'./data/{self.country}/p3_data_preparation/select_data/df_fs.xlsx', index=True)
            df.to_excel(f'./data/{self.country}/p3_data_preparation/df_selected.xlsx', index=True)

        return df
    

class Modeling:

    def __init__(self, country: str, var_resp: str = 'result', var_pred: str = 'predicted_result'):
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
        
    def select_test_set(self, df, test_size, retrain, n_months: int = 3, n_max_reg: int = 100, verbose: int = 1):
        """
        Determina qué registros pueden ser utilizados en el test
        Requisitos para el test
            -1: Que id_competition sea publica (lo mismo que hago en assess) --> Nuevo
            -2: Que sean partidos del ultimo año? --> Nuevo
                # Podria levantar df_match y ver para tal id_match su valor en id_competition y su fecha.
            -3: Que no este rellenado

        # Parameters
            df: Dataframe.
            test_size: Porcentaje maximo del total de datos que iran al test.
            n_years_to_select: Numero de años para seleccionar los ultimos partidos los cuales iran al df_test.
            n_max_reg: Numero maximo de registros para X_test. (int)
            verbose: 

        # Return
            X_test: Dataframe de testeo sin variable respuesta.
            y_test: Dataframe de testeo solo la variable respuesta.
        """
        # Levanto df_match del pais
        if retrain:
            df_match = pd.read_excel(f'data/{self.country}/p6_deployment/missing/old_updated/df_match.xlsx', index_col=0)
            df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
            logger.warning(f"Se levanto el df_match con los missing pues retrain=True. Shape: {df_match.shape}")
        else:
            df_match = pd.read_excel(f"data/{self.country}/p3_data_preparation/clean_data/df_match_cleaned.xlsx", index_col=0)  
        
        # Ordeno por fecha descendiente
        df_match = df_match.sort_values(by='date', ascending=False)
        
        if verbose >= 2:
            logger.info(df_match.columns) # No quiero "unnamed"
            logger.info(df_match['date'].head(10))

        # Requisito 1: id competition.   # En df_match obtengo id_competition por match y determino posibles id_matches
        from p3_data_preparation.select_data import select_league_matches
        df1 = select_league_matches(df_match)
        index_comp = df1.index

        if verbose >= 2:
            df_filt_1 = df[df.index.isin(index_comp)]
            logger.info(f"Registros que pasan el requisito 1 (solo competencia publica): {len(df_filt_1)}")
        
        # Requisito 2: Last matches (6 meses?)
        fecha_last_match = df_match.iloc[0]['date']
        n_days = n_months * 30
        fecha_limite = fecha_last_match - datetime.timedelta(days=n_days)
        df2 = df_match[df_match['date'] >= fecha_limite] 
        index_last_matches = df2.index

        if verbose >= 1:

            df_match_filt = df_match[df_match.index.isin(index_comp) & df_match.index.isin(index_last_matches)] # "ultimos partidos de id_competition en df_match"
            n_part_expected = len(df_match_filt)

            df_filt_2 = df[df.index.isin(index_comp) & df.index.isin(index_last_matches)]
            n_part_final = len(df_filt_2)

            logger.info(f"Nº partidos expected: {n_part_expected}. Nº partidos que pasaron: {n_part_final}")
            # logger.info(f"Registros que pasan requisitos 1 y 2: {len(df_filt_2)}")

        if verbose >= 2:
            logger.info(f"Partido mas reciente: {fecha_last_match} ({max(df_match['date'])}). Fecha limite: {fecha_limite}")
            logger.warning(f"Hay {len(index_last_matches)} partidos en los ultimos {n_years_to_select*365/30:.1f} meses.")

            df_aux = df[df.index.isin(index_last_matches)]
            logger.info(f"Registros que pasan requisito 2 (todas las competiciones): {len(df_aux)}")

            # Si se han eliminado "ultimos partidos" en treat_nan_values() o clean_data_2()
            if len(df_aux) != len(index_last_matches):
                logger.warning(f"(1 de 3) PROBLEMA DE NAN EN ULTIMOS PARTIDOS. De los {len(index_last_matches)}, solo hay {len(df_aux)} en el df que recibe select_test_data(). Deberian ser iguales --> {len(index_last_matches)} = {len(df_aux)}")
                logger.warning(f"(2 de 3) Por que no son iguales? Se estan eliminando 'ultimos partidos' en 1) clean_data_2 (eliminacion de filas por tener mucho NaN) o 2) treat nan values (dropna de columnas con 'poco' nan).")
                logger.warning(f"(3 de 3) Que podes hacer? No podemos hacer mucho sino investigar por qué los ultimos partidos tienen tanto NaN y evitar que tenga NaN. Seguramente el problema es en la extraccion de missing debido a algun cambio de Flashscore")

        # Selecciono registros que cumplen los 3 requisitos
        filters = df.index.isin(index_comp) & df.index.isin(index_last_matches) # df.index.isin(index_comp) & df.index.isin(index_last_matches)& df.index.isin(index_no_rellenado) 
        df_filt = df[filters]

        # Determino si hay suficientes registros no rellenados para poner en el dataframe de testeo
        n_reg_test = min(int(len(df) * test_size), n_max_reg) # Defino numero de registros necesarios
        n_reg_test_max = len(df_filt)

        if n_reg_test > n_reg_test_max: # Si no hay suficientes filas no rellenadas disponibles
            # Ajusta n para tomar todas las filas no rellenadas disponibles
            n_reg_test_old = n_reg_test
            n_reg_test = n_reg_test_max
            
            if verbose >= 0:
                logger.warning(f"ACHICO DF_TEST. Reduzco la cantidad de registros en test debido a que no hay los suficientes que satisfagan los requisitos. Necesito {n_reg_test_old} pero hay solo {n_reg_test_max} registros posibles.")

        # Construyo el dataset de testeo a partir de registros que no han sido rellenados
        df_test = df_filt.sample(n_reg_test, random_state=42)
        X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]

        return X_test, y_test
        
    def generate_test_design(self, df: pd.DataFrame, bal_type, val_size: float = 0.15, test_size: float = 0.15, verbose: int = 0, retrain: bool = False, export: bool = True):
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
        if verbose >= 0:
            print("\nSeparating data in train, val and test...")

        # Selecciono test set
        X_test, y_test = self.select_test_set(df, test_size, retrain=retrain)

        # Separo validation y train (dejo de tener en cuenta si lo rellene o no)
        df_train_val = df[~df.index.isin(X_test.index)]
        X_train_val, y_train_val = df_train_val.drop(self.var_resp, axis=1), df_train_val[self.var_resp]

        # Calcula el tamaño relativo del conjunto de validación
        test_size_ratio = len(X_test) / len(df)  # Calcula el tamaño relativo del conjunto de prueba
        val_size_ratio = val_size / (1 - test_size_ratio) 

        # Separo en train y validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_ratio, random_state=randint(1, 1000), shuffle=True)

        # Balanceo el dataset de entrenamiento (No se debe balancear el de validacion)
        if bal_type is not None:
            X_train, y_train = generate_test_design.balance_dataset(X_train, y_train, bal_type=bal_type)

        if verbose >= 0:
            print(f'Train: {X_train.shape} {y_train.shape}', f'\nVal: {X_val.shape} {y_val.shape}', f'\nTest: {X_test.shape} {y_test.shape}')

        if export:
            X_train.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/X_train.xlsx', index=True)
            X_val.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/X_val.xlsx', index=True)
            X_test.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/X_test.xlsx', index=True)
            y_train.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/y_train.xlsx', index=True)
            y_val.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/y_val.xlsx', index=True)
            y_test.to_excel(f'./data/{self.country}/p4_modeling/generate_test_design/y_test.xlsx', index=True)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, default_model, X_val: pd.DataFrame, y_val: pd.DataFrame, X_train: pd.DataFrame, y_train, k: int, params: dict = None, 
                    bayes: bool = True, compare_tuning: bool = False, export: bool = True):
        """
        Selecciona el mejor modelo a partir de la accuracy.
        
        # Parameters
            model: Modelo de Machine Learning. (sklearn.ensemble)
            X_val: Dataframe de validacion con variables predictoras. (DataFrame)
            y_val: Dataframe de validacion solo con variable respuesta. (DataFrame)
            X_train: Dataframe de entrenamiento con variables predictoras.  (DataFrame)
            y_train: Dataframe de entrenamiento solo con variable respuesta. (DataFrame)
            k: Numero de folds. (int)
        
        # Return
            model_best_params: Modelo entrenado con hiperparaemtros optimos (sklearn.ensemble?)
            params: Combinacion de hiperparametros del modelo (dict)
            train_accuracy: Precision de entrenamiento (float)
        """
        # warnings.filterwarnings("ignore")
        print("\nTraining model...")
        self.classes = np.unique(y_train)

        if default_model == "neural_network": # A diferencia de los otros modelos, la tengo que crear                
            logger.info("Entrenando red neuronal")

            # Creo instancia de clase NeuralNetwork()
            red = build_model.TrainNeuralNetwork()

            # Seleccion mejor arquitectura con la validacion y entreno el modelo
            model_best_params, params, train_accuracy, results = red.select_best_arquitecture(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        else:
            # Train model searching for best hiper
            if params is None:
                model_best_params, params, best_metric, results = build_model.select_best_hiperparameters(default_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k, 
                                                                                                          bayes=bayes, all_tuning=compare_tuning, verbose=1)
                # model_best_params, params, best_metric, results = build_model.compare_scoring_methods(default_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k, bayes=True, all_tuning=compare_tuning, verbose=1)
                train_accuracy = best_metric # no es train_acc... es el scoring que uso, en este caso, f1_macro..
      
            # Train model with prefix params
            else:
                model_best_params = default_model.set_params(**params)

                # Fit model
                model_best_params.fit(X_train, y_train)

                # Eval en val...

                # Evaluo el modelo con Cross Validation
                train_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
                print(f"\nAccuracy promedio de validación cruzada: {train_accuracy:.1f}%")

        if export:
            pickle.dump(model_best_params, open(f"./data/{self.country}/p4_modeling/modelo.pkl", "wb"))
            # results.to_excel(f"./data/{self.country}/{ite_date}/p4_modeling/models/hiperparametros.xlsx")    # Exportar metricas por cada combinacion de hiperparametros (En vez de retornar best_metric.)

        return model_best_params, params, train_accuracy, results

    def assess_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, retrain: bool = False, export: bool = False, verbose: int = 0):
        """
        Evalúa un modelo de machine learning utilizando datos de prueba y calcula métricas de desempeño.

        # Parameters:
            model: Modelo de Machine Learning entrenado. (sklearn.ensemble)
            X_test: Dataframe de prueba con variables predictoras. (DataFrame)
            y_test: Dataframe de prueba solo con variable respuesta. (DataFrame)
            retrain: Boolean para definir si es un reentrenamiento de modelos con missing o no. (bool)
            export: Booleano para indicar si se debe exportar el DataFrame seleccionado. True para exportar, de lo contrario, False.  (bool)

        # Returns:
            Precisión del modelo y ROI en el conjunto de prueba. (int) y (float)
        """
        print("\nEvaluating trained model with test sets...")
        
        # Predigo sobre X_test
        y_pred_prob, y_pred = self.predict(model, X_test, verbose=verbose)

        # Crear el DataFrame con las probabilidades
        df_pred_proba = pd.DataFrame({
                self.var_resp: y_test,
                self.var_pred: y_pred,
                f'prob_class_{self.classes[1]}': y_pred_prob[:, 1],  # Probabilidad de la clase 1
                f'prob_class_{self.classes[0]}': y_pred_prob[:, 0],  # Probabilidad de la clase 0
                f'prob_class_{self.classes[2]}': y_pred_prob[:, 2]   # Probabilidad de la clase 2 (si hay 3 clases)
            }, index=X_test.index)
        

        # Calculo metricas
        df_predicciones, d_metrics = self.calculate_metrics(df_pred_proba, retrain=retrain, export=export, verbose=verbose)

        if export:
            df_predicciones.to_excel(f'./data/{self.country}/p4_modeling/modeling/df_predicciones.xlsx')

        return df_predicciones, d_metrics
    
    def predict(self, model, X_test: pd.DataFrame, verbose: int = 0):
        """
        Predigo sobre X_test
        """
        # Predecir las etiquetas para los datos de prueba
        try:
            y_pred_prob = model.predict_proba(X_test) # Te da las probabilidad de cada clase. Funciona para todos los modelos? # AttributeError: predict_proba is not available when probability=False
            
            if verbose >= 1:
                class_distribution = y_pred_prob.mean(axis=0)
                print("Distribución promedio de probabilidades por clase:", class_distribution)
                
        except AttributeError: # AttributeError: 'Sequential' object has no attribute 'predict_proba'
            y_pred_prob = model.predict(X_test)
            
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad  # y_pred = model.predict(X_test)  # es un numpy array      
        return y_pred_prob, y_pred
    
    def calculate_metrics(self, df_pred_proba: pd.DataFrame, retrain: bool = False, export: bool = False, verbose: int = 0):
        """
        Calculo metricas como precision y ROI de las predicciones del modelo entrenado.
        """
        # Defino variables
        bs = betting_strategy.BettingStrategy()
        self.var_pred_bm = 'bookmaker_result'  
        y_test = df_pred_proba[self.var_resp].values  # Etiquetas reales
        y_pred = df_pred_proba[self.var_pred].values  # Predicciones del modelo

        # Calculo metricas
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100
        d_metrics = {'test_accuracy': test_accuracy, 'recall': recall, 'f1_score': f1}
        if verbose >= 1:
            # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
            df_conf_mat = asses_model.confusion_matrix(y_test, y_pred)
            if export:
                df_conf_mat.to_excel(f'./data/{self.country}/p4_modeling/modeling/df_conf_matrix.xlsx')


        ## df_match --> # Podria levantar el df_match para agregar equipos y saber que partido es cada cual en el df_predicciones... tal como hago en produccion.
        path_match = f'data/{self.country}/p6_deployment/missing/old_updated/df_match.xlsx' if retrain else f'data/{self.country}/p2_data_understanding/df_match.xlsx'
        df_match = pd.read_excel(path_match, index_col=0) # --> missing no lo necesita y el otro si?
        df_match = df_match.loc[:, ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition', 'goals_home', 'goals_away',  'expected_goals_(xg)_home', 'expected_goals_(xg)_away']] 
        # Determino expected ROI --> Lo uso en calculate_roi...
        df_match = df_match[df_match.index.isin(df_pred_proba.index)]
        # df_match = df_match.reindex(df_pred_proba.index

        # Df_match_odds
        path_match_odds = f'data/{self.country}/p6_deployment/missing/old_updated/df_match_odds.xlsx' if retrain else f'data/{self.country}/p2_data_understanding/df_match_odds.xlsx'
        df_match_odds = pd.read_excel(path_match_odds, index_col=0) # --> missing no lo necesita y el otro si?
        # Filtro df_match_odds dejando solo los partidos de X_test
        df_match_odds = df_match_odds[df_match_odds.index.isin(df_pred_proba.index)]  # Selecciono los partidos que estan en df_test
        df_match_odds = df_match_odds.reindex(df_pred_proba.index)  # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test
        
        if verbose >= 2:
            logger.info(f"Path odds: {path_match_odds}")
            logger.info(df_match_odds)

        # Agrego predicciones de bookmaker
        df_match_odds = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta
        df_match_odds = asses_model.determine_result_by_bookmaker(df_match_odds, self.var_pred_bm)  # Determino resultado predicho segun cuota minima (e.g. "Home")
        y_pred_bm = df_match_odds[self.var_pred_bm].values

        # Calculo precision de casa de apuesta
        test_precision_bookmaker = accuracy_score(y_test, y_pred_bm) * 100  # Calcula bien tras el reindex()
        dif_prec = test_accuracy - test_precision_bookmaker
        d_metrics.update({'test_accuracy_bm': test_precision_bookmaker, 'dif_prec_bm': dif_prec})
        

        # Calculo distribucion en df_predicciones? .... lo guardo en df_metrics pues luego va a df_test.
        d_distrib = asses_model.determine_distribution(df_pred_proba)
        d_metrics.update(d_distrib)


        # Concateno dfs --> Concateno antes de calcular ROI porque alli uso cuotas y expected result de df_match_odds y de df_match respectivamente
        df_predicciones = pd.concat([df_match, df_match_odds, df_pred_proba], axis=1)


        # Calculo ROI
        # thr_ajustado = construct_data.adjust_thr_to_match_distributions(df, initial_thr=0.3, tolerance=0.05)
        df_predicciones = construct_data.determine_expected_result(df_predicciones) # Intento hacerlo antes con df_match pero rompia.
        df_predicciones, d_roi = bs.calculate_roi_by_betting_strategy(df_predicciones, strategy='train', save_strategy=False)
        d_metrics.update(d_roi)

       # Convierto ids de equipos a nombres --> Hacerlo afuera de def assess_model...
        df_predicciones = self.map_teams(df_predicciones)

        if verbose >=1:
            print(d_metrics)

        return df_predicciones, d_metrics

    def map_teams(self, df):
        """
        Convierto id_team_home e id_team_away de ids a nombre de equipos.
        """
        # Levanto df_teams
        df_teams = pd.read_excel(f'data/{self.country}/p3_data_preparation/integrate_data/df_teams.xlsx', index_col=0)

        # Revierto etiquetas para tener nombres de equipos en vez de ids
        d_mapeo = dict(zip(df_teams.index, df_teams['team_name']))        
        df['id_team_home'] = df['id_team_home'].replace(d_mapeo)
        df['id_team_away'] = df['id_team_away'].replace(d_mapeo)
        return df

    def train_and_assess_models(self, l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, ruta_base_mod_seg, cont_iter, retrain: bool = False, export=True):
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
                # Entreno modelo
                model, params, cv_accuracy, results = self.build_model(modelo, X_val, y_val, X_train, y_train, k, export=False)

                # Evaluo modelo en test
                df_predicciones, d_metrics = self.assess_model(model, X_test, y_test, retrain=retrain)

                # Hiperparametros del modelo y Metricas en testeo y train
                new_row = {'n_iteration': cont_iter, 'model_name': model_name, 'cv_accurracy': cv_accuracy, 'model_hiper': params}
                new_row.update(d_metrics)
                df_metrics_new = pd.DataFrame([new_row])  # 1. Convertir el diccionario d_metrics en un DataFrame de una fila
                df_metrics = pd.concat([df_metrics, df_metrics_new], ignore_index=True)  # 2. Concatenar este nuevo DataFrame con df_metrics existente

                # Exporto datos del modelo
                pickle.dump(model, open(f"{ruta_base_mod_seg}/{cont_iter}_{model_name}.pkl", "wb"))
                results.to_excel(f'{ruta_base_mod_seg}/{cont_iter}__{model_name}_params.xlsx')
                df_predicciones.to_excel(f'{ruta_base_mod_seg}/{cont_iter}__{model_name}_predicciones.xlsx', index=True)

            except KeyboardInterrupt as e:
                logger.warning(f"Se evitó entrenar este modelo mediante {e}")
                
        return df_metrics  # return model, results, df_predicciones, df_metrics

def crear_variables(diccionario):
    return SimpleNamespace(**diccionario)

##################################################### MAIN #####################################################
def main(id_country, d_run, d_params, modelo, export: bool = True):
    """
    Extraction, processing and analysis of matches to predict match results.
    """
    # Definicion de variables
    d_par = crear_variables(d_params)
    d_run = crear_variables(d_run)     # data_unders, data_prep, modeling = d_run['data_unders'], d_run['data_prep'], d_run['modeling']

    if id_country > 0:
        df_countries = pd.read_excel('./data/df_countries.xlsx')
        country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0].lower()
    else:
        country = 'all'

    # Creo instancias de clases
    du = DataUnderstanding(id_country, country) # Creo objeto de clase DataPreparation
    dp = DataPreparation(country) # Creo objeto de clase DataPreparation
    mo = Modeling(country=country)  # Creo objeto de clase Modeling

    #------------------------------------------- DATA UNDERSTANDING -------------------------------------------#
    if d_run.data_unders:
        print(" Data understanding ".center(120, "#"))
        # Extriago datos o los levanto
        df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = du.collect_initial_data(export=export)

        # Describo datos
        du.describe_data(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa)

    elif d_run.data_prep:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match_odds.xlsx', index_col=0)
        df_player_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_player_sofifa.xlsx', index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_player_fifa_sofifa.xlsx') #  index_col=0 --> si lo uso falla la integracion porque pone 'id_player' como index
        df_teams_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_teams_sofifa.xlsx', index_col=0)
        
        du.describe_data(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa)

    #------------------------------------------- DATA PREPARATION -------------------------------------------#
    if d_run.data_prep:
        print(" Data preparation ".center(120, "#"))
        # Hiperparametros # PODRIA PONERLOS EN UN DICT Y HACER EL DATAFRAME MAS AUTOMATICO
 
        # df = pd.read_excel(f'./data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)
        # print(df.head(2))

        # Preparo el dataset para el analisis
        df_match, df_match_player, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_player_fifa_sofifa, export=False)
        df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export)
        df = dp.integrate_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export) 
       
        if not d_run.until_integrate:
            df = dp.construct_data(df, l_days=d_par.l_days, n_years_h2h=d_par.n_years_h2h, segun_localia=d_par.segun_localia, dif_con_against=d_par.dif_con_against, export=export)
            df, df_etiquetas = dp.tag_string_data_to_integer(df, export=export)
            df, scaler, columns_used = dp.clean_data_2(df, d_par.n_years_to_select, d_par.comp_to_select, export=export)
            df = dp.select_data(df, thr_corr=d_par.thr_corr, thr_fs=d_par.thr_fs, export=export)
            df = dp.treat_nan_values(df, fill_na=d_par.fill_na, export=export)
        
        # if export:
        #     df_hiper_prep.to_excel(f'./data/{country}/p3_data_preparation/df_hiper_prep.xlsx', index=False)

    elif not d_run.data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./data/{country}/p3_data_preparation/df_selected_nan.xlsx', index_col=0)
        print(df.head(3), df.shape)

    #------------------------------------------- MODELING -------------------------------------------#
    if d_run.modeling:
        print(" Modeling ".center(120, "#"))
        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df, d_par.bal_type, d_par.val_size, d_par.test_size, export=export)

        # Analizo datos con un modelo
        model, d_hiper_model, cv_accuracy = mo.build_model(modelo, X_val=X_val, y_val=y_val, X_train=X_train, y_train=y_train, k=d_par.k, export=export)
        df_pred, d_metrics = mo.assess_model(model, X_test, y_test, export=export)

        # Construyo dataframe con hiperparametros de Modeling() (incluyendo los de la estrategia de apuesta)
        # d_hiper_mod.update(d_metrics)
        # df_hiper_mod = pd.DataFrame(data=d_hiper_mod, index=[0])        

        if export:
            # df_hiper_mod.to_excel(f'./data/{country}/p4_modeling/modeling/df_hiper_mod.xlsx', index=True)
            pickle.dump(model, open(f"./data/{country}/p4_modeling/modeling/modelo.pkl", "wb"))

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Definicion declea variables
    id_country = 59 # 55, 59, 77, 148, 167 -1
    d_run = {'data_unders': False, 'data_prep': True, 'modeling': False, 'until_integrate': True}

    # Hiperparametros
    d_comps = select_data.determine_country_competitions(id_country)
    modelo = LogisticRegression()  # LogisticRegression(), RandomForestClassifier()
    d_params = {
        'l_days': [30, 180], 
        'n_years_h2h': 3,
        'segun_localia': False,
        'dif_con_against': False,
        'thr_corr': 0.85,
        'thr_fs': 0.5,
        'n_years_to_select': 3, 
        'comp_to_select': d_comps['all_comp'],
        'fill_na': None,
        'val_size':  0.125,
        'test_size': 0.125,
        'bal_type': None,
        'k': 10
    }

    # df_hiper_prep = pd.DataFrame(data=d_params) # df_hiper_prep = pd.DataFrame(data={'n_dias_ult_part': [l_days], 'n_anios_hist': [n_years_h2h], 'segun_localia': [segun_localia], 'dif_con_against': [dif_con_against], 'thr_corr': [thr_corr], 'thr_fs': [thr_fs], 'fill_na': [fill_na], 'n_years_to_select': [n_years_to_select], 'comp_to_select': [comp_to_select]}, index=[0])
    # d_hiper_mod = {'val_size': [val_size], 'test_size': [test_size], 'bal_type': [bal_type], 'k': [k]}

    main(id_country, d_run, d_params, modelo, export=True)