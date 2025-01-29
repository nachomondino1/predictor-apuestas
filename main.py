# Importo librerias
import pandas as pd
import numpy as np
import os
import datetime
import time
from utils.set_up_logging import logger
from utils import directories
## Data understanding
from p2_data_understanding.collect_initial_data import scraper_flashscore, scraper_sofifa
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data, integrate_sofifa_to_flashscore
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
        directories.make_directories(l_directorios=l_directorios)

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
            df_match, df_match_player, df_match_odds = scraper_flashscore.extract_data(self.id_country, self.country, row['id_competition'], 
                                                                                       row['competition_flashscore'], row['is_cup'], n_seasons_max=11,
                                                                                       export=export
                                                                                       )
            
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

    def __init__(self, id_country, country, date, var_resp: str = 'result', verbose: int = 0):
        self.id_country = id_country
        self.country = country.lower()
        self.date = date
        self.var_resp = var_resp
        self.verbose = verbose

        # Tiene que estar aca por prod?
        if self.date is not None:
            path = f'./data/{self.country}/p3_data_preparation/{self.date}'
        else:
            path = f'./data/{self.country}/p3_data_preparation'
        self.base_path = path
        
        self.make_directories()

    def make_directories(self):

        # Creo directorios para la preparacion actual
        l_directorios = [
            f'{self.base_path}/format_data',
            f'{self.base_path}/clean_data',
            f'{self.base_path}/integrate_data',
            # f'{self.base_path}/fill_data',
            f'{self.base_path}/construct_data',
            f'{self.base_path}/clean_data_2',
            f'{self.base_path}/treat_nan',
            f'{self.base_path}/tag',
            f'{self.base_path}/select_data',
        ]  
        directories.make_directories(l_directorios=l_directorios)

    def format_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, reformat: bool = True, export: bool = True):
        """
        Arreglo el data data_type de algunas variables.

        :param df_match: Dataframe de los datos de los partidos. (DataFrame)
        :param df_player_fifa_sofifa: Dataframe de los datos de los players por fifa. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        logger.info("Formatting data...")

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

        if reformat: # Solo para missing
            # Formateo nuevas columnas...
            logger.warning("Reformateo columnas stats nuevas")
            base_columns = ['passes', 'passes_in_the_final_third', 'crosses', 'tackles']
            df_match = format_data.format_percentage_columns(df_match, base_columns)  # Uso nombres ≠ que los df_match x si tmb existen dichas columnas en missing. (eso lo tendre en cuenta en el renombre de columnas...)

            # Renombro columnas para usar el mismo nombre que en los datos viejos (por ende, se concatenen juntas) 
            rename_dict = {
                # nombre en missing: --> nombre en datos viejos
                "n_passes_home" : "total_passes_home",
                "n_passes_away": "total_passes_away",
                "n_correct_passes_home" : "completed_passes_home",
                "n_correct_passes_away": "completed_passes_away",
                "accuracy_passes_home": 'pass_success_%_home',
                "accuracy_passes_away": 'pass_success_%_away',
                'n_clearances_home': 'clearances_completed_home', # No hay n_correct clearances a veces?
                'n_clearances_away': 'clearances_completed_away',
                'n_correct_tackles_home': 'tackles_home',  # "total_tackles_home" : "tackles_home",
                'n_correct_tackles_away': 'tackles_away',  # "total_tackles_away": "tackles_away",
            }
            df_match = format_data.rename_and_merge_columns(df_match, rename_dict)
                
        # Dataframe player_fifa_sofifa
        ## Fecha
        df_player_fifa_sofifa['date'] = pd.to_datetime(df_player_fifa_sofifa['date'], format='%b %d, %Y')
        ## Market value
        df_player_fifa_sofifa = format_data.convert_value_to_int(df_player_fifa_sofifa)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_match.to_excel(f'{self.base_path}/format_data/df_match_formated.xlsx', index=True)
            df_match_player.to_excel(f'{self.base_path}/format_data/df_match_player_formated.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'{self.base_path}/format_data/df_player_fifa_sofifa_formated.xlsx', index=True)

        return df_match, df_match_player, df_player_fifa_sofifa

    def clean_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, export: bool = True):
        """
        Limpieza inicial de los dataframes
        """
        start = time.time()
        logger.info("\nCleanning data...")
        # Elimino partidos viejos sin estadisticas y sin datos de jugadores
        n_rows_inic = len(df_match)
        df_match['date'] = pd.to_datetime(df_match['date'])  # Asegurarte de que la columna 'date' sea de tipo datetime (si no lo es ya)
        
        start_date = '2015-01-01' # Filtrar por fecha (por ejemplo, para filtrar datos desde una fecha específica)
        df_match = df_match[df_match['date'] >= start_date]
        df_match_player = df_match_player[df_match_player.index.isin(df_match.index)] # Es clave para eliminar jugadores y hacer una mejor integracion (tener menos falsos positivos)
        logger.warning(f"Partidos jugados antes de {start_date} eliminados. {n_rows_inic} --> {len(df_match)}. {len(df_match_player)}")

        # Elimino columnas de jugadores que son todo NaN (se ve que hay porque las creo y no les guardo nada eso debe ser porque obtengo nombres solo si tiene url)
        non_object_columns = df_match_player.select_dtypes(exclude=['object']).columns
        df_match_player.drop(columns=non_object_columns, inplace=True)
        
        if self.verbose >= 1:
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
        if self.verbose >= 0 and n_players_eliminated > 0:
            logger.warning(f"Se eliminaron {n_players_eliminated} jugadores de los {len_inic} de Sofifa que habia.")
        ### Dataframe player sofifa (df)
        df_player_sofifa = clean_data.prepare_text_columns(df_player_sofifa, l_cols_to_process=['player_name', 'player_name_short'])  # Preaparo texto para integrar

        # Correcion de valores
        if self.verbose >= 1:
            print("\nCorrecion de valores")
        df_player_fifa_sofifa['fifa_year'] = df_player_fifa_sofifa['fifa'].str.split(' ').str[-1]  # Agrego columna "fifa_year" quedandome solo con el año del fifa (e.g. "22" en vez de "FIFA 22")
    
        end = time.time()
        print(f"Clean data in {(end - start) / 60:.1f} minutes")

        if export:
            df_match.to_excel(f'{self.base_path}/clean_data/df_match_cleaned.xlsx', index=True)
            df_match_player.to_excel(f'{self.base_path}/clean_data/df_match_player_cleaned.xlsx', index=True)
            df_player_sofifa.to_excel(f'{self.base_path}/clean_data/df_player_sofifa_cleaned.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'{self.base_path}/clean_data/df_player_fifa_sofifa_cleaned.xlsx', index=True)
    
        return df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa

    def verify_format(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, export: bool = True):
        """
        Verificacion de formato
        """
        logger.info("\nVerifying data format...")

        # Flashscore
        df_match = format_data.format_df_match(df_match)
        # df_match_player = format_data.format_df_match_player(df_match_player)
        df_match_odds = format_data.format_df_match_odds(df_match_odds)

        # Sofifa
        df_player_sofifa = format_data.format_df_player_sofifa(df_player_sofifa)
        df_player_fifa_sofifa = format_data.format_df_player_fifa_sofifa(df_player_fifa_sofifa)
        return df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa

    def integrate_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, prod: bool = False, export: bool = True):
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        Parameters:
            df_match (pd.DataFrame): Dataframe de los datos de los partidos.
            df_match_player (pd.DataFrame): Dataframe de los datos de los jugadores en cada partido.
            df_player_sofifa (pd.DataFrame): Dataframe de los datos de los jugadores en Sofifa.
            df_player_fifa_sofifa (pd.DataFrame): Dataframe de los datos de los jugadores en Fifa-Sofifa.
            prod: Cuando entreno tengo que mapear, en cambio, cuando estoy en produccion tengo que usar el mapeo de cuando entrené (bool)
            export (bool): Booleano para indicar si se debe exportar el dataframe integrado. True para exportar y False para no exportar.
        
        Returns:
            pd.DataFrame: Dataframe integrado.
        """
        start = time.time()
        logger.info("\nIntegrating data...")
        export = False if prod else export  # No exporto datos en produccion para no sobreescribir los de train y poder reutilizarlos.

        # (Temporalmente) Obtengo el listado de equipos unicos de Flashscore
        # if export:
        df_teams = integrate_sofifa_to_flashscore.create_df_teams(df_match)
        if not prod:
            df_teams.to_excel(f"{self.base_path}/integrate_data/df_teams.xlsx", index=True)

        # PLAYERS --> MATCH (Mapeo df_player_sofifa con df_player e integro a df_match)
        print("\nIntegrating player's data to df_match...")

        # Segun si es train o produccion (en el 1ero hago el mapeo, en el 2do uso el mapeo ya hecho)
        if prod:
            logger.critical("Integración para produccion")

            df_map_players_fs_so = pd.read_excel(f'{self.base_path}/integrate_data/df_map_players_fs_so.xlsx', index_col=0)
            logger.info(f"No vuelvo a mapear sino que levanto df_map del pais (para producción) \n {df_map_players_fs_so.head(3)}")

        else: 
            logger.critical("Integración para train")

            # Matcheo jugadores de Sofifa y Flashscore
            df_map_players_fs_so = integrate_sofifa_to_flashscore.map_players(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, id_country=self.id_country, base_path=self.base_path)

        # Integro datos de jugadores a df_match usando el mapeo
        df, df_aux = integrate_sofifa_to_flashscore.integrate_player_data_in_match(df_match, df_match_player, df_map_players_fs_so, df_player_sofifa, df_player_fifa_sofifa)

        # Drop de columnas que use para df_teams, df_player, df_coaches, etc..
        cols_to_drop = ['team_home', 'team_away', 'coach_home', 'coach_away', 'fifa_year'] # main_next a veces no tiene coaches.. deeberia copiar antes...
        cols_to_drop_filt = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(cols_to_drop_filt, axis=1)

        # Verificar cantidad de NaN values en variables jugadores
        l_player_cols  = [col for col in df.columns if ('player_start' in col) or ('player_sub' in col)]  # Selecciono las variables que corresponden a jugadores
        for col in l_player_cols:
            nan_percentage = df[col].isna().mean() * 100  # Porcentaje de NaN
            if nan_percentage < 50:  # Si el porcentaje de NaN es menor al 50%
                if prod: # En prod a veces recoje un solo partido y por ahi justo ni siquiera es de la comp public..
                    logger.warning(f"La columna '{col}' tiene menos del 50% de valores NaN: {nan_percentage:.2f}%. Revisar posible diferencia en formato en columnas usadas al integrar.")
                else:
                    raise ValueError(f"La columna '{col}' tiene menos del 50% de valores NaN: {nan_percentage:.2f}%. Revisar posible diferencia en formato en columnas usadas al integrar.")

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")
        
        if export:
            df.to_excel(f'{self.base_path}/df_integrated.xlsx', index=True)
            df_aux.to_excel(f'{self.base_path}/integrate_data/n_players_integrated.xlsx', index=True)

        return df

    def clean_data_3(self, df: pd.DataFrame, competencies_to_select: list = None, export: bool = True):
        """
        CLEAN DATA ANTES DE CONSTRUIR. Eliminacion de columnas
        """
        # Ordeno valores por fecha y separo X e y
        df = df.sort_values(by='date', ascending=False)

        # (1) Eliminacion de filas 
        ## Para evitar ciertas competencias
        if competencies_to_select is not None:
            n_reg_inic = len(df)
            df = df[df['id_competition'].isin(competencies_to_select)]

            if self.verbose >= 0:
                print("Eliminacion de filas...")
                print(f"Eliminacion por competencias. Cantidad de filas: {n_reg_inic} --> {len(df)}")

        # (2) Eliminacion de columnas         
        ## usadas solo para construir y constantes
        cols_constants = list(df.columns[df.nunique() == 1])  # Elimino columnas constantes
        cols_basics_noise = ['attendance', 'capacity', 'venue', 'referee']  # --> generan problemas de convergencia por ser nros altos y ademas su info puede ser importante junta y no separada.        
        col_players_noise = [col for col in df.columns if 'rep_player' in col or 'hei_player' in col] # 'wage_player' in col  # Elimino variables jugadores que meten ruido (lo hago aqui antes de que construya mil columnas mas...)
        cols_to_drop = cols_constants + cols_basics_noise + col_players_noise
        df.drop(columns=cols_to_drop, inplace=True)
        if self.verbose >= 0:
            print("Eliminación de columnas...")
            print(f"Columnas constantes eliminadas: {cols_constants}")
            print(f"Columnas eliminadas x posible ruido: {cols_to_drop}")
            print(df.shape)

        ## Elimino stats irrelevantes
        # Determino cuales son las variables stats automaticamente
        stats_columns = construct_data.determine_stats_columns(df)
        relevant_stats = self.determine_stats_to_use()
        df = clean_data.delete_not_relevant_stats(df, stats_columns=stats_columns, relevant_stats_columns=relevant_stats)
        print(df.shape)

        return df
    
    def determine_stats_to_use(self):
        """
        Es necesaria para usarla desde prod.
        Mejoras: Garantizar que esten en df.columns...
        """
        self.stats_to_derive = ['fouls', 'tackles', 'interceptions', 'clearances_total', 'blocked_shots', 'throw-ins', 'corner_kicks', 'free_kicks']

        self.stats_to_construct = [
            # Ofensive
            'expected_goals_(xg)', 
            'shots_on_goal', 'goal_attempts', 'goals', 'points','PPS', 'goal_ratio', 
            'dead_balls', 
            'ball_possession', 'total_passes', 'attacking_efficiency',
            # Defensive
            'yellow_cards', 'red_cards', 'defensive_actions', 
            'PPDA', 'clean_sheet', 'defensive_efficiency',  "efficiency" 
        ]
        return self.stats_to_derive + self.stats_to_construct

    def construct_data(self, df: pd.DataFrame, n_last_matches: list, l_days: list , n_years_h2h: int, segun_localia: bool, with_h2h: bool = True, with_historic: bool = True, 
                       dif_con_against: bool = True, export: bool = True):
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de players. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        start = time.time()
        logger.info("Constructing data...")

        # Si quiero construir variables historicas
        if with_historic:

            # VARIABLE RESPUESTA (no son historicas estan filtradas)
            df = construct_data.determine_result(df, self.var_resp)
            df = construct_data.determine_points(df)

            # VARIABLES DERIVADAS
            ## OFENSIVE
            ## Goal ratio
            df = construct_data.construct_percentaje_column(df, col_num='goals', col_den="goal_attempts", laplace=True,  column_name="goal_ratio") # G2S # Similar a G2A
            # df = construct_data.construct_percentaje_column(df, col_num='goals', col_den="shots_on_goal", laplace=True,  column_name="G2SOG")  # Similar a G2A

            # Traduccion de posesion a tiros
            df = construct_data.construct_percentaje_column(df, col_num='goal_attempts', col_den="total_passes", laplace=True,  column_name="PPS")  # home = home / home

            # Dead balls
            df = construct_data.construct_sum_columns(df, l_columns=['throw-ins', 'corner_kicks', 'free_kicks'], column_name="dead_balls") #  # home = home + home

            # Attacking efficiency --> (lo evito por cantidad de NaN)
            df['attacking_efficiency_home'] = np.where(df['expected_goals_(xg)_home'].notna(), df['goals_home'] - df['expected_goals_(xg)_home'], None)
            df['attacking_efficiency_away'] = np.where(df['expected_goals_(xg)_away'].notna(), df['goals_away'] - df['expected_goals_(xg)_away'],  None)

            ## DEFENSIVE 
            ## Passess per defensive action (PPDA) --> (no es solamente en el 60% de la cancha pues no tengo ese dato)
            df = construct_data.construct_sum_columns(df, l_columns=['fouls', 'tackles', 'interceptions', 'clearances_total', 'blocked_shots'], column_name="defensive_actions") # Calculo defesive actions  # home = home + home
            df['PPDA_home'] = np.where(df['defensive_actions_home'].notna(),  df['total_passes_away'] / df['defensive_actions_home'], None)
            df['PPDA_away'] = np.where(df['defensive_actions_away'].notna(), df['total_passes_home'] / df['defensive_actions_away'],  None)

            # Clean Sheets
            df['clean_sheet_home'] = (df['goals_away'] == 0).astype(int)
            df['clean_sheet_away'] = (df['goals_home'] == 0).astype(int)

            # Defensive efficiency (en la teoria esto es KGP) --> (lo evito por cantidad de NaN)
            df['defensive_efficiency_home'] = np.where(df['expected_goals_(xg)_away'].notna(),  df['expected_goals_(xg)_away'] - df['goals_away'], None)
            df['defensive_efficiency_away'] = np.where(df['expected_goals_(xg)_home'].notna(), df['expected_goals_(xg)_home'] - df['goals_home'],  None)

            # Efficiency --> (lo evito por cantidad de NaN)
            df['efficiency_home'] = df['attacking_efficiency_home'] + df['defensive_efficiency_home']
            df['efficiency_away'] = df['attacking_efficiency_away'] + df['defensive_efficiency_away']

            # VARIABLES HISTORICAS
            ## 1) EN ULTIMOS N PARTIDOS
            for n_matches in n_last_matches:
                df = construct_data.determine_number_results_last_matches(df, n_matches=n_matches, segun_localia=False) # Hay que ver si funciona tanto sin como con localia.

            ## 2) EN PARTIDOS EN ULTIMOS N DAYS
            if with_h2h:
                df = construct_data.h2h_by_date(df, n_years=n_years_h2h)
                df = construct_data.h2h_by_date_by_localia(df, n_years=n_years_h2h)

            # Numero de partidos jugados en ultimos n days
            for n_days in l_days:
                df = construct_data.determine_number_matches_last_days(df, n_days=n_days) 

            # Por stat (e.g. shots_on_goal)
            logger.info(f"Stats a promediar en ultimos partidos: {self.stats_to_construct}")
            for var in self.stats_to_construct: 
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
            df.to_excel(f'{self.base_path}/df_constructed.xlsx', index=True)

        return df

    def tag_string_data_to_integer(self, df: pd.DataFrame, export: bool = True):
        """
        Conversion de columnas tipo "object" a "integer"
        """
        # Elimino columna 'season'
        df = df.drop(['season'], axis=1)  # Arroja error TypeError porque tiene tanto str como int en los valores originales y el label solo puede recibir un tipo (str o int). Season tiene valores como "2021" y "2020_2021", los primeros los entiende como int y los segundos como str.
        
        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df, verbose=self.verbose)

        if export:
            df_etiquetas.to_excel(f'{self.base_path}/df_etiquetas.xlsx', index=False)
            df.to_excel(f'{self.base_path}/df_constructed_etiquetado.xlsx', index=True)
        return df, df_etiquetas
    
    def clean_data_2(self, df: pd.DataFrame, n_years_to_select: int = None, fill_na: str = None, index_test_set: list = None, export: bool = True):
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
        if n_years_to_select is not None:
            n_reg_inic = len(X)
            fecha_limite = X.iloc[0]['date'] - datetime.timedelta(days=n_years_to_select*365)
            X = X[X['date'] >= fecha_limite] 

            if self.verbose >= 0:
                print("Eliminacion de filas...")
                print(f"Eliminacion por fecha. Cantidad de filas: {n_reg_inic} --> {len(X)}")

        # (2) Eliminacion de columnas usadas para construir
        cols_for_construct = ['date', 'id_team_home', 'id_team_away']  # Elimino variables que no usare en el modelo fecha (la idea es usar todas las posibles)
        cols_to_derive_others = [col for col in X.columns if any(stat in col for stat in self.stats_to_derive)]
        cols_to_drop = cols_for_construct + cols_to_derive_others
        X.drop(columns=cols_to_drop, inplace=True)
        logger.warning(f"Columnas eliminadas: {cols_for_construct}")
        logger.warning(f"Columnas eliminadas (solo usadas para derivar otras): {cols_to_derive_others}") # Cuidado en que se eliminen todas las stats usadas para derivar.

        # (2) Tratamiento de NaN values
        shape_inicial = X.shape
        X, df_filled_columns = self.treat_nan_values(X=X, fill_na=fill_na, index_test_set=index_test_set)
        if self.verbose >= 1:
            print(f"Tras fill_na={fill_na}. Shape X_sin_col_mucho_nan: {shape_inicial} --> {X.shape}")

        # (3) Escalado de datos
        if self.verbose >= 1:
            print("\nEscalado de datos...")
        scaler = StandardScaler()
        scaler.fit(X) # Paso 1: Ajusta el StandardScaler a tus datos
        X_scaled = scaler.transform(X) # Paso 2: Transforma tus datos utilizando el StandardScaler ajustado
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Concateno X e y
        y_sin_nan = y[y.index.isin(X.index)] # Dado que elimine filas de X
        df = pd.concat([X_scaled_df, y_sin_nan], axis=1)

        end = time.time()
        logger.info(f"Clean data 2 en {(end - start)/60:.1f} minutos")

        if export: 
            joblib.dump((scaler, X.columns), f"{self.base_path}/scaler_model.pkl")       
            df.to_excel(f'{self.base_path}/df_constructed_clean.xlsx', index=True)

        return df, scaler, X.columns, df_filled_columns
    
    def treat_nan_values(self, X: pd.DataFrame , fill_na: str = None, index_test_set: list = None, porc_nan_max: float = 0.4, percentil_nan: int = 75, export: bool = True):
        """
        Tratamiento de nan values

        # Para ENG y SPA usé porc_min_no_nan = 0.5. Pero para FRA tengo que usar 0.8 porque sino X_test queda vacio.
        Tan alto puede tirar error porque elimina todas las columnas.

        # Parameters
            df: Dataframe a tratar nan values. (DataFrame)
            fill_na: Tipo de rellenado de NaN values.
            porc_nan_max: Porcentaje de NaN values maximo tolerado (tanto en filas como columnas) (float)
            percentil_nan: Percentil para definir que columnas son consideradas con mucho nan y cuales con poco nan. Solo cuando haces fillna.
            export: 
            _print:

        # Returns
            Dataframe sin NaN values
        """ 
        start = time.time()
        print("\nTreating NaN values to avoid input=NaN in Modeling...")

        # Separo test y train/val
        df_test = X[X.index.isin(index_test_set)]
        df_train_val = X[~X.index.isin(index_test_set)]
        logger.info(f"{X.shape} --> {df_train_val.shape} {df_test.shape}")

        # (1) Eliminacion de filas con mucho NaN (filas sin estadisticas ni formaciones)
        n_reg_inic = len(df_train_val)
        df_train_val = clean_data.delete_rows_nan(df_train_val, porc_nan_max=porc_nan_max) # no mas del 50% de nan 
        if self.verbose >= 0:
            print(f"(1) Eliminaccion por mucho NaN. Cantidad de filas: {n_reg_inic} --> {len(df_train_val)}")
            logger.warning(f"Cantidad de filas: {n_reg_inic} --> {len(df_train_val)}")

        # (2) Eliminacion de columnas con mucho NaN --> Elimino columnas con alto porcentaje de NaN values (de manera que tras el dropna quedarian menos de n_reg_min)
        X_sin_col_mucho_nan = df_train_val.copy()
        df_train_val = clean_data.delete_columns_nan(df_train_val, porc_nan_max=porc_nan_max)
        l_col_eliminated = list(X_sin_col_mucho_nan.columns.difference(df_train_val.columns))
        if self.verbose >= 0 and len(df_train_val.columns) != len(X_sin_col_mucho_nan.columns):
            logger.warning(f"(2) Eliminacion de columnas con mucho Nan. Se eliminaron {len(X_sin_col_mucho_nan.columns) - len(df_train_val.columns) } columnas de {len(X_sin_col_mucho_nan.columns)} por tener mas de {porc_nan_max*100:.0f}% de NaN values. Columnas eliminadas: {l_col_eliminated}")
            logger.info(f"Shape X_sin_col_mucho_nan: {X_sin_col_mucho_nan.shape} --> {df_train_val.shape} ")

        # (3) Eliminacion de todo NaN ya sea drop o fill_na
        ## Reemplazo NaN en TRAIN y VALIDATION 
        if fill_na is not None:
            
            # Determino las columns con mucho NaN (mas de nan_threshold%)
            l_columns_poco_nan, l_columns_mucho_nan = clean_data.determine_columns_to_fill(df_train_val, percentil_nan=percentil_nan) 

            # Elimino registros con al menos un NaN 
            largo_inic = len(df_train_val)
            df_train_val = df_train_val.dropna(subset=l_columns_poco_nan)
            if self.verbose >= 1:
                print(f"De las {largo_inic} filas, se han eliminado {largo_inic-len(df_train_val)} por tener al menos un Nan value. Quedan {len(df_train_val)} filas. Shape final: {df_train_val.shape}") 

            # Determino que filas relleno y cuales no (antes de fill porque despues de rellenar no puedo diferenciar que filas rellene y cuales no)
            df_rellenado = pd.DataFrame(index=df_train_val.index)
            df_rellenado['rellenado'] = df_train_val[l_columns_mucho_nan].isnull().any(axis=1)
            df_rellenado.to_excel(f'{self.base_path}/treat_nan/df_rellenado.xlsx', index=True)

            # Relleno nan de las columnas con mucho NaN
            df_train_val_filled = clean_data.fill_nan_values(df_train_val, l_columns_mucho_nan, fill_type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las accuracyes casi siempre seran mayores que dropna() en train y test, lo que cuenta es la accuracy en next_matches o en un dataset que no haya sido filleado...
            if self.verbose >= 1:
                print(f"Columnas consideradas con mucho nan (a las cuales rellenar): {l_columns_mucho_nan}")
                print(f"\tSe realizó el rellenado de NaN values. Shape X luego de rellenado: {df_train_val_filled.shape}")

        else:
            # Elimino registros con al menos un NaN 
            df_train_val_filled = df_train_val.dropna(subset=df_train_val.columns)

        # Tratamiento de nan values para test
        ## Eliminar columnas que se eliminaron x nan
        columns_to_drop = [col for col in l_col_eliminated if col in df_test.columns]
        df_test_filt = df_test.drop(columns=columns_to_drop)
        logger.info(f"{df_test.shape} --> {df_test_filt.shape}")
        ## Reemplazo NaN en TEST por 0
        df_test_filled, df_filled_columns = self.emergency_fill_for_test(df_test_filt)

        # Concateno df_test y df_train_val ya rellenados
        X = pd.concat([df_train_val_filled, df_test_filled], axis=0)

        # Imprimo cantidad de registros que quedan en test
        if self.verbose >= 0:        
            n_rows = generate_test_design.n_rows_to_test(X, df_test_filled)

            if n_rows != len(index_test_set):
                logger.warning(f" Se han eliminado registros de df_test por tener NaN values cuando no deberia borrarse ninguno.")

        logger.info(f"(3) Tras eliminar todo NaN con drop o fill_na: {df_train_val_filled.shape} {df_test_filled.shape} --> {X.shape}")

        end = time.time()
        print(f"Tratamiento de NaN values en {(end - start)/60:.1f} minutos")

        if export:
            X.to_excel(f'{self.base_path}/df_selected_nan.xlsx', index=True)

        return X, df_filled_columns
        
    def emergency_fill_for_test(self, df, export: bool = True):
        """
        Remoción de valores NaN en df_test

        # Parameters:
            df: Dataframe al cual remover NaN values. (DataFrame)
            columns_selected: Listado de columnas seleccioandas para usar en produccion. (list)
        
        # Returns:
            df: Dataframe pasado como parametro sin registros con al menos un NaN value. (DataFrame)
        """
        logger.info("Treating NaN values in df_test...")
        df_filled_columns = pd.DataFrame(0, index=df.index, columns=['emergency_fill', 'player_emergency_fill', 'n_col_filled_sin_player', 'n_col_filled', 'perc_col_filled', 'l_col_filled']) # Inicializo el df

        # Rellenar NaN en algunas columnas espeecificas
        columns_to_fill = [col for col in df.columns if df[col].isna().any()]  # En teoria, solo rellena las variables historicas que son nan.
        if self.verbose >= 1:
            logger.info(f'Nº columnas a rellenar: {len(columns_to_fill)}')

        if columns_to_fill:
            # Crear una copia del DataFrame y rellenar los NaN
            df_copy = df.copy()
            df_copy[columns_to_fill] = df_copy[columns_to_fill].fillna(0)

            # Identificar filas donde se rellenaron NaN
            filled_rows = (df[columns_to_fill].isna() & (df_copy[columns_to_fill] == 0)).any(axis=1)

            # Crear DataFrame con las columnas rellenadas y `emergency_fill`
            df_filled_columns = df_copy.loc[filled_rows, columns_to_fill]
            df_filled_columns['emergency_fill'] = 1

            # Crear una columna con el listado de columnas rellenadas por cada registro
            df_filled_columns['l_col_filled'] = df[columns_to_fill].apply(
                lambda row: [col for col in columns_to_fill if pd.isna(row[col]) and df_copy.at[row.name, col] == 0], axis=1
            )

            # Columna con valor 0 o 1 según si se rellenaron columnas con "player_start" o "player_sub"
            player_columns = [col for col in columns_to_fill if "player_start" in col or "player_sub" in col]  # Esta bien missing tmb?
            columns_to_fill_sin_player = [col for col in columns_to_fill if col not in player_columns]
            df_filled_columns['player_emergency_fill'] = (
                (df[player_columns].isna() & (df_copy[player_columns] == 0)).any(axis=1).astype(int)
            )

            # Calcular cuántas columnas se rellenaron de emergencia para cada fila
            df_filled_columns['n_col_filled'] = (
                (df[columns_to_fill].isna() & (df_copy[columns_to_fill] == 0)).sum(axis=1)
            )

            # Calcular cuántas columnas se rellenaron de emergencia para cada fila
            df_filled_columns['n_col_filled_sin_player'] = (
                (df[columns_to_fill_sin_player].isna() & (df_copy[columns_to_fill_sin_player] == 0)).sum(axis=1)
            )
            
            # Calcular el porcentaje de columnas rellenadas de emergencia para cada fila --> es ANTES de seleccionar las columnas... TAl vez ni siquiera usas esas columnas rellenadas.
            df_filled_columns['perc_col_filled'] = (
                df_filled_columns['n_col_filled'] / len(columns_to_fill) * 100
            )

            # Calcular el porcentaje de columnas rellenadas de emergencia para cada fila --> es ANTES de seleccionar las columnas... TAl vez ni siquiera usas esas columnas rellenadas.
            df_filled_columns['perc_col_filled_sin_player'] = (
                df_filled_columns['n_col_filled_sin_player'] / len(columns_to_fill) * 100
            )
            
            if self.verbose >= 0:
                # Calcular y mostrar el porcentaje de NaN por cada columna
                for col in columns_to_fill:
                    nan_percentage = df[col].isna().mean() * 100

                    if self.verbose >= 1:
                        logger.warning(f"Columna '{col}' tiene {nan_percentage:.1f}% de valores NaN.")

            # Actualizar el DataFrame original
            df = df_copy

        # Elimino partidos con al menos un NaN value --> Tal vez lo deberia poner al ppio para imprimir warning de cuantos partidos eliminaria...
        df_sin_dup = df.dropna()
        if self.verbose >= 1 and (len(df) != len(df_sin_dup)):
            logger.warning(f"De los {len(df)} partidos, no se hará la prediccion para {len(df)-len(df_sin_dup)} partidos puesto que tienen al menos un valor NaN y el modelo no puede tener input NaN.")

        if export:
            df_sin_dup.to_excel(f'{self.base_path}/treat_nan/df_treat_nan.xlsx', index=True)
            df_filled_columns.to_excel(f'{self.base_path}/treat_nan/df_filled_columns.xlsx', index=True)

        return df_sin_dup, df_filled_columns
        
    def select_data(self, df: pd.DataFrame, thr_corr: float = None, thr_fs: float = None, export: bool = True):
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

            if self.verbose >= 1:
                print('\n Eliminando columnas correlacionadas...')
                print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")

            if export:
                df_corr_tri_X.to_excel(f'{self.base_path}/select_data/df_correlation.xlsx', index=True)

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            n_cols = len(df.columns)-1  # -1 por variable respuesta
            l_important_features, df_normalized = select_data.select_best_features(df, self.var_resp, thr_fs, graf=False)
            l_col_eliminated = list(df.columns.difference(l_important_features))
            df = df.loc[:, l_important_features + [self.var_resp]]

            if self.verbose >= 1:
                print('\n Feature Selection...')
                print(f"\tSe eliminaron {n_cols-len(l_important_features)} de {n_cols} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%. Columnas eliminadas: {l_col_eliminated}")

            if export:
                df_normalized.to_excel(f'{self.base_path}/select_data/df_fs.xlsx', index=True)

        if self.verbose >= 1:
            print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")
        
        end = time.time()
        logger.info(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'{self.base_path}/df_selected.xlsx', index=True)

        return df
    

class Modeling:

    def __init__(self, country: str, date: str = None, var_resp: str = 'result', var_pred: str = 'predicted_result', verbose: int = 0):
        if not isinstance(var_resp, str) or not isinstance(var_pred, str):
            raise TypeError("Los parámetros var_resp y var_pred deben ser cadenas de texto.")
        if not isinstance(country, str):
            raise TypeError("El parámetro country debe ser una cadena de texto.")

        self.country = country.lower()
        self.date = date
        self.var_resp = var_resp
        self.var_pred = var_pred
        self.verbose = verbose
        self.make_directories()

    def make_directories(self):

        if self.date is not None:
            path = f'./data/{self.country}/p4_modeling/{self.date}'
            path_dp = f'./data/{self.country}/p3_data_preparation/{self.date}'
        
        else:
            path = f'./data/{self.country}/p4_modeling'
            path_dp = f'./data/{self.country}/p3_data_preparation'

        self.base_path = path
        self.base_path_dp = path_dp

        # Levanto df_teams (lo hago 1 vez para todas las veces que use el reformateo)
        self.df_teams = pd.read_excel(f'{self.base_path_dp}/integrate_data/df_teams.xlsx', index_col=0)

    def generate_test_design(self, df: pd.DataFrame, bal_type: str = None, val_size: float = 0.15, index_test_set: list = None, export: bool = True):
        """
        Separa conjuntos de datos en train, validacion y test, balancea las clases del dataset y elimina los NaN values.

        # Parameters
            df: Dataframe a dividir en test, validation y train. (Dataframe)
            bal_type: Tipo de balanceo de clases a realizar. (String)
            val_size: Porcentaje del total de datos destinado a validacion. (Float)
            export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (Bool)
            
        # Returns
            Dataframe de entrenamiento y de testeo balanceados (DataFrame)
        """
        # warnings.filterwarnings('ignore') # no son mias, son de openpyxl
        if self.verbose >= 0:
            print("\nSeparating data in train, val and test...")

        # Selecciono test set
        df_test = df[df.index.isin(index_test_set)]
        X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]

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

        if self.verbose >= 0:
            print(f'Train: {X_train.shape} {y_train.shape}', f'\nVal: {X_val.shape} {y_val.shape}', f'\nTest: {X_test.shape} {y_test.shape}')

        if export:
            X_train.to_excel(f'{self.base_path}/generate_test_design/X_train.xlsx', index=True)
            X_val.to_excel(f'{self.base_path}/generate_test_design/X_val.xlsx', index=True)
            X_test.to_excel(f'{self.base_path}/generate_test_design/X_test.xlsx', index=True)
            y_train.to_excel(f'{self.base_path}/generate_test_design/y_train.xlsx', index=True)
            y_val.to_excel(f'{self.base_path}/generate_test_design/y_val.xlsx', index=True)
            y_test.to_excel(f'{self.base_path}/generate_test_design/y_test.xlsx', index=True)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, default_model, X_val: pd.DataFrame, y_val: pd.DataFrame, X_train: pd.DataFrame, y_train, k: int, params: dict = None, 
                    bayes: bool = False, compare_tuning: bool = False, export: bool = True):
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
            pickle.dump(model_best_params, open(f"{self.base_path}/modelo.pkl", "wb"))
            # results.to_excel(f"./data/{self.country}/{ite_date}/p4_modeling/models/hiperparametros.xlsx")    # Exportar metricas por cada combinacion de hiperparametros (En vez de retornar best_metric.)

        return model_best_params, params, train_accuracy, results

    def assess_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, df_match: pd.DataFrame, df_match_odds: pd.DataFrame, df_filled: pd.DataFrame = None, prod: bool = False, export: bool = False):
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
        df_probabilities, y_pred = self.predict(model, X_test)
            
        # Combinar ambos DataFrames
        if not prod:
            df_pred_proba = pd.DataFrame({
                    self.var_resp: y_test,
                    self.var_pred: y_pred,
                }, index=X_test.index)
        else:
            # En producción (excluye y_test)
            df_pred_proba = pd.DataFrame({
                self.var_pred: y_pred,
            }, index=X_test.index)
        df_pred_proba = pd.concat([df_pred_proba, df_probabilities], axis=1)

        # Concateno todos los dfs en uno solo --> Necesario para roi?
        df_predicciones = asses_model.concatenate_dfs(df_pred_proba=df_pred_proba, df_match=df_match, df_match_odds=df_match_odds, df_filled=df_filled)

        # Calculo metricas
        d_metrics = None
        if not prod:
            df_predicciones, d_metrics = self.calculate_metrics(df_predicciones, export=export)
        
        df_predicciones = self.reformat_pred(df_predicciones)
        
        if export:
            df_predicciones.to_excel(f'{self.base_path}/modeling/df_predicciones.xlsx')

        return (df_predicciones, d_metrics) if not prod else df_predicciones
    
    def predict(self, model, X_test: pd.DataFrame):
        """
        Predigo sobre X_test + Mapeo clases entre test y pred (pred genera clases con ≠ valor).
        """
        # Predecir las etiquetas para los datos de prueba
        try:
            y_pred_prob = model.predict_proba(X_test) # Te da las probabilidad de cada clase. Funciona para todos los modelos? # AttributeError: predict_proba is not available when probability=False
            
            if self.verbose >= 1:
                class_distribution = y_pred_prob.mean(axis=0)
                print("Distribución promedio de probabilidades por clase:", class_distribution)
                
        except AttributeError: # AttributeError: 'Sequential' object has no attribute 'predict_proba'
            y_pred_prob = model.predict(X_test)
            
        # Mapeo clases de y_test e y_pred (y_pred son ≠ nros)
        if hasattr(model, "classes_"):
            model_classes = model.classes_

            # Obtener la clase predicha basada en el índice con mayor probabilidad
            y_pred_indices = np.argmax(y_pred_prob, axis=1)  # Índices de las clases predichas

            # Reemplazar los índices por las clases del modelo
            y_pred = np.array([model_classes[idx] for idx in y_pred_indices])
            
            # Verificar el mapeo de índices a clases (opcional)
            if self.verbose >= 1:
                logger.info(f"Mapeo de índices a clases: {dict(enumerate(model_classes))}")
                logger.info(y_pred)
        else:
            raise ValueError("El modelo no tiene el atributo 'classes_', no se puede determinar el mapeo.")

        # Crear un DataFrame para visualizar las probabilidades con sus clases
        df_pred_proba = pd.DataFrame(
            y_pred_prob,
            columns=[f'prob_class_{cls}' for cls in model_classes],
            index=X_test.index
        )

        if self.verbose >= 1:
            logger.info(df_pred_proba)

        return df_pred_proba, y_pred   

    def calculate_metrics(self, df_pred_proba, export: bool = False):
        
        d_metrics = {}
        df_pred_proba = asses_model.calculate_result_probabilities_by_bookmaker(df_pred_proba) # Caculo probabilidades segun casa de apuesta
        df_pred_proba = asses_model.determine_result_by_bookmaker(df_pred_proba, col_name="bookmaker_result")  # Determino resultado predicho segun cuota minima (e.g. "Home")

        # Calculo metricas
        d_metrics.update(asses_model.calculate_basic_metrics(df_pred_proba, country=self.country, export=export))
        d_metrics.update(asses_model.calculate_bet_metrics(df_pred_proba))
        d_metrics.update({'dif_prec_bm': d_metrics['test_accuracy'] -  d_metrics['test_accuracy_bm']})

        # Calculo ROI
        bs = betting_strategy.BettingStrategy()  # Al no pasarle iteration_date no inicializa directories de betting strategy
        d_params = bs.define_hiperparameters(strategy='train')
        df_predicciones, _, d_roi = bs.calculate_roi_in_combinations(df_pred_proba, d_params=d_params)
        d_metrics.update(d_roi)
        
        if self.verbose >= 0:
            print(d_metrics)

        # Calculo otras metricas
        d_metrics.update(asses_model.determine_distribution(df_predicciones))
        d_metrics.update(asses_model.calculate_nan_metrics(df_predicciones)) # Necesita 'ROI'
        d_metrics.update(asses_model.calculate_gp_by_result(df_predicciones)) # Necesita 'ROI'
        d_metrics.update(asses_model.calculate_accuracy_by_result(df_predicciones)) # Necesita 'acerte'

        return df_predicciones, d_metrics
    
    def reformat_pred(self, df):
        # Convierto ids de equipos a nombres --> Hacerlo afuera de def assess_model...
        df = format_data.map_teams(df,df_teams=self.df_teams)
        return df

    def train_and_assess_models(self, X_val, y_val, X_train, y_train, X_test, y_test, l_modelos: list, ruta_base_mod_seg: str, cont_iter: int,  df_match:pd.DataFrame, df_match_odds: pd.DataFrame, retrain: bool = False, k: int = 5, verbose: int = 0):
        """
        Pruebo varios modelos 
        Me gusta que este en Modeling() (y no en find_best_hyper) puesto que usa build_model y asses_model.
        """
        # Defino variables
        df_metrics = pd.DataFrame()
        rows_to_features_min, min_row_test = 5, 30
        rows_test = len(X_test)
        rows_to_features = len(X_train) / len(X_train.columns)  # Idealmente mayor a 10. En caso de redes neuronales entre 30 y 100 veces mas.
        
        if verbose >= 0:
            logger.info(f"Rows X_test: {rows_test}")
            logger.info(f"Relacion rows to features: {rows_to_features:.0f}")

        # Si hay suficientes datos
        if (rows_test >= min_row_test) and (rows_to_features >= rows_to_features_min):

            # Por modelo
            for modelo in l_modelos:
                
                model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")
                print(f" Modelo: {model_name} ".center(120, '-'))

                # Entreno modelo y evaluo su rendimiento 
                try:
                    # Entreno modelo
                    model, params, cv_accuracy, results = self.build_model(modelo, X_val, y_val, X_train, y_train, k, export=False)

                    # Evaluo modelo en test
                    df_predicciones, d_metrics = self.assess_model(model, X_test, y_test, df_match, df_match_odds, retrain=retrain)

                    # Hiperparametros del modelo y Metricas en testeo y train
                    new_row = {'n_iteration': cont_iter, 'model_name': model_name, 'cv_accurracy': cv_accuracy, 'model_hiper': params, 'X_train': X_train.shape,
                                'X_val': X_val.shape, 'X_test': X_test.shape, "X_columns": list(X_train.columns)}
                    new_row.update(d_metrics)
                    df_metrics_new = pd.DataFrame([new_row])  # 1. Convertir el diccionario d_metrics en un DataFrame de una fila
                    df_metrics = pd.concat([df_metrics, df_metrics_new], ignore_index=True)  # 2. Concatenar este nuevo DataFrame con df_metrics existente

                    # Exporto datos del modelo
                    pickle.dump(model, open(f"{ruta_base_mod_seg}/{cont_iter}_{model_name}.pkl", "wb"))
                    results.to_excel(f'{ruta_base_mod_seg}/{cont_iter}__{model_name}_params.xlsx')
                    df_predicciones.to_excel(f'{ruta_base_mod_seg}/{cont_iter}__{model_name}_predicciones.xlsx', index=True)

                except KeyboardInterrupt as e:
                    logger.warning(f"Se evitó entrenar este modelo mediante {e}")
        
        else:
            if rows_to_features >= rows_to_features_min:
                logger.warning(f"EVITO TRAIN. Se evita entrenar modelo por pocas filas en X_test. {rows_test} menor a {min_row_test}. Probablemente los 'ultimos partidos' tienen mucho NaN y se estan eliminando en clean_data_2 (en la eliminacion de filas por mucho NaN) o treat_nan_values (si el fill_na=None no podes hacer nada..., en este caso el fill_na es {fill_na})")
            else:
                logger.warning(f"EVITO TRAIN. Se evita entrenar modelo por pocas filas respecto a columnas. {rows_to_features} menor a {rows_to_features_min} ")
                
        return df_metrics

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
    dp = DataPreparation(id_country, country) # Creo objeto de clase DataPreparation
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
        
        du.describe_data(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa)

    #------------------------------------------- DATA PREPARATION -------------------------------------------#
    if d_run.data_prep:
        print(" Data preparation ".center(120, "#"))

        # Preparo el dataset para el analisis
        ## Hasta integrate
        if not d_run.from_integrate:
            df_match, df_match_player, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_player_fifa_sofifa, export=False)
            df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export)
            df = dp.integrate_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, export=export) 
        else:
            df = pd.read_excel(f'./data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)
            print(df.head(2))

        ## Desde construct hasta el final
        if not d_run.until_integrate:
            df = dp.construct_data(df, l_days=d_par.l_days, n_years_h2h=d_par.n_years_h2h, segun_localia=d_par.segun_localia, dif_con_against=d_par.dif_con_against, export=export)
            # df = pd.read_excel(f'./data/{country}/p3_data_preparation/df_constructed.xlsx', index_col=0)
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
    id_country = 6
    d_run = {'data_unders': True, 'data_prep': False, 'modeling': False, 'until_integrate': False, 'from_integrate': True}

    # Hiperparametros
    d_comps = select_data.determine_country_competitions(id_country)
    modelo = LogisticRegression()  # LogisticRegression(), RandomForestClassifier()
    d_params = {
        'l_days': [30, 180], 
        'n_years_h2h': 3,
        'segun_localia': False,
        'dif_con_against': False,
        'thr_corr': 0.85,
        'thr_fs': 0.25,
        'n_years_to_select': 3, 
        'comp_to_select': d_comps['comp_solo_liga'],
        'fill_na': None,
        'val_size':  0.125,
        'test_size': 0.125,
        'bal_type': None,
        'k': 10
    }

    # df_hiper_prep = pd.DataFrame(data=d_params) # df_hiper_prep = pd.DataFrame(data={'n_dias_ult_part': [l_days], 'n_anios_hist': [n_years_h2h], 'segun_localia': [segun_localia], 'dif_con_against': [dif_con_against], 'thr_corr': [thr_corr], 'thr_fs': [thr_fs], 'fill_na': [fill_na], 'n_years_to_select': [n_years_to_select], 'comp_to_select': [comp_to_select]}, index=[0])
    # d_hiper_mod = {'val_size': [val_size], 'test_size': [test_size], 'bal_type': [bal_type], 'k': [k]}

    main(id_country, d_run, d_params, modelo, export=True)