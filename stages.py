# Importo librerias
import pandas as pd
import numpy as np
import os
import datetime
import time
from utils.set_up_logging import logger
from utils import directories
## Data understanding
from p2_data_understanding import scraper_flashscore, scraper_sofifa
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data, integrate_sofifa_to_flashscore
from sklearn.preprocessing import StandardScaler
import joblib
## Modeling
from sklearn.metrics import log_loss
from p4_modeling import generate_test_design, build_model, assess_model, betting_strategy
### Generate test design
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
        describe_data.scatter_plot(df_match, f"{self.country}/scatter_plot_df_match")

        print("\n DF_MATCH_PLAYER \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_player)

        print("\n DF_MATCH_ODDS \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_odds)
        describe_data.scatter_plot(df_match, f"{self.country}/scatter_plot_df_match_odds")

        print("\n DF_PLAYER_SOFIFA \n".center(240, "-"))
        describe_data.getting_to_know_data(df_player_sofifa)
        describe_data.verificar_unicidad_registros(df_player_sofifa)
        describe_data.scatter_plot(df_match, f"{self.country}/scatter_plot_df_player_sofifa")

        print("\n DF_PLAYER_FIFA_SOFIFA \n".center(240, "-"))
        describe_data.getting_to_know_data(df_player_fifa_sofifa)
        # describe_data.check_ids_in_both_dataframes(df_player_sofifa, df_player_fifa_sofifa, column='id_player')  # Verifico consistencia en campos que relacionan entidades
        describe_data.scatter_plot(df_match, f"{self.country}/scatter_plot_df_player_fifa_sofifa")


class DataPreparation:

    def __init__(self, id_country, country, date, var_resp: str = 'result', verbose: int = 0):
        self.id_country = id_country
        self.country = country.lower()
        self.date = date
        self.var_resp = var_resp
        self.verbose = verbose

        self.base_path = f'./data/{self.country}/p3_data_preparation/{self.date}'
        # path = f'./data/{self.country}/p3_data_preparation'        
        self.make_directories()

    def make_directories(self):

        # Creo directorios para la preparacion actual
        l_directorios = [
            f'{self.base_path}/format_data',
            f'{self.base_path}/clean_data',
            f'{self.base_path}/integrate_data',
            f'{self.base_path}/clean_post_integrate',
            f'{self.base_path}/describe_integrate_data',
            f'{self.base_path}/construct_data',
            f'{self.base_path}/clean_post_construct',
            f'{self.base_path}/tag',
            f'{self.base_path}/clean_post_select',
            f'{self.base_path}/select_data',
        ]  
        directories.make_directories(l_directorios=l_directorios)

    # Format and clean
    def format_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, reformat: bool = True, prod: bool = False, export: bool = True):
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
        if not prod:
            ## Ball posession
            df_match = format_data.convert_ball_possession_to_int(df_match)
            
            ## Goals
            df_match = format_data.convert_goals_to_int(df_match)
            df_match = format_data.format_penalties(df_match)
            df_match = clean_data.corregir_goals(df_match)

            # Goals per Half  
            # df = format_data.format_result_per_half(df)

        df_match_player = df_match_player[df_match_player.index.isin(df_match.index)]
        ## Todas las columnas
        df_match = format_data.convert_columns_to_float(df_match)  # Formateo estadisticas a float (no se por que son object)
        df_match_odds = format_data.convert_columns_to_float(df_match_odds)  # Formateo estadisticas a float (no se por que son object)

        if reformat: # Solo para missing
            # Formateo nuevas columnas...
            logger.warning("Reformateo columnas stats nuevas")
            base_columns = ['passes', 'passes_in_the_final_third', 'crosses', 'tackles']
            df_match = format_data.format_percentage_columns(df_match, base_columns)  # Uso nombres ≠ que los df_match x si tmb existen dichas columnas en missing. (eso lo tendre en cuenta en el renombre de columnas...)

            # Renombro columnas para usar el mismo nombre que en los datos viejos (por ende, se concatenen juntas) 
            rename_dict = {
                # nombre en missing: --> nombre en datos viejos
                'total_shots_home': 'goal_attempts_home', 'total_shots_away': 'goal_attempts_away',
                'shots_on_target_home': 'shots_on_goal_home', 'shots_on_target_away': 'shots_on_goal_away',
                'shots_off_target_home': 'shots_off_goal_home', 'shots_off_target_away': 'shots_off_goal_away',
                "n_passes_home" : "total_passes_home", "n_passes_away": "total_passes_away",
                "n_correct_passes_home" : "completed_passes_home", "n_correct_passes_away": "completed_passes_away",
                "accuracy_passes_home": 'pass_success_%_home', "accuracy_passes_away": 'pass_success_%_away',
                'n_clearances_home': 'clearances_completed_home', 'n_clearances_away': 'clearances_completed_away', # No hay n_correct clearances a veces? 
                'n_correct_tackles_home': 'tackles_home', 'n_correct_tackles_away': 'tackles_away',
            }
            df_match = format_data.rename_and_merge_columns(df_match, rename_dict)
                
        # Dataframe player_fifa_sofifa
        ## Fecha
        df_player_fifa_sofifa['date'] = pd.to_datetime(df_player_fifa_sofifa['date'], format='%b %d, %Y')
        ## Market value
        df_player_fifa_sofifa = format_data.convert_value_to_int(df_player_fifa_sofifa)

        # Dataframe con dtypes de cada columna
        df_match_dtype = pd.DataFrame({'Variable': df_match.columns, 'Dtype': df_match.dtypes.astype(str)})
        df_match_player_dtype = pd.DataFrame({'Variable': df_match_player.columns, 'Dtype': df_match_player.dtypes.astype(str)})
        df_match_odds_dtype = pd.DataFrame({'Variable': df_match_odds.columns, 'Dtype': df_match_odds.dtypes.astype(str)})
        # df_player_sofifa_dtype = pd.DataFrame({'Variable': df_player_fifa_sofifa.columns, 'Dtype': df_player_fifa_sofifa.dtypes.astype(str)})
        df_player_fifa_sofifa_dtype = pd.DataFrame({'Variable': df_player_fifa_sofifa.columns, 'Dtype': df_player_fifa_sofifa.dtypes.astype(str)})

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_match.to_excel(f'{self.base_path}/format_data/df_match_formated.xlsx', index=True)
            df_match_player.to_excel(f'{self.base_path}/format_data/df_match_player_formated.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'{self.base_path}/format_data/df_player_fifa_sofifa_formated.xlsx', index=True)

            # Exporto dtypes de columnas de cada df
            df_match_dtype.to_excel(f'{self.base_path}/format_data/df_match_dtype.xlsx', index=True)
            df_match_player_dtype.to_excel(f'{self.base_path}/format_data/df_match_player_dtype.xlsx', index=True)
            df_match_odds_dtype.to_excel(f'{self.base_path}/format_data/df_match_odds_dtype.xlsx', index=True)
            df_player_fifa_sofifa_dtype.to_excel(f'{self.base_path}/format_data/df_player_fifa_sofifa_dtype.xlsx', index=True)

        return df_match, df_match_player, df_match_odds, df_player_fifa_sofifa

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
        
        # Elimino columnas de jugadores que son todo NaN (se ve que hay porque las creo y no les guardo nada eso debe ser porque obtengo nombres solo si tiene url)
        non_object_columns = df_match_player.select_dtypes(exclude=['object']).columns
        df_match_player.drop(columns=non_object_columns, inplace=True)
        
        # Mensajes sobre limpieza de datos de Flashscore
        if self.verbose >= 1:
            logger.warning(f"Partidos jugados antes de {start_date} eliminados. {n_rows_inic} --> {len(df_match)}. {len(df_match_player)}")

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

        # Eliminacion de outliers...
        if export: # en prod no (pues no hay accuracy aun)
            # Filtrar columnas con "accuracy" en su nombre y reemplazar valores fuera de rango
            accuracy_cols = [col for col in df_match.columns if 'accuracy' in col]
            df_match[accuracy_cols] = df_match[accuracy_cols].apply(lambda col: col.map(lambda x: x if 0 <= x <= 100 else np.nan))
            ## (hay total passess con valor ridiculo...)
            ## (hay muchas stats que Flashscore les da valor 0 en vez de nan. Fijate en attacks y essas

        if export:
            df_match.to_excel(f'{self.base_path}/clean_data/df_match_cleaned.xlsx', index=True)
            df_match_player.to_excel(f'{self.base_path}/clean_data/df_match_player_cleaned.xlsx', index=True)
            df_player_sofifa.to_excel(f'{self.base_path}/clean_data/df_player_sofifa_cleaned.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'{self.base_path}/clean_data/df_player_fifa_sofifa_cleaned.xlsx', index=True)
    
        return df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa

    def verify_format(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, prod: bool = False):
        """
        Verificacion de formato
        """
        logger.info("\nVerifying data format...")

        # Flashscore
        df_match = format_data.format_df_match(df_match, prod=prod)
        # df_match_player = format_data.format_df_match_player(df_match_player)
        df_match_odds = format_data.format_df_match_odds(df_match_odds)

        # Sofifa
        df_player_sofifa = format_data.format_df_player_sofifa(df_player_sofifa)
        df_player_fifa_sofifa = format_data.format_df_player_fifa_sofifa(df_player_fifa_sofifa)
        return df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa

    # Integracion de fuentes de datos (Flashscore y Sofifa)
    def integrate_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, prod: bool = False, fifa_not_released_yet: bool = False, export: bool = True):
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
        df_teams = integrate_sofifa_to_flashscore.create_df_teams(df_match)
        if not prod:
            df_teams.to_excel(f"{self.base_path}/integrate_data/df_teams.xlsx", index=True)

        # PLAYERS --> MATCH (Mapeo df_player_sofifa con df_player e integro a df_match)
        print("\nIntegrating player's data to df_match...")

        # Segun si es train o produccion (en el 1ero hago el mapeo, en el 2do uso el mapeo ya hecho)
        if prod or fifa_not_released_yet:
            logger.critical("Integración para produccion. No vuelvo a mapear sino que levanto df_map del pais ")            
            df_map_players_fs_so = pd.read_excel(f'{self.base_path}/integrate_data/df_map_players_fs_so.xlsx', index_col=0)
            print(df_map_players_fs_so.head(3))

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

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")
        
        if export:
            df.to_excel(f'{self.base_path}/df_integrated.xlsx', index=True)
            df_aux.to_excel(f'{self.base_path}/integrate_data/n_players_integrated.xlsx', index=True)

        return df

    def describe_integrate_data(self, df, export: bool = True):
        """
        Describo los datos una vez integradas todas las fuentes
        """
        df_copy = df.copy()

        # Data types
        df_dtype = pd.DataFrame({'Variable': df_copy.columns, 'Dtype': df_copy.dtypes.astype(str)})

        # Nan Values
        ## COLUMNAS
        df_nan_col = df_copy.isna().mean()
        df_nan_col_sorted = df_nan_col.sort_values(ascending=False)

        ## FILAS
        ## % nan mean x fila
        nan_mean_total = df_copy.isnull().mean().mean()  # Promedio sobre todas las filas y columnas
    
        ## % nan mean x competicion
        # Calcular el porcentaje de NaN por fila
        df_copy['nan_percent_per_row'] = df_copy.isnull().sum(axis=1) / df_copy.shape[1] * 100  # En porcentaje

        # Agrupar por competición y obtener la media del porcentaje de NaN
        df_nan_by_competition = df_copy.groupby('id_competition')['nan_percent_per_row'].mean().reset_index()

        # Renombrar columnas para claridad
        df_nan_by_competition.columns = ['id_competition', 'mean_nan_percent']
        
        if self.verbose >= 2:
            logger.warning(f"Las 10 variables con más valores NaN: \n {df_nan_col_sorted.head(10)}")
            logger.info(f"Promedio de NaN values en todas las filas y columnas: {nan_mean_total}")
            print(df_nan_by_competition)
        
        if export:
            df_dtype.to_excel(f'{self.base_path}/describe_integrate_data/dtypes.xlsx', index=True)
            df_nan_col.to_excel(f'{self.base_path}/describe_integrate_data/nan_per_col.xlsx', index=True)
            df_nan_by_competition.to_excel(f'{self.base_path}/describe_integrate_data/nan_per_competition.xlsx', index=True)

    def clean_post_integrate(self, df: pd.DataFrame, n_years_to_select: int = None, competencies_to_select: list = None, prod: bool = False):
        """
        CLEAN DATA ANTES DE CONSTRUIR. Eliminacion de columnas
        """
        if not prod:
            self.describe_integrate_data(df)
        
        # Ordeno valores por fecha y separo X e y
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        df = df.sort_values(by='date', ascending=False)

        # (1) Filtrado de filas 
        n_years = None if n_years_to_select is None else n_years_to_select+1
        df = self.filter_rows(df, n_years_to_select=n_years, competencies_to_select=competencies_to_select)

        # (2) Eliminacion de columnas
        ## constantes
        if not prod:
            cols_constants = list(df.columns[df.nunique() == 1])  # Elimino columnas constantes
        else:
            cols_constants = [] # En prod se elimina is_cup pero hay modelos que la usan...
        ## ruido
        cols_basics_noise = ['attendance', 'capacity', 'referee']   # --> generan problemas de convergencia por ser nros altos y ademas su info puede ser importante junta y no separada.        
        strings_to_avoid = ['rep_player', 'hei_player', 'wage_player', 'value_player', 'pot_player']
        col_players_noise = [col for col in df.columns if any(s in col for s in strings_to_avoid)] # Elimino variables jugadores que meten ruido (lo hago aqui antes de que construya mil columnas mas...)

        # Elimino todas las cols al mismo tiempo
        cols_to_drop = list(set(cols_constants + cols_basics_noise + col_players_noise))
        cols_to_drop = [col for col in cols_to_drop if col in df.columns] # En prod falla la elim de "attendance"
        if self.verbose >= 0:
            logger.warning(f"Eliminación de {len(cols_to_drop)} de {len(df.columns)} columnas...\n Por ser contantes: {cols_constants} \n Por ruido: {cols_basics_noise + col_players_noise}")

        df.drop(columns=cols_to_drop, inplace=True)
        if self.verbose >= 0:
            logger.warning(df.shape)

        # Nan en red_cards (en prod: no lo uso para df_next_matches pero si lo uso para df old matches con el cual construir datos)
        if 'red_cards_home' in df.columns:
            df['red_cards_home'] = df['red_cards_home'].fillna(0)
            df['red_cards_away'] = df['red_cards_away'].fillna(0)
        
        # (3) Tratamiento de nan inicial (solo elimino lo que es absurdamente nan)
        if not prod:
            df = self.treat_nan_in_cols(df, porc_nan_max=0.9) # Columnas
            df.dropna(subset=['mean_rat_player_start_home', 'mean_rat_player_sub_home'], inplace=True)

        # Exporto datos
        if self.verbose >= 0 and not prod:
            df.to_excel(f'{self.base_path}/clean_post_integrate/df_cleaned.xlsx', index=True)

        return df

    def filter_rows(self, df, n_years_to_select: int, competencies_to_select: list = None):

        # (1) Eliminacion de filas
        ## Para evitar ciertas competencias (podria eliminar competencias segun nan values? competencias con mucho nan, afuera.)
        if competencies_to_select is not None:
            n_reg_inic = len(df)
            df = df[df['id_competition'].isin(competencies_to_select)]
            logger.warning(f"Eliminacion por competencias. Cantidad de filas: {n_reg_inic} --> {len(df)}")

        ## Para evitar partidos muy viejos
        if n_years_to_select is not None:
            n_reg_inic = len(df)
            fecha_limite = df.iloc[0]['date'] - datetime.timedelta(days=n_years_to_select*365)
            df = df[df['date'] >= fecha_limite] 
            logger.warning(f"Eliminacion por fecha. Cantidad de filas: {n_reg_inic} --> {len(df)}")

        return df

    # Construccion de nuevos datos a partir de los datos existentes
    def determine_stats_to_use(self):
        """
        Es necesaria para usarla desde prod.
        Mejoras: Garantizar que esten en df.columns...
        """
        self.stats_to_derive = ['offsides', 'fouls', 'tackles', 'interceptions', 'yellow_cards', 'red_cards', 
                                'clearances_total', 'blocked_shots', 'throw-ins', 'corner_kicks', 'free_kicks']
        self.stats_to_construct = [
            # (1) Ofensive
            'goals', 'points', 'expected_goals_(xg)', 'expected_points',
            'shots_on_goal', 'goal_attempts', 'PPS', 'S2G', 'xS2G', "SOG2S",
            'dead_balls', 'O2S',
            'ball_possession', 'total_passes', 'attacking_efficiency',
            # (2) Defensive ("against")
            'defensive_actions', "KGP", "x_KGP", "cards",
            'PPDA', 'clean_sheet', 'defensive_efficiency',  # "efficiency" 
            'ELO', 'expected_ELO'
        ]
        cols = self.stats_to_derive + self.stats_to_construct
        return cols

    def construct_data(self, df: pd.DataFrame, n_last_matches: list, n_years_h2h: int, segun_localia: bool = True, calculate_dif: bool = False, with_historic: bool = True, prod: bool = False, decay_rate: float = 0, prod_idxs: list = None, path_prod: str = None, export: bool = True):
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de players. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)

        Mejoras:
         - Evitar argumentos 'with_historic' y 'with_h2h'. Casi no los uso.
        """
        start = time.time()
        logger.info("Constructing data...")        
        cols_to_use = self.determine_stats_to_use()

        # Alerta si dejo de medir alguna variable en los ultimos partidos
        self.warning_variables_no_longer_measured(df, cols_to_use)

        # Si quiero construir variables historicas
        if with_historic:

            # VARIABLE RESPUESTA (no son historicas estan filtradas)
            df = construct_data.determine_result(df, self.var_resp)
            df = construct_data.determine_points(df)

            # VARIABLES DERIVADAS
            ## (1) OFENSIVE
            df = construct_data.construct_percentaje_column(df, col_num='shots_on_goal', col_den="goal_attempts", laplace=True, column_name="SOG2S") # Shots on goal --> Shots
            df = construct_data.construct_percentaje_column(df, col_num='goal_attempts', col_den="goals", laplace=True, column_name="S2G") # Tiros por gol
            df = construct_data.construct_sum_columns(df, l_columns=['throw-ins', 'corner_kicks', 'free_kicks'], column_name="dead_balls") # Dead balls
            df = construct_data.construct_percentaje_column(df, col_num='offsides', col_den="goal_attempts", laplace=True, column_name="O2S") # Concentracion

            ## (2) DEFENSIVE  --> es como tener en cuenta against ya desde la construccion misma
            ## Passess per defensive action (PPDA) --> (no es solamente en el 60% de la cancha pues no tengo ese dato)
            df = construct_data.construct_sum_columns(df, l_columns=['fouls', 'tackles', 'interceptions', 'clearances_total', 'blocked_shots'], column_name="defensive_actions") # Calculo defesive actions  # home = home + home
  
            # Tiros recibidos por gol concedido (Keep goals prevented)
            df['KGP_home'] = np.where(df['goal_attempts_away'].notna(), df['goal_attempts_away'] / (df['goals_away'] + 1), None) 
            df['KGP_away'] = np.where(df['goal_attempts_home'].notna(), df['goal_attempts_home'] / (df['goals_home'] + 1),  None)
                 
            # Disciplina (sumar falta tmb?)
            df['cards_home'] = df['yellow_cards_home'] + 5 * df['red_cards_home']
            df['cards_away'] = df['yellow_cards_away'] + 5 * df['red_cards_away']

            # Clean Sheets
            df['clean_sheet_home'] = (df['goals_away'] == 0).astype(int)
            df['clean_sheet_away'] = (df['goals_home'] == 0).astype(int)

            # Si esta la columna pases
            if 'total_passes_home' in df.columns:
                # Pases por gol
                df = construct_data.construct_percentaje_column(df, col_num='total_passes', col_den="goal_attempts", laplace=True,  column_name="PPS")  # Passes por tiro # home = home + home
                
                # Passes per defensive action
                df['PPDA_home'] = np.where(df['defensive_actions_home'].notna(),  df['total_passes_away'] / df['defensive_actions_home'], None)
                df['PPDA_away'] = np.where(df['defensive_actions_away'].notna(), df['total_passes_home'] / df['defensive_actions_away'],  None)

            # # Si esta la expected goals
            if 'expected_goals_(xg)_home' in df.columns:
                df = construct_data.determine_expected_result(df, verbose=0)
                df = construct_data.determine_points(df, suffix='expected_') 

                df = construct_data.construct_percentaje_column(df, col_num='goal_attempts', col_den="expected_goals_(xg)", laplace=True,  column_name="xS2G") # Tiros por x_gol
                df = construct_data.construct_percentaje_column(df, col_num='goals', col_den="expected_goals_(xg)", laplace=True,  column_name="attacking_efficiency") # Goles por x_goals

                df['x_KGP_home'] = np.where(df['goal_attempts_away'].notna(), df['goal_attempts_away'] / (df['expected_goals_(xg)_away'] + 1), None)
                df['x_KGP_away'] = np.where(df['goal_attempts_home'].notna(), df['goal_attempts_home'] / (df['expected_goals_(xg)_home'] + 1),  None)

                df['defensive_efficiency_home'] = np.where(df['expected_goals_(xg)_away'].notna(),  df['goals_away'] - df['expected_goals_(xg)_away'], None)
                df['defensive_efficiency_away'] = np.where(df['expected_goals_(xg)_home'].notna(), df['goals_home'] - df['expected_goals_(xg)_home'],  None)

                df = construct_data.assign_elo_before_match(df, k=30, base_rating=1500, expected=True) 
            
            # (3) GENERAL: ELO o ranking fifa --> deberia hacerlo para ≠ timelapses? No tarda nada en construirse en prod.
            df = construct_data.assign_elo_before_match(df, k=30, base_rating=1500)
            df_preconstructed = df.copy()

            # VARIABLES HISTORICAS (EN ULTIMOS N DAYS)
            df = construct_data.h2h_by_date(df, n_years=n_years_h2h, idxs_to_construct=prod_idxs) # no mas por localia por alto nan.
            
            for n_days in n_last_matches:
                df = construct_data.determine_number_matches_last_days(df, n_days=n_days, idxs_to_construct=prod_idxs)  # Lo determino aqui para no hacerlo una vez por cada stat 

            # Por stat (e.g. shots_on_goal)
            stats = [col for col in self.stats_to_construct if f'{col}_home' in df.columns]
            logger.info(f"Stats a promediar en ultimos partidos: {stats}")
            for var in stats:
                logger.info(f"Estadística a promediar: {var}")
                
                cols_to_drop = []
                variable = f'dif_{var}' if calculate_dif else var # (e.g. dif_goals o goals)

                if calculate_dif:
                     df[variable] = df[f'{var}_home'] - df[f'{var}_away']
                     cols_to_drop.append(variable)
                cols_to_drop.extend([f'{var}_home', f'{var}_away'])

                # Por numero de days
                for n_days in n_last_matches:

                    func = construct_data.determine_mean_last_matches_difference if calculate_dif else construct_data.determine_mean_last_matches_home_away

                    # Calculo promedio en ultimos partidos
                    df = func(df, n_days=n_days, variable=variable, segun_localia=False, decay_rate=decay_rate, diff=True, idxs_to_construct=prod_idxs)

                    # Si quiero calcular la diferencia por localia
                    if segun_localia:
                        df = func(df, n_days=n_days, variable=variable, segun_localia=segun_localia, decay_rate=decay_rate, diff=True, idxs_to_construct=prod_idxs)

                # Elimino variables utilizadas para construir historicas
                df.drop(columns=cols_to_drop, inplace=True)
        
        else:
            logger.warning("Evito construccion de variables historicas debido a la falta de ultimos partidos")

        # VARIABLE DE JUGADORES
        df = construct_data.calculate_dif_col_players(df)  # Construyo variables de diferencias para las variables promedio de los players

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")
        
        # Exporto datos
        sp = path_prod if prod else self.base_path
        df.to_excel(f'{sp}/df_constructed.xlsx', index=True)
        if with_historic:  # df_preconstructed solo existe si se construyeron variables historicas
            df_preconstructed.to_excel(f'{sp}/df_pre_constructed.xlsx', index=True) # Para ver como queda el df
        
        return df

    def warning_variables_no_longer_measured(self, df, cols_to_use):

        # Warning si una columna en particular tiene mayoria de nan en ultimos partidos (indica que flashscore la dejo de medir)
        recent_df = df.sort_values('date', ascending=False).head(100)
        logger.info(f"Para identificar si Flashscore dejó de medir alguna variable en el ultimo tiempo (e.g. attacks):")

        cols_to_review = [col for col in recent_df.columns if any(substring in col for substring in cols_to_use)]
        # print(cols_to_review)

        for col in cols_to_review:  # Revisar cada columna (menos la columna de fecha)
            if col == 'date':
                continue
            total = recent_df[col].shape[0]
            num_nans = recent_df[col].isna().sum()
            if num_nans > total / 2:
                logger.warning(f"⚠️ Warning: La columna '{col}' tiene {num_nans} NaN de {total} registros recientes.")

    def tag_string_data_to_integer(self, df: pd.DataFrame, df_etiquetas: pd.DataFrame = None, prod: bool = False):
        """
        Conversion de columnas tipo "object" a "integer"
        """        
        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df, df_etiquetas=df_etiquetas, verbose=self.verbose, prod=prod)

        if self.verbose >= 1:
            df_etiquetas.to_excel(f'{self.base_path}/df_etiquetas.xlsx', index=False)
            df.to_excel(f'{self.base_path}/df_constructed_etiquetado.xlsx', index=True)

        return df, df_etiquetas
    
    def clean_post_construct(self, df: pd.DataFrame, n_years_to_select: int = None, col_nan_max: float = 0.7, prod: bool = False):
        """
        Eliminacion de filas y columnas con mucho NaN y escalado de datos

        # Parameters:
            df: Dataframe a tratar NaN y escalar.

        # Returns:
            df: Dataframe pasado como parametro sin filas y columnas con mucho NaN y con datos escalados.
        """
        logger.info("\nCleaning data post construct...")

        # Reemplazo infinitos (generados en la construccion de variables porcentage)
        df = clean_data.replace_infinite(df)

        if not prod:
            # Elimino los partidos del +1 que hice en el filter de clean_data_post_integrate (usados para construir bien las var historicas)
            df = self.filter_rows(df, n_years_to_select=n_years_to_select, competencies_to_select=None)

            # (2) Eliminacion de columns 
            # Elimino cols con alto porcentaje de nan values
            df = self.treat_nan_in_cols(df, porc_nan_max=col_nan_max)
        
        # Eliminacion de columnas usadas para construir
        cols_not_constructed = df.filter(regex='(_home|_away)$').columns.tolist()  # Eliminar toda stat "..._home" y "..._away" --> para eliminar stats no construidas como "attacks_home", "dang_attacks_home", 'goalkeeper_saves', etc.
        cols_data_leakage = ['season', 'date', 'expected_result', 'penalties']
        cols_noise = ['id_team_home', 'id_team_away']  # Elimino variables que no usare en el modelo fecha (la idea es usar todas las posibles)
        
        cols_set = set(cols_data_leakage + self.stats_to_derive + cols_not_constructed + cols_noise)
        cols_to_drop = [col for col in df.columns if col in cols_set]
        df.drop(columns=cols_to_drop, inplace=True)

        # Mensaje de warning
        logger.warning(f"\nColumnas a eliminar... \n x data leakage: {cols_data_leakage} \n x ruido: {cols_noise} \n x no usarse para construir: {cols_not_constructed}")
        return df

    def treat_nan_in_cols(self, df: pd.DataFrame, porc_nan_max: float, export: bool = True):
        """
        Tratamiento de nan values

        # Parameters
            df: Dataframe a tratar nan values. (DataFrame)
            fill_na: Tipo de rellenado de NaN values.
            porc_nan_max: Porcentaje de NaN values maximo tolerado en columnas. (float)
            percentil_nan: Percentil para definir que columnas son consideradas con mucho nan y cuales con poco nan. Solo cuando haces fillna.
            export: 
            _print:

        # Returns
            Dataframe sin NaN values
        """ 
        start = time.time()
        print("\nTreating NaN values in cols...")

        # Calcula la proporción de NaN en cada columna
        X_sin_col_mucho_nan = df.copy()

        # Elimino cols
        df = clean_data.delete_columns_nan(df, porc_nan_max=porc_nan_max)
        if self.verbose >= 1:
            logger.info(f"Shape X_sin_col_mucho_nan: {X_sin_col_mucho_nan.shape} --> {df.shape} ")

        end = time.time()
        print(f"Tratamiento de NaN values en {(end - start)/60:.1f} minutos")

        if self.verbose >= 1:
            df.to_excel(f'{self.base_path}/clean_post_construct/df_treat_nan.xlsx', index=True)

        return df
    
    # Seleccion de variables
    def select_data(self, df: pd.DataFrame, thr_corr: float = None, thr_fs: float = None, export: bool = True):
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        start = time.time()
        logger.info("Selecting data...")

        # Determino variables seleccionadas con el df pero sin nan values (pues no puede recibir ni uno)
        df_aux = df.copy()
        df = df.dropna(axis=0, how='any')  # Puede construir y generar col con nan por ej cdo no hay ultimos partidos
        logger.warning(f"\tSe eliminaron registros con al menos un NaN value para seleccionar. {df_aux.shape} --> {df.shape}")

        # Elimino variables altamente correlacionadas
        if thr_corr is not None:
            l_columnas_a_eliminar, df_corr_tri_X = select_data.delete_correlated_columns(df, self.var_resp, thr_corr)
            df = df.drop(l_columnas_a_eliminar, axis=1)

            if self.verbose >= 0:
                print('\n Eliminando columnas correlacionadas...')
                print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")

            if self.verbose >= 0:
                df_corr_tri_X.to_excel(f'{self.base_path}/select_data/df_correlation.xlsx', index=True)

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            n_cols = len(df.columns)-1  # -1 por variable respuesta
            l_important_features, df_normalized = select_data.select_best_features(df=df, var_resp=self.var_resp, thr_fs=thr_fs, graf=False)
            l_col_eliminated = list(df.columns.difference(l_important_features))
            df = df.loc[:, l_important_features + [self.var_resp]]

            if self.verbose >= 0:
                print('\n Feature Selection...')
                print(f"\tSe eliminaron {n_cols-len(l_important_features)} de {n_cols} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%. Columnas eliminadas: {l_col_eliminated}")

            if self.verbose >= 0:
                df_normalized.to_excel(f'{self.base_path}/select_data/df_fs.xlsx', index=True)

        if self.verbose >= 0:
            print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")
        
        end = time.time()
        logger.info(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        # Selecciono variables seleccionadas (trataré nan values en proximo paso)
        df_aux = df_aux.loc[:, df.columns]  # Selecciono las columnas que quedaron en el df
        logger.critical(f"Shape df_aux: {df_aux.shape}")

        if self.verbose >= 1:
            df_aux.to_excel(f'{self.base_path}/df_selected.xlsx', index=True)

        return df_aux
    
    def clean_post_select(self, df: pd.DataFrame, fill_na: str = None, scaler_loaded = None, path_save: str = None, prod: bool = False):
        """
        Limpieza de datos 4 (drop nan en rows + escalado)
        """
        # Tratamiento de nan values 
        if prod:
            df = df.fillna(0) # Si o si tengo que predecir, no puede ser None.
        else:
            df = self.treat_nan_in_rows(df, fill_na=fill_na)  # Elimino registros con al menos un NaN value

        # Escalado de datos para eliminar diferencias x escala
        df = self.scale_data(df, scaler_loaded=scaler_loaded, path_save=path_save, prod=prod)  # Escalado de datos para eliminar diferencias x escala

        if self.verbose >= 1:
            df.to_excel(f'{self.base_path}/clean_post_select/df_sel_cleaned.xlsx', index=True)

        return df

    def treat_nan_in_rows(self, df, fill_na: str = None):
        # Solo de las columnas importantes

        if fill_na is not None:
            df_filled = clean_data.fill_nan_values(df, fill_type=fill_na)

        else:
            # Elimino registros con al menos un NaN 
            df_filled = df.dropna(axis=0, how='any')  # Elimino registros con al menos un NaN value

        logger.warning(f"\tSe eliminaron registros con al menos un NaN value. Shape X luego de dropna: {df.shape} --> {df_filled.shape}")
    
        return df_filled
    
    def scale_data(self, df: pd.DataFrame, scaler_loaded = None, path_save:str = None, prod: bool = False):

        print("\nEscalado de datos...")

        if not prod:
            # Separo X e y
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]  
        
            # Paso 1: Ajusta el StandardScaler a tus datos
            scaler = StandardScaler()
            scaler.fit(X) 

            # Exporto scaler entrenado (para usar en prod)
            joblib.dump((scaler, df.columns), path_save)

        else:
            X = df.copy()
            scaler = scaler_loaded

        # Paso 2: Transforma tus datos utilizando el StandardScaler ajustado
        try:
            X_scaled = scaler.transform(X) 
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
            if self.verbose >= 1:
                logger.critical("El escalado fue un exito!")
                    
        except ValueError as e: # Found array with 0 sample(s) (shape=(0, 47)) while a minimum of 1 is required by StandardScaler.
            logger.error(f"El escalado tuvo un error: {e}")
            
            raise ValueError

        if not prod:
            # Concateno X e y
            df = pd.concat([X_scaled_df, y], axis=1)
        else:
            df = X_scaled_df.copy()

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

    def generate_test_design(self, df: pd.DataFrame, bal_type: str = None, index_val: list = None, index_test_set: list = None, export: bool = True):
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

        # Separo en test, val y train
        ## Selecciono test set
        df_test = df[df.index.isin(index_test_set)]
        X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]
        df_train_val = df[~df.index.isin(X_test.index)]

        ## Selecciono val set
        df_val = df_train_val[df_train_val.index.isin(index_val)]
        X_val, y_val = df_val.drop(self.var_resp, axis=1), df_val[self.var_resp]
        df_train = df_train_val[~df_train_val.index.isin(X_val.index)]

        ## seleccion train set
        X_train, y_train = df_train.drop(self.var_resp, axis=1), df_train[self.var_resp]

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
                    bayes: bool = False, export: bool = True):
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
        print("\nTraining model...")

        if default_model == "neural_network": # A diferencia de los otros modelos, la tengo que crear                
            logger.info("Entrenando red neuronal")

            # Creo instancia de clase NeuralNetwork()
            red = build_model.TrainNeuralNetwork()

            # Seleccion mejor arquitectura con la validacion y entreno el modelo
            model_best_params, params, train_accuracy, results = red.select_best_arquitecture(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            pickle.dump(model_best_params, open(f"{self.base_path}/modelo.pkl", "wb"))
            return model_best_params, params, train_accuracy, results

        # Train model searching for best hiper
        if params is None:
            model_best_params, params, d_metrics, results = build_model.select_best_hiperparameters(
                default_model, 
                X_train=X_train, 
                y_train=y_train, 
                X_val=X_val, 
                y_val=y_val, 
                k=k, 
                bayes=bayes, 
                verbose=1
                )

        # Training metrics (≠ a las de cross validation que son en el validation set)
        _, d_metrics_train = self.assess_model(model=model_best_params, X_test=X_train, y_test=y_train, suffix="_train")
        print("Train Metrics:", d_metrics_train)
        d_metrics.update(d_metrics_train)

        if export:
            pickle.dump(model_best_params, open(f"{self.base_path}/modelo.pkl", "wb"))
            # results.to_excel(f"./data/{self.country}/{ite_date}/p4_modeling/models/hiperparametros.xlsx")    # Exportar metricas por cada combinacion de hiperparametros (En vez de retornar best_metric.)

        return model_best_params, params, d_metrics, results

    def predict_model(self, model, X_test: pd.DataFrame):
        """
        Predigo sobre X_test
        """
        # Predecir las etiquetas para los datos de prueba
        try:
            y_pred_prob = model.predict_proba(X_test) # Te da las probabilidad de cada clase. Funciona para todos los modelos? # AttributeError: predict_proba is not available when probability=False
            
            if self.verbose >= 1:
                class_distribution = y_pred_prob.mean(axis=0)
                print("Distribución promedio de probabilidades por clase:", class_distribution)
                
        except AttributeError: # AttributeError: 'Sequential' object has no attribute 'predict_proba'
            # y_pred_prob = model.predict(X_test)
            logger.error("Se necesitan las probabilidades de cada clase.")
            raise ValueError

        y_pred = self.map_classes_test_and_pred(model, y_pred_prob)
        return y_pred_prob, y_pred

    def map_classes_test_and_pred(self, model, y_pred_prob):
        """
        Mapeo clases entre test y pred (por si pred genera clases con ≠ valor).
        """
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
        return y_pred
    
    def construct_predictions_dataframe(self, model, X_test, y_pred_prob, y_pred, y_test = None):
        # Crear DataFrame con las probabilidades
        df_pred_proba = pd.DataFrame(
            y_pred_prob,
            columns=[f'prob_class_{cls}' for cls in model.classes_],
            index=X_test.index
        )

        # Agregar las columnas de la variable real (y_test) y la predicción (y_pred)
        if y_test is not None:
            # df_pred_proba[self.var_resp] = y_test
            df_pred_proba[self.var_resp] = pd.Series(y_test, index=X_test.index) # Asegurar que y_test e y_pred tengan el mismo índice que X_test

        df_pred_proba[self.var_pred] = pd.Series(y_pred, index=X_test.index)
        # df_pred_proba[self.var_pred] = y_pred
        return df_pred_proba
    
    def assess_model(self, model, X_test, y_test, suffix: str = None):
        """
        Evalúa un modelo de machine learning utilizando datos de prueba y calcula métricas (basicas) de desempeño.

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
        y_pred_prob, y_pred = self.predict_model(model, X_test)

        # Construo df_pred_proba
        df_pred_proba = self.construct_predictions_dataframe(model=model, X_test=X_test, y_pred_prob=y_pred_prob, y_pred=y_pred, y_test=y_test)

        # Calculo metricas
        d_metrics = assess_model.calculate_metrics(df_pred_proba, suffix=suffix, bet_metrics=False, gp_result=False) 
        return df_pred_proba, d_metrics

    def assess_model_with_roi(self, df_pred_proba, df_match, df_match_odds, expected_metrics: bool = False):
        """
        Aplico estrategia de apuesta y calculo ROI
        """
        # Concateno dfs
        df_predicciones = self.prepare_dataframe_to_assess_with_roi(df_pred_proba, df_match, df_match_odds)

        # Aplico estrategia "sin ea" para tener bank, stakes y rois 
        bs = betting_strategy.BettingStrategy()
        df_predicciones = bs.apply_strategy(df_predicciones, bs.define_hiperparameters(strategy='train'), prod=False)

        # G/P x rdo
        df_predicciones, d_metrics_roi = bs.calculate_roi_in_combination(df_predicciones)
        d_metrics_roi.update(assess_model.calculate_gp_by_result(df_predicciones))

        # Calculo metricas de bet (en assess no tengo cuotas)
        d_metrics_bm = assess_model.calculate_bookie_metrics(df_predicciones)
        d_metrics_roi.update(d_metrics_bm)

        # Calculo metricas "Expected" --> necesita df_match por expected_goals 
        if expected_metrics:
            # Construyo 'expected_result'
            df_predicciones = construct_data.determine_expected_result(df_predicciones, verbose=0)  # Durante la prep la elimino x fuga de info.

            # Eliminar partidos sin expected_goals (puede no estar)
            df_predicciones_ex = df_predicciones.dropna(subset=['expected_result']) 
            
            # Calculo metricas
            d_metric_sin_ea_ex = assess_model.calculate_metrics(df_predicciones_ex, var_resp='expected_result', prefix='expected_', bet_metrics=False, gp_result=False)
            d_metrics_roi.update(d_metric_sin_ea_ex)

        return df_predicciones, d_metrics_roi

    def prepare_dataframe_to_assess_with_roi( 
            self,
            df_pred_proba: pd.DataFrame,
            df_match: pd.DataFrame,
            df_match_odds: pd.DataFrame
            ):      
        # Selecciono los registros de df_pred_proba
        df_match = df_match[df_match.index.isin(df_pred_proba.index)]
        df_match_odds = df_match_odds[df_match_odds.index.isin(df_pred_proba.index)]

        # Selecciono columnas de df_match
        l_cols_match = [col for col in ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition', 'country', 'competition', 'goals_home', 'goals_away',  'expected_goals_(xg)_home', 'expected_goals_(xg)_away'] if col in df_match.columns]
        df_match = df_match[l_cols_match]

        # Calculo 'bookmaker_result" y sus probas
        df_match_odds = assess_model.calculate_result_probabilities_by_bookmaker(df_match_odds=df_match_odds)
        df_match_odds = assess_model.determine_result_by_bookmaker(df=df_match_odds, col_name='bookmaker_result')

        # Concatenar todos alineados por index
        df_predicciones = pd.concat([
            df_match,
            df_match_odds,
            df_pred_proba,
        ], axis=1)

        # Reformateo id_teams de id a nombre
        df_predicciones = self.reformat_pred(df_predicciones)
        
        return df_predicciones
    
    def reformat_pred(self, df):
        # Convierto ids de equipos a nombres --> Hacerlo afuera de def assess_model...
        df_teams = pd.read_excel(f'{self.base_path_dp}/integrate_data/df_teams.xlsx', index_col=0)
        df = format_data.map_teams(df,df_teams=df_teams)
        return df

    def train_and_assess_models(self, X_val, y_val, X_train, y_train, X_test, y_test, l_modelos: list, ruta_base_mod_seg: str, cont_iter: int,  df_match:pd.DataFrame, df_match_odds: pd.DataFrame, k: int = 5, verbose: int = 0):
        """
        Pruebo varios modelos 
        Me gusta que este en Modeling() (y no en find_best_hyper) puesto que usa build_model y assess_model.
        """
        # Defino variables
        train_rows, test_rows = [], []
        rows_to_features_min = 5
        rows_to_features = len(X_train) / len(X_train.columns)  # Idealmente mayor a 10. En caso de redes neuronales entre 30 y 100 veces mas.
        
        if verbose >= 0:
            logger.info(f"Rows X_test: {len(X_test)}")
            logger.info(f"Relacion rows to features: {rows_to_features:.0f}")

        # Si hay suficientes datos
        if rows_to_features >= rows_to_features_min:

            # Por modelo
            for modelo in l_modelos:
                
                model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")
                print(f" Modelo: {model_name} ".center(120, '-'))

                # Entreno modelo y evaluo su rendimiento 
                try:
                    # Entreno modelo
                    model, params, d_metrics_train, results = self.build_model(modelo, X_val, y_val, X_train, y_train, k, export=False)
                    
                    # Evaluo modelo en test
                    df_pred_proba, d_metrics_test = self.assess_model(model, X_test, y_test)
                    df_predicciones, d_metrics_roi = self.assess_model_with_roi(df_pred_proba, df_match, df_match_odds, expected_metrics=True)
                    d_metrics_test.update(d_metrics_roi)

                    # Evaluar overfitting
                    f1_score_train = d_metrics_train['f1_score_train']
                    f1_score_test = d_metrics_test['f1_score']
                    var = (f1_score_test - f1_score_train) / f1_score_train # (50% - 60%)/60% = -16%
                    d_metrics_test['var_f1_score'] = var
                    logger.info(f'f1_score_train = {f1_score_train:.1f}% --> f1_score_test = {f1_score_test:.1f}% ({var*100:.0f})%')
                    if var <= -0.10:
                        logger.warning(f"Posible OVERFITTING. Caida del f1 score de {var*100:.0f}%")

                    # Hiperparametros del modelo y Metricas en testeo y train
                    new_row = {'n_iteration': cont_iter, 'model_name': model_name, 'model_hiper': params, 'X_train': X_train.shape, 
                               'X_val': X_val.shape, 'X_test': X_test.shape, "X_columns": list(X_train.columns),
                               **d_metrics_train}
                    new_row_test = {'n_iteration': cont_iter, 'model_name': model_name, **d_metrics_test}
                    train_rows.append(new_row)
                    test_rows.append(new_row_test)
                    
                    # Exporto datos del modelo
                    pickle.dump(model, open(f"{ruta_base_mod_seg}/{cont_iter}_{model_name}.pkl", "wb"))
                    results.to_excel(f'{ruta_base_mod_seg}/{cont_iter}__{model_name}_params.xlsx')
                    df_predicciones.to_excel(f'{ruta_base_mod_seg}/{cont_iter}__{model_name}_predicciones.xlsx', index=True)

                except KeyboardInterrupt as e:
                    logger.warning(f"Entrenamiento interrumpido: {e}")

                except Exception as e:
                    logger.error(f"Error inesperado al entrenar el modelo {model_name}: {e}", exc_info=True)  
        else:
            logger.warning(f"EVITO TRAIN. Se evita entrenar modelo por pocas filas respecto a columnas. {rows_to_features} menor a {rows_to_features_min} ")

        return train_rows, test_rows


# NOTA: este modulo es una BIBLIOTECA de clases (DataUnderstanding, DataPreparation,
# Modeling). La orquestacion real vive en:
#   - main_train_models.py                     -> entrenamiento
#   - p6_deployment/main_next_matches.py       -> prediccion de proximos partidos
# El antiguo main() / __main__ de este archivo estaba desactualizado (firmas que ya
# no existen) y se removio en el refactor. Ver docs/REFACTOR.md.
