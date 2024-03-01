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
                         f'{ruta_base}/per_season/df_player_fifa_sofifa_sofifa/',

                         f'{ruta_base}/per_competition/df_match/',
                         f'{ruta_base}/per_competition/df_match_player/',
                        f'{ruta_base}/per_competition/df_match_odds/',
                         f'{ruta_base}/per_competition/df_player/',
                        f'{ruta_base}/per_competition/df_teams/',
                        f'{ruta_base}/per_competition/df_coaches/',
                        f'{ruta_base}/per_competition/df_player_sofifa/',
                        f'{ruta_base}/per_competition/df_player_fifa_sofifa_sofifa/'
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
            # print(df_match_concat.shape, df_match_player_concat.shape, df_match_odds_concat.shape, df_player_concat.shape)
            l_competition_already_extracted = df_match_concat['id_competition'].unique()
        except:
            df_match_concat, df_match_player_concat, df_player_sofifa_concat, df_match_odds_concat, df_player_fifa_sofifa_sofifa_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
            df_match, df_match_player, df_match_odds, df_teams, df_coaches, df_player = scraper_flashscore.extract_data( self.id_country, self.country, row['id_competition'], row['competition_flashscore'], row['is_cup'], n_seasons_max=16, export=export)

            # Guardo datos de competition
            df_match_concat = pd.concat([df_match_concat, df_match], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player], axis=0)
            df_match_odds_concat = pd.concat([df_match_odds_concat, df_match_odds], axis=0)
            df_teams_concat = pd.concat([df_teams_concat, df_teams], axis=0)
            df_coaches_concat = pd.concat([df_coaches_concat, df_coaches], axis=0)
            df_player_concat = pd.concat([df_player_concat, df_player], axis=0)

            # Si la competition es una liga
            if row['is_cup'] == 0:

                # Extraigo datos de players de Sofifa (df_player)
                df_player_sofifa, df_player_fifa_sofifa_sofifa = scraper_sofifa.extract_players(self.id_country, self.country, row['id_competition'], row['competition_sofifa'], export=export)

                # Save data
                df_player_sofifa_concat = pd.concat([df_player_sofifa_concat, df_player_sofifa], axis=0)
                df_player_fifa_sofifa_sofifa_concat = pd.concat([df_player_fifa_sofifa_sofifa_concat, df_player_fifa_sofifa_sofifa], axis=0)
            
            # Exporto por seguridad
            if export:
                df_match_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_match.xlsx', index=True)
                df_match_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_match_player.xlsx', index=True)
                df_match_odds_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_match_odds.xlsx', index=True)
                df_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_player.xlsx', index=True)
                df_teams_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_teams.xlsx', index=True)
                df_coaches_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_coaches.xlsx', index=True)
                df_player_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_player_sofifa.xlsx', index=True)
                df_player_fifa_sofifa_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/data_seg/df_player_fifa_sofifa_sofifa.xlsx', index=True)

        # Exporto datasets con competiciones del country
        if export:
            df_match_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match_player.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match_odds.xlsx', index=True)
            df_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_player.xlsx', index=True)
            df_teams_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_teams.xlsx', index=True)
            df_coaches_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_coaches.xlsx', index=True)
            df_player_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_player_fifa_sofifa.xlsx', index=True)
            df_player_fifa_sofifa_sofifa_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_player_fifa_sofifa.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat, df_player_concat, df_coaches_concat, df_player_sofifa_concat, df_player_fifa_sofifa_sofifa_concat

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
        directorio = f'./p3_data_preparation/data/{self.country.lower()}'

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
        
        # Dataframe player temp
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

    def clean_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame, df_teams: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, export: bool = True):
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

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df_match, df_etiquetas = format_data.convert_columns_to_int(df_match)  # puedo hacerlo solo a df_match porque los otros no tiene columnas object. # aun no limpie las columnas object... lo tengo que hacer post clean?

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

        # Verificar que no haya outliers
        # algo (sacar de mi tesis)

        end = time.time()
        print(f"Clean data in {(end - start) / 60:.1f} minutes")

        if export:
            df_match.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_form_clean.xlsx', index=True)
            df_match_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_player_form_clean.xlsx', index=True)
            df_etiquetas.to_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx', index=False)
            df_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_player_form_clean.xlsx', index=True)
            df_player_sofifa.to_excel(f'./p3_data_preparation/data/{self.country}/df_player_sofifa_form_clean.xlsx', index=True)
            df_player_fifa_sofifa.to_excel(f'./p3_data_preparation/data/{self.country}/df_player_fifa_sofifa_form_clean.xlsx', index=True)

        return df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa

    def integrate_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame, df_teams: pd.DataFrame, df_player_sofifa: pd.DataFrame, df_player_fifa_sofifa: pd.DataFrame, export: bool = True):
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

        # Mapeo df_teams_sofifa con df_teams e integro a df_match
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')
        df_teams_sofifa = pd.read_excel('./p2_data_understanding/data/df_teams_sofifa.xlsx')
        df_map_teams_name_id = match_teams_by_name(df_teams, df_teams_sofifa, thr_coincidence_min=90)
        # df_map_teams_name_id.to_excel('/Users/nachomondino/Desktop/df_map_teams_name_id.xlsx', index=True)
        # Integro todo a df_match
        df_match = integrate_team_data_in_match(df_match, df_etiquetas, df_teams_sofifa, df_map_teams_name_id)
        # df_match.to_excel('/Users/nachomondino/Desktop/df_match_integrated_with_df_teams.xlsx', index=True)

        # Mapeo df_player_sofifa con df_player e integro a df_match
        # Mapeo jugadores por nombre
        df_map_players_name_id = match_dataframes_by_str_column(df_player, df_player_sofifa, column_to_relation="player_name", column_to_integrate='id_player', thr_coincidence_min=90)
        # df_map_players_name_id.to_excel('/Users/nachomondino/Desktop/df_map_players_name_id.xlsx')

        # Reemplazo nombre de jugadores por id en df_match_player
        df_match_player = replace_players_with_sofifa_id(df_match_player, df_map_players_name_id)
        # df_match_player.to_excel('/Users/nachomondino/Desktop/df_match_player.xlsx')

        # Sintetizar la data de df_player (Sofifa) en df_match (Flashscore) gracias al vinculo con df_match_player (Flashscore) -->   Aca dentro hago esto:  # Traer fecha, equipo y no se que mas de df_match (Flashscore) y agregar a df_match_player (Flashscore) para poder saber en que momento traer la info del player (Sofifa tiene varias veces un mismo player porque es el player en ≠ fifas)
        df = integrate_player_data_in_match(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa)
        
        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")
        
        if export:
            # df_map_teams_name_id.to_excel(f"./p3_data_preparation/data/{self.country}/df_map_teams_name_id.xlsx")
            df_map_players_name_id.to_excel(f"./p3_data_preparation/data/{self.country}/df_map_players_name_id.xlsx")
            df_match_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_player_with_id.xlsx', index=True)
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_integrated.xlsx', index=True)
        
        return df

    def construct_data(self, df: pd.DataFrame, n_days: int, n_years_h2h: int , export: bool = True):
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

        # Construyo variables porcentajes (funciona ok!) No tira error de division ni nada. Es nan solo cuando es 0/0 (sin tiirar error).
        df = construct_data.construct_percentaje_column(df, col_num="dangerous_attacks", col_den="attacks")
        df = construct_data.construct_percentaje_column(df, col_num="shots_on_goal", col_den="goal_attempts")
        df = construct_data.construct_percentaje_column(df, col_num="goals", col_den="goal_attempts")

        # Variables historicas
        df = construct_data.h2h_by_date(df, n_years=n_years_h2h)

        # Determino cuales son las variables stats automaticamente
        l_stats = construct_data.determine_stats_columns(df)
        print(f"Stats a promediar en ultimos partidos: {l_stats}")

        # Por estadistica del partido
        for var in l_stats:
            print(f"\tEstadistica a promediar: {var}", df[f"{var}_home"].dtype, df[f"{var}_away"].dtype)

            # Determine la diferencia de la estadistica entre equipo local y visitante de cada partido
            df[f'dif_{var}'] = df[f'{var}_home'] - df[f'{var}_away']  # (e.g. dif_goles = goles_home - goles_away)
            df = df.drop([f'{var}_home', f'{var}_away'], axis=1)  # (e.g. borro goles_home y goles_away)

            # Determine para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
            df = construct_data.determine_mean_in_last_matches(df, n_days=n_days, variable=f'dif_{var}', tipo='mean')  # mean_last_match_dif_points_home
            df = df.drop([f'dif_{var}'], axis=1)

            # Determine la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            df[f'dif_mean_last_match_dif_{var}'] = df[f'mean_last_match_dif_{var}_home'] - df[f'mean_last_match_dif_{var}_away']  # KeyError: 'mean_last_match_dif_points_home'
            df = df.drop(columns=[f'mean_last_match_dif_{var}_home', f'mean_last_match_dif_{var}_away'], axis=1)
        
        # Construyo variables de diferencias para las variables promedio de los players
        df = clean_data.replace_nan_with_zero(df, 'n_player_miss_home', 'n_player_miss_away')  # Funciona perfecto
        df = construct_data.suma_rat_player_missing(df)  # Funciona perfecto
        df = construct_data.calculate_dif_col_players(df)
    
        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_constructed.xlsx', index=True)

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
        print("\nSelecting data...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df = df.drop(['date'], axis=1)  
        # Elimino columnas con 100% de nan values (puede que construyas y queden con todo nan...)
        df = clean_data.delete_columns_nan(df, porc_nan_max=0.99)

        # # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        # df, df_etiquetas = format_data.convert_columns_to_int(df)

        # Elimino variables altamente correlacionadas
        if thr_corr is not None:
            l_columnas_a_eliminar = select_data.delete_correlated_columns(df, self.var_resp, thr_corr)
            df = df.drop(l_columnas_a_eliminar, axis=1)
            print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            n_cols = len(df.columns)-1  # -1 por variable respuesta
            l_important_features = select_data.select_best_features(df, self.var_resp, thr_fs, graf=export)
            df = df.loc[:, l_important_features + [self.var_resp]]
            print(f"\tSe eliminaron {n_cols-len(l_important_features)} de {n_cols} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%. Columnas importantes: {l_important_features}")

        print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")
        
        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            # df_etiquetas.to_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx', index=True)
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_selected.xlsx', index=True)
        return df

    def nan_values_treatment(self, df , fill_na, percentil_nan: int = 75, export: bool = True):
        """
        Tratamiento de nan values
        """        
        if fill_na is None:
            print("\n Dropping rows with NaN values...")
            # Elimino filas con al menos un NaN puesto que al modelo no le pueden ingresar NaN values (solo en variables selected)
            df = clean_data.drop_columns_until_drop_nan_not_empty(df, n_reg_min=100) # elimina las columnas hasta que pueda hacer dropna() # CAMBIAR NOMBRE DE FUNCION
            df = clean_data.delete_rows_nan(df, porc_nan_max=0, _print=True)  # df = df.dropna()

        # Si relleno NaN values
        else:
            # Separo en X e y ?
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

            # Determino las columns con mucho NaN (mas de nan_threshold%)
            l_columns_con_poco_nan, l_columns_con_mucho_nan = clean_data.determine_columns_to_fill(X, percentil_nan=percentil_nan)

            # Elimino registros NaN en las columns con bajo % de NaN 
            X = X.dropna(subset=l_columns_con_poco_nan)
            X = clean_data.delete_columns_nan(X, porc_nan_max=0.99)  # Elimino columnas que quedan nan tras el dropna anterior. Esto evita error al rellenar una columna vacia.
            l_columnas_to_fill = X.drop(l_columns_con_poco_nan, axis=1).columns
            print(f"\t Eliminacion de filas con nan en columnas con menos % de nan: {X.shape}")

            # Determino que filas relleno y cuales no (antes de fill porque despues de rellenar no puedo diferenciar que filas rellene y cuales no)
            df_rellenado = pd.DataFrame(index=X.index)
            df_rellenado['rellenado'] = X[l_columnas_to_fill].isnull().any(axis=1)
            if export:
                df_rellenado.to_excel(f'./p3_data_preparation/data/{self.country}/df_rellenado.xlsx', index=True)

            # Relleno nan en X
            X = clean_data.fill_nan_values(X, l_columnas_to_fill, fill_type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las accuracyes casi siempre seran mayores que dropna() en train y test, lo que cuenta es la accuracy en next_matches o en un dataset que no haya sido filleado...
            print(f"\tSe realizó el rellenado de NaN values. Shape X luego de rellenado: {X.shape}")

            # Agrego columna reellenado a X (post fill puesto que no quiero limpiar la columna "rellenado")
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
        directorio = f'./p4_modeling/data/{self.country.lower()}'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)
        
    def generate_test_design(self, df: pd.DataFrame, bal_type, test_val_size: float = 0.2, test_size: float = 0.5, fill_na = None, with_pca: bool = False, export: bool = True):
        """
        Separa conjuntos de datos en train, validacion y test, balancea las clases del dataset y elimina los NaN values.

        :param bal_type: Tipo de balanceo de clases a realizar. (string)
        :param test_val_size: Porcentaje del total de datos destinado a validacion y test. (float)
        :param test_size: # Porcentaje de test_val_size destinado a test. (float)
        :param treat_nan: Tipo de tratamiento de NaN values. (string)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe de entrenamiento y de testeo balanceados (DataFrame)
        """
        print("\n Separating data in train, val and test...")
        # Separo conjunto de datos en train, validation y test
        if fill_na is None:     

            # Separo en X e y
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

            # Separo train y val_test
            X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_val_size, random_state=randint(1, 1000), shuffle=True)
            
            # Separo val y test
            X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=randint(1, 1000), shuffle=True)
       
        else:
            # Obtengo indice de filas rellenadas y no rellenadas
            index_no_rellenado = df[~df['rellenado']].index  # index_rellenado = df[df['rellenado']].index
            df = df.drop('rellenado', axis=1)
        
            # Todos los registros con al menos un NaN value los guardo en el conjunto de entrenamiento
            n = int(len(df) * test_size * test_val_size)
            max_possible_n = len(index_no_rellenado)

            # Verifica si hay suficientes filas no rellenadas disponibles
            if n > max_possible_n:
                # Actualizo test_val_size para que val no se lleve mas registros de lo especificado
                test_val_size_orig = test_val_size
                test_val_size = n / (len(df)-max_possible_n)
                print(f'test_val_size: {test_val_size_orig} --> {test_val_size}')

                # Ajusta n para tomar todas las filas no rellenadas disponibles
                print(f"Tamaño que deberia tener df_test: {n} pero hay solo {max_possible_n} registros disponibles.")
                n = max_possible_n

            # Construyo el dataset de prueba a partir de registros que no han sido rellenados
            df_test = df.loc[index_no_rellenado].sample(n, random_state=42) # df_test = df[~df_rellenado['rellenado']].sample(n, random_state=42)
            X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]
            # print(df_test.shape)

            # Construyo train y val a partir de las filas que quedan
            df_train_val = df[~df.index.isin(df_test.index)]
            X_train_val, y_train_val = df_train_val.drop(self.var_resp, axis=1), df_train_val[self.var_resp]
            # print(df_train_val.shape)

            # Separo train y val
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_val_size, random_state=randint(1, 1000), shuffle=True)

        # Implemento PCA
        if with_pca:
            print("Implementando PCA()...")

            # Selecciono los mejores hiperparametros
            pca = build_model.select_best_hiperparameters(PCA(), X_val, y_val, k=10)

            # Si conviene implementar PCA
            n_comp_opt = pca.get_params()['n_components']
            if n_comp_opt != None:

                # Entreno el modelo
                pca.fit(X_train)

                # Transformo X
                X_train = pd.DataFrame(pca.transform(X_train))
                X_val = pd.DataFrame(pca.transform(X_val))
                X_test = pd.DataFrame(pca.transform(X_test))
            else:
                print("No hago PCA() porque gano n_components=None")

        # Balanceo el dataset de entrenamiento
        if bal_type is not None:
            X_train, y_train = generate_test_design.balance_dataset(X_train, y_train, bal_type=bal_type)
            print(f"Shape X_train luego de balanceo: {X_train.shape}")

        print(f'Train: {X_train.shape} {y_train.shape}', f'\nVal: {X_val.shape} {y_val.shape}', f'\nTest: {X_test.shape} {y_test.shape}')

        if export:
            X_train.to_excel(f'./p4_modeling/data/{self.country}/X_train.xlsx', index=True)
            X_val.to_excel(f'./p4_modeling/data/{self.country}/X_val.xlsx', index=True)
            X_test.to_excel(f'./p4_modeling/data/{self.country}/X_test.xlsx', index=True)
            y_train.to_excel(f'./p4_modeling/data/{self.country}/y_train.xlsx', index=True)
            y_val.to_excel(f'./p4_modeling/data/{self.country}/y_val.xlsx', index=True)
            y_test.to_excel(f'./p4_modeling/data/{self.country}/y_test.xlsx', index=True)

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
        print("Training model...")
        
        # Find best hiperparameters
        if params is None:
            model_best_params = build_model.select_best_hiperparameters(model, X_val, y_val, k=10, _print=True)
        else:
            model_best_params = model.set_params(**params)
        print("Hiperparametros:", model_best_params.get_params())

        # Fit model
        model_best_params.fit(X_train, y_train)

        # Evaluo el modelo con Cross Validation
        cv_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
        print(f"\nAccuracy promedio de validación cruzada: {cv_accuracy:.1f}%")

        if export:
            pickle.dump(model_best_params, open(f"./p4_modeling/data/{self.country}/modelo.pkl", "wb"))
            df_hiperparametros = pd.DataFrame.from_dict(model_best_params.get_params(), orient='index', columns=['Valor'])
            df_hiperparametros.to_csv(f"./p4_modeling/data/{self.country}/hiperparametros.csv")

        return model_best_params, cv_accuracy

    def assess_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, export: bool = True):
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
        # Levanto df_etiquetas
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')
        df_etiquetas_y = df_etiquetas[df_etiquetas['variable'] == self.var_resp]  # solo etiquetas de la var resp

        # Levanto df_match_odds (solo los partidos en X_test)
        df_match_odds = pd.read_excel(f'./p2_data_understanding/data/{self.country}/df_match_odds.xlsx', index_col=0)
        df_match_odds = df_match_odds[df_match_odds.index.isin(X_test.index)]  # Selecciono los partidos que estan en df_test
        df_match_odds = df_match_odds.reindex(X_test.index)  # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test
        self.var_pred_bm = 'bookmaker_result'  

        # Predecir las etiquetas para los datos de prueba
        y_pred_prob = model.predict_proba(X_test) # Te da las probabilidad de cada clase. Funciona para todos los modelos? # AttributeError: predict_proba is not available when probability=False
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad  # y_pred = model.predict(X_test)  # es un numpy array
        nombres_clases = df_etiquetas_y.set_index('int_value').reindex(model.classes_ )['str_value']  # Reordeno segun el orden de las clases en y_pred_prob
        df_pred_proba = pd.DataFrame(y_pred_prob, columns=nombres_clases, index=X_test.index)
        df_pred = pd.DataFrame({self.var_resp: y_test, self.var_pred: y_pred}, index=X_test.index)

        # Calculo metricas
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100

        # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
        df_conf_mat = asses_model.confusion_matrix(y_test, y_pred, df_etiquetas_y)

        # Agrego predicciones de bookmaker
        df_match_odds = asses_model.calculate_probas_bookmarker(df_match_odds) # Caculo probabilidades segun casa de apuesta
        df_match_odds = asses_model.determine_bookmaker_result(df_match_odds, self.var_pred_bm)  # Determino resultado predicho segun cuota minima (e.g. "Home")
        df_match_odds = format_data.convert_pred_str_to_int(df_match_odds, name_var_str=self.var_pred_bm, name_var_int='y_pred_bm', df_etiquetas_y= df_etiquetas_y) # Agrego columna con prediccion numerica (e.g. "Home" --> 2)
        y_pred_bm = df_match_odds['y_pred_bm'].values

        # Calculo precision de casa de apuesta
        test_precision_bookmaker = accuracy_score(y_test, y_pred_bm) * 100  # Calcula bien tras el reindex()
        dif_prec = test_accuracy - test_precision_bookmaker
        d_metrics = {'test_accuracy': test_accuracy, 'recall': recall, 'f1_score': f1, 'test_accuracy_bm': test_precision_bookmaker, 'dif_prec_bm': dif_prec}

        # Concateno dfs
        df_concat = pd.concat([df_pred, df_pred_proba, df_match_odds], axis=1)
       
        # Convierto de int a str 1) resultado real y 2) predicciones del modelo 
        df_concat = format_data.convert_pred_int_to_str(df_concat, name_var_int=self.var_resp, name_var_str=f'{self.var_resp}_str', df_etiquetas_y=df_etiquetas_y)
        df_concat = format_data.convert_pred_int_to_str(df_concat, name_var_int=self.var_pred, name_var_str=f'{self.var_pred}_str', df_etiquetas_y=df_etiquetas_y)

        # Calculoo ROI
        d_roi = asses_model.calculate_roi_by_betting_strategy(df_concat)
        d_metrics.update(d_roi)
        print(d_metrics)

        if export:
            df_conf_mat.to_excel(f'./p4_modeling/data/{self.country}/df_conf_matrix.xlsx')
            df_concat.to_excel('/Users/nachomondino/Desktop/df_prueba.xlsx')

        return d_metrics
    
    def select_best_model(self, l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, export=True):
        """
        Pruebo varios modelos 
        Me gusta que este en Modeling() (y no en find_best_hyper) puesto que usa build_model y asses_model.
        """
        # Definicion de variables
        d = {}
        prec_test_max = 0
        
        # Por modelo
        for modelo in l_modelos:
            
            model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")
            print(f" Modelo: {model_name} ".center(120, '-'))

            # Entreno modelo y evaluo su rendimiento     
            try:
                model_best_params, cv_accuracy = self.build_model(modelo, X_val, y_val, X_train, y_train, k, export=False)
                d_metrics = self.assess_model(model_best_params, X_test, y_test)

                # Si la precision_test_es mayor, guardar datos...
                if d_metrics['test_accuracy'] > prec_test_max:
                    prec_test_max = d_metrics['test_accuracy']
                    d = {'model_name': modelo, 'model_trained': model_best_params, 'train_cv_accuracy': cv_accuracy}
                    d.update(d_metrics)

            except KeyboardInterrupt:
                print("Se evitó entrenar este modelo")
            
        if export:
            pickle.dump(d['model_trained'], open(f"./p4_modeling/data/{self.country}/modelo.pkl", "wb"))
        return d

def main():
    """
    Extraction, processing and analysis of matches to predict match results.
    """
    # Definicion de variables
    var_resp, var_pred = 'result', 'predicted_result'
    data_unders, data_prep, modeling = True, False, False
    export = True
    
    # Hiperparametros
    fill_na = 'ml'  # Relleno de nan values [None, mode, ml]

    # Selecciono country a extraer por terminal
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    country = 'England'  # country = str(input("Choose country to extract (e.g. England, Germany, etc): "))
    id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]

    # Creo instancias de clases
    du = DataUnderstanding(id_country, country) # Creo objeto de clase DataPreparation
    dp = DataPreparation(country) # Creo objeto de clase DataPreparation
    mo = Modeling(var_resp, var_pred, country)  # Creo objeto de clase Modeling

    # DATA UNDERSTANDING
    if data_unders:
        print(" Data understanding ".center(120, "#"))
        # Extriago datos o los levanto
        df_match, df_match_player, df_match_odds, df_player, df_teams, df_coaches, df_player_sofifa, df_player_fifa_sofifa = du.collect_initial_data(export=export)

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

        du.describe_data(df_match, df_match_player, df_match_odds, df_player, df_teams, df_coaches, df_player_sofifa, df_player_fifa_sofifa)

    # DATA PREPARATION
    if data_prep:
        print(" Data preparation ".center(120, "#"))
        # Hiperparametros # PODRIA PONERLOS EN UN DICT Y HACER EL DATAFRAME MAS AUTOMATICO
        n_days = 30  # Numero de dias anteriores para calcular las estadisticas promedio
        n_years_h2h = 3  # Años para construir el historial entre los equipos
        thr_corr = 0.7  # Correlacion umbral para la eliminacion de variables altamente correlacionadas
        thr_fs = 0.3  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)
        df_hiper_prep = pd.DataFrame(data={'n_days': [n_days], 'n_years_h2h': [n_years_h2h], 'thr_corr': [thr_corr], 'thr_fs': [thr_fs], 'fill_na': [fill_na]}, index=[0])
        
        # df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
        # print(df.head(2))

        # Preparo el dataset para el analisis
        df_match, df_match_player, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_player_fifa_sofifa, export=False)
        df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, export=export)
        df = dp.integrate_data(df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, export=export) 
        # df = dp.construct_data(df, n_days=n_days, n_years_h2h=n_years_h2h, export=export)
        # df = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, export=export)
        # df = dp.nan_values_treatment(df, fill_na=fill_na, export=export)
        
        if export:
            df_hiper_prep.to_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx', index=False)

    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_selected_nan.xlsx', index_col=0)
        print(df.head(3), df.shape)

    # MODELING
    if modeling:
        print(" Modeling ".center(120, "#"))
        
        modelo = RandomForestClassifier()  # LogisticRegression(), RandomForestClassifier()
        model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")

        build_specific_model = True
        if build_specific_model:

            # LogisticRegression(C=0.1, fit_intercept=False, penalty='l1', solver='saga')
            d_params = {
                'RandomForestClassifier': {
                    'n_estimators': 500,  # Número de árboles en el bosque.
                    'criterion': 'entropy',  # Función para medir la calidad de una división.
                    'max_depth': 5, # Podria reemplazar 10 y 15 por 8 # Profundidad máxima de los árboles.
                    'min_samples_split': 2,  # Número mínimo de muestras requeridas para realizar una división en un nodo interno.
                    'min_samples_leaf': 4, # Número mínimo de muestras requeridas para estar en un nodo hoja.
                    'max_features': 'sqrt',  # Número máximo de características a considerar al buscar la mejor división.
                    'bootstrap': True,  # Indica si se deben realizar muestras bootstrap al construir árboles.
                },
                'LogisticRegression': {
                    'penalty': 'l1',  # Tipo de regularización a aplicar.
                    'C': 0.1,  # Podria probar un 3.0 en vez de 5  # Inverso de la fuerza de regularización.
                    'solver': 'saga', # Algoritmo a utilizar en la optimización del problema.
                    'fit_intercept': False,  # Especifica si se debe ajustar o no el intercepto.  # Mas del 75% de las veces es True
                    # 'max_iter': 100,  # Podria prescindir de 1000 # Número máximo de iteraciones para la convergencia del algoritmo.
                    'multi_class': 'auto',  # Esquema de clasificación multiclase.
                },
            }
            hiperparametros = d_params[model_name]
        else:
            hiperparametros = None

        # Hiperparametros
        test_val_size = 0.25  # Porcentaje del total de datos destinado a validacion y test.
        test_size = 0.5  # Porcentaje de test_val_size destinado a test.
        bal_type = None # bal_type de balanceo a realizar [None, 'over', 'under']
        k = 5  # Numero de folds para seleccionar best parameters y para entrenar modelo
        df_hiper_mod = pd.DataFrame(data={'test_val_size': [test_val_size], 'test_size': [test_size], 'bal_type': [bal_type], 'k': [k]}, index=[0])        

        # Analizo datos con un modelo
        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df, bal_type, test_val_size, test_size, fill_na=fill_na, export=export)
        model_best_params, cv_accuracy = mo.build_model(modelo, X_val=X_val, y_val=y_val, X_train=X_train, y_train=y_train, k=k, params=hiperparametros, export=export)
        d_metrics = mo.assess_model(model_best_params, X_test, y_test, export=export)

        if export:
            df_hiper_mod.to_excel(f'./p4_modeling/data/{country}/df_hiper_mod.xlsx', index=True)
        
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()