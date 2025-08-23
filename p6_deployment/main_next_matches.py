# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
from utils.set_up_logging import logger
from utils import directories
import os
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_next_matches, extract_data
from p2_data_understanding import describe_data
## Data preparation
from main import DataPreparation, Modeling
from p3_data_preparation.format_data import value_nan_to_none
from p3_data_preparation import clean_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
from p3_data_preparation.select_data import determine_country_competitions
# Modeling
from p4_modeling import betting_strategy
import pickle
import joblib


class DataUnderstandingNew():

    def __init__(self, id_country, country, export: bool = True):
        self.id_country = id_country
        self.country = country.lower()
        self.export = export
        self.make_directories()

    def make_directories(self):
        base_path = f'./data/{self.country}/p6_deployment'
        self.path_missing = f'{base_path}/missing'
        self.path_unders = f'{base_path}/data_understanding'
        
        l_directorios = [
            self.path_unders,
            f'{self.path_missing}/data_understanding/all',
            f'{self.path_missing}/old_updated',   
            f'{self.path_missing}/data_preparation/all',
        ]

        directories.make_directories(l_directorios=l_directorios)
        
    def collect_initial_data_new(self, l_competencies, df_comp_country: pd.DataFrame, n_days: int = 7, _print: bool = False):
        """
        Extraccion de datos de los partidos en los proximos dias en todas las competiciones del pais.

        # Parameters
            n_days: Numero de dias desde hoy para recolectar proximos partidos
            export:
            _print: 

        # Returns
            dfs...
        """
        logger.info("Collecting data...")
        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # POR COMPETITION
        for id_competition in l_competencies:
            
            row_comp = df_comp_country[df_comp_country['id_competition'].astype(int) == int(id_competition)]
            competition, is_cup = row_comp['competition_flashscore'].values[0], row_comp['is_cup'].values[0]
            if _print:
                print(f" Competition: {competition} id_comp: {id_competition}".center(120, '+'))

            # Extraigo proximos partidos
            df_match_next, df_match_player_next, df_match_odds = extract_next_matches(self.id_country, self.country, id_competition, competition, is_cup, n_days=n_days)

            # Agrego pais y competicion
            df_match_next['country'] = self.country
            df_match_next['competition'] = competition

            # Guarda datos de competition
            df_match_concat = pd.concat([df_match_concat, df_match_next], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_next], axis=0)
            df_match_odds_concat = pd.concat([df_match_odds_concat, df_match_odds], axis=0)

        # Verificaciones
        if len(df_match_concat) > 0:

            ## Df_match_player
            if len(df_match_player_concat.columns) == 0:
                logger.warning(f"No se tiene las formaciones de ninguno de los {len(df_match_concat)} partidos a predecir. Si ya esta al menos la seccion 'Will not play', no deberia fallar.")

            ## Df_match_odds
            if len(df_match_odds_concat.columns) != 3:
                logger.warning("No se recolectaron las odds en df_match_odds. Probablemente cambió el XPATH de Flashscore.") # Puede que sea un solo partido y falte mucho y aun no tenga cuotas..
                df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            elif df_match_odds_concat.isna().any().any(): 
                logger.warning("El DataFrame df_match_odds contiene al menos un valor NaN. Puede deberse a que BET aun no asigno cuotas a ciertos partidos para los que falta mucho")
                df_match_odds_concat.dropna(subset=['odds_home'], inplace=True)
                df_match_concat = df_match_concat[df_match_concat.index.isin(df_match_odds_concat.index)]
                df_match_player_concat = df_match_player_concat[df_match_player_concat.index.isin(df_match_odds_concat.index)]
              
        # Exporto datasets
        if self.export:
            df_match_concat.to_excel(f'{self.path_unders}/df_match_next.xlsx', index=True)
            df_match_player_concat.to_excel(f'{self.path_unders}/df_match_player_next.xlsx', index=True)
            df_match_odds_concat.to_excel(f'{self.path_unders}/df_match_next_odds.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat

    def collect_missing_data(self, df_match: pd.DataFrame, df_comp_country: pd.DataFrame, n_seasons_max: int = 1, _print: bool = False):
        """
        Extraccion de varias competencias de un mismo country.
        """
        logger.info("Collecting data...")
        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Solo extriago las competencias que tengo en los datos viejos
        l_ids_extracted = list(df_match.index) 
        l_competencies = df_match['id_competition'].unique()
        print("Competencias extraidas: ", l_competencies)

        # POR COMPETITION (solo las que hay en df_match)
        for id_competition in l_competencies:

            # evito competencias que extraje en df_match pero no quiero recolectar missing
            if id_competition in [1672, 1673]:
                continue

            # Obtengo nombre de competicion y is_cup
            df_comp_filt = df_comp_country[df_comp_country['id_competition'] == id_competition] 
            competition, is_cup = df_comp_filt['competition_flashscore'].values[0], df_comp_filt['is_cup'].values[0]
            if _print:
                print(f" Competition: {competition} ".center(120, '+'))

            # Actualizo df_match y df_match_player con los partidos faltantes
            df_match_miss, df_match_player_miss, df_match_odds_miss = extract_data(self.id_country, self.country, id_competition, competition, is_cup, n_seasons_max=n_seasons_max, l_ids_already_collected=l_ids_extracted, export=False)
            if _print:
                print(f"Cantidad de partidos faltantes en df_match: {df_match_miss.shape[0]}")

            # Concateno dfs
            df_match_concat = pd.concat([df_match_concat, df_match_miss], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_miss], axis=0)
            df_match_odds_concat =  pd.concat([df_match_odds_concat, df_match_odds_miss], axis=0)
            
        # Verificaciones
        if len(df_match_concat) > 0:

            ## Df_match_player
            missing_start_cols = not any("_start_" in col for col in df_match_player_concat.columns)
            missing_sub_cols = not any("_sub_" in col for col in df_match_player_concat.columns)

            if len(df_match_player_concat.columns) == 0:
                logger.error(f"No se tiene las formaciones de ninguno de los {len(df_match_player_concat.columns)} partidos ya jugados. Esto no es comun. Solo es correcto si realmente no existe ningun dato de las formaciones para estos partidos (ni missing players).")
                raise ValueError
            
            elif missing_start_cols or missing_sub_cols:
                
                warning_message = "No se encontraron datos de jugadores en las columnas.\n"
                if missing_start_cols:
                    warning_message += "  - Faltan columnas que contengan '_start_'.\n"
                if missing_sub_cols:
                    warning_message += "  - Faltan columnas que contengan '_sub_'.\n"
                
                warning_message += "El unico caso en que esto no es un problema es si los partidos missing recien recolectados son poco importantes o de competencias no tan seguidas. Pero sino, posiblemente se deba a un cambio en el XPATH de los datos de las formaciones.\n"

                logger.warning(warning_message)
                
                # Pausa el script y espera la entrada del usuario
                user_input = input("Queres continuar igual? (y/n)")
                
                # Opcional: Podrías salir del script si el usuario ingresa un valor específico
                if user_input.lower() != 'y':
                    sys.exit("Script detenido por el usuario.")

            ## Df_match_odds
            porcentaje_nan = df_match_odds_concat.isna().mean().mean()
            umbral = 0.5
            if porcentaje_nan > umbral: 
                logger.error(f"El DataFrame df_match_odds_concat tiene {porcentaje_nan:.2%} valores NaN, lo cual supera el umbral de {umbral:.2%}. Esto no es posible una vez jugado el partido, se deben tener las cuotas.")
                raise ValueError

            if len(df_match_odds_concat.columns) != 3:
                logger.error("No se recolectaron todas las odds en df_match_odds. Esto no es posible una vez jugado el partido, se debe tener las cuotas.")
                raise ValueError

        # Exporto datasets
        if self.export:
            df_match_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_miss.xlsx', index=True)
            df_match_player_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_player_miss.xlsx', index=True)
            df_match_odds_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_odds_miss.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat

    def describe_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame, verbose: int = 1):
        """
        Descripción de dataframes en terminos de dtypes, nan values, registros unicos, etc.
        """
        logger.info("Describing data... ")

        d = {'df_match': df_match, 'df_match_player': df_match_player, 'df_match_odds': df_match_odds}

        for name, df in d.items():
            print(name)
            if verbose >= 0:
                print(df.shape)
            elif verbose >= 1:
                print(df.head(2))
            elif verbose >= 2:
                describe_data.getting_to_know_data(df, verbose=verbose)
                describe_data.verificar_unicidad_registros(df) # Verifico unicidad de registros segun campos id

class DataPreparationNew(DataPreparation):

    def __init__(self, id_country, country, iteration_date, export: bool = True):
        self.id_country = id_country
        self.country = country
        self.iteration_date = iteration_date
        self.export = export
        self.make_directories()
        super().__init__(self.id_country , self.country, self.iteration_date)

    def make_directories(self):

        self.BASE_DIR = f"./data/{self.country}/p6_deployment/data_preparation"

        l_directorios = [
            f'{self.BASE_DIR}/format_data',
            f'{self.BASE_DIR}/clean_data',
            f'{self.BASE_DIR}/fill_data',
            f'{self.BASE_DIR}/construct_data',
        ]

        if self.export:
            for directorio in l_directorios:
                if not os.path.exists(directorio):
                    # Si no existe, crear el directorio
                    os.makedirs(directorio)

    # Rellenado de datos necesarios para predecir
    def fill_data_not_available_yet(self, df_next_matches: pd.DataFrame, df_last_old_matches: pd.DataFrame, verbose: int = 0):
        """
        Relleno datos aun no disponibles debido a que aun falta mas de 30 min para el partido. Asi, poder predecir a pesar de tener datos aun no 
        disponibles.

        # Parameters:
            df_next_matches: Dataframe con proximos partidos ya integrado. (DataFrame)
            df_last_old_matches: Dataframe con los ultimos partidos ya jugados (DataFrame)

        # Returns:
            Dataframe con proximos partidos habiendo reemplezaso los datos aun no disponible por valores en ultimos partidos. 
        """
        logger.info("Rellenando datos aun no disponibles...")

        # En caso que aun no se cuente con las formaciones, asigno promedio en ultimos partidos
        l_player_cols  = [col for col in df_last_old_matches.columns if ('player_start' in col) or ('player_sub' in col)]  # Selecciono las variables que corresponden a jugadores
        df_next_matches, df_copiado_formaciones = clean_data.fillna_with_mean_in_last_matches_with_df(df_to_fill=df_next_matches, df=df_last_old_matches, cols_to_fill=l_player_cols)            
        
        # A futuro: Reduccion de medias segun cantidad de lesionados
        # df_next_matches = self.reduce_mean_by_missing_players(df_next_matches, df_last_old_matches)
        
        # Copio valores en ultimos partidos (deberia copiar solo referee y coaches)
        miss_player_columns = [col for col in df_last_old_matches.columns if ('player_miss' in col)]  # --> ojo porque no se si las rellena ok... es complejo el rellenado.
        l_var_to_copy = miss_player_columns + ['id_coach_home', 'id_coach_away']  # Es clave copiar coaches porque no suele estar hasta que esten las formaciones..
        df_next_matches, df_copiado = self.fillna_with_last_match_value(df_next_matches, df_last_old_matches, cols_to_fill=l_var_to_copy) 
        if self.export:
            df_copiado_formaciones.to_excel(f"{self.BASE_DIR}/fill_data/df_copiado_formaciones.xlsx", index=True)
            # df_copiado.to_excel(f"{self.BASE_DIR}/fill_data/df_copiado_ref_and_coaches.xlsx", index=True)
            df_next_matches.to_excel(f"{self.BASE_DIR}/df_filled.xlsx", index=True)

        return df_next_matches, df_copiado_formaciones, df_copiado
    
    def reduce_mean_by_missing_players(self, df_next_matches, df_last_old_matches, threshold_injury_increase = 0.2):
        # 20% más lesionados se considera significativo

        ## Calcular variacion de missing del partido actual respecto de la media en los ultimos partidos
        df_last_old_matches['sum_rat_player_miss_home_avg'] = df_last_old_matches.groupby('id_team_home')['sum_rat_player_miss_home'].transform('mean')
        df_last_old_matches['sum_rat_player_miss_away_avg'] = df_last_old_matches.groupby('id_team_away')['sum_rat_player_miss_away'].transform('mean')
        
        df_next_matches['variation_miss_home']  = (df_next_matches['sum_rat_player_miss_home'] - df_last_old_matches['sum_rat_player_miss_home_avg']) / df_last_old_matches['sum_rat_player_miss_home_avg']
        df_next_matches['variation_miss_away'] = (df_next_matches['sum_rat_player_miss_away'] - df_last_old_matches['sum_rat_player_miss_away_avg']) / df_last_old_matches['sum_rat_player_miss_away_avg']

        ## Si hay mas missing en este partido, perjudicar la media de los ultimos partidos copiada para START y SUB.
        df_next_matches.loc[df_next_matches['variation_miss_home'] > threshold_injury_increase, ['mean_rat_player_start_home', 'mean_rat_player_sub_home']] *= 0.9
        df_next_matches.loc[df_next_matches['variation_miss_away'] > threshold_injury_increase, ['mean_rat_player_start_away', 'mean_rat_player_sub_away']] *= 0.9

        df_next_matches.to_excel("ruta.xlsx", index=True)
        df_last_old_matches.to_excel("ruta2.xlsx", index=True)

        return df_next_matches
    
    def fillna_with_last_match_value(self, df_new: pd.DataFrame, df: pd.DataFrame, cols_to_fill: list):
        """
        En los partidos nuevos, rellena los datos no disponibles con los datos de partidos anteriores.
        """  
        logger.info(f"Remplazando NaN por valor en ultimo partido en {cols_to_fill}...")
        df_copiado = pd.DataFrame(columns=['copiado_avoid_nan'], index=df_new.index)

        for var in cols_to_fill:

            # En caso que ningun proximo partido tenga formaciones, creo la columna jugador correspondient
            # print(f"\nVariable a promediar: {var}")
            if var not in df_new.columns:
                df_new[var] = np.nan

            # Por partido nuevo
            for id_match, row in df_new.iterrows():

                # Si el valor actual es NaN
                if (pd.isna(row[var])):

                    team = row['id_team_home'] if "_home" in var else row['id_team_away']
                    df_match_team = df.loc[(df['id_team_home'] == team) | (df['id_team_away'] == team)] # el referee lo copia del partido de un solo equipo

                    # Busco el partido anterior del team  --> podria hacer ciclo para que busque hasta que encuentre un valor no nan...
                    for i, row_prev_match in df_match_team.iterrows():

                        home_or_away_prev_match = "home" if row_prev_match['id_team_home'] == team else "away"
                        variable_prev_match = var if var in cols_to_fill else f"{var}_{home_or_away_prev_match}"

                        # Si el valor en el partido anterior no es NaN
                        if not (pd.isna(row_prev_match[variable_prev_match])):
                            # Guardo valor del partido anterior y dejo de revisar partidos del equipo puesto que ya rellene el valor
                            df_new.loc[id_match, var] = row_prev_match[variable_prev_match]
                            df_copiado.loc[id_match, 'copiado_avoid_nan'] = 1
                            df_copiado.loc[id_match, var] = row_prev_match[variable_prev_match]
                            break

        return df_new, df_copiado
    
    # Construccion de datos
    def filter_rows_to_construct(self, df_integrated_updated, df_constructed_train, predict_missing):
        """
        Preparar df tal que me aseguro de construir con los mismos datos que cuando entrené (y asi construir bien vars como ELO)
        """
        logger.info("Preparing to construct...")
        
        # Obtener los índices de entrenamiento
        train_indices = df_constructed_train.index

        # Filtrar registros posteriores al entrenamiento y obtener sus índices
        last_train_date = df_constructed_train['date'].max()
        missing_indices = df_integrated_updated.loc[df_integrated_updated['date'] > last_train_date].index

        # Unir los índices de entrenamiento con los nuevos registros
        final_indices = train_indices.union(missing_indices)

        # Filtrar el dataframe con los índices determinados
        df_integrated_updated_clean = df_integrated_updated.loc[final_indices]

        if self.verbose >= 0:
            print(f"Total registros en entrenamiento: {len(train_indices)}")
            print(f"Total registros nuevos con competencias seleccionadas: {len(missing_indices)}")
            print(f"Total registros después de combinación: {len(final_indices)}")
            print(f"Total registros finales después de limpieza: {len(df_integrated_updated_clean)}")

        if not predict_missing and len(df_integrated_updated_clean) != len(final_indices):
            logger.error("El dataframe utilizado para construir no tiene las filas que deberia.")
            raise ValueError
        
        return df_integrated_updated_clean

    def construct_data_new(self, df_next_matches: pd.DataFrame, df_last_old_matches,
                           n_last_matches:list, n_years_h2h: int, segun_localia: bool, calculate_dif: bool, decay_rate: float,
                           columns_used: list, verbose: int = 0):
        """
        Construye nuevos datos a partir de un dataframe existente.

        # Parameters
            df_next_matches: Dataframe con proximos partidos ya integrado. (DataFrame)
            df_old_matches: Dataframe con partidos ya jugados e integrado. (DataFrame)
            df_last_old_matches: Dataframe con los ultimos partidos ya jugados (DataFrame)
            n_last_matches: Número de últimos partidos a considerar para el cálculo de variables. (int)
            n_years_h2h: Numero de años para construir historial entre equipos. (int)
            segun_localia: Construir variables por localia o no. (bool)
        
        # Usar prod_idxs y prod_cols para agilizar construccion, construyendo solo las columnas necesarias y para los partidos necesarios (next_matches)
        # Returns
            Dataframe con proximos partidos construido utilizando los partidos ya jugados. (DataFrame)
        """
        # Construyo historicas solo si hay "ultimos partidos"
        with_historic = True if len(df_last_old_matches) > 0 else False

        # En caso que los proximos partidos ya esten en df_old_last_matches (o sea, los partidos ya se jugaron y los recolectaste como missing, tirara error al momento de predecir por indice repetido.)
        df_concat_last = pd.concat([df_next_matches, df_last_old_matches], axis=0)

        # Construyo datos (sin historiales) luego de concatenar proximos partidos (df_next_matches) y los ultimos partidos ya jugados (df_last_old_matches)
        df_constructed = self.construct_data(
            df_concat_last, 
            n_last_matches=n_last_matches,
            n_years_h2h=n_years_h2h, 
            segun_localia=segun_localia, 
            calculate_dif=calculate_dif, 
            with_historic=with_historic,
            decay_rate=decay_rate, 
            prod=True, 
            prod_idxs=df_next_matches.index,
            path_prod=f'{self.BASE_DIR}/construct_data',
            export=False
            )
        
        # Separo datos construidos entre los proximos partidos y los ya jugados
        df_next_matches = df_constructed[df_constructed.index.isin(df_next_matches.index)]

        # Agregar las columnas que faltan ("las que se deberian construir tambien") y rellenar con NaN
        if not with_historic:

            user_input = input("No hay ultimos partidos. ¿Inicializar columnas historicas con NaN? (Recomendado solo a inicios de temporada o tras baches grandes)'").strip().lower() 
            if user_input == "y":
                for columna in columns_used: 
                    if columna not in df_next_matches.columns:
                        df_next_matches[columna] = np.nan  # no le des valor 0 porque sino las predicciones son 33-33-33.
            else:
                raise SystemExit("⛔ Predicción cancelada por el usuario.")

        if self.export:
            df_last_old_matches.to_excel(f'{self.BASE_DIR}/construct_data/df_matches_to_construct_historic_values.xlsx', index=True)
            df_next_matches.to_excel(f'{self.BASE_DIR}/df_constructed.xlsx', index=True)

        return df_next_matches

    # Limpieza post select     
    def describe_and_verify_nan(self, df: pd.DataFrame):
        logger.info("Describing nan values in prod...")

        # Calcular el porcentaje de valores NaN por columna
        df_nan_col = df.isna().mean()

        # Identificar cols con mucho nan
        nan_min = 0.5  # Umbral del 50% de NaN para considerar una columna como problemática
        full_nan_cols = df_nan_col[df_nan_col == 1].index.tolist()  # Identificar columnas con 100% NaN
        nan_cols = df_nan_col[df_nan_col > nan_min].index.tolist()  # Identificar columnas con más del x% de NaN
        perc = len(nan_cols) / len(df.columns)

        # Imprimir información y generar un error si se cumplen las condiciones
        if full_nan_cols:
            msg = f"Error: Hay {len(full_nan_cols)} columnas que tienen 100% de NaN values. Columnas 100% NaN: {full_nan_cols}."
            logger.error(msg)
            raise ValueError(msg)

        elif perc > 0.5:
            msg = f"Error: Hay {len(nan_cols)} columnas con mucho NaN value. Columnas > {nan_min*100:.0f}% NaN: {nan_cols}."
            logger.error(msg)
            raise ValueError(msg)

        # Guardar datos en Excel para referencia
        df_nan_col_sorted = df_nan_col.sort_values(ascending=False)

        # Crear columnas 'emergency_fill' y 'player_emergency_fill' 
        df_filled = self.add_emergency_fill_flags(df)

        # Exporto datos
        df_nan_col_sorted.to_excel(f'{self.BASE_DIR}/df_cols_nan.xlsx', index=True)
        df_filled.to_excel(f'{self.BASE_DIR}/df_filled.xlsx', index=True)

    def add_emergency_fill_flags(self, df):
        """
        Agrega dos columnas booleanas:
        - 'emergency_fill': 1 si hay al menos un NaN en la fila (en cualquier columna).
        - 'player_emergency_fill': 1 si hay al menos un NaN en columnas que contienen 'player' en su nombre.
        """
        df_aux = df.copy()

        # Marca 1 si la fila tiene al menos un NaN en cualquier columna
        df_aux['emergency_fill'] = df_aux.isna().any(axis=1).astype(int)

        # Marca 1 si hay al menos un NaN en columnas que contienen 'player' en su nombre
        player_cols = [col for col in df_aux.columns if 'player' in col]
        df_aux['player_emergency_fill'] = df_aux[player_cols].isna().any(axis=1).astype(int)

        return df_aux

    
class TrainingDataLoader():
    """
    Carga los hiperparametros y ciertos modelos (e.g. scaler) que se usaron durante el entrenamiento para luego poder usarlos en producción.
    """
    def __init__(self, country, n_model, model_name, iteration_date, verbose: int = 1):
        self.country = country
        self.n_model = n_model
        self.model_name = model_name
        self.iteration_date = iteration_date
        self.verbose = verbose
        self.construct_directories()

    def construct_directories(self):
        self.BASE_DIR_dp = f"./data/{self.country}/p3_data_preparation/{self.iteration_date}"
        self.BASE_DIR_mod = f"./data/{self.country}/p4_modeling/{self.iteration_date}"
    
    # Data preparation
    def load_data_preparation_hyperparameters(self):
        """
        Cargo hiperparametros de DataPreparation()
        """
        d = {}
            
        df_iteration = pd.read_excel(f"{self.BASE_DIR_mod}/df_iteration.xlsx")

        # Selecciono la primera. Hay una por modelo entrenado pero los hiper son =.
        try:
            row_hiper = df_iteration[df_iteration['n_iteration'] == self.n_model].iloc[0]  
        except IndexError:
            logger.error(f"El modelo {self.n_model} no fue entrenado en el entrenamiento del {self.iteration_date}.")
            raise IndexError
            
        # Guardo hiperparametros en diccionario
        print(row_hiper)
        # clean_post_integrate
        d['comp_to_select'] = eval(row_hiper['comp_to_select']) # .values[0]
        d['n_years_to_select'] = value_nan_to_none(row_hiper['n_years_to_select'])
        ## Construct_data
        d['n_last_matches'] = eval(row_hiper['n_last_matches']) 
        d['n_years_h2h'] = int(row_hiper['n_anios_hist']) # .values[0]
        d['segun_localia'] = row_hiper['segun_localia'] # .values[0]
        d['calculate_dif'] = row_hiper['calculate_dif'] # .values[0]
        d['decay_rate'] = 0 if row_hiper['decay_rate'] == 0.0 else row_hiper['decay_rate'] 
        ## Select_data
        d['thr_corr'] = value_nan_to_none(row_hiper['thr_corr'])
        d['thr_fs'] = value_nan_to_none(row_hiper['thr_fs'])
        # clean_post_select
        d['fill_na'] = value_nan_to_none(row_hiper['fill_na'])
        if isinstance(d['fill_na'], float):
            d['fill_na'] = '0'
        d['selected_columns'] = eval(row_hiper['X_columns'])  # Columnas utilizadas para entrenar el modelo # eval() para pasar de string a lista

        # Construyo paths para levantar modelos / datos de cuando entrené
        self.path_clean = f'{d['comp_to_select']}_{d['n_years_to_select']}'
        self.path_construct = f'{self.path_clean}__{d['n_last_matches']}_{d['n_years_h2h']}_{d['segun_localia']}_{d['calculate_dif']}_{d['decay_rate']}'
        self.path_sel = f'{self.path_construct}__{d['thr_corr']}_{d['thr_fs']}_{d['fill_na']}'

        # Imprimo hiper levantados
        if self.verbose >= 0:
            print("Hiperparametros cargados:")
            for key, value in d.items():
                print(f'\t {key}: {value}')

        return d

    def load_df_constructed(self):
        """
        Para construir los datos tal y como lo hicimos durante el entrenamiento
        """
        logger.info("Levento df_constructed con el que entrené")
        path = f'{self.BASE_DIR_dp}/construct_data/df_constructed_{self.path_construct}.xlsx'       
        df = pd.read_excel(path, index_col=0)

        if self.verbose >= 1:  
            print(df.head(3))

        return df
    
    def load_df_etiquetas(self):

        logger.info("Levento etiquetas con el que entrené")
        path_tag = f'{self.BASE_DIR_dp}/tag/df_etiquetas_{self.path_construct}.xlsx'       
        df_etiquetas = pd.read_excel(path_tag, index_col=0)

        if self.verbose >= 1:  
            print(df_etiquetas.head(3))

        return df_etiquetas

    def load_scaler_model(self):
        """
        Levanto modelo utilizado en entrenamiento para escalar datos
        """
        path_scaler = f'{self.BASE_DIR_dp}/clean_post_select/scaler_model_{self.path_sel}.pkl'
        scaler, columns_scaled = joblib.load(path_scaler)
        return scaler, columns_scaled

    # Modeling
    def load_model(self):
        """
        Levanto modelo a utilizar para hacer predicciones.
        """
        # Si se levanta de main_find_best_hyper.py
        if self.n_model is not None:
            path_model_1 = f"{self.BASE_DIR_mod}/models/{self.n_model}_{self.model_name}.pkl" 
            loaded_model = pickle.load(open(path_model_1, "rb"))
        # Si se levanta de main.py
        else:
            logger.error("Se levanta el modelo y el scaler desde de main.py")
            path_model_1 = f"./data/{self.country}/p4_modeling/modelo.pkl"
            loaded_model = pickle.load(open(path_model_1, "rb"))

        return loaded_model

    def load_modeling_hyperparameters(self):
        """
        Cargo hiperparametros de Modeling()

        Mejora: 
         - Levantar parametros por resultado...
        """
        # Estrategia por resultado
        try:
            df_hiper = pd.read_excel(f"{self.BASE_DIR_mod}/best_model/3_bet_strategy/df_strategy_{self.n_model}_{self.model_name}.xlsx", index_col=0)

            # Si es por resultado
            if len(df_hiper) == 3:
                logger.critical("Se levantó la estrategia de apuesta por resultado")
            else:
                logger.warning("Se levanta una estrategia comun a todos los resultados")
                d = {
                    'prob_dp': value_nan_to_none(df_hiper['prob_dp'].values[0]), 
                    'curva': df_hiper['curva'].values[0], 
                    'm': df_hiper['m'].values[0], 
                    'b': df_hiper['b'].values[0], 
                    'k': df_hiper['k'].values[0]
                    }
                return d
            if self.verbose >= 0:
                logger.info("Hiperparametros de apuesta cargados:")
                logger.info(df_hiper)

        except FileNotFoundError as e:
            logger.error("Falló la carga del df_strategy")
            return {'prob_dp': None, 'curva': 'linear', 'm': 10, 'b': 0, 'k': 1}
        
        return df_hiper

class MissingData:
    def __init__(self, country, iteration_date, verbose: int = 1):
        self.country = country
        self.iteration_date = iteration_date
        self.verbose = verbose
        self.construct_directories()

    def construct_directories(self):
        self.BASE_DIR_du = f"./data/{self.country}/p2_data_understanding"
        self.BASE_DIR_dp = f"./data/{self.country}/p3_data_preparation/{self.iteration_date}"
        self.BASE_DIR_mod = f"./data/{self.country}/p4_modeling/{self.iteration_date}"
        
        path_missing = f'./data/{self.country}/p6_deployment/missing'
        self.BASE_DIR_MISSING_AND_OLD = f'{path_missing}/old_updated'
        self.BASE_DIR_MISSING_DP = f"{path_missing}/data_preparation"
        self.BASE_DIR_MISSING_ALL_du = f"{path_missing}/data_understanding/all"
        self.BASE_DIR_MISSING_ALL_dp = f"{path_missing}/data_preparation/all"
    
    def read_last_flashscore_data(self):
        """
        Obtengo ultima version de df_match, df_match_player y df_match odds (con missing)
        No lo pongo en la clase puesto que no son datos usados durante el entrenamiento.
        """
        try:
            df_match = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match.xlsx', index_col=0)
            df_match_player = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_player.xlsx', index_col=0)
            df_match_odds = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_odds.xlsx', index_col=0)
            logger.info('Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')

        except FileNotFoundError:
            
            user_input = input("No se encontraron los dfs con missing concatenados. ¿Quiere levantar los dataframes de partidos viejos? (y/n): ")
            if user_input.strip().lower()  == "y":
                df_match = pd.read_excel(f'data/{self.country}/p2_data_understanding/df_match.xlsx', index_col=0) 
                df_match_player = pd.read_excel(f'data/{self.country}/p2_data_understanding/df_match_player.xlsx', index_col=0) 
                df_match_odds = pd.read_excel(f'data/{self.country}/p2_data_understanding/df_match_odds.xlsx', index_col=0) 
            else:
                raise SystemExit("⛔ Predicción cancelada por el usuario.")
        
        logger.info(f"Shapes: \t df_match: {df_match.shape} \t df_match_player:{df_match_player.shape} \t df_match_odds: {df_match_odds.shape}")
        return df_match, df_match_player, df_match_odds

    def read_last_integrate_data(self):

        # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
        try:
            df_integrated = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_integrated.xlsx', index_col=0)
            logger.info('Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')

        # Si no existe un df_integrated concatenado entre old y missing
        except FileNotFoundError:

            warning_msg = f"No se pudo levantar el df_integrated con old + missing. Esto es correcto solo si nunca se ha extraido / integrado missing. Desea levantar el df_integrated con el que se entrenó? (y/n): "
            user_input = input(warning_msg).strip().lower() 
            if user_input == "y":
                df_integrated = pd.read_excel(f'{self.BASE_DIR_dp}/df_integrated.xlsx', index_col=0)  # Tiene missing hasta el dia en el que entrené (por no desde ese dia en adelante)
                logger.warning('Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
            else: 
                raise SystemExit("⛔ Predicción cancelada por el usuario.")            
        
        logger.info(f"df_integrated: {df_integrated.shape}")
        return df_integrated

    def read_last_missing_data(self):
        """
        Leer todos los datos missing ya extraidos
        """
        try:
            df_match_miss = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_miss.xlsx', index_col=0)
            df_match_player_miss = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_player_miss.xlsx', index_col=0)
            df_match_odds_miss = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_odds_miss.xlsx', index_col=0)

        # Si es la primera vez que extraigo partidos missing
        except FileNotFoundError:

            warning_msg = f"No se pudo levantar datos missing ya extraidos. Esto es correcto solo si nunca se ha extraido missing. Desea inicializar crear los dataframes? (y/n): "
            user_input = input(warning_msg).strip().lower() 
            if user_input == "y":
                df_match_miss, df_match_player_miss, df_match_odds_miss = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            else: 
                raise SystemExit("⛔ Predicción cancelada por el usuario.")            

        return df_match_miss, df_match_player_miss, df_match_odds_miss
    
    def read_last_integrate_missing_data(self):
        try:
            df_integrated_missing_all = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_dp}/df_integrated_missing.xlsx', index_col=0)

        # Si es la primera vez que extraigo partidos missing
        except FileNotFoundError:
            logger.error("Nunca se ha integrado missing")
            warning_msg = f"No se pudo levantar el df_integrated_missing. Esto es correcto solo si nunca se ha integrado missing. Desea inicializar crear el dataframe? (y/n): "
            user_input = input(warning_msg).strip().lower() 
            if user_input == "y":
                df_integrated_missing_all = pd.DataFrame()  # Es importante para que se guarde por primera vez df_integrated_missing en /all 
            else: 
                raise SystemExit("⛔ Predicción cancelada por el usuario.")
        
        return df_integrated_missing_all

    def concat_old_with_missing(self, df_match, df_match_player, df_match_odds, df_match_miss, df_match_player_miss, df_match_odds_miss):
        """
        Exporta datos de partidos con los que se entrena el modelo y los partidos missing,
        evitando duplicar datos ya existentes en los DataFrames originales.
        """
        len_inicial = len(df_match)
        len_inicial_miss = len(df_match_miss)

        # Función auxiliar para evitar duplicados antes de concatenar
        def avoid_duplicate_concat(df_original, df_miss):
            return pd.concat(
                [df_original, df_miss.loc[~df_miss.index.isin(df_original.index)]],
                axis=0
            )
        
        # Concatenar evitando duplicados
        df_concat_match = avoid_duplicate_concat(df_match, df_match_miss)
        df_concat_match_player = avoid_duplicate_concat(df_match_player, df_match_player_miss)
        df_concat_match_odds = avoid_duplicate_concat(df_match_odds, df_match_odds_miss)

        len_final = len(df_concat_match)
        logger.warning(f"Old: {df_match.shape} + Missing: {df_match_miss.shape} = {df_concat_match.shape}")

        # Verificación
        verif = (len_inicial + len_inicial_miss) == len_final
        if not verif:
            logger.error("Fallo la concatenación de partidos missing a los datos viejos")

        # Exportar datos
        df_concat_match.to_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match.xlsx')
        df_concat_match_player.to_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_player.xlsx')
        df_concat_match_odds.to_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_odds.xlsx')
        logger.info(f"Shape de df_match concatenado con missing: {len_inicial} --> {len_final}")

    def concat_with_missing_already_extracted(self, df_match_miss, df_match_player_miss, df_match_odds_miss, df_match_miss_comp, df_match_player_miss_comp, df_match_odds_miss_comp):
        """
        Guarda los nuevos partidos missing con los que ya tenía, evitando duplicados.

        Funciona pero hay que ver cuando hay repetidos si los evita... (si corres bien, nunca deberia siquiera tener que evitarlo... pero bueno).
        """
        # Identificar las claves primarias únicas en los datos previos
        keys_match = df_match_miss_comp.index if df_match_miss_comp.index.is_unique else df_match_miss_comp['id_match']
        keys_player = df_match_player_miss_comp.index if df_match_player_miss_comp.index.is_unique else df_match_player_miss_comp['id_player']
        keys_odds = df_match_odds_miss_comp.index if df_match_odds_miss_comp.index.is_unique else df_match_odds_miss_comp['id_match']

        # Filtrar nuevos registros que no estén ya en los datos previos
        df_match_miss_new = df_match_miss[~df_match_miss.index.isin(keys_match)]
        df_match_player_miss_new = df_match_player_miss[~df_match_player_miss.index.isin(keys_player)]
        df_match_odds_miss_new = df_match_odds_miss[~df_match_odds_miss.index.isin(keys_odds)]

        # Concatenar solo los registros nuevos
        df_match_miss_comp_ct = pd.concat([df_match_miss_comp, df_match_miss_new], axis=0)
        df_match_player_miss_comp_ct = pd.concat([df_match_player_miss_comp, df_match_player_miss_new], axis=0)
        df_match_odds_miss_comp_ct = pd.concat([df_match_odds_miss_comp, df_match_odds_miss_new], axis=0)

        logger.info(f"Se han añadido {len(df_match_miss_new)} nuevos registros a df_match_miss.")
        logger.info(f"Se han añadido {len(df_match_player_miss_new)} nuevos registros a df_match_player_miss.")
        logger.info(f"Se han añadido {len(df_match_odds_miss_new)} nuevos registros a df_match_odds.")

        # Exporto datos
        df_match_miss_comp_ct.to_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_miss.xlsx', index=True)
        df_match_player_miss_comp_ct.to_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_player_miss.xlsx', index=True)
        df_match_odds_miss_comp_ct.to_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_odds_miss.xlsx', index=True)
    
# Generales
def read_data_of_best_model(id_country, d_model = None, verbose : int = 1):
    """
    # Parameters

    # Return
        n_model: Numero del modelo (int)
        model_name: Tipo de modelo (str)    
    """
    
    if d_model is not None:
        logger.warning("Se usa modelo especificado como parametro y no necesariamente es el que se esta usando en produccion.")
        n_model, model_name = d_model['n_model'], d_model['model_name']
    else:
        # Levanto dataframe con los modelos a usar por pais
        df_best_models = pd.read_excel("./data/df_best_models.xlsx")

        # Selcciono fila del pais
        row_country = df_best_models[df_best_models['id_country'] == id_country]

        # Obtengo el modelo a usar
        n_model = int(row_country['n_model'].values[0])
        model_name = str(row_country['model_name'].values[0])
    
    if verbose >= 1:
        logger.critical(f"Model: {n_model} ; Model name: {model_name}")

    return n_model, model_name

def filter_dataframe_by_date(df: pd.DataFrame, initial_date, n_days: int, holgura: int = 0.5):
    """
    Filtrar registros de dataframe por fecha segun columna 'date'.

    # Parameters:
        df: Dataframe con columna 'date' al cual filtrar. (DataFrame)
        initial_date: Fecha. (Datetime)
        n_days: Dias desde la last_date.

    # Returns:
        df: Dataframe solo con registros en el periodo de tiempo especificado. (DataFrame)
    """
    # Asegurar que 'date' es datetime
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M') 

    # Determino fecha inical 
    limit_date = initial_date - datetime.timedelta(days=n_days) 
    print(f"Seleccion de ultimos partidos jugados: {limit_date} <-- {n_days}d -- {initial_date}")

    # Filtro segun fechas inicial y final
    df_filt = df[df['date'] >= limit_date]  # df_filt = df[(df['date'] >= limit_date) & (df['date'] <= last_date)]
    df_filt = df_filt.sort_values(by='date', ascending=False) # Ordeno por fecha ascendente
    logger.info(f"Seleccion de ultimos partidos: {len(df)} --> {len(df_filt)}")

    # Cálculo del mínimo de partidos necesarios
    min_last_matches = (n_days / 7) * 10 * holgura # 1 fecha cada 7 dias y 10 partidos por fecha. Algo asi seria el minimo.

    # Verificar si hay suficientes partidos para rellenar
    if len(df_filt) < min_last_matches:
        warning_msg = f"⚠️ Solo hay {len(df_filt)} partidos disponibles, mínimo recomendado: {min_last_matches:.0f}. Este mensaje deberia saltar solo cdo hubo un periodo largo sin partidos (sino puede que haya un error en el filtrado). \n¿Seguro que querés continuar? (y/n): "
        user_input = input(warning_msg).strip().lower() 
        if user_input != "y":
            raise SystemExit("⛔ Predicción cancelada por el usuario.")

    # Si no hay ultimos partidos
    if len(df_filt) == 0:
        logger.warning(f"No hay partidos en los ultimos {n_days} dias.")

    return df_filt


########################################################################## MAIN #######################################################################
def main(
        d_run: dict, id_country: int, iteration_date: str,
        n_seasons_missing : int = 1,                                                                    # Missing
        n_days_max_next_matches: int = 7, predict_missing: bool = False,                                # Data understanding
        n_days_fill_data: int = 30,                                                                     # Data preparation
        d_model: dict = None,                                                     # Modeling
        verbose: int = 1, export: bool = True, country: str = None,
        date_missing = None
        ):
    """
    Recoleccion de proximos partidos, preparacion y prediccion
    """
    start = time.time()

    # si el fifa aun no salió
    if datetime.datetime.now().month in [8, 9]: 
        n_days_fill_data = 120
        fifa_not_released_yet = True
        print(f"n_days_fill_data: {n_days_fill_data}")

    # Determino country si es None
    if country is None:
        df_countries = pd.read_excel('./data/df_countries.xlsx')
        country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0].lower()

    # Determino competence e ite_date
    iteration_date_dt = pd.to_datetime(iteration_date, format='%Y-%m-%d').date()  # con .date() saco hora y minutos
    
    d_comps = determine_country_competitions(id_country)
    print(d_comps['all_comp'])

    df_comp = pd.read_excel('./data/df_competencies.xlsx')
    if id_country == -1:
        df_comp_country = df_comp[df_comp['id_competition'].isin(d_comps['all_comp'])] 
    else:
        df_comp_country = df_comp[df_comp['id_country'] == id_country] 
    comp_public = d_comps['comp_solo_liga'] 

    if verbose >= 0:
        logger.info("\n" + "#"*120 + "\n" + f"COUNTRY: {country.upper()}".center(120) + "\n" + "#"*120 + "\n")
        print(f'Competencias: \n {df_comp_country}  \n Competencias publicas: {comp_public}')
        logger.info(f"Iteration date: {iteration_date_dt}")

    # Creo objetos de clases
    du = DataUnderstandingNew(id_country, country, export=export) # Creo objeto de clase DataUnderstanding
    dp = DataPreparationNew(id_country=id_country, country=country, iteration_date=iteration_date_dt, export=export) # Creo objeto de clase DataPreparation
    mo = Modeling(country=country, date=iteration_date_dt) # Creo objeto de clase DataPreparation
    mis = MissingData(country=country, iteration_date=iteration_date_dt)

    # Read data usada en mas de una seccion (para levantarla 1 sola vez)
    ## Missing data
    if d_run['run_missing'] or predict_missing:
        df_match_miss, df_match_player_miss, df_match_odds_miss = mis.read_last_missing_data()
        df_integrated_missing = mis.read_last_integrate_missing_data()
    df_integrated_upd = mis.read_last_integrate_data() # Last df_integrated con missing + old    
    ## SOFIFA
    df_player_sofifa = pd.read_excel(f"data/{country}/p2_data_understanding/old_updated/{iteration_date_dt}/df_player_sofifa.xlsx", index_col=0)
    df_player_fifa_sofifa = pd.read_excel(f"data/{country}/p2_data_understanding/old_updated/{iteration_date_dt}/df_player_fifa_sofifa.xlsx")  

    # _____________________________________________________________ MISSING DATA _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "MISSING DATA".center(120) + "\n" + "+"*120 + "\n")
    if d_run['run_missing']: 
        
        # Levanto datos: old + los ultimos missing extraidos
        df_match_upd, df_match_player_upd, df_match_odds_upd = mis.read_last_flashscore_data() # Last df_integrated con missing + old
        extract_missing = True
        if extract_missing:
            # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
            df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new = du.collect_missing_data(df_match_upd, df_comp_country=df_comp_country, n_seasons_max=n_seasons_missing)
            logger.info(f"Cantidad de partidos missing extraidos: {len(df_match_miss_new)}")
        else:
            # Unicamente util para cuando falla la preparacion de missing pero ya extrajiste...
            df_match_miss_new = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/df_match_miss.xlsx', index_col=0)
            df_match_player_miss_new = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/df_match_player_miss.xlsx', index_col=0)
            df_match_odds_miss_new = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/df_match_odds_miss.xlsx', index_col=0)
            print("Shape:", df_match_miss_new.shape, df_match_player_miss_new.shape, df_match_odds_miss_new.shape)

        # Si extrajo missing
        if len(df_match_miss_new) > 0:
            
            # Preparo datos missing            
            df_match_miss_new_f, df_match_player_miss_new_f, df_match_odds_miss_new_f, df_player_fifa_sofifa = dp.format_data(df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new, df_player_fifa_sofifa, reformat=True, export=False)            
            df_match_miss_new_c, df_match_player_miss_new_c, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match_miss_new_f, df_match_player_miss_new_f, df_player_sofifa, df_player_fifa_sofifa, export=False)
            # df_match_miss_new_vf, df_match_player_miss_new_vf, df_match_odds_miss_new_vf, df_player_sofifa, df_player_fifa_sofifa = dp.verify_format(df_match_miss_new_c, df_match_player_miss_new_c, df_match_odds_miss_new_f, df_player_sofifa, df_player_fifa_sofifa, prod=False) # prod=False pues los partidos ya se jugaron..
            df_integrated_missing_new = dp.integrate_data(df_match_miss_new_c, df_match_player_miss_new_c, df_player_sofifa, df_player_fifa_sofifa, prod=True, fifa_not_released_yet=fifa_not_released_yet, export=False) 

            # Concateno missing y old (que puede tener algunos missing ya)
            df_integrated_updated = pd.concat([df_integrated_upd, df_integrated_missing_new], axis=0)
            df_integrated_updated = df_integrated_updated[~df_integrated_updated.index.duplicated(keep='first')]  # El df_integrated tiene missing hasta el dia en que entrené
            logger.warning(f"Concatenación old + missing: {df_integrated_upd.shape} + {df_integrated_missing_new.shape} --> {df_integrated_updated.shape} (si nunca preparaste, missing no se agrega pues ya está)")

            # Guardo registro de todos los partidos missing juntos (los recien recolectados y los que ya tenia)
            df_integrated_missing_all = pd.concat([df_integrated_missing, df_integrated_missing_new], axis=0)
            
            if export:
                if extract_missing:    
                    # Exporto datos extraidos una vez que la integracion funcionó (sino lo extrae pero no lo integra) --> # Mucho cuidado si falla la preparacion pues los missing estaran en old_updated pero no integrados correctamente. (deberias exportar si la prep funciona o algo asi)
                    mis.concat_with_missing_already_extracted(df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new, df_match_miss, df_match_player_miss, df_match_odds_miss)  # missing all --> NO HACERLO CUANDO SOLO QUIERO PREPARAR... Deberia evitar que concatene si los partidos missing ya estan...
                    mis.concat_old_with_missing(df_match_upd, df_match_player_upd, df_match_odds_upd, df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new) # Old + missing # # No lo quiero cuando ya extraje missing y solo quiero preparar...
                df_integrated_missing_new.to_excel(f'{mis.BASE_DIR_MISSING_DP}/df_integrated_missing.xlsx', index=True)
                df_integrated_updated.to_excel(f'{mis.BASE_DIR_MISSING_AND_OLD}/df_integrated.xlsx', index=True)
                df_integrated_missing_all.to_excel(f'{mis.BASE_DIR_MISSING_ALL_dp}/df_integrated_missing.xlsx', index=True)
  
        else:
            df_integrated_updated = df_integrated_upd.copy()
            logger.warning(f"Ya se habian extriado todos los partidos missing. Aun no hay partidos nuevos. {df_integrated_updated.shape}")
       
    else:
        df_integrated_updated = df_integrated_upd.copy()
        logger.warning(f"Se evito por comando la extraccion de missing. Levanto integrated ya concatenado {df_integrated_updated.shape}...")

    # Si solo queria missing, cortar.
    if not d_run['data_unders'] and not d_run['data_prep'] and not d_run['modeling']:
        return None

    # _____________________________________________________________ DATA UNDERSTANDING _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "DATA UNDERSTANDING".center(120) + "\n" + "+"*120 + "\n")
    if d_run['data_unders']:
        # Extriago datos de los partidos en los proximos dias
        df_match, df_match_player, df_match_odds = du.collect_initial_data_new(l_competencies=comp_public, df_comp_country=df_comp_country, n_days=n_days_max_next_matches)
    
         # Si no hay proximos partidos
        if len(df_match) == 0:
            # Evito preparacion y modelado
            logger.error("No hay próximos partidos para los cuales predecir su resultado.")
            return ValueError

        du.describe_data_new(df_match, df_match_player, df_match_odds, verbose=0)            

    else:
        logger.warning("Se evitó por comando la extraccion de proximos partidos.")

        if predict_missing:

            logger.info("Uso los partidos df_match MISSING ya extraidos.")
            df_match, df_match_player, df_match_odds = df_match_miss.copy(), df_match_player_miss.copy(), df_match_odds_miss.copy()
            df_match = df_match[df_match['id_competition'].isin(comp_public)]
 
            # Levanto df_integrated de cuando entrené modelos (tmb los que use en test...)
            df_integrated_train = pd.read_excel(f'data/{country}/p3_data_preparation/{iteration_date}/df_integrated.xlsx', index_col=0) 
            
            # Podria volver a predecir como prox partido un partido de test (con el que entrené) ? --> Para comparar probas del = partido en test y prod. Son muy similares :
            if date_missing is not None:
                len_inic = len(df_integrated_train)
                date_missing = pd.to_datetime()
                df_integrated_train = df_integrated_train[df_integrated_train['date'] <= date_missing]
                logger.info(f'Tras seleccionar partidos anteriores a {date_missing}: {len_inic} --> {len(df_integrated_train)}') 
            
            # Selecciono los missing con los que no se entrenó
            df_match = df_match[~df_match.index.isin(df_integrated_train.index)]
            df_match_player = df_match_player[~df_match_player.index.isin(df_match.index)]
            df_match_odds = df_match_odds[~df_match_odds.index.isin(df_match.index)]
            print("Shapes df_missing:", df_match.shape, df_match_player.shape, df_match_odds.shape)

            # Elimino los partidos missing de df_integrated_updated para evitar duplicated labels
            print(df_integrated_updated.shape)
            df_integrated_updated = df_integrated_updated[~df_integrated_updated.index.isin(df_match.index)]
            print(df_integrated_updated.shape)

        else:
            # Levanto datos de proximos partidos ya extraidos
            logger.info("Uso los partidos df_match_next ya extraidos.")
            BASE_DIR_NEXT_MATCHES = f'./data/{country}/p6_deployment/data_understanding'
            df_match = pd.read_excel(f'{BASE_DIR_NEXT_MATCHES}/df_match_next.xlsx', index_col=0)
            df_match_player = pd.read_excel(f'{BASE_DIR_NEXT_MATCHES}/df_match_player_next.xlsx', index_col=0)
            df_match_odds = pd.read_excel(f'{BASE_DIR_NEXT_MATCHES}/df_match_next_odds.xlsx', index_col=0)
    
    # Error por intentar predecir al menos un partido que ya se jugó y tenes en missing
    rows_rep = df_match[df_match.index.isin(df_integrated_updated.index)] # df_integrated_udated
    if len(rows_rep) > 0:
        logger.warning(f"IDXS REPETIDOS: {len(rows_rep)}")

        logger.error("En caso que los proximos partidos ya esten en df_old_last_matches (o sea, los partidos ya se jugaron y los recolectaste como missing, tirara error al momento de predecir por indice repetido.)")
        user_input = str(input("Escribe 'y' para eliminar indices duplicados y seguir la prediccion: "))

        if user_input == 'y':

            user_input_2 = str(input("Escribe 'y' para predecir solo los proximos partidos: "))
            if user_input_2 == 'y':
                # Op1)  Elimino repetidos de proximos partidos a predecir (y no de df_int_updated..) --> no te quedan missing.
                # No podes usar esto para comparar con assess. La diferencia puede ser grande, sobrotodo si extrajiste varios dias antes del partido.
                logger.info(df_match.shape)
                df_match = df_match[~df_match.index.isin(rows_rep.index)]
                logger.info(df_match.shape)
            else:
                # Podria volver a cierta fecha y predecir como si fueran proximos partidos. El unico problema es que levanta partidos con cuotas desactualizadas. no se si algo mas...
                # Op2) Elimino repetidos de los partidos viejos para predecir los ya jugados tmb
                print(df_integrated_updated.shape)
                df_integrated_updated = df_integrated_updated[~df_integrated_updated.index.isin(rows_rep.index)]
                print(df_integrated_updated.shape)
        else:
            logger.error("Se cortó la prediccion.")
            raise KeyError
        
    # En caso que no haya proximos partidos para predecir, no los preparo.
    elif len(df_match) == 0:
        return pd.DataFrame()

    # _____________________________________________________________ DATA PREPARATION _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "DATA PREPARATION".center(120) + "\n" + "+"*120 + "\n")
    if d_run['data_prep']:

        # Determino n_model, iteration date y nombre --> Lo uso para levantar hiper no solo en modeling sino tmb en data prep.
        n_model, model_name = read_data_of_best_model(id_country, d_model)

        # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
        lo = TrainingDataLoader(country=country, n_model=n_model, model_name=model_name, iteration_date=iteration_date_dt)
        d_hiper = lo.load_data_preparation_hyperparameters()
        df_etiquetas = lo.load_df_etiquetas()
        scaler, columns_scaled = lo.load_scaler_model()

        # Definir rango de fechas 
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        initial_date = df_match['date'].min()  # Obtiene la fecha mínima (para filtrar dfs para rellenar y construir) 
        logger.info(f"📅 Fecha inicial: {initial_date}.")

        # Preparacion de datos hasta integrate
        if not predict_missing:

            # Deberia levantar los datos de sofifa tal como cuando entrené.... ?
            # ...

            # Format a integrate (usar lo mismo que en df_int_missing)
            df_match, df_match_player, df_match_odds, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_match_odds, df_player_fifa_sofifa, reformat=False, prod=True, export=False)            
            df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, export=False)
            df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa = dp.verify_format(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, prod=True)
            df = dp.integrate_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, prod=True, export=False) 
        
        else:
            # Forma 2: desde int_missing 
            df = df_integrated_missing.copy()
            df = df[df.index.isin(df_match.index)]

        # Clean post integrate
        ## Next matches
        df = dp.clean_post_integrate(df, competencies_to_select=d_hiper['comp_to_select'], prod=True)

        ## Old matches (con los cuales construir var historicas) -->         # Filtrar df_integrated con las competencias seleccionadas (ya hago fill solo de las cols players que no elimino por ruido)
        df_integrated_updated = dp.clean_post_integrate(
            df_integrated_updated, n_years_to_select=None, competencies_to_select=d_hiper['comp_to_select'], prod=True
        )

        ### Filter rows to construct
        df_constructed_train = lo.load_df_constructed()
        df_integrated_updated_clean = dp.filter_rows_to_construct(df_integrated_updated, df_constructed_train, predict_missing)

        # Fill data (si es missing no pues ya deberia tener las formaciones)
        if not predict_missing: 
            logger.info("Fill data...")
            ### Selecciono los ultimos partidos de los ya jugados para rellenar
            logger.info(f"Seleccion de ultimos partidos (last {n_days_fill_data} dias) para rellenar formaciones...")
            df_last_old_matches_fill = filter_dataframe_by_date(df=df_integrated_updated_clean, initial_date=initial_date, n_days=n_days_fill_data) # Los parates pueden ser de 3 meses o mas. Por eso tomo 5 meses para tener un poco de margen de seguridad.
            
            ### Relleno datos
            df, df_c1, df_c2 = dp.fill_data_not_available_yet(df, df_last_old_matches_fill)

        ## Construct
        df = dp.construct_data_new(
            df_next_matches=df, 
            df_last_old_matches=df_integrated_updated_clean,
            n_last_matches=d_hiper['n_last_matches'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'], 
            calculate_dif=d_hiper['calculate_dif'], decay_rate=d_hiper['decay_rate'], columns_used=d_hiper['selected_columns']
            )
        
        ## Clean data post construct
        df = dp.clean_post_construct(df=df, prod=True)
        
        ## Tag
        df, _ = dp.tag_string_data_to_integer(df, df_etiquetas, prod=True)
        
        ## Select --> Selecciono las variables que necesita el modelo
        n_col_inic = len(df.columns)
        df = df[columns_scaled] # Es igual a d_hiper['selected_columns'] + "result" --> pero necesito el mismo orden de las cols exacto que cuando entrené el scaler...
        df = df.drop(columns=['result'])
        if verbose >= 1:
            logger.info(f"Columnas luego de filtrar x mas importantes: {n_col_inic} --> {len(df.columns)}")

        ## Clean data post select'
        dp.describe_and_verify_nan(df=df)
        df = dp.clean_post_select(df, scaler_loaded=scaler, prod=True)

        # Exporto datos
        df.to_excel(f"data/{country}/p6_deployment/data_preparation/df_to_predict.xlsx", index=True)

        # Mensajes antes de predecir
        logger.info(f"Shape Dataframe antes de Modeling(): {df.shape}")
        if len(df) == 0:
            logger.warning("Se evitó seguir la preparacion luego de clean_data puesto que no hay partidos para la competencia.")
            return df
        
        if len(df_match) != len(df):
            logger.warning(f"\nDe los {len(df_match)} proximos partidos, quedan {len(df)} luego de la preparacion")

    else:
        if d_run['modeling']:
            logger.warning("Se evitó por comando la preparacion de proximos partidos.")
            # Levanto dataset para prueba
            df = pd.read_excel(f'./data/{country}/p6_deployment/data_preparation/df_selected_nan.xlsx', index_col=0)
            print(df.head(2), df.shape)

            df_match = pd.read_excel(f'data/{country}/p6_deployment/data_understanding/df_match_next.xlsx', index_col=0)
            df_match_odds = pd.read_excel(f'./data/{country}/p6_deployment/data_understanding/df_match_next_odds.xlsx', index_col=0)
            df_c1 = pd.read_excel(f'./data/{country}/p6_deployment/data_preparation/fill_data/df_copiado_formaciones.xlsx', index_col=0)
            df_c2 = pd.read_excel(f'./data/{country}/p6_deployment/data_preparation/fill_data/df_copiado_ref_and_coaches.xlsx', index_col=0)
            df_fill = pd.read_excel(f'./data/{country}/p6_deployment/data_preparation/df_filled.xlsx', index_col=0)
  
    #_____________________________________________________________ MODELING _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "MODELING".center(120) + "\n" + "+"*120 + "\n")
    if d_run['modeling']:
        logger.critical(f"n_model: {n_model} model_name: {model_name} iteration_date: {iteration_date}")
        df_fill = pd.read_excel(f'./data/{country}/p6_deployment/data_preparation/df_filled.xlsx', index_col=0)

        # Predigo con modelo cargado
        if predict_missing:
            df_filled = df_fill.copy() 
        else:
            df_filled = pd.concat([df_c1['copiado_formaciones'], df_fill.loc[:, ['player_emergency_fill', 'emergency_fill']]], axis=1) 

        # Predigo sobre proximos partidos usando modelo cargado
        y_pred_proba, y_pred = mo.predict_model(model=lo.load_model(), X_test=df)

        if predict_missing:
            df_match = df_match_miss.copy()
            df_match_odds = df_match_odds_miss.copy()

        # Guardo como df
        df_pred_proba = mo.construct_predictions_dataframe(model=lo.load_model(), X_test=df, y_pred_prob=y_pred_proba, y_pred=y_pred)
        df_predicciones = mo.prepare_dataframe_to_assess_with_roi(df_pred_proba=df_pred_proba, df_match=df_match, df_match_odds=df_match_odds)         # Concateno todos los dfs en uno solo 
        df_predicciones = pd.concat([df_predicciones, df_filled], axis=1)

        # Aplico estrategia de apuesta
        bs = betting_strategy.BettingStrategy(country=country, iteration_date=iteration_date_dt)

        # Pasarle "strategy" prod o bien ya pasarle el d_params...
        d_m = {'england': 10, 'france': 5, 'germany': 10, 'italy': 10, 'spain': 5}
        d_strategy = {'prob_dp': None, 'curva': 'kelly_linear', 'm': d_m[country], 'b': 0, 'k': 1}
        if isinstance(d_strategy, dict):
            logger.warning("Aplico MISMA estrategia A TODOS LOS RDOS. ")
            df = bs.apply_strategy(df_predicciones, param_dict=d_strategy)

        else:
            logger.warning("Aplico estrategia DISTINTA POR RESULTADO. ")
            df = bs.apply_strategy_by_result(df_predicciones, df_hiper=d_strategy)
            
        if export:
            df.to_excel(f'./data/{country}/p6_deployment/predicciones.xlsx', index=True)

        logger.critical("LA PREDICCION FUE UN EXITO!")

    else:
        logger.warning("Se evitó por comando la predicción de proximos partidos.")
        return pd.DataFrame()

    end = time.time()
    logger.info(f"Main_next_matches en {(end - start)/60:.1f} minutos \n\n")
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":    
    directorio = os.getenv('BASE_DIR_LOCAL')

    # Defino country
    d_countries = {
        48: ["england", '2025-08-18'], 
        55: ["france", '2025-08-18'], 
        59: ["germany", '2025-08-18'], 
        77: ["italy", '2025-08-19'],
        148: ["spain", '2025-08-19'], 
        167: ["usa", '2025-08-14'], 
        }

    id_country = 48
    key, value = 'predict', 'next_matches'
    n_days = 7

    # iteration date y modelo
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]
    # d_model = {'n_model': 24, 'model_name': "LogisticRegression"} # LogisticRegression


    d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
    df = main(d_run, id_country, iteration_date=iteration_date, n_days_max_next_matches=n_days, export=True, country=country)  #d_model=d_model


    if isinstance(df, pd.DataFrame):
        df.to_excel(f"{directorio}/predicciones.xlsx")