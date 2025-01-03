# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
from utils.set_up_logging import logger
import re
import os
import json
from dotenv import load_dotenv
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_next_matches, extract_data
from p2_data_understanding import describe_data
## Data preparation
from main import DataPreparation, Modeling
from p3_data_preparation import format_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
# Modeling
from p4_modeling import asses_model, betting_strategy
import pickle
import joblib


class DataUnderstandingNew():

    def __init__(self, id_country, country, export: bool = True):
        self.id_country = id_country
        self.country = country.lower()
        self.export = export
        self.make_directories()

    def make_directories(self):
        l_directorios = [
            f'./data/{self.country}/p6_deployment/data_understanding',
            f'./data/{self.country}/p6_deployment/missing/data_understanding/all',
            f'./data/{self.country}/p6_deployment/missing/old_updated',   
            f'./data/{self.country}/p6_deployment/missing/data_preparation/all',
        ]
    
        if self.export:
            for directorio in l_directorios:
                if not os.path.exists(directorio):
                    # Si no existe, crear el directorio
                    os.makedirs(directorio)

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
        for df in [df_match_concat, df_match_player_concat, df_match_odds_concat]:
            self.verify_data_quality(df)

        # Exporto datasets
        if self.export:
            df_match_concat.to_excel(f'./data/{self.country}/p6_deployment/data_understanding/df_match_next.xlsx', index=True)
            df_match_player_concat.to_excel(f'./data/{self.country}/p6_deployment/data_understanding/df_match_player_next.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./data/{self.country}/p6_deployment/data_understanding/df_match_next_odds.xlsx', index=True)

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

            # Obtengo nombre de competicion y is_cup
            df_comp_filt = df_comp_country[df_comp_country['id_competition'] == id_competition]  # Para extrar varios countryes?: df = df_comp[df_comp['country'].isin(l_countryes)]
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
        for df in [df_match_concat, df_match_player_concat, df_match_odds_concat]:
            self.verify_data_quality(df)

        # Exporto datasets
        if self.export:
            df_match_concat.to_excel(f'./data/{self.country}/p6_deployment/missing/data_understanding/df_match_miss.xlsx', index=True)
            df_match_player_concat.to_excel(f'./data/{self.country}/p6_deployment/missing/data_understanding/df_match_player_miss.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./data/{self.country}/p6_deployment/missing/data_understanding/df_match_odds_miss.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat

    def verify_data_quality(self, df, raise_error:bool = False):
        """
        Verificacion de que dataframe extraido tiene al menos una fila y columna.
        """
        if len(df) == 0:
            logger.warning("El dataframe no tiene registros")
            if raise_error:
                raise ValueError("El dataframe no tiene registros")
        elif len(df.columns) == 0:
            logger.warning("El dataframe no tiene columnas")
            if raise_error:
                raise ValueError("El dataframe no tiene columnas")
            
    def describe_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame, verbose: int = 1):
        """
        Descripción de dataframes en terminos de dtypes, nan values, registros unicos, etc.
        """
        logger.info("Describing data... ")
        logger.info("DF_MATCH")
        describe_data.getting_to_know_data(df_match, verbose=verbose)
        describe_data.verificar_unicidad_registros(df_match) # Verifico unicidad de registros segun campos id

        logger.info("DF_MATCH_PLAYER")
        describe_data.getting_to_know_data(df_match_player, verbose=verbose)

        logger.info("DF_MATCH_ODDS")
        describe_data.getting_to_know_data(df_match_odds, verbose=verbose)

class DataPreparationNew(DataPreparation):

    def __init__(self, country, export: bool = True):
        self.country = country
        self.BASE_DIR = f"./data/{country}/p6_deployment/data_preparation"
        self.export = export
        self.make_directories()
        super().__init__(country)

    def make_directories(self):  # Pasarle direcotio o l_directorios como argumento...
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

    def format_data_new(self, df_match: pd.DataFrame, df_match_odds: pd.DataFrame, verbose: int = 0):
        """
        Arreglo el data fill_type de algunas variables.

        # Parameters:
            df_match: Dataframe de los datos de los partidos. (DataFrame)
            
        # Returns:
            Dataframe pasado como parametro formateado. (DataFrame)
        """
        start = time.time()
        logger.info("Formating data...")

        # Dataframe partido
        ## Fecha
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        ## Capacity & Attendance
        df_match = format_data.convert_capacity_to_int(df_match)        

        # Convierto columnas a float
        df_match = format_data.convert_columns_to_float(df_match)
        df_match_odds = format_data.convert_columns_to_float(df_match_odds)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if self.export:
            df_match.to_excel(f'{self.BASE_DIR}/format_data/df_match_form.xlsx')
            df_match_odds.to_excel(f'{self.BASE_DIR}/format_data/df_match_odds_form.xlsx')
        return df_match, df_match_odds

    def clean_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, verbose:int = 0):
        """
        Limpieza inicial de los dataframes

        # Parameters:
            df_match:
            df_match_player:
        
        # Returns:
            Dataframe
        """
        start = time.time()
        logger.info("Cleaning new data...")

        # Dataframe match
        if 'attendance' in df_match.columns:
            df_match = df_match.drop(['attendance'], axis=1)

        # Preparo columnas texto
        columns_to_keep = [col for col in df_match.columns if df_match[col].dtype == 'object' and 'id_' not in col]
        df_match = clean_data.prepare_text_columns(df_match, l_cols_to_process=columns_to_keep) # Ver si selecciona bien.. # ['team_home', 'team_away', 'coach_home', 'coach_away', 'venue', 'referee'])
        columns_player_names = list(df_match_player.filter(like='player_name').columns)
        df_match_player = clean_data.prepare_text_columns(df_match_player, l_cols_to_process=columns_player_names)
        df_match = clean_data.clean_teams_names(df_match)  # una vez que ya aplique el lower()

        end = time.time()
        print(f"Limpieza inicial de datos en {(end - start) / 60:.1f} minutos")

        if self.export:
            df_match.to_excel(f'{self.BASE_DIR}/clean_data/df_match_cleaned.xlsx')
            df_match_player.to_excel(f'{self.BASE_DIR}/clean_data/df_match_player_cleaned.xlsx')

        return df_match, df_match_player

    def integrate_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, verbose: int = 0):
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        :param df_match: Dataframe de los datos de los partidos.
        :param df_match_player: Dataframe de los datos de los jugadores en cada partido.
        :param df_player: Dataframe de los datos de los jugadores.
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de
        lo contrario. (bool)

        :return: Dataframe integrado. (DataFrame)

        Podria guardar el nuevo mapeo para jugadores nuevo o no hace falta? No hace falta creo.
        """
        # Uso integracion de main.py
        df = self.integrate_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) 

        if self.export:
            df.to_excel(f'{self.BASE_DIR}/df_integrated.xlsx')
        return df

    def fill_data_not_available_yet(self, df_next_matches: pd.DataFrame, df_last_old_matches: pd.DataFrame, comp_to_select=list, verbose: int = 0):
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

        # Filtro df_last_old_matches con competencias a predecir...
        n_inic = df_last_old_matches.shape
        df_last_old_matches = df_last_old_matches[df_last_old_matches['id_competition'].isin(comp_to_select)]  # Para rellenar solo con los partidos de la misma liga...
        print(n_inic, df_last_old_matches.shape)

        # En caso que aun no se cuente con las formaciones, asigno promedio en ultimos partidos
        l_player_cols  = [col for col in df_last_old_matches.columns if ('player_start' in col) or ('player_sub' in col)]  # Selecciono las variables que corresponden a jugadores
        df_next_matches, df_copiado_formaciones = clean_data.fillna_with_mean_in_last_matches_with_df(df_to_fill=df_next_matches, df=df_last_old_matches, cols_to_fill=l_player_cols)            

        # Copio valores en ultimos partidos (deberia copiar solo referee y coaches)
        miss_player_columns = [col for col in df_last_old_matches.columns if ('player_miss' in col)]  # --> ojo porque no se si las rellena ok... es complejo el rellenado.
        l_var_to_copy = miss_player_columns # ['id_coach_home', 'id_coach_away'] + 
        df_next_matches, df_copiado = self.fillna_with_last_match_value(df_next_matches, df_last_old_matches, cols_to_fill=l_var_to_copy) 
        if self.export:
            df_copiado_formaciones.to_excel(f"{self.BASE_DIR}/fill_data/df_copiado_formaciones.xlsx", index=True)
            # df_copiado.to_excel(f"{self.BASE_DIR}/fill_data/df_copiado_ref_and_coaches.xlsx", index=True)
            df_next_matches.to_excel(f"{self.BASE_DIR}/df_filled.xlsx", index=True)

        return df_next_matches, df_copiado_formaciones, df_copiado
    
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
    
    def construct_data_new(self, df_next_matches: pd.DataFrame, df_old_matches, df_last_old_matches, n_days: list, n_years_h2h: int, 
                           segun_localia: bool, columns_used: list, dif_con_against: bool = True, verbose: int = 0):
        """
        Construye nuevos datos a partir de un dataframe existente.

        # Parameters
            df_next_matches: Dataframe con proximos partidos ya integrado. (DataFrame)
            df_old_matches: Dataframe con partidos ya jugados e integrado. (DataFrame)
            df_last_old_matches: Dataframe con los ultimos partidos ya jugados (DataFrame)
            n_days: Número de últimos partidos a considerar para el cálculo de variables. (int)
            n_years_h2h: Numero de años para construir historial entre equipos. (int)
            segun_localia: Construir variables por localia o no. (bool)
        
        # Returns
            Dataframe con proximos partidos construido utilizando los partidos ya jugados. (DataFrame)
        """
        logger.info("Constructing new data...")
        start = time.time()

        # 1) Construyo historial entre si  (podria evitar construirlas si no estan en columns_used...)
        # df_next_matches = self.construct_h2h_next_matches(df_next_matches, df_old_matches, n_years_h2h, columns_used)

        # Si hay "ultimos partidos"
        if len(df_last_old_matches) > 0:            
            # Construyo datos (sin historiales) luego de concatenar proximos partidos (df_next_matches) y los ultimos partidos ya jugados (df_last_old_matches)
            n_rows_inic = len(df_next_matches)
            df_concat_last = pd.concat([df_next_matches, df_last_old_matches], axis=0)
            df_constructed = self.construct_data(df_concat_last, n_days, n_years_h2h, segun_localia=segun_localia, with_h2h=False, dif_con_against=dif_con_against, export=False)
            df_next_matches = df_constructed[df_constructed.index.isin(df_next_matches.index)]  # Separo datos construidos entre los proximos partidos y los ya jugados  # En caso que los proximos aprtidos ya esten en df_old_last_matches (o sea, los partidos ya se jugeron y los recolectaste como missing, tirara error al momento de predecir por indice repetido.)

        # Si no hay "ultimos partidos"
        else:
            # evito construir variables historicas
            logger.warning("Evito construccion de variables historicas debido a la falta de ultimos partidos")
            df_next_matches = self.construct_data(df_next_matches, n_days, n_years_h2h, segun_localia=segun_localia, with_historic=False, verbose=verbose, export=False)

            # Agregar las columnas que faltan ("las que se deberian construir tambien") y rellenar con NaN
            for columna in columns_used: 
                if columna not in df_next_matches.columns:
                    df_next_matches[columna] = np.nan  # no le des valor 0 porque sino las predicciones son 33-33-33.

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")
        
        if self.export:
            df_last_old_matches.to_excel(f'{self.BASE_DIR}/construct_data/df_matches_to_construct_historic_values.xlsx', index=True)
            df_next_matches.to_excel(f'{self.BASE_DIR}/df_constructed.xlsx', index=True)

        return df_next_matches

    def construct_h2h_next_matches(self, df_next_matches, df_old_matches, n_years_h2h, columns_used):
        """
        Construccion de variables historiales para proximos partidos.
        Mejora a hacer: podria evitar la construccion de las variables si no estan en columns_used...
        """
        # Determine years to construct h2h
        df_integrated = pd.read_excel(f'data/{self.country}/p3_data_preparation/df_integrated.xlsx', index_col=0)  # no puedo usar df_old_matches porque tiene missing y la date se estira... (intente usar df_match pero falla n_years)
        n_years = (max(df_integrated['date']) - min(df_integrated['date'])).days / 365  # Determino n_years solo con partidos ya jugados
        n_years = int(-(-n_years // 1)) # redondeo hacia arriba numero de años

        # Construyo columna "result" para poder calcular h2h
        df_old_matches = construct_data.determine_result(df_old_matches, self.var_resp) # Construyo columna resultado en el old para poder calcular historial

        ## Construyo historiales
        df_concat = pd.concat([df_next_matches, df_old_matches], axis=0)
        df = construct_data.h2h_by_date(df_concat, n_years=n_years)
        df = construct_data.h2h_by_date(df, n_years=n_years_h2h)
        df = construct_data.h2h_by_date_by_localia(df, n_years=n_years)
        df = construct_data.h2h_by_date_by_localia(df, n_years=n_years_h2h)

        ## Vuelvo a seleccionar df_next_matches pero con historiales construidos
        columnas_deseadas = list(df_next_matches.columns) + [col for col in df.columns if 'h2h_' in col]  # Reemplazo historiales nan por 0
        df = df[columnas_deseadas]
        df_next_matches = df[df.index.isin(df_next_matches.index)]
        return df_next_matches
        
    def tag_string_data_to_integer_new(self, df: pd.DataFrame, df_etiquetas_loaded, verbose: int = 0):
        """
        Utilizando las mismas etiquetas que cuando se entreno el modelo para el pais, convierto columnas string a integer
        """
        logger.info("Tagging string data to integer..")

        # Elimino season para no etiquetarla?
        df = df.drop(['season'], axis=1)

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int_already_tagged(df, df_etiquetas_loaded, verbose=2)  # Si o si tengo que devolver df_etiquetas?
    
        if self.export:
            df.to_excel(f'{self.BASE_DIR}/df_tagged.xlsx', index=True) 
            df_etiquetas.to_excel(f'{self.BASE_DIR}/df_etiquetas_actualizado.xlsx', index=False) 
        return df

    def clean_data_2_new(self, df: pd.DataFrame, scaler_loaded, columns_used, comp_to_select, columns_selected, verbose: int = 0):
        """
        Filtrado por competencias y escalado de datos.
        """
        logger.info("Cleaning data 2...")

        # Reemplazo infinitos
        df = clean_data.replace_infinite(df)

        # Para evitar ciertas competencias
        n_reg_inic, n_col_inic= len(df), len(df.columns)
        df = df[df['id_competition'].isin(comp_to_select)]
        
        if verbose >= 1:
            logger.info(f"Filas luego de filtrar x competencia: {n_reg_inic} --> {len(df)}")

        # Selecciono las mismas caracteristicas con las que entrene el scaler (sino, falla)
        try:
            df = df.loc[:, columns_used]

            if verbose >= 1:
                logger.info(f"Columnas luego de filtrar x columnas scaled: {n_col_inic} --> {len(df.columns)}")

        except KeyError as e:
            logger.error("Las columnas del scaler no coinciden con las de los proximos partidos.")
            
            if verbose >= 1:
                logger.info(df)
                logger.info(df.shape)

            raise e
        
        # Tratamiento de NaN values (antes de escalar para que el 0 realmente sea 0.)
        df, df_emergency_fill = self.treat_nan_values_new(df=df, columns_selected=columns_selected)

        # Transforma los nuevos datos de predicción utilizando el StandardScaler cargado
        try: 
            X_scaled = scaler_loaded.transform(df)
            X_scaled_df = pd.DataFrame(X_scaled, columns=columns_used, index=df.index)

            if verbose >= 1:
                logger.critical("El escalado fue un exito!")
                
        except ValueError as e: # Found array with 0 sample(s) (shape=(0, 47)) while a minimum of 1 is required by StandardScaler.
            logger.error(f"El escalado tuvo un error: {e}")
            return pd.DataFrame()
        
        return X_scaled_df, df_emergency_fill

    def treat_nan_values_new(self, df: pd.DataFrame, columns_selected, verbose: int = 0):
        """
        Remoción de valores NaN 

        # Parameters:
            df: Dataframe al cual remover NaN values. (DataFrame)
            columns_selected: Listado de columnas seleccioandas para usar en produccion. (list)
        
        # Returns:
            df: Dataframe pasado como parametro sin registros con al menos un NaN value. (DataFrame)
        """
        logger.info("Treating NaN values...")
        df_filled_columns = pd.DataFrame(0, index=df.index, columns=['emergency_fill', 'player_emergency_fill']) # Inicializo el df

        # Rellenar NaN en algunas columnas espeecificas
        columns_to_fill = [col for col in df.columns if df[col].isna().any()]  # En teoria, solo rellena las variables historicas que son nan.

        # Determino que  variables rellena Y que fueron seleccionadas (podria rellenarla y no ser seleccionada)
        columns_to_fill_sel =  [col for col in columns_to_fill if col in columns_selected]
        if self.verbose >= 1:
            logger.info(f'Nº columnas a rellenar: {len(columns_to_fill)}')
            logger.info(f'Nº columnas a rellenar (que fueron seleccionadas): {len(columns_to_fill_sel)}')
            logger.info(f'Nº columnas seleccionadas: {len(columns_selected)}')

        if columns_to_fill:
            # Crear una copia del DataFrame y rellenar los NaN
            df_copy = df.copy()
            df_copy[columns_to_fill] = df_copy[columns_to_fill].fillna(0)

            # Identificar filas donde se rellenaron NaN
            filled_rows = (df[columns_to_fill_sel].isna() & (df_copy[columns_to_fill_sel] == 0)).any(axis=1)

            # Crear DataFrame con las columnas rellenadas y `emergency_fill`
            df_filled_columns = df_copy.loc[filled_rows, columns_to_fill_sel]
            df_filled_columns['emergency_fill'] = 1

            # Columna con valor 0 o 1 según si se rellenaron columnas con "player_start" o "player_sub"
            player_columns = [col for col in columns_to_fill_sel if "player_start" in col or "player_sub" in col]
            df_filled_columns['player_emergency_fill'] = (
                (df[player_columns].isna() & (df_copy[player_columns] == 0)).any(axis=1).astype(int)
            )

            # Calcular cuántas columnas se rellenaron de emergencia para cada fila
            df_filled_columns['n_cols_filled'] = (
                (df[columns_to_fill].isna() & (df_copy[columns_to_fill_sel] == 0)).sum(axis=1)
            )

            # Calcular el porcentaje de columnas rellenadas de emergencia para cada fila --> es ANTES de seleccionar las columnas... TAl vez ni siquiera usas esas columnas rellenadas.
            df_filled_columns['perc_cols_filled'] = (
                df_filled_columns['n_cols_filled'] / len(columns_selected) * 100
            )

            # Crear una columna con el listado de columnas rellenadas por cada registro
            try:
                df_filled_columns['l_col_filled'] = df[columns_to_fill_sel].apply(
                    lambda row: [col for col in columns_to_fill_sel if pd.isna(row[col]) and df_copy.at[row.name, col] == 0], axis=1
                )
            except ValueError: # ValueError: Length of values (0) does not match length of index (12)
                pass
                
            if self.export:
                df_filled_columns.to_excel(f'{self.BASE_DIR}/df_filled_columns.xlsx', index=True)

            # Calcular y mostrar el porcentaje de NaN por cada columna
            for col in columns_to_fill:
                nan_percentage = df[col].isna().mean() * 100

                if verbose >= 1:
                    logger.warning(f"Columna '{col}' tiene {nan_percentage:.1f}% de valores NaN.")

            # Actualizar el DataFrame original
            df = df_copy

        # Elimino partidos con al menos un NaN value --> Tal vez lo deberia poner al ppio para imprimir warning de cuantos partidos eliminaria...
        df_sin_dup = df.dropna()
        if verbose >= 1 and (len(df) != len(df_sin_dup)):
            logger.warning(f"De los {len(df)} partidos, no se hará la prediccion para {len(df)-len(df_sin_dup)} partidos puesto que tienen al menos un valor NaN y el modelo no puede tener input NaN.")

        if self.export: 
            df_sin_dup.to_excel(f'{self.BASE_DIR}/df_selected_nan.xlsx', index=True)

        return df_sin_dup, df_filled_columns
        
    def select_data_new(self, df: pd.DataFrame, l_columns: list, verbose: int = 0):
        """
        Selecciona las variables que necesita el modelo ya entrenado.

        # Parameters:
            df: Dataframe con los proximos partidos. (DataFrame)
            l_columns: Lista de columnas a seleccionar. (list)
        
        # Returns
            Dataframe con las variables seleccionadas. (DataFrame)
        """
        logger.info("Selecting data...")
        
        # Selecciono las variables que necesita el modelo
        n_col_inic = len(df.columns)
        df = df[l_columns]

        if verbose >= 1:
            logger.info(f"Columnas luego de filtrar x mas importantes: {n_col_inic} --> {len(df.columns)}")

        if self.export:
            df.to_excel(f'{self.BASE_DIR}/df_selected.xlsx', index=True)
        return df

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
        # self.BASE_DIR_ALL_MISSING = f'./data/{self.country}/p6_deployment/missing/data_understanding/all'
        # self.BASE_DIR_NEXT_MATCHES = f'./data/{self.country}/p6_deployment/data_understanding'
        self.BASE_DIR_mod = f"./data/{self.country}/p4_modeling/{self.iteration_date}"
        self.BASE_DIR_dp = f"{self.BASE_DIR_mod}/p3_data_preparation/"
    
    # Data preparation
    def load_data_preparation_hyperparameters(self):
        """
        Cargo hiperparametros de DataPreparation()
        """
        d = {}

        # Si se levanta de main_find_best_hyper.py
        if self.n_model is not None:
            
            df_iteration = pd.read_excel(f"{self.BASE_DIR_mod}/df_iteration_train.xlsx")
            
            # Selecciono la primera. Hay una por modelo entrenado pero los hiper son =.
            row_hiper = df_iteration[df_iteration['n_iteration'] == self.n_model].iloc[0]  
            # logger.info(row_hiper)
            
            ## Levanto columnas utilizadas para entrenar el modelo
            # selected_columns = eval(row_hiper['X_columns'].values[0])  # eval() para pasar de string a lista
            selected_columns = eval(row_hiper['X_columns'])  # eval() para pasar de string a lista
        # Si se levanta de main.py
        else:
            row_hiper = pd.read_excel(f'./data/{self.country}/p3_data_preparation/df_hiper_prep.xlsx')

            ## Levanto columnas utilizadas para entrenar el modelo
            df_selected = pd.read_excel(f'./data/{self.country}/p3_data_preparation/df_selected.xlsx', index_col=0)
            df_selected = df_selected.drop(['result'], axis=1)
            selected_columns = list(df_selected.columns)
            logger.error("Se levantan los hiperparametros de Data Preparation desde de main.py")

        # Guardo hiperparametros en diccionario
        ## Construct_data
        d['n_dias_ult_part'] = eval(row_hiper['n_dias_ult_part']) # eval(row_hiper['n_dias_ult_part'].values[0])
        d['n_years_h2h'] = int(row_hiper['n_anios_hist']) # .values[0]
        d['segun_localia'] = row_hiper['segun_localia'] # .values[0]
        d['dif_con_against'] = row_hiper['dif_con_against'] # .values[0]
        ## Clean_data_2
        n_years_to_select = row_hiper['n_years_to_select'] # .values[0]
        d['n_years_to_select'] = None if pd.isna(n_years_to_select) else int(n_years_to_select) # Si n_years_to_select es NaN, lo paso de np.nan a None
        d['comp_to_select'] = row_hiper['comp_to_select'] # .values[0]
        ## Select_data
        d['selected_columns'] = selected_columns

        if self.verbose >= 0:
            logger.info("Hiperparametros cargados:")
            for key, value in d.items():
                logger.info(f'\t {key}: {value}')
        return d

    def load_df_etiquetas(self, d):

        logger.info("Levento etiquetas con el que entrené")
        n_ult_part, n_years_h2h, segun_localia, dif_con_against = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia'], d['dif_con_against']
        path_tag = f'{self.BASE_DIR_dp}/df_etiquetas_{n_ult_part}_{n_years_h2h}_{segun_localia}_{dif_con_against}.xlsx'       
        df_etiquetas = pd.read_excel(path_tag, index_col=0)

        '''
        # Si se levanta de find_best_hyper.py
        if self.n_model is not None:
            n_ult_part, n_years_h2h, segun_localia, dif_con_against = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia'], d['dif_con_against']
            path_tag = f'{self.BASE_DIR_dp}/df_etiquetas_{n_ult_part}_{n_years_h2h}_{segun_localia}_{dif_con_against}.xlsx'       
        # Si se levanta de main.py
        else:
            path_tag = f"./data/{country}/p3_data_preparation/df_etiquetas.xlsx"
        '''
        if self.verbose >= 1:  
            print(df_etiquetas.head(3))

        return df_etiquetas

    def load_scaler_model(self, d):
        """
        Levanto modelo utilizado en entrenamiento para escalar datos
        """
        n_ult_part, n_years_h2h, segun_localia, dif_con_against, n_years_sel, comp = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia'], d['dif_con_against'], d['n_years_to_select'], d['comp_to_select']
        path_scaler = f'{self.BASE_DIR_dp}/scaler_model_{n_ult_part}_{n_years_h2h}_{segun_localia}_{dif_con_against}_{n_years_sel}_{comp}.pkl'

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
            path_model_2 =  f"{self.BASE_DIR_mod}/models/{self.n_model}_model.pkl"
        # Si se levanta de main.py
        else:
            logger.error("Se levanta el modelo y el scaler desde de main.py")
            path_model_1 = f"./data/{self.country}/p4_modeling/modelo.pkl"
        
        try:
            loaded_model = pickle.load(open(path_model_1, "rb"))
        except FileNotFoundError:
            loaded_model = pickle.load(open(path_model_2, "rb"))

        return loaded_model

    def load_modeling_hyperparameters(self):
        """
        Cargo hiperparametros de Modeling()

        Mejora: 
         - Levantar parametros por resultado...
        """
        d = {}
        l_curvas = ['linear', 'kelly']

        # Estrategia por resultado
        try:
            df_hiper = pd.read_excel(f"{self.BASE_DIR_mod}/best_model/3_bet_strategy/df_strategy_{self.n_model}_{self.model_name}.xlsx", index_col=0)  # desde que separé estrategia de apuesta de entrenamiento...
            print(df_hiper)

            # Si es por resultado
            if len(df_hiper) == 3:
                logger.critical("Se levantó la estrategia de apuesta por resultado")
                return df_hiper

            df_iteration = df_iteration.reset_index()  # Convierte el índice en una columna
            df_iteration.rename(columns={'n_model': 'n_iteration'}, inplace=True)  # Renombra la columna creada

        # Misma Estrategia para los resultados
        except FileNotFoundError:
            logger.warning("Falló la obtencion de hiperparametros de estrategia de apuesta")
            try:
                logger.warning("No está df_iteration_with_strategy")
                df_iteration = pd.read_excel(f"{self.BASE_DIR_mod}/df_iteration.xlsx")  # index_col=0 (ya no lo uso?)

            except FileNotFoundError:
                logger.warning("No está df_iteration_with_strategy ni df_iteration")
                df_iteration = pd.read_excel(f"{self.BASE_DIR_mod}/df_iteration_test.xlsx")

            row_ite = df_iteration[df_iteration['n_iteration'] == self.n_model]

            if self.verbose >= 2:
                logger.info(df_iteration)
                logger.info(row_ite)

            if len(row_ite) > 1:
                col_model = 'model_name' if 'model_name' in df_iteration.columns else 'model_name_test'
                row_hiper_bet_strat = row_ite[row_ite[col_model] == self.model_name]
                logger.warning(len(row_hiper_bet_strat))
            else:
                row_hiper_bet_strat = row_ite

            # Verificacion de que se selecciono una sola row
            if self.verbose >= 0 and len(row_hiper_bet_strat) != 1:
                logger.error(f"Falló la carga de hiperparametros de la estrategia de apuesta. El modelo a cargar era el {self.n_model}, un {self.model_name}. La fila con los hiperparametros tiene un largo de {len(row_hiper_bet_strat)} (≠ de 1:).")
                logger.info(df_iteration)
                raise ValueError
  
        try:
            d['prob_dp'] = float(self.load_value_from_series(row_hiper_bet_strat, 'thr_prob_min_best'))
        except:
            d['prob_dp'] = float(self.load_value_from_series(row_hiper_bet_strat, 'thr_prob_min'))

        d['curva'] = self.load_value_from_series(row_hiper_bet_strat, 'curva') # str
        param1 = self.load_value_from_series(row_hiper_bet_strat, 'param1')  # int?
        param2 = self.load_value_from_series(row_hiper_bet_strat, 'param2') 
        d['m'] = param1 if d['curva'] in l_curvas else None
        d['b'] = param2 if d['curva'] in l_curvas else None
        d['curva_p1'] = eval(str(param1)) if d['curva'] not in l_curvas else None
        d['curva_p2'] = eval(str(param2)) if d['curva'] not in l_curvas else None
        d['odd_weight'] = int(self.load_value_from_series(row_hiper_bet_strat, 'odd_weight'))
        d['lim_sup'] = float(self.load_value_from_series(row_hiper_bet_strat, 'dif_prob_sup_cap'))
        d['normalized'] = self.load_value_from_series(row_hiper_bet_strat, 'normalized')

        if self.verbose >= 0:
            logger.info("Hiperparametros cargados:")
            for key, value in d.items():
                logger.info(f'\t {key}: {value}')
        return d

    def load_value_from_series(self, row, col_name):
        try:
            val = row[col_name].values[0]
        except IndexError:
            val = row[col_name]
        
        if self.verbose >= 2:
            logger.warning(f'{col_name}: {val}')
        return val

# Missing data
def load_last_version_extracted_matches(country):
    """
    No lo pongo en la clase puesto que no son datos usados durante el entrenamiento.
    """
    # Esta bien levantar df_integrated aca tambien. para asegurarme que todos los missing estan en df_integrated tambien.
    # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
    try:
        BASE_DIR_MISSING_AND_OLD = f'./data/{country}/p6_deployment/missing/old_updated'
        df_match = pd.read_excel(f'{BASE_DIR_MISSING_AND_OLD}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'{BASE_DIR_MISSING_AND_OLD}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{BASE_DIR_MISSING_AND_OLD}/df_match_odds.xlsx', index_col=0)
        df_integrated = pd.read_excel(f'{BASE_DIR_MISSING_AND_OLD}/df_integrated.xlsx', index_col=0)
        logger.info('Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')

    except FileNotFoundError:
        df_match = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match_odds.xlsx', index_col=0)
        df_integrated = pd.read_excel(f'./data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)
        logger.warning('Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
    
    logger.info(f"Shapes: \t df_match: {df_match.shape} \t df_match_player:{df_match_player.shape} \t df_match_odds: {df_match_odds.shape} \t df_integrated: {df_integrated.shape}")
    return df_match, df_match_player, df_match_odds, df_integrated

def concat_and_export_old_with_missing(df_match, df_match_player, df_match_odds, df_match_miss, df_match_player_miss, df_match_odds_miss, country):
    """
    Exporto datos de partidos con los que entreno el modelo y los partidos missing.
    """
    BASE_PATH=f"./data/{country}/p6_deployment/missing/old_updated"

    len_inicial = len(df_match)
    len_inicial_miss = len(df_match_miss)
    # Concateno old (que puede tener ya algunos missing) y nuevos missing
    df_concat_match = pd.concat([df_match, df_match_miss], axis=0)
    df_concat_match_player = pd.concat([df_match_player, df_match_player_miss], axis=0)
    df_concat_match_odds = pd.concat([df_match_odds, df_match_odds_miss], axis=0)
    len_final = len(df_concat_match)
    logger.warning(f"Old:{df_match.shape} + Missing: {df_match_miss.shape} = {df_concat_match.shape}")

    verif = (len_inicial + len_inicial_miss) == len_final
    if not verif:
        logger.error("Fallo la concatenacion de partidos missing a los datos viejos")

    # Exporto datos
    df_concat_match.to_excel(f'{BASE_PATH}/df_match.xlsx')
    df_concat_match_player.to_excel(f'{BASE_PATH}/df_match_player.xlsx')
    df_concat_match_odds.to_excel(f'{BASE_PATH}/df_match_odds.xlsx')
    logger.info(f"Shape de df_match concatenado con missing:: {len_inicial} --> {len_final}")

def concat_and_export_missing_extracted(df_match_miss, df_match_player_miss, df_match_odds_miss, country):
    """
    Guardo los nuevos partidos missing con los que ya tenia
    """
    BASE_PATH=f"./data/{country}/p6_deployment/missing/data_understanding/all"

    # Si aun no extraje partidos missing
    try: 
        df_match_miss_comp = pd.read_excel(f'{BASE_PATH}/df_match_miss.xlsx', index_col=0)
        df_match_player_miss_comp = pd.read_excel(f'{BASE_PATH}/df_match_player_miss.xlsx', index_col=0)
        df_match_odds_miss_comp = pd.read_excel(f'{BASE_PATH}/df_match_odds_miss.xlsx', index_col=0)
        len_inicial = len(df_match_miss_comp)

        df_match_miss_comp_ct = pd.concat([df_match_miss_comp, df_match_miss], axis=0)
        df_match_player_miss_comp_ct = pd.concat([df_match_player_miss_comp, df_match_player_miss], axis=0)
        df_match_odds_miss_comp_ct = pd.concat([df_match_odds_miss_comp, df_match_odds_miss], axis=0)

    # Si es la primera vez que extraigo partidos missing
    except FileNotFoundError:
        df_match_miss_comp_ct = df_match_miss
        df_match_player_miss_comp_ct = df_match_player_miss
        df_match_odds_miss_comp_ct = df_match_odds_miss
        len_inicial = 0

    len_inicial_miss = len(df_match_miss)
    len_final = len(df_match_miss_comp_ct)
    verif = (len_inicial + len_inicial_miss) == len_final
    if not verif:
        logger.error("Fallo la concatenacion de partidos missing a los datos viejos")
    logger.info(f"Shape de todos los partidos missing hasta hoy : {len_inicial} --> {len_final}")

    # Exporto datos
    df_match_miss_comp_ct.to_excel(f'{BASE_PATH}/df_match_miss.xlsx', index=True)
    df_match_player_miss_comp_ct.to_excel(f'{BASE_PATH}/df_match_player_miss.xlsx', index=True)
    df_match_odds_miss_comp_ct.to_excel(f'{BASE_PATH}/df_match_odds_miss.xlsx', index=True)

# Data understanding
def load_data_to_prepare(country, iteration_date, predict_missing, verbose: int = 0):
    """
    Cargo datos de Flashscore a preparar.
    No lo pongo en la clase puesto que no son datos usados durante el entrenamiento.
    
    # Parameters:
        predict_missing: True para levantar los partidos missing con los que no se entrenó. False, para levantar los proximos partidos ya extraidos.
    """  

    # Missing
    if predict_missing:

        logger.warning("Uso los partidos DF_MATCH_MISS quitando los que usé para entrenar.")
        
        BASE_DIR_ALL_MISSING = f'./data/{country}/p6_deployment/missing/data_understanding/all'

        # Obtengo la fecha del ultimo partido con el que entrené los modelos
        df_integrated_updated = pd.read_excel(f"./data/{country}/p4_modeling/{iteration_date}/p2_data_understanding/df_integrated.xlsx", index_col=0)
        df_integrated_updated['date'] = pd.to_datetime(df_integrated_updated['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        last_date = df_integrated_updated['date'].max()
        logger.info(f'Last date: {last_date}')

        # Opcion 2: df_match miss, df_match_player miss, df_match_odds_miss
        # Esta bien? --> Falla cuando usa df_match al final de la preparacion y en modeling sino...
        df_match = pd.read_excel(f'{BASE_DIR_ALL_MISSING}/df_match_miss.xlsx', index_col=0)
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        df_match = df_match[df_match['date'] > last_date] 

        df_match_player = pd.read_excel(f'{BASE_DIR_ALL_MISSING}/df_match_player_miss.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{BASE_DIR_ALL_MISSING}/df_match_odds_miss.xlsx', index_col=0)
        if verbose >= 2:
            logger.info(df_match.shape)
            logger.info(df_match_player.shape)
            logger.info(df_match_odds.shape)

        df_match_player = df_match_player[df_match_player.index.isin(df_match.index)]
        df_match_odds = df_match_odds[df_match_odds.index.isin(df_match.index)]
        if verbose >= 2:
            logger.info(df_match_player.shape)
            logger.info(df_match_odds.shape)

    # Next matches
    else:
        BASE_DIR_NEXT_MATCHES = f'./data/{country}/p6_deployment/data_understanding'

        logger.info("Uso los partidos df_match_next ya extraidos.")
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'{BASE_DIR_NEXT_MATCHES}/df_match_next.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'{BASE_DIR_NEXT_MATCHES}/df_match_player_next.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{BASE_DIR_NEXT_MATCHES}/df_match_next_odds.xlsx', index_col=0)
            
    return df_match, df_match_player, df_match_odds
    
# Generales
def read_data_of_best_model(id_country, verbose : int = 1):
    
    # Levanto dataframe con los modelos a usar por pais
    df_best_models = pd.read_excel("./data/df_best_models.xlsx")

    # Selcciono fila del pais
    row_country = df_best_models[df_best_models['id_country'] == id_country]

    # Obtengo el modelo a usar
    n_model = int(row_country['n_model'].values[0])
    model_name = str(row_country['model_name'].values[0])
    iteration_date_str = row_country['iteration_date'].values[0]
    iteration_date_dt = pd.to_datetime(iteration_date_str, format='%Y-%m-%d').date()  # con .date() saco hora y minutos

    if verbose >= 1:
        logger.info(f"Model: {n_model} Date ite: {iteration_date_dt}")

    return n_model, model_name, iteration_date_dt

def filter_dataframe_by_date(df: pd.DataFrame, initial_date, n_days: int):
    """
    Filtrar registros de dataframe por fecha segun columna 'date'.

    # Parameters:
        df: Dataframe con columna 'date' al cual filtrar. (DataFrame)
        initial_date: Fecha. (Datetime)
        n_days: Dias desde la initial_date.

    # Returns:
        df: Dataframe solo con registros en el periodo de tiempo especificado. (DataFrame)
    """
    # Determino fecha inical 
    limit_date = initial_date - datetime.timedelta(days=n_days) 
    # print(f"Seleccion de ultimos partidos jugados: {limit_date} -- {n_days} days --> {initial_date}")

    # Filtro segun fechas inicial y final
    df_filt = df[(df['date'] >= limit_date) & (df['date'] <= initial_date)]
    df_filt = df_filt.sort_values(by='date', ascending=False) # Ordeno por fecha ascendente
    logger.info(f"Seleccion de ultimos partidos: {len(df)} --> {len(df_filt)}")

    # Si no hay ultimos partidos
    if len(df_filt) == 0:
        logger.warning(f"No hay partidos en los ultimos {n_days} dias.")

    return df_filt

########################################################################## MAIN #######################################################################
def main(
        d_run: dict, id_country: int, d_model: dict = None,                # Params
        n_seasons_missing : int = 1, extract_missing: bool = True,         # Missing
        n_days_max_next_matches: int = 7, predict_missing: bool = False,   # Data understanding
        n_days_fill_data: int = 30,                                        # Data preparation
        porc_m: float = 0.35,                                              # Modeling
        verbose: int = 1, export: bool = True):
    """
    Recoleccion de proximos partidos, preparacion y prediccion
    """
    start = time.time()
    # Cargo variables entorno (x las dudas para collect_predictions.py)
    load_dotenv()
    env = os.getenv('ENVIRONMENT')
    logger.info(f"Environment: {env}")
    
    # Determino country y competence
    df_countries = pd.read_excel('./data/df_countries.xlsx')
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0].lower()
    df_comp = pd.read_excel('./data/df_competencies.xlsx')
    df_comp_country = df_comp[df_comp['id_country'] == id_country]
    if env == 'dev':
        comp_public = df_comp_country[(df_comp_country['is_cup'] == 0) & (df_comp_country['is_second_division'] == 0)]['id_competition'].values  # dev --> para incluir ARG y USA
    elif env == 'prod':
        comp_public = df_comp_country[df_comp_country['is_public'] == 1]['id_competition'].values  # prod

    # Determino n_model, iteration date y nombre --> Lo uso para levantar hiper no solo en modeling sino tmb en data prep.
    if d_model is not None:
        logger.warning("Se usa modelo especificado como parametro y no necesariamente es el que se esta usando en produccion.")
        n_model, model_name, iteration_date_dt = d_model['n_model'], d_model['model_name'], d_model['iteration_date']
    else:
        n_model, model_name, iteration_date_dt = read_data_of_best_model(id_country)

    if verbose >= 0:
        logger.info("\n" + "#"*120 + "\n" + f"COUNTRY: {country.upper()}".center(120) + "\n" + "#"*120 + "\n")
        logger.info(f'Competencias: \n {df_comp_country}  \n Competencias publicas: {comp_public}')
        logger.critical(f"n_model: {n_model} ; iteration_date: {iteration_date_dt}")

    # Creo objetos de clases
    du = DataUnderstandingNew(id_country, country, export) # Creo objeto de clase DataUnderstanding
    dp = DataPreparationNew(country=country, export=export) # Creo objeto de clase DataPreparation
    mo = Modeling(country=country) # Creo objeto de clase DataPreparation
    lo = TrainingDataLoader(country=country, n_model=n_model, model_name=model_name, iteration_date=iteration_date_dt)


    # _____________________________________________________________ MISSING DATA _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "MISSING DATA".center(120) + "\n" + "+"*120 + "\n")
    if d_run['run_missing']: 
        
        # Levanto datos 
        df_match, df_match_player, df_match_odds, df_integrated = load_last_version_extracted_matches(country) # Obtengo ultima version de df_match, df_match_player y df_match odds (con missing)
        df_player_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_player_sofifa.xlsx', index_col=0) # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_player_fifa_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_player_fifa_sofifa.xlsx') # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_teams_sofifa = pd.read_excel(f'./data/{country}/p2_data_understanding/df_teams_sofifa.xlsx', index_col=0)

        # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
        if extract_missing:
            df_match_miss, df_match_player_miss, df_match_odds_miss = du.collect_missing_data(df_match, df_comp_country=df_comp_country, n_seasons_max=n_seasons_missing)
        else:
            df_match_miss = pd.read_excel(f'./data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx', index_col=0)
            df_match_player_miss = pd.read_excel(f'./data/{country}/p6_deployment/missing/data_understanding/all/df_match_player_miss.xlsx', index_col=0)
            df_match_odds_miss = pd.read_excel(f'./data/{country}/p6_deployment/missing/data_understanding/all/df_match_odds_miss.xlsx', index_col=0)
            logger.warning(f"Levanto missing ya extraido: {df_match_miss.shape} {df_match_player_miss.shape} {df_match_odds_miss.shape}")

        # Si hay partidos missing que no extraje aun
        if len(df_match_miss) > 0:
            logger.info(f"Cantidad de partidos missing extraidos: {len(df_match_miss)}")

            if export:
                concat_and_export_old_with_missing(df_match, df_match_player, df_match_odds, df_match_miss, df_match_player_miss, df_match_odds_miss, country)

                if extract_missing:
                    # Guardo datos con los que entrenó el modelo y los missing
                    concat_and_export_missing_extracted(df_match_miss, df_match_player_miss, df_match_odds_miss, country)

            # Preparo datos
            df_match_miss, df_match_player_miss, df_player_fifa_sofifa = dp.format_data(df_match_miss, df_match_player_miss, df_player_fifa_sofifa, reformat=True, export=False)            
            df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False)
            df_integrated_missing = dp.integrate_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) 

            # Concateno missing y old (que puede tener algunos missing ya)
            df_integrated_updated = pd.concat([df_integrated, df_integrated_missing], axis=0)
            logger.warning(f"Concatenación old + missing: {df_integrated.shape} + {df_integrated_missing.shape} --> {df_integrated_updated.shape}")

            if export:
                df_integrated_missing.to_excel(f'./data/{country}/p6_deployment/missing/data_preparation/df_integrated_missing.xlsx', index=True)
                df_integrated_updated.to_excel(f'./data/{country}/p6_deployment/missing/old_updated/df_integrated.xlsx', index=True)

            # Guardo registro de todos los partidos missing juntos (los recien recolectados y los que ya tenia)
            try:
                df_integrated_missing_all = pd.read_excel(f'./data/{country}/p6_deployment/missing/data_preparation/all/df_integrated_missing.xlsx', index_col=0)
            except FileNotFoundError:
                df_integrated_missing_all = pd.DataFrame()  # Es importante para que se guarde por primera vez df_integrated_missing en /all 

            df_integrated_missing_all = pd.concat([df_integrated_missing_all, df_integrated_missing], axis=0)
            if export:
                df_integrated_missing_all.to_excel(f'./data/{country}/p6_deployment/missing/data_preparation/all/df_integrated_missing.xlsx', index=True)
    
            # Verificacion de concatatenacion de nuevos partidos missing con partidos ya jugados
            if len(df_integrated_updated) != (len(df_match_miss) + len(df_integrated)):
                logger.info(f"\nShape of df_integrated_with_missing (siempre es un poco menor a df_match_with_missing pero no se por qué): {df_integrated_updated.shape}")  # Calculo que debe ser por la eliminacion de partidos con goles="-" que hice de df_integrated en main.py
                logger.info(df_integrated_updated.head(3))
                logger.error("Falló la concatenación de nuevos partidos missing con los partidos ya jugados.")
                raise ValueError("Falló la concatenación de nuevos partidos missing con los partidos ya jugados.")
        
        else:
            logger.warning("Ya se habian extriado todos los partidos missing. Aun no hay partidos nuevos.")
            df_integrated_updated = pd.read_excel(f'./data/{country}/p6_deployment/missing/old_updated/df_integrated.xlsx', index_col=0)

    else:
        logger.warning("Se evitó por comando la extraccion y preparacion de partidos missing.")

        try:
            df_integrated_updated = pd.read_excel(f'./data/{country}/p6_deployment/missing/old_updated/df_integrated.xlsx', index_col=0)
        except FileNotFoundError:
            df_integrated_updated = pd.read_excel(f'./data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)
            logger.warning("No hay un dataframe integrado con missing aun. Tuve que levantar el df_integrated de main.py...")
    
        if predict_missing:
            df_integrated_updated = pd.read_excel(f"./data/{country}/p4_modeling/{iteration_date_dt}/p2_data_understanding/df_integrated.xlsx", index_col=0)


    # _____________________________________________________________ DATA UNDERSTANDING _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "DATA UNDERSTANDING".center(120) + "\n" + "+"*120 + "\n")
    if d_run['data_unders']:
        # Extriago datos de los partidos en los proximos dias
        df_match, df_match_player, df_match_odds = du.collect_initial_data_new(l_competencies=comp_public, df_comp_country=df_comp_country, n_days=n_days_max_next_matches)
    
    else:
        logger.warning("Se evitó por comando la extraccion de proximos partidos.")

        if d_run['data_prep'] or d_run['modeling']:
            df_match, df_match_player, df_match_odds = load_data_to_prepare(country=country, iteration_date=iteration_date_dt, predict_missing=predict_missing)
        
    # Si no hay proximos partidos
    if len(df_match) == 0:
        # Evito preparacion y modelado
        logger.error("No hay próximos partidos para los cuales predecir su resultado.")
        return ValueError

    if verbose >= 2:
        du.describe_data_new(df_match, df_match_player, df_match_odds, verbose=verbose)
    
    if verbose >= 3:
        print("\n DF MATCH \n", df_match.head(2))
        print("\n DF MATCH PLAYER \n", df_match_player.head(2))
        print("\n df_player_sofifa \n", df_player_sofifa.head(2))
        print("\n df_player_fifa_sofifa \n", df_player_fifa_sofifa.head(2))

    # SOFIFA
    df_player_sofifa = pd.read_excel(f"data/{country}/p3_data_preparation/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0)
    df_player_fifa_sofifa = pd.read_excel(f"data/{country}/p3_data_preparation/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
    df_teams_sofifa = pd.read_excel(f"data/{country}/p3_data_preparation/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)


    # _____________________________________________________________ DATA PREPARATION _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "DATA PREPARATION".center(120) + "\n" + "+"*120 + "\n")
    if d_run['data_prep']:

        # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
        d_hiper = lo.load_data_preparation_hyperparameters()
        df_etiquetas = lo.load_df_etiquetas(d_hiper)
        scaler, columns_scaled = lo.load_scaler_model(d_hiper)

        # Preparacion de datos hasta integrate
        df_match, df_match_odds = dp.format_data_new(df_match, df_match_odds)
        df_match, df_match_player = dp.clean_data_new(df_match, df_match_player)
        df = dp.integrate_data_new(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc


        # Tirar error si estoy prediciendo proximos partidos que ya se jugaron y ya fueron recolectados en missing...
        rows_rep = df[df.index.isin(df_integrated_updated.index)]
        logger.warning(f"IDXS REPETIDOS: {len(rows_rep)}")

        if len(rows_rep) > 0:
            logger.error("En caso que los proximos partidos ya esten en df_old_last_matches (o sea, los partidos ya se jugaron y los recolectaste como missing, tirara error al momento de predecir por indice repetido.)")
                
            user_input = str(input("Escribe 'y' para eliminar indices duplicados y seguir la prediccion: "))
            if user_input == 'y':
                logger.info(df_integrated_updated.shape)
                df_integrated_updated = df_integrated_updated[~df_integrated_updated.index.isin(rows_rep.index)]
                logger.info(df_integrated_updated.shape)
            else:
                raise KeyError
        

        # Selecciono los ultimos partidos de los ya jugados
        initial_date = datetime.datetime.now()  # initial_date = datetime.datetime(2024, 8, 16)  # Prueba para establecer initial date en una fecha especifica (e.g. 16/08/2024)
        ## Para fill_data
        logger.info("Seleccion de ultimos partidos para rellenar formaciones...")
        df_last_old_matches_fill = filter_dataframe_by_date(df=df_integrated_updated, initial_date=initial_date, n_days=n_days_fill_data) # Los parates pueden ser de 3 meses o mas. Por eso tomo 5 meses para tener un poco de margen de seguridad.
        ## Para construct_data
        logger.info("Seleccion de ultimos partidos para construccion de variables...")
        n_days_max = max(d_hiper['n_dias_ult_part'])
        n_days_period = n_days_max * 2 if d_hiper['segun_localia'] == True else n_days_max
        df_last_old_matches_construct = filter_dataframe_by_date(df=df_integrated_updated, initial_date=initial_date, n_days=n_days_period) # No sirve de nada hacerlo flex dado que construct_data() de main.py usa n_days

        # Sigo con la preparacion de datos desde fill_data
        df, df_c1, df_c2 = dp.fill_data_not_available_yet(df, df_last_old_matches_fill, comp_to_select=comp_public)
        df = dp.construct_data_new(df_next_matches=df, df_last_old_matches=df_last_old_matches_construct, df_old_matches=df_integrated_updated, n_days=d_hiper['n_dias_ult_part'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'], dif_con_against=d_hiper['dif_con_against'], columns_used=columns_scaled)
        df = dp.tag_string_data_to_integer_new(df, df_etiquetas)
        df, df_fill = dp.clean_data_2_new(df=df, scaler_loaded=scaler, columns_used=columns_scaled, comp_to_select=comp_public, columns_selected=d_hiper['selected_columns']) # Antes usaba comp_to_select pero me quedaban los partidos de todas las comp en predicciones.xlsx
        df = dp.select_data_new(df, d_hiper['selected_columns'])
        
        # Guardo df justo antes de predecir cuando hago pred_missing para poder comparar ASSESS Y PROD
        if predict_missing: 
            df.to_excel(f'./data/{country}/p6_deployment/data_preparation/df_selected_MISS.xlsx', index=True)

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
            df_fill = pd.read_excel(f'./data/{country}/p6_deployment/data_preparation/df_filled_columns.xlsx', index_col=0)
  
    #_____________________________________________________________ MODELING _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "MODELING".center(120) + "\n" + "+"*120 + "\n")
    if d_run['modeling']:

        # Levanto hiperparametros de modeling
        loaded_model = lo.load_model()
        classes = [0, 1, 2] # loaded_model.classes_

        try:
            df_match = df_match.loc[:, ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition', 'country', 'competition']] 
        except KeyError:
            df_match = df_match.loc[:, ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition']] # falla cuando uso predict_missing porque falta 'country' y 'competition'
        df_match_odds = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta

        # Realizo predicciones sobre los nuevos partidos
        y_pred_prob, y_pred = mo.predict(model=loaded_model, X_test=df)
        df_pred_proba = pd.DataFrame({
                'predicted_result': y_pred,
                f'prob_class_{classes[1]}': y_pred_prob[:, 1],  # Probabilidad de la clase 1
                f'prob_class_{classes[0]}': y_pred_prob[:, 0],  # Probabilidad de la clase 0
                f'prob_class_{classes[2]}': y_pred_prob[:, 2]   # Probabilidad de la clase 2 (si hay 3 clases)
            }, index=df.index)

        # Concateno conjunto de datos # df_fill puede tirar error. Si tira, arreglar.
        df_predicciones = pd.concat(
            [df_match, 
             df_match_odds, 
             df_pred_proba, 
             df_c1['copiado_formaciones'], 
             df_fill.loc[:, ['player_emergency_fill', 'emergency_fill']]
             ], 
            axis=1) 
    
        # Aplico estrategia de apuesta
        bs = betting_strategy.BettingStrategy(country=country, iteration_date=iteration_date_dt)
        d_strategy = {'prob_dp': -1, 'curva': 'linear', 'm': 10, 'b': 0, 'odd_weight':0, 'lim_sup': 0, 'normalized': False} if predict_missing else lo.load_modeling_hyperparameters()
        df = bs.calculate_dif_proba_in_predicted_result(df_predicciones)

        # Pasarle "strategy" prod o bien ya pasarle el d_params...
        if  isinstance(d_strategy, dict):
            logger.warning("Aplico MISMA estrategia A TODOS LOS RDOS. ")
            df = bs.apply_strategy(df, param_dict=d_strategy)
        else:
            logger.warning("Aplico estrategia DISTINTA POR RESULTADO. ")
            df = bs.apply_strategy_by_result(df, df_hiper=d_strategy)

        # Aplico reduccion a stake --> Ver si funciona.. Creo que si.
        df['stake_to_bet'] = df['stake_to_bet'] * porc_m
        # logger.info(f"m_to_use: {d_strategy['curva_m']} tras aplicar {porc_m}")

        # Ultimos preparativos
        df = format_data.map_teams(df, country=country)     # Revierto etiquetas para tener nombres de equipos en vez de ids  # --> Podria usar mapeo 
        df = df[df['id_competition'].isin(comp_public)]     # Filtro partidos para quedarme solo con los de competencias publicas.

        if export:
            df.to_excel(f'./data/{country}/p6_deployment/predicciones.xlsx', index=True)

        if predict_missing: 
            df.to_excel(f'./data/{country}/p6_deployment/df_predicciones_missing.xlsx', index=True)

        logger.critical("LA PREDICCION FUE UN EXITO!")

    else:
        logger.warning("Se evitó por comando la predicción de proximos partidos.")
        return pd.DataFrame()

    end = time.time()
    logger.info(f"Main_next_matches en {(end - start)/60:.1f} minutos \n\n")
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":    

    n_days = 15
    # d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True}
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': True}  # No puedo correr data_unders = False si los proximos partidos ya estan en missing.
    directorio = os.getenv('BASE_DIR_LOCAL')

    l_countries = [48, 55, 59, 77, 148]
    l_countries = [77]
    d_countries = {
        6: ["argentina", '2024-12-05'], 
        48: ["england", '2024-12-23'], 
        55: ["france", '2024-12-26'], 
        59: ["germany", '2024-12-26'], 
        77: ["italy", '2024-12-23'], 
        # 77: ["italy", '2025-01-02'], 
        148: ["spain", '2024-12-25'], 
        167: ["usa", '2024-12-05']
        }
    
    # Definir condiciones del análisis
    for id_country in l_countries:

        # recolectar missing
        # df = main(d_run, id_country, extract_missing=True, n_seasons_missing=3, export=d_run['export']) 
        # df = main(d_run, id_country, extract_missing=False, export=d_run['export']) 

        # Probar un modelo
        d_model = {'n_model': 902, 'model_name': "LogisticRegression", 'iteration_date': d_countries[id_country][1]} # DecisionTreeClassifier, XGBClassifier, neural_networ, SVC, LogisticRegression, MLPClassifier
        ## Prox partidos
        df = main(d_run, id_country, n_days_max_next_matches=n_days, d_model=d_model, predict_missing=False, export=d_run['export']) 
        ## En partidos missing
        # df = main(d_run, id_country, n_days_max_next_matches=n_days, d_model=d_model, predict_missing=True, export=False) 
        
        # Prod 
        # df = main(d_run, id_country, n_days_max_next_matches=n_days, export=d_run['export']) 

        if isinstance(df, pd.DataFrame):
            df.to_excel(f"{directorio}/predicciones.xlsx")