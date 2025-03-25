# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
from utils.set_up_logging import logger
from utils import directories
import os
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
        for df in [df_match_concat, df_match_player_concat, df_match_odds_concat]:
            self.verify_data_quality(df)

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
            df_match_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_miss.xlsx', index=True)
            df_match_player_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_player_miss.xlsx', index=True)
            df_match_odds_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_odds_miss.xlsx', index=True)

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

        # Copio valores en ultimos partidos (deberia copiar solo referee y coaches)
        miss_player_columns = [col for col in df_last_old_matches.columns if ('player_miss' in col)]  # --> ojo porque no se si las rellena ok... es complejo el rellenado.
        l_var_to_copy = miss_player_columns + ['id_coach_home', 'id_coach_away']  # Es clave copiar coaches porque no suele estar hasta que esten las formaciones..
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
    
    def clean_data_3_new(self, df, competencies_to_select = None):
        """
        Es clave para que no prediga todas las comp?
        """
        n_reg_inic = len(df)
        df = df[df['id_competition'].isin(competencies_to_select)]
        
        if self.verbose >= 1:
            logger.info(f"Filas luego de filtrar x competencia: {n_reg_inic} --> {len(df)}")
        return df
    
    def construct_data_new(self, df_next_matches: pd.DataFrame, df_old_matches, df_last_old_matches,
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
        
        # Returns
            Dataframe con proximos partidos construido utilizando los partidos ya jugados. (DataFrame)
        """
        logger.info("Constructing new data...")
        start = time.time()

        # 1) Construyo historial entre si  (podria evitar construirlas si no estan en columns_used...)
        df_next_matches = self.construct_h2h_next_matches(df_next_matches, df_old_matches, n_years_h2h, columns_used)  # usar with_h2h=False para no reemplazarlo.

        # Si hay "ultimos partidos"
        if len(df_last_old_matches) > 0:
            self.determine_stats_to_use()
            # Construyo datos (sin historiales) luego de concatenar proximos partidos (df_next_matches) y los ultimos partidos ya jugados (df_last_old_matches)
            df_concat_last = pd.concat([df_next_matches, df_last_old_matches], axis=0)
            df_constructed = self.construct_data(
                df_concat_last, 
                n_last_matches=n_last_matches,
                n_years_h2h=n_years_h2h, 
                segun_localia=segun_localia, 
                calculate_dif=calculate_dif, 
                decay_rate=decay_rate, 
                prod=True, 
                export=False
                )
            df_next_matches = df_constructed[df_constructed.index.isin(df_next_matches.index)]  # Separo datos construidos entre los proximos partidos y los ya jugados  # En caso que los proximos aprtidos ya esten en df_old_last_matches (o sea, los partidos ya se jugeron y los recolectaste como missing, tirara error al momento de predecir por indice repetido.)

        # Si no hay "ultimos partidos"
        else:
            # evito construir variables historicas
            logger.warning("Evito construccion de variables historicas debido a la falta de ultimos partidos")
            df_next_matches = self.construct_data(df_next_matches, n_last_matches, n_years_h2h, segun_localia=segun_localia, with_historic=False, prod=True, verbose=verbose, export=False)

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
        Mejora a hacer: 
        - podria evitar la construccion de las variables si no estan en columns_used...
        """
        # Construyo columna "result" para poder calcular h2h
        df_old_matches = construct_data.determine_result(df_old_matches, self.var_resp) # Construyo columna resultado en el old para poder calcular historial
        idxs_to_construct = df_next_matches.index

        ## Construyo historiales
        df = pd.concat([df_next_matches, df_old_matches], axis=0)
        df = construct_data.h2h_by_date(df, n_years=n_years_h2h, prod=True, idxs_to_construct=idxs_to_construct)
        df = construct_data.h2h_by_date_by_localia(df, n_years=n_years_h2h, prod=True, idxs_to_construct=idxs_to_construct)

        ## Vuelvo a seleccionar df_next_matches pero con historiales construidos
        columnas_deseadas = list(df_next_matches.columns) + [col for col in df.columns if 'h2h_' in col]  # Reemplazo historiales nan por 0
        df = df[columnas_deseadas]
        df_next_matches = df[df.index.isin(df_next_matches.index)]
        return df_next_matches
        
    def tag_string_data_to_integer_new(self, df: pd.DataFrame, df_etiquetas_loaded, columns_scaled, verbose: int = 0):
        """
        Utilizando las mismas etiquetas que cuando se entreno el modelo para el pais, convierto columnas string a integer

        # Parameters
            df: Dataframe con algunas columnas string las cuales necesitamos convertir a integer
            df_etiquetas_loaded: Conversion de valores string a integer (usada en el entrenamiento).
            columns_selected: Para determinar las columnas a codificar 

        # Return
            df: Dataframe pasado como parametro habiendo convertido sus columnas string a integer tal como en el entrenamiento.
        """
        logger.info("Tagging string data to integer..")

        # Determino columnas a codificar
        l_col_codificadas = df_etiquetas_loaded['variable'].unique()
        l_col_a_codificar = [col for col in l_col_codificadas if col in columns_scaled]  # Las col selected no puedo porque falla el scaler..

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int_already_tagged(df=df, df_etiquetas=df_etiquetas_loaded, l_col_a_codificar=l_col_a_codificar, verbose=2)  # Si o si tengo que devolver df_etiquetas?
    
        if self.export:
            df.to_excel(f'{self.BASE_DIR}/df_tagged.xlsx', index=True) 
            df_etiquetas.to_excel(f'{self.BASE_DIR}/df_etiquetas_actualizado.xlsx', index=False) 
        return df

    def clean_data_2_new(self, df: pd.DataFrame, scaler_loaded, columns_scaled, columns_selected, verbose: int = 0):
        """
        Filtrado por competencias y escalado de datos.
        """
        logger.info("Cleaning data 2...")

        # Reemplazo infinitos
        df = clean_data.replace_infinite(df)

        # Para evitar ciertas competencias --> Moverlo a clean antes de construir?
        n_col_inic = len(df.columns)

        # El scaler necesita exactamenete las mismas cols que escaló durante el train (no se por que ger tiene loc_efficiency de mas...)
        missing_cols = set(columns_scaled) - set(df.columns)
        extra_cols = set(df.columns) - set(columns_scaled)

        if missing_cols:
            logger.warning(f"Faltan columnas en los nuevos datos: {missing_cols}")
            for col in missing_cols:
                df[col] = 0  # O usa df[col] = df[col].mean() si prefieres
            
        if extra_cols:
            logger.info(f"Se eliminarán columnas extra: {extra_cols}")
            df = df[columns_scaled]  # Ahora sí seleccionas las columnas correctas

        # Selecciono las mismas caracteristicas con las que entrene el scaler (sino, falla)
        df = df.loc[:, columns_scaled]
        if verbose >= 1:
            logger.info(f"Columnas luego de filtrar x columnas scaled: {n_col_inic} --> {len(df.columns)}")
        
        # Tratamiento de NaN values (antes de escalar para que el 0 realmente sea 0.)
        df, df_emergency_fill = self.treat_nan_values_new(df=df, columns_selected=columns_selected)

        # Transforma los nuevos datos de predicción utilizando el StandardScaler cargado
        try: 
            X_scaled = scaler_loaded.transform(df)
            X_scaled_df = pd.DataFrame(X_scaled, columns=columns_scaled, index=df.index)

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
        self.BASE_DIR_dp = f"./data/{self.country}/p3_data_preparation/{self.iteration_date}"
        self.BASE_DIR_mod = f"./data/{self.country}/p4_modeling/{self.iteration_date}"
    
    # Data preparation
    def load_data_preparation_hyperparameters(self):
        """
        Cargo hiperparametros de DataPreparation()
        """
        d = {}

        # Si se levanta de main_find_best_hyper.py
        if self.n_model is not None:
            
            df_iteration = pd.read_excel(f"{self.BASE_DIR_mod}/df_iteration.xlsx")
   
            # Selecciono la primera. Hay una por modelo entrenado pero los hiper son =.
            try:
                row_hiper = df_iteration[df_iteration['n_iteration'] == self.n_model].iloc[0]  
            except IndexError:
                logger.error(f"El modelo {self.n_model} no fue entrenado en el entrenamiento del {self.iteration_date}.")
                raise IndexError

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
        d['n_last_matches'] = eval(row_hiper['n_last_matches']) 
        d['n_years_h2h'] = int(row_hiper['n_anios_hist']) # .values[0]
        d['segun_localia'] = row_hiper['segun_localia'] # .values[0]
        d['calculate_dif'] = row_hiper['calculate_dif'] # .values[0]
        ## Clean_data_2
        n_years_to_select = row_hiper['n_years_to_select'] # .values[0]
        d['n_years_to_select'] = None if pd.isna(n_years_to_select) else int(n_years_to_select) # Si n_years_to_select es NaN, lo paso de np.nan a None
        d['comp_to_select'] = eval(row_hiper['comp_to_select']) # .values[0]
        if pd.isna(row_hiper['fill_na']):  # Verifica si es NaN o None
            d['fill_na'] = row_hiper['fill_na'] = None
        elif isinstance(row_hiper['fill_na'], float):
            d['fill_na'] = '0'
        else:
            d['fill_na'] = row_hiper['fill_na']
        d['decay_rate'] = 0 if row_hiper['decay_rate'] == 0.0 else row_hiper['decay_rate'] 
        ## Select_data
        d['selected_columns'] = selected_columns

        self.path_clean = f'{d['comp_to_select']}'
        self.path_construct = f'{d['n_last_matches']}_{d['n_years_h2h']}_{d['segun_localia']}_{d['calculate_dif']}_{d['decay_rate']}'
        self.path_clean_2 = f'{d['n_years_to_select']}_{d['fill_na']}'

        if self.verbose >= 0:
            logger.info("Hiperparametros cargados:")
            for key, value in d.items():
                logger.info(f'\t {key}: {value}')
        return d

    def load_df_etiquetas(self):

        logger.info("Levento etiquetas con el que entrené")
        subpath = f'{self.path_clean}__{self.path_construct}'
        path_tag = f'{self.BASE_DIR_dp}/tag/df_etiquetas_{subpath}.xlsx'       
        df_etiquetas = pd.read_excel(path_tag, index_col=0)

        if self.verbose >= 1:  
            print(df_etiquetas.head(3))

        return df_etiquetas

    def load_scaler_model(self):
        """
        Levanto modelo utilizado en entrenamiento para escalar datos
        """
        subpath = f'{self.path_clean}__{self.path_construct}__{self.path_clean_2}'
        path_scaler = f'{self.BASE_DIR_dp}/clean_data_2/scaler_model_{subpath}.pkl'

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
                return {'prob_dp': df_hiper['prob_dp'].values[0], 'curva': df_hiper['curva'].values[0], 'm': df_hiper['m'].values[0], 'b': df_hiper['b'].values[0]}

            if self.verbose >= 0:
                logger.info("Hiperparametros cargados:")
                logger.info(df_hiper)

        except FileNotFoundError as e:
            print("Falló la carga del df_strategy")

            user_input = str(input("Quiere predecir sin estrategia igual (y para aceptar)?: "))
            if user_input == 'y':
                return {'prob_dp': 0, 'curva': 'linear', 'm': 10, 'b': 0, 'k': 1}
            
            raise ValueError(e)

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
            logger.warning('Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
        
        logger.info(f"Shapes: \t df_match: {df_match.shape} \t df_match_player:{df_match_player.shape} \t df_match_odds: {df_match_odds.shape}")
        return df_match, df_match_player, df_match_odds

    def read_last_integrate_data(self):

        # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
        try:
            df_integrated = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_integrated.xlsx', index_col=0)
            logger.info('Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')

        # Si no existe un df_integrated concatenado entre old y missing
        except FileNotFoundError:

            warning_msg = f"No se pudo levantar el df_integrated con old + missing. Esto es correcto solo si nunca se ha extraido / integrado missing. Desea levantar el df_integrated con el que se entrenó? (y/n)"
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

            warning_msg = f"No se pudo levantar df_match, df_match_player y df_match_odds missing. Esto es correcto solo si nunca se ha extraido missing. Desea inicializar crear los dataframes? (y/n)"
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

            warning_msg = f"No se pudo levantar el df_integrated_missing. Esto es correcto solo si nunca se ha integrado missing. Desea inicializar crear el dataframe? (y/n)"
            user_input = input(warning_msg).strip().lower() 
            if user_input != "y":
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
        logger.info(f"Model: {n_model} ; Model name: {model_name}")

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
        porc_m: float = None, d_model: dict = None,                                                     # Modeling
        verbose: int = 1, export: bool = True,
        ):
    """
    Recoleccion de proximos partidos, preparacion y prediccion
    """
    start = time.time()

    # Levanto datasets 
    df_countries = pd.read_excel('./data/df_countries.xlsx')
    df_comp = pd.read_excel('./data/df_competencies.xlsx')

    # Determino country, competence e ite_date
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0].lower()
    df_comp_country = df_comp[df_comp['id_country'] == id_country]
    iteration_date_dt = pd.to_datetime(iteration_date, format='%Y-%m-%d').date()  # con .date() saco hora y minutos
    comp_public = df_comp_country[(df_comp_country['is_cup'] == 0) & (df_comp_country['is_second_division'] == 0)]['id_competition'].values  # dev --> para incluir ARG y USA

    if verbose >= 0:
        logger.info("\n" + "#"*120 + "\n" + f"COUNTRY: {country.upper()}".center(120) + "\n" + "#"*120 + "\n")
        logger.info(f'Competencias: \n {df_comp_country}  \n Competencias publicas: {comp_public}')
        logger.warning(iteration_date_dt)

    # Creo objetos de clases
    du = DataUnderstandingNew(id_country, country, export=export) # Creo objeto de clase DataUnderstanding
    dp = DataPreparationNew(id_country=id_country, country=country, iteration_date=iteration_date_dt, export=export) # Creo objeto de clase DataPreparation
    mo = Modeling(country=country, date=iteration_date_dt) # Creo objeto de clase DataPreparation
    mis = MissingData(country=country, iteration_date=iteration_date_dt)

    # Read data usada en mas de una seccion
    # Missing data
    if d_run['run_missing'] or predict_missing:
        df_match_miss, df_match_player_miss, df_match_odds_miss = mis.read_last_missing_data()
        df_integrated_missing = mis.read_last_integrate_missing_data()
    df_integrated_upd = mis.read_last_integrate_data() # Last df_integrated con missing + old    
    # SOFIFA
    df_player_sofifa = pd.read_excel(f"data/{country}/p2_data_understanding/old_updated/{iteration_date_dt}/df_player_sofifa.xlsx", index_col=0)
    df_player_fifa_sofifa = pd.read_excel(f"data/{country}/p2_data_understanding/old_updated/{iteration_date_dt}/df_player_fifa_sofifa.xlsx")  

    # _____________________________________________________________ MISSING DATA _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "MISSING DATA".center(120) + "\n" + "+"*120 + "\n")
    if d_run['run_missing']: 
        
        # Levanto datos: old + los ultimos missing extraidos
        df_match_upd, df_match_player_upd, df_match_odds_upd = mis.read_last_flashscore_data() # Last df_integrated con missing + old

        # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
        df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new = du.collect_missing_data(df_match_upd, df_comp_country=df_comp_country, n_seasons_max=n_seasons_missing)
        logger.info(f"Cantidad de partidos missing extraidos: {len(df_match_miss_new)}")

        if export and len(df_match_miss_new) > 0:
            # Mucho cuidado si falla la preparacion pues los missing estaran en old_updated pero no integrados correctamente. (deberias exportar si la prep funciona o algo asi)
            mis.concat_with_missing_already_extracted(df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new, df_match_miss, df_match_player_miss, df_match_odds_miss)  # missing all --> NO HACERLO CUANDO SOLO QUIERO PREPARAR... Deberia evitar que concatene si los partidos missing ya estan...
            mis.concat_old_with_missing(df_match_upd, df_match_player_upd, df_match_odds_upd, df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new) # Old + missing # # No lo quiero cuando ya extraje missing y solo quiero preparar...
        
        # Si extrajo missing
        if len(df_match_miss_new) > 0:
            
            # Preparo datos missing            
            df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new, df_player_fifa_sofifa = dp.format_data(df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new, df_player_fifa_sofifa, reformat=True, export=False)            
            df_match_miss_new, df_match_player_miss_new, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match_miss_new, df_match_player_miss_new, df_player_sofifa, df_player_fifa_sofifa, export=False)
            # df_match_miss, df_match_player_miss, df_match_odds_miss, df_player_sofifa, df_player_fifa_sofifa = dp.verify_format(df_match_miss, df_match_player_miss, df_match_odds_miss, df_player_sofifa, df_player_fifa_sofifa, prod=False) # prod=False pues los partidos ya se jugaron..
            df_integrated_missing_new = dp.integrate_data(df_match_miss_new, df_match_player_miss_new, df_player_sofifa, df_player_fifa_sofifa, prod=True, export=False) 

            # Concateno missing y old (que puede tener algunos missing ya)
            df_integrated_updated = pd.concat([df_integrated_upd, df_integrated_missing_new], axis=0)
            df_integrated_updated = df_integrated_updated[~df_integrated_updated.index.duplicated(keep='first')]  # El df_integrated tiene missing hasta el dia en que entrené
            logger.warning(f"Concatenación old + missing: {df_integrated_upd.shape} + {df_integrated_missing_new.shape} --> {df_integrated_updated.shape} (si nunca preparaste, missing no se agrega pues ya está)")

            # Guardo registro de todos los partidos missing juntos (los recien recolectados y los que ya tenia)
            df_integrated_missing_all = pd.concat([df_integrated_missing, df_integrated_missing_new], axis=0)
            
            if export:                
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

        if verbose >= 2:
            du.describe_data_new(df_match, df_match_player, df_match_odds, verbose=verbose)
        
        # SOFIFA
        df_player_sofifa = pd.read_excel(f"data/{country}/p2_data_understanding/old_updated/{iteration_date_dt}/df_player_sofifa.xlsx", index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f"data/{country}/p2_data_understanding/old_updated/{iteration_date_dt}/df_player_fifa_sofifa.xlsx")

        if verbose >= 1:
            print("\n DF MATCH \n", df_match.head(2))
            print("\n DF MATCH PLAYER \n", df_match_player.head(2))
            print("\n df_player_sofifa \n", df_player_sofifa.head(2))
            print("\n df_player_fifa_sofifa \n", df_player_fifa_sofifa.head(2))

    else:
        logger.warning("Se evitó por comando la extraccion de proximos partidos.")

        if predict_missing:
            logger.info("Uso los partidos df_match MISSING ya extraidos.")
            df_match, df_match_player, df_match_odds = df_match_miss.copy(), df_match_player_miss.copy(), df_match_odds_miss.copy()
            df_match = df_match[df_match['id_competition'].isin(comp_public)]
 
            # Levanto df_integrated de cuando entrené modelos
            df_integrated_train = pd.read_excel(f'data/{country}/p3_data_preparation/{iteration_date}/df_integrated.xlsx', index_col=0)
            
            # Selecciono los missing con los que no se entrenó
            df_match = df_match[~df_match.index.isin(df_integrated_train.index)]
            df_match_player = df_match_player[~df_match_player.index.isin(df_integrated_train.index)]
            df_match_odds = df_match_odds[~df_match_odds.index.isin(df_integrated_train.index)]
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
    logger.warning(f"IDXS REPETIDOS: {len(rows_rep)}")
    if len(rows_rep) > 0:

        logger.error("En caso que los proximos partidos ya esten en df_old_last_matches (o sea, los partidos ya se jugaron y los recolectaste como missing, tirara error al momento de predecir por indice repetido.)")
        user_input = str(input("Escribe 'y' para eliminar indices duplicados y seguir la prediccion: "))

        if user_input == 'y':
            # Op1)  Elimino repetidos de proximos partidos a predecir (y no de df_int_updated..)
            # logger.info(df_match.shape)
            # df_match = df_match[~df_match.index.isin(rows_rep.index)]
            # logger.info(df_match.shape)

            # Op2) Elimino repetidos de los partidos viejos para predecir los ya jugados tmb
            print(df_integrated_updated.shape)
            df_integrated_updated = df_integrated_updated[~df_integrated_updated.index.isin(rows_rep.index)]
            print(df_integrated_updated.shape)
        else:
            raise KeyError
        
    # En caso que no haya proximos partidos para predecir, no los preparo.
    elif len(df_match) == 0:
        return pd.DataFrame()

    # _____________________________________________________________ DATA PREPARATION _____________________________________________________________ #
    logger.info("\n" + "+"*120 + "\n" + "DATA PREPARATION".center(120) + "\n" + "+"*120 + "\n")
    if d_run['data_prep']:

        # Determino n_model, iteration date y nombre --> Lo uso para levantar hiper no solo en modeling sino tmb en data prep.
        n_model, model_name = read_data_of_best_model(id_country, d_model)
        if verbose >= 0:
            logger.critical(f"n_model: {n_model} ; model_name: {model_name}")

        # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
        lo = TrainingDataLoader(country=country, n_model=n_model, model_name=model_name, iteration_date=iteration_date_dt)
        d_hiper = lo.load_data_preparation_hyperparameters()
        df_etiquetas = lo.load_df_etiquetas()
        scaler, columns_scaled = lo.load_scaler_model()

        # Filtro df_integrated con las competencias seleccionadas
        df_integrated_updated = df_integrated_updated[df_integrated_updated['id_competition'].isin(d_hiper['comp_to_select'])]  # Filtro partidos por comptencia --> # Para construir como en train, solo las comp que corresponden # Para rellenar solo con los partidos de la misma liga...

        # Definir rango de fechas 
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        initial_date = df_match['date'].min()  # Obtiene la fecha mínima (para filtrar dfs para rellenar y construir) 
        logger.info(f"📅 Fecha inicial: {initial_date}.")

        # Preparacion de datos
        if not predict_missing: # En missing ya tengo las formaciones
            # Format a integrate (usar lo mismo que en df_int_missing)
            df_match, df_match_player, df_match_odds, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_match_odds, df_player_fifa_sofifa, reformat=False, prod=True, export=False)            
            df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, export=False)
            # df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa = dp.verify_format(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, prod=prod_vf)
            df = dp.integrate_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, prod=True, export=False) 
            df = df[df['id_competition'].isin(d_hiper['comp_to_select'])]  # Clean data antes de construir
    
            ## Fill data
            ### Selecciono los ultimos partidos de los ya jugados para rellenar
            logger.info("Seleccion de ultimos partidos para rellenar formaciones...")
            df_last_old_matches_fill = filter_dataframe_by_date(df=df_integrated_updated, initial_date=initial_date, n_days=n_days_fill_data) # Los parates pueden ser de 3 meses o mas. Por eso tomo 5 meses para tener un poco de margen de seguridad.
            df_last_old_matches_fill = df_last_old_matches_fill[df_last_old_matches_fill['id_competition'].isin(comp_public)] # Quiero rellenar solo con las competencias publicas.
            ### Relleno datos
            df, df_c1, df_c2 = dp.fill_data_not_available_yet(df, df_last_old_matches_fill)

        else:
            # Verificacion de assess y test
            verify_assess_and_test = False # True solo si queres verificar que assess = test. Podes comparar las predicciones de un modelo en test y lo que sale del assess (predice los partidos de test tmb)
            if verify_assess_and_test:
                # Format a integrate (usar lo mismo que en df_int_missing)
                df_match, df_match_player, df_match_odds, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_match_odds, df_player_fifa_sofifa, reformat=False, prod=True, export=False)            
                df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, export=False)
                # df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa = dp.verify_format(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, prod=prod_vf)
                df = dp.integrate_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, prod=True, export=False) 
                df = df[df['id_competition'].isin(d_hiper['comp_to_select'])]  # Clean data antes de construir

            else:
                # Forma 2: desde int_missing 
                df = df_integrated_missing.copy()
                print(df.shape)
                df = df[df.index.isin(df_match.index)]
                df = df[df['id_competition'].isin(d_hiper['comp_to_select'])] 
                print(df.shape)

            """
            # Verificacion de prod y assess sin formaciones...
            cols_to_replace = [col for col in df.columns if "_player_" in col]
            df[cols_to_replace] = np.nan

            ## Fill data
            ### Selecciono los ultimos partidos de los ya jugados para rellenar
            logger.info("Seleccion de ultimos partidos para rellenar formaciones...")
            df_last_old_matches_fill = filter_dataframe_by_date(df=df_integrated_updated, initial_date=initial_date, n_days=n_days_fill_data) # Los parates pueden ser de 3 meses o mas. Por eso tomo 5 meses para tener un poco de margen de seguridad.
            df_last_old_matches_fill = df_last_old_matches_fill[df_last_old_matches_fill['id_competition'].isin(comp_public)] # Quiero rellenar solo con las competencias publicas.
            ### Relleno datos
            df, df_c1, df_c2 = dp.fill_data_not_available_yet(df, df_last_old_matches_fill)     
            """
        
        ## Construct
        ### Selecciono los ultimos partidos de los ya jugados para construir
        logger.info("Seleccion de ultimos partidos para construccion de variables...")
        df_last_old_matches_construct = filter_dataframe_by_date(df=df_integrated_updated, initial_date=initial_date, n_days=max(d_hiper['n_last_matches'])) # No sirve de nada hacerlo flex dado que construct_data() de main.py usa n_days
        df_last_old_matches_h2h = filter_dataframe_by_date(df=df_integrated_updated, initial_date=initial_date, n_days=d_hiper['n_years_h2h']*365 + 100) # No sirve de nada hacerlo flex dado que construct_data() de main.py usa n_days
        ### Construyo datos
        df = dp.construct_data_new(
            df_next_matches=df, df_last_old_matches=df_last_old_matches_construct, df_old_matches=df_last_old_matches_h2h, 
            n_last_matches=d_hiper['n_last_matches'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'], 
            calculate_dif=d_hiper['calculate_dif'], decay_rate=d_hiper['decay_rate'],
            columns_used=columns_scaled
            )
        
        ## Tag
        df = dp.tag_string_data_to_integer_new(df, df_etiquetas, columns_scaled=columns_scaled)
        
        ## Clean data 2
        df, df_fill = dp.clean_data_2_new(df=df, scaler_loaded=scaler, columns_scaled=columns_scaled, columns_selected=d_hiper['selected_columns']) # Antes usaba comp_to_select pero me quedaban los partidos de todas las comp en predicciones.xlsx
        
        ## Select
        df = dp.select_data_new(df, d_hiper['selected_columns'])
        
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
        logger.critical(f"n_model: {n_model} model_name: {model_name} iteration_date: {iteration_date}")
        
        # Predigo con modelo cargado
        if predict_missing:
            df_filled = df_fill.copy() 
            d_strategy = {'prob_dp': 0, 'curva': 'linear', 'm': 10, 'b': 0}
        else:  
            df_filled = pd.concat([df_c1['copiado_formaciones'], df_fill.loc[:, ['player_emergency_fill', 'emergency_fill']]], axis=1) 
            d_strategy = lo.load_modeling_hyperparameters()

        # Predigo sobre proximos partidos usando modelo cargado
        y_pred_proba, y_pred = mo.predict_model(model=lo.load_model(), X_test=df)
        df_pred_proba = mo.construct_predictions_dataframe(model=lo.load_model(), X_test=df, y_pred_prob=y_pred_proba, y_pred=y_pred)

        # Concateno dfs + Reformateo teams
        df_predicciones = mo.prepare_dataframe_to_assess_with_roi(df_pred_proba=df_pred_proba, df_match=df_match, df_match_odds=df_match_odds, df_filled=df_filled)         # Concateno todos los dfs en uno solo 
        # df_predicciones = df_predicciones[df_predicciones['id_competition'].isin(comp_public)] # Filtro partidos para quedarme solo con los de competencias publicas.

        # Aplico estrategia de apuesta
        bs = betting_strategy.BettingStrategy(country=country, iteration_date=iteration_date_dt)

        # Pasarle "strategy" prod o bien ya pasarle el d_params...
        if  isinstance(d_strategy, dict):
            logger.warning("Aplico MISMA estrategia A TODOS LOS RDOS. ")
            df = bs.apply_strategy(df_predicciones, param_dict=d_strategy)
        else:
            logger.warning("Aplico estrategia DISTINTA POR RESULTADO. ")
            df = bs.apply_strategy_by_result(df_predicciones, df_hiper=d_strategy)

        # Aplico reduccion a stake
        if porc_m is not None and not predict_missing:
            df['stake_to_bet'] = df['stake_to_bet'] * porc_m

            # Determino confidence margin
            df = asses_model.determine_confidence_margin(df)

            # Reducir stake si confidence_margin < threshold
            df.loc[(df['confidence_margin'] < 0.04) & (df['result_to_bet'] != 0), 'stake_to_bet'] *= 0.2
        
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
    d_run_type = {
        'missing': [0],
        'predict': ['try_a_specific_model', 'predict_missing', 'prod'],
    }

    id_country = 77
    key, value = 'predict', 'try_a_specific_model'
    data_unders = False
    n_days = 0.5

    # Defino country, iteration date y modelo
    d_countries = {
        # 6: ["argentina", '2025-02-06'], 
        48: ["england", '2025-03-23'], 
        55: ["france", '2025-03-23'], 
        59: ["germany", '2025-03-23'], 
        77: ["italy", '2025-03-23'],
        148: ["spain", '2025-03-24'], 
        # 167: ["usa", '2024-12-05']
        }
    iteration_date = d_countries[id_country][1]
    d_model = {'n_model': 184, 'model_name': "LogisticRegression"} # DecisionTreeClassifier, XGBClassifier, neural_networ, SVC, LogisticRegression, MLPClassifier

    if key == 'missing':
        
        d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
        logger.warning("Extract and prepare missing matches")
        df = main(d_run, id_country, iteration_date=iteration_date, export=d_run['export']) 

    elif key == 'predict':
        d_run = {'run_missing': False, 'data_unders': data_unders, 'data_prep': True, 'modeling': True, 'export': False} 
            
        if value == "try_a_specific_model":
            logger.warning("Get predictions of specific model")
            df = main(d_run, id_country, iteration_date=iteration_date, n_days_max_next_matches=n_days, d_model=d_model, export=False) 

        elif value == 'predict_missing':
            df = main(d_run, id_country, iteration_date=iteration_date, predict_missing=True, d_model=d_model, export=False) 

        elif value == "prod":
            df = main(d_run, id_country, iteration_date=iteration_date, n_days_max_next_matches=n_days, porc_m=1, export=True) 

    if isinstance(df, pd.DataFrame):
        df.to_excel(f"{directorio}/predicciones.xlsx")