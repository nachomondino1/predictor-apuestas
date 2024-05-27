# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
import re
import os
from main import DataPreparation
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_next_matches, extract_data
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
# Modeling
import pickle
import joblib
from p4_modeling import asses_model


class DataUnderstandingNew():

    def __init__(self, id_country, country, export: bool = True):
        self.id_country = id_country
        self.country = country.lower()
        self.export = export
        self.make_directories()

    def make_directories(self):
        l_directorios = [
            f'./p6_deployment/data/{self.country}/data_understanding',
            f'./p6_deployment/data/{self.country}/missing/data_understanding/all',
            f'./p6_deployment/data/{self.country}/missing/old_concat_missing',   
            f'./p6_deployment/data/{self.country}/missing/data_preparation/all',
        ]
    
        if self.export:
            for directorio in l_directorios:
                if not os.path.exists(directorio):
                    # Si no existe, crear el directorio
                    os.makedirs(directorio)

    def collect_initial_data_new(self, l_competencies, n_days: int = 7, _print: bool = True):
        """
        Extraccion de datos de los partidos en los proximos dias en todas las competiciones del pais.

        # Parameters
        n_days: Numero de dias desde hoy para recolectar proximos partidos
        export:
        _print: 

        # Returns
        dfs...
        """
        print(" Collecting data... ")
        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
        df_comp_country = df_comp[df_comp['id_country'] == self.id_country]
        if _print:
            print(f' COUNTRY: {self.country} '.center(120, '#'))
            print(f"Competiciones a extraer del pais {self.country}: {l_competencies}")

        # POR COMPETITION
        for id_competition in l_competencies:
            
            df_comp_filt = df_comp_country[df_comp_country['id_competition'].astype(int) == int(id_competition)]
            competition, is_cup = df_comp_filt['competition_flashscore'].values[0], df_comp_filt['is_cup'].values[0]
            if _print:
                print(f" Competition: {competition} id_comp: {id_competition}".center(120, '+'))

            # Extraigo proximos partidos
            df_match_next, df_match_player_next, df_match_odds = extract_next_matches(self.id_country, self.country, id_competition, competition, is_cup, n_days=n_days)

            # Guarda datos de competition
            df_match_concat = pd.concat([df_match_concat, df_match_next], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_next], axis=0)
            df_match_odds_concat = pd.concat([df_match_odds_concat, df_match_odds], axis=0)

        # Exporto datasets
        if self.export:
            df_match_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_match_next.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_match_player_next.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_match_next_odds.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat

    def collect_missing_data(self, df_match: pd.DataFrame, _print: bool = False):
        """
        Extraccion de varias competencias de un mismo country.
        """
        print(" Collecting data... ")
        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
        df_comp_country = df_comp[df_comp['id_country'] == self.id_country]
        if _print:
            print(f' COUNTRY: {self.country} '.center(120, '#'))
            print(f'Competencias de {self.country}: \n {df_comp_country}')

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
            df_match_miss, df_match_player_miss, df_match_odds_miss = extract_data(self.id_country, self.country, id_competition, competition, is_cup, n_seasons_max=1, l_ids_already_collected=l_ids_extracted, export=False)
            if _print:
                print(f"Cantidad de partidos faltantes en df_match: {df_match_miss.shape[0]}")

            # Concateno dfs
            df_match_concat = pd.concat([df_match_concat, df_match_miss], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_miss], axis=0)
            df_match_odds_concat =  pd.concat([df_match_odds_concat, df_match_odds_miss], axis=0)
            
        # Exporto datasets
        if self.export:
            df_match_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_match_miss.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_match_player_miss.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_match_odds_miss.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat

    def describe_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame):

        print("\nDescribing data... ")
        print("\n DF_MATCH \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match)
        describe_data.verificar_unicidad_registros(df_match) # Verifico unicidad de registros segun campos id

        print("\n DF_MATCH_PLAYER \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_player)

        print("\n DF_MATCH_ODDS \n".center(240, "-"))
        describe_data.getting_to_know_data(df_match_odds)

class DataPreparationNew(DataPreparation):

    def __init__(self, country, export: bool = True):
        self.country = country
        self.ruta_base = f"./p6_deployment/data/{country}/data_preparation"
        self.export = export
        self.make_directories()
        super().__init__(country)

    def make_directories(self):  # Pasarle direcotio o l_directorios como argumento...
        l_directorios = [
            f'{self.ruta_base}/format_data',
            f'{self.ruta_base}/clean_data',
            f'{self.ruta_base}/fill_data',
            f'{self.ruta_base}/construct_data',
        ]

        if self.export:
            for directorio in l_directorios:
                if not os.path.exists(directorio):
                    # Si no existe, crear el directorio
                    os.makedirs(directorio)

    def format_data_new(self, df_match: pd.DataFrame, df_match_odds: pd.DataFrame):
        """
        Arreglo el data fill_type de algunas variables.

        # Parameters:
            df_match: Dataframe de los datos de los partidos. (DataFrame)
            
        # Returns:
            Dataframe pasado como parametro formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormating data...")

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
            df_match.to_excel(f'{self.ruta_base}/format_data/df_match_form.xlsx')
            df_match_odds.to_excel(f'{self.ruta_base}/format_data/df_match_odds_form.xlsx')
        return df_match, df_match_odds

    def clean_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame):
        """
        Limpieza inicial de los dataframes

        # Parameters:
            df_match:
            df_match_player:
        
        # Returns:
            Dataframe
        """
        start = time.time()
        print("\nCleaning new data...")

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
            df_match.to_excel(f'{self.ruta_base}/clean_data/df_match_cleaned.xlsx')
            df_match_player.to_excel(f'{self.ruta_base}/clean_data/df_match_player_cleaned.xlsx')

        return df_match, df_match_player

    def integrate_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa):
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
            df.to_excel(f'{self.ruta_base}/df_integrated.xlsx')
        return df

    def fill_data_not_available_yet(self, df_new: pd.DataFrame, df_old_int, _print: bool = False):
        """
        Relleno datos aun no disponibles debido a que aun falta mas de 30 min para el partido. Asi, poder predecir a pesar de tener datos aun no 
        disponibles.

        # Parameters:
            df_new:
            df_old_int: 

        # Returns:
            Dataframe pasado como parametro con datos aun no disponibles reemplazados por valores en ultimos partidos. 
        """
        print("\n Rellenando datos aun no disponibles...")
        print("Shape de partidos ya jugados con los cuales rellenar datos aun no disponibles en los proximos partidos: ", df_old_int.shape)

        # En caso que aun no se cuente con las formaciones, asigno promedio en ultimos partidos
        l_player_cols = [col for col in df_old_int.columns if re.search(r'_player_', col) and "_miss" not in col]  # Selecciono las variables que corresponden a jugadores
        df_new, df_copiado_formaciones = fillna_with_mean_in_last_matches(df_new, df_old_int, cols_to_fill=l_player_cols)            
            
        # Copio valores en ultimos partidos (deberia copiar solo referee y coaches)
        l_var_to_copy = ['referee', 'id_coach_home', 'id_coach_away']
        df_new, df_copiado = fillna_with_last_match_value(df_new, df_old_int, cols_to_fill=l_var_to_copy) 

        if self.export:
            df_copiado_formaciones.to_excel(f"{self.ruta_base}/fill_data/df_copiado_formaciones.xlsx", index=True)
            df_copiado.to_excel(f"{self.ruta_base}/fill_data/df_copiado_ref_and_coaches.xlsx", index=True)
            df_new.to_excel(f"{self.ruta_base}/df_filled.xlsx", index=True)

        return df_new, df_copiado_formaciones, df_copiado

    def construct_data_new(self, df_new: pd.DataFrame, df_old_int, df_old_int_filt, n_days: int, n_years_h2h: int, segun_localia: bool, _print: bool = False):
        """
        Construye nuevos datos a partir de un dataframe existente.

        # Parameters
        df_new: Dataframe con proximos partidos ya integrado.
        df: Dataframe con partidos ya jugados e integrado. (DataFrame)
        N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        
        # Returns
        Dataframe construido. (DataFrame)
        """
        print("\nConstructing new data...")
        start = time.time()

        # Construyo historial entre si
        df_old_int = construct_data.determine_result(df_old_int, self.var_resp) # en el old para poder calcular historial
        df_new = construct_data.h2h_by_date_new_matches(df_new, df_old_int, n_years=-1, segun_localia=True)  # TENEMOOS QUE DARLE EL DF ADICIONAL CON EL CUAL CALCULAR EL HISTORIAL SOLO PARA EL DF ORIGINAL
        df_new = construct_data.h2h_by_date_new_matches(df_new, df_old_int, n_years=-1, segun_localia=False)
        df_new = construct_data.h2h_by_date_new_matches(df_new, df_old_int, n_years=n_years_h2h, segun_localia=True)
        df_new = construct_data.h2h_by_date_new_matches(df_new, df_old_int, n_years=n_years_h2h, segun_localia=False)

        # Concateno df_new y df filtrado y construyo
        df_concat = pd.concat([df_new, df_old_int_filt], axis=0)
        print("Shape de partidos ya jugados con los cuales construir los proximos partidos: ", df_old_int_filt.shape)
        df_constructed = self.construct_data(df_concat, n_days, n_years_h2h, segun_localia=segun_localia, without_h2h=True, export=False)

        # Separo datos construidos entre los proximos partidos y los ya jugados
        df_new = df_constructed[df_constructed.index.isin(df_new.index)]

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if self.export:
            df_old_int_filt.to_excel(f'{self.ruta_base}/construct_data/df_matches_to_construct_historic_values.xlsx', index=True)
            df_new.to_excel(f'{self.ruta_base}/df_constructed.xlsx', index=True)

        return df_new

    def tag_string_data_to_integer_new(self, df: pd.DataFrame, df_etiquetas_loaded, _print: bool = True):
        """
        Utilizando las mismas etiquetas que cuando se entreno el modelo para el pais, convierto columnas string a integer
        """
        print("\nTagging string data to integer..")
        # Elimino season para no etiquetarla?
        df = df.drop(['season'], axis=1)

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int_already_tagged(df, df_etiquetas_loaded)  # Si o si tengo que devolver df_etiquetas?
    
        if self.export:
            df.to_excel(f'{self.ruta_base}/df_tagged.xlsx', index=True) 
            df_etiquetas.to_excel(f'{self.ruta_base}/df_etiquetas_actualizado.xlsx', index=False) 
        return df

    def clean_data_2_new(self, df: pd.DataFrame, scaler_loaded, columns_used, comp_to_select):

        # Para evitar ciertas competencias
        df = df[df['id_competition'].isin(comp_to_select)]

        # Separo en X e y
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp] # Separo en X e y

        # Selecciono las mismas caracteristicas con las que entrene el scaler (sino, falla)
        X = X.loc[:, columns_used]

        # Transforma los nuevos datos de predicción utilizando el StandardScaler cargado
        X_scaled = scaler_loaded.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=columns_used, index=X.index)

        df = pd.concat([X_scaled_df, y], axis=1)
        return df

    def select_data_new(self, df: pd.DataFrame, l_columns: list, _print: bool = True):
        """
        Selecciona las variables que necesita el modelo ya entrenado.

        # Parameters:
            df: Dataframe con los proximos partidos. (DataFrame)
            l_columns: Lista de columnas a seleccionar. (list)
        
        # Returns
            Dataframe con las variables seleccionadas. (DataFrame)
        """
        start = time.time()
        print("\n Selecting data...")

        # Selecciono las variables que necesita el modelo
        df = df[l_columns]

        end = time.time()
        print(f"Las siguientes {len(l_columns)} columnas son las seleccionadas: {list(l_columns)}")
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if self.export:
            df.to_excel(f'{self.ruta_base}/df_selected.xlsx', index=True)
        return df
    
    def treat_nan_values_new(self, df: pd.DataFrame):
        """
        Remoción de valores NaN
       
        # Parameters:
            df: Dataframe al cual remover NaN values. (DataFrame)
        
        # Returns:
            df: Dataframe pasado como parametro sin registros con al menos un NaN value. (DataFrame)
        """
        # Rellenar NaN en algunas columnas espeecificas
        columnas_h2h = [col for col in df.columns if 'h2h_' in col]  # Reemplazo historiales nan por 0
        columnas_player_miss = [col for col in df.columns if '_miss' in col]  # columnas_player_miss = ['dif_mean_age_player_miss', 'dif_mean_hei_player_miss', 'dif_mean_int_rep_player_miss']
        historic_columns =  [col for col in df.columns if 'mean_last_match' in col]  # El tema es que no tiene que rellenar normalmente y al hacerlo podria tapar un "error"
        columns_to_fill = [col for col in (columnas_h2h + columnas_player_miss + historic_columns) if col in df.columns]
        if columns_to_fill:
            df[columns_to_fill] = df[columns_to_fill].fillna(0)

        # Elimino partidos con al menos un NaN value
        df_sin_dup = df.dropna()
        if len(df) != len(df_sin_dup):
            text = f"Cuidado! De los {len(df)} partidos, no se hará la prediccion para {len(df)-len(df_sin_dup)} partidos puesto que tienen al menos un valor NaN y el modelo no puede tener input NaN."
            warnings.warn(text)

        if self.export: 
            df_sin_dup.to_excel(f'{self.ruta_base}/df_selected_nan.xlsx', index=True)

        return df_sin_dup
    

# fill_data_not_available_yet()
def fillna_with_mean_in_last_matches(df_new: pd.DataFrame, df: pd.DataFrame, cols_to_fill, _print: bool = False): 
    """
    Para cada variable de cols_to_fill, reemplaza valores NaN por el valor promedio de dicha variable en los ultimos partidos.

    # Parameters
        df_new: Dataframe con los proximos partidos. (DataFrame)
        df: DataFrame con partidos ya jugados para rellenar df_new (DataFrame)
        cols_to_fill: Columnas a reemplazar valores NaN. (list)

    # Returns
        Dataframe df_new pasado como parametro habiendo reemplazado en cols_to_fill NaN por promedio en ultimos partidos.
    """
    print(f"Remplazando NaN por valor promedio en ultimos partidos en {cols_to_fill}...")
    df_copiado_form = pd.DataFrame(columns=["copiado_formaciones"], index=df_new.index)
    
    # Por variable mean_player 
    for variable in cols_to_fill: # (e.g. mean_val_player_sub_home, mean_rat_player_start_away)  

        # En caso que ningun proximo partido tenga formaciones, creo la columna jugador correspondiente
        if variable not in df_new.columns:
            df_new[variable] = np.nan
        if _print: 
            print(f"\nVariable a promediar: {variable}")
        col_sin_suffix = variable.replace("_home", "").replace("_away", "")

        # Por partido nuevo
        for id_match, row in df_new.iterrows():

            # Si el valor es nan en el partido
            if pd.isna(row[variable]):

                team = row['id_team_home'] if "_home" in variable else row['id_team_away']

                # Busco promedio en ultimos partidos
                df_matches_home_team = df[df['id_team_home'] == team]
                df_matches_away_team = df[df['id_team_away'] == team]
                if _print:
                    print("\n DF_MATCH_TEAM_HOME \n", df_matches_home_team.loc[:, ['date', 'id_team_home', 'id_team_away', f'{col_sin_suffix}_home']].head(5))
                    print("\n DF_MATCH_TEAM_AWAY \n", df_matches_away_team.loc[:, ['date', 'id_team_home', 'id_team_away', f'{col_sin_suffix}_away']].head(5))

                # Obtener los valores de la variable para los partidos en casa y fuera de casa
                values_home = df_matches_home_team[f'{col_sin_suffix}_home'].values
                values_away = df_matches_away_team[f'{col_sin_suffix}_away'].values

                # Remover los valores NaN
                values_home_clean = values_home[~np.isnan(values_home)]
                values_away_clean = values_away[~np.isnan(values_away)]

                # Calcular el número total de partidos
                total_partidos = (len(values_home_clean) + len(values_away_clean))
                suma = (np.sum(values_home_clean) + np.sum(values_away_clean))

                # Si hay al menos un valor que promediar, guardo promedio
                if total_partidos > 0:
                    df_new.loc[id_match, variable] = suma / total_partidos
                    df_copiado_form.loc[id_match, 'copiado_formaciones'] = 1
                    df_copiado_form.loc[id_match, variable] = suma / total_partidos
                    if _print:
                        print(f"Valor a rellenar: {suma / total_partidos} en {variable}")

    return df_new, df_copiado_form

def fillna_with_last_match_value(df_new: pd.DataFrame, df: pd.DataFrame, cols_to_fill: list, _print: bool = False):
    """
    En los partidos nuevos, rellena los datos no disponibles con los datos de partidos anteriores.
    """  
    print(f"Remplazando NaN por valor en ultimo partido en {cols_to_fill}...")
    df_copiado = pd.DataFrame(columns=['copiado_avoid_nan'], index=df_new.index)

    for var in cols_to_fill:

        # En caso que ningun proximo partido tenga formaciones, creo la columna jugador correspondient
        if var not in df_new.columns:
            df_new[var] = np.nan
        if _print:
            print(f"\nVariable a promediar: {var}")

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

# Missing data
def read_last_version_matches(country, _print: bool = True):
    # Esta bien levantar df_integrated aca tambien. para asegurarme que todos los missing estan en df_integrated tambien.
    # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
    try:
        df_match = pd.read_excel(f'./p6_deployment/data/{country}/missing/old_concat_missing/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p6_deployment/data/{country}/missing/old_concat_missing/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country}/missing/old_concat_missing/df_match_odds.xlsx', index_col=0)
        df_integrated = pd.read_excel(f'./p6_deployment/data/{country}/missing/old_concat_missing/df_integrated.xlsx', index_col=0)
        if _print:
            print('1) Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')

    except FileNotFoundError:
        df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_odds.xlsx', index_col=0)
        df_integrated = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
        if _print:
            print('2) Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
    print(df_match.shape, df_match_player.shape, df_match_odds.shape, df_integrated.shape)
    return df_match, df_match_player, df_match_odds, df_integrated

def load_df_etiquetas(country, n_model, ruta_base, d):

    # Intento levantar df_etiquetas con etiquetas nuevas # Una vez que df_eti_2 funcione ok, Exportar df_etiquetas_2 y usar este en lugar de df_etiquetas puesto que esta mas actualizado...    
    try:    
        df_etiquetas = pd.read_excel(f'./p6_deployment/data/{country}/data_preparation/df_etiquetas_actualizado.xlsx')
        print("1) Levento etiquetas actualizado")

    except FileNotFoundError:
        print("2) Levento etiquetas viejo puesto que no hay uno actualizado")
        # Si se levanta de main.py
        if n_model is None:
            path_tag = f"./p3_data_preparation/data/{country}/df_etiquetas.xlsx"

        # Si se levanta de find_best_hyper.py
        else:
            n_ult_part, n_years_h2h, segun_localia = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia']
            path_tag = f'{ruta_base}/data_preparation/df_etiquetas_{n_ult_part}_{n_years_h2h}_{segun_localia}.xlsx'

        df_etiquetas = pd.read_excel(path_tag, index_col=0)
        
    print(df_etiquetas.head(3))
    return df_etiquetas

def load_data_preparation_hyperparameters(country, n_model, ruta_base):
    """
    Cargo hiperparametros de DataPreparation()
    """
    d = {}

    # Si se levanta de main.py
    if n_model is None:
        
        row_hiper = pd.read_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx')

        ## Levanto columnas utilizadas para entrenar el modelo
        df_selected = pd.read_excel(f'./p3_data_preparation/data/{country}/df_selected.xlsx', index_col=0)
        df_selected = df_selected.drop(['result'], axis=1)
        selected_columns = list(df_selected.columns)

    else:
        df_iteration = pd.read_excel(f"{ruta_base}/df_iteration.xlsx")
        row_hiper = df_iteration[df_iteration['n_iteration'] == n_model]

        ## Levanto columnas utilizadas para entrenar el modelo
        selected_columns = eval(row_hiper['X_columns'].values[0])  # eval() para pasar de string a lista

    # Guardo hiperparametros en diccionario
    ## Construct_data
    d['n_dias_ult_part'] = int(row_hiper['n_dias_ult_part'].values[0])
    d['n_years_h2h'] = int(row_hiper['n_anios_hist'].values[0])
    d['segun_localia'] = row_hiper['segun_localia'].values[0]
    ## Clean_data_2
    n_years_to_select = row_hiper['n_years_to_select'].values[0]
    d['n_years_to_select'] = None if pd.isna(n_years_to_select) else int(n_years_to_select) # Si n_years_to_select es NaN, lo paso de np.nan a None
    d['comp_to_select'] = row_hiper['comp_to_select'].values[0]
    ## Select_data
    d['selected_columns'] = selected_columns

    print("Hiperparametros cargados:")
    for key, value in d.items():
        print(f'\t {key}: {value}')
    return d

def load_modeling_hyperparameters(country, n_model, ruta_base):
    """
    Cargo hiperparametros de Modeling()
    """
    d = {}

    # Si se levanta de main.py
    if n_model is None:
        row_hiper_bet_strat = pd.read_excel(f'./p4_modeling/data/{country}/modeling/df_hiper_mod.xlsx')

    else:
        df_iteration_prod = pd.read_excel(f"{ruta_base}/df_iteration_prod.xlsx", index_col=0)
        row_hiper_bet_strat = df_iteration_prod.loc[n_model]

    # Guardo hiperparametros en diccionario
    d['thr_prob_min'] = row_hiper_bet_strat['thr_prob_min_best']
    d['curva'] = row_hiper_bet_strat['curva']
    param1 = row_hiper_bet_strat['param1']
    param2 = row_hiper_bet_strat['param2']
    d['curva_m'] = param1 if d['curva'] == 'linear' else None
    d['curva_b'] = param2 if d['curva'] == 'linear' else None
    d['curva_p1'] = eval(param1) if d['curva'] != 'linear' else None
    d['curva_p2'] = eval(param2) if d['curva'] != 'linear' else None

    print("Hiperparametros cargados:")
    for key, value in d.items():
        print(f'\t {key}: {value}')
    return d

def load_models(country, n_model, ruta_base, d):

    # Si se levanta de main.py
    if n_model is None:
        path_scaler = f"./p3_data_preparation/data/{country}/scaler_model.pkl"
        path_model = f"./p4_modeling/data/{country}/modelo.pkl"

    # Si se levanta de find_best_hyper.py
    else:
        n_ult_part, n_years_h2h, segun_localia, n_years_sel, comp = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia'], d['n_years_to_select'], d['comp_to_select']
        path_scaler = f'{ruta_base}/data_preparation/scaler_model_{n_ult_part}_{n_years_h2h}_{segun_localia}_{n_years_sel}_{comp}.pkl'
        path_model = f"{ruta_base}/modeling/{n_model}_model.pkl"

    scaler, columns_scaled = joblib.load(path_scaler)
    loaded_model = pickle.load(open(path_model, "rb"))
    return scaler, columns_scaled, loaded_model
    
def concat_and_export_old_with_missing(df_match, df_match_player, df_match_odds, df_match_miss, df_match_player_miss, df_match_odds_miss, BASE_PATH):
    """
    Exporto datos de partidos con los que entreno el modelo y los partidos missing.
    """
    # Concateno old (que puede tener ya algunos missing) y nuevos missing
    df_concat_match = pd.concat([df_match, df_match_miss], axis=0)
    df_concat_match_player = pd.concat([df_match_player, df_match_player_miss], axis=0)
    df_concat_match_odds = pd.concat([df_match_odds, df_match_odds_miss], axis=0)

    # Exporto datos
    df_concat_match.to_excel(f'{BASE_PATH}/df_match.xlsx')
    df_concat_match_player.to_excel(f'{BASE_PATH}/df_match_player.xlsx')
    df_concat_match_odds.to_excel(f'{BASE_PATH}/df_match_odds.xlsx')
    print(f"Shape de df_match with missing: {len(df_concat_match)}")

def concat_and_export_all_missing(df_match_miss, df_match_player_miss, df_match_odds_miss, BASE_PATH):
    """
    Guardo los nuevos partidos missing con los que ya tenia
    """
    # Si aun no extraje partidos missing
    try: 
        df_match_miss_comp = pd.read_excel(f'{BASE_PATH}/df_match_miss.xlsx', index_col=0)
        df_match_player_miss_comp = pd.read_excel(f'{BASE_PATH}/df_match_player_miss.xlsx', index_col=0)
        df_match_odds_miss_comp = pd.read_excel(f'{BASE_PATH}/df_match_odds_miss.xlsx', index_col=0)

        df_match_miss_comp_ct = pd.concat([df_match_miss_comp, df_match_miss], axis=0)
        df_match_player_miss_comp_ct = pd.concat([df_match_player_miss_comp, df_match_player_miss], axis=0)
        df_match_odds_miss_comp_ct = pd.concat([df_match_odds_miss_comp, df_match_odds_miss], axis=0)

    # Si es la primera vez que extraigo partidos missing
    except FileNotFoundError:
        df_match_miss_comp_ct = df_match_miss
        df_match_player_miss_comp_ct = df_match_player_miss
        df_match_odds_miss_comp_ct = df_match_odds_miss
        pass

    # Exporto datos
    df_match_miss_comp_ct.to_excel(f'{BASE_PATH}/df_match_miss.xlsx', index=True)
    df_match_player_miss_comp_ct.to_excel(f'{BASE_PATH}/df_match_player_miss.xlsx', index=True)
    df_match_odds_miss_comp_ct.to_excel(f'{BASE_PATH}/df_match_odds_miss.xlsx', index=True)

def read_data_of_best_model(country):
    
    df_best_models = pd.read_excel("./main_find_best_hyper/data/df_best_models.xlsx")
    row_country = df_best_models[df_best_models['country'] == country]
    n_model = int(row_country['n_model'].values[0])
    iteration_date_str = row_country['iteration_date'].values[0]
    iteration_date_dt = pd.to_datetime(iteration_date_str, format='%Y-%m-%d').date()  # con .date() saco hora y minutos
    print(f"COUNTRY: {country} --> n_model: {n_model} ; iteration_date: {iteration_date_dt}")
    return n_model, iteration_date_dt

################################################### MAIN ###################################################
def main(d_run:dict, country:str, n_days_max_next_matches:int = 7, export:bool = True):
    """
    Recoleccion de proximos partidos, preparacion y prediccion
    """
    start = time.time()

    # Determino id_country
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    id_country = df_countries[df_countries['country_name'] == country.capitalize()]['id_country'].values[0]

    # Creo objetos de clases
    du = DataUnderstandingNew(id_country, country, export) # Creo objeto de clase DataUnderstanding
    dp = DataPreparationNew(country=country, export=export) # Creo objeto de clase DataPreparation

    #______________________________________________ MISSING DATA ______________________________________________#
    print("\n", "#"*120, "\n", "MISSING DATA".center(120), "\n", "#"*120, "\n")
    if d_run['run_missing']: 
        
        # Levanto datos
        df_match, df_match_player, df_match_odds, df_integrated = read_last_version_matches(country) # Obtengo ultima version de df_match y df_match_player
        df_player_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_sofifa.xlsx', index_col=0) # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_player_fifa_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_fifa_sofifa.xlsx') # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_teams_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams_sofifa.xlsx', index_col=0)

        # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
        df_match_miss, df_match_player_miss, df_match_odds_miss = du.collect_missing_data(df_match)
        print(f"Cantidad de partidos missing extraidos: {len(df_match_miss)}")

        # Si hay partidos missing que no extraje aun
        if len(df_match_miss) > 0:

            if export:
                # Guardo datos con los que entrenó el modelo y los missing
                # df_match_odds = format_data.convert_columns_to_float(df_match_odds)  # Formateo odds a float (no se por que son object)
                concat_and_export_old_with_missing(df_match, df_match_player, df_match_odds, df_match_miss, df_match_player_miss, df_match_odds_miss, BASE_PATH=f"./p6_deployment/data/{country}/missing/old_concat_missing")
                concat_and_export_all_missing(df_match_miss, df_match_player_miss, df_match_odds_miss, BASE_PATH=f"./p6_deployment/data/{country}/missing/data_understanding/all")

            # Preparo datos
            df_match_miss, df_match_player_miss, df_player_fifa_sofifa = dp.format_data(df_match_miss, df_match_player_miss, df_player_fifa_sofifa, export=False)
            df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False)
            df_integrated_missing = dp.integrate_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) 
            # df_integrated_missing = format_data.convert_columns_to_float(df_integrated_missing)  # Formateo estadisticas a float (no se por que son object)

            # Concateno missing y old (que puede tener algunos missing ya)
            df_int_with_missing = pd.concat([df_integrated, df_integrated_missing], axis=0)

            # Levanto partidos missing ya integrados
            try:
                df_integrated_missing_all = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_preparation/all/df_integrated_missing.xlsx', index_col=0)
            except FileNotFoundError:
                df_integrated_missing_all = pd.DataFrame()
            # Concateno partidos missing ya integrados con los nuevos partidos missing integrados
            df_integrated_missing_all = pd.concat([df_integrated_missing_all, df_integrated_missing], axis=0)
            
            if export:
                df_integrated_missing.to_excel(f'./p6_deployment/data/{country}/missing/data_preparation/df_integrated_missing.xlsx', index=True)
                df_integrated_missing_all.to_excel(f'./p6_deployment/data/{country}/missing/data_preparation/all/df_integrated_missing.xlsx', index=True)
                df_int_with_missing.to_excel(f'./p6_deployment/data/{country}/missing/old_concat_missing/df_integrated.xlsx', index=True)

        else:
            print("Ya se habian extriado todos los partidos missing. Aun no hay partidos nuevos.")
            df_int_with_missing = pd.read_excel(f'./p6_deployment/data/{country}/missing/old_concat_missing/df_integrated.xlsx', index_col=0)
        
    else:
        try:
            df_int_with_missing = pd.read_excel(f'./p6_deployment/data/{country}/missing/old_concat_missing/df_integrated.xlsx', index_col=0)
        except FileNotFoundError:
            df_int_with_missing = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
            print("No hay un dataframe integrado con missing aun. Tuve que levantar el df_integrated de main.py...")
   
    print(f"\nShape of df_integrated_with_missing (siempre es un poco menor a df_match_with_missing pero no se por qué): {df_int_with_missing.shape}")  # Calculo que debe ser por la eliminacion de partidos con goles="-" que hice de df_integrated en main.py
    print(df_int_with_missing.head(3))

    # Defino variables  # Podria llevarlo despues de Data Understanding solo si no usase "comp_to_select" en la extraccion para filtrar las competencias a extraer.
    n_model, iteration_date_dt = read_data_of_best_model(country)
    ruta_base = f"./main_find_best_hyper/data/{country}/{iteration_date_dt}"

    # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
    d_hiper = load_data_preparation_hyperparameters(country, n_model, ruta_base)
    scaler, columns_scaled, loaded_model = load_models(country, n_model, ruta_base, d_hiper)
    df_etiquetas = load_df_etiquetas(country, n_model, ruta_base, d_hiper)
    comp_to_select = eval(d_hiper['comp_to_select'])

    #______________________________________________ DATA UNDERSTANDING ______________________________________________#
    print("\n DATA UNDERSTANDING \n".center(240, "#"))
    if d_run['data_unders']:
        # Extriago datos de los partidos en los proximos dias
        df_match, df_match_player, df_match_odds = du.collect_initial_data_new(l_competencies=comp_to_select, n_days=n_days_max_next_matches)
        
        # SOFIFA
        df_player_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
        df_teams_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)

        if len(df_match) > 0:
            du.describe_data_new(df_match, df_match_player, df_match_odds)
        else:
            d_run['data_prep'], d_run['modeling'], export = False, False, False
            print("No hay proximos partidos para los cuales predecir su resultado.")
    
    elif d_run['data_prep']:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_match_player_next.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_match_next_odds.xlsx', index_col=0)
        print("\n DF MATCH \n", df_match.head(2))
        print("\n DF MATCH PLAYER \n", df_match_player.head(2))

        # SOFIFA
        df_player_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
        df_teams_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)
        print("\n df_player_sofifa \n", df_player_sofifa.head(2))
        print("\n df_player_fifa_sofifa \n", df_player_fifa_sofifa.head(2))

        # du.describe_data_new(df_match, df_match_player, df_match_odds)

    #______________________________________________ DATA PREPARATION ______________________________________________#
    print("\n DATA PREPARATION \n".center(240, "#"))
    if d_run['data_prep']:

        # df = pd.read_excel(f'./p6_deployment/data/{country}/df_constructed.xlsx', index_col=0)
        # print(df.head(2))

        # Formateo, limpio e integro proximos partidos 
        df_match, df_match_odds = dp.format_data_new(df_match, df_match_odds)
        df_match, df_match_player = dp.clean_data_new(df_match, df_match_player)
        df = dp.integrate_data_new(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc
        
        # Relleno datos aun no disponibles con datos en ultimos partidos 
        days_to_filt = d_hiper['n_dias_ult_part'] * 2 if d_hiper['segun_localia'] == True else d_hiper['n_dias_ult_part']
        df_old_int = df_int_with_missing.sort_values(by='date', ascending=False) # Ordeno por fecha ascendente
        fecha_limite_seguridad = datetime.datetime.now() - datetime.timedelta(days=days_to_filt)  # uso mas dias por si justo no hay partidos dentro de "n_dias_ult_part"
        df_old_int_to_fill = df_old_int[df_old_int['date'] >= fecha_limite_seguridad]  # Funciona ok, tiene los partidos missing.
        df, df_c1, df_c2 = dp.fill_data_not_available_yet(df, df_old_int_to_fill)

        # Construyo datos usando "df_old_int" para poder construir variables historicas
        print(f"n_dias_ult_part: {d_hiper['n_dias_ult_part']} ; Segun localia: {d_hiper['segun_localia']} --> days_to_filt: {days_to_filt}" )
        fecha_minima = df['date'].min()
        fecha_limite = fecha_minima - datetime.timedelta(days=days_to_filt) 
        df_old_int_to_construct = df_old_int[df_old_int['date'] >= fecha_limite]  # Filtro df para seleccionar ultimos x dias. Minimizo tiempo de computo
        df = dp.construct_data_new(df, df_old_int, df_old_int_to_construct, n_days=d_hiper['n_dias_ult_part'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'])
        
        # Sigo con la preparacion solo con proximos partidos: Etiquetado, limpieza, seleccion y tratamiento de NaN values
        df = dp.tag_string_data_to_integer_new(df, df_etiquetas)
        df = dp.clean_data_2_new(df, scaler, columns_scaled, comp_to_select)
        df = dp.select_data_new(df, d_hiper['selected_columns'])
        df = dp.treat_nan_values_new(df)
        print("Shape Dataframe antes de Modeling(): ", df.shape)

    elif not d_run['data_unders']:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p6_deployment/data/{country}/data_preparation/df_selected_nan.xlsx', index_col=0)
        print(df.head(2), df.shape)

        df_match = pd.read_excel(f'p6_deployment/data/{country}/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_match_next_odds.xlsx', index_col=0)
        df_c1 = pd.read_excel(f'./p6_deployment/data/{country}/data_preparation/fill_data/df_copiado_formaciones.xlsx', index_col=0)
        df_c2 = pd.read_excel(f'./p6_deployment/data/{country}/data_preparation/fill_data/df_copiado_ref_and_coaches.xlsx', index_col=0)
    
    #______________________________________________ MODELING ______________________________________________#
    print("\n MODELING \n".center(240, "#"))
    if d_run['modeling']:

        # Cargo hiperparametros
        d_hiper_mod = load_modeling_hyperparameters(country, n_model, ruta_base)

        # Levanto datasets
        df_teams = pd.read_excel(f'p3_data_preparation/data/{country}/integrate_data/df_teams.xlsx', index_col=0)
        df_match = df_match.loc[:, ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition']]

        # Realizo predicciones sobre los nuevos partidos
        y_pred_prob = loaded_model.predict_proba(df)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad 
        df_pred_proba = pd.DataFrame({'predicted_result': y_pred, f'prob_class_{loaded_model.classes_[1]}': y_pred_prob[:, 1], f'prob_class_{loaded_model.classes_[0]}': y_pred_prob[:, 0], f'prob_class_{loaded_model.classes_[2]}': y_pred_prob[:, 2]}, index=df.index)

        # Concateno conjunto de datos
        df_match_odds = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta
        df_predicciones = pd.concat([df_match, df_match_odds, df_pred_proba, df_c1, df_c2, df], axis=1)

        # Determino estrategia de apuuesta
        func = lambda x: x if x is None else int(x)  # Para formatear m y b (Falla para españa) --> lo deberia implementar cdo levanto hiper
        df = asses_model.calculate_dif_proba_in_predicted_result(df_predicciones)
        df = asses_model.determine_result_to_bet(df, thr_prob_min=d_hiper_mod['thr_prob_min'])
        df = asses_model.determine_stake_to_bet(df, stake_base=1, type_relation=d_hiper_mod['curva'], m=func(d_hiper_mod['curva_m']), b=func(d_hiper_mod['curva_b']), p1=d_hiper_mod['curva_p1'], p2=d_hiper_mod['curva_p2'])

        # Revierto etiquetas para tener nombres de equipos en vez de ids
        d_mapeo = dict(zip(df_teams.index, df_teams['team_name']))        
        df['id_team_home'] = df['id_team_home'].replace(d_mapeo)
        df['id_team_away'] = df['id_team_away'].replace(d_mapeo)

        if export:
            df.to_excel(f'./p6_deployment/data/{country}/predicciones.xlsx', index=True)

    end = time.time()
    print(f"Main_next_matches en {(end - start)/60:.1f} minutos")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Defino condiciones del analisis
    country = "argentina"
    n_days_max_next_matches = 1 # Numero de dias maximo desde hoy para extraer partidos
    d_run = {'run_missing': False, 'data_unders': True, 'data_prep': True, 'modeling': True, 'export': True}

    # Extraigo, preparo y predigo proximos partidos
    main(d_run, country, n_days_max_next_matches, export=d_run['export'])