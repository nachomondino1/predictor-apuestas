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
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
# Modeling
import pickle
import joblib
from p4_modeling import asses_model


class DataUnderstandingNew():

    def __init__(self, id_country, country, export: bool = True):
        #super().__init__(country)
        self.id_country = id_country
        self.country = country.lower()
        self.make_directories()
        self.export = export

    def make_directories(self):
        l_directorios = [
            f'./p6_deployment/data/{self.country}/data_understanding',
            f'./p6_deployment/data/{self.country}/missing/data_understanding',    
        ]
    
        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def collect_initial_data_new(self, n_days: int = 7, _print: bool = False):
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
        if _print:
            print(f' COUNTRY: {self.country} '.center(120, '#'))
        # Levanto datasets
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
        df_match = pd.read_excel(f'./p2_data_understanding/data/{self.country}/df_match.xlsx', index_col=0)

        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        l_competencies = df_match['id_competition'].unique()
        if _print:
            print(f"Competiciones a extraer del pais {self.country}: {l_competencies}")

        # POR COMPETITION
        for id_competition in l_competencies:
            
            df_comp_filt = df_comp[(df_comp['id_country'] == self.id_country) & (df_comp['id_competition'] == id_competition)]  # Para extrar varios countryes?: df = df_comp[df_comp['country'].isin(l_countryes)]
            competition, is_cup = df_comp_filt['competition_flashscore'].values[0], df_comp_filt['is_cup'].values[0]
            if _print:
                print(f" Competition: {competition} ".center(120, '+'))

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
            df_match_miss, df_match_player_miss, df_match_odds_miss = extract_data(self.id_country, self.country, id_competition, competition, is_cup, n_seasons_max=1, l_ids_already_collected=list(df_match.index), export=False)
            if _print:
               print(f"Cantidad de partidos faltantes en df_match: {df_match_miss.shape[0]}")

            # Concateno dfs
            df_match_concat = pd.concat([df_match_concat, df_match_miss], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_miss], axis=0)
            df_match_odds_concat =  pd.concat([df_match_odds_concat, df_match_odds_miss], axis=0)
            
        # Exporto datasets
        if self.export and len(df_match_miss) > 0:
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

    def __init__(self, id_country, country, export: bool = True):

        super().__init__(country)
        self.id_country = id_country
        self.country = country.lower()
        self.make_directories()
        self.export = export

    def make_directories(self):  # Pasarle direcotio o l_directorios como argumento...
        l_directorios = [
            f'./p6_deployment/data/{self.country}/missing/data_preparation',    
            f'./p6_deployment/data/{self.country}/data_preparation'   
        ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def format_data_new(self, df_match: pd.DataFrame):
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
        ## Capacity & Attendance  # Deberia fallar attendance porque aun no existe el dato...
        df_match = format_data.convert_capacity_to_int(df_match)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if self.export:
            df_match.to_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_match_form.xlsx')
        return df_match

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
        df_match = df_match.drop(['attendance'], axis=1)

        # Preparo columnas texto
        columns_to_keep = [col for col in df_match.columns if df_match[col].dtype == 'object' and 'id_' not in col]
        df_match = clean_data.prepare_text_columns(df_match, l_cols_to_process=columns_to_keep) # Ver si selecciona bien.. # ['team_home', 'team_away', 'coach_home', 'coach_away', 'venue', 'referee'])
        columns_player_names = list(df_match_player.filter(like='player_name').columns)
        df_match_player = clean_data.prepare_text_columns(df_match_player, l_cols_to_process=columns_player_names)
     
        end = time.time()
        print(f"Limpieza inicial de datos en {(end - start) / 60:.1f} minutos")

        if self.export:
            df_match.to_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_match_cleaned.xlsx')
            df_match_player.to_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_match_player_cleaned.xlsx')

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
            df.to_excel(f'./p6_deployment/data/{self.country}/df_integrated.xlsx')
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

        # En caso que aun no se cuente con las formaciones, asigno promedio en ultimos partidos
        l_player_cols = [col for col in df_old_int.columns if re.search(r'_player_', col) and "_miss" not in col]  # Selecciono las variables que corresponden a jugadores  # Al parecer funcionaria # Puesto que los missing pueden ser nulos efectivamente y estan siempre pre-partido...
        df_new, df_copiado_formaciones = fillna_with_mean_in_last_matches(df_new, df_old_int, cols_to_fill=l_player_cols)            
            
        # Copio valores en ultimos partidos (deberia copiar solo referee y coaches)
        l_var_to_copy = ['referee', 'id_coach_home', 'id_coach_away']  # coach_home y coach_away no harian falta pues se usa solo para integrar.  
        df_new, df_copiado = fillna_with_last_match_value(df_new, df_old_int, cols_to_fill=l_var_to_copy) 

        if self.export:
            df_copiado_formaciones.to_excel(f"./p6_deployment/data/{self.country}/df_copiado_formaciones.xlsx", index=True)
            df_copiado.to_excel(f"./p6_deployment/data/{self.country}/df_copiado_ref_and_coaches.xlsx", index=True)
            df_new.to_excel(f"./p6_deployment/data/{self.country}/df_data_not_avai_fill.xlsx", index=True)

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
        df_constructed = self.construct_data(df_concat, n_days, n_years_h2h, segun_localia=segun_localia, without_h2h=True, export=False)

        # Separo datos construidos entre los proximos partidos y los ya jugados
        df_new = df_constructed[df_constructed.index.isin(df_new.index)]

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")
        if _print:
            df_concat.to_excel("/Users/nachomondino/Desktop/df_concat.xlsx")
            df_constructed.to_excel("/Users/nachomondino/Desktop/df_constructed.xlsx")
            df_new.to_excel("/Users/nachomondino/Desktop/df_new_constructed.xlsx")

        if self.export:
            df_new.to_excel(f'./p6_deployment/data/{self.country}/df_constructed.xlsx', index=True)

        return df_new

    def tag_string_data_to_integer_new(self, df: pd.DataFrame, tager_loaded, _print: bool = True):
        """
        Utilizando las mismas etiquetas que cuando se entreno el modelo para el pais, convierto columnas string a integer
        """
        print("\nTagging string data to integer..")
        # Elimino season para no etiquetarla?
       
        df = df.drop(['season'], axis=1)

        # Intento levantar df_etiquetas con etiquetas nuevas # Una vez que df_eti_2 funcione ok, Exportar df_etiquetas_2 y usar este en lugar de df_etiquetas puesto que esta mas actualizado...
        # df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')
        ''' 
        try:    
            df_etiquetas = pd.read_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_etiquetas_actualizado.xlsx')
            print("Levento etiquetas actualizado")
        except:
            df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')
            print("Levento etiquetas viejo puesto que no hay uno actualizado")
        '''

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int_2(df, tager_loaded)  # Si o si tengo que devolver df_etiquetas?
        if _print:
            df.to_excel('/Users/nachomondino/Desktop/df_codificacion_next_matches.xlsx')

        if self.export:
            df_etiquetas.to_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_etiquetas_actualizado.xlsx', index=False) 
        return df

    def clean_data_2_new(self, df: pd.DataFrame, scaler_loaded, columns_used):

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
        if _print:
            start = time.time()
            print("\n Selecting data...")

        # Selecciono las variables que necesita el modelo
        df = df[l_columns]
        if _print:
            end = time.time()
            print(f"Las siguientes {len(l_columns)} columnas son las seleccionadas: {list(l_columns)}")
            print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if self.export:
            df.to_excel(f'./p6_deployment/data/{self.country}/df_selected.xlsx', index=True)
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
        columnas_player_miss = ['dif_mean_last_match_n_matches_last_days', 'dif_mean_age_player_miss', 'dif_mean_hei_player_miss', 'dif_mean_int_rep_player_miss', 'dif_sum_rat_player_miss', 'dif_sum_val_player_miss']         # A veces, las columnas dif de jugadores ausentes es NaN dado que uno delos equipos no tiene jug ausentes. Podria evitarlo.
        columns_to_fill = [col for col in (columnas_player_miss + columnas_h2h) if col in df.columns]
        if columns_to_fill:
            df[columns_to_fill] = df[columns_to_fill].fillna(0)

        # Elimino partidos con al menos un NaN value
        df_sin_dup = df.dropna()
        if len(df) != len(df_sin_dup):
            text = f"Cuidado! No se hara la prediccion para {len(df_sin_dup)} partidos puesto que tienen al menos un valor NaN y el modelo no puede tener input NaN."
            warnings.warn(text)

        if self.export: 
            df_sin_dup.to_excel(f'./p6_deployment/data/{self.country}/df_selected_nan.xlsx', index=True)

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
    # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
    try:
        df_match = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_with_missing.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_player_with_missing.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_odds_with_missing.xlsx', index_col=0)
        df_integrated = pd.read_excel(f'./p6_deployment/data/{country}/missing/df_integrated_with_missing.xlsx', index_col=0)
        if _print:
            print('1) Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')
            print(df_match.shape, df_match_player.shape, df_match_odds.shape, df_integrated.shape)
   
    except FileNotFoundError:
        df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_odds.xlsx', index_col=0)
        df_integrated = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
        if _print:
            print('2) Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
            print(df_match.shape, df_match_player.shape, df_match_odds.shape, df_integrated.shape)
  
    return df_match, df_match_player, df_match_odds, df_integrated
            
def convert_columns_to_float(df: pd.DataFrame, _print: bool = False):
    """
    Intenta convertir las columnas object a float
    """
    # Selecciono las columnas object
    l_columnas_a_codificar = df.select_dtypes(include=['object']).columns
    
    if _print:
        print("\n df_integrated_missing \n")
        df.info()

    # Por columna object
    for col in l_columnas_a_codificar:

        # Intento convertirla a float
        try:
            df[col] = df[col].astype(float)
            if _print:
                print(f"Se convirtio la columna {col} a float!")
        except:
            pass
    return df
  
# Data preparation
def get_construct_hyper(country, n_mejor_modelo = None):
    """
    Levanto los mismos hiperparametros con los que construi los datos con los que entrene el modelo.

    # Parameters:
        country: Nombre del pais al cual extraer proximos partidos. (str)
        n_mejor_modelo: Numero de iteracion del modelo a utilizar para predecir resultados de proximos partidos. (int)

    # Returns:
        Hiperparametros utilizados para construir el dataframe con el cual se entrenó el modelo. (tuple)
    """
    # 1) Si entrene el modelo en main.py
    if n_mejor_modelo is None:
        df_hiper_prep = pd.read_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx')
        n_dias_ult_part = int(df_hiper_prep['n_days'].values[0])
        n_years_h2h = int(df_hiper_prep['n_years_h2h'].values[0])
        segun_localia = df_hiper_prep['segun_localia'].values[0]
        
    # 2) Si entrene el modelo en find_best_hyper.py
    else:
        ## Levanto iteraciones en find_best_hyper.py
        df_iteration = pd.read_excel(f"./main_find_best_hyper/data/{country}/df_iteration.xlsx")
        row_ite = df_iteration[df_iteration['n_iteration'] == n_mejor_modelo]

        ## Obtengo los hiperparametros utilizados para construir datos cuando entrené el modelo
        n_dias_ult_part, n_years_h2h, segun_localia = int(row_ite['n_dias_ult_part'].values[0]), int(row_ite['n_anios_hist'].values[0]), row_ite['segun_localia'].values[0]
        print(f"Numero de dias ult part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h} ; segun_localia: {segun_localia}")

    return n_dias_ult_part, n_years_h2h, segun_localia

def load_scaler(country, n_model, n_dias_ult_part, n_years_h2h, segun_localia):

    # 1) Si entrene el modelo en main.py
    if n_model is None:
        scaler, columns_used = joblib.load(f"./p3_data_preparation/data/{country}/scaler_model.pkl")
        # scaler_loaded = joblib.load(f"./p3_data_preparation/data/{country}/scaler_model.pkl")
    else:
        scaler, columns_used = joblib.load(f'./main_find_best_hyper/data/{country}/data_preparation/scaler_model_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}.pkl')
        # scaler_loaded = joblib.load(f'./main_find_best_hyper/data/{country}/data_preparation/scaler_model_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}.pkl')

    return scaler, columns_used

def load_tager(country, n_model, n_dias_ult_part, n_years_h2h, segun_localia):
    # 1) Si entrene el modelo en main.py
    if n_model is None:
        tager_loaded = pd.read_excel(f"./p3_data_preparation/data/{country}/df_etiquetas.xlsx")
    else:
        tager_loaded = pd.read_excel(f'./main_find_best_hyper/data/{country}/data_preparation/df_etiquetas_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}.xlsx')
    return tager_loaded

def get_columns_of_trained_model(country, n_mejor_modelo = None): # La tengo que probar... Deberia funcionar ok.
    """
    Levanto las columnas utilizadas para entrenar el modelo

    # Parameters:
        country: Nombre del pais al cual extraer proximos partidos. (str)
        n_mejor_modelo: Numero de iteracion del modelo a utilizar para predecir resultados de proximos partidos. (int)

    # Returns:
        Lista de columnas utilizadas para entrenar el modelo. (list)
    """
    # 1) Si entrene el modelo en main.py
    if n_mejor_modelo is None:
        ## Levanto columnas utilizadas para entrenar el modelo
        df_selected = pd.read_excel(f'./p3_data_preparation/data/{country}/df_selected.xlsx', index_col=0)
        df_selected = df_selected.drop(['result'], axis=1)
        l_columns = list(df_selected.columns)
        
    # 2) Si entrene el modelo en find_best_hyper.py
    else:
        ## Levanto iteraciones en find_best_hyper.py
        df_iteration = pd.read_excel(f"./main_find_best_hyper/data/{country}/df_iteration.xlsx")
        row_ite = df_iteration[df_iteration['n_iteration'] == n_mejor_modelo]
        l_columns = row_ite['X_columns'].values[0]  # Lista como string
        l_columns = eval(l_columns)  # Convierto lista string a lista

    print(f"Las {len(l_columns)} columnas seleccionadas con las que se entrenó el modelo son: {l_columns}")
    return l_columns

# Modeling
def load_model(country, n_mejor_modelo = None):
    """
    Levanto modelo con el cual predecir los proximos partidos.
    # Parameters:
        country: Nombre del pais al cual extraer proximos partidos. (str)
        n_mejor_modelo: Numero de iteracion del modelo a utilizar para predecir resultados de proximos partidos. (int)

    # Returns:
        Modelo para predecir resultado de proximos partidos.
    """
    # Si se levanta de main.py
    if n_mejor_modelo is None:
        path = f"./p4_modeling/data/{country}/modelo.pkl"
    # Si se levanta de find_best_hyper.py
    else:
        path = f"./main_find_best_hyper/data/{country}/modeling/{n_mejor_modelo}_model.pkl"

    loaded_model = pickle.load(open(path, "rb"))
    return loaded_model

################################################### MAIN ###################################################
def main():
    """
    Recoleccion de proximos partidos
    """
    start = time.time()

    # Definicion de variables
    var_resp, var_pred = 'result', 'predicted_result'
    run_missing, data_unders, data_prep, modeling = False, False, True, True
    export = True
    country = "england"  # country = str(input("Choose country to extract (e.g. England, Germany, etc): "))
    numero_mejor_modelo = 1 # ING: 10 ARG: 10 (LOG) (podria ser o 46 (SVM) o 90 (GB) pero me gusta mas LOG pq su prec es mas alta sinceramente ademas de un ROI mayor)
    n_days_max_next_matches = 2  # Numero de dias maximo desde hoy para extraer partidos

    # Determino id_country
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    id_country = df_countries[df_countries['country_name'] == country.capitalize()]['id_country'].values[0]

    # Creo instancias de clases
    du = DataUnderstandingNew(id_country, country, export) # Creo objeto de clase DataPreparation
    dp = DataPreparationNew(id_country, country, export) # Creo objeto de clase DataPreparation

    #______________________________________________ MISSING DATA ______________________________________________#
    print("\n", "#"*120, "\n", "MISSING DATA".center(120), "\n", "#"*120, "\n")
    if run_missing:  # Lo puedo correr atemporal de los proximos partidos, dado que tarda,esta bueno correlo seguido para no tener una gran extraccion y tarde mucho
        
        # Obtengo ultima version de df_match y df_match_player
        df_match, df_match_player, df_match_odds, df_integrated = read_last_version_matches(country)
        df_player_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_sofifa.xlsx', index_col=0) # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_player_fifa_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_fifa_sofifa.xlsx') # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_teams_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams_sofifa.xlsx', index_col=0)

        # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
        df_match_miss, df_match_player_miss, df_match_odds_miss = du.collect_missing_data(df_match)
        print(f"Cantidad de partidos missing extraidos: {len(df_match_miss)}")

        # Si hay partidos missing que no extraje aun
        if len(df_match_miss) > 0:

            # Guardo df_match y df_match_player concatenados con missing
            if export:
                # df_match_odds = convert_columns_to_float(df_match_odds)  # Formateo odds a float (no se por que son object)
                df_concat_match = pd.concat([df_match, df_match_miss], axis=0)
                df_concat_match_player = pd.concat([df_match_player, df_match_player_miss], axis=0)
                df_concat_match_odds = pd.concat([df_match_odds, df_match_odds_miss], axis=0)
                df_concat_match.to_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_with_missing.xlsx')
                df_concat_match_player.to_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_player_with_missing.xlsx')
                df_concat_match_odds.to_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_odds_with_missing.xlsx')
                print(f"Shape de df_match with missing: {len(df_concat_match)}")

            # Preparo datos
            df_match_miss, df_match_player_miss, df_player_fifa_sofifa = dp.format_data(df_match_miss, df_match_player_miss, df_player_fifa_sofifa, export=False)
            df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False)
            df_integrated_missing = dp.integrate_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) 
            # df_integrated_missing = convert_columns_to_float(df_integrated_missing)  # Formateo estadisticas a float (no se por que son object)
      
            # Concateno missing y old
            df_int_with_missing = pd.concat([df_integrated, df_integrated_missing], axis=0)

            if export:
                df_integrated_missing.to_excel(f'./p6_deployment/data/{country}/missing/data_preparation/df_integrated_missing.xlsx', index=True)
                df_int_with_missing.to_excel(f'./p6_deployment/data/{country}/missing/df_integrated_with_missing.xlsx', index=True)

        else:
            print("Ya se habian extriado todos los partidos missing. Aun no hay partidos nuevos.")
            df_int_with_missing = pd.read_excel(f'./p6_deployment/data/{country}/missing/df_integrated_with_missing.xlsx', index_col=0)
        
    else:
        try:
            df_int_with_missing = pd.read_excel(f'./p6_deployment/data/{country}/missing/df_integrated_with_missing.xlsx', index_col=0)
        except FileNotFoundError:
            df_int_with_missing = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
            print("No hay un dataframe integrado con missing aun. Tuve que levantar el df_integrated de main.py...")
        print(df_int_with_missing.head(3))
        print(df_int_with_missing.shape)

    #______________________________________________ DATA UNDERSTANDING ______________________________________________#
    print("\n DATA UNDERSTANDING \n".center(240, "#"))
    if data_unders:
        # Extriago datos de los partidos en los proximos dias
        df_match, df_match_player, df_match_odds = du.collect_initial_data_new(n_days=n_days_max_next_matches)

        if len(df_match) > 0:
            # Describo datos
            du.describe_data_new(df_match, df_match_player, df_match_odds)
        else:
            data_prep, modeling, export = False, False, False
            print("No hay proximos partidos para los cuales predecir su resultado.")
    
        # SOFIFA
        df_player_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
        df_teams_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)

    elif data_prep:
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

        du.describe_data_new(df_match, df_match_player, df_match_odds)

    #______________________________________________ DATA PREPARATION ______________________________________________#
    print("\n DATA PREPARATION \n".center(240, "#"))
    if data_prep:

        # Levanto hiperparametros de construct y columnas con las que se entrenó el modelo
        n_dias_ult_part, n_years_h2h, segun_localia = get_construct_hyper(country=country, n_mejor_modelo=numero_mejor_modelo)
        tager_loaded = load_tager(country, numero_mejor_modelo, n_dias_ult_part, n_years_h2h, segun_localia)
        scaler_loaded, columns_used = load_scaler(country, numero_mejor_modelo, n_dias_ult_part, n_years_h2h, segun_localia)
        l_columns = get_columns_of_trained_model(country=country, n_mejor_modelo=numero_mejor_modelo)
        fecha_limite = datetime.datetime.now() - datetime.timedelta(days=n_dias_ult_part)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual

        # df = pd.read_excel(f'./p6_deployment/data/{country}/df_integrated.xlsx', index_col=0)
        # print(df.head(2))

        # Filtro df para seleccionar ultimos x dias  ## Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada team...
        df_old_int = df_int_with_missing.sort_values(by='date', ascending=False) # Ordeno por fecha ascendente
        df_old_int_filt = df_old_int[df_old_int['date'] >= fecha_limite]  # Funciona ok, tiene los partidos missing.
        print("Shape de partidos ya jugados con los cuales construir los proximos partidos: ", df_old_int_filt.shape)
        
        # Preparo el dataset para el analisis
        df_match = dp.format_data_new(df_match)
        df_match, df_match_player = dp.clean_data_new(df_match, df_match_player)
        df = dp.integrate_data_new(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc
        df, df_c1, df_c2 = dp.fill_data_not_available_yet(df, df_old_int_filt)
        df = dp.construct_data_new(df, df_old_int, df_old_int_filt, n_days=n_dias_ult_part, n_years_h2h=n_years_h2h, segun_localia=segun_localia)
        df = dp.tag_string_data_to_integer_new(df, tager_loaded)
        df = dp.clean_data_2_new(df, scaler_loaded, columns_used)
        df = dp.select_data_new(df, l_columns)
        df = dp.treat_nan_values_new(df)
        print("Shape Dataframe antes de Modeling(): ", df.shape)

    elif not data_unders:
        df_match = pd.read_excel(f'p6_deployment/data/{country}/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_match_next_odds.xlsx', index_col=0)
        df_c1 = pd.read_excel(f'./p6_deployment/data/{country}/df_copiado_formaciones.xlsx', index_col=0)
        df_c2 = pd.read_excel(f'./p6_deployment/data/{country}/df_copiado_ref_and_coaches.xlsx', index_col=0)
       
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p6_deployment/data/{country}/df_selected_nan.xlsx', index_col=0)
        print(df.head(2), df.shape)

    #______________________________________________ MODELING ______________________________________________#
    print("\n MODELING \n".center(240, "#"))
    if modeling:

        # Levanto datasets
        df_teams = pd.read_excel(f'p3_data_preparation/data/{country}/integrate_data/df_teams.xlsx', index_col=0)
        df_match = df_match.loc[:, ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition']]

        # Levanto modelo ya entrenado
        loaded_model = load_model(country, n_mejor_modelo=numero_mejor_modelo)

        # Realizo predicciones sobre los nuevos partidos
        y_pred_prob = loaded_model.predict_proba(df)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad 
        df_pred_proba = pd.DataFrame({var_pred: y_pred, f'prob_class_{loaded_model.classes_[1]}': y_pred_prob[:, 1], f'prob_class_{loaded_model.classes_[0]}': y_pred_prob[:, 0], f'prob_class_{loaded_model.classes_[2]}': y_pred_prob[:, 2]}, index=df.index)

        # Concateno conjunto de datos
        df_match_odds = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta
        df_predicciones = pd.concat([df_match, df_match_odds, df_pred_proba, df_c1, df_c2, df], axis=1)

        # Determino estrategia de apuuesta
        df = asses_model.calculate_dif_proba_in_predicted_result(df_predicciones)
        df = asses_model.determine_result_to_bet(df)
        df = asses_model.determine_stake_to_bet(df, stake_base=1, type_relation='linear', m=20, b=0)

        # Revierto etiquetas para tener nombres de equipos en vez de ids
        d_mapeo = dict(zip(df_teams.index, df_teams['team_name']))        
        df['id_team_home'] = df['id_team_home'].replace(d_mapeo)
        df['id_team_away'] = df['id_team_away'].replace(d_mapeo)
        df.to_excel(f'./p6_deployment/data/{country}/predicciones.xlsx')

    end = time.time()
    print(f"Main_next_matches en {(end - start)/60:.1f} minutos")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()