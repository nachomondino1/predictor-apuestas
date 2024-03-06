# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
import re
import os
from main import DataUnderstanding, DataPreparation
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_next_matches, extract_missing_data
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation.construct_data import determine_mean_in_last_matches_next_matches, determine_stats_columns
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
# Modeling
import pickle
from p4_modeling import asses_model


class DataUnderstandingNew():

    def __init__(self, id_country, country):
        #super().__init__(country)
        self.id_country = id_country
        self.country = country
        self.make_directories()

    def make_directories(self):
        l_directorios = [
            f'./p6_deployment/data/{self.country.lower()}/data_understanding',
            f'./p6_deployment/data/{self.country.lower()}/missing/data_understanding',    
        ]
    
        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def collect_initial_data_new(self, n_days: int = 7, export=True, _print: bool = False):

        print(" Collecting data... ")
        if _print:
            print(f' COUNTRY: {self.country} '.center(120, '#'))
        # Levanto datasets
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
        df_match = pd.read_excel(f'./p2_data_understanding/data/{self.country}/df_match.xlsx', index_col=0)

        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_teams_concat, df_coaches_concat, df_player_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
            df_match_next, df_match_player_next, df_match_odds, df_teams, df_coaches, df_player = extract_next_matches(self.id_country, self.country, id_competition, competition, is_cup, n_days=n_days)

            # Guarda datos de competition
            df_match_concat = pd.concat([df_match_concat, df_match_next], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_next], axis=0)
            df_match_odds_concat = pd.concat([df_match_odds_concat, df_match_odds], axis=0)

            df_teams_concat = pd.concat([df_teams_concat, df_teams], axis=0)
            df_coaches_concat = pd.concat([df_coaches_concat, df_coaches], axis=0)
            df_player_concat = pd.concat([df_player_concat, df_player], axis=0)

        # Exporto datasets
        if export:
            df_match_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_match_next.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_match_player_next.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_match_next_odds.xlsx', index=True)

            df_player_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_player.xlsx', index=True)
            df_teams_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_teams.xlsx', index=True)
            df_coaches_concat.to_excel(f'./p6_deployment/data/{self.country}/data_understanding/df_coaches.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat, df_player_concat, df_teams_concat, df_coaches_concat

    def collect_missing_data(self, df_match: pd.DataFrame, export: bool = True, _print=False):
        """
        Extraccion de varias competencias de un mismo country.
        """
        print(" Collecting data... ")
        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_teams_concat, df_coaches_concat, df_player_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx') # /Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx'
        df_comp_country = df_comp[df_comp['id_country'] == self.id_country]
        if _print:
            print(f' COUNTRY: {self.country} '.center(120, '#'))

        # Solo extriago las competencias que tengo en los datos viejos 
        l_competencies = df_match['id_competition'].unique()

        # POR COMPETITION (solo las que hay en df_match)
        for id_competition in l_competencies:
            
            # Obtengo nombre de competicion y is_cup
            df_comp_filt = df_comp_country[df_comp_country['id_competition'] == id_competition]  # Para extrar varios countryes?: df = df_comp[df_comp['country'].isin(l_countryes)]
            competition, is_cup = df_comp_filt['competition_flashscore'].values[0], df_comp_filt['is_cup'].values[0]
            if _print:
                print(f" Competition: {competition} ".center(120, '+'))

            # Actualizo df_match y df_match_player con los partidos faltantes
            df_match_miss, df_match_player_miss, df_match_odds_miss, df_teams_miss, df_coaches_miss, df_player_miss = extract_missing_data(self.id_country, self.country, id_competition, competition, is_cup, list(df_match.index), _print=True)
            if _print:
               print(f"Cantidad de partidos faltantes en df_match: {df_match_miss.shape[0]}")

            # Concateno dfs
            df_match_concat = pd.concat([df_match_concat, df_match_miss], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_miss], axis=0)
            df_match_odds_concat =  pd.concat([df_match_odds_concat, df_match_odds_miss], axis=0)
            df_teams_concat = pd.concat([df_teams_concat, df_teams_miss], axis=0)
            df_coaches_concat = pd.concat([df_coaches_concat, df_coaches_miss], axis=0)
            df_player_concat = pd.concat([df_player_concat, df_player_miss], axis=0)
            
        # Exporto datasets
        if export and len(df_match_miss) > 0:
            df_match_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_match_miss.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_match_player_miss.xlsx', index=True)
            df_match_odds_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_match_odds_miss.xlsx', index=True)
            df_player_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_player_miss.xlsx', index=True)
            df_teams_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_teams_miss.xlsx', index=True)
            df_coaches_concat.to_excel(f'./p6_deployment/data/{self.country}/missing/data_understanding/df_coaches_miss.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat, df_player_concat, df_teams_concat, df_coaches_concat

    def describe_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_match_odds: pd.DataFrame,  df_player:  pd.DataFrame, df_teams:  pd.DataFrame, df_coaches: pd.DataFrame):

        print("\nDescribing data... ")
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

class DataPreparationNew(DataPreparation):

    def __init__(self, id_country, country):

        super().__init__(country)
        self.id_country = id_country
        self.country = country.lower()
        self.make_directories()

    def make_directories(self):  # Pasarle direcotio o l_directorios como argumento...
        l_directorios = [
            f'./p6_deployment/data/{self.country.lower()}/missing/data_preparation',    
            f'./p6_deployment/data/{self.country.lower()}/data_preparation'   
        ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def format_data_new(self, df_match: pd.DataFrame, export: bool = True):
        """
        Arreglo el data fill_type de algunas variables.
        :param df_match: Dataframe de los datos de los partidos. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
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

        if export:
            df_match.to_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_match_form.xlsx')
        return df_match

    def clean_data_new(self, df_match, df_player: pd.DataFrame, df_teams: pd.DataFrame, export: bool = True):
        """
        Limpieza inicial de los dataframes
        :param df_match:
        :param df_match_player:
        :param df_player:
        :param export:
        :return:
        """
        start = time.time()
        print("\nCleaning new data...")

        # Dataframe player
        df_player = clean_data.prepare_text_columns(df_player, l_cols_to_process=['player_name'])

        # Dataframe teams
        df_teams = clean_data.prepare_text_columns(df_teams, l_cols_to_process=['team_name'])
        df_teams = clean_data.clean_teams_names(df_teams)  # Eliminar strings adicionales en names de equipos

        end = time.time()
        print(f"Limpieza inicial de datos en {(end - start) / 60:.1f} minutos")

        if export:
            df_player.to_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_player_cleaned.xlsx')
            df_teams.to_excel(f'./p6_deployment/data/{self.country}/data_preparation/df_teams_cleaned.xlsx')

        return df_match, df_player, df_teams

    def integrate_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export: bool = True):
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
        df = self.integrate_data(df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) 

        if export:
            df.to_excel(f'./p6_deployment/data/{self.country.lower()}/df_integrated.xlsx')
        return df

    def construct_data_new(self, df_new: pd.DataFrame, df, n_days: int, n_years_h2h: int, export: bool = True): # actualizar df_old tiene que ser df_updated...
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        print("\nConstructing new data...")
        start = time.time()

        # Variables historicas
        df = construct_data.determine_result(df, self.var_resp) # en el old para poder calcular historial
        df_new = construct_data.h2h_by_date_new_matches(df_new, df, n_years=n_years_h2h)  # TENEMOOS QUE DARLE EL DF ADICIONAL CON EL CUAL CALCULAR EL HISTORIAL SOLO PARA EL DF ORIGINAL

        # Filtro df para seleccionar ultimos x dias  ## Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada team...
        fecha_limite = datetime.datetime.now() - datetime.timedelta(days=n_days)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual
        df = df[df['date'] >= fecha_limite]
        print(df.shape)

        # Concateno df_new y df filtrado
        df_concat = pd.concat([df_new, df], axis=0)
        df_concat.to_excel("/Users/nachomondino/Desktop/df_concat.xlsx")  # Por que no genera valores a estadisticas??

        # Construyo 
        df_constructed = self.construct_data(df_concat, n_days, n_years_h2h, export=False)
        df_constructed.to_excel("/Users/nachomondino/Desktop/df_constructed.xlsx")  # Por que no genera valores a estadisticas??

        # 4) Selecciono solo los partidos nuevos de los datos construidos 
        df_new = df_constructed[df_constructed.index.isin(df_new.index)]
        df = df_constructed[df_constructed.index.isin(df.index)]
        df_new.to_excel("/Users/nachomondino/Desktop/df_new_constructed.xlsx")
        df.to_excel("/Users/nachomondino/Desktop/df_constructed_sin_new_matches.xlsx")

        # JUGADORES 
        # 3) Relleno datos no disponibles en partidos nuevos (rating formacion titular, etc) usando los partidos viejos
        df_new = copy_last_matches_mean_value(df_new, df, n_days)
        df_new.to_excel("/Users/nachomondino/Desktop/df_new_copy_mean.xlsx")
      
      
        df_new = copy_last_matches_value(df_new, df) 
        df_new.to_excel("/Users/nachomondino/Desktop/df_new_copy.xlsx")

        # # VERIFICAR QUE EL DATASET NO TIENEN NAN  --> Aca o en select_data? Para eliminar solo si no tiene datos en las variables selected y no borrar mal.
        # df_sin_dup = df_new.dropna()
        # if len(df_sin_dup) > 0:
        #     text = f"Cuidado! No se hara la prediccion para {len(df_sin_dup)} partidos puesto que tienen al menos un valor NaN y el modelo no puede tener input NaN."
        #     warnings.warn(text)
    
        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_new.to_excel(f'./p6_deployment/data/{self.country}/df_constructed.xlsx', index=True)
        return df_new

    def select_data_new(self, df, export=True, _print: bool = True):
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        if _print:
            start = time.time()
            print("\n Selecting data...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df = df.drop(['date'], axis=1)

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')
        df, df_etiquetas = format_data.convert_columns_to_int(df, df_etiquetas)  # Si o si tengo que devolver df_etiquetas?
        # df.to_excel('/Users/nachomondino/Desktop/df_codificacion_next_matches.xlsx')  #  Comprobé que codifica bien

        # Selecciono las variables que necesita el modelo
        X_test = pd.read_excel(f'./p4_modeling/data/{self.country}/X_test.xlsx', index_col=0)
        df = df[X_test.columns]

        if _print:
            end = time.time()
            print(f"Las siguientes {len(X_test.columns)} columnas son las seleccionadas: {list(X_test.columns)}")
            print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p6_deployment/data/{self.country}/df_selected.xlsx', index=True)
        return df
 

# Missing data
def read_last_version_matches(country, _print: bool = False):
    # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
    try:
        df_match = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_with_missing.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_player_with_missing.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_odds_with_missing.xlsx', index_col=0)
        # df_player = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_player_miss.xlsx', index_col=0)
        # df_teams = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_teams_miss.xlsx', index_col=0)
        # df_coaches = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_coaches_miss.xlsx', index_col=0)
        if _print:
            print('1) Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')
            print(df_match.shape, df_match_player.shape, df_match_odds.shape)
   
    except:
        df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_odds.xlsx', index_col=0)
        # df_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx', index_col=0)
        # df_teams = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams.xlsx', index_col=0)
        # df_coaches = pd.read_excel(f'./p2_data_understanding/data/{country}/df_coaches.xlsx', index_col=0)

        if _print:
            print('2) Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
            print(df_match.shape, df_match_player.shape, df_match_odds.shape)
  
    return df_match, df_match_player, df_match_odds # , df_player, df_teams, df_coaches

def read_last_version_integrated(country, _print: bool = False):
    """
    Levanta la ultima version de los datos viejos integrados (puede que ya le haya concatenado algunos partidos missing)
    Extriago df_integrated_missing si ya existe.
    """
    try:
        df_integrated = pd.read_excel(f'./p6_deployment/data/{country}/missing/data_preparation/df_integrated_with_missing.xlsx', index_col=0)
        if _print:
            print('1) Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')
            print(df_integrated.shape)
    except:
        df_integrated = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
        if _print:
            print('2) Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
            print(df_integrated.shape)
    return df_integrated
            
def convert_columns_to_float(df: pd.DataFrame, _print: bool = False):
    """
    Intenta convertir las columnas object a float
    """
    # Selecciono las columnas object
    l_columnas_a_codificar = df.select_dtypes(include=['object']).columns
    
    if _print:
        print("\n df_integrated_missing \n")
        df_integrated_missing.info()

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

# Construct_data_new
def copy_last_matches_value(df_new: pd.DataFrame, df: pd.DataFrame):
    """
    # En los partidos nuevos, rellena los datos no disponibles (necesarios antes de construir datos) con los datos de partidos anteriores.
    
    Cosas a agregar:
    - Si los dts son nan en el partido anterior, entonces buscar en el sigueinte y asi...
    """  
    print("Rellenando datos no disponibles...")
    # Tengo que agregar copiar los datos del ultimo partido para: dt
    # l_var_to_copy = ['id_coach']  # Puedo copiar attendance tambien? o la elimino?

    # Identificar las columnas con al menos un valor NaN
    l_var_to_copy = df_new.columns[df_new.isna().any()].tolist()
    print("Columnas con al menos un NaN:", l_var_to_copy)
    l_columns_sin_suffix = {re.sub(r'_(home|away)$', '', col) for col in l_var_to_copy}
    l_var_to_copy = list(l_columns_sin_suffix)
    print("Columnas con al menos un NaN:", l_var_to_copy)

    for var in l_var_to_copy:

        # Por partido nuevo
        for idx, row in df_new.iterrows():

            # Obtengo id de equipos local y visitante
            d_teams = {row['id_team_home']: 'home', row['id_team_away']: 'away'}
            # print(f"Team home: {row['id_team_home']}, Team away: {row['id_team_away']}")

            # Por equipos
            for team, home_or_away in d_teams.items():

                # Busco partidos anteriores del equipo
                df_team_matches = df.loc[(df['id_team_home'] == team) | (df['id_team_away'] == team)]

                # Busco el partido anterior del team  --> podria hacer ciclo para que busque hasta que encuentre un valor no nan...
                if len(df_team_matches) > 0:
                    fila_part_ant = df_team_matches.iloc[0]

                    # Cargo datos a partido nuevo
                    try: # e.g. id_coach_home
                        elem_a_cambiar = df_new.loc[idx, f"{var}_{home_or_away}"]
                        tit = "home" if fila_part_ant['id_team_home'] == team else "away"

                        if pd.isna(elem_a_cambiar):
                            df_new.loc[idx, f"{var}_{home_or_away}"] = fila_part_ant[f"{var}_{tit}"]  # (e.g. df_new['dt_away'] = fila_part_ant['dt_home'])
                    except:
                        pass 

                    try: # e.g. todas las de jugadores y estadisticas...
                        elem_a_cambiar = df_new.loc[idx, var]

                        if pd.isna(elem_a_cambiar):
                            df_new.loc[idx, var] = fila_part_ant[var]  # (e.g. df_new['dt_away'] = fila_part_ant['dt_home'])
                    except:
                        pass 
    return df_new

def copy_last_matches_mean_value(df_new: pd.DataFrame, df: pd.DataFrame, n_days: int):
    """
    Copio el promedio de los valores en los ultimos partidos
    # Obtengo valor para columnas promedio de jugadores promediando el valor en ultimos partidos

    # Cuidado algunas variables de jugadores ya tieneene calculado el valor y tengo que dejar ese valor. Sobretodo miss players auqnue tambein podrian serr titulares y suplentes si falta poco para el partido.
    """
    # Selecciono las variables que corresponden a jugadores
    l_columns_sin_suffix = {re.sub(r'_(home|away)$', '', elemento) for elemento in df.columns}
    # print("Columnas a agregar a df de next matches: ", l_columns_sin_suffix)
    l_var_mean_player = [col for col in l_columns_sin_suffix if re.search(r'mean_.*_player_', col)]  # r'_jug_'  --> tmb modificaba n_jug_home pero crasheaba para hacer promedio.
    print(f"Lista de columnas prom jug: {l_var_mean_player}")

    l_var_mean_player.append('dif_sum_rat_player_miss')
    l_var_mean_player.append('dif_n_player_miss')

    # Por variable mean_player
    for var in l_var_mean_player:
        print(f"\tVariable a promediar: {var}")  # (e.g. dif_mean_val_player_sub)             

        df_new = determine_mean_in_last_matches_next_matches(df_new, df, n_days=n_days, variable=var, tipo="mean") # (e.g. mean_last_match_dif_mean_val_player_sub)

    # df_new.to_excel(f'/Users/nachomondino/Desktop/df_copy_last_matchs_mean_value.xlsx', index=False)
    return df_new

def main():
    """
    Recoleccion de proximos partidos
    """
    start = time.time()

    # Definicion de variables
    var_resp, var_pred = 'result', 'predicted_result'
    # Selecciono country a extraer por terminal
    # df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    # country = str(input("Choose country to extract (e.g. England, Germany, etc): "))
    # id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]
    id_country, country = 48, "England"
    du = DataUnderstandingNew(id_country, country) # Creo objeto de clase DataPreparation
    dp = DataPreparationNew(id_country, country) # Creo objeto de clase DataPreparation

    run_missing, data_unders, data_prep, modeling = False, False, True, True
    export = True

    # Hiperparametro
    n_days = 7  # Numero de dias maximo desde hoy para extraer partidos
 
    print("\n", "#"*120, "\n", "MISSING DATA".center(120), "\n", "#"*120, "\n")
    if run_missing:  # Lo puedo correr atemporal de los proximos partidos, dado que tarda,esta bueno correlo seguido para no tener una gran extraccion y tarde mucho
        
        # Obtengo ultima version de df_match y df_match_player
        df_match, df_match_player, df_match_odds = read_last_version_matches(country)  # df_match, df_match_player, df_match_odds, df_player, df_teams, df_coaches = read_last_version_matches(country)
        df_player_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_sofifa.xlsx', index_col=0) # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_player_fifa_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_fifa_sofifa.xlsx') # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
        df_teams_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams_sofifa.xlsx', index_col=0)

        # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
        df_match_miss, df_match_player_miss, df_match_odds_miss, df_player_miss, df_teams_miss, df_coaches_miss = du.collect_missing_data(df_match, export=export)
        print(f"Cantidad de partidos missing extraidos: {len(df_match_miss)}")
    
        # Si hay partidos missing que no extraje aun
        if len(df_match_miss) > 0 :

            # Guardo df_match y df_match_player concatenados con missing
            if export:
                df_match_odds = convert_columns_to_float(df_match_odds)  # Formateo odds a float (no se por que son object)
                df_concat_match = pd.concat([df_match, df_match_miss], axis=0)
                df_concat_match_player = pd.concat([df_match_player, df_match_player_miss], axis=0)
                df_concat_match_odds = pd.concat([df_match_odds, df_match_odds_miss], axis=0)
                df_concat_match.to_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_with_missing.xlsx')
                df_concat_match_player.to_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_player_with_missing.xlsx')
                df_concat_match_odds.to_excel(f'./p6_deployment/data/{country}/missing/data_understanding/df_match_odds_with_missing.xlsx')

            # Preparo datos
            df_match_miss, df_match_player_miss, df_player_fifa_sofifa = dp.format_data(df_match_miss, df_match_player_miss, df_player_fifa_sofifa, export=False)
            df_match_miss, df_match_player_miss, df_player_miss, df_teams_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match_miss, df_match_player_miss, df_player_miss, df_teams_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False)
            df_integrated_missing = dp.integrate_data(df_match_miss, df_match_player_miss, df_player_miss, df_teams_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) 
            df_integrated_missing = convert_columns_to_float(df_integrated_missing)  # Formateo estadisticas a float (no se por que son object)
      
            # Concateno missing y old
            df_integrated = read_last_version_integrated(country)  # Levanto df_integrated de partidos viejos
            df_int_with_missing = pd.concat([df_integrated, df_integrated_missing], axis=0)

            if export:
                df_integrated_missing.to_excel(f'./p6_deployment/data/{country}/missing/data_preparation/df_integrated_missing.xlsx', index=True)
                df_int_with_missing.to_excel(f'./p6_deployment/data/{country}/missing/df_integrated_with_missing.xlsx', index=True)

        else:
            print("Ya se habian extriado todos los partidos missing. Aun no hay partidos nuevos.")
            df_int_with_missing = pd.read_excel(f'./p6_deployment/data/{country}/missing/df_integrated_with_missing.xlsx', index_col=0)
        
    else:
        df_int_with_missing = pd.read_excel(f'./p6_deployment/data/{country}/missing/df_integrated_with_missing.xlsx', index_col=0)
        # df_int_with_missing = pd.read_excel('./p3_data_preparation/data/england/df_integrated_etiquetado.xlsx', index_col=0)
        print(df_int_with_missing.head(3))
        print(df_int_with_missing.shape)

    print("\n DATA UNDERSTANDING \n".center(240, "#"))
    if data_unders:
        # Extriago datos o los levanto
        df_match, df_match_player, df_match_odds, df_player, df_teams, df_coaches = du.collect_initial_data_new(n_days=n_days, export=export)

        if len(df_match) > 0:
            # Describo datos
            du.describe_data_new(df_match, df_match_player, df_match_odds,  df_player, df_teams, df_coaches)
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
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country.lower()}/data_understanding/df_match_next_odds.xlsx', index_col=0)
        df_teams = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_teams.xlsx', index_col=0)
        df_player = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_player.xlsx', index_col=0)
        df_coaches = pd.read_excel(f'./p6_deployment/data/{country}/data_understanding/df_coaches.xlsx', index_col=0)
        print("\n DF MATCH \n", df_match.head(2))
        print("\n DF MATCH PLAYER \n", df_match_player.head(2))

        # SOFIFA
        df_player_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
        df_teams_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)

        # du.describe_data(df_match, df_match_player, df_match_odds, df_player)

    print("\n DATA PREPARATION \n".center(240, "#"))
    if data_prep:

        # Hiperparametros (tengo que usar los mismos que con los que construi los datos con los que entrene el modelo)
        df_hiper_prep = pd.read_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx')
        n_days = int(df_hiper_prep['n_days'].values[0])
        n_years_h2h = int(df_hiper_prep['n_years_h2h'].values[0])

        # df = pd.read_excel(f'./p6_deployment/data/{country}/data_preparation/df_integrated.xlsx', index_col=0)
        # print(df.head(2))

        # Preparo el dataset para el analisis
        df_match = dp.format_data_new(df_match, export=False)
        df_match, df_player, df_teams = dp.clean_data_new(df_match, df_player, df_teams, export=export)
        df = dp.integrate_data_new(df_match, df_match_player, df_player, df_teams, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=export)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc
        df = dp.construct_data_new(df, df_int_with_missing, n_days=n_days, n_years_h2h=n_years_h2h, export=export)
        df = dp.select_data_new(df, export=export)

    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p6_deployment/data/{country}/df_selected.xlsx', index_col=0)
        print(df.head(2), df.shape)

    print("\n MODELING \n".center(240, "#"))
    if modeling:

        # Levanto datasets
        df_match_next = pd.read_excel(f'p6_deployment/data/{country.lower()}/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'./p6_deployment/data/{country.lower()}/data_understanding/df_match_next_odds.xlsx', index_col=0)

        # Levanto modelo ya entrenado
        loaded_model = pickle.load(open(f"./p4_modeling/data/{country}/modelo.pkl", "rb"))

        # Realizo predicciones sobre los nuevos partidos
        y_pred_prob = loaded_model.predict_proba(df)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad 
        df_pred_proba = pd.DataFrame({var_pred: y_pred, f'prob_class_{loaded_model.classes_[1]}': y_pred_prob[:, 1], f'prob_class_{loaded_model.classes_[0]}': y_pred_prob[:, 0], f'prob_class_{loaded_model.classes_[2]}': y_pred_prob[:, 2]}, index=df.index)

        # Concateno conjunto de datos
        df_predicciones = pd.concat([df_match_next, df_match_odds, df_pred_proba], axis=1)

        # Determino estrategia de inversion
        df_predicciones = asses_model.calculate_odds_model(df_predicciones)
        df = asses_model.construct_stake_modified(df_predicciones, stake_base=7000, type_stake='linear', m=3, b=-1)

        # Revierto etiquetas para tener nombres de equipos en vez de ids
        df_teams = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams.xlsx', index_col=0)
        df = format_data.revert_columns_from_int(df, df_teams, _print=True)
        df.to_excel(f'./p6_deployment/data/{country}/predicciones.xlsx')

    end = time.time()
    print(f"Main_next_matches en {(end - start)/60:.1f} minutos")
    
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()