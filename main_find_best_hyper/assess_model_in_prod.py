# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
import os
from main import DataPreparation
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_data
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
# Modeling
import pickle
import joblib
from p4_modeling import asses_model
from p4_modeling.asses_model import calculate_roi_by_betting_strategy


class DataUnderstandingNew_2():

    def __init__(self, id_country, country, iteration_date, export: bool = True):
        #super().__init__(country)
        self.id_country = id_country
        self.country = country.lower()  # Proximamente lo podre quitar, ahora ya le paso la ruta base y listo
        self.iteration_date = iteration_date
        self.export = export
        self.make_directories()

    def make_directories(self):
        l_directorios = [
            f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_understanding',
        ]
    
        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

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
        if self.export:
            df_match_concat.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_understanding/df_match_next.xlsx', index=True)
            df_match_player_concat.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_understanding/df_match_player_next.xlsx', index=True)
            df_match_odds_concat.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_understanding/df_match_next_odds.xlsx', index=True)

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

class DataPreparationNew_2(DataPreparation):

    def __init__(self, id_country, country, iteration_date, export: bool = True):

        self.id_country = id_country
        self.country = country.lower()  # Proximamente lo podre quitar, ahora ya le paso la ruta base y listo
        self.iteration_date = iteration_date
        self.export = export
        self.make_directories()
        super().__init__(country) # si esta arriba, como que no se inician los atributos de la clase hija... y me tira error.


    def make_directories(self):  # Pasarle direcotio o l_directorios como argumento...
        l_directorios = [
            f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation',
            f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/modeling',
        ]

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
        ## Ball posession --> LA AGREGO TEMPORALMENTE. No la necesito en realidad puesto que no tengo estadisticas en realidad.
        df_match = format_data.convert_ball_possession_to_int(df_match)
        ## Capacity & Attendance  # Deberia fallar attendance porque aun no existe el dato...
        df_match = format_data.convert_capacity_to_int(df_match)

        # Convierto columnas a float
        df_match = convert_columns_to_float(df_match)
        df_match_odds = convert_columns_to_float(df_match_odds)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if self.export:
            df_match.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_match_form.xlsx')
            df_match_odds.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_match_odds_form.xlsx')
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
        df_match = df_match.drop(['attendance'], axis=1)

        # Elimino estadisticas que no quiero promediar porque no sirven y solo introducen ruido en el analisis -->  No es necesario puesto que ni las recolecto. Pero por el momento lo necesito.
        print("\nEliminacion de estadisticas irrelevantes")
        stats_columns = construct_data.determine_stats_columns(df_match)
        relevant_stats_columns = ['ball_possession', 'goal_attempts', 'interceptions', 'shots_on_goal', 'goals', 'points', 'expected_goals_(xg)', 'fouls'] # 'perc_shots_on_goal_of_goal_attempts', 'perc_goals_of_goal_attempts']
        df_match = clean_data.delete_not_relevant_stats(df_match, stats_columns, relevant_stats_columns)

        # Elimino columnas de jugadores que son todo NaN (se ve que hay porque las creo y no les guardo nada eso debe ser porque obtengo nombres solo si tiene url)
        non_object_columns = df_match_player.select_dtypes(exclude=['object']).columns
        df_match_player.drop(columns=non_object_columns, inplace=True)
        print("Shape df_match_player: ", df_match_player.shape)

        # Preparo columnas texto
        columns_to_keep = [col for col in df_match.columns if df_match[col].dtype == 'object' and 'id_' not in col]
        df_match = clean_data.prepare_text_columns(df_match, l_cols_to_process=columns_to_keep) # Ver si selecciona bien.. # ['team_home', 'team_away', 'coach_home', 'coach_away', 'venue', 'referee'])
        columns_player_names = list(df_match_player.filter(like='player_name').columns)
        df_match_player = clean_data.prepare_text_columns(df_match_player, l_cols_to_process=columns_player_names)

        end = time.time()
        print(f"Limpieza inicial de datos en {(end - start) / 60:.1f} minutos")

        if self.export:
            df_match.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_match_cleaned.xlsx')
            df_match_player.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_match_player_cleaned.xlsx')

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
            df.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_integrated.xlsx')
        return df
    
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

        # Agrego variable respuesta a df_match
        df_new = construct_data.determine_result(df_new, 'result')
        
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
            df_new.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_constructed.xlsx', index=True)

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
            df_etiquetas.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_etiquetas_actualizado.xlsx', index=False) 
        return df

    def clean_data_2_new(self, df: pd.DataFrame, scaler_loaded, columns_used, comp_to_select):

        ## Para evitar ciertas competencias (no tiene sentido filtrar por n_years_to_select pues los partidos son proximos y no los quiero filtrar)
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
            df.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_selected.xlsx', index=True)
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
        columnas_player_miss = ['dif_mean_val_player_sub' ,'dif_mean_age_player_miss', 'dif_mean_hei_player_miss', 'dif_mean_int_rep_player_miss', 'dif_sum_rat_player_miss', 'dif_sum_val_player_miss']         # A veces, las columnas dif de jugadores ausentes es NaN dado que uno delos equipos no tiene jug ausentes. Podria evitarlo.
        columns_to_fill = [col for col in (columnas_player_miss + columnas_h2h) if col in df.columns]
        if columns_to_fill:
            df[columns_to_fill] = df[columns_to_fill].fillna(0)
        # df= df.fillna(0)

        # Elimino partidos con al menos un NaN value
        df_sin_dup = df.dropna()
        if len(df) != len(df_sin_dup):
            text = f"Cuidado! No se hara la prediccion para {len(df_sin_dup)} partidos puesto que tienen al menos un valor NaN y el modelo no puede tener input NaN."
            warnings.warn(text)

        if self.export: 
            df_sin_dup.to_excel(f'main_find_best_hyper/data/{self.country}/{self.iteration_date}/assess_model_in_prod/data_preparation/df_selected_nan.xlsx', index=True)

        return df_sin_dup

# Data understanding
def read_last_version_matches(country, ruta_base, _print: bool = True):
    # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
    try:
        df_match = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_understanding/df_match_with_missing.xlsx', index_col=0) #WITH MISSING
        if _print:
            print('1) Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')
            print(df_match.shape)
   
    except:  # FileNotFoundError Tambien hay otro error que es cuando el archivo excel esta pero esta dañado y no abre "OptionError"
        df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
        if _print:
            print('2) Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
            print(df_match.shape)
  
    return df_match
            
def convert_columns_to_float(df: pd.DataFrame, _print: bool = False):
    """
    Intenta convertir las columnas object a float
    """
    # Selecciono las columnas object
    l_columnas_a_codificar = df.select_dtypes(include=['object']).columns

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

def load_hyperparameters(country, n_model, ruta_base):

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
        row_hiper = df_iteration[df_iteration['n_iteration'] == n_model] # row_ite

        ## Levanto columnas utilizadas para entrenar el modelo
        selected_columns = eval(row_hiper['X_columns'].values[0])  # eval() para pasar de string a lista

    # Guardo hiperparametros en diccionario
    ## Construct_data
    d['n_dias_ult_part'] =  int(row_hiper['n_dias_ult_part'].values[0])
    d['n_years_h2h'] = int(row_hiper['n_anios_hist'].values[0])
    d['segun_localia'] = row_hiper['segun_localia'].values[0]
    ## Clean_data_2
    n_years_to_select = row_hiper['n_years_to_select'].values[0]
    d['n_years_to_select'] = None if pd.isna(n_years_to_select) else int(n_years_to_select) # Si n_years_to_select es NaN, lo paso de np.nan a None
    d['comp_to_select'] = row_hiper['comp_to_select'].values[0]
    ## Select_data
    d['selected_columns'] = selected_columns
    ## Estrategia de apuesta
    d['thr_prob_min'] = row_hiper['thr_prob_min_best'].values[0]
    d['thr_prob_win'] = row_hiper['thr_prob_win_best'].values[0]
    d['curva'] = row_hiper['curva'].values[0]
    param1 = row_hiper['param1'].values[0]
    param2 = row_hiper['param2'].values[0]
    d['curva_m'] = param1 if d['curva'] == 'linear' else None
    d['curva_b'] = param2 if d['curva'] == 'linear' else None
    d['curva_p1'] = eval(param1) if d['curva'] != 'linear' else None
    d['curva_p2'] = eval(param2) if d['curva'] != 'linear' else None

    for key, value in d.items():
        print(f'\t {key}: {value}')
    return d

def load_models(country, n_model, ruta_base, d):

    # Si se levanta de main.py
    if n_model is None:
        path_tag = f"./p3_data_preparation/data/{country}/df_etiquetas.xlsx"
        path_scaler = f"./p3_data_preparation/data/{country}/scaler_model.pkl"
        path_model = f"./p4_modeling/data/{country}/modelo.pkl"

    # Si se levanta de find_best_hyper.py
    else:
        n_ult_part, n_years_h2h, segun_localia, n_years_sel, comp = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia'], d['n_years_to_select'], d['comp_to_select']
        path_tag = f'{ruta_base}/data_preparation/df_etiquetas_{n_ult_part}_{n_years_h2h}_{segun_localia}.xlsx'
        path_scaler = f'{ruta_base}/data_preparation/scaler_model_{n_ult_part}_{n_years_h2h}_{segun_localia}_{n_years_sel}_{comp}.pkl'
        path_model = f"{ruta_base}/modeling/{n_model}_model.pkl"

    tager_loaded = pd.read_excel(path_tag)
    scaler, columns_scaled = joblib.load(path_scaler)
    loaded_model = pickle.load(open(path_model, "rb"))
    return tager_loaded, scaler, columns_scaled,loaded_model

################################################### MAIN ###################################################
def main(d_run, country, iteration_date, n_model):
    """
    Extrae datos missing, los prepara y predice con modelo ya entrenado. 
    """
    start = time.time()

    # Defino condiciones del analisis
    data_unders, data_prep, modeling, export = d_run['data_unders'], d_run['data_prep'], d_run['modeling'], d_run['export']

    # Definicion de variables
    ruta_base = f"./main_find_best_hyper/data/{country}/{iteration_date}"  # Le agrego assess_model_in_prod
    var_pred = 'predicted_result'
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    id_country = df_countries[df_countries['country_name'] == country.capitalize()]['id_country'].values[0]

    # Creo objeto de clases DataUnderstanding y DataPreparation
    du = DataUnderstandingNew_2(id_country=id_country, country=country, iteration_date=iteration_date, export=export) 
    dp = DataPreparationNew_2(id_country=id_country, country=country, iteration_date=iteration_date, export=export)

    # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
    d_hiper = load_hyperparameters(country, n_model, ruta_base)
    tager, scaler, columns_scaled, loaded_model = load_models(country, n_model, ruta_base, d_hiper)
    
    #______________________________________________ DATA UNDERSTANDING ______________________________________________#
    print("\n", "#"*120, "\n", "DATA UNDERSTANDING".center(120), "\n", "#"*120, "\n")
    if data_unders:
        
        # Obtengo ultima version de df_match
        df_match_old = read_last_version_matches(country, ruta_base)
    
        # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
        df_match, df_match_player, df_match_odds = du.collect_missing_data(df_match_old)
        print(f"Cantidad de partidos missing extraidos: {len(df_match)}")

        # Concateno
        if len(df_match) > 0:
            df_match_ct = pd.concat([df_match_old, df_match], axis=0)
            df_match_ct.to_excel(f'{ruta_base}/assess_model_in_prod/data_understanding/df_match_with_missing.xlsx', index=True)

    else:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_understanding/df_match_player_next.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_understanding/df_match_next_odds.xlsx', index_col=0)

        # du.describe_data_new(df_match, df_match_player, df_match_odds)

    # print("Shapes: ", df_match.shape, df_match_player.shape, df_match_odds.shape)
    # print("\n DF MATCH \n", df_match.head(2))
    # print("\n DF MATCH PLAYER \n", df_match_player.head(2))

    # SOFIFA
    df_player_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0)
    df_player_fifa_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
    df_teams_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)
    # print("\n df_player_sofifa \n", df_player_sofifa.head(2))
    # print("\n df_player_fifa_sofifa \n", df_player_fifa_sofifa.head(2))

    # Levanto datos viejos
    df_integrated = pd.read_excel(f'p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)

    # Filtro df para seleccionar ultimos x dias  ## Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada team...
    fecha_limite = datetime.datetime.now() - datetime.timedelta(days=d_hiper['n_dias_ult_part']*3)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual
    df_old_int = df_integrated.sort_values(by='date', ascending=False) # Ordeno por fecha ascendente
    df_old_int_filt = df_old_int[df_old_int['date'] >= fecha_limite]  # Funciona ok, tiene los partidos missing.
    print("Shape de partidos ya jugados con los cuales construir los proximos partidos: ", df_old_int_filt.shape)
        
    #______________________________________________ DATA PREPARATION ______________________________________________#
    print("\n DATA PREPARATION \n".center(240, "#"))
    if data_prep:

        # Verifico si es necesario integrar o si ya lo hice
        try:
            df = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_preparation/df_integrated.xlsx', index_col=0)
            print(len(df), len(df_match))

            if not(len(df) == len(df_match)):
                print("Hay un dataframe integrado pero le faltan partidos.")
                raise TypeError
            else:
                print("Se levanto df ya integrado y se evito integrar nuevamente...")
        except:
            print("Formateo, limpio e integro datos")
            df_match, df_match_odds = dp.format_data_new(df_match, df_match_odds)
            df_match, df_match_player = dp.clean_data_new(df_match, df_match_player)
            df = dp.integrate_data_new(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc 
        # Preparo
        df = dp.construct_data_new(df, df_old_int, df_old_int_filt, n_days=d_hiper['n_dias_ult_part'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'])
        df = dp.tag_string_data_to_integer_new(df, tager)
        df = dp.clean_data_2_new(df, scaler, columns_scaled, eval(d_hiper['comp_to_select']))
        df = dp.select_data_new(df, d_hiper['selected_columns'])
        df = dp.treat_nan_values_new(df)
        print("Shape Dataframe antes de Modeling(): ", df.shape)

    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_preparation/df_selected_nan.xlsx', index_col=0)
        print(df.head(2), df.shape)

        df_match = pd.read_excel(f'main_find_best_hyper/data/{country}/assess_model_in_prod/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_understanding/df_match_next_odds.xlsx', index_col=0)

    #______________________________________________ MODELING ______________________________________________#
    print("\n MODELING \n".center(240, "#"))
    if modeling:

        # Levanto datasets
        df_teams = pd.read_excel(f'p3_data_preparation/data/{country}/integrate_data/df_teams.xlsx', index_col=0)
        df_constructed = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_preparation/df_constructed.xlsx', index_col=0)

        # Agrego el resultado a df_match
        df_result = df_constructed['result']
        df_result = df_result[df_result.index.isin(df.index)]
        # Es importante no concatenar partidos de mas porque afecta los partidos a predecir
        df_match_odds = df_match_odds[df_match_odds.index.isin(df.index)]
        df_match = df_match[df_match.index.isin(df.index)]
        df_match = df_match.loc[:, ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition']]
        df_match = pd.concat([df_match, df_result], axis=1)

        # Realizo predicciones sobre los nuevos partidos
        y_pred_prob = loaded_model.predict_proba(df)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad 
        df_pred_proba = pd.DataFrame({var_pred: y_pred, f'prob_class_{loaded_model.classes_[1]}': y_pred_prob[:, 1], f'prob_class_{loaded_model.classes_[0]}': y_pred_prob[:, 0], f'prob_class_{loaded_model.classes_[2]}': y_pred_prob[:, 2]}, index=df.index)

        # Concateno conjunto de datos
        df_match_odds = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta
        df_predicciones = pd.concat([df_match, df_match_odds, df_pred_proba], axis=1)

        # Evaluo predicciones del modelo
        df_predicciones, d_roi = calculate_roi_by_betting_strategy(df_predicciones, _print=False)
        print("Metricas: ", d_roi)

        # Revierto etiquetas para tener nombres de equipos en vez de ids
        d_mapeo = dict(zip(df_teams.index, df_teams['team_name']))        
        df_predicciones['id_team_home'] = df_predicciones['id_team_home'].replace(d_mapeo)
        df_predicciones['id_team_away'] = df_predicciones['id_team_away'].replace(d_mapeo)
        return df_predicciones, d_roi

    end = time.time()
    print(f"Main_next_matches en {(end - start)/60:.1f} minutos")

def assess_models(country, fecha_iteracion, df_iteration):
    """
    Evaluo modelos en produccion con datos de partidos recientes
    """
    d_run = {'data_unders': False, 'data_prep': True, 'modeling': True, 'export': True}  # True o False para data_unders --> True solo cdo quiero recolectar missing..

    # Definicion de variables
    df = pd.DataFrame()
    # progress_bar =

    # Por iteracion
    for idx, row in df_iteration.iterrows():

        # Preparo datos y predigo rdos segun modelo
        df_predicciones, d_roi = main(d_run, country, fecha_iteracion, row['n_iteration'])
        print("1 Shape df_reality: ", df_predicciones.shape)

        # Concateno datos
        datos = {k: ', '.join(map(str, v)) if isinstance(v, tuple) else v for k, v in d_roi.items()}
        df_roi = pd.DataFrame.from_dict(datos, orient='index').T
        df_roi['n_iteration'] = row['n_iteration']
        df_roi.set_index('n_iteration', inplace=True) # Establecer 'n_iteration' como índice del DataFrame
        df = pd.concat([df, df_roi], axis=0)
        print(df)

        # Exporto datos
        df_predicciones.to_excel(f'main_find_best_hyper/data/{country}/{iteration_date}/assess_model_in_prod/modeling/{row['n_iteration']}_df_pred_metrics.xlsx', index=True)
        df.to_excel(f'main_find_best_hyper/data/{country}/{iteration_date}/df_iteration_prod.xlsx')
        
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Defino condiciones del analisis
    country = "england"
    iteration_date = '2024-04-22'
    df_iteration = pd.read_excel(f'main_find_best_hyper/data/{country}/{iteration_date}/df_iteration.xlsx')

    # Para filtrar por competiciones (si queres)
    # print("1", df_iteration.shape)
    # df_iteration = df_iteration[df_iteration['comp_to_select'].apply(lambda x: eval(x) == [482, 483, 484, 485])]
    # print("2", df_iteration.shape)

    assess_models(country, iteration_date, df_iteration)