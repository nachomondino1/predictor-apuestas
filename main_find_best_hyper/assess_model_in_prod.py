# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
import os
## Data preparation
from p6_deployment.main_next_matches import DataPreparationNew, filter_dataframe_by_date
from p3_data_preparation import construct_data
# Modeling
import pickle
import joblib
from p4_modeling import asses_model
from set_up_logging import logger
import os
from dotenv import load_dotenv


def make_directories(ruta_base):  # Pasarle direcotio o l_directorios como argumento...
    l_directorios = [
        f'{ruta_base}/assess_model_in_prod/data_preparation',
        f'{ruta_base}/assess_model_in_prod/modeling',
    ]

    for directorio in l_directorios:
        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

# Data understanding
def select_league_matches(df):
    """
    Filtra partidos seleccionado solo aquellos que son de liga (eliminando partidos de copa)
    """
    # Levanto df_competencies
    df_comp = pd.read_excel('p2_data_understanding/data/df_competencies.xlsx')

    # Selecciono solo las ligas del pais
    l_leagues = list(df_comp[df_comp['is_cup']==0]['id_competition'].values) 
    print("Ligas: ", l_leagues)

    df = df[df['id_competition'].isin(l_leagues)]
    print(f"Shape sin copas: {df.shape}")
    return df

def load_hyperparameters(row_hiper):

    d = {}

    # Guardo hiperparametros en diccionario
    ## Construct_data
    d['n_dias_ult_part'] =  int(row_hiper['n_dias_ult_part'].values[0])
    d['n_years_h2h'] = int(row_hiper['n_anios_hist'].values[0])
    d['segun_localia'] = row_hiper['segun_localia'].values[0]
    ## Clean_data_2
    n_years_to_select = row_hiper['n_years_to_select'].values[0]
    d['n_years_to_select'] = None if pd.isna(n_years_to_select) else int(n_years_to_select) # Si n_years_to_select es NaN, lo paso de np.nan a None
    try:
        d['comp_to_select'] = eval(row_hiper['comp_to_select'].values[0])
    except TypeError: # Falla aqui cuando corro el find_best.
        d['comp_to_select'] = list(row_hiper['comp_to_select'].values[0])
    ## Select_data
    try:
        d['selected_columns'] = eval(row_hiper['X_columns'].values[0])  # eval() para pasar de string a lista
    except TypeError:  # Falla aqui cuando corro el find_best.
        d['selected_columns'] = list(row_hiper['X_columns'].values[0])  # eval() para pasar de string a lista

    print("\nHiperparametros cargados:")
    for key, value in d.items():
        print(f'\t {key}: {value}')
    return d

def load_models(n_model, ruta_base, d):

    # Levanto hiperparametros de DataPreparation de la iteracion 
    n_ult_part, n_years_h2h, segun_localia, n_years_sel, comp = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia'], d['n_years_to_select'], d['comp_to_select']

    # Cargo modelos segun hiperparametros
    tager_loaded = pd.read_excel(f'{ruta_base}/data_preparation/df_etiquetas_{n_ult_part}_{n_years_h2h}_{segun_localia}.xlsx')
    scaler, columns_scaled = joblib.load(f'{ruta_base}/data_preparation/scaler_model_{n_ult_part}_{n_years_h2h}_{segun_localia}_{n_years_sel}_{comp}.pkl')
    loaded_model = pickle.load(open(f"{ruta_base}/modeling/{n_model}_model.pkl", "rb"))
    return tager_loaded, scaler, columns_scaled,loaded_model


################################################### MAIN ###################################################
def main(df_iteration, country, iteration_date, export: bool = True):
    """
    Levanta los datos missing, los prepara y predice con modelo ya entrenado. 
    """
    # Definicion de variables
    df_iteration_prod = pd.DataFrame()
    ruta_base = f"./main_find_best_hyper/data/{country}/{iteration_date}"  # Le agrego assess_model_in_prod
    ruta_base_data_prep =f"./main_find_best_hyper/data/{country}/{iteration_date}/assess_model_in_prod/data_preparation" 
    make_directories(ruta_base)  # Creo directorios 

    # Creo objeto de clase DataPreparationNew
    dp = DataPreparationNew(country=country, export=False)

    #______________________________________________ DATA UNDERSTANDING ______________________________________________#  # --> Levanto dfs missing de p6_deployment
    print("\n", "#"*120, "\n", "DATA UNDERSTANDING".center(120), "\n", "#"*120, "\n")
    # Levanto datos
    df_match_odds = pd.read_excel(f'p6_deployment/data/{country}/missing/data_understanding/all/df_match_odds_miss.xlsx', index_col=0)
    df_teams = pd.read_excel(f'p3_data_preparation/data/{country}/integrate_data/df_teams.xlsx', index_col=0)
    df_int_missing = pd.read_excel(f'p6_deployment/data/{country}/missing/data_preparation/all/df_integrated_missing.xlsx', index_col=0)  # tienen que ser /all... Solo partidos missing.
    
    # Determino la fecha del partido missing mas "viejo"
    df_int_missing['date'] = pd.to_datetime(df_int_missing['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime --> estoy casi seguro que no hace falta.
    initial_date = df_int_missing['date'].min()
    print(f"Initial_date (es decir, la del partido missing mas viejo): {initial_date}")

    # Filtro por competencias. No quiero partidos de copas (e.g. FA cup) solo de la liga
    df_int_missing = select_league_matches(df_int_missing)
   
    # Construyo la variable "result"
    df_int = construct_data.determine_result(df_int_missing, 'result')  # Es necesaria? Creo que si porque en main_next_matches.py no le construyo result...

    # Por iteracion
    for idx, row in df_iteration.iterrows():

        # Definicion de variables
        n_model = row['n_iteration']
        row_hiper = df_iteration[df_iteration['n_iteration'] == n_model]
        print("\n", "#"*120, "\n", f"MODEL Nº {n_model}".center(120), "\n", "#"*120, "\n")

        # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
        d_hiper = load_hyperparameters(row_hiper)
        tager, scaler, columns_scaled, loaded_model = load_models(n_model, ruta_base, d_hiper)
        
        #______________________________________________ DATA PREPARATION ______________________________________________#
        print("DATA PREPARATION".center(120, "-"))

        # Levanto datos ya construidos
        path_cons = f'{ruta_base_data_prep}/df_constructed_{d_hiper['n_dias_ult_part']}_{d_hiper['n_years_h2h']}_{d_hiper['segun_localia']}.xlsx'
        try:
            df_cons = pd.read_excel(path_cons, index_col=0)
            print("Evito construir datos dado que levanto dataframe ya construido")

        # Levanto df_integrated y construyo datos
        except FileNotFoundError:      
            # Levanto datos viejos  
            df_old_int = pd.read_excel(f'p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)  ## Datos con los que entrenó el modelo
            df_old_int = df_old_int.sort_values(by='date', ascending=False)  # Ordeno por fecha ascendente. Funciona? Es entendida como datetime la columna? Si.

            # Selecciono los ultimos partidos de los ya jugados
            n_days_period = d_hiper['n_dias_ult_part'] * 2 if d_hiper['segun_localia'] == True else d_hiper['n_dias_ult_part']
            df_last_old_matches = filter_dataframe_by_date(df=df_old_int, initial_date=initial_date, n_days=n_days_period) 

            # Construyo datos usando partidos viejos
            df_cons = dp.construct_data_new(df_next_matches=df_int, df_last_old_matches=df_last_old_matches, df_old_matches=df_old_int, n_days=d_hiper['n_dias_ult_part'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'])
            df_cons.to_excel(path_cons, index=True)
        
        # Sigo preparando datos
        df_tag = dp.tag_string_data_to_integer_new(df_cons, tager)
        df_clean = dp.clean_data_2_new(df_tag, scaler, columns_scaled, d_hiper['comp_to_select'])
        df_sel = dp.select_data_new(df_clean, d_hiper['selected_columns'])
        df_treat = dp.treat_nan_values_new(df_sel)

        print("\nShape Dataframe antes de Modeling(): ", df_treat.shape)
        if len(df_sel) != len(df_treat):
            logger.warning(f"WARNING! De los {len(df_sel)} proximos partidos, quedan {len(df_treat)} luego de la preparacion")

        #______________________________________________ MODELING ______________________________________________#
        print("MODELING".center(120, "-"))
        # Realizo predicciones sobre los nuevos partidos
        y_pred_prob = loaded_model.predict_proba(df_treat)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad 
        df_pred_proba = pd.DataFrame({'predicted_result': y_pred, f'prob_class_{loaded_model.classes_[1]}': y_pred_prob[:, 1], f'prob_class_{loaded_model.classes_[0]}': y_pred_prob[:, 0], f'prob_class_{loaded_model.classes_[2]}': y_pred_prob[:, 2]}, index=df_treat.index)

        # Agrego resultado y cuotas a df_match
        df_match_odds_2 = df_match_odds[df_match_odds.index.isin(df_treat.index)]
        df_match_odds_2 = df_match_odds.reindex(df_treat.index)  # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test
        df_match = df_int.loc[df_int.index.isin(df_treat.index), ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition', 'result']]

        # Concateno conjunto de datos
        df_match_odds_2 = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds_2) # Caculo probabilidades segun casa de apuesta
        df_predicciones = pd.concat([df_match, df_match_odds_2, df_pred_proba], axis=1)
        df_predicciones['date'] = pd.to_datetime(df_predicciones['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        df_predicciones = df_predicciones.sort_values(by='date', ascending=True)  # Ordeno por fecha de menos reciente a mas reciente para calcular ROI bien.

        # Evaluo predicciones del modelo
        df_predicciones, d_roi = asses_model.calculate_roi_by_betting_strategy(df_predicciones, _print=False)
        print("Metricas: ", d_roi)

        # Revierto etiquetas para tener nombres de equipos en vez de ids
        d_mapeo = dict(zip(df_teams.index, df_teams['team_name']))        
        df_predicciones['id_team_home'] = df_predicciones['id_team_home'].replace(d_mapeo)
        df_predicciones['id_team_away'] = df_predicciones['id_team_away'].replace(d_mapeo)
        
        # Concateno datos
        d_roi['param1'], d_roi['param2'] = str(d_roi['param1']), str(d_roi['param2'])
        df_roi = pd.DataFrame(d_roi, index=[row['n_iteration']])
        df_roi['X_shape_missing'] = [df_treat.shape]  # Me interesa saber el largo del df_missing
        df_iteration_prod = pd.concat([df_iteration_prod, df_roi], axis=0)

        # Exporto datos
        if export:
            df_predicciones.to_excel(f'{ruta_base}/assess_model_in_prod/modeling/{row['n_iteration']}_df_pred_metrics.xlsx', index=True)
            df_iteration_prod.to_excel(f'{ruta_base}/assess_model_in_prod/df_iteration_prod_seg.xlsx')

    return df_iteration_prod

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    # Cargar las variables de entorno desde el archivo .env
    load_dotenv() 
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    # Defino condiciones del analisis
    country = "usa"
    iteration_date = '2024-06-24'
    df_iteration = pd.read_excel(f'main_find_best_hyper/data/{country}/{iteration_date}/df_iteration.xlsx')

    # Evaluo modelos en produccion
    df_iteration_prod = main(df_iteration, country, iteration_date)

    # Concateno df_iteration y df_iteration_prod para tener df_iteration_completo
    df_iteration.set_index('n_iteration', inplace=True) # Establecer 'n_iteration' como índice del DataFrame
    df_concat = pd.concat([df_iteration, df_iteration_prod], axis=1)
    df_concat.to_excel(f'{BASE_DIR_LOCAL}/df_iteration_completo.xlsx', index=True)
    