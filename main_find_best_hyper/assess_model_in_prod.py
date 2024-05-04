# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import datetime
import os
## Data preparation
from p6_deployment.main_next_matches import DataPreparationNew
from p3_data_preparation import construct_data
# Modeling
import pickle
import joblib
from p4_modeling import asses_model


def make_directories(ruta_base):  # Pasarle direcotio o l_directorios como argumento...
    l_directorios = [
        f'{ruta_base}/assess_model_in_prod/data_preparation/format_data',
        f'{ruta_base}/assess_model_in_prod/data_preparation/clean_data',
        f'{ruta_base}/assess_model_in_prod/modeling',
    ]

    for directorio in l_directorios:
        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

# Data understanding
def read_data(country):
    ## Flashscore
    df_match = pd.read_excel(f'p6_deployment/data/{country}/missing/data_understanding/all/df_match_miss.xlsx', index_col=0)
    df_match_player = pd.read_excel(f'p6_deployment/data/{country}/missing/data_understanding/all/df_match_player_miss.xlsx', index_col=0)
    df_match_odds = pd.read_excel(f'p6_deployment/data/{country}/missing/data_understanding/all/df_match_odds_miss.xlsx', index_col=0)
    ## Sofifa
    df_player_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0)
    df_player_fifa_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
    df_teams_sofifa = pd.read_excel(f"p3_data_preparation/data/{country}/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)
    print(df_match.shape, df_match_player.shape, df_match_odds.shape, df_player_sofifa.shape, df_player_fifa_sofifa.shape, df_teams_sofifa.shape)
    return df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa

def select_league_matches(df_match, df_match_player, df_match_odds):
    """
    Filtra partidos seleccionado solo aquellos que son de liga (eliminando partidos de copa)
    """
    # Levanto df_competencies
    df_comp = pd.read_excel('p2_data_understanding/data/df_competencies.xlsx')

    # Selecciono solo las ligas del pais
    l_leagues = list(df_comp[df_comp['is_cup']==0]['id_competition'].values) # l_comp = [481, 485]
    print("Ligas: ", l_leagues)

    df_match = df_match[df_match['id_competition'].isin(l_leagues)]
    df_match_player = df_match_player[df_match_player.index.isin(df_match.index)]
    df_match_odds = df_match_odds[df_match_odds.index.isin(df_match.index)]
    print(f"Shape sin copas: {df_match.shape}")
    return df_match, df_match_player, df_match_odds

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
    make_directories(ruta_base)  # Creo directorios
    var_pred = 'predicted_result'
    df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    id_country = df_countries[df_countries['country_name'] == country.capitalize()]['id_country'].values[0]

    # Creo objeto de clase DataPreparationNew
    ruta_base_data_prep =f"./main_find_best_hyper/data/{country}/{iteration_date}/assess_model_in_prod/data_preparation" 
    dp = DataPreparationNew(id_country=id_country, country=country, ruta_base=ruta_base_data_prep, export=export)

    #______________________________________________ DATA UNDERSTANDING ______________________________________________#  # --> Levanto dfs missing de p6_deployment
    print("\n", "#"*120, "\n", "DATA UNDERSTANDING".center(120), "\n", "#"*120, "\n")
    # Levanto datos
    df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = read_data(country)  ## Datos "missing"
    df_integrated = pd.read_excel(f'p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)  ## Datos con los que entrenó el modelo

    # Filtro por competencias. No quiero partidos de copas (e.g. FA cup) solo de la liga
    df_match, df_match_player, df_match_odds = select_league_matches(df_match, df_match_player, df_match_odds)

    # Dataframe match
    if 'attendance' in df_match.columns:
        df_match = df_match.drop(['attendance'], axis=1)

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
        # Verifico si es necesario integrar o si ya lo hice
        try:
            df_int = pd.read_excel(f'{ruta_base}/assess_model_in_prod/data_preparation/df_integrated.xlsx', index_col=0)
            print(len(df_int), len(df_match))

            if not(len(df_int) == len(df_match)):
                print("Hay un dataframe integrado pero le faltan partidos.")
                raise TypeError
            else:
                print("Se levanto df ya integrado y se evito integrar nuevamente...")
                
        except (FileNotFoundError, TypeError):
            print("Formateo, limpio e integro datos")
            df_match, df_match_odds = dp.format_data_new(df_match, df_match_odds)
            df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) # uso de main.py puesto que tengo estadisticas y demas
            df_int = dp.integrate_data_new(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc 
    
        # Filtro df para seleccionar ultimos x dias  ## Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada team...
        fecha_limite = datetime.datetime.now() - datetime.timedelta(days=d_hiper['n_dias_ult_part']*3)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual
        df_old_int = df_integrated.sort_values(by='date', ascending=False) # Ordeno por fecha ascendente
        df_old_int_filt = df_old_int[df_old_int['date'] >= fecha_limite]  # Funciona ok, tiene los partidos missing.
        print("Shape de partidos ya jugados con los cuales construir los proximos partidos: ", df_old_int_filt.shape)
    
        # Preparo
        df_int = construct_data.determine_result(df_int, 'result')  # Es necesaria?
        df_cons = dp.construct_data_new(df_int, df_old_int, df_old_int_filt, n_days=d_hiper['n_dias_ult_part'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'])
        df_tag = dp.tag_string_data_to_integer_new(df_cons, tager)
        df_clean = dp.clean_data_2_new(df_tag, scaler, columns_scaled, d_hiper['comp_to_select'])
        df_sel = dp.select_data_new(df_clean, d_hiper['selected_columns'])
        df_treat = dp.treat_nan_values_new(df_sel)
        print("Shape Dataframe antes de Modeling(): ", df_treat.shape)

        #______________________________________________ MODELING ______________________________________________#
        print("MODELING".center(120, "-"))
        # Levanto datasets
        df_teams = pd.read_excel(f'p3_data_preparation/data/{country}/integrate_data/df_teams.xlsx', index_col=0)

        # Agrego el resultado a df_match
        df_result = df_cons['result']
        df_result = df_result[df_result.index.isin(df_treat.index)]
        # Es importante no concatenar partidos de mas porque puede hacer el ROI nan (al no tener un predicted_result ni nada al evaluar el ROI)
        df_match_odds_2 = df_match_odds[df_match_odds.index.isin(df_treat.index)]
        df_match_odds_2 = df_match_odds.reindex(df_treat.index)  # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test
        df_match_2 = df_match[df_match.index.isin(df_treat.index)]
        df_match_2 = df_match_2.loc[:, ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition']]
        df_match_2 = pd.concat([df_match_2, df_result], axis=1)

        # Realizo predicciones sobre los nuevos partidos
        y_pred_prob = loaded_model.predict_proba(df_treat)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad 
        df_pred_proba = pd.DataFrame({var_pred: y_pred, f'prob_class_{loaded_model.classes_[1]}': y_pred_prob[:, 1], f'prob_class_{loaded_model.classes_[0]}': y_pred_prob[:, 0], f'prob_class_{loaded_model.classes_[2]}': y_pred_prob[:, 2]}, index=df_treat.index)

        # Concateno conjunto de datos
        df_match_odds_2 = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds_2) # Caculo probabilidades segun casa de apuesta
        df_predicciones = pd.concat([df_match_2, df_match_odds_2, df_pred_proba], axis=1)
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
        print(df_iteration_prod)

        # Exporto datos
        if export:
            df_predicciones.to_excel(f'{ruta_base}/assess_model_in_prod/modeling/{row['n_iteration']}_df_pred_metrics.xlsx', index=True)
            df_iteration_prod.to_excel(f'{ruta_base}/assess_model_in_prod/df_iteration_prod_seg.xlsx')

    # Exporto datos
    if export:
        df_iteration_prod.to_excel(f'{ruta_base}/df_iteration_prod.xlsx')
    
    return df_iteration_prod

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Defino condiciones del analisis
    country = "argentina"
    iteration_date = '2024-05-03'
    df_iteration = pd.read_excel(f'main_find_best_hyper/data/{country}/{iteration_date}/df_iteration.xlsx')

    # Evaluo modelos en produccion
    df_iteration_prod = main(df_iteration, country, iteration_date) # df_iteration_prod = pd.read_excel(f'main_find_best_hyper/data/{country}/{iteration_date}/df_iteration_prod.xlsx', index_col=0)

    # Concateno df_iteration y df_iteration_prod
    df_iteration.set_index('n_iteration', inplace=True) # Establecer 'n_iteration' como índice del DataFrame
    df_concat = pd.concat([df_iteration, df_iteration_prod], axis=1)
    df_concat.to_excel(f'main_find_best_hyper/data/{country}/{iteration_date}/df_iteration_completo.xlsx', index=True)  # df_concat.to_excel(f'/Users/nachomondino/Desktop/df_iteration_completo_{country}.xlsx', index=True)