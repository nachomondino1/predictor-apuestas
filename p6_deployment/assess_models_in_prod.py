import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import pandas as pd
import os
from dotenv import load_dotenv
'''ImportError: cannot import name 'DataPreparation' from partially initialized module 'main' (most likely due to a circular import) (/Users/nachomondino/Documents/GitHub/predictor-apuestas/main.py)
from p3_data_preparation.construct_data import determine_result, determine_expected_result
from p6_deployment import main_next_matches
from p4_modeling import betting_strategy
import datetime
'''
# El objetivo es evaluar las predicciones de los mejores modelos de un pais en los ultimos partidos jugados sin tener que hacerlo manualmente.
def assess_model_in_prod(id_country, n_model, model_name, iteration_date):

    d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Recolecta predicciones de modelo en partidos "missing"
    df = collect_prediction_in_missing_matches(id_country, n_model, model_name, iteration_date)

    # Preparo el df 
    df = prepare_for_betting_strategy(df, country)
    return df


def collect_prediction_in_missing_matches(id_country, n_model, model_name, iteration_date):
    """
    Recolecta predicciones de modelos en partidos "missing"

    :return: df_predicciones del modelo en partidos missing
    """
    # Defino variables (no tocar)
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': False}  # No puedo correr data_unders = False si los proximos partidos ya estan en missing.
    d_model = {'n_model': n_model, 'model_name': model_name, 'iteration_date': iteration_date} # XGBClassifier, neural_networ, SVC, LogisticRegression, MLPClassifier

    # Usar mnm.py con predict_missing=True y data_unders=False.
    df = main_next_matches.main(d_run, id_country, d_model=d_model, predict_missing=True, no_strategy=True, export=d_run['export'], verbose=0) # Probar un modelo

    df = df.sort_values(by='date', ascending=False)
    # df.to_excel(f'/Users/nachomondino/Desktop/df_pred_missing_{n_model}.xlsx')
    # print(df)
    return df

def prepare_for_betting_strategy(df, country):  # Ponerlo como funcion dentro de BettingStrategy????
    """
    Agregar columnas result y acerte en df_pred. Determino result y expected result para poder determinar "acerte"
    """
    ## Levanto df_match_miss para obtener goals? ??
    df_match_miss = pd.read_excel(f"data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx", index_col=0)
    logger.info(df_match_miss)
    
    # Loggear información básica del DataFrame
    logger.info(f"Shape of df_match_miss: {df_match_miss.shape}")
    logger.info(f"Columns in df_match_miss: {df_match_miss.columns.tolist()}")

    # Columnas a copiar
    l_columns_to_copy = ['goals_home', 'goals_away', 'expected_goals_(xg)_home', 'expected_goals_(xg)_away']

    # Asignar valores de df_match_miss a df solo en las columnas y filas correspondientes
    for idx, row in df.iterrows():

        for column in l_columns_to_copy:

            df.loc[idx, column] = df_match_miss.loc[idx, column] 

    # df.loc[df_match_miss.index, l_columns_to_copy] = df_match_miss[l_columns_to_copy]

    ## Determino result y expected result segun goals
    df = determine_result(df) # Intento hacerlo antes con df_match pero rompia.
    df = determine_expected_result(df, goals_to_xg_ratio=0.3) # Intento hacerlo antes con df_match pero rompia.


    # Por que no calculo ROI y expected roi --> No deberia pues ahora calculo metricas del df_concatenado.. --> Creo que no ahce falta gracias a calculate_roi_by_betting_strategy()
   
    # # Agrego predicciones de bookmaker --> no tengo df_match_odds
    # df_match_odds = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta
    # df_match_odds = asses_model.determine_result_by_bookmaker(df_match_odds, self.var_pred_bm)  # Determino resultado predicho segun cuota minima (e.g. "Home")
    return df


def nose():
    # Calcular metricas de df_predicciones ya sea de missing (?) o de prox partidos para cierto modelo...

    id_country = 77
    model = 314
    iteration_date = '2024-12-23'
    pred_missing = True

    # Defino variables
    d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Obtener predicciones missing
    if pred_missing:
        df_pred = collect_prediction_in_missing_matches(id_country=id_country, n_model=model, model_name='LogisticRegression', iteration_date=iteration_date)
    else:
        # Levanto predicciones
        df_pred = pd.read_excel(f'data/{country}/p6_deployment/ASSESS/semana 19/predicciones_{model}.xlsx', index_col=0)
        print(df_pred)

    # Eliminar partidos aun no jugados
    print(len(df_pred))
    fecha_hoy = datetime.datetime.now()
    df_pred = df_pred[df_pred['date'] <= fecha_hoy]
    print(len(df_pred))

    # Obtengo result y expected result
    df_pred = prepare_for_betting_strategy(df_pred, country=country)
    print(df_pred)

    # Calculo metricas
    bs = betting_strategy.BettingStrategy(strategy='train')
    d_hiper, best_df_pred, best_d_rois  = bs.calculate_roi_by_betting_strategy(df_pred, roi_weight=1)

    print(best_df_pred)
    print(f'Ganancias sin bank: {best_df_pred['G/P_sin_bank'].sum()}')
    print(f'Expected Ganancias sin bank: {best_df_pred['expected_G/P_sin_bank'].sum()}')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    '''
    # Defino parametros
    id_country = 148
    country = 'spain'
    n_model = 1
    model_name = "LogisticRegression"
    iteration_date = "2024-12-03"
    
    # Obtengo predicciones en partidos missing
    df_predicciones = collect_prediction_in_missing_matches(id_country, n_model, model_name, iteration_date)

    # Exporto datos
    load_dotenv() 
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')
    df_predicciones.to_excel(f'{BASE_DIR_LOCAL}/df_pred_missing_{n_model}.xlsx')
    '''
    nose()