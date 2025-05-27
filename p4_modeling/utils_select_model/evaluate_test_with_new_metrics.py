import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
import datetime
from p3_data_preparation import construct_data
from p4_modeling import asses_model, betting_strategy
from p4_modeling import main_select_model as msm 
from p6_deployment import main_next_matches
import os


# Main
def main(
        df_ite,
        country, 
        iteration_date,
        ):
    """
    Asi puedo seleccionar candidatos usando las mismas metricas que al seleccionar el modelo ganador. 
    Se usa en el caso que cree una nueva metrica luego de entrenar el modelo como sucedio en el ultimo tiempo con las metricas "expected"
    del tipo "f1_score_expected" o si en el train fallo el calculo de metricas avanzadas algo asi.
    """
    # Definicion de paths
    rows = []
    cont = 0
    bs = betting_strategy.BettingStrategy(country, iteration_date, verbose=0)

    # Por modelo
    for idx, row in df_ite.iterrows():

        n_model, model_name = row['n_iteration'], row['model_name']
        logger.info(f'{n_model} {model_name}')
        
        # Levanto predicciones del modelo (test o test + assess)
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred_test = pd.read_excel(path_test, index_col=0)
        df_pred = df_pred_test.copy()

        # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = asses_model.drop_old_metrics(df_pred)

        # 📌 Aplicar estrategia "sin_ea"
        d_params = bs.define_hiperparameters(strategy='train')  
        # d_params = {'prob_dp': None, 'curva': 'kelly', 'm': 10, 'b': 0, 'k': 1}
        df_pred_met, d_metrics = bs.calculate_roi_in_combination(df_pred, d_params)

        # Calculo metricas
        d_metric_sin_ea = asses_model.calculate_metrics(df_pred_met, var_resp='result')
        d_metric_sin_ea_ex = asses_model.calculate_metrics(df_pred_met, var_resp='expected_result', prefix='expected_')

        # Guardo datos
        new_row = {'n_model': n_model, 'model_name': model_name, **d_metrics, **d_metric_sin_ea, **d_metric_sin_ea_ex}
        rows.append(new_row)
        df_ite_bs = pd.DataFrame(data=rows)

        cont += 1
        if cont % 100 == 0:
            df_ite_bs.to_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration_test_new.xlsx', index=False)

    # Exporto datos
    df_ite_bs.to_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration_test_new.xlsx', index=False)

    return df_ite_bs

def concat_dfs(df_ite, df_ite_train, df_ite_new_test):

    # Merge de los DataFrames
    df_iteration = pd.merge(df_ite, df_ite_train, on='n_iteration', how='outer')

    # Renombrar 'n_model' a 'n_iteration' en df_ite_new_test
    df_ite_new_test = df_ite_new_test.rename(columns={'n_model': 'n_iteration'})

    df_iteration = pd.merge(df_iteration, df_ite_new_test, on=['n_iteration', 'model_name'], how='outer')
    return df_iteration

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]

    d_countries = {
        48: ["england", '2025-05-07'],
        55: ["france", '2025-05-07'], 
        59: ["germany", '2025-05-08'],
        77: ["italy", '2025-05-08'],
        148: ["spain", '2025-05-07'],
        }
    
    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]

        # Levanto datos
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite.xlsx")
        df_ite_train = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite_train.xlsx")
        df_ite_test = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite_test.xlsx")
        print(df_ite.shape, df_ite_train.shape, df_ite_test.shape)

        # Calculo nuevas metricas en test
        reevaluate = True
        if reevaluate:
            df_ite_test_new = main(df_ite=df_ite_test, country=country, iteration_date=iteration_date)
        else:
            df_ite_test_new = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration_test_new.xlsx')
            print(df_ite_test_new.head(5))
        
        # Concateno nuevo test a trian para generar el nuevo df_ite
        df = concat_dfs(df_ite, df_ite_train, df_ite_test_new)
        df.to_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration_new.xlsx', index=False)

        