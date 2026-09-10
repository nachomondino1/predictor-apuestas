import pandas as pd
from utils.set_up_logging import logger
from utils import directories
from p4_modeling import assess_model, betting_strategy
from main_train_models import concat_dataframes_on_iteration



# Calculo nuevas metricas
def add_metric(df_ite, country,  iteration_date, path):
    """
    Asi puedo seleccionar candidatos usando las mismas metricas que al seleccionar el modelo ganador. 
    Se usa en el caso que cree una nueva metrica luego de entrenar el modelo como sucedio en el ultimo tiempo con las metricas "expected"
    del tipo "f1_score_expected" o si en el train fallo el calculo de metricas avanzadas algo asi.

    # Parameters:
        df_ite: Df_test original del entrenamiento. (DataFrame)
        country: Pais (string)
        iteration_date: Fecha del entrenamiento (string)
        path: Path donde guardar el df_test con las nuevas metricas. (string)
    
    # Return
        df_ite_test: Df_test con nuevas metricas (x cambio en bs o por calculo de ≠ metricas)
    """
    # Definicion de paths
    rows = []

    # Por modelo
    for idx, row in df_ite.iterrows():

        n_model, model_name = row['n_iteration'], row['model_name']
        logger.info(f'{n_model} {model_name}')
        
        # Levanto predicciones del modelo (test o test + assess)
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred_test = pd.read_excel(path_test, index_col=0)

        # Correlacion entre expected_result y result por resultado.
        match = len(df_pred_test[df_pred_test['result'] == df_pred_test['expected_result']])
        pos_match = len(df_pred_test)
        porc = match / pos_match * 100
        d_metrics = {'match': porc}

        # 1: Calculo correlacion de columna con todas las otras stats
        # cols_sel = [
        #     'result_to_bet', 'prob_result_to_bet', 'odd_to_bet', 'stake_to_bet', 
        #     'acerte', 'bank_inicial', 'stake_to_bet_en_$', 'G/P', 'bank_final', 'yield', 'kelly_criterion',
        #     'expected_acerte', 'expected_bank_inicial', 'expected_stake_to_bet_en_$', 'expected_G/P', 'expected_bank_final', 'expected_yield'
        # ]
        # d_metrics = df_pred_test[cols_sel].corr()['acerte'].to_dict()
        
        '''
        # 2: Calculo corerlacion entre dos columnas
        # Confidence margin --> acerté
        df_acerte = df_pred_test[df_pred_test['acerte'] == 1]
        df_falle = df_pred_test[df_pred_test['acerte'] == 0]

        # calculo nuevas metricas
        mean_acerte_cm = df_acerte['prob_result_to_bet'].mean()
        mean_falle_cm = df_falle['prob_result_to_bet'].mean()
        coef_pearson = df_pred_test['prob_result_to_bet'].corr(df_pred_test['acerte'])

        d_metrics = {'coef_pearson': coef_pearson, 'mean_prob_acerte': mean_acerte_cm, 'mean_prob_falle': mean_falle_cm}
        '''
        # Guardo datos
        new_row = {'n_iteration': n_model, 'model_name': model_name, **d_metrics}
        rows.append(new_row)
        

    # Exporto datos
    df_ite_bs = pd.DataFrame(data=rows)
    df_ite_bs.to_excel(f'{path}/df_experiment.xlsx', index=False)
    print(df_ite_bs)
    
    return df_ite_bs

def main_add_metric(l_countries, d_countries, folder_name):

    df = pd.DataFrame()

    # Por pais
    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        
        path = f'data/{country}/p4_modeling/{iteration_date}/bs/{folder_name}'
        directories.make_directories(l_directorios=[path])

        # Levanto datos
        df_ite_test = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite_test.xlsx")
        print(df_ite_test.shape)

        df_country = add_metric(df_ite=df_ite_test, country=country, iteration_date=iteration_date, path=path)

        df = pd.concat([df, df_country])
    
    df.to_excel('data/_metrics/df_ite_test_with_new_metrics.xlsx', index=False)


# Pruebo ≠ estrategias de apuesta en df_test
def try_strategy(df_ite, country,  iteration_date, path):
    """
    Asi puedo seleccionar candidatos usando las mismas metricas que al seleccionar el modelo ganador. 
    Se usa en el caso que cree una nueva metrica luego de entrenar el modelo como sucedio en el ultimo tiempo con las metricas "expected"
    del tipo "f1_score_expected" o si en el train fallo el calculo de metricas avanzadas algo asi.

    # Parameters:
        df_ite: Df_test original del entrenamiento. (DataFrame)
        country: Pais (string)
        iteration_date: Fecha del entrenamiento (string)
        path: Path donde guardar el df_test con las nuevas metricas. (string)
    
    # Return
        df_ite_test: Df_test con nuevas metricas (x cambio en bs o por calculo de ≠ metricas)
    """
    # Definicion de paths
    rows = []
    max_roi = -100
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
        df_pred = assess_model.drop_old_metrics(df_pred)

        # 📌 Aplicar estrategia "sin_ea"
        d_params = bs.define_hiperparameters(strategy='train')  
        # d_params = {'prob_dp': None, 'curva': 'kelly_linear', 'm': 10, 'b': 0, 'k': 1}
        df_pred_met, d_metrics = bs.calculate_roi_in_combination(df_pred, d_params)

        # Calculo metricas
        d_metric_sin_ea = assess_model.calculate_metrics(df_pred_met, var_resp='result', var_pred="result_to_bet")
        d_metric_sin_ea_ex = assess_model.calculate_metrics(df_pred_met, var_resp='expected_result', var_pred="result_to_bet", prefix='expected_')

        # Guardo datos
        new_row = {'n_iteration': n_model, 'model_name': model_name, **d_metrics, **d_metric_sin_ea, **d_metric_sin_ea_ex}
        rows.append(new_row)

        if d_metrics['roi'] > max_roi:
            best_pred = df_pred_met
            max_roi = d_metrics['roi'] 
            best_n_model, best_model = n_model, model_name

    # Exporto datos
    df_ite_bs = pd.DataFrame(data=rows)
    df_ite_bs.to_excel(f'{path}/df_ite_test.xlsx', index=False)
    best_pred.to_excel(f'{path}/best_{best_n_model}_{best_model}.xlsx', index=False)

    return df_ite_bs

def main_strategy(l_countries, d_countries, folder_name):

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        
        path = f'data/{country}/p4_modeling/{iteration_date}/bs/{folder_name}'
        directories.make_directories(l_directorios=[path])

        # Levanto datos
        df_ite_test = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite_test.xlsx")
        print(df_ite_test.shape)

        # Calculo nuevas metricas en test
        df_ite_test_new = try_strategy(df_ite=df_ite_test, country=country, iteration_date=iteration_date, path=path)
        # df_ite_test_new = pd.read_excel(f'{path}/df_ite_test.xlsx')
        # print(df_ite_test_new.head(5))

        # Calcular las métricas para el país actual
        roi_mean = df_ite_test_new['roi'].mean()
        accuracy_mean = df_ite_test_new['test_accuracy'].mean() # test_accuracy_dp
        
        metrics = {
            'country': id_country,
            'avg_roi': roi_mean,
            'avg_precision': accuracy_mean
        }
        l_metrics.append(metrics)

        # Concateno nuevo test a trian para generar el nuevo df_ite
        df_params_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_params_ite.xlsx")
        df_ite_train = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite_train.xlsx")
        print(df_params_ite.shape, df_ite_train.shape)

        df = concat_dataframes_on_iteration(df_params_ite, df_ite_train, df_ite_test_new)
        df.to_excel(f'{path}/df_iteration.xlsx', index=False)

    ### Calcular el promedio general de todos los países
    # 1. Convertir la lista de diccionarios a un DataFrame para un cálculo más sencillo
    df_metrics = pd.DataFrame(l_metrics)

    # 2. Calcular el promedio de las columnas de métricas
    avg_roi_all_countries = df_metrics['avg_roi'].mean()
    avg_precision_all_countries = df_metrics['avg_precision'].mean()

    # 3. Imprimir los resultados
    print("\nMétricas por país:")
    print(df_metrics)

    print(f"\nROI promedio de todos los países: {avg_roi_all_countries:.2f}%")
    print(f"Precisión promedio de todos los países: {avg_precision_all_countries:.2f}%")

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    folder_name='corr_res_xres'

    d_countries = {
        48: ["england", '2025-08-26'],
        55: ["france", '2025-08-26'], 
        59: ["germany", '2025-08-26'],
        77: ["italy", '2025-08-26'],
        148: ["spain", '2025-08-26'],
        }
    
    l_metrics = []

    # main_strategy(l_countries, d_countries, folder_name)
    main_add_metric(l_countries, d_countries, folder_name)