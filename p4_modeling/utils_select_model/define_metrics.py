
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from p3_data_preparation.select_data import delete_correlated_columns
import p4_modeling.main_select_model as msm
from p4_modeling import betting_strategy, asses_model
from tqdm import tqdm


def determine_metrics_by_model(df_ite, country, iteration_date, perc_matches_test: float = 0.75):
    """
    Automatizo el experimento para definir metricas segun correlacion con ROI prod y Ex ROI prod.
    Es un experimento retroactivo. Tengo el ROI de cada modelos en los partidos futuros y veo como seleccionar a los modelos que mejor les fue.
    """
    rows, rows_prod = [], []
    bs = betting_strategy.BettingStrategy(country, iteration_date, verbose=0)

    progress_bar = tqdm(total=len(df_ite), ncols=80)  # Inicializo barra de progreso

    # Itero sobre cada modelo
    for idx, row in df_ite.iterrows():

        n_model, model_name = row['n_iteration'], row['model_name_x']
        # print(n_model)

        '''
        # Filtrar columnas que contienen 'cv_' o '_train'
        train_metrics_cols = [col for col in df_ite.columns if "cv_" in col or "_train" in col]
        # Seleccionar solo las métricas de train
        d_metrics_train = df_ite.loc[idx, train_metrics_cols].to_dict()
        '''

        # Obtengo predicciones
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred = pd.read_excel(path_test, index_col=0)
        # print(df_pred.shape)

        # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = msm.drop_old_metrics(df_pred)

        # Separo 75% como test y 25% como ROI en prod
        n_matches_test = int(perc_matches_test * len(df_pred))
        n_matches_prod = len(df_pred) - n_matches_test
        df_first_matches = df_pred.head(n_matches_test)
        df_last_matches = df_pred.tail(n_matches_prod)
        # print(df_first_matches.shape, df_last_matches.shape)

        # Aplico estrategia
        d_params_sin_ea = bs.define_hiperparameters(strategy='train') 

        # Aplico estrategia a "TEST" (first_matches)
        # 📌 Sin ea 
        df_test, d_rois = bs.calculate_roi_in_combination(df_first_matches, d_params_sin_ea)
        d_metrics_test_sin_ea = asses_model.calculate_metrics(df_test)
        d_metrics_test_sin_ea_ex = asses_model.calculate_metrics(df_test, var_resp='expected_result', prefix='expected_')

        # Aplico estrategia a "PROD" o "ASSESS" (last_matches) 
        df_prod, d_rois_prod = bs.calculate_roi_in_combination(df_last_matches, d_params_sin_ea)
        d_metrics_prod_sin_ea = asses_model.calculate_metrics(df_prod, suffix='_prod')

        # Renombro metricas para evitar sobreescribirlas
        d_rois_prod = asses_model.rename_dict_keys(d_rois_prod, suffix='_prod')

        # Guardo métricas del modelo en un solo diccionario
        row_dict = {
            "n_model": n_model,
            "model_name": model_name,
            # **d_metrics_train,
            **d_metrics_test_sin_ea,  # Métricas de test sin ea
            **d_metrics_test_sin_ea_ex,
            **d_rois
        }

        row_dict_prod = {
            "n_model": n_model,
            "model_name": model_name,
            **d_metrics_prod_sin_ea,  # Agrego métricas de producción (prod)
            **d_rois_prod
        }

        # Agrego la fila a la lista
        rows.append(row_dict)
        rows_prod.append(row_dict_prod)
        progress_bar.update(1)

    progress_bar.close()
    df_ite = pd.DataFrame(data=rows)
    df_ite_prod = pd.DataFrame(data=rows_prod)

    return df_ite, df_ite_prod

def calculate_correlation(df_ite, corr_col: str = 'roi_prod'):
    """
    Calculo de correlacion de metricas con roi_prod
    """
    metrics = list(df_ite.columns)
    metrics.remove(corr_col)
    
    # Crear un DataFrame vacío para la correlación
    df_corr = pd.DataFrame(index=metrics, columns=['corr_roi']) 

    # Calculo correlación entre métricas de init y prod con los ROIs
    for col in metrics:
        try:
            df_corr.loc[col, 'corr_roi'] = df_ite[col].corr(df_ite[corr_col])
            # df_corr.loc[col, 'corr_ex_roi'] = df_ite[col].corr(df_ite[ex_roi_col])
        except ValueError:
            print(f"Falló el calculo de corr de {col}")
            # Elimino metricas que no son float
            # df_ite_test_with_metric = df_ite_test_with_metric.select_dtypes(include=['float64'])
            # print(f"B: {len(df_ite.columns)}", df_ite.columns)
            
    # Ordeno por correlación con ROI
    df_corr = df_corr.sort_values(by='corr_roi', ascending=False)
    return df_corr

def eliminate_metrics_with_corr(df_ite, thr_corr: float, metric_corr: str, verbose: int = 0):
     
    ## x alta correlacion entre si
    l_cols_to_elim, df_corr_triang = delete_correlated_columns(df_ite, var_resp=metric_corr, thr_corr=thr_corr, verbose=0) # 'roi_prod'
    l_cols = [col for col in df_ite.columns if col not in l_cols_to_elim]
    df_ite = df_ite[l_cols]
    if verbose >= 1:
        print(f"C: {len(df_ite.columns)}", df_ite.columns)

    df_corr_triang.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr_triang.xlsx')
    return df_ite

def eliminate_metrics_low_corr_metric(df_corr, verbose: int = 0):

    metrics = df_corr.index
    ## con poca correlacion con roi_prod
    df_corr = df_corr[df_corr.index.isin(metrics)]
    thr = df_corr['corr_roi'].quantile(0.75)    
    df_corr_filt = df_corr[(df_corr['corr_roi'] >= thr) & (df_corr['corr_roi'] >= 0.01)]

    if verbose >= 1:
        print(f"Columnas que pasan en {country}: {list(df_corr_filt.index)}")

    # df_corr_triang.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr_triang.xlsx')
    return df_corr_filt

def compute_weights(df):
    """
    Defino pesos por pais para cada metrica.
    """
    # Selecciona columnas válidas: Filtra las que no sean 'n_selected' ni 'mean'.
    cols_to_process = [col for col in df.columns if col not in ['n_selected', 'mean']]
    
    # Calcula los pesos de manera vectorizada
    df_weights = df[cols_to_process].div(df[cols_to_process].sum(), axis=1)

    # Renombra las columnas con 'weight_'.
    df_weights.columns = [f'weight_{col}' for col in df_weights.columns]
    return df_weights
        
if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [55]

    d_countries = {
        # train viejos
        # 48: ["england", '2025-02-05'],
        # 55: ["france", '2025-02-05'], 
        # 59: ["germany", '2025-02-05'],
        # 77: ["italy", '2025-02-05'],
        # 148: ["spain", '2025-02-05'], 
        # train nuevos
        48: ["england", '2025-03-18'],
        55: ["france", '2025-03-20'], 
        59: ["germany", '2025-03-20'],
        77: ["italy", '2025-03-20'],
        148: ["spain", '2025-03-20']
        }
    
    df_corr_roi = pd.DataFrame()
    calculate_metrics = True
    corr_col = 'error'
    corr_metric = f'{corr_col}_prod' # Si max f1_score_prod ? el roi es relativo para cada modelo por prob_res_to_bet...

    for id_country in l_countries:

        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        print(f" {country.upper()} ".center(120, "$"))
        
        # 0: Levanto df_ite_test (test) --> NUNCA REDUCIR EL NRO DE MODELOS PUES LOS RDOS PUEDEN SER MUY ≠ A LOS QUE REALMENTE SON.
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
        # print(df_ite)

        # Para que la coreelacion positiva signifique que minimiza el error
        if corr_col == 'error':
            df_ite[corr_col] = -df_ite.loc[:, corr_col]

        # Elimino modelos con f1_score bajo.
        # df_ite = msm.filter_models_by_metric(df_ite, metric_col='expected_roi', perc_cutoff=50)
        # l_metrics, l_weights = ['n_emp', 'acc_draw'],  [0.6, 0.4]
        # df_ite = asses_model.calculate_combined_metric(df_ite, l_metrics=l_metrics, l_weights=l_weights, metric_name='metric_test')
        # df_ite = msm.filter_models_by_metric(df_ite, metric_col='metric_test', perc_cutoff=20) # Creo que hasta 100 esta ok, mas no. En FRA gana el 220, y yo prefiero otro.
        
        # 1. Por modelo: division en "test" y "prod" + Calculo metricas
        if calculate_metrics:
            df_ite_test, df_ite_prod = determine_metrics_by_model(df_ite, country, iteration_date, perc_matches_test=0.75)
            df_ite_test.to_excel(f'/Users/nachomondino/Desktop/{country}/df_ite.xlsx', index=False)
            df_ite_prod.to_excel(f'/Users/nachomondino/Desktop/{country}/df_ite_prod.xlsx', index=False)
        else:
            df_ite_test = pd.read_excel(f"/Users/nachomondino/Desktop/{country}/df_ite.xlsx")
            df_ite_prod = pd.read_excel(f"/Users/nachomondino/Desktop/{country}/df_ite_prod.xlsx")

        # Agrego corr_metric a test
        df_ite_test_with_metric = pd.merge(df_ite_test, df_ite_prod.loc[:, ['n_model', corr_metric]], on='n_model', how='outer') 
        df_ite = pd.merge(df_ite_test, df_ite_prod, on='n_model', how='outer') 
        
        # Calculo correlacion entre metricas de test y corr_metric de prod
        df_corr = calculate_correlation(df_ite_test_with_metric, corr_col=corr_metric)
        df_corr.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr.xlsx', index=True)
     
        # Guardo datos del country
        df_corr_country = df_corr[['corr_roi']].rename(columns={'corr_roi': country})
        df_corr_roi = pd.concat([df_corr_roi, df_corr_country], axis=1)

    # 5. Calculo correlacion promedio (≠ paises) entre el roi y cada metrica 
    df_corr_roi['mean'] = df_corr_roi.mean(axis=1) # df_corr_roi.drop(columns=['n_selected'], errors='ignore').mean(axis=1)
    df_corr_roi.to_excel('/Users/nachomondino/Desktop/df_corr_roi.xlsx')
        

