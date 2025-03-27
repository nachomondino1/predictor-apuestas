
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

        # Filtrar columnas que contienen 'cv_' o '_train'
        train_metrics_cols = [col for col in df_ite.columns if "cv_" in col or "_train" in col]
        # Seleccionar solo las métricas de train
        d_metrics_train = df_ite.loc[idx, train_metrics_cols].to_dict()

        # Obtengo predicciones
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred = pd.read_excel(path_test, index_col=0)
        # print(df_pred.shape)

        # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = asses_model.drop_old_metrics(df_pred)

        # Separo x% como "test" y (1-x)% como "prod"
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
        d_metrics_test_sin_ea.update(asses_model.calculate_gp_by_result(df_test))
        d_metrics_test_sin_ea_ex = asses_model.calculate_metrics(df_test, var_resp='expected_result', prefix='expected_')

        # Aplico estrategia a "PROD" o "ASSESS" (last_matches) 
        df_prod, d_rois_prod = bs.calculate_roi_in_combination(df_last_matches, d_params_sin_ea)
        d_metrics_prod_sin_ea = asses_model.calculate_metrics(df_prod, suffix='_prod')
        d_metrics_prod_sin_ea_ex = asses_model.calculate_metrics(df_prod, var_resp='expected_result', prefix='expected_', suffix='_prod')
        d_metrics_prod_sin_ea.update(asses_model.calculate_gp_by_result(df_prod))

        # Renombro metricas para evitar sobreescribirlas
        d_rois_prod = asses_model.rename_dict_keys(d_rois_prod, suffix='_prod')

        # Guardo métricas del modelo en un solo diccionario
        row_dict = {
            "n_model": n_model,
            "model_name": model_name,
            **d_metrics_train,
            **d_metrics_test_sin_ea,  # Métricas de test sin ea
            **d_metrics_test_sin_ea_ex,
            **d_rois
        }

        row_dict_prod = {
            "n_model": n_model,
            "model_name": model_name,
            **d_metrics_prod_sin_ea,  # Agrego métricas de producción (prod)
            **d_metrics_prod_sin_ea_ex,
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
        
if __name__ == "__main__":
        
    # Defino variables
    df_ct = pd.DataFrame()
    calculate_metrics = False

    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [148]

    d_countries = {
        # train nuevos
        48: ["england", '2025-03-23'],
        55: ["france", '2025-03-23'], 
        59: ["germany", '2025-03-23'],
        77: ["italy", '2025-03-23'],
        148: ["spain", '2025-03-24']
        }
    
    # Definir metrica a maximizar en produccion
    corr_col = 'error' # f1_score
    corr_metric = f'{corr_col}_prod'

    for id_country in l_countries:

        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        print(f" {country.upper()} ".center(120, "$"))
        
        # 0: Levanto df_ite_test (test) --> NUNCA REDUCIR EL NRO DE MODELOS PUES LOS RDOS PUEDEN SER MUY ≠ A LOS QUE REALMENTE SON.
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")

        # 1. Por modelo: division en "test" y "prod" + Calculo metricas
        if calculate_metrics:
            df_ite_test, df_ite_prod = determine_metrics_by_model(df_ite, country, iteration_date, perc_matches_test=0.75)
            df_ite = pd.merge(df_ite_test, df_ite_prod, on='n_model', how='outer') 
            df_ite_test.to_excel(f'/Users/nachomondino/Desktop/{country}/df_ite_test.xlsx', index=False)
            df_ite_prod.to_excel(f'/Users/nachomondino/Desktop/{country}/df_ite_prod.xlsx', index=False)
            df_ite.to_excel(f'/Users/nachomondino/Desktop/{country}/df_iteration.xlsx', index=False)
        else:
            df_ite_test = pd.read_excel(f"/Users/nachomondino/Desktop/{country}/df_ite_test.xlsx")
            df_ite_prod = pd.read_excel(f"/Users/nachomondino/Desktop/{country}/df_ite_prod.xlsx")
            df_ite =  pd.read_excel(f'/Users/nachomondino/Desktop/{country}/df_iteration.xlsx')
     
        # multiplico error por -1 para que la corr sea positiva
        df_ite_test['error_train'] = df_ite_test['error_train'] * -1
        df_ite_test['error'] = df_ite_test['error'] * -1
        df_ite_test['expected_error'] = df_ite_test['expected_error'] * -1

        df_ite['error_train'] = df_ite['error_train'] * -1
        df_ite['error'] = df_ite['error'] * -1
        df_ite['expected_error'] = df_ite['expected_error'] * -1

        df_ite_prod['error_prod'] = df_ite_prod['error_prod'] * -1
        # df_ite_prod['expected_error'] = df_ite_prod['expected_error'] * -1
        # print(df_ite)

        # Drop columns 
        # Filtrar las columnas que contienen los strings en cols_drop
        cols_drop = ['dif_', '%_dif', 'gp_total', '_train']
        df_ite_test = df_ite_test.drop(columns=[col for col in df_ite_test.columns if any(substring in col for substring in cols_drop)])

        # Concateno metrica de prod con test
        df_ite_test_with_metric = pd.merge(df_ite_test, df_ite_prod.loc[:, ['n_model', corr_metric]], on='n_model', how='outer') 
        cols_float = df_ite_test_with_metric.select_dtypes(include=['float']).columns.tolist()
        df_ite_test_with_metric = df_ite_test_with_metric[cols_float]
        print(df_ite_test_with_metric.shape)

        df_ct = pd.concat([df_ct, df_ite_test_with_metric], axis=0)
        print(df_ct.shape)

        # Imprimir correlacion de metrica entre test y prod
        corr = df_ite_test_with_metric[corr_metric].corr(df_ite_test_with_metric[corr_col]) * 100
        print(f"La correlacion de la metrica {corr_col} entre test y prod es de: {corr:.1f}%")

    # Imprimir correlacion de metrica entre test y prod
    corr = df_ct[corr_metric].corr(df_ct[corr_col]) * 100
    print(f"\n La correlacion de la metrica {corr_col} entre test y prod es de: {corr:.1f}%")

    df_ct = df_ct.dropna(axis=1, how='any')
    df_ct.to_excel("/Users/nachomondino/Desktop/AAA.xlsx")

    # Lasso
    import p3_data_preparation.select_data as sd
    l_important_features, df_normalized = sd.select_best_features(df_ct, var_resp=corr_metric, thr_fs=0.2)
    df_normalized.to_excel("/Users/nachomondino/Desktop/df_normalized.xlsx")
