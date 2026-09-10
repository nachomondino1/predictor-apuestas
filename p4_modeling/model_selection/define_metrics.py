
import pandas as pd
import numpy as np
from utils import directories
from utils.set_up_logging import logger
from p3_data_preparation.select_data import normalize_column
import p3_data_preparation.select_data as sd # --> le da importancia a metricas con corr negativa pero que deberian ser max no min.
from p4_modeling import betting_strategy, assess_model
from p4_modeling.model_selection import assess_in_prod
from tqdm import tqdm
import datetime
import numpy as np

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

        col1 = 'n_iteration' if 'n_iteration' in df_ite.columns else 'n_model'
        col2 = 'model_name' if 'model_name' in df_ite.columns else 'model_name_x'
        n_model, model_name = row[col1], row[col2]
        # print(n_model)

        # Filtrar columnas que contienen 'cv_' o '_train'
        train_metrics_cols = [col for col in df_ite.columns if "cv_" in col or "_train" in col]
        d_metrics_train = df_ite.loc[idx, train_metrics_cols].to_dict()  # Seleccionar solo las métricas de train

        # Obtengo predicciones
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred = pd.read_excel(path_test, index_col=0)
        df_pred = assess_model.drop_old_metrics(df_pred) # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = df_pred.head(300)
        print(df_pred.shape)

        # Separo x% como "test" y (1-x)% como "prod"
        n_matches_test = int(perc_matches_test * len(df_pred))
        n_matches_prod = len(df_pred) - n_matches_test
        df_first_matches = df_pred.head(n_matches_test)
        df_last_matches = df_pred.tail(n_matches_prod)
        print(df_first_matches.shape, df_last_matches.shape)

        # Aplico estrategia "sin_ea"
        d_params_sin_ea = bs.define_hiperparameters(strategy='train')  # {'prob_dp': None, 'curva': 'kelly', 'm': 10, 'b': 0, 'k': 1}

        # Aplico estrategia a "TEST" (first_matches)
        # 📌 Sin ea 
        df_test, d_rois = bs.calculate_roi_in_combination(df_first_matches, d_params_sin_ea)
        d_metrics_test_sin_ea = assess_model.calculate_metrics(df_test)
        d_metrics_test_sin_ea_ex = assess_model.calculate_metrics(df_test, var_resp='expected_result', prefix='expected_')

        # Aplico estrategia a "PROD" o "ASSESS" (last_matches) 
        df_prod, d_rois_prod = bs.calculate_roi_in_combination(df_last_matches, d_params_sin_ea)
        d_metrics_prod_sin_ea = assess_model.calculate_metrics(df_prod, suffix='_prod')
        d_metrics_prod_sin_ea_ex = assess_model.calculate_metrics(df_prod, var_resp='expected_result', prefix='expected_', suffix='_prod')

        # Renombro metricas para evitar sobreescribirlas
        d_rois_prod = assess_model.rename_dict_keys(d_rois_prod, suffix='_prod')

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

# Metodos para determinar importancia de metricas de test respecto de la metrica a optimizar en prod
def calculate_correlation(df_ite, metrics, corr_col: str = 'roi_prod', n_models: int = None):
    """
    Calculo de correlacion de metricas con roi_prod
    """
    # Crear un DataFrame vacío para la correlación
    df_corr = pd.DataFrame(index=metrics, columns=['corr_roi']) 

   # Por metrica
    for col in metrics:

        # 2.1. Selecciono mejores modelos by metric (para quitar ruido?)
        if n_models is not None:
            df_ite_filt = df_ite.sort_values(by=col, ascending=False).head(n_models)
            df_ite_filt.to_excel(f"{path_save}/input_data/models_{col}.xlsx", index=False)
        else:
            df_ite_filt = df_ite.copy()

        # Calculo correlación entre métricas de init y prod con los ROIs
        try:
            df_corr.loc[col, 'corr_roi'] = df_ite_filt[col].corr(df_ite_filt[corr_col])
            # df_corr.loc[col, 'corr_ex_roi'] = df_ite[col].corr(df_ite[ex_roi_col])
        except ValueError:
            print(f"Falló el calculo de corr de {col}")
            # Elimino metricas que no son float
            # df_ite_test_with_metric = df_ite_test_with_metric.select_dtypes(include=['float64'])
            # print(f"B: {len(df_ite.columns)}", df_ite.columns)
            
    # Ordeno por correlación con ROI
    df_corr = df_corr.sort_values(by='corr_roi', ascending=False)
    return df_corr
        
def weightened_average(df_ite, metrics, path_save, n_models: int = 100):
    """
    Calculo de promedio ponderado entre metricas de test y metrica de prox a optimizar. Ponderado pues + peso 
    en donde mejor es la metrica de test. 

    # Parameters:
    df_ite: Dataframe con metricas de testeo y metrica de prod. (DataFrame)
    metrics: Metricas de testeo a evaluar. (list)
    """
    rows = {}
    df_prod_only = df_ite[[col for col in df_ite.columns if '_prod' in col]]
    prod_cols = df_prod_only.select_dtypes(include=["number"]).columns

    # Por metrica
    for metric in metrics:

        # 2.1. Selecciono mejores modelos by metric (para quitar ruido?)
        if n_models is not None:
            df_ite_filt = df_ite.sort_values(by=metric, ascending=False).head(n_models)
            df_ite_filt.to_excel(f"{path_save}/input_data/models_{metric}.xlsx", index=False)
        else:
            df_ite_filt = df_ite.copy()

        # Crear pesos para los registros (mayor peso a las primeras filas) --> Normalizando metrica entre 0 y 1
        metric_values = df_ite_filt[metric].values
        min_val, max_val = metric_values.min(), metric_values.max()

        if min_val == max_val:
            weights = np.ones_like(metric_values)
        else:
            weights = (metric_values - min_val) / (max_val - min_val)

        weights /= weights.sum()  # Normalizar pesos para que sumen 1
        
        # Calcular media ponderada para cada columna numérica
        weighted_means = {
            f"weighted_mean_{col}": np.average(df_ite_filt[col], weights=weights)
            for col in prod_cols
        }
        
        # Agregar las medias ponderadas al diccionario con la métrica como clave
        rows[metric] = weighted_means

    # Convertir el diccionario en un DataFrame para almacenar todo organizado
    df_results = pd.DataFrame.from_dict(rows, orient="index")
    return df_results

if __name__ == "__main__":

    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]

    d_countries = {
        # train nuevos
        48: ["england", '2025-05-28'],
        55: ["france", '2025-05-28'], 
        59: ["germany", '2025-05-28'],
        77: ["italy", '2025-05-28'],
        148: ["spain", '2025-05-28']
        }
    
    # Condiciones 
    perc_matches_test = 0.6
    n_models = 10
    method = ['corr', 'fs', 'mean', 'select_best_model'][3] # deberia tener un fs que considere metricas juntas... tal vez ese que va eliminando variables de a una.
    l_metrics_test = ["roi", "error", "expected_roi", "test_accuracy", "recall", "f1_score",  "expected_f1_score", 
                      "cv_accuracy", "cv_f1_score_wei", "cv_f1_score", "cv_f1_score_draw", "cv_cross_entropy_loss"]
    l_metrics_prod = ['test_accuracy'] # El maldito roi es demasiado volatil? ['roi', 'expected_roi', 'error', "recall", 'f1_score', 'test_accuracy', 'expected_f1_score'] # 'expected_error',
    
    # Por metrica de prod
    for corr_col in l_metrics_prod:

        # Definir metrica a optimizar en produccion
        corr_metric = f'{corr_col}_prod'
        print("Corr col: ", corr_col)

        # Defino variables
        df_ct = pd.DataFrame()

        for id_country in l_countries:

            # Defino pais y directorios
            country = d_countries[id_country][0]
            iteration_date = d_countries[id_country][1]
            print(f" {country.upper()} ".center(120, "$"))
            
            path_save = f"data/_metrics/{country}/{iteration_date}"
            directories.make_directories(l_directorios=[f'{path_save}/input_data', f"{path_save}/results/{iteration_date}", f'data/_metrics/results/{method}' ])

            # 2. Preseleccionar modelos 
            df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx") 
 
            col_name = 'model_name' if 'model_name' in df_ite.columns else 'model_name_x'
            df_ite.drop_duplicates(subset=['n_iteration', col_name], inplace=True) # df_iteration esta mal x +1 models? # pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/best_model/3_bet_strategy/df_ite_bs.xlsx") 

            df_ite.loc[df_ite['error'] > 0, 'error'] *= -1  # Convierto error a negativo
            if 'expected_error' in df_ite.columns:
                df_ite.loc[df_ite['expected_error'] > 0, 'expected_error'] *= -1 

            # 3. Por modelo: division en "test" y "prod" + Calculo metricas
            # 3.1. Divido en "test" y "prod" y 3.2. recalculo metricas
            try:
                df_ite_test = pd.read_excel(f"{path_save}/input_data/df_ite_test_{perc_matches_test}.xlsx", index_col=0)
                df_ite_prod = pd.read_excel(f"{path_save}/input_data/df_ite_prod_{1-perc_matches_test}.xlsx", index_col=0)
                print(df_ite_test)
                print(df_ite_prod)
            except FileNotFoundError:
                df_ite_test, df_ite_prod = determine_metrics_by_model(df_ite, country, iteration_date, perc_matches_test=perc_matches_test)
                df_ite_test.to_excel(f'{path_save}/input_data/df_ite_test_{perc_matches_test}.xlsx', index=True)
                df_ite_prod.to_excel(f'{path_save}/input_data/df_ite_prod_{1-perc_matches_test}.xlsx', index=True)

            # calcular correlacion entre metricas de test
            df_correlacion = df_ite_test.select_dtypes(exclude=['object']).corr().abs()
            df_correlacion.to_excel(f'{path_save}/input_data/df_corr.xlsx', index=True)

            # Concateno "test" y "prod" en un solo df    
            df_ite = pd.merge(
                df_ite_test,
                df_ite_prod,
                on=['n_model', 'model_name'],
                how='outer'
            )        
            df_ite.to_excel(f'{path_save}/input_data/df_iteration_{perc_matches_test}.xlsx', index=False)

            # 5. Metodo estadistico para definir importancias de metricas test respecto de la metrica a opt de prod
            ## Op 1: Calculo correlacion de metricas test con la metrica de prod 
            if method == 'corr':
                df = calculate_correlation(df_ite=df_ite, metrics=l_metrics_test, corr_col=corr_metric, n_models=n_models)
                df.rename(columns={'corr_roi': country}, inplace=True)

            # Op nueva
            elif method == 'select_best_model':

                df = pd.DataFrame()

                # Por metrica
                for metric in l_metrics_test:

                    # seleccionar el mejor modelo (o n mejores modelos)
                    df_ite_aux = df_ite.sort_values(by=metric, ascending=False).head(n_models)
                    print(df_ite_aux)

                    # Obtener metrica de prod para ese mpdelo
                    mean_metric = df_ite_aux[corr_metric].mean()
                    print(f"Media de {corr_col} en prod de los modelos con mejor {metric} en test: {mean_metric}")

                    df.loc[metric, f'mean_{corr_col}'] = mean_metric

            elif method == 'fs':
                ## Op 2: Feature selection # Usar df_ite en vez de df_ite_test para poder calcular los promedios ponderados de todas las metricas de prod... (asi no tengo que definirlo de antemano.)
                l_important_features, df_normalized = sd.select_best_features(df_ct, var_resp=corr_metric, thr_fs=0.2)

            elif method == 'mean':
                # Op 3: Prom ponderado
                df = weightened_average(
                    df_ite=df_ite, 
                    metrics=l_metrics_test,
                    path_save=path_save,
                    n_models=n_models
                )
                df = df.loc[:, f'weighted_mean_{corr_metric}'] 
            
            else:
                logger.error(f"El method {method} no es un metodo disponible. Revisar.")
                raise ValueError

            df.to_excel(f"{path_save}/results/{iteration_date}/df_{corr_col}_country.xlsx", index=True)

            # Guardo resultados del pais
            df_ct = pd.concat([df_ct, df], axis=1) # weighted_mean_roi_prod
            print(df_ct.shape)

        # Calculo media de todos los paises
        desv = df_ct.std(axis=1) # para evitar mean
        if method == 'mean':
            col_name = 'sum'
            df_ct["sum"] = df_ct.sum(axis=1)
        else:
            col_name = 'mean'
            df_ct["mean"] = df_ct.mean(axis=1)
        
        df_ct["std"] = desv
        df_ct = df_ct.sort_values(by=col_name, ascending=False)
        df_ct.to_excel(f"data/_metrics/results/{method}/{corr_col}_{iteration_date}.xlsx")

