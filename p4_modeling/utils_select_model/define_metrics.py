
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils import directories
from p4_modeling import betting_strategy, asses_model
from p4_modeling.utils_select_model import assess_in_prod
from tqdm import tqdm
import datetime
import numpy as np


def determine_metrics_by_model(df_ite, country, iteration_date, perc_matches_test: float = 0.75, assess: bool = False):
    """
    Automatizo el experimento para definir metricas segun correlacion con ROI prod y Ex ROI prod.
    Es un experimento retroactivo. Tengo el ROI de cada modelos en los partidos futuros y veo como seleccionar a los modelos que mejor les fue.
    """
    rows, rows_prod = [], []
    bs = betting_strategy.BettingStrategy(country, iteration_date, verbose=0)

    progress_bar = tqdm(total=len(df_ite), ncols=80)  # Inicializo barra de progreso

    # Itero sobre cada modelo
    for idx, row in df_ite.iterrows():

        col_name = 'model_name' if 'model_name' in df_ite.columns else 'model_name_x'
        n_model, model_name = row['n_iteration'], row[col_name]
        # print(n_model)

        # Filtrar columnas que contienen 'cv_' o '_train'
        train_metrics_cols = [col for col in df_ite.columns if "cv_" in col or "_train" in col]
        # Seleccionar solo las métricas de train
        d_metrics_train = df_ite.loc[idx, train_metrics_cols].to_dict()

        # Obtengo predicciones
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_test = pd.read_excel(path_test, index_col=0)
        
        if assess:
            df_first_matches = df_test.copy()

            # Obtengo predicciones assess de modelos
            try:
                date_assess = datetime.datetime.now().date()
                path_assess = f"data/{country}/p4_modeling/{iteration_date}/best_model/2_assess/{date_assess}/{n_model}__{model_name}_predicciones.xlsx" 
                df_last_matches = pd.read_excel(path_assess, index_col=0)

            except FileNotFoundError:
                df_last_matches = assess_in_prod.assess_model_in_prod(df_ite=df_ite, id_country=id_country, country=country, iteration_date=iteration_date, concat_with_test=False)
        
        else:
            df_pred = df_test.copy()

            # Separo x% como "test" y (1-x)% como "prod"
            n_matches_test = int(perc_matches_test * len(df_pred))
            n_matches_prod = len(df_pred) - n_matches_test
            df_first_matches = df_pred.head(n_matches_test)
            df_last_matches = df_pred.tail(n_matches_prod)
            # print(df_first_matches.shape, df_last_matches.shape)

        # Dropeo old metrics (sino calcula mal las nuevas)
        df_first_matches = asses_model.drop_old_metrics(df_first_matches)
        df_last_matches = asses_model.drop_old_metrics(df_last_matches)

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

def calculate_correlation(df_ite, metrics, corr_col: str = 'roi_prod'):
    """
    Calculo de correlacion de metricas con roi_prod
    """
    
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
        
def weightened_average(
        df_ite,
        metrics,
        prod_metric,
        export: bool = True
    ):
    rows = {}

    for metric in metrics:

        # Crear pesos para los registros (mayor peso a las primeras filas) --> Normalizando metrica entre 0 y 1
        metric_values = df_ite[metric].values
        min_val, max_val = metric_values.min(), metric_values.max()

        if min_val == max_val:
            weights = np.ones_like(metric_values)
        else:
            weights = (metric_values - min_val) / (max_val - min_val)

        weights /= weights.sum()  # Normalizar pesos para que sumen 1
        
        # Calcular media ponderada para cada columna numérica
        weighted_means = {
            f"weighted_mean_{col}": np.average(df_ite[prod_metric], weights=weights)
            for col in df_ite.select_dtypes(include=["number"]).columns
        }
        
        # Agregar las medias ponderadas al diccionario con la métrica como clave
        rows[metric] = weighted_means

    # Convertir el diccionario en un DataFrame para almacenar todo organizado
    df_results = pd.DataFrame.from_dict(rows, orient="index")
    return df_results

if __name__ == "__main__":
        
    # Defino variables
    df_ct = pd.DataFrame()

    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    assess = False

    d_countries = {
        # train nuevos
        48: ["england", '2025-04-08'],
        55: ["france", '2025-04-08'], 
        59: ["germany", '2025-04-08'],
        77: ["italy", '2025-04-08'],
        148: ["spain", '2025-04-08']
        }
    
    # 1. Definir metrica a optimizar en produccion
    corr_col = 'roi'
    corr_metric = f'{corr_col}_prod'

    for id_country in l_countries:

        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        print(f" {country.upper()} ".center(120, "$"))
        
        path_save = f"data/{country}/p4_modeling/{iteration_date}/best_model/0_define_metrics"
        directories.make_directories(l_directorios=[path_save])

        # 2. Preseleccionar modelos 
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite_test.xlsx") # df_iteration esta mal x +1 models?
        # df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/best_model/3_bet_strategy/df_ite_bs.xlsx")        
        # df_ite = df_ite.head(100) 

        # By metric? 
        # 2.1. By metric
        # df_ite.sort_values(by=metr)
    
        # 3. Por modelo: division en "test" y "prod" + Calculo metricas
        try:
            df_ite_test = pd.read_excel(f"{path_save}/df_ite_test.xlsx")
            df_ite_prod = pd.read_excel(f"{path_save}/df_ite_prod.xlsx")
            df_ite =  pd.read_excel(f'{path_save}/df_iteration.xlsx')
        
        except FileNotFoundError:
            # 3.1. Divido en "test" y "prod" y 3.2. recalculo metricas
            df_ite_test, df_ite_prod = determine_metrics_by_model(df_ite, country, iteration_date, perc_matches_test=0.75, assess=assess)
            df_ite = pd.merge(df_ite_test, df_ite_prod, on='n_model', how='outer') 
            
            df_ite_test.to_excel(f'{path_save}/df_ite_test.xlsx', index=False)
            df_ite_prod.to_excel(f'{path_save}/df_ite_prod.xlsx', index=False)
            df_ite.to_excel(f'{path_save}/df_iteration.xlsx', index=False)
       
        # 4. Elimino metricas de test no deseadas
        # Filtrar las columnas que contienen los strings en cols_drop
        # l_metrics_test = ["roi", "expected_roi", "error", "expected_error", "test_accuracy", "f1_score", "expected_f1_score"]
        # df_ite_test = df_ite_test.loc[l_metrics_test]
        cols_drop = ['dif_', '%_dif', 'gp_total', '_train', '_home', '_draw', '_away'] #  'expected_', 'accuracy_', 'f1_score_'
        df_ite_test = df_ite_test.drop(columns=[col for col in df_ite_test.columns if any(substring in col for substring in cols_drop)])

        # Concateno metrica de prod con test
        df_ite_test_with_metric = pd.merge(df_ite_test, df_ite_prod.loc[:, ['n_model', corr_metric]], on='n_model', how='outer') 
        cols_float = df_ite_test_with_metric.select_dtypes(include=['float', 'int']).columns.tolist()
        df_ite_test_with_metric = df_ite_test_with_metric[cols_float]
        print(df_ite_test_with_metric.shape)

        metrics = list(df_ite_test_with_metric.columns)
        metrics.remove(corr_col)

        # 5. Metodo estadistico para definir importancias de metricas test respecto de la metrica a opt de prod
        ## Op 1: Calculo correlacion de metricas test con la metrica de prod
        # df = calculate_correlation(df_ite_test_with_metric, metrics=metrics, corr_col=corr_metric)
        # df.rename(columns={'corr_roi': country}, inplace=True)
        # df_corr.to_excel("/Users/nachomondino/Desktop/df_normalized.xlsx")

        ## Op 2: Feature selection 
        # import p3_data_preparation.select_data as sd --> le da importancia a metricas con corr negativa pero que deberian ser max no min.
        # l_important_features, df_normalized = sd.select_best_features(df_ct, var_resp=corr_metric, thr_fs=0.2)
        # df_normalized.to_excel("/Users/nachomondino/Desktop/df_normalized.xlsx")

        ## Op 3: Prom ponderado
        df = weightened_average(
            df_ite=df_ite_test_with_metric, 
            metrics=metrics,
            prod_metric=corr_metric
            )
        df.to_excel(f"{path_save}/df_results.xlsx", index=True)

        # Guardo resultados del pais
        df_ct = pd.concat([df_ct, df], axis=1)
        print(df_ct.shape)
        # df_ct.to_excel("/Users/nachomondino/Desktop/df_ct.xlsx")

        # Imprimir correlacion de metrica entre test y prod
        # corr = df_ite_test_with_metric[corr_metric].corr(df_ite_test_with_metric[corr_col]) * 100
        # print(f"La correlacion de la metrica {corr_col} entre test y prod es de: {corr:.1f}%")

    # Calculo correlacion de metricas test con la metrica de prod
    # df_ct["mean_corr"] = df_ct.mean(axis=1)
    df_ct.to_excel(f"{path_save}/df_normalized.xlsx")

