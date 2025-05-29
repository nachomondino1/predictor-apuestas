
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils import directories
from utils.set_up_logging import logger
from p3_data_preparation.select_data import normalize_column
import p3_data_preparation.select_data as sd # --> le da importancia a metricas con corr negativa pero que deberian ser max no min.
from p4_modeling import betting_strategy, asses_model
from p4_modeling.utils_select_model import assess_in_prod
from tqdm import tqdm
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools

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
        df_pred = asses_model.drop_old_metrics(df_pred) # Dropeo old metrics (sino calcula mal las nuevas)
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
        d_metrics_test_sin_ea = asses_model.calculate_metrics(df_test)
        d_metrics_test_sin_ea_ex = asses_model.calculate_metrics(df_test, var_resp='expected_result', prefix='expected_')

        # Aplico estrategia a "PROD" o "ASSESS" (last_matches) 
        df_prod, d_rois_prod = bs.calculate_roi_in_combination(df_last_matches, d_params_sin_ea)
        d_metrics_prod_sin_ea = asses_model.calculate_metrics(df_prod, suffix='_prod')
        d_metrics_prod_sin_ea_ex = asses_model.calculate_metrics(df_prod, var_resp='expected_result', prefix='expected_', suffix='_prod')

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

# Metodos para determinar importancia de metricas de test respecto de la metrica a optimizar en prod
def greedy_backward_elimination(df_ite, l_metrics_test, n_models, corr_col):

    current_metrics = l_metrics_test.copy()
    history = []

    while len(current_metrics) > 1:
        print(f"\nEvaluando combinación de métricas: {current_metrics}")

        # Calcular promedio de métricas actuales
        df_ite['mean_score'] = df_ite[current_metrics].mean(axis=1)

        # Seleccionar mejores modelos según promedio
        top_models = df_ite.sort_values(by='mean_score', ascending=False).head(n_models)

        # Evaluar metrica en prod (ej: precision_prod)
        current_prod_score = top_models[corr_col].mean()
        print(f"Precisión en prod actual: {current_prod_score:.4f}")

        history.append({
            'kept_metrics': current_metrics.copy(),
            f'mean_{corr_col}': current_prod_score
        })

        # Evaluar impacto de eliminar cada métrica individualmente
        best_new_score = -np.inf
        worst_metric = None

        for metric in current_metrics:
            test_metrics = [m for m in current_metrics if m != metric]
            df_ite['mean_temp'] = df_ite[test_metrics].mean(axis=1)
            top_temp = df_ite.sort_values(by='mean_temp', ascending=False).head(n_models)
            temp_score = top_temp[corr_col].mean()
            print(f"  Sin {metric}: {temp_score:.4f}")

            if temp_score > best_new_score:
                best_new_score = temp_score
                worst_metric = metric

        if best_new_score >= current_prod_score:
            print(f"Eliminando métrica: {worst_metric}")
            current_metrics.remove(worst_metric)
        else:
            # No mejora quitando ninguna → salimos
            print("No mejora quitando ninguna métrica, detenemos el proceso.")
            break

    return pd.DataFrame(history)

def exhaustive_metric_combination_search(df, l_metrics_test, n_models, corr_col, max_comb_size=None, verbose=True):
    best_score = -float("inf")
    best_combination = None
    history = []

    max_comb_size = max_comb_size or len(l_metrics_test)

    for r in range(1, max_comb_size + 1):
        for combo in itertools.combinations(l_metrics_test, r):
            df["combined_metric"] = df[list(combo)].mean(axis=1)

            # Selecciono los mejores n modelos
            df_top = df.sort_values("combined_metric", ascending=False).head(n_models)
            mean_corr = df_top[corr_col].mean()
            std_corr = df_top[corr_col].std()

            history.append((combo, mean_corr, std_corr))

            if verbose:
                print(f"Probando combinación {combo}: {mean_corr:.4f} en {corr_col}")

            if mean_corr > best_score:
                best_score = mean_corr
                best_combination = combo

    logger.critical(f"\nMejor combinación: {best_combination} con score {best_score:.4f} en {corr_col}")
    # return pd.DataFrame(history)
    return pd.DataFrame(history, columns=["metrics", f"mean_{corr_col}", f"std_{corr_col}"]).set_index("metrics")

if __name__ == "__main__":

    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    # l_countries = [48, 55, 148]

    d_countries = {
        # train nuevos
        48: ["england", '2025-04-22'],
        55: ["france", '2025-04-23'], 
        59: ["germany", '2025-04-23'],
        77: ["italy", '2025-04-23'],
        148: ["spain", '2025-04-23'],

        # train viejos
        # 48: ["england", '2025-05-07'],
        # 55: ["france", '2025-05-07'], 
        # 59: ["germany", '2025-05-08'],
        # 77: ["italy", '2025-05-08'],
        # 148: ["spain", '2025-05-07']

        # new train
        # 48: ["england", '2025-05-28'],
        # 55: ["france", '2025-05-28'], 
        # 59: ["germany", '2025-05-29'],
        # 77: ["italy", '2025-05-29'],
        # 148: ["spain", '2025-05-28']
        }
    
    # Condiciones 
    perc_matches_test = 0.6
    n_models = 20
    method = ['corr', 'fs', 'mean', 'select_best_model'][3] # deberia tener un fs que considere metricas juntas... tal vez ese que va eliminando variables de a una.
    l_metrics_test = ["roi", "error", "expected_roi", "test_accuracy", "recall", "f1_score",  "expected_f1_score", 
                      "cv_accuracy", "cv_f1_score_wei", "cv_f1_score", "cv_f1_score_draw", "cv_cross_entropy_loss"]
    l_metrics_prod = ['roi'] # El maldito roi es demasiado volatil? ['roi', 'expected_roi', 'error', "recall", 'f1_score', 'test_accuracy', 'expected_f1_score'] # 'expected_error',
    
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

            # Concateno "test" y "prod" en un solo df    
            df_ite = pd.merge(df_ite_test, df_ite_prod, on=['n_model', 'model_name'], how='outer')        
            df_ite.to_excel(f'{path_save}/input_data/df_iteration_{perc_matches_test}.xlsx', index=False)


            # Normalizo las metricas de test
            df_ite_scaled = df_ite.copy()
            scaler = MinMaxScaler()
            # Invertir las métricas donde menor es mejor
            # invertir = ['logloss_test', 'mae_test']  # si aplica
            # for col in invertir:
            #     df_ite_scaled[col] = -df_ite_scaled[col]
            df_ite_scaled[l_metrics_test] = scaler.fit_transform(df_ite_scaled[l_metrics_test])
            df_ite_scaled.to_excel(f"{path_save}/results/{iteration_date}/df_scaled.xlsx", index=True)

            # Defino mejor combinacion de metricas de test en metrica de produccion
            df = exhaustive_metric_combination_search(df_ite_scaled, l_metrics_test=l_metrics_test, n_models=n_models, corr_col=corr_metric, verbose=True)
            # df = greedy_backward_elimination(df_ite_scaled, l_metrics_test=l_metrics_test, n_models=n_models, corr_col=corr_metric)
            df.to_excel(f"{path_save}/results/{iteration_date}/df_{corr_col}_{country}.xlsx", index=True)

            # Guardo resultados del pais
            df_ct = pd.concat([df_ct, df], axis=1) # weighted_mean_roi_prod
            print(df_ct.shape)

        # Calculo media de todos los paises
        df_ct["mean"] = df_ct.filter(like="mean_").mean(axis=1)
        df_ct["std"] = df_ct.filter(like="std_").std(axis=1)

        # Seleccionar la combinacion que maximiza la suma pero minimiza el desvio.
        df_ct["score"] = df_ct["mean"] - 3 * df_ct["std"]  # Maximizar sum, minimizar std
        df_ct = df_ct.sort_values(by='score', ascending=False)

        # Exportar resultados finales
        df_ct.to_excel(f"data/_metrics/results/{method}/{corr_col}_{iteration_date}.xlsx")

