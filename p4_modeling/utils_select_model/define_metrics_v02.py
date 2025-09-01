
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
        
        df_pred = df_pred.head(300) # Selecciono menos registros de test pues 500 son muchos...
        print(df_pred.shape)

        # Separo x% como "test" y (1-x)% como "prod"
        n_matches_test = int(perc_matches_test * len(df_pred))
        n_matches_prod = len(df_pred) - n_matches_test

        df_first_matches = df_pred.head(n_matches_test)
        df_last_matches = df_pred.tail(n_matches_prod)
        print(df_first_matches.shape, df_last_matches.shape)

        # Aplico estrategia "sin_ea"
        # d_params_sin_ea = bs.define_hiperparameters(strategy='train') 
        d_params_sin_ea = {'prob_dp': 0.5, 'curva': 'kelly', 'm': 5, 'b': 0, 'k': 10}

        # Aplico estrategia a "TEST" (first_matches)
        # 📌 Sin ea 
        df_test, d_rois = bs.calculate_roi_in_combination(df_first_matches, d_params_sin_ea)
        d_metrics_test_sin_ea = asses_model.calculate_metrics(df_test, bet_metrics=False)
        d_metrics_test_sin_ea_ex = asses_model.calculate_metrics(df_test, var_resp='expected_result', prefix='expected_', bet_metrics=False)

        d_metrics_test_sin_ea.update({
            'sum_aciertos_draw': d_metrics_test_sin_ea['aciertos_draw'] + d_metrics_test_sin_ea_ex['expected_aciertos_draw'], 
            'sum_aciertos_away': d_metrics_test_sin_ea['aciertos_away'] + d_metrics_test_sin_ea_ex['expected_aciertos_away'], 
            'sum_aciertos_home': d_metrics_test_sin_ea['aciertos_home'] + d_metrics_test_sin_ea_ex['expected_aciertos_home'], 
            })

        # Aplico estrategia a "PROD" o "ASSESS" (last_matches) 
        df_prod, d_rois_prod = bs.calculate_roi_in_combination(df_last_matches, d_params_sin_ea)
        d_metrics_prod_sin_ea = asses_model.calculate_metrics(df_prod, suffix='_prod', bet_metrics=False)
        d_metrics_prod_sin_ea_ex = asses_model.calculate_metrics(df_prod, var_resp='expected_result', prefix='expected_', suffix='_prod', bet_metrics=False)

        d_metrics_test_sin_ea.update({
            'sum_aciertos_draw_prod': d_metrics_prod_sin_ea['aciertos_draw_prod'] + d_metrics_prod_sin_ea_ex['expected_aciertos_draw_prod'], 
            'sum_aciertos_away_prod': d_metrics_prod_sin_ea['aciertos_away_prod'] + d_metrics_prod_sin_ea_ex['expected_aciertos_away_prod'], 
            'sum_aciertos_home_prod': d_metrics_prod_sin_ea['aciertos_home_prod'] + d_metrics_prod_sin_ea_ex['expected_aciertos_home_prod'], 
            })


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

def exhaustive_metric_combination_search(df, l_metrics_test, col_prod, n_models: int = None, max_comb_size=None, verbose=True):
    """
    Realiza una busqueda exhaustiva de combinaciones de metricas de test para maximizar la correlacion con una metrica de produccion.
    
    # Parameters
        df: DataFrame con las metricas de test y produccion.
        l_metrics_test: Lista de nombres de las metricas de test a considerar.
        n_models: Numero de modelos a considerar en la seleccion.
        col_prod: Nombre de la columna de la metrica de produccion con la que se quiere maximizar la correlacion.
        max_comb_size: Tamaño maximo de las combinaciones a considerar. Si es None, se considera el tamaño de l_metrics_test.
        verbose: Si es True, imprime el progreso de la busqueda.

    # Returns
        Devuelve un DataFrame con las combinaciones de metricas, su media y desviacion estandar de la correlacion con la metrica de produccion.
    """
    best_score = -float("inf")
    best_combination = None
    history = []

    max_comb_size = max_comb_size or len(l_metrics_test)

    for r in range(1, max_comb_size + 1):
        for combo in itertools.combinations(l_metrics_test, r):

            # Calculo la métrica combinada
            df["combined_metric"] = df[list(combo)].mean(axis=1)

            # Selecciono solo los mejores modelos (si usas media simple, tenes que filtrar por n_models si o si, para evitar = medias)
            if n_models is None:
                df_top = df.copy()
            else:
                # n_models = int(perc_models * len(df))
                df_top = df.sort_values("combined_metric", ascending=False).head(n_models)
                logger.info(f"Seleccionando los {n_models} mejores modelos de {len(df)}")
      
            # Metodo 1: Media simple
            mean_corr = df_top[col_prod].mean()
            std_corr = df_top[col_prod].std()

            # Metodo 2: Media ponderada según combined_metric
            weights = np.arange(len(df_top), 0, -1)  # [n, n-1, ..., 1] # pesos segun ranking
            # weights = df_top["combined_metric"].values # pesos segun la metrica combinada
            weights = weights / weights.sum()  # Normalizar
            weighted_mean_corr = np.average(df_top[col_prod].values, weights=weights) #  np.average(df_top[col_prod], weights=weights)

            history.append((combo, mean_corr, weighted_mean_corr, std_corr))

            if verbose:
                print(f"Probando combinación {combo}: {mean_corr:.4f} en {col_prod}")

            if weighted_mean_corr > best_score:
                best_score = weighted_mean_corr
                best_combination = combo

    logger.critical(f"\nMejor combinación: {best_combination} con score {best_score:.4f} en {col_prod}")
    return pd.DataFrame(history, columns=["metrics", f"mean_{col_prod}", f"wei_mean_{col_prod}", f"std_{col_prod}"]).set_index("metrics")

def calculate_mean_cols(df, filter_cols: bool = False):
    
    if filter_cols:
        mean_cols = [col for col in df.columns if col.startswith('mean_')]
        media = df[mean_cols].mean(axis=1)
        wei_mean = df.filter(like='wei_mean_').mean(axis=1)
        desv = df.filter(like='std_').std(axis=1)

        df["wei_mean"] = wei_mean

    else:
        media = df_corr_ct.mean(axis=1)
        desv = df.std(axis=1)

    # Guardo res  
    df["mean"] = media
    df["std"] = desv
    return df

if __name__ == "__main__":

    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]

    d_countries = {
        # train nuevos
        48: ["england", '2025-08-26'],
        55: ["france", '2025-08-26'], 
        59: ["germany", '2025-08-26'],
        77: ["italy", '2025-08-26'],
        148: ["spain", '2025-08-26'],
        }
    
    # Condiciones 
    perc_matches_test = 0.7
    n_models = 20
    used_saved = False # Usar FALSE si queres usar ≠ metricas de test o dividir segun distinta proporcion  
    l_metrics_test = [
        "yield", "expected_yield", "error", "expected_error", "test_accuracy", "recall", "f1_score", "expected_f1_score",  # roi_por_partido, expected_roi_por_partido
        # "cv_accuracy", "cv_f1_score", "cv_cross_entropy_loss", "cv_f1_score_draw",
        "aciertos_draw", "aciertos_home", "aciertos_away", 
        "sum_aciertos_draw", "sum_aciertos_home", "sum_aciertos_away", 
        # "n_draw", "n_away", "n_home", 
        "f1_score_draw", "f1_score_away", "f1_score_home", 
        "expected_f1_score_draw", "expected_f1_score_away", "expected_f1_score_home",
        ]
    # Definir metrica a optimizar en produccion
    col_prod = 'test_accuracy_rtb'  # yield

    # Defino variables
    df_ct, df_corr_ct = pd.DataFrame(), pd.DataFrame()
    d = {}
    corr_metric = f'{col_prod}_prod'
    perc_matches_prod = round(1 - perc_matches_test, 1)

    # Por pais
    for id_country in l_countries:

        # Defino pais y directorios
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        print(f" {country.upper()} ".center(120, "$"))
        
        # Defino paths
        base_path = f"data/_metrics/{country}/{iteration_date}"
        path_input = f'{base_path}/input_data/{perc_matches_test}-{perc_matches_prod}'
        path_res = f"{base_path}/results/{perc_matches_test}-{perc_matches_prod}"
        path_res_all = f'data/_metrics/results/{iteration_date}/'
        directories.make_directories(l_directorios=[path_input, path_res, path_res_all])

        # PASO 1: Division en "test" y "prod" por cada modelo
        ## Levanto df_ite
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx") 

        ## Invierto las metricas donde menor es mejor
        df_ite.loc[df_ite['error'] > 0, 'error'] *= -1  # Convierto error a negativo
        if 'expected_error' in df_ite.columns:
            df_ite.loc[df_ite['expected_error'] > 0, 'expected_error'] *= -1 

        ## Divido en "test" y "prod" y recalculo metricas
        if used_saved:
            try:
                df_ite_test = pd.read_excel(f"{path_input}/df_ite_test.xlsx", index_col=0)
                df_ite_prod = pd.read_excel(f"{path_input}/df_ite_prod.xlsx", index_col=0)
                print(df_ite_test)
                print(df_ite_prod)
            except FileNotFoundError as e:
                logger.warning(f"No se encontraron los archivos guardados en {base_path}/input_data/. Se vuelven a calcular.")
                raise e
        else:
            df_ite_test, df_ite_prod = determine_metrics_by_model(df_ite, country, iteration_date, perc_matches_test=perc_matches_test)
            df_ite_test.to_excel(f'{path_input}/df_ite_test.xlsx', index=True)
            df_ite_prod.to_excel(f'{path_input}/df_ite_prod.xlsx', index=True)

        ## Concateno "test" y "prod" en un solo df    
        df_ite = pd.merge(df_ite_test, df_ite_prod, on=['n_model', 'model_name'], how='outer')        


        # PASO 2: Normalizo las metricas de test
        df_ite_scaled = df_ite.copy()
        scaler = MinMaxScaler()
        df_ite_scaled[l_metrics_test] = scaler.fit_transform(df_ite_scaled[l_metrics_test]) # Cuidado!: Invertir las métricas donde menor es mejor


        # PASO 3: Uso ≠ metodos para definir las mejores metricas de test
        ## PASO 3.a: Calculo correlacion entre metricas de test y produccion
        ### Filtrar columnas de test y prod
        prod_cols = [col for col in df_ite_scaled.columns if col.endswith('_prod')]

        ### Seleccionar solo las columnas relevantes
        df_correlacion = df_ite_scaled[l_metrics_test + prod_cols].corr()
        
        ### Dropear las filas de prod_cols y las columnas de test_cols (para evitar redundancia)
        df_correlacion = df_correlacion.drop(index=prod_cols, columns=l_metrics_test)

        ### Selecciono solo la columna de la metrica de produccion 
        df_corr = df_correlacion[['roi_por_partido_prod']]  
        df_corr.rename(columns={"roi_por_partido_prod": f"{country}_corr_{corr_metric}"}, inplace=True)

        ## PASO 3.b: Defino mejor combinacion de metricas de test en metrica de produccion
        df = exhaustive_metric_combination_search(df_ite_scaled, l_metrics_test=l_metrics_test, n_models=n_models, col_prod=corr_metric, max_comb_size=2, verbose=True)
        
        # Exporto datos 
        df_ite.to_excel(f'{path_input}/df_iteration.xlsx', index=False)
        df_ite_scaled.to_excel(f"{path_input}/df_scaled.xlsx", index=True)
        df_correlacion.to_excel(f'{path_res}/df_corr.xlsx', index=True)
        df.to_excel(f"{path_res}/df_{col_prod}_best_models.xlsx", index=True)
        
        # Guardo rdos
        d[id_country] = df_ite_scaled.copy()
        df_corr_ct = pd.concat([df_corr_ct, df_corr], axis=1)
        df_ct = pd.concat([df_ct, df], axis=1)
        print(df_ct.shape)

    # PASO 4: Calculo media de todos los paises
    df_ct = calculate_mean_cols(df_ct, filter_cols=True)
    df_corr_ct = calculate_mean_cols(df_corr_ct)
    print(df_ct)
    
    # Seleccionar la combinacion que maximiza la suma pero minimiza el desvio.
    # df_ct["score"] = (df_ct["mean"] - 1 * df_ct["std"]) # Maximizar sum, minimizar std
    # df_ct = df_ct.sort_values(by='score', ascending=False)

    # Exportar resultados finales
    df_ct.to_excel(f"{path_res_all}/select_best_model_{col_prod}_{perc_matches_test}.xlsx") 
    df_corr_ct.to_excel(f"{path_res_all}/corr_{col_prod}_{perc_matches_test}.xlsx") 


    '''
    # PASO 5: ?
    # Defino las mejores metricas de test
    max_mean = df_ct["mean"].max()
    l_metrics_test_filt = df_ct[df_ct["mean"].between(max_mean * 0.8, max_mean)].index
    logger.critical(f"\nMejores metricas de test: {l_metrics_test_filt} con media mayor a {max_mean * 0.8:.4f} y maximo {max_mean:.4f}")

    df_ct_2 = pd.DataFrame()

    # Defino mejor combinacion de metricas de test en metrica de produccion
    for id_country in l_countries:

        # Defino mejor combinacion de metricas de test en metrica de produccion
        df = exhaustive_metric_combination_search(d[id_country], l_metrics_test=l_metrics_test_filt, perc_models=perc_models, col_prod=corr_metric, verbose=True)
        df_ct_2 = pd.concat([df_ct_2, df], axis=1)
    

    # Calculo media de todos los paises
    media = df_ct_2.filter(like='mean_').mean(axis=1)
    desv = df_ct_2.filter(like='mean_').std(axis=1)    
    df_ct_2["mean"] = media
    df_ct_2["std"] = desv

    df_ct_2.to_excel(f"data/_metrics/results/{method}/2_{col_prod}_{iteration_date}.xlsx") 
    '''