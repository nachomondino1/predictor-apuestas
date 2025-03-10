import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
import datetime
from p3_data_preparation import construct_data
from p4_modeling import betting_strategy, asses_model
from p6_deployment import main_next_matches
import os

def initialize_directories(country, iteration_date, assess):
    """
    Guarda la selección de modelo vieja en una carpeta y crea los directorios necesarios.

    Mejoras:
    - Hacer funciones separadas para mover archivos dentro de assess y dentro de bs..
    """

    fecha_hoy = datetime.datetime.now().date()
    
    # Definir paths principales
    base_path = f"data/{country}/p4_modeling/{iteration_date}"
    base_path_sbm = f"{base_path}/best_model"
    
    d_paths = {
        'base_path': base_path,
        'base_path_sbm': base_path_sbm,
        'path_old': f'{base_path}/best_model_old/{fecha_hoy}',
        'path_select': f'{base_path_sbm}/1_filter_models',
        'path_assess': f'{base_path_sbm}/2_assess',
        'path_bet_strategy': f'{base_path_sbm}/3_bet_strategy',
        # 'path_assess_dep': f"data/{country}/p6_deployment/assess"
    }

    # Mover toda la carpeta si assess y bet_strategy son True
    if assess:
        if os.path.exists(base_path_sbm):
            directories.make_directories([d_paths['path_old']])
            directories.mover_archivo(base_path_sbm, d_paths['path_old'])
    
    # Crear directorios necesarios
    directories.make_directories([
        d_paths['path_assess'], 
        d_paths['path_select'], 
        d_paths['path_bet_strategy'], 
        # d_paths['path_assess_dep']
    ])

    return d_paths

# Select candidates
def filter_models_by_metric(df, metric_col, prop_to_max: float = 0, perc_cutoff: float = 100, n_models_max: int = 100, verbose: int = 0):
    """
    Selecciona los mejores modelos (sin tener en cuenta la estrategia de apuesta aun).

    # Parameters
    df: Un DataFrame que contiene información sobre los modelos, incluyendo las columnas roi_por_partido y expected_roi_por_partido.
    cutoff: Proporción de los mejores registros a seleccionar (por defecto, 0.02 o el 2% superior).
    method: Metodo para filtrar. 
        'percentile' o 'prop_to_max'
    verbose: Nivel de detalle en los mensajes de salida.
        0 (por defecto): Salida básica.
        Valores mayores producen más detalles.

    # Return 
    La función devuelve un DataFrame (df_filt) que contiene solo los registros seleccionados con los valores más altos en la métrica combinada.
    """
    logger.info("Paso 1: Descartando modelos segun metrica en df_test")
    df = df.sort_values(by=metric_col, ascending=False)  # Ordenar los registros por 'metric' en orden descendente

    # Determino metrica de corte
    ## segun roi proporcional al max
    max_metric = df[metric_col].max()
    metric_cut_prop = max_metric * prop_to_max

    ## segun percentil 
    metric_cut_perc = np.percentile(df[metric_col], (100-perc_cutoff)) # Seleccionar el 20% de los registros con los valores más altos de 'metric'
    
    ## segun cantidad maxima de modelos
    if len(df) >= n_models_max:
        metric_cut_fixed = df.iloc[n_models_max-1][metric_col]
    else:
        metric_cut_fixed = 0

    # Determino el minimo
    metric_cut = max(metric_cut_prop, metric_cut_perc, metric_cut_fixed)
    logger.info(f"\nMetricas de corte: \n(1) {metric_cut_prop} \n(2) {metric_cut_perc} \n (3) {metric_cut_fixed} \n => {metric_cut} ")

    # Filtro modelos segun roi to cut
    df_filt = df[df[metric_col] >= metric_cut]

    if verbose >= 1:
        logger.warning(f"Descarte por {metric_col.upper()}: {len(df)} --> {len(df_filt)}")
        error_empty_dataframe(df_filt)

    return df_filt

def error_empty_dataframe(df):
    if len(df) == 0:
        logger.error("El dataframe esta vació. Probablemente uno de los filtros eliminó todos los modelos que quedaban.")
        raise ValueError

# Betting Strategy
def drop_old_metrics(df_predicciones):
    # Elimino columnas de metricas dejando las predicciones raw (evitar eliminar 'player_emergency_fill' pues genera dif entre los mismos partidos del test y assess. Tmb evitar eliminar goals y demas.)
    columns_to_exclude = [
        'result_to_bet', 'prob_result_to_bet', 'odd_to_bet', 'strategy', 'stake_to_bet', 
        'acerte', 'bank_inicial', 'stake_to_bet_en_$', 'G/P', 'bank_final', 'G/P_sin_bank'
        'expected_acerte', 'expected_bank_inicial', 'expected_stake_to_bet_en_$', 'expected_G/P', 'expected_bank_final', 'expected_G/P_sin_bank'
    ]
    df_predicciones = df_predicciones.drop(columns=columns_to_exclude, errors='ignore')
    return df_predicciones

# Main
def main(
        df_ite,
        id_country, 
        country, 
        iteration_date,
        l_metrics: list, 
        l_weights: list, 
        select_candidates: bool = True,
        n_max_candidates: int = None,
        bet_strat: bool = True,
        assess: bool = True,
        update_missing: bool = True,
        predict_missing: bool = True,
        export: bool = True,
        verbose: int = 0
        ):
    """
    Assess model in prod + Seleccion del modelo + Estrategia de apuesta

    # Parameters
        df_ite: Dataframe con el train y test de los modelos del pais. (DataFrame)
        assess: True para recolectar las predicciones de los modelos en los partidos missing y concatenarlas a las de test. (bool)
        update_missing: Para actualizar los missing y tener los ultimos partidos jugados si hay. (bool)

    # Return
        Modelo a usar en produccion con su estrategia de apuesta optima (teniendo en cuenta test + missing). 

    Posibles mejoras:
        - A partir de nuevos train (donde ya calculo metrics en last matches), usar mismas metricas entre seleccion de candidatos y seleccion de modelo. Por el momento no puedo porque no estoy calculando metricas en ultimos partidos en df_test.
    """
    # Definicion de paths
    rows = []
    d_paths = initialize_directories(country, iteration_date, predict_missing)
    path_cand = f'{d_paths['path_select']}/df_filt_by_metric_cand.xlsx'

    # Determino correlacion entre ROI y Expected ROI --> no creo que este bien. Creo que conviene usar roi_weight de 1 y listo. No es representativo la corr sino la diferencia absoluta entre roi y ex_roi (posiblemente porcentual)
    ex_weight = df_ite['roi'].corr(df_ite['expected_roi'])
    roi_weight = 1 - ex_weight if ex_weight > 0 else 1
    logger.info(f"Correlacion ROI y Expected ROI: {ex_weight:.2f}. --> ROI weight: {roi_weight:.2f} y Ex ROI weight: {ex_weight:.2f}")

    # (1) SELECCION DE MODELOS CANDIDATOS
    if select_candidates:
        
        # 1.1. Calculo metrica combinada
        l_metrics_cand = ['f1_score', 'roi']
        l_weights_cand = [1/len(l_metrics_cand) for _ in l_metrics_cand] 
        metric_cand = 'metric_cand' 
        df_ite = asses_model.calculate_combined_metric(df_ite, l_metrics=l_metrics_cand, l_weights=l_weights_cand, metric_name=metric_cand)

        ## 1.2. Filtro modelos segun metrica (y no por ROI para evitar modelos con alto ROI pero predicciones malas)
        df_ite_filt = filter_models_by_metric(df_ite, metric_col=metric_cand, prop_to_max=0.35, n_models_max=n_max_candidates) # Creo que hasta 100 esta ok, mas no. En FRA gana el 220, y yo prefiero otro.
        logger.warning(f'Shape: {df_ite.shape} --> {df_ite_filt.shape}')
        df_ite_filt.to_excel(path_cand, index=False)

    else:
        df_ite_filt = pd.read_excel(path_cand)

    # (2) BETTING STRATEGY + CALCULO DE METRICAS
    if bet_strat:
        bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths, verbose=0)

        # Actualizo missing (1 sola vez para todos los modelos)
        if assess and update_missing:
            d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
            main_next_matches.main(d_run, id_country, iteration_date=iteration_date, extract_missing=True, prepare_missing=True, export=d_run['export']) 

        # Por modelo
        for idx, row in df_ite_filt.iterrows():
            n_model, model_name= row['n_iteration'], row['model_name']
            logger.info(f'{n_model} {model_name}')
            
            # Levanto predicciones del modelo (test o test + assess)
            path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
            df_pred_test = pd.read_excel(path_test, index_col=0)

            if assess:
                logger.warning("Se estan concatenando las predicciones de TEST y ASSESS...")

                # 1. Predict missing
                d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': True} 
                df_pred_missing = main_next_matches.main(d_run, id_country, iteration_date=iteration_date, predict_missing=True, export=False) 

                # 2. Concat test + missing
                df_pred = pd.concat([df_pred_test, df_pred_missing], axis=0)

                # 3. Agrego columnas 'result' y 'expected_result' --> Lo podria implementar en betting strategy no?
                df_pred = construct_data.determine_result(df_pred) # Intento hacerlo antes con df_match pero rompia.
                df_pred = construct_data.determine_expected_result(df_pred, goals_to_xg_ratio=0.42) # Intento hacerlo antes con df_match pero rompia.

                df_pred.to_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index=True) # sin ea pero con metricas
            else:
                df_pred = df_pred_test.copy()

            # Dropeo old metrics (sino calcula mal las nuevas)
            df_pred = drop_old_metrics(df_pred)

            # 📌 Aplicar estrategia "sin_ea"
            d_params = bs.define_hiperparameters(strategy='train')  
            df_pred_met, _ = bs.calculate_roi_in_combination(df_pred, d_params)
            d_metric_sin_ea = asses_model.calculate_metrics(df_pred_met, var_resp='result', advanced_metrics=True)
            d_metric_sin_ea_ex = asses_model.calculate_metrics(df_pred_met, var_resp='expected_result', advanced_metrics=True)

            # 📌 Aplicar estrategia "con ea"
            per_res, vary_dp = True, False
            d_params = bs.define_hiperparameters(strategy='kelly_linear', big_space_m=False, vary_dp=vary_dp)  # Defino hiperparametros de estrategia de apuesta a probar. Con linear no tiene en cuenta cuotas y puede llegar a apostar mucho en cuota baja.
            if per_res:
                func = bs.define_model_betting_strategy_by_result
            else:
                func = bs.define_model_betting_strategy
            df_strat, df_pred_with_stra = func(df_pred, d_params=d_params, roi_weight=roi_weight)
            ## Solo calculo el roi que es lo unico que cambia..  --> d_metric_con_ea = asses_model.calculate_metrics(df_pred_with_stra, var_pred='predicted_result', advanced_metrics=False)
            roi_con_ea = asses_model.determine_roi(df_pred_with_stra, var_resp='result')
            ex_roi_con_ea = asses_model.determine_roi(df_pred_with_stra, var_resp='expected_result') 
            d_metric_con_ea = ({'roi_con_ea': roi_con_ea, 'expected_roi_con_ea': ex_roi_con_ea})

            # Renombro metricas para evitar sobreescribirlas
            d_metric_sin_ea = asses_model.rename_dict_keys(d_metric_sin_ea) #  suffix="sin_ea"
            d_metric_sin_ea_ex = asses_model.rename_dict_keys(d_metric_sin_ea_ex, prefix='expected_') #suffix="sin_ea"

            mult = (d_metric_con_ea['roi_con_ea'] - d_metric_sin_ea['roi'])  / abs(d_metric_sin_ea['roi'])
            mult_ex = (d_metric_con_ea['expected_roi_con_ea'] - d_metric_sin_ea_ex['expected_roi']) / abs(d_metric_sin_ea_ex['expected_roi'])

            # Guardo datos
            new_row = {'n_model': n_model, 'model_name': model_name, **d_metric_sin_ea, **d_metric_sin_ea_ex, **d_metric_con_ea, 'x_roi': mult, 'x ex_roi': mult_ex}
            rows.append(new_row)
            df_ite_bs = pd.DataFrame(data=rows)

            # Exporto datos
            df_pred_met.to_excel(f"{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx", index=True)
            df_strat.to_excel(f"{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx", index=True)
            df_pred_with_stra.to_excel(f"{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx", index=True)
            df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

    else:
        # Usar test viejo
        df_ite_bs = df_ite_filt.copy() # pd.read_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx')
        # n_model_name = 'n_iteration' # 'n_model'

    # (3) SELECCION DEL MODELO (que maximiza la metrica combinada)
    ## 3.1. Calculo metrica combinada
    metric = 'metric_assess'
    df_ite_bs = asses_model.calculate_combined_metric(df_ite_bs, l_metrics=l_metrics, l_weights=l_weights, metric_name=metric)

    ## 3.2. Ordeno por metrica combinada
    df_ite_bs = df_ite_bs.sort_values(by=metric, ascending=False)  # Ordenar los registros por 'metric' en orden descendente
    idx_max = df_ite_bs[metric].idxmax() # Pero ahora sin ea realmente. No uso kelly sino linear sin cuotas.
    logger.critical(f"Modelo seleccionado: {df_ite_bs.loc[idx_max, 'n_model']} {df_ite_bs.loc[idx_max, 'model_name']}")
    
    # Exporto datos
    if export:
        df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

    return df_ite_bs

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [55, 59, 77, 148]

    # Defino hiperparametros
    select_candidates, n_max_candidates = True, 100
    bet_strat = True
    assess = False if bet_strat else False   # Tarda banda. Ver de volver a poner predict_missing en mnm. 
    update_missing = False if assess else False
    predict_missing = False if assess else False
    export = True

    d_countries = {
        # Train nuevos
        6: ["argentina", '2025-02-06'], 
        48: ["england", '2025-03-03'],
        55: ["france", '2025-03-03'], 
        59: ["germany", '2025-03-04'],
        77: ["italy", '2025-03-04'],
        148: ["spain", '2025-03-04'], 
        }
    
    # Defino pesos
    # l_metrics = ['acerte_draw_sin_ea', 'expected_acerte_draw_sin_ea', 'n_emp_sin_ea'] # total f1 socre lo uso para candidatos y no me interesa tanto ahora (busco max el 0)
    # l_weights = [0.5, 0.5, 0.25]

    # l_metrics = ['f1_score_sin_ea', 'acerte_draw_sin_ea', 'expected_acerte_draw_sin_ea'] # 'n_emp_sin_ea'? para favorecer mas empates predichos? --> no creo que es mejor tener un test mas robusto...
    # l_weights = [0.48, 0.46, 0.07]

    d_metrics = {
        48: {'gp_draw': 0.393902, 'expected_gp_draw': 0.3127, 'f1_score': 0.2933},
        55: {'f1_score_home': 0.2767, 'f1_score': 0.255, 'acerte_draw':  0.24004, 'expected_f1_score': 0.2279},
        59: {'expected_gp_away': 0.4171, 'gp_home': 0.2919, 'gp_away': 0.2908},
        77: {'expected_f1_score': 0.52601, 'acerte_draw': 0.4739},
        148: {'roi': 0.4364, 'f1_score_draw': 0.3201, 'gp_away': 0.24337}
    }

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        l_metrics, l_weights = d_metrics[id_country].keys(), d_metrics[id_country].values()
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
        print(df_ite)

        main(
            df_ite=df_ite,
            id_country=id_country, country=country, iteration_date=iteration_date, 
            l_metrics=l_metrics, l_weights=l_weights, 
            select_candidates=select_candidates,
            n_max_candidates=n_max_candidates,
            bet_strat=bet_strat,
            assess=assess,
            update_missing=update_missing,
            predict_missing=predict_missing,
            export=export
            )
        