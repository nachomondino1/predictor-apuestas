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
def filter_models_by_metric(df, metric_col, prop_to_max: float = None, perc_cutoff: float = 100, n_models_max: int = None, verbose: int = 0):
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
    if prop_to_max is None:
        metric_cut_prop = -100
    else:
        metric_cut_prop = max_metric * prop_to_max

    ## segun percentil 
    metric_cut_perc = np.percentile(df[metric_col], (100-perc_cutoff)) # Seleccionar el 20% de los registros con los valores más altos de 'metric'
    
    ## segun cantidad maxima de modelos
    n_models_max = len(df) if n_models_max is None else n_models_max
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

def filter_models_by_distribution(df, verbose: int = 0):
    """
    Selecciono solo los modelos con una distribución de predicted_result similar 
    a la distribución de resultados en la realidad.
    """
    logger.info("Paso 2: Descartando modelos según distribución en df_test")

    # Hago la diferencia absoluta
    df['abs_dif'] = abs(df['%_dif'])

    # Determino diferencia maxima tolerada
    thr_fixed = 0.3
    thr_perc = df['abs_dif'].quantile(0.5)
    thr = min(thr_fixed, thr_perc)
    print(f"Diferencia maxima tolerada: {thr_fixed} {thr_perc} --> {thr}")

    # Quiero eliminar modelos solo si el %_dif es mayor a cierto threshold y no es porque el empate tiene de menos y se lo da a otro resultado?
    df_filtered = df[df['abs_dif'] < thr]

    if verbose >= 0:
        logger.warning(f'Descarte por distribucion: {len(df)} --> {len(df_filtered)}' )
        # logger.info(f"Se eliminaron {len(l_idx_to_remove)} modelos por distribucion muy distinta a la de results.")

    return df_filtered

def error_empty_dataframe(df):
    if len(df) == 0:
        logger.error("El dataframe esta vació. Probablemente uno de los filtros eliminó todos los modelos que quedaban.")
        raise ValueError

# Main
def main(
        df_ite,
        id_country, 
        country, 
        iteration_date,
        l_metrics: list, 
        l_weights: list, 
        assess: bool = False,
        bet_strat: bool = True,
        update_missing: bool = True,
        predict_missing: bool = True,
        use_assess_saved: bool = True,
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
    d_paths = initialize_directories(country, iteration_date, assess)

    # (1) Assess --> Fuera de servicio desde el 17/03/2025
    if assess:
        # (2) BETTING STRATEGY + CALCULO DE METRICAS
        if bet_strat:
            bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths, verbose=0)

            # Actualizo missing (1 sola vez para todos los modelos)
            if update_missing:
                d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
                main_next_matches.main(d_run, id_country, iteration_date=iteration_date, export=d_run['export']) 

            # Por modelo
            for idx, row in df_ite.iterrows():
                n_model, model_name = row['n_iteration'], row['model_name']
                logger.info(f'{n_model} {model_name}')
                
                # Levanto predicciones del modelo (test o test + assess)
                path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
                df_pred_test = pd.read_excel(path_test, index_col=0)

                if predict_missing:
                    logger.warning("Se estan concatenando las predicciones de TEST y ASSESS...")

                    # 1. Predict missing
                    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': True} 
                    d_model = {'n_model': n_model, 'model_name': model_name}
                    df_pred_missing = main_next_matches.main(d_run, id_country, iteration_date=iteration_date, predict_missing=True, d_model=d_model, export=False) 

                    # 2. Concat test + missing
                    df_pred = pd.concat([df_pred_test, df_pred_missing], axis=0)

                    # 3. Agrego columnas 'result' y 'expected_result' --> Lo podria implementar en betting strategy no?
                    df_pred = construct_data.determine_result(df_pred) # Intento hacerlo antes con df_match pero rompia.
                    df_pred = construct_data.determine_expected_result(df_pred, goals_to_xg_ratio=0.42) # Intento hacerlo antes con df_match pero rompia.

                    df_pred.to_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index=True) # sin ea pero con metricas
                
                elif use_assess_saved:
                    df_pred = pd.read_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index_col=0)
                
                else:
                    df_pred = df_pred_test.copy()

                # Dropeo old metrics (sino calcula mal las nuevas)
                df_pred = asses_model.drop_old_metrics(df_pred)

                # 📌 Aplicar estrategia "sin_ea"
                d_params = bs.define_hiperparameters(strategy='train')  
                df_pred_met, _ = bs.calculate_roi_in_combination(df_pred, d_params)
                
                ## Calculo metricas
                d_metric_sin_ea = asses_model.calculate_metrics(df_pred_met, var_resp='result', advanced_metrics=True)
                d_metric_sin_ea_ex = asses_model.calculate_metrics(df_pred_met, var_resp='expected_result', advanced_metrics=True)
                
                ## Renombro metricas para evitar sobreescribirlas
                d_metric_sin_ea_ex = asses_model.rename_dict_keys(d_metric_sin_ea_ex, prefix='expected_') #suffix="sin_ea"

                # 📌 Aplicar estrategia "con ea"
                per_res, vary_dp = True, False
                d_params = bs.define_hiperparameters(strategy='linear', big_space_m=True, vary_dp=vary_dp) # kelly_linear  # Defino hiperparametros de estrategia de apuesta a probar. Con linear no tiene en cuenta cuotas y puede llegar a apostar mucho en cuota baja.
                if per_res:
                    func = bs.define_model_betting_strategy_by_result
                else:
                    func = bs.define_model_betting_strategy
                df_strat, df_pred_with_stra = func(df_pred, d_params=d_params, roi_weight=1)
                ## Calculo metricas  ## Solo calculo el roi que es lo unico que cambia.. o que me interesa medir
                roi_con_ea = asses_model.determine_roi(df_pred_with_stra, var_resp='result')
                ex_roi_con_ea = asses_model.determine_roi(df_pred_with_stra, var_resp='expected_result') 
                d_metric_con_ea = ({'roi_con_ea': roi_con_ea, 'expected_roi_con_ea': ex_roi_con_ea})
                mult = (d_metric_con_ea['roi_con_ea'] - d_metric_sin_ea['roi'])  / abs(d_metric_sin_ea['roi'])
                mult_ex = (d_metric_con_ea['expected_roi_con_ea'] - d_metric_sin_ea_ex['expected_roi']) / abs(d_metric_sin_ea_ex['expected_roi'])

                # Guardo datos
                new_row = {'n_model': n_model, 'model_name': model_name, **d_metric_sin_ea, **d_metric_sin_ea_ex, **d_metric_con_ea, 'x_roi': mult, 'x ex_roi': mult_ex}
                rows.append(new_row)
                df_ite_bs = pd.DataFrame(data=rows)

                # Exporto datos
                if use_assess_saved or predict_missing:
                    df_pred_met.to_excel(f"{d_paths['path_assess']}/{n_model}__{model_name}_predicciones_met.xlsx", index=True) # Cuando haces assess
                df_strat.to_excel(f"{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx", index=True)
                df_pred_with_stra.to_excel(f"{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx", index=True)
                df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

        else:
            df_ite_bs = pd.read_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx')
            # n_model_name = 'n_iteration' # 'n_model'
    else:
        # Usar test viejo
        df_ite_bs = df_ite.copy()

    # (2) SELECCION DEL MODELO (que maximiza la metrica combinada)
    ## 3.1. Calculo metrica combinada
    metric = 'metric_test_assess'
    df_ite_bs = asses_model.calculate_combined_metric(df_ite_bs, l_metrics=l_metrics, l_weights=l_weights, metric_name=metric)

    ## 3.2. Ordeno por metrica combinada
    df_ite_bs = df_ite_bs.sort_values(by=metric, ascending=False)  # Ordenar los registros por 'metric' en orden descendente

    # 3.3. Seleccion del modelo
    idx_max = df_ite_bs[metric].idxmax() # Pero ahora sin ea realmente. No uso kelly sino linear sin cuotas.
    n_model, model_name = df_ite_bs.loc[idx_max, 'n_iteration'], df_ite_bs.loc[idx_max, 'model_name_x']
    logger.critical(f"Modelo seleccionado: {n_model} {model_name}") # n_model

    # Exporto datos
    if export:
        df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

    return df_ite_bs

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [48]

    d_countries = {
        # 6: ["argentina", '2025-02-06'], 
        48: ["england", '2025-03-23'],
        55: ["france", '2025-03-23'], 
        59: ["germany", '2025-03-23'],
        77: ["italy", '2025-03-23'],
        148: ["spain", '2025-03-24']
        }
    
    # Defino metricas y pesos
    l_metrics = ['f1_score_away', 'f1_score_draw', 'error']  # paso 'expected_error' por poca corr entre si? ademas alta corr con expected_f1_score_away (74%)
    l_weights = [0.25, 0.25, 0.5]

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        # l_metrics, l_weights = d_metrics[id_country].keys(), d_metrics[id_country].values()
        # l_weights = d_weights[id_country]

        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
        print(df_ite)

        # para maximizar error en metrica
        df_ite['error'] = df_ite['error'] * (-1)
        df_ite['expected_error'] = df_ite['expected_error'] * (-1)

        main(
            df_ite=df_ite,
            id_country=id_country, country=country, iteration_date=iteration_date, 
            l_metrics=l_metrics, l_weights=l_weights,
            assess=False,
            export=True
            )
        