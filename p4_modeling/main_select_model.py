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

def main(
        df_ite,
        id_country, 
        country, 
        iteration_date,
        select_candidates: bool = True,
        n_max_candidates: int = 30,
        recalculate_metrics: bool = True,
        assess: bool = True,
        update_missing: bool = True,
        predict_missing: bool = True,
        select_model: bool = True,
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
    d_paths = initialize_directories(country, iteration_date, predict_missing)
    path_cand = f'{d_paths['path_select']}/df_filt_by_metric_cand.xlsx'

    # Definicion de variables
    rows = []
    # l_metrics = ['roi_sin_ea', 'roi_last_50_sin_ea', 'roi_last_25_sin_ea', 'f1_score_sin_ea', 'f1_score_last_50_sin_ea', 'f1_score_last_25_sin_ea']
    l_metrics = ['roi_con_ea', 'roi_last_50_con_ea', 'roi_last_25_con_ea', 'f1_score_sin_ea', 'f1_score_last_50_sin_ea', 'f1_score_last_25_sin_ea']
    l_weights = [1/len(l_metrics) for _ in l_metrics]
    

    # (1) SELECCION DE MODELOS CANDIDATOS
    if select_candidates:
        
        # 1.1. Calculo metrica combinada
        l_metrics_cand, l_weights_cand = ['roi', 'f1_score'], [0.5, 0.5] # Temporal pues no tengo n last matches en test...
        metric = 'metric_cand' 
        df_ite = asses_model.calculate_combined_metric(df_ite, l_metrics=l_metrics_cand, l_weights=l_weights_cand, metric_name=metric)

        ## 1.2. Filtro modelos segun metrica (y no por ROI para evitar modelos con alto ROI pero predicciones malas)
        df_ite_filt = filter_models_by_metric(df_ite, metric_col=metric, prop_to_max=0.35, n_models_max=n_max_candidates) # Creo que hasta 100 esta ok, mas no. En FRA gana el 220, y yo prefiero otro.
        logger.warning(f'Shape: {df_ite.shape} --> {df_ite_filt.shape}')
        df_ite_filt.to_excel(path_cand, index=False)

    else:
        df_ite_filt = pd.read_excel(path_cand)

    # (2) BETTING STRATEGY + CALCULO DE METRICAS
    if recalculate_metrics:
        bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths, verbose=0)

        # Por modelo
        for idx, row in df_ite_filt.iterrows():
            n_model, model_name= row['n_iteration'], row['model_name']
            logger.info(f'{n_model} {model_name}')
            
            # Levanto predicciones del modelo (test o test + assess)
            path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
            if assess:
                df_pred = update_test_with_assess(path_test, n_model, model_name, id_country, country, iteration_date, d_paths)
            else:
                df_pred = pd.read_excel(path_test, index_col=0)

            # 📌 Aplicar estrategia "sin_ea"
            d_params = bs.define_hiperparameters(strategy='train')  
            df_pred_met, _, __ = bs.calculate_roi_in_combinations(df_pred, d_params=d_params)
            d_metric_sin_ea = calculate_all_metrics(df_pred_with_stra, suffix="sin_ea", advanced_metrics=True)

            # 📌 Aplicar estrategia "con ea"
            df_strat, df_pred_with_stra = apply_betting_strategy(df_pred, bs, per_res=True, vary_dp=True, vary_m=True)
            d_metric_con_ea = calculate_all_metrics(df_pred_with_stra, suffix="con_ea")

            # Guardo datos
            new_row = {'n_model': n_model, 'model_name': model_name, **d_metric_sin_ea, **d_metric_con_ea}
            rows.append(new_row)
            df_ite_bs = pd.DataFrame(data=rows)

            # Exporto datos
            df_pred_met.to_excel(f"{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx", index=True)
            df_strat.to_excel(f"{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx", index=True)
            df_pred_with_stra.to_excel(f"{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx", index=True)
            df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

        ## 3.1. Calculo metrica combinada
        metric, n_model_name = 'metric_assess', 'n_model'
        df_ite_bs = asses_model.calculate_combined_metric(df_ite_bs, l_metrics=l_metrics, l_weights=l_weights, metric_name=metric)

    else:
        # Usar test viejo
        df_ite_bs = df_ite_filt.copy() # pd.read_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx')
        n_model_name = 'n_iteration' # 'n_model'
        # metric = 'metric_assess'

    # (3) SELECCION DEL MODELO (que maximiza la metrica combinada)
    if select_model:
        df_ite_bs = df_ite_bs.sort_values(by=metric, ascending=False)  # Ordenar los registros por 'metric' en orden descendente
        idx_max = df_ite_bs[metric].idxmax() # Pero ahora sin ea realmente. No uso kelly sino linear sin cuotas.
        n_model, model_name = df_ite_bs.loc[idx_max, n_model_name], df_ite_bs.loc[idx_max, 'model_name']
        logger.critical(f"Modelo seleccionado: {n_model} {model_name}")
    
    else:
        logger.warning("Se evitó seleccionar un modelo")

    # Exporto datos
    if export:
        df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

    return df_ite_bs

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

# Assess
def update_test_with_assess(path_test, n_model, model_name, id_country, country, iteration_date, d_paths):

    logger.warning("Se estan concatenando las predicciones de TEST y ASSESS...")

    # 2.1. Actualizo missing
    if update_missing:
        d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
        main_next_matches.main(d_run, id_country, iteration_date=iteration_date, extract_missing=True, prepare_missing=True, export=d_run['export']) 

    # 2.2. Predict missing
    df_pred_test = pd.read_excel(path_test, index_col=0)
    df_pred = get_model_predictions_with_missing(df_pred_test=df_pred_test, n_model=n_model, model_name=model_name, id_country=id_country, country=country, iteration_date=iteration_date) # sin ea
    df_pred.to_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index=True) # sin ea pero con metricas
    
    return df_pred

def get_model_predictions_with_missing(df_pred_test, n_model, model_name, id_country, country, iteration_date):
    """
    Obtengo df_probabilities test + missing.
    """
    # 1. Predict missing
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True}  # No puedo correr data_unders = False si los proximos partidos ya estan en missing.
    d_model = {'n_model': n_model, 'model_name': model_name} # XGBClassifier, neural_networ, SVC, LogisticRegression, MLPClassifier

    # Usar mnm.py con predict_missing=True y data_unders=False.
    df_pred_missing = main_next_matches.main(d_run, id_country, iteration_date=iteration_date, d_model=d_model, predict_missing=True, export=False, verbose=0) 

    if not isinstance(df_pred_missing, pd.DataFrame):
        raise ValueError("No se generó un dataframe.")

    elif len(df_pred_missing) == 0:
        logger.warning("No hay predicciones de partidos missing...")
        raise ValueError
    
    # 2. Concateno df_pred_test y df_pred missing.
    df_predicciones = pd.concat([df_pred_test, df_pred_missing], axis=0)
    l_cols = ['date', 'id_team_home', 'id_team_away', 'predicted_result', 'prob_class_1', 'prob_class_2', 'prob_class_0', 'odds_home', 'odds_draw', 'odds_away']
    df_predicciones = df_predicciones.loc[:, l_cols]

    #  Agrego columnas result y expected result
    df_predicciones = get_goals(df_predicciones, country)  

    # Determino result y expected result segun goals
    df_predicciones = construct_data.determine_result(df_predicciones) # Intento hacerlo antes con df_match pero rompia.
    df_predicciones = construct_data.determine_expected_result(df_predicciones, goals_to_xg_ratio=0.42) # Intento hacerlo antes con df_match pero rompia.
    return df_predicciones

def get_goals(df, country):  # Ponerlo como funcion dentro de BettingStrategy????
    """
    Agregar columnas result y acerte en df_pred. Determino result y expected result para poder determinar "acerte"
    """
    ## Levanto df_match_miss para obtener goals? ??
    df_match_miss = pd.read_excel(f"data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx", index_col=0)
    l_columns_to_copy = ['goals_home', 'goals_away', 'expected_goals_(xg)_home', 'expected_goals_(xg)_away']   # Columnas a copiar
    
    # Ordeno df por date
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
    df = df.sort_values(by='date', ascending=True)

    # Asignar valores de df_match_miss a df solo en las columnas y filas correspondientes
    for idx, row in df.iterrows():

        for column in l_columns_to_copy:

            df.loc[idx, column] = df_match_miss.loc[idx, column] 

    return df

# Betting Strategy
def calculate_all_metrics(df, suffix, advanced_metrics=False):
    """
    Calcula métricas generales y para los últimos 50 y 25 partidos, agregando un sufijo a las claves.

    :param df: DataFrame con las predicciones.
    :param suffix: Sufijo para renombrar las métricas (ejemplo: "sin_ea" o "con_ea").
    :param advanced_metrics: Si True, calcula métricas avanzadas.
    :return: Diccionario con las métricas calculadas.
    """
    d_metrics = asses_model.calculate_metrics(df, advanced_metrics=advanced_metrics)

    for num_matches in [50, 25]:
        d_metrics.update(asses_model.calculate_metrics_for_last_matches(df, num_matches=num_matches))

    return {f"{k}_{suffix}": v for k, v in d_metrics.items()}

def apply_betting_strategy(df_pred, bs, per_res: bool = False, vary_dp: bool = False, vary_m: bool = False):
    """
    Aplico ≠ estrategias de apuesta.

    # Parameters:
        df_pred: Dataframe con predicciones a las cuales aplicar estrategia.
        bs: Instancia de clase BettingStrategy()
        mode: Tipo de estrategia a usar. 
            'equal' para usar mismo m en todos los rdos. 
            'per_res' para usar un m y dp ≠ por res. 
            'dp_per_res' para usar mismo m pero dp ≠ por rdo.
    """
    d_params = bs.define_hiperparameters(strategy='linear')  # Defino hiperparametros de estrategia de apuesta a probar. Con linear no tiene en cuenta cuotas y puede llegar a apostar mucho en cuota baja.
        
    # estrategia x rdo + dp
    if per_res:

        # Completo
        if vary_m and vary_dp:
            df_strat, df_pred_with_stra = bs.define_model_betting_strategy_by_result(df_pred, d_params=d_params)

        # Solo vario el m
        elif vary_m:
            d_params['prob_dp'] = [0]
            df_strat, df_pred_with_stra = bs.define_model_betting_strategy_by_result(df_pred, d_params=d_params)

        # Solo vario el dp
        elif vary_dp:
            # Defino m comun a todos los rdos
            d_params_m = {'prob_dp': [0], 'curva': ['linear'], 'm': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 65, 80, 95, 110, 140, 170, 200], 'b': [0]}
            df_strat_1, df_pred_with_stra_1 = bs.define_model_betting_strategy(df_pred, d_params=d_params_m)
            m_sel = df_strat_1['m'].values[0]

            # Defino dp por resultado usando el m ya definido
            d_params_dp = {'prob_dp': [0, 0.45, 0.6, 0.75, 0.9], 'curva': ['linear'], 'm': [m_sel], 'b': [0]}
            df_strat, df_pred_with_stra = bs.define_model_betting_strategy_by_result(df_pred, d_params=d_params_dp)  # antes no lo hacia por rdo.
        
        # para tener mismo bank across all results.
        df_pred_with_stra, _ = asses_model.calculate_roi(df_pred_with_stra) 
        
    ## Mismo m todos los rdos
    else:
        df_strat, df_pred_with_stra = bs.define_model_betting_strategy(df_pred, d_params=d_params)

    return df_strat, df_pred_with_stra

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    # l_countries = [59, 77, 148]
    l_countries = [6]

    # Defino hiperparametros
    select_candidates = True
    recalculate_metrics = True
    assess = False if recalculate_metrics else False
    update_missing = False if assess else False
    predict_missing = False if assess else False
    export = True

    d_countries = {
        # Train nuevos
        6: ["argentina", '2025-02-06'], 
        48: ["england", '2025-02-05'],
        55: ["france", '2025-02-05'], 
        59: ["germany", '2025-02-05'],
        77: ["italy", '2025-02-05'],
        148: ["spain", '2025-02-05'], 
        }

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]

        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
        print(df_ite)

        main(
            df_ite=df_ite,
            id_country=id_country, country=country, iteration_date=iteration_date, 
            select_candidates=select_candidates,
            recalculate_metrics=recalculate_metrics,
            assess=assess,
            update_missing=update_missing,
            predict_missing=predict_missing,
            export=export
            )
        