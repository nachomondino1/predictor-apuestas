import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
from utils.set_up_logging import logger
from utils import directories
import datetime
from p4_modeling import select_model_for_prod, betting_strategy, assess_models_in_prod, asses_model
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
    # Definicion de variables
    rows = []
    d_paths = initialize_directories(country, iteration_date, predict_missing)
    bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths, verbose=0)
    l_metrics = ['roi_sin_ea', 'roi_last_50_sin_ea', 'roi_last_25_sin_ea', 'f1_score_sin_ea', 'f1_score_last_50_sin_ea', 'f1_score_last_25_sin_ea']
    l_weights = [1/len(l_metrics) for _ in l_metrics]

    # (1) SELECCION DE MODELOS CANDIDATOS
    if select_candidates:
        sbm = select_model_for_prod.SelectBestModel(id_country=id_country, path_save=d_paths['path_select'])  # Creo instancia de sbm
        
        # 1.1. Calculo metrica combinada
        l_metrics_cand, l_weights_cand = ['roi', 'f1_score'], [0.5, 0.5] # Temporal pues no tengo n last matches en test...
        metric = 'metric_cand' 
        df_ite = asses_model.calculate_combined_metric(df_ite, l_metrics=l_metrics_cand, l_weights=l_weights_cand, metric_name=metric)

        ## 1.2. Filtro modelos segun metrica (y no por ROI para evitar modelos con alto ROI pero predicciones malas)
        df_ite_filt = sbm.filter_models_by_metric(df_ite, metric_col=metric, prop_to_max=0.35, n_models_max=n_max_candidates) # Creo que hasta 100 esta ok, mas no. En FRA gana el 220, y yo prefiero otro.
        logger.warning(f'Shape: {df_ite.shape} --> {df_ite_filt.shape}')
    
    else:
        df_ite_filt = pd.read_excel(f'{d_paths['path_select']}/df_filt_by_metric_cand.xlsx')

    # (2) Assses + Recalculo de metricas
    # Por modelo
    if assess:

        # 2.1. Actualizo missing
        if update_missing:
            d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
            main_next_matches.main(d_run, id_country, iteration_date=iteration_date, extract_missing=True, prepare_missing=True, export=d_run['export']) 

        # 2.2. Predicciones test + missing
        # Por modelo
        for idx, row in df_ite_filt.iterrows():

            n_model, model_name= row['n_iteration'], row['model_name']
            logger.info(f'{n_model} {model_name}')
            
            # Obtengo predicciones missing + concateno test y missing
            if predict_missing:
                # Levanto df_pred_test
                df_pred_test = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx", index_col=0)
                logger.info(df_pred_test.shape)

                df_pred = assess_models_in_prod.get_model_predictions_with_missing(df_pred_test=df_pred_test, n_model=n_model, model_name=model_name, id_country=id_country, country=country, iteration_date=iteration_date) # sin ea
                df_pred.to_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index=True) # sin ea pero con metricas
            else:
                df_pred = pd.read_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx') # sin ea pero con metricas
                df_pred = df_pred.loc[:, ~df_pred.columns.str.startswith('Unnamed')]
            
            # Defino predicciones sin estrategia 
            d_params = bs.define_hiperparameters(strategy='train')  # Defino hiperparametros de estrategia de apuesta a probar. Con linear no tiene en cuenta cuotas y puede llegar a apostar mucho en cuota baja.
            df_pred_met, _, __ = bs.calculate_roi_in_combinations(df_pred, d_params=d_params)
            if predict_missing:
                df_pred_met.to_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index=True) # sin ea pero con metricas

            # Defino predicciones con estrategia (lo hago aqui solo para tener los rdos de todos los modelos con su ea)
            df_strat, df_pred_with_stra = apply_betting_strategy(df_pred, bs, per_res=True, vary_dp=True, vary_m=False)
            df_strat.to_excel(f'{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx', index=True)
            df_pred_with_stra.to_excel(f'{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx', index=True)
          
            # Recalculo metricas (test + assess)
            ## sin ea 
            d_metric_sin_ea = asses_model.calculate_metrics(df_pred_met, advanced_metrics=True)
            for num_matches in [50, 25]:
                d_metrics_last = asses_model.calculate_metrics_for_last_matches(df_pred_met, num_matches=num_matches)
                d_metric_sin_ea.update(d_metrics_last)
            ## con ea
            d_metric_con_ea = asses_model.calculate_metrics(df_pred_with_stra) 

            ## Renombro metricas para que no se sobreescriban
            d_metric_sin_ea_renamed = {f"{k}_sin_ea": v for k, v in d_metric_sin_ea.items()}
            d_metric_con_ea_renamed = {f"{k}_con_ea": v for k, v in d_metric_con_ea.items()}

            # Guardo datos
            new_row = {
                'n_model': n_model, 'model_name': model_name, 
                **d_metric_sin_ea_renamed,
                **d_metric_con_ea_renamed
                }
            rows.append(new_row)
            df_ite_bs = pd.DataFrame(data=rows)

            # Exporto datos
            df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

        ## 3.1. Calculo metrica combinada
        metric = 'metric_assess'
        df_ite_bs = asses_model.calculate_combined_metric(df_ite_bs, l_metrics=l_metrics, l_weights=l_weights, metric_name=metric)
        n_model_name = 'n_model'

    else:
        # Usar test viejo
        df_ite_bs = df_ite_filt.copy()
        n_model_name = 'n_iteration'

        # Usar test + assess viejo
        # df_ite_bs = pd.read_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx')
        # n_model_name = 'n_model'
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
            d_params_dp = {'prob_dp': [0, 0.45, 0.55, 0.65, 0.75, 0.85], 'curva': ['linear'], 'm': [m_sel], 'b': [0]}
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
    # l_countries = [148]

    # Defino hiperparametros
    select_candidates = False
    assess = True
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
            assess=assess,
            update_missing=update_missing,
            predict_missing=predict_missing,
            export=export
            )
        