import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
from utils.set_up_logging import logger
from utils import directories
import datetime
from p4_modeling import select_model_for_prod, betting_strategy, assess_models_in_prod, asses_model
from p6_deployment import main_next_matches
import os


def initialize_directories(country, iteration_date, predict_missing, bet_strategy):
    """
    Guarda la selección de modelo vieja en una carpeta y crea los directorios necesarios.
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
        'path_assess_dep': f"data/{country}/p6_deployment/assess"
    }

    # Mover toda la carpeta si predict_missing y bet_strategy son True
    if predict_missing and bet_strategy:
        if os.path.exists(base_path_sbm):
            directories.make_directories([d_paths['path_old']])
            directories.mover_archivo(base_path_sbm, d_paths['path_old'])

    # Mover solo path_assess si predict_missing es True
    elif predict_missing:
        directories.make_directories([d_paths['path_old']])
        directories.mover_archivo(d_paths['path_assess'], d_paths['path_old'])

    # Mover solo path_bet_strategy si bet_strategy es True
    elif bet_strategy:
        directories.make_directories([d_paths['path_old']])
        directories.mover_archivo(d_paths['path_bet_strategy'], d_paths['path_old'])

    # Crear directorios necesarios
    directories.make_directories([
        d_paths['path_assess'], 
        d_paths['path_select'], 
        d_paths['path_bet_strategy'], 
        d_paths['path_assess_dep']
    ])

    return d_paths

def main(
        df_ite,
        id_country, 
        country, 
        iteration_date,
        update_missing: bool = True,               # Assess
        predict_missing: bool = True,     # Assess
        betting_strat: bool = True,
        strategy: str = 'kelly',                  # Betting Strategy
        export: bool = True,
        verbose: int = 0
        ):
    """
    Assess model in prod + Seleccion del modelo + Estrategia de apuesta

    # Parameters
        df_ite: Dataframe con el train y test de los modelos del pais. (DataFrame)
        assess: True para recolectar las predicciones de los modelos en los partidos missing y concatenarlas a las de test. (bool)

    # Return
        Modelo a usar en produccion con su estrategia de apuesta optima (teniendo en cuenta test + missing). 
    """
    # Definicion de variables
    rows = []
    d_rows = {}
    d_paths = initialize_directories(country, iteration_date, predict_missing, betting_strat)
    ## Betting strategy
    bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths, verbose=0)
    d_params = bs.define_hiperparameters(strategy=strategy)
    ## Seleccion de modelo
    l_metrics = ['roi_last_25_matches_sin_ea', 'ROI_sin_ea', 'roi_last_25_matches_con_ea', 'ROI_con_ea']
    l_weights = [1 / len(l_metrics) for _ in l_metrics]

    # (0) Actualizo missing
    if update_missing:
        d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
        main_next_matches.main(d_run, id_country, iteration_date=iteration_date, extract_missing=True, prepare_missing=True, export=d_run['export']) 

    # (1) SELECCION DE MODELOS CANDIDATOS
    ## Creo instancia de sbm
    sbm = select_model_for_prod.SelectBestModel(id_country=id_country, path_save=d_paths['path_select'])

    ## Determino metrica mas importante entre test_acc, recall y f1-score
    metrics = asses_model.select_metrics(df_ite, col_corr="roi_por_partido", l_metrics=['test_accuracy', 'recall', 'f1_score'] , n_metrics=1)

    ## Filtro modelos segun metrica mas importante (y no por ROI para evitar modelos con alto ROI pero predicciones malas)
    df_ite_filt = sbm.filter_models_by_metric(df_ite, metric_col=metrics[0], perc_cutoff=1, n_models_max=15)
    logger.warning(f'Shape: {df_ite.shape} --> {df_ite_filt.shape}')

    # (2) ESTRATRAGIA DE APUESTA
    if betting_strat:
        
        # Por modelo
        for idx, row in df_ite_filt.iterrows():

            n_model, model_name= row['n_iteration'], row['model_name']
            logger.info(f'{n_model} {model_name}')
            
            # Levanto df_predicciones
            if predict_missing:
                df_pred = assess_models_in_prod.get_model_predictions_with_missing(n_model=n_model, model_name=model_name, id_country=id_country, country=country, iteration_date=iteration_date) # sin ea
            else: 
                try:
                    df_pred = pd.read_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index_col=0) # sin ea
                    logger.info("Levanto df_pred sin ea del assess ya recolectado...")
                except FileNotFoundError:
                    df_pred = pd.read_excel(f"{d_paths['base_path']}/models/{n_model}__{model_name}_predicciones.xlsx", index_col=0) # sin ea
                    logger.info("Levanto df_pred sin ea de cuando entrene modelos (solo test)...")
                logger.info(df_pred)

            # Recalculo metricas   
            df_pred, d_metric = assess_models_in_prod.determine_metrics(df_pred, country, iteration_date)
            if predict_missing:
                df_pred.to_excel(f'{d_paths['path_assess']}/{n_model}__{model_name}_predicciones.xlsx', index=True) # sin ea pero con metricas
            
            # Defino estrategia
            df, df_pred_with_stra = bs.define_model_betting_strategy_by_result(df_pred, col_to_max='G/P', d_params=d_params)

            # Calculo metricas para guardar en df_ite_bs
            ## Calculo ROIs en last matches 
            roi_values_sin_ea = asses_model.calculate_last_matches_roi(df_pred, l_last_matches=[25], roi_col='G/P', extension='sin_ea')  # Sin ea
            roi_values_con_ea = asses_model.calculate_last_matches_roi(df_pred_with_stra, l_last_matches=[25], roi_col='G/P', extension='con_ea') # Con ea # sin bank 
            ## Calculo ROI con y sin ea
            roi_sin_bank_sin_ea, roi_sin_bank_con_ea = df_pred['G/P_sin_bank'].sum(), df_pred_with_stra['G/P_sin_bank'].sum()
            roi_sin_ea, roi_con_ea = df_pred['G/P'].sum(), df_pred_with_stra['G/P'].sum()
            multiplicador = (roi_con_ea - roi_sin_ea) / abs(roi_sin_ea)
            
            # Guardo datos
            new_row = {
                'n_model': n_model, 'model_name': model_name, 
                **d_metric,
                **roi_values_sin_ea, **roi_values_con_ea, 
                'ROI_sin_bank_sin_ea': roi_sin_bank_sin_ea, 'ROI_sin_bank_con_ea': roi_sin_bank_con_ea, 
                'ROI_sin_ea': roi_sin_ea, 'ROI_con_ea': roi_con_ea, 'x ea': multiplicador,
                }
            rows.append(new_row)
            d_rows[n_model] = [df, df_pred_with_stra]

            # Exporto datos (x seg)
            df_ite_bs = pd.DataFrame(data=rows)
            df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)
            if verbose >= 2:
                df.to_excel(f'{d_paths['path_bet_strategy']}/__df_strategy_{n_model}_{model_name}.xlsx', index=True)
                df_pred_with_stra.to_excel(f'{d_paths['path_bet_strategy']}/__predicciones_{n_model}_{model_name}.xlsx')
    
    else:
        df_ite_bs = pd.read_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx')
        logger.warning(f"Se evito redefinir estrategia de apuesta por modelo. \n{df_ite_bs}")

    # (3) SELECCION DEL MODELO (el que maximiza el ROI con ea)
    ## Calculo metrica combinada
    metric_col = 'metric'
    df_ite_bs = asses_model.calculate_combined_metric(df_ite_bs, l_metrics=l_metrics, l_weights=l_weights)
    df_ite_bs = df_ite_bs.sort_values(by=metric_col, ascending=False)  # Ordenar los registros por 'metric' en orden descendente

    ## Selecciono el modelo que maximiza la metrica combinada
    idx_max = df_ite_bs[metric_col].idxmax()
    n_model, model_name = df_ite_bs.loc[idx_max, 'n_model'], df_ite_bs.loc[idx_max, 'model_name']
    logger.critical(f"Modelo seleccionado: {n_model} {model_name}")

    # Exporto datos
    if export:
        df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)
        df, df_pred_with_stra = d_rows[n_model]
        df.to_excel(f'{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx', index=True)
        df_pred_with_stra.to_excel(f'{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx')


if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    # l_countries = [6]
    
    # Defino hiperparametros
    update_missing = False  # Extract missing + Prepare missing
    betting_strat = True # Recalcular estrategia de apuesta por modelo 
    predict_missing = False # Predecir missing x modelo. betting_strategy debe ser True.
    export = True

    d_countries = {
        # Train actuales
        6: ["argentina", '2025-01-28'], 
        48: ["england", '2025-01-22'],
        55: ["france", '2025-01-22'], 
        59: ["germany", '2025-01-23'], 
        77: ["italy", '2025-01-20'],
        148: ["spain", '2025-01-20'], 
        167: ["usa", '2024-12-05']
        }

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")

        main(
            df_ite=df_ite,
            id_country=id_country, country=country, iteration_date=iteration_date, 
            update_missing=update_missing, predict_missing=predict_missing,
            betting_strat=betting_strat,
            export=export
            )