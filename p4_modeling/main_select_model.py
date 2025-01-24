import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
import datetime
from p4_modeling import select_model_for_prod, betting_strategy, assess_models_in_prod, asses_model
from p6_deployment import main_next_matches

def initialize_directories(country, iteration_date, predict_missing):
    """
    Guardar seleccion de modelo vieja en carpeta
    """
    fecha_hoy = datetime.datetime.now().date()
    
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

    if predict_missing:
        create_and_move_directories(d_paths=d_paths)
    return d_paths

def create_and_move_directories(d_paths):
    """
    Guardar seleccion de modelo vieja en carpeta

    Mejoras: 
        - Evitar mover old si uso assess_already_extracted?
    """
    # Mover anterior seleccion y assess a old..
    import os
    if os.path.exists(d_paths['base_path_sbm']):
        directories.make_directories(l_directorios=[d_paths['path_old']]) # Por si nunca corri el main_select para el pais.
        directories.mover_archivo(origen=d_paths['base_path_sbm'], destino=d_paths['path_old'])

    # Creo directorios para nuevo assess y seleccion
    directories.make_directories(l_directorios=[d_paths['path_assess'], d_paths['path_select'], d_paths['path_bet_strategy'], d_paths['path_assess_dep']])

def read_predicciones(n_model, model_name, assess, d_paths):
    """
    Aqui deberia ser capaz de levantar las predicciones sobre los missing tambien y evaluar todo junto (test + missing).             
    """
    filename = f'{n_model}__{model_name}_predicciones.xlsx'
    path_pred = f"{d_paths['path_assess']}/{filename}" if assess else f"{d_paths['base_path']}/models/{filename}"
    df_pred = pd.read_excel(path_pred, index_col=0)
    
    logger.info(f'n_model: {n_model} model_name: {model_name}')
    logger.info(df_pred.shape)
    return df_pred

def main(
        id_country, 
        country, 
        iteration_date,
        assess: bool = True,                        # Assess
        update_missing: bool = True,               # Assess
        predict_missing: bool = False,     # Assess
        strategy: str = 'kelly',                  # Betting Strategy
        ):
    """
    Assess model in prod + Seleccion del modelo + Estrategia de apuesta

    # Parameters
        prop_to_max: Porcentaje de la metrica maxima para filtrar modelos. Cuanto mayor es, menos modelos. (float)
        assess: True para recolectar las predicciones de los modelos en los partidos missing y concatenarlas a las de test. (bool)
        assess_already_extracted: Si queres evitar la recoleccion y prediccion en partidos missing y usar un test y assess ya actualizado.
        extract_missing: Para extraer missing y tenerlos actualizados antes de predecir.
        select_best_model: True para hacer la seleccion del modelo a usar en produccion. (bool)
        strategy: Estrategia de apuesta para determinar hiperparametros a probar. (str)

    # Return
        Modelo a usar en produccion con su estrategia de apuesta optima (teniendo en cuenta test + missing). 

    Mejoras:
        - Posibilidad de hacer filtrado de modelos antes de determinar el roi weight? --> Eliminaria modelos outlier o chotos y calcularia una correlacion mas precisa?
    """
    # Creo objeto de clase select_best_model
    d_paths = initialize_directories(country, iteration_date, predict_missing)

   # Levanto df_iteration
    df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")

    # (1) SELECCION DE MODELOS CANDIDATOS
    sbm = select_model_for_prod.SelectBestModel(id_country=id_country, path_save=d_paths['path_select'])

    # 1.1. Descarte por METRIC
    df_ite_filt = sbm.filter_models_by_metric(df_ite, metric_col='roi_por_partido', prop_to_max=0.2, perc_cutoff=20, n_models_max=30)
    logger.warning(f'Shape: {df_ite.shape} --> {df_ite_filt.shape}')

    rows = []
    d_rows = {}

    # (2) ESTRATRAGIA DE APUESTA
    # Por modelo
    for idx, row in df_ite_filt.iterrows():

        n_model = row['n_iteration']
        model_name = row['model_name']
        logger.info(f'{n_model} {model_name}')

        # Levanto df_predicciones
        if assess:
            df_pred = assess_models_in_prod.get_model_predictions_with_missing(n_model=n_model, model_name=model_name, id_country=id_country, country=country, iteration_date=iteration_date, path_save=d_paths['path_assess'])
        else:
            df_pred = read_predicciones(n_model, model_name, assess, d_paths)

        # Defino estrategia
        bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths, verbose=0)
        d_params = bs.define_hiperparameters(strategy=strategy)
        df, df_pred_with_stra = bs.define_model_betting_strategy_by_result(df_pred, d_params=d_params)

        # Guardo metrics
        roi_sin_ea = df_pred_with_stra['G_P_sin_ea'].sum()
        roi_con_ea = df['roi'].sum()
        gp_home = df.loc[1 ,'%_G/P']
        gp_draw = df.loc[0 ,'%_G/P']
        gp_away = df.loc[2 ,'%_G/P']

        multiplicador = (roi_con_ea - roi_sin_ea) / abs(roi_sin_ea)
        new_row = {
            'n_model': n_model, 'model_name': model_name, 'test_accuracy': row['test_accuracy'], 'recall': row['recall'], 'f1_score': row['f1_score'], 
            'gp_home': gp_home, 'gp_draw': gp_draw, 'gp_away': gp_away,
            'ROI_sin_ea': roi_sin_ea, 'ROI_con_ea': roi_con_ea, 'x ea': multiplicador
            }
        rows.append(new_row)
        d_rows[n_model] = [df, df_pred_with_stra]

        # Exporto datos (x seg)
        # df.to_excel(f'{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx', index=True)
        # df_pred_with_stra.to_excel(f'{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx')
        df_ite_bs = pd.DataFrame(data=rows)
        df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

    # (3) SELECCION DEL MODELO (el que maximiza el ROI con ea)
    ## Determino componentes de metrica combinada y pesos 
    l_pos_metrics = ['ROI_con_ea', 'test_accuracy', 'recall', 'f1_score', 'gp_home', 'gp_draw', 'gp_away'] # 'gp_total' no pues es = ROI.  # si usas cv_acc ojo que en el recalculo de emtricas por assess deberia ser "cv_accuracy_train"
    l_metrics = asses_model.select_metrics(df_ite_bs, col_corr="ROI_con_ea", l_metrics=l_pos_metrics, n_metrics=2)
    # l_weights = asses_model.define_weights(df_ite_bs, col_corr="ROI_con_ea", l_metrics=l_metrics) --> Si lo calculas con el ROI_con_ea se genera bias cdo los modelos que ganan tienen metricas malas por fuera del ROI_con_ea
    l_weights = [0.5, 0.5]

    ## Calculo metrica combinada
    metric_col = 'metric'
    df_ite_bs = asses_model.calculate_combined_metric(df_ite_bs, l_metrics=l_metrics, l_weights=l_weights)
    df_ite_bs = df_ite_bs.sort_values(by=metric_col, ascending=False)  # Ordenar los registros por 'metric' en orden descendente

    ## Selecciono el modelo que maximiza la metrica combinada
    idx_max = df_ite_bs[metric_col].idxmax()
    n_model = df_ite_bs.loc[idx_max, 'n_model']
    model_name = df_ite_bs.loc[idx_max, 'model_name']
    logger.critical(f"Modelo seleccionado: {n_model} {model_name}")
    
    # Exporto datos
    df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)
    df, df_pred_with_stra = d_rows[n_model]
    df.to_excel(f'{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx', index=True)
    df_pred_with_stra.to_excel(f'{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx')

if __name__ == "__main__":
    # Defino parametros
    id_country = 148
    
    # Defino hiperparametros
    assess = False
    # update_missing = False  # Extract + Prepare
    # predict_missing = True

    # Defino variables
    d_countries_old = {
        48: ["england", '2025-01-07'],
        55: ["france", '2025-01-08'], 
        59: ["germany", '2025-01-08'], 
        77: ["italy", '2025-01-06'],
        148: ["spain", '2025-01-07'], 
        }
    
    d_countries = {
        6: ["argentina", '2024-12-05'], 
        48: ["england", '2025-01-22'],
        55: ["france", '2025-01-22'], 
        59: ["germany", '2025-01-23'], 
        77: ["italy", '2025-01-20'],
        148: ["spain", '2025-01-20'], 
        167: ["usa", '2024-12-05']
        }

    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]
    
    main(
        id_country=id_country, country=country, iteration_date=iteration_date, 
        assess=assess, #update_missing=update_missing, predict_missing=predict_missing
        )