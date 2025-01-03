import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
import datetime
from p4_modeling import select_model_for_prod, betting_strategy, assess_models_in_prod, asses_model
from p6_deployment import main_next_matches

def main(
        id_country, 
        country, 
        iteration_date,
        prop_to_max: float = 0.7,                   # Filtrado inicial
        assess: bool = True,                        # Assess
        extract_missing: bool = True,               # Assess
        assess_already_extracted: bool = False,     # Assess
        select_best_model: bool = True,             # Select model
        strategy: str = 'general',                  # Betting Strategy
        by_result: bool = True                      # Betting Strategy
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
    """
    d_paths = initialize_directories(country, iteration_date)

   # Levanto df_iteration
    df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")

    # Defino roi_weight
    roi_weight = asses_model.asignar_roi_weight(df_ite)  # Segun correlacion entre ROI y Expected ROI
   
   # (0) DESCARTE POR ROI (pues el assess consume muchisimo tiempo, hacer solo a unos pocos modelos)
    # Calculo metrica combinada entre ROI y Expected ROI
    name_extension = '_sin_ea_test'
    metric_col_test = f'metric{name_extension}'
    df_ite = asses_model.calculate_combined_metric(df_ite, roi_weight=roi_weight, name_extension=name_extension)
    df_ite = df_ite.sort_values(by=metric_col_test, ascending=False)  # Ordenar los registros por 'metric' en orden descendente
    
    # Filtro
    df_ite_filt = filter_models_by_roi(df_ite, method='prop_to_max', prop_to_max=prop_to_max, metric_col=metric_col_test)
    df_ite_filt.to_excel(f'{d_paths['base_path_sbm']}/0_df_filt.xlsx', index=False)
    print(df_ite_filt.shape)

    # (1) ASSESS: Actualizar df_prediccion test con missing. --> Funcion ok incluso cuando no hay partidos missing. Chequeado.
    if assess:
        logger.warning("Estas por actualizar el df_iteration con los ultimos partidos missing...")

        if not assess_already_extracted:
            # Evaluo modelos en test y missing

            # Extraer missing
            if extract_missing:
                logger.warning(f"Se definió extract_missing={extract_missing}, por lo que, se está extrayendo los ultimos partidos missing...")
                # Usar mnm.py con predict_missing=True y data_unders=False.
                d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True}
                main_next_matches.main(d_run, id_country, export=d_run['export'], verbose=-1) 

            # Concatenar df_predicciones missing a test para cada modelo
            df_ite_filt = assess_models_in_prod.update_predicciones_test_with_missing(
                df_ite=df_ite_filt, id_country=id_country, country=country, iteration_date=iteration_date, path_save=d_paths['path_assess'])
        # Usar test + missing ya actualizado
        else:
            df_ite_filt = pd.read_excel(f'{d_paths["path_assess"]}/df_iteration.xlsx')

        # Recalculo metrica con assess
        name_extension = '_sin_ea'
        metric_col_assess = f'metric{name_extension}'
        df_ite_filt = asses_model.calculate_combined_metric(df_ite_filt, roi_weight=roi_weight, name_extension=name_extension)
        print(df_ite_filt.shape)

    # (2) Seleccion del modelo
    if select_best_model:
        # Creo objeto de clase select_best_model
        sbm = select_model_for_prod.SelectBestModel(id_country=id_country, iteration_date=iteration_date, path_save=d_paths['path_select'])

        # 1.1. DESCARTE POR DISTRIBUCION
        df_ite = sbm.filter_models_by_distribution(df_ite_filt)

        # 1.2. DESCARTE POR RELLENO DE NAN EN TEST
        if '%_gp_filled' in df_ite.columns and 'average_col_filled' in df_ite.columns:
            df_ite = sbm.filter_models_by_fill_nan(df_ite)
        else:
            logger.warning("Evito eliminacion de modelos por relleno de nan values en test puesto que no le medí lo cuando entrené")

        # 1.3: Seleccionar el modelo que maximiza ROI y expected ROI (sin estrategia)
        metric_col = metric_col_assess if assess else metric_col_test
        row = sbm.select_model(df_ite, metric_col=metric_col)

    else:
        # Levanto el df del ultimo paso de la seleccion y obtengo el mejor modelo
        df = pd.read_excel(f'{d_paths['path_select']}/df_selected_model.xlsx', index_col=0)
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]:.1f}")

    # Levanto df_predicciones --> Aqui deberia ser capaz de levantar las predicciones sobre los missing tambien y evaluar todo junto (test + missing).             # Falta levantar las predicciones de los missing y concatenerlas (si hubiera) --> no haria falta el assess_models_in_prod.py????
    n_model, model_name = row.index[0], row['model_name'].values[0]
    filename = f'{n_model}__{model_name}_predicciones.xlsx'
    path_pred = f"{d_paths['path_assess']}/{filename}" if assess else f"{d_paths['base_path']}/models/{filename}"
    df_pred = pd.read_excel(path_pred, index_col=0)
    
    logger.info(f'n_model: {n_model} model_name: {model_name}')
    logger.info(df_pred.shape)

    # (4) Determinar estrategia de apuesta optima para el modelo seleccionado        
    bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths, verbose=0)
    df, df_pred_with_stra = bs.define_model_betting_strategy(df_pred, strategy=strategy, roi_weight=roi_weight, by_result=by_result)
    
    df.to_excel(f'{d_paths['path_bet_strategy']}/df_strategy_{n_model}_{model_name}.xlsx', index=True)
    df_pred_with_stra.to_excel(f'{d_paths['path_bet_strategy']}/predicciones_{n_model}_{model_name}.xlsx')


def initialize_directories(country, iteration_date):
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
        'path_assess': f'{base_path_sbm}/1_assess/',
        'path_select': f'{base_path_sbm}/2_select_model/',
        'path_bet_strategy': f'{base_path_sbm}/3_bet_strategy/',
    }

    # Mover anterior seleccion y assess a old..
    import os
    if os.path.exists(d_paths['base_path_sbm']):
        directories.make_directories(l_directorios=[d_paths['path_old']]) # Por si nunca corri el main_select para el pais.
        directories.mover_archivo(origen=d_paths['base_path_sbm'], destino=d_paths['path_old'])

    # Creo directorios para nuevo assess y seleccion
    directories.make_directories(l_directorios=[d_paths['path_assess'], d_paths['path_select'], d_paths['path_bet_strategy']])
    return d_paths

def filter_models_by_roi(df, method: str = 'prop_to_max', prop_to_max: float = 0.7, perc_cutoff: float = 20, metric_col: str = 'metric_sin_ea', verbose: int = 2):
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
    logger.info("Paso 1: Descartando modelos con bajo ROI en df_test")

    if method == 'prop_to_max':
        max_roi = df[metric_col].max()
        roi_cut = max_roi * prop_to_max

    elif method == 'percentile':
        # Seleccionar el 20% de los registros con los valores más altos de 'metric'
        roi_cut = np.percentile(df[metric_col], (100-perc_cutoff))

    else:
        logger.error(f"No existe el metodo {method} para filtrar por ROI los modelos.")
        raise ValueError

    # Filtro modelos segun roi to cut
    df_filt = df[df[metric_col] >= roi_cut]

    if verbose >= 2:
        val1 = df[metric_col].max()
        val2 = np.percentile(df[metric_col], (100-perc_cutoff))
        logger.info(f"\n(1) Metric Max: {val1} --> ROI cut: {val1 * prop_to_max} \n(2) Roi cut percentile {perc_cutoff}: {val2}")

    if verbose >= 1:
        logger.info(df_filt.head())
        logger.warning(f"Descarte por ROI: {len(df)} --> {len(df_filt)}")

    return df_filt

if __name__ == "__main__":
    # Defino parametros
    id_country = 148

    # Defino hiperparametros
    asssess_models_in_prod = False
    select_best_model = True

    # Defino variables
    d_countries = {
        6: ["argentina", '2024-12-05'], 
        48: ["england", '2024-12-23'],
        # 48: ["england", '2025-01-02'],
        55: ["france", '2024-12-26'], 
        59: ["germany", '2024-12-26'], 
        77: ["italy", '2024-12-23'],
        # 77: ["italy", '2025-01-01'],
        # 77: ["italy", '2025-01-02'],
        148: ["spain", '2024-12-25'], 
        167: ["usa", '2024-12-05']
        }
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]

    main(id_country=id_country, country=country, iteration_date=iteration_date, 
         assess=asssess_models_in_prod, extract_missing=False, assess_already_extracted=False,
         select_best_model=select_best_model)