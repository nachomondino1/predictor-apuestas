import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
import datetime
from p4_modeling import asses_model
from p4_modeling.utils_select_model import assess_in_prod

def initialize_directories(country, iteration_date):
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
        'path_bet_strategy': f'{base_path_sbm}/3_bet_strategy',
    }

    # Mover toda la carpeta si assess y bet_strategy son True
    # if os.path.exists(base_path_sbm):
    #     directories.make_directories([d_paths['path_old']])
    #     directories.mover_archivo(base_path_sbm, d_paths['path_old'])
    
    # Crear directorios necesarios
    directories.make_directories([
        d_paths['path_select'], 
        d_paths['path_bet_strategy'], 
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
        country, 
        iteration_date,
        l_metrics: list, 
        l_weights: list, 
        assess: bool = False,
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
    d_paths = initialize_directories(country, iteration_date)
    metric = 'metric_test_assess'

    ## 1. Calculo metrica combinada
    df_ite_bs = asses_model.calculate_combined_metric(df_ite, l_metrics=l_metrics, l_weights=l_weights, metric_name=metric, penalize_std=False)

    ## 2. Ordeno por metrica combinada
    df_ite_bs = df_ite_bs.sort_values(by=metric, ascending=False)  # Ordenar los registros por 'metric' en orden descendente

    # Si assesss
    if assess:
        # Selecciono candidatos
        df_ite_bs = df_ite_bs.head(10)

        # Actualizo con assess
        logger.warning("Tengo en cuenta tanto 'test' como 'assess' para seleccionar modelo...")
        df_ite_bs = assess_in_prod.assess_models_in_prod(
            df_ite=df_ite_bs,
            id_country=id_country, 
            country=country, 
            iteration_date=iteration_date, 
            concat_with_test=True,
            export=False
        )       

        df_ite_bs = asses_model.calculate_combined_metric(df_ite_bs, l_metrics=l_metrics, l_weights=l_weights, metric_name=metric)
        df_ite_bs = df_ite_bs.sort_values(by=metric, ascending=False)  # Ordenar los registros por 'metric' en orden descendente

    # 3. Seleccion del modelo
    idx_max = df_ite_bs[metric].idxmax() # Pero ahora sin ea realmente. No uso kelly sino linear sin cuotas.
    col_name_1 = 'n_iteration' if 'n_iteration' in df_ite_bs.columns else 'n_model'
    col_name_2 = 'model_name' if 'model_name' in df_ite_bs.columns else 'model_name_x'
    n_model, model_name = df_ite_bs.loc[idx_max, col_name_1], df_ite_bs.loc[idx_max, col_name_2]
    logger.critical(f"Modelo seleccionado: {n_model} {model_name}") # n_model

    # Exporto datos
    if export:
        df_ite_bs.to_excel(f'{d_paths['path_bet_strategy']}/df_ite_bs.xlsx', index=False)

    return df_ite_bs

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    assess = True

    d_countries = {
        -1: ['all', '2025-04-22'],
        48: ["england", '2025-04-22'],
        55: ["france", '2025-04-23'], 
        59: ["germany", '2025-04-23'],
        77: ["italy", '2025-04-23'],
        148: ["spain", '2025-04-23']
        }
    
    # Defino metricas y pesos (Metricas comunes pero pesos ≠ por pais) 
    l_metrics = ['f1_score', 'expected_f1_score', 'error', 'expected_error'] 
    l_weights = [0.25, 0.25, 0.25, 0.25]

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]

        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_ite_test.xlsx")
        print(df_ite)

        # Para maximizar error en metrica
        df_ite.loc[df_ite['error'] > 0, 'error'] *= -1
        df_ite.loc[df_ite['expected_error'] > 0, 'expected_error'] *= -1 
        # df_ite['cv_cross_entropy_loss'] = df_ite['cv_cross_entropy_loss'] * (-1)

        main(
            df_ite=df_ite,
            country=country, iteration_date=iteration_date, 
            l_metrics=l_metrics, l_weights=l_weights,
            assess=assess,
            export=True
            )
        