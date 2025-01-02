import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
import datetime
from p4_modeling import select_model_for_prod, betting_strategy, assess_models_in_prod, asses_model


def main(
        id_country, 
        country, 
        iteration_date,
        first_cutoff: float = 0.2, # 0.02 creo que es muy grande. Salvo que la diferencia de ROI sea pequeña como en SPA...
        assess: bool = True, 
        avoid_assess: bool = False,
        select_best_model: bool = True, 
        strategy: str = 'general'
        ):
    """
    Assess model in prod + Seleccion del modelo + Estrategia de apuesta

    # Parameters
        first_cutoff: Porcentaje de modelos para los cuales hacer el assess y la seleccion. Cuanto menor es, menor tiempo tarda. (float)
        assess: True para recolectar las predicciones de los modelos en los partidos missing y concatenarlas a las de test. (bool)
        select_best_model: True para hacer la seleccion del modelo a usar en produccion. (bool)
        strategy: Estrategia de apuesta para determinar hiperparametros a probar. (str)

    # Return
        Modelo a usar en produccion con su estrategia de apuesta optima (teniendo en cuenta test + missing). 
    """
    d_paths = initialize_directories(country, iteration_date)
    
    # Defino parametros --> Futuros argumentos
    second_cutoff = 0.1

   # Levanto df_iteration
    df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")

    # Defino roi_weight
    roi_weight = asses_model.asignar_roi_weight(df_ite)  # Segun correlacion entre ROI y Expected ROI
    # roi_weight = 0.75 if with_assess else None

    # (0) Filtro df_ite seleccionado top x% de registros. --> Hacerlo segun el roi_por_partido maximo menos un 20%... para evaluar los modelos realmente buenos...
    df_ite = df_ite.sort_values(by='roi_por_partido', ascending=False) ## Ordenar los registros por 'metric' en orden descendente
    cutoff = int(len(df_ite) * first_cutoff)  # Calcular el 20% superior
    df_ite_filt = df_ite.iloc[:cutoff] ## Seleccionar el 20% de los registros con los valores más altos de 'metric'
    print(df_ite_filt)

    # (1) Actualizar df_prediccion test con missing? --> Funcion ok incluso cuando no hay partidos missing. Chequeado.
    if assess:
        logger.warning("Estas por actualizar el df_iteration con los ultimos partidos missing...")

        if avoid_assess:
            df_ite_filt = pd.read_excel(f'{d_paths["path_assess"]}/df_iteration.xlsx')
        else:
            # Evaluo modelos en test y missing
            df_ite_filt = assess_models_in_prod.update_test_with_missing(df_ite=df_ite_filt, id_country=id_country, country=country, iteration_date=iteration_date, path_save=d_paths['path_assess'])

    logger.info(df_ite_filt)
    
    # (2) Seleccion del modelo
    if select_best_model:
        # Creo objeto de clase select_best_model
        sbm = select_model_for_prod.SelectBestModel(id_country=id_country, iteration_date=iteration_date, d_paths=d_paths)

        # Selecciono el mejor modelo
        row = sbm.main(df_ite_filt, roi_weight=roi_weight, perc_cutoff=second_cutoff)

    else:
        # Levanto el df del ultimo paso de la seleccion y obtengo el mejor modelo
        df = pd.read_excel(f'{bs.BASE_PATH_sbm}/df_p4.xlsx', index_col=0)
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]:.1f}")

    # (3) Determinar estrategia de apuesta optima para el modelo seleccionado
    bs = betting_strategy.BettingStrategy(country, iteration_date, d_paths=d_paths)
    bs.define_model_betting_strategy(row=row, strategy=strategy, roi_weight=roi_weight, with_assess=assess, by_result=True)

    
def initialize_directories(country, iteration_date):
    """
    Guardar seleccion de modelo vieja en carpeta
    """
    fecha_hoy = datetime.datetime.now().date()
    
    base_path = f"data/{country}/p4_modeling/{iteration_date}"
    base_path_sbm = f"{base_path}/best_model"
    directories.make_directories(l_directorios=[base_path_sbm]) # Por si nunca corri el main_select para el pais.

    d_paths = {
        'base_path': base_path,
        'base_path_sbm': base_path_sbm,
        'path_old': f'{base_path_sbm}/old/{fecha_hoy}',
        'path_assess': f'{base_path_sbm}/1_assess/',
        'path_select': f'{base_path_sbm}/2_select_model/',
    }

    # Mover anterior seleccion y assess a old..
    directories.mover_archivo(origen=d_paths['path_assess'], destino=d_paths['path_old'])
    directories.mover_archivo(origen=d_paths['path_select'], destino=d_paths['path_old'])

    # Creo directorios para nuevo assess y seleccion
    directories.make_directories(l_directorios=[d_paths['path_assess'], d_paths['path_select']])
    return d_paths


if __name__ == "__main__":
    # Defino parametros
    id_country = 77

    # Defino hiperparametros
    asssess_models_in_prod = False
    avoid_assess = True
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

    main(id_country=id_country, country=country, iteration_date=iteration_date, assess=asssess_models_in_prod, select_best_model=select_best_model)