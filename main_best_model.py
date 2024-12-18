# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
from p3_data_preparation.select_data import determine_country_competitions
import utils.directories as directories
import pickle
from itertools import product
from main import DataPreparation, Modeling
import joblib
import time
# from itertools import product
import re
from p3_data_preparation.clean_data import fillna_with_mean_in_last_matches
import numpy as np
from p4_modeling.select_model_for_prod import SelectBestModel


def comprehensive_search(
    country, 
    date, 
    d_params, 
    l_modelos, 
    retrain: bool = True, 
    continue_old_train: bool = False, 
    verbose: int = 0, 
    export: bool = True
):
    """
    Busca los hiperparámetros óptimos en las etapas de DataPreparation y Modeling de main.py.
    
    Parámetros:
    ----------
    country : str
        Nombre del país sobre el cual se realizará la búsqueda de hiperparámetros.
        
    ruta_base_mod : str
        Ruta base donde se guardarán los datos generados durante la ejecución de la función.
        
    d_params : dict
        Diccionario con los hiperparámetros a probar. Estos hiperparámetros se aplican tanto
        a la preparación de datos como al modelado.
        
    l_modelos : list
        Lista de modelos que se entrenarán. Puede incluir instancias como `SVC`, 
        `LogisticRegression`, etc.
        
    retrain : bool, opcional (por defecto True)
        Si es True, los modelos se entrenarán utilizando tanto los datos extraídos como 
        los datos missing con información de los últimos partidos jugados.
        
    continue_old_train : bool, opcional (por defecto False)
        Si es True, permite continuar un entrenamiento que se interrumpió previamente. 
        Retoma desde la última construcción completada.
        
    verbose : int, opcional (por defecto 0)
        Nivel de detalle de los mensajes impresos en la consola.
        - 0: Sin mensajes.
        - 1: Mensajes básicos.
        - 2 o mayor: Mensajes detallados para depuración.
        
    export : bool, opcional (por defecto True)
        Si es True, exporta los DataFrames generados durante la ejecución de la función
        a la ubicación especificada en `ruta_base_mod`.

    Return:
    ----------
    df_iteration_comp: DataFrame
        Una fila por modelos entrenado detallando los hiperparametros usados al entrenar y su 
        evaluacion en el testeo.
    """
    # Definicion de variables
    one_time, avoid_construct = True, False
    rows_to_features_min, min_row_test = 10, 30
    dp, mo = DataPreparation(country), Modeling(country) # Creo objetos de clases DataPreparation y Modeling

    # Imprimo largo de iteraciones
    n_iter = define_n_iterations(d_params)
    if verbose >= 0:
        logger.info(f"Numero de iteraciones totales: {n_iter}")
        start_train = time.time()  # segundos desde el 1 de enero de 1970 UTC
    
    # Defino rutas segun country y date
    BASE_DIR = f"./data/{country}/p4_modeling/{date}"
    ruta_base_du, ruta_base_dp, ruta_base_modelos = f"{BASE_DIR}/p2_data_understanding", f"{BASE_DIR}/p3_data_preparation", f"{BASE_DIR}/models" 
    # directories.make_directories(l_directorios=[ruta_base_du, ruta_base_dp, ruta_base_modelos])
    
    # Defino otras variables
    if continue_old_train:
        
        # Defino dataframes a devolver como copia de los que ya habia generado
        df_ite_old = pd.read_excel(f'{BASE_DIR}/df_iteration_train.xlsx')
        df_ite_test_old = pd.read_excel(f'{BASE_DIR}/df_iteration_test.xlsx')
        df_iteration = df_ite_old.copy()
        df_ite_test = df_ite_test_old.copy()

        # Determino hiperparametros de construccion de la ultima iteracion entrenada
        row = df_ite_old[df_ite_old['n_iteration'] == df_ite_old['n_iteration'].max()]
        idx = row.index[0]
        n_last_model, lm_n_dias_ult_part, lm_n_anios_hist, lm_segun_localia, lm_dif_con_against = row.loc[idx, 'n_iteration'], eval(row.loc[idx, 'n_dias_ult_part']), row.loc[idx, 'n_anios_hist'], row.loc[idx, 'segun_localia'], row.loc[idx, 'dif_con_against']
        logger.warning("Se esta continuando el entrenamiento anterior dado que continue_old_train=True.")

        # Determino numero de iteracion desde la que se sigue
        cont_iter = n_last_model
        # n_comb_construct = d_params['construct']
        # frecuencia_construct = n_iter / n_comb_construct
    
    else:
        cont_iter = 0
        df_iteration, df_ite_test = pd.DataFrame(), pd.DataFrame()
        directories.make_directories(l_directorios=[ruta_base_du, ruta_base_dp, ruta_base_modelos])

    # Construct_data
    for i, param_values_2 in enumerate(product(*d_params['construct'].values()), start=1):

        # Asigno valor a cada hiperpametro
        n_dias_ult_part, n_years_h2h, segun_localia, dif_con_against = param_values_2  # n_dias_ult_part, n_years_h2h, segun_localia, dif_con_against = param_values_2[0], param_values_2[1], param_values_2[2], param_values_2[3]
        if verbose >= 0:
            logger.info(f" Iteracion Construct Nº {i} ".center(120, "#"))
            print(f'Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h}; segun_localia: {segun_localia} ; dif_con_against: {dif_con_against}')

        if continue_old_train and one_time:
            # Una vez que alcance la construccion de last model
            if (n_dias_ult_part == lm_n_dias_ult_part) and (n_years_h2h==lm_n_anios_hist) and (segun_localia==lm_segun_localia) and (dif_con_against==lm_dif_con_against):
                one_time = False
                avoid_construct = False
            else:
                logger.warning("Se evito construccion.")
                avoid_construct = True

        if not avoid_construct:

            # Construyo datos
            path_1 = f"{n_dias_ult_part}_{n_years_h2h}_{segun_localia}_{dif_con_against}"
            path_construct = f'{ruta_base_dp}/df_constructed_{path_1}.xlsx'
            try:
                df_constructed = pd.read_excel(path_construct, index_col=0)
                # print("\n DF_CONSTRUCTED \n", df_constructed.head(2))
            
            except FileNotFoundError:
                # Levanto dataset formateado e integrado (estos no cambian entre iteraciones)
                if retrain:
                    df_integrated = pd.read_excel(f'./data/{country}/p6_deployment/missing/old_updated/df_integrated.xlsx', index_col=0)
                    logger.warning(f"Se levanto el df_integrated con los missing. Shape: {df_integrated.shape}")

                    directories.copy_directory(origen=f'./data/{country}/p6_deployment/missing/old_updated', destino=ruta_base_du)
                    # df_integrated.to_excel(f'{ruta_base_dp}/df_integrated.xlsx') 

                else:
                    df_integrated = pd.read_excel(f'./data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)
                    
                df_constructed = dp.construct_data(df_integrated, l_days=n_dias_ult_part, n_years_h2h=n_years_h2h, segun_localia=segun_localia, dif_con_against=dif_con_against, export=False)
                if export:
                    df_constructed.to_excel(path_construct, index=True)

            # Etiqueto df_constructed
            df_cons_etiquetado, df_etiquetas = dp.tag_string_data_to_integer(df_constructed, export=False)
            path_etiqueta = f'{ruta_base_dp}/df_etiquetas_{path_1}.xlsx'
            if export:
                df_etiquetas.to_excel(path_etiqueta, index=True)

            # Clean data 2 + Treat nan
            for zz, param_values_00 in enumerate(product(*d_params['clean_data_2'].values()), start=1):

                comp_to_select, n_years_to_select, fill_na =  param_values_00[0], param_values_00[1], param_values_00[2]

                path_2 = f'{n_years_to_select}_{comp_to_select}'
                path_clean_data = f'{ruta_base_dp}/df_clean_data_2_{path_1}_{path_2}_{fill_na}.xlsx'
                if verbose >= 0:
                    logger.info(f" Iteracion clean_data 2 Nº {i}.{zz} ".center(120, "#"))
                    print(f"Hiper clean_data_2 --> n_years_to_select: {n_years_to_select} ; comp_to_select: {comp_to_select} ; fill_na: {fill_na}")            

                df_cons_clean, scaler, columns_used = dp.clean_data_2(df=df_cons_etiquetado, n_years_to_select=n_years_to_select, competencies_to_select=comp_to_select, 
                                                                        fill_na=fill_na, export=False)
                joblib.dump((scaler, columns_used), f'{ruta_base_dp}/scaler_model_{path_1}_{path_2}.pkl')

                if verbose >= 2:
                    df_cons_clean.to_excel(path_clean_data, index=True)


                # Select data
                for j, param_values_4 in enumerate(product(*d_params['select'].values()), start=1):

                    # Asigno valor a cada hiperpametro
                    thr_corr, thr_fs = param_values_4[0], param_values_4[1]
                    path_3 = f'{thr_corr}_{thr_fs}'
                    if verbose >= 0:
                        logger.info(f" Iteracion Select Nº {i}.{zz}.{j} ".center(120, "#"))
                        print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")            

                    # Selecciono datos
                    path_select = f'{ruta_base_dp}/df_selected_{path_1}_{path_2}_{path_3}.xlsx'
                    try:
                        df_sel = pd.read_excel(path_select, index_col=0)
                    except FileNotFoundError:
                        df_sel = dp.select_data(df_cons_clean, thr_corr=thr_corr, thr_fs=thr_fs, export=False)

                        if verbose >= 2:
                            df_sel.to_excel(path_select, index=True)
            
                    # Modeling
                    for h, param_values_5 in enumerate(product(*d_params['modeling'].values()), start=1):
                        
                        # Asigno valor a cada hiperparametro
                        val_size, n_reg_test, bal_type, k = param_values_5[0], param_values_5[1], param_values_5[2], param_values_5[3]
                        if verbose >= 0:
                            cont_iter += 1
                            logger.info(f" Iteracion Modeling Nº {i}.{zz}.{j}.{h} ".center(120, "#"))
                            print(f'\n - Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h} ; segun_localia: {segun_localia} \n - Hiper clean_data_2 n_years_to_sel: {n_years_to_select} comp_to_select: {comp_to_select} \n- Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs} \n - Hiper treat_nan --> {fill_na} \n - Hiper modeling --> val_size: {val_size} ; n_reg_test: {n_reg_test}; bal_type: {bal_type} ; k: {k}')
                            logger.critical(f" Iteracion Nº {cont_iter} de {n_iter} ({cont_iter*100/n_iter:.0f}%)")

                        # Generar el diseño de la prueba
                        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df_sel, bal_type=bal_type, val_size=val_size, n_reg_test=n_reg_test, retrain=retrain, export=False)
                        
                        rows_test = len(X_test)
                        rows_to_features = len(X_train) / len(X_train.columns)  # Idealmente mayor a 10. En caso de redes neuronales entre 30 y 100 veces mas.
                        if verbose >= 0:
                            logger.info(f"Rows X_test: {rows_test}")
                            logger.info(f"Relacion rows to features: {rows_to_features:.0f}")

                        # Si hay suficientes datos
                        if (rows_test >= min_row_test) and (rows_to_features >= rows_to_features_min):
                            
                            df_metrics = mo.train_and_assess_models(l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, ruta_base_modelos, cont_iter, retrain=retrain, export=False)

                            # Guardo datos en dataframe
                            row_data = {'n_iteration': cont_iter, 
                                        'n_dias_ult_part': n_dias_ult_part, 'n_anios_hist': n_years_h2h, 'segun_localia': segun_localia, 'dif_con_against': dif_con_against,
                                        'thr_corr': thr_corr, 'thr_fs': thr_fs,
                                        'n_years_to_select': n_years_to_select, 'comp_to_select': comp_to_select,
                                        'fill_na': fill_na, 'bal_type': bal_type,
                                        'val_size': val_size, 'n_reg_test': n_reg_test, 'X_train': X_train.shape,
                                        'X_val': X_val.shape, 'X_test': X_test.shape, "X_columns": list(X_train.columns),
                                        'k': k}
                            
                            # Concateno y exporto datos
                            df_iteration = pd.concat([df_iteration, pd.DataFrame([row_data])], axis=0)
                            df_ite_test = pd.concat([df_ite_test, df_metrics], ignore_index=True) 
                            df_iteration_comp = pd.merge(df_iteration, df_ite_test, on='n_iteration', how='outer')     # Realizamos un merge por 'n_iteration' para combinar los DataFrames

                            if export:    
                                df_iteration.to_excel(f'{BASE_DIR}/df_iteration_train.xlsx', index=False)
                                df_ite_test.to_excel(f'{BASE_DIR}/df_iteration_test.xlsx', index=False)
                                df_iteration_comp.to_excel(f'{BASE_DIR}/df_iteration.xlsx', index=False)

                        else:
                            if rows_to_features >= rows_to_features_min:
                                logger.warning(f"EVITO TRAIN. Se evita entrenar modelo por pocas filas en X_test. {rows_test} menor a {min_row_test}. Probablemente los 'ultimos partidos' tienen mucho NaN y se estan eliminando en clean_data_2 (en la eliminacion de filas por mucho NaN) o treat_nan_values (si el fill_na=None no podes hacer nada..., en este caso el fill_na es {fill_na})")
                            else:
                                logger.warning(f"EVITO TRAIN. Se evita entrenar modelo por pocas filas respecto a columnas. {rows_to_features} menor a {rows_to_features_min} ")

                        if verbose >= 0:
                            current_train = time.time()
                            ritmo = cont_iter / ((current_train - start_train) / 60 / 60)  # ite / hora
                            ite_restantes = n_iter - cont_iter
                            horas_restantes = ite_restantes / ritmo
                            min_restantes = horas_restantes * 60
                            horas_train = n_iter / ritmo
                            logger.info(f"Dado el ritmo de {ritmo:.1f} ite/hora (ideal >60) y que quedan {ite_restantes} iteraciones, el tiempo estimado de finalizacion es en {min_restantes:.1f} minutos (={horas_restantes:.1f} horas)") # Proyeccion de cuantas horas quedan.
                            logger.info(f"Tiempo total de entrenamiento proyectado de {horas_train:.1f} horas.")
                            print()

    if verbose >= 0:
        end_train = time.time()
        logger.info(f"Tiempo total de entrenamiento: {(end_train - start_train) / 60:.1f} minutos")

    # Guardo datos de todas las iteraciones
    if export:
        df_iteration.to_excel(f'{BASE_DIR}/df_iteration_train.xlsx', index=False)
        df_ite_test.to_excel(f'{BASE_DIR}/df_iteration_test.xlsx', index=False)

    return df_iteration_comp

def define_n_iterations(d_params):
    """
    Calcula el numero de iteraciones y el tiempo estimado para terminar

    Parameteres
        d_params: Diccionario con hiperparametros a probar (dict)
    
    Return
        n_iter: Numero de iteraciones. (int)
    """
    # Cuidado con el nro de iteraciones sobretodo en select. nº comb = producto de posibles comb de cada hiper  EJ: {'thr_corr': [0.5, 0.6, 0.7], 'thr_fs': [0.25, 0.2, 0.15, 0.1, 0.05], 'thr_nan_col': [0.2, 0.5, None]} --> nºcomb = 3x5x3=45
    # n comb totales = 3 x 45 x 6 = 810 --> 90 iteraciones en 17 horas --> 5,35 iter/hora => 810 iteraciones = 151 horas  # n comb totales = 1 x 18 x 6 = 108 --> 90 iteraciones en 17 horas --> 5,35 iter/hora => 216 iteraciones = 151 horas
    n_iter = 1
    for task, d_params_task in d_params.items():
        for key in d_params_task.keys():
            n_iter *= len(d_params_task[key])
    return n_iter

def define_params_space(id_country, fast: bool = False):

    # Defino hiperparametros a probar
    d_comps = determine_country_competitions(id_country)
    # l_modelos = [LogisticRegression(), 'neural_network', SVC(), XGBClassifier()] # GradientBoostingClassifier(), MLPClassifier()]
    l_modelos = [LogisticRegression()]

    # 1728 iteraciones
    d_params = {
        'construct': {
            'n_dias_ult_part': [[60], [30, 180]], # [30, 180] 
            'n_years_h2h': [3],
            'segun_localia': [True, False],
            'dif_con_against': [False, True] 
        },
        'clean_data_2': {
            'competencies_to_select': [d_comps['comp_solo_liga'], d_comps['comp_sin_b'], d_comps['comp_sin_cups'], d_comps['all_comp']], # d_comps['comp_sin_cups'], d_comps['comp_sin_b'],
            'n_years_to_select': [2, 3, 5, 10], #, None --> no tiene sentido porque el fifa arranca en 2007 (hace 17 años). Tampoco tiene sentido usar 15 años si elimino los datos de antes de 2012
            'fill_na': [None, 'ml'],  # None se eliminan todos los ultimos partidos y el X_test queda vacio, por ende, no entrena.
        },
        'select': {
            'thr_corr': [0.7, 0.85, None],
            'thr_fs': [None, 0.25, 0.5, 0.75],
        },
        'modeling': {
            'val_size': [0.10],
            'n_reg_test': [100], 
            'bal_type': [None, 'under'], # None (ni con f1_score..)
            'k': [5] 
            # scoring : ['accuracy', 'f1_macro'] 
        }
    }

    if fast:
        l_modelos = [LogisticRegression()]  # Neural network se corta x memoria. SVC() nunca ganó.

        d_params = {  
            'construct': {
                'n_dias_ult_part': [[30, 180]], # [180], [90], [30], [60, 240] --> Perdió claramente en los nuevos entrenam.
                'n_years_h2h': [3],
                'segun_localia': [True, False],
                'dif_con_against': [False, True] 
            },
            'clean_data_2': {
                'competencies_to_select': [d_comps['comp_solo_liga'], d_comps['all_comp']],  # d_comps['comp_sin_cups'], d_comps['comp_sin_b'],
                'n_years_to_select': [2, 3, 5, 10], #, None --> no tiene sentido porque el fifa arranca en 2007 (hace 17 años). Tampoco tiene sentido usar 15 años si elimino los datos de antes de 2012
                'fill_na': [None, 'ml'], #   # None se eliminan todos los ultimos partidos y el X_test queda vacio, por ende, no entrena.
            },
            'select': {
                'thr_corr': [0.7, 0.85, None],
                'thr_fs': [None, 0.25, 0.5, 0.75],
            },
            'modeling': {
                'val_size': [0.1],
                'n_reg_test': [100], # 100 partidos son aprox los ultimos 3 meses.
                'bal_type': [None, 'under'], # (ni con f1_score..) # ,
                'k': [5] 
                # scoring : ['accuracy', 'f1_macro'] 
            }
        }
 
    logger.info(f"Parametros para entrenar: {d_params}")

    # Exportar un archivo .txt con los hiperparametros probados. --> Asi tengo que hiper probe en cada entrenamiento...
    # ...

    return d_params, l_modelos

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
        
    # Parametros de ejecucion
    id_country = 59
    only_select_best_model = False
    continue_old_train, date_old_train = False, '2024-11-27'

    d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Preparao datos, entreno modelos y evaluo en df_test
    if not only_select_best_model:

        # Determino date 
        date = date_old_train if continue_old_train else datetime.datetime.now().date() # datetime.datetime.now().date() 
        logger.info(f"Country: {country} Date: {date}")

        # Defino hiperparametros a probar
        d_params, l_modelos = define_params_space(id_country, fast=True)
        
        # Preparo y entreno modelos para todas las combinaciones de hiper posibles 
        df_iteration_comp = comprehensive_search(country, date, d_params, l_modelos, continue_old_train=continue_old_train)

    else:
        # Determino date 
        d_dates = {48: "2024-12-04", 55: "2024-11-14", 59: "2024-12-05", 77: "2024-11-10", 148: "2024-12-03"}
        date = d_dates[id_country]
        logger.info(f"Country: {country} Date: {date}")

        df_iteration_comp = pd.read_excel(f'./data/{country}/p4_modeling/{date}/df_iteration.xlsx') # index_col=0

    # Selecciono el mejor modelo --> Ver si funca...
    # sbm = SelectBestModel(id_country=id_country, country=country, iteration_date=date)
    # sbm.main(df_iteration_comp)