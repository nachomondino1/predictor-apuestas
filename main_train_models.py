# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import pandas as pd
import numpy as np
import datetime
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
from p2_data_understanding.collect_initial_data import update_sofifa_data
from p3_data_preparation.select_data import select_league_matches
from p3_data_preparation import concat_mapeos
from p3_data_preparation.select_data import determine_country_competitions
from p6_deployment import main_next_matches
import utils.directories as directories
from itertools import product
from main import DataPreparation, Modeling
import joblib
import time


def comprehensive_search(
    country, 
    date, 
    d_params, 
    l_modelos, 
    data_unders: bool = True,
    data_prep_int: bool = True,
    update_sofifa: bool = True,
    retrain: bool = True, 
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

    Posibles mejoras:
        - Separar paths entre clean, construct y demas por dos underscore "__" en vez de uno solo "_"
        - Entrenar evitando integrate pero variando los l_models por ejemplo. Usa mismo: old_updated/ integrate_data/ clean_data/ y df_integrated que el entrenamiento actual...  pero tenes que guardar los missing en p2...
    """
    # Definicion de variables
    cont_iter = 0
    df_iteration, df_ite_test = pd.DataFrame(), pd.DataFrame()
    dp, mo = DataPreparation(id_country=id_country, country=country, date=date), Modeling(country, date=date) # Creo objetos de clases DataPreparation y Modeling

    # Imprimo largo de iteraciones
    n_iter = define_n_iterations(d_params)
    if verbose >= 0:
        logger.info(f"Numero de iteraciones totales: {n_iter}")
        start_train = time.time()  # segundos desde el 1 de enero de 1970 UTC
    
    # DIRECTORIOS
    # Defino rutas segun country y date
    BASE_DIR_du = f"./data/{country}/p2_data_understanding/old_updated/{date}"
    BASE_DIR_flashscore = f'data/{country}/p6_deployment/missing/old_updated'
    BASE_DIR_sofifa = f'data/{country}/p2_data_understanding/sofifa_update/{date}' if retrain else f'data/{country}/p2_data_understanding'
    BASE_DIR_dp = f"./data/{country}/p3_data_preparation/{date}"     # BASE_DIR_mod = f"./data/{country}/p4_modeling/{date}"
    BASE_DIR_mod = f"./data/{country}/p4_modeling/{date}"
    ruta_base_modelos = f"{BASE_DIR_mod}/models" 
    # save_old_train(l_directories=[f"./data/{country}/p2_data_understanding/old_updated", f"./data/{country}/p3_data_preparation", f"./data/{country}/p4_modeling"], base_path_old = f'./data/{country}/old') 
    directories.make_directories(l_directorios=[BASE_DIR_dp, BASE_DIR_mod, ruta_base_modelos])
 
    if data_unders:
        directories.make_directories(l_directorios=[BASE_DIR_du, BASE_DIR_sofifa])

        ####################################################################### DATA UNDERSTANDING ####################################################################### --> Si hubo missing, esta bueno correrlo...
        # Defino paths de donde levantar los datos
        df_match, df_match_player, df_match_odds = get_flashscore_data(BASE_DIR_flashscore)
        df_player_sofifa, df_player_fifa_sofifa = get_sofifa_data(country, update_sofifa=update_sofifa, BASE_DIR_sofifa=BASE_DIR_sofifa)

        # Exporto los datos para saber que datos use en el entrenamiento actual (no copio directorios porque me borra lo que ya hay en el directorio.)
        df_match.to_excel(f'{BASE_DIR_du}/df_match.xlsx', index=True)
        df_match_player.to_excel(f'{BASE_DIR_du}/df_match_player.xlsx', index=True)
        df_match_odds.to_excel(f'{BASE_DIR_du}/df_match_odds.xlsx', index=True)
        df_player_sofifa.to_excel(f'{BASE_DIR_du}/df_player_sofifa.xlsx', index=True)
        df_player_fifa_sofifa.to_excel(f'{BASE_DIR_du}/df_player_fifa_sofifa.xlsx', index=True)
    
    else:
        df_match = pd.read_excel(f'{BASE_DIR_du}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'{BASE_DIR_du}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{BASE_DIR_du}/df_match_odds.xlsx', index_col=0)
        df_player_sofifa = pd.read_excel(f'{BASE_DIR_du}/df_player_sofifa.xlsx', index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f'{BASE_DIR_du}/df_player_fifa_sofifa.xlsx', index_col=0)

    if data_prep_int:

        ####################################################################### DATA PREPARATION (hasta integrate) #######################################################################
        # Format data
        df_match, df_match_player, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_player_fifa_sofifa, reformat=True, export=True)
        
        # Clean data
        df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, export=True)

        # Integrate data (tengo que volver a integrar... para generar df_player de fs bien y tener los nuevos jugadores que surgen en missing y mapearlos..)
        # df_map, df_player_sofifa, df_player_fifa_sofifa = concat_mapeos.concat_integrate_data_by_country(l_countries=d_countries.values()) No se como lo implementaria...
        df_integrated = dp.integrate_data(df_match, df_match_player, df_player_sofifa=df_player_sofifa, df_player_fifa_sofifa=df_player_fifa_sofifa, export=True) 
    
        ####################################################################### DATA PREPARATION (MISSING) #######################################################################
        # ACTUALIZAR PREPARACION DE MISSING (con el ultimo mapeo y los ultimos datos de sofifa, es clave)
        ## Eliminar preparacion de missing actual
        l_dirs_to_remove = [
            f'data/{country}/p6_deployment/missing/data_preparation',
            f'data/{country}/p6_deployment/missing/old_updated/df_integrated.xlsx'
        ]    
        directories.remove_directories(directories=l_dirs_to_remove)

        ## Volver a preaparar missing
        d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True}
        main_next_matches.main(d_run, id_country, iteration_date=date, extract_missing=False, prepare_missing=True, export=d_run['export'])

    else:
        df_integrated = pd.read_excel(f'{BASE_DIR_dp}/df_integrated.xlsx', index_col=0)
        print(df_integrated)
    
    
    ####################################################################### DATA PREPARATION (desde construct) #######################################################################

    # Determino registros a usar en test_set
    n_reg_test = d_params['modeling'].pop('n_reg_test', None)     # Obtener el valor de 'n_reg_test' y eliminarlo del diccionario
    if n_reg_test is not None:
        n_reg_test = n_reg_test[0]  # Si es una lista, obtenemos el primer elemento
    logger.info(d_params['modeling'].values())
    logger.info(f"N_REG_TEST: {n_reg_test}")

    df_match_old = read_df_match(country=country, iteration_date=date, retrain=retrain)
    index_test_set = determine_rows_for_test_set(df_match=df_match_old, n_reg_test=n_reg_test)  # Usar n_reg_test.

     # Clean data 3
    for zz, param_values_00 in enumerate(product(*d_params['clean_data_3'].values()), start=1):

        comp_to_select =  param_values_00[0]

        path_clean = f'{comp_to_select}'
        path_clean_data = f'{BASE_DIR_dp}/clean_data_3/df_clean_data_3_{path_clean}.xlsx'
        if verbose >= 0:
            logger.info(f" Iteracion clean_data 3".center(120, "#"))
            print(f"Hiper clean_data_3 --> comp_to_select: {comp_to_select}")            

        df_int_clean = dp.clean_data_3(df=df_integrated, competencies_to_select=comp_to_select, export=True)
        if verbose >= 2:
            df_int_clean.to_excel(path_clean_data, index=True)

        # Construct_data
        for i, param_values_2 in enumerate(product(*d_params['construct'].values()), start=1):

            # Asigno valor a cada hiperpametro
            n_last_matches, n_dias_ult_part, n_years_h2h, segun_localia, dif_con_against = param_values_2  # n_dias_ult_part, n_years_h2h, segun_localia, dif_con_against = param_values_2[0], param_values_2[1], param_values_2[2], param_values_2[3]
            if verbose >= 0:
                logger.info(f" Iteracion Construct Nº {i} ".center(120, "#"))
                print(f'Hiper construct --> n_last_matches: {n_last_matches} ; n_dias_ult_part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h}; segun_localia: {segun_localia} ; dif_con_against: {dif_con_against}')

            # Construyo datos
            path_cons = f"{path_clean}_{n_last_matches}_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}_{dif_con_against}"
            path_construct = f'{BASE_DIR_dp}/construct_data/df_constructed_{path_cons}.xlsx'
            df_constructed = dp.construct_data(df_int_clean, n_last_matches=n_last_matches,l_days=n_dias_ult_part, n_years_h2h=n_years_h2h, segun_localia=segun_localia, dif_con_against=dif_con_against, export=True)
            if export:
                df_constructed.to_excel(path_construct, index=True)

            # Etiqueto df_constructed
            df_cons_etiquetado, df_etiquetas = dp.tag_string_data_to_integer(df_constructed, export=True)
            path_etiqueta = f'{BASE_DIR_dp}/tag/df_etiquetas_{path_cons}.xlsx'
            if export:
                df_etiquetas.to_excel(path_etiqueta, index=True)

            # Clean data 2 (Treat nan + Escalado)
            for zz, param_values_00 in enumerate(product(*d_params['clean_data_2'].values()), start=1):

                n_years_to_select, fill_na = param_values_00[0], param_values_00[1]
                path_clean_2 = f'{path_cons}_{n_years_to_select}_{fill_na}'
                path_clean_data = f'{BASE_DIR_dp}/clean_data_2/df_clean_data_2_{path_clean_2}.xlsx'
                if verbose >= 0:
                    logger.info(f" Iteracion clean_data 2 Nº {i}.{zz} ".center(120, "#"))
                    print(f"Hiper clean_data_2 --> n_years_to_select: {n_years_to_select} ; fill_na: {fill_na}")            

                df_cons_clean, scaler, columns_used = dp.clean_data_2(df=df_cons_etiquetado, n_years_to_select=n_years_to_select, fill_na=fill_na, index_test_set=index_test_set, export=True)
                joblib.dump((scaler, columns_used), f'{BASE_DIR_dp}/clean_data_2/scaler_model_{path_clean_2}.pkl')

                if verbose >= 2:
                    df_cons_clean.to_excel(path_clean_data, index=True)

                # Select data
                for j, param_values_4 in enumerate(product(*d_params['select'].values()), start=1):

                    # Asigno valor a cada hiperpametro
                    thr_corr, thr_fs = param_values_4[0], param_values_4[1]
                    path_sel = f'{path_clean_2}_{thr_corr}_{thr_fs}'
                    if verbose >= 0:
                        logger.info(f" Iteracion Select Nº {i}.{zz}.{j} ".center(120, "#"))
                        print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")            

                    # Selecciono datos
                    path_select = f'{BASE_DIR_dp}/select_data/df_selected_{path_sel}.xlsx'
                    try:
                        df_sel = pd.read_excel(path_select, index_col=0)
                    except FileNotFoundError:
                        df_sel = dp.select_data(df_cons_clean, thr_corr=thr_corr, thr_fs=thr_fs, export=True)

                        if verbose >= 2:
                            df_sel.to_excel(path_select, index=True)
            
                    ####################################################################### MODELING #######################################################################
                    for h, param_values_5 in enumerate(product(*d_params['modeling'].values()), start=1):
                        
                        # Asigno valor a cada hiperparametro
                        val_size, bal_type, k = param_values_5[0], param_values_5[1], param_values_5[2]
                        if verbose >= 0:
                            cont_iter += 1
                            logger.info(f" Iteracion Modeling Nº {i}.{zz}.{j}.{h} ".center(120, "#"))
                            print(f'\n - Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h} ; segun_localia: {segun_localia} \n - Hiper clean_data_2 n_years_to_sel: {n_years_to_select} comp_to_select: {comp_to_select} \n- Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs} \n - Hiper treat_nan --> {fill_na} \n - Hiper modeling --> val_size: {val_size} ; n_reg_test: {n_reg_test}; bal_type: {bal_type} ; k: {k}')
                            logger.critical(f" Iteracion Nº {cont_iter} de {n_iter} ({cont_iter*100/n_iter:.0f}%)")

                            # Generar el diseño de la prueba
                            X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df_sel, bal_type=bal_type, val_size=val_size, index_test_set=index_test_set, export=False)

                            # Entreno y evaluo modelos
                            df_metrics = mo.train_and_assess_models(X_val, y_val, X_train, y_train,  X_test, y_test, l_modelos, k, ruta_base_modelos, cont_iter, retrain=retrain)

                        if len(df_metrics) > 0:
                            # Guardo datos en dataframe
                            row_data = {
                                'n_iteration': cont_iter, 
                                'comp_to_select': comp_to_select,
                                'n_last_matches': n_last_matches, 'n_dias_ult_part': n_dias_ult_part, 'n_anios_hist': n_years_h2h, 'segun_localia': segun_localia, 'dif_con_against': dif_con_against,
                                'thr_corr': thr_corr, 'thr_fs': thr_fs,
                                'n_years_to_select': n_years_to_select, 'fill_na': fill_na, 
                                'bal_type': bal_type,'val_size': val_size, 'n_reg_test': n_reg_test, 
                                'k': k
                                }
                            
                            # Concateno y exporto datos
                            df_iteration = pd.concat([df_iteration, pd.DataFrame([row_data])], axis=0)
                            df_ite_test = pd.concat([df_ite_test, df_metrics], ignore_index=True) 
                            df_iteration_comp = pd.merge(df_iteration, df_ite_test, on='n_iteration', how='outer')     # Realizamos un merge por 'n_iteration' para combinar los DataFrames

                            if export:    
                                df_iteration.to_excel(f'{BASE_DIR_mod}/df_iteration_train.xlsx', index=False)
                                df_ite_test.to_excel(f'{BASE_DIR_mod}/df_iteration_test.xlsx', index=False)
                                df_iteration_comp.to_excel(f'{BASE_DIR_mod}/df_iteration.xlsx', index=False)

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
        df_iteration.to_excel(f'{BASE_DIR_mod}/df_iteration_train.xlsx', index=False)
        df_ite_test.to_excel(f'{BASE_DIR_mod}/df_iteration_test.xlsx', index=False)

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

def save_old_train(l_directories, base_path_old): # Por ahora lo hago manual porque no funciona como quiero… Arreglar.

    directories.make_directories(l_directorios=[base_path_old])

    import os 
    for directorio in l_directories:
        # if not os.path.exists(directorio):
        #     # Si no existe, crear el directorio
        #     os.makedirs(directorio)
        directories.mover_archivo(origen=directorio, destino=base_path_old)

def get_flashscore_data(BASE_DIR_flashscore, update_missing: bool = True, verbose: int = 0):
    """
    Obtengo datos de Flashscore a usar en el nuevo entrenamiento.
    """    
    # Podria recolectar missing para tener lo ultimos partidos actualzados
    if update_missing:
        d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True}
        main_next_matches.main(d_run, id_country, iteration_date=date, extract_missing=True, prepare_missing=False, export=d_run['export'])

    # Levanto datos --> Correr missing con Extract_missing=True pero Prepare_missing=False ?--> Deberia usarlos en modeling cuando hago retrain...
    df_match = pd.read_excel(f'{BASE_DIR_flashscore}/df_match.xlsx', index_col=0)
    df_match_player = pd.read_excel(f'{BASE_DIR_flashscore}/df_match_player.xlsx', index_col=0)
    df_match_odds = pd.read_excel(f'{BASE_DIR_flashscore}/df_match_odds.xlsx', index_col=0)
    
    # Verificacion 
    df_match_sin_dup = df_match[~df_match.index.duplicated()]
    if len(df_match) != len(df_match_sin_dup):
        logger.error("Hay partidos repetidos en df_match. Probablemente fallo la concatenacion de old y missing, habiendo concatenado mas de una vez un partido missing.")
        raise ValueError

    print("Flashscore data:")
    if verbose >= 0:
        print(df_match.shape)
        print(df_match_player.shape)
        print(df_match_odds.shape)

    if verbose >= 1:
        print(df_match)
        print(df_match_player)
        print(df_match_odds)

    return df_match, df_match_player, df_match_odds

def get_sofifa_data(country, update_sofifa, BASE_DIR_sofifa, verbose: int = 0):
    """
    Obtengo datos de Sofifa a usar en el nuevo entrenamiento.
    """
    if update_sofifa:
        # Actualizar sofifa con las ultimas seasons
        df_comp = pd.read_excel('./data/df_competencies.xlsx')
        df_comp_country = df_comp[(df_comp['id_country'] == id_country) & (df_comp['is_cup'] == 0)]
        
        # Levanto los datos viejos
        df_player_sofifa_old, df_player_fifa_sofifa_old = update_sofifa_data.read_last_player_data(country)

        # Obtengo ultimas seasons
        df_player, df_player_fifa = update_sofifa_data.get_player_data(id_country, country, df_comp_country, n_seasons_update=1, path_save=f'{BASE_DIR_sofifa}/data_seg')

        # Actualizar sofifa con las ultimas seasons
        df_player_sofifa, df_player_fifa_sofifa = update_sofifa_data.concat_player_data(df_player, df_player_fifa, df_player_sofifa_old, df_player_fifa_sofifa_old, path_save=BASE_DIR_sofifa)

    else:
        df_player_sofifa = pd.read_excel(f'{BASE_DIR_sofifa}/df_player_sofifa.xlsx', index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f'{BASE_DIR_sofifa}/df_player_fifa_sofifa.xlsx', index_col=0)

    if verbose >= 0: 
        print("\n\nSofifa data:")
        print(df_player_sofifa.shape)
        print(df_player_fifa_sofifa.shape)
    
    if verbose >= 1: 
        print(df_player_sofifa)
        print(df_player_fifa_sofifa)   

    return df_player_sofifa, df_player_fifa_sofifa

def read_df_match(country, iteration_date, retrain: bool = True):
    """
    Levanto el df_match
    """
    # Levanto df_match del pais
    if retrain:
        try:
            # Levanto desde los datos usados para el nuevo train
            path_new_train = f"./data/{country}/p2_data_understanding/old_updated/{iteration_date}"
            df_match = pd.read_excel(f'{path_new_train}/df_match.xlsx', index_col=0)
            # logger.warning(df_match)

        except FileNotFoundError:
            logger.warning(f"Fallo la carga del archivo df_match. No se encontró el archivo en '{path_new_train}'. Por ello recurro a levantarlo desde p6_deployment/missing")

        except IsADirectoryError:
            logger.warning(f"Fallo la carga del archivo df_match. No existe el directorio '{path_new_train}'. Por ello recurro a levantarlo desde p6_deployment/missing")

        # Levanto desde deployment/missing
        path = f'data/{country}/p6_deployment/missing/old_updated/df_match.xlsx' 
        df_match = pd.read_excel(path, index_col=0)

    else:
        logger.warning(f"Se esta obteniendo el df_test del df_match viejo (sin missing). En caso de querer extrarlo con missing tambien, usar retrain=True.")
        df_match = pd.read_excel(f"data/{country}/p3_data_preparation/clean_data/df_match_cleaned.xlsx", index_col=0)  

    if 'Unnamed: 0' in df_match.columns:
        df_match = df_match.drop(columns=['Unnamed: 0'])

    return df_match

def determine_rows_for_test_set(df_match, n_reg_test: int = 100, verbose : int = 0):
    """
    Determina qué registros pueden ser utilizados en el test
    Requisitos para el test
        -1: Que id_competition sea publica (lo mismo que hago en assess).
        -2: Que sean partidos jugados en los ultimos meses.

    # Parameters
        df: Dataframe.
        df_match: Del cual determinar que partidos son los ultimos partidos y la competicion (tiene date e id_comp)
        n_reg_test: Numero de registros los ultimos partidos a selecciona los cuales iran al df_test.

    # Return
        index: indice de los registros para test set
    """
    # Ordeno por fecha descendiente
    df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
    df_match = df_match.sort_values(by='date', ascending=False)

    if verbose >= 2:
        logger.info(df_match['date'].head(10))
    
    # Requisito 1: id competition.   # En df_match obtengo id_competition por match y determino posibles id_matches
    df1 = select_league_matches(df_match)
    index_comp = df1.index

    if verbose >= 2:
        df_filt_1 = df[df.index.isin(index_comp)]
        logger.info(f"Registros que pasan el requisito 1 (solo competencia publica): {len(df_filt_1)}")
    
    # Requisito 2: Last matches 
    df_match_comp = df_match[df_match.index.isin(index_comp)]  # Dejo solo las ligas / comp publicas
    df2 = df_match_comp.head(n_reg_test)
    index_last_matches = df2.index

    # Imprimo rango de fechas de df_test
    if verbose >= 1:
        date_hoy = datetime.datetime.now().date()
        col_index = df_match_comp.columns.get_loc('date')  # Índice de la columna "date"
        date_final = df_match_comp.iloc[0, col_index]
        date_inic = df_match_comp.iloc[100, col_index]

        # Los convierto a datetime
        date_final = pd.to_datetime(date_final, format='%d.%m.%Y %H:%M').date()   # Convierto fecha de object a datetime
        date_inic =  pd.to_datetime(date_inic, format='%d.%m.%Y %H:%M').date()   # Convierto fecha de object a datetime

        logger.info(f"Date hoy: {date_hoy}. Dates en df_test: {date_inic} --> {date_final}")

        dif_dias = date_hoy - date_final
        dif_dias_max = 20

        # Si no hay partidos de los ultimos x dias en df_test
        if dif_dias.days >= dif_dias_max:
            logger.warning(f"No hay registros de los ultimos {dif_dias.days} dias en df_test. Puede haber fallado algo en la extraccion de missing o en la seleccion del df_test.")

    if verbose >= 2:
        df_filt_2 = df[df.index.isin(index_last_matches)]
        logger.info(f"Registros que pasan el requisito 2 (solo last matches): {len(df_filt_2)}")
    
        # Selecciono registros que cumplen los requisitos

    logger.info(f"Index test set: {len(index_last_matches)}")
    return index_last_matches

def define_params_space(id_country, fast: bool = False):

    # Defino hiperparametros a probar
    d_comps = determine_country_competitions(id_country)
    # l_modelos = [DecisionTreeClassifier(), XGBClassifier(), SVC(), MLPClassifier()]   # , RandomForestClassifier(), GradientBoostingClassifier()
    # l_modelos = [LogisticRegression(), DecisionTreeClassifier(), XGBClassifier()]   # SVC(), RandomForestClassifier(), GradientBoostingClassifier()

    # 1728 iteraciones
    d_params = {
        'construct': {
            'n_dias_ult_part': [[60], [30, 180]], # [30, 180] 
            'n_years_h2h': [2],
            'segun_localia': [True, False],
            'dif_con_against': [False, True] 
        },
        'clean_data_2': {
            'competencies_to_select': [d_comps['comp_solo_liga'], d_comps['comp_sin_b'], d_comps['comp_sin_cups'], d_comps['all_comp']],
            'n_years_to_select': [2, 3, 5, 10], 
            'fill_na': [None, "0", 'ml'],
        },
        'select': {
            'thr_corr': [0.7, 0.85, None],
            'thr_fs': [None, 0.25, 0.5, 0.75],
        },
        'modeling': {
            'val_size': [0.10],
            'n_reg_test': [100], 
            'bal_type': [None, 'under'], # None
            'k': [5] 
        }
    }

    if fast:
        l_modelos = [LogisticRegression()]

        d_params = {  
            'clean_data_3': {
                'competencies_to_select': [d_comps['comp_solo_liga'], d_comps['comp_sin_b'], d_comps['all_comp']], 
            },
            'construct': {
                'n_last_matches': [[10]],  # Variables historicas en ultimos n partidos
                'n_dias_ult_part': [[30, 180]], # Variables historicas en partidos de ultimos n_days
                'n_years_h2h': [2],
                'segun_localia': [True, False],
                'dif_con_against': [False, True] 
            },
            'clean_data_2': {
                'n_years_to_select': [2, 3, 5, 10],
                'fill_na': [None, "0", 'ml'], 
            },
            'select': {
                'thr_corr': [0.7, 0.85, None],
                'thr_fs': [None, 0.25, 0.5, 0.75],
            },
            'modeling': {
                'val_size': [0.1],
                'n_reg_test': [100],
                'bal_type': ['under'], # None, 
                'k': [5] 
            }
        }
 
    logger.info(f"Parametros para entrenar: {d_params}")

    # Exportar un archivo .txt con los hiperparametros probados. --> Asi tengo que hiper probe en cada entrenamiento...
    # ...

    return d_params, l_modelos

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
        
    # Parametros de ejecucion
    id_country = 48
    data_unders = False
    update_sofifa = False
    data_prep_int = False # si queres entrenar ≠ con mismos datos, copiar df_int e integrate_data/ en nuevo p3_data_prep.
    
    d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Determino date 
    date = datetime.datetime.now().date() # datetime.datetime.now().date() 
    logger.info(f"Country: {country} Date: {date}")
    
    # Preparao datos, entreno modelos y evaluo en df_test
    # Defino hiperparametros a probar
    d_params, l_modelos = define_params_space(id_country, fast=True)
    
    # Preparo y entreno modelos para todas las combinaciones de hiper posibles 
    df_iteration_comp = comprehensive_search(country=country, date=date, 
                                             data_unders=data_unders, update_sofifa=update_sofifa, 
                                             data_prep_int=data_prep_int, 
                                             d_params=d_params, l_modelos=l_modelos)
