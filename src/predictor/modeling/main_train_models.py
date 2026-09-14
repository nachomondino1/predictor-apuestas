# Importo librerias
from predictor.utils.set_up_logging import logger
import pandas as pd
import numpy as np
import datetime
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
from predictor.data_understanding import update_sofifa_data
from predictor.data_preparation.select_data import select_league_matches
from predictor.data_preparation import concat_mapeos
from predictor.data_preparation.select_data import determine_country_competitions
from predictor.deployment import main_next_matches
import predictor.utils.directories as directories
from predictor.utils.io import read_df
from predictor.utils import training_log
from predictor.config import SEED, WALK_FORWARD_N_FOLDS, WALK_FORWARD_FOLD_SIZE

# Minimo de filas de train (sin NaN) para que valga la pena entrenar un fold.
# 5x el ~36 de features tipico: por debajo de eso `train_and_assess_models` ya
# avisa "EVITO TRAIN" por relacion filas/columnas.
MIN_TRAIN_ROWS_FOLD = 180
from itertools import product
from predictor.stages import DataUnderstanding, DataPreparation, Modeling
import time


def comprehensive_search(
    country, 
    date, 
    d_params, 
    l_modelos, 
    data_unders: bool = True,
    update_missing = False,
    data_prep_int: bool = True,
    data_prep_int_miss: bool = True,
    update_sofifa: bool = True,
    retrain: bool = True, 
    verbose: int = 0,
    checkpoint: int = 5,
    export: bool = True,
    run_type: str = "train",
    notes: str = "",
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

    run_type : str, opcional (por defecto "train")
        "smoke" | "train" | "retrain". Queda en el historial de corridas para poder
        distinguir pruebas de humo de entrenamientos reales.

    notes : str, opcional (por defecto "")
        Comentario libre que queda en el historial de corridas: qué cambió en el
        código respecto de la corrida anterior, por qué se corrió. Es lo que
        permite leer el historial como una tabla de experimentos
        (ver docs/EXPERIMENTOS.md).

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
    df_ite_train, df_ite_test, df_params_ite = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows_ite_list, rows_train_list, rows_test_list = [], [], []
    rows_test_by_fold_list = []  # detalle sin promediar (1 fila por combinacion x modelo x fold)
    du, dp, mo = DataUnderstanding(id_country=id_country, country=country), DataPreparation(id_country=id_country, country=country, date=date), Modeling(country, date=date) # Creo objetos de clases DataPreparation y Modeling

    # Imprimo largo de iteraciones
    n_iter = define_n_iterations(d_params)
    start_train = time.time()  # segundos desde el 1 de enero de 1970 UTC. Sin gate de verbose: siempre se loguea la corrida (ver utils/training_log.py)
    if verbose >= 0:
        logger.info(f"Numero de iteraciones totales: {n_iter}")

    # DIRECTORIOS
    # Defino rutas segun country y date
    BASE_DIR_du = f"./data/{country}/data_understanding/old_updated/{date}"
    BASE_DIR_flashscore = f'data/{country}/deployment/missing/old_updated'
    BASE_DIR_sofifa = f'data/{country}/data_understanding/sofifa_update' if retrain else f'data/{country}/data_understanding'
    BASE_DIR_dp = f"./data/{country}/data_preparation/{date}"     # BASE_DIR_mod = f"./data/{country}/modeling/{date}"
    BASE_DIR_mod = f"./data/{country}/modeling/{date}"
    ruta_base_modelos = f"{BASE_DIR_mod}/models" 
    directories.make_directories(l_directorios=[BASE_DIR_dp, BASE_DIR_mod, ruta_base_modelos])
    
    # Determino si el nuevo fifa ya salio o nó
    if datetime.datetime.now().month in [8, 9]: 
        fifa_not_released_yet = True
        logger.warning("FIFA NOT RELEASED YET = TRUE")
    else:
        fifa_not_released_yet = False
    
    ####################################################################### DATA UNDERSTANDING ####################################################################### --> Si hubo missing, esta bueno correrlo...
    if data_unders:
        directories.make_directories(l_directorios=[BASE_DIR_du, BASE_DIR_sofifa])

        # Podria recolectar missing para tener lo ultimos partidos actualzados
        if update_missing:
            d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True}
            main_next_matches.main(d_run, id_country, iteration_date=date, extract_missing=True, prepare_missing=False, export=d_run['export'])

        # Levanto datos --> Correr missing con Extract_missing=True pero Prepare_missing=False ?--> Deberia usarlos en modeling cuando hago retrain...
        df_match = pd.read_excel(f'{BASE_DIR_flashscore}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'{BASE_DIR_flashscore}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{BASE_DIR_flashscore}/df_match_odds.xlsx', index_col=0)
        if verbose > 1:
            describe_fs_data(df_match, df_match_player, df_match_odds)

        if fifa_not_released_yet:

            logger.warning("Mapeo con todos los paises pues aun no salio el nuevo fifa. Te recomiendo haber ejecutado concat_mapeos.py antes para tener datos lo mas recientes posibles. ")
            directories.duplicate_archivo(
                source_path='./data/_shared/data_preparation/integrate_data/df_map_players_fs_so.xlsx',
                destination_path=f'./data/{country}/data_preparation/{date}/integrate_data/df_map_players_fs_so.xlsx'
            )
        
            # Acordate de ejecutar concat_mapeos.py recientemente para tener datos relativamente nuevos.
            df_player_sofifa = read_df('./data/_shared/data_preparation/clean_data/df_player_sofifa_cleaned.xlsx')
            df_player_fifa_sofifa = read_df('./data/_shared/data_preparation/clean_data/df_player_fifa_sofifa_cleaned.xlsx')
        
        else:
            df_player_sofifa, df_player_fifa_sofifa = get_sofifa_data(country, update_sofifa=update_sofifa, BASE_DIR_sofifa=BASE_DIR_sofifa, n_seasons_update=1)

        # Exporto los datos para saber que datos use en el entrenamiento actual (no copio directorios porque me borra lo que ya hay en el directorio.)
        df_match.to_excel(f'{BASE_DIR_du}/df_match.xlsx', index=True)
        df_match_player.to_excel(f'{BASE_DIR_du}/df_match_player.xlsx', index=True)
        df_match_odds.to_excel(f'{BASE_DIR_du}/df_match_odds.xlsx', index=True)
        df_player_sofifa.to_excel(f'{BASE_DIR_du}/df_player_sofifa.xlsx', index=True)
        df_player_fifa_sofifa.to_excel(f'{BASE_DIR_du}/df_player_fifa_sofifa.xlsx', index=True)

        # Desbribe data
        if verbose >= 1:
            du.describe_data(df_match, df_match_player, df_match_odds, df_player_sofifa, df_player_fifa_sofifa)
    
    else:
        df_match = pd.read_excel(f'{BASE_DIR_du}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'{BASE_DIR_du}/df_match_player.xlsx', index_col=0)
        df_match_odds = pd.read_excel(f'{BASE_DIR_du}/df_match_odds.xlsx', index_col=0)
        df_player_sofifa = pd.read_excel(f'{BASE_DIR_du}/df_player_sofifa.xlsx', index_col=0)
        df_player_fifa_sofifa = pd.read_excel(f'{BASE_DIR_du}/df_player_fifa_sofifa.xlsx', index_col=0)

    ####################################################################### DATA PREPARATION (hasta integrate) #######################################################################
    if data_prep_int:

        # Format data
        df_match, df_match_player, df_match_odds, df_player_fifa_sofifa = dp.format_data(df_match, df_match_player, df_match_odds, df_player_fifa_sofifa, reformat=True, export=True)
        
        # Clean data
        df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, export=True)

        # Integrate data (tengo que volver a integrar... para generar df_player de fs bien y tener los nuevos jugadores que surgen en missing y mapearlos..)
        df_integrated = dp.integrate_data(df_match, df_match_player, df_player_sofifa=df_player_sofifa, df_player_fifa_sofifa=df_player_fifa_sofifa, fifa_not_released_yet=fifa_not_released_yet, export=True) 
        df_integrated.to_excel(f'{BASE_DIR_flashscore}/df_integrated.xlsx', index=True) # Exporto como df_int_old_updated

    else:
        df_integrated = pd.read_excel(f'{BASE_DIR_dp}/df_integrated.xlsx', index_col=0)
        print(df_integrated)

     ####################################################################### DATA PREPARATION (MISSING) #######################################################################
    
    # ACTUALIZAR PREPARACION DE MISSING (con el ultimo mapeo y los ultimos datos de sofifa, es clave)
    if data_prep_int_miss:

        # Levanto datos missing (solamente missing)
        df_match_miss = pd.read_excel(f'data/{country}/deployment/missing/data_understanding/all/df_match_miss.xlsx', index_col=0)
        idxs_missing = df_match_miss.index
        print(df_match_miss.shape)

        ## De df_integrated selecciono unicamente los missing (df_integrated ya tiene missing... no tengo que volver a integrar)
        df_integrated_missing = df_integrated[df_integrated.index.isin(idxs_missing)]
        print(df_integrated_missing.shape)

        if len(df_match_miss) != len(df_integrated_missing):
            logger.error(f"Fallo la obtencion del df_integrated_missing a partir del df_integrated. {len(df_match_miss)} ≠ {len(df_integrated_missing)}.")
            raise ValueError
        
        df_integrated_missing.to_excel(f'data/{country}/deployment/missing/data_preparation/all/df_integrated_missing.xlsx', index=True)
        
    ####################################################################### DATA PREPARATION (desde construct) #######################################################################
    # Determino registros a usar en test_set
    n_reg_val = d_params['modeling'].pop('n_reg_val', None)
    n_reg_test = d_params['modeling'].pop('n_reg_test', None)     # Obtener el valor de 'n_reg_test' y eliminarlo del diccionario
    if n_reg_test is not None:
        n_reg_test = n_reg_test[0]  # Si es una lista, obtenemos el primer elemento
        n_reg_val = n_reg_val[0]  # Si es una lista, obtenemos el primer elemento
        
    logger.info(d_params['modeling'].values())
    logger.info(f"N_REG_VAL: {n_reg_val} y FOLD_SIZE (ex N_REG_TEST): {n_reg_test}")

    # Walk-forward: en vez de UN split de 100 partidos, N folds consecutivos
    # (default 5 x 200 = 1000 partidos de evaluacion). Ver predictor/config.py.
    folds = determine_walk_forward_folds(df_match=df_match, fold_size=n_reg_test, n_reg_val=n_reg_val, verbose=1)

    # Fechas por partido, para resolver el train de cada fold (todo lo anterior a
    # su date_cutoff, copas incluidas).
    dates_by_match = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') if df_match['date'].dtype == object else df_match['date']

    # Clean post integrate
    for zz, param_values_00 in enumerate(product(*d_params['clean_post_integrate'].values()), start=1):
        comp_to_select, n_years_to_select =  param_values_00[0], param_values_00[1]

        logger.info(f" Iteracion clean_data post integrate".center(120, "#"))
        print(f"Hiper clean_post_construct --> comp_to_select: {comp_to_select} ; n_years_to_select: {n_years_to_select}")            
        path_clean = f'{comp_to_select}_{n_years_to_select}'
        df_int_clean = dp.clean_post_integrate(df_integrated, n_years_to_select=n_years_to_select, competencies_to_select=comp_to_select)

        # Construct_data
        for i, param_values_2 in enumerate(product(*d_params['construct'].values()), start=1):

            # Asigno valor a cada hiperpametro
            n_last_matches, n_years_h2h, segun_localia, calculate_dif, decay_rate = param_values_2
            logger.info(f" Iteracion Construct Nº {i} ".center(120, "#"))
            print(f'Hiper construct --> n_last_matches: {n_last_matches} ; n_years_h2h: {n_years_h2h}; segun_localia: {segun_localia} ; calculate_dif:{calculate_dif}')

            # Construyo datos
            path_cons = f"{path_clean}__{n_last_matches}_{n_years_h2h}_{segun_localia}_{calculate_dif}_{decay_rate}"
            path_construct = f'{BASE_DIR_dp}/construct_data/df_constructed_{path_cons}.xlsx'
            df_constructed = dp.construct_data(df_int_clean, n_last_matches=n_last_matches, n_years_h2h=n_years_h2h, segun_localia=segun_localia, calculate_dif=calculate_dif, decay_rate=decay_rate, export=True)
            if export:
                df_constructed.to_excel(path_construct, index=True)

            # Clean post construct
            df_cons_etiquetado = dp.clean_post_construct(df_constructed, n_years_to_select=n_years_to_select)

            # Etiqueto df_constructed
            df_cons_etiquetado, df_etiquetas = dp.tag_string_data_to_integer(df_cons_etiquetado)
            path_etiqueta = f'{BASE_DIR_dp}/tag/df_etiquetas_{path_cons}.xlsx'
            if export:
                df_etiquetas.to_excel(path_etiqueta, index=True)

            # Select data
            for j, param_values_4 in enumerate(product(*d_params['select'].values()), start=1):

                # Asigno valor a cada hiperpametro
                thr_corr, thr_fs, fill_na = param_values_4[0], param_values_4[1], param_values_4[2]

                logger.info(f" Iteracion Select Nº {i}.{zz}.{j} ".center(120, "#"))
                print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")            
                path_sel = f'{path_cons}__{thr_corr}_{thr_fs}_{fill_na}'

                ####################################################################### MODELING #######################################################################
                for h, param_values_5 in enumerate(product(*d_params['modeling'].values()), start=1):

                    # Asigno valor a cada hiperparametro
                    bal_type, k = param_values_5[0], param_values_5[1]
                    cont_iter += 1
                    if verbose >= 0:
                        logger.info(f" Iteracion Modeling Nº {i}.{zz}.{j}.{h} ".center(120, "#"))
                        print(f'\n - Hiper construct --> n_last_matches: {n_last_matches} ; n_years_h2h: {n_years_h2h} ; segun_localia: {segun_localia} \n - Hiper clean_post_construct n_years_to_sel: {n_years_to_select} comp_to_select: {comp_to_select} \n- Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs} \n - Hiper treat_nan --> {fill_na} \n - Hiper modeling --> n_reg_val: {n_reg_val} ; fold_size: {n_reg_test}; bal_type: {bal_type} ; k: {k}')
                        logger.critical(f" Iteracion Nº {cont_iter} de {n_iter} ({cont_iter*100/n_iter:.0f}%)")

                    # WALK-FORWARD: entreno y evaluo una vez por fold, y despues
                    # promedio. La feature selection y el scaler se ajustan DENTRO
                    # de cada fold, solo con sus filas de train: antes corrian
                    # sobre el dataset completo (test incluido), lo cual filtraba
                    # informacion y inflaba las metricas del backtest.
                    rows_train_folds, rows_test_folds = [], []
                    for fold in folds:
                        n_fold, index_test, index_val = fold['n_fold'], fold['index_test'], fold['index_val']
                        idx_train_fold = dates_by_match[dates_by_match < fold['date_cutoff']].index

                        df_train_rows = df_cons_etiquetado[df_cons_etiquetado.index.isin(idx_train_fold)]

                        # `select_data` dropea toda fila con algun NaN antes de medir
                        # importancias, y la disponibilidad de features cae hacia atras
                        # en el tiempo (e.g. expected_goals no existe en partidos
                        # viejos): en los folds mas antiguos eso puede dejar 0 filas.
                        # Salteo el fold con un aviso fuerte en vez de romper la corrida.
                        n_train_sin_nan = len(df_train_rows.dropna(axis=0, how='any'))
                        if n_train_sin_nan < MIN_TRAIN_ROWS_FOLD:
                            logger.error(
                                f"Fold {n_fold}: solo {n_train_sin_nan} filas de train sin NaN "
                                f"(de {len(df_train_rows)}), menos que el minimo {MIN_TRAIN_ROWS_FOLD}. "
                                f"SALTEO el fold -- el promedio va a salir de menos folds (ver n_folds). "
                                f"Si pasa en varios, hay que acortar el walk-forward o rellenar NaN antes de seleccionar."
                            )
                            continue

                        try:
                            # (a) Feature selection ajustada SOLO con el train del fold
                            df_sel_train = dp.select_data(df_train_rows, thr_corr=thr_corr, thr_fs=thr_fs, export=False)
                            cols_sel = list(df_sel_train.columns)
                        except Exception as e:
                            logger.error(f"Fold {n_fold}: fallo la seleccion de features, lo salteo. {e}", exc_info=True)
                            continue

                        # (b) Aplico esas columnas a train + val + test del fold
                        idx_fold = idx_train_fold.union(index_val).union(index_test)
                        df_sel_fold = df_cons_etiquetado.loc[df_cons_etiquetado.index.isin(idx_fold), cols_sel]

                        # (c) Scaler ajustado SOLO con el train del fold
                        df_sel_fold = dp.clean_post_select(
                            df=df_sel_fold, fill_na=fill_na,
                            path_save=f'{BASE_DIR_dp}/clean_post_select/scaler_model_{path_sel}_f{n_fold}.pkl',
                            fit_rows=idx_train_fold,
                        )
                        if verbose >= 2:
                            df_sel_fold.to_excel(f'{BASE_DIR_dp}/select_data/df_selected_{path_sel}_f{n_fold}.xlsx', index=True)

                        # (d) Split y entrenamiento de este fold
                        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(
                            df_sel_fold, bal_type=bal_type, index_val=index_val, index_test_set=index_test, export=False)

                        rows_train_f, rows_test_f = mo.train_and_assess_models(
                            X_val=X_val, y_val=y_val, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                            l_modelos=l_modelos, ruta_base_mod_seg=ruta_base_modelos, cont_iter=cont_iter,
                            df_match=df_match, df_match_odds=df_match_odds, k=k,
                            save_artifacts=(n_fold == 1),  # 1 solo .pkl por combinacion (el fold mas reciente)
                        )
                        for r in rows_train_f + rows_test_f:
                            r['n_fold'] = n_fold
                        rows_train_folds.extend(rows_train_f)
                        rows_test_folds.extend(rows_test_f)

                    # Promedio entre folds: 1 fila por (combinacion, modelo), con el
                    # desvio de las metricas clave para saber si una mejora supera
                    # al ruido.
                    rows_train = aggregate_folds(rows_train_folds)
                    rows_test = aggregate_folds(rows_test_folds)
                    rows_test_by_fold_list.extend(rows_test_folds)
                    
                    if len(rows_test) > 0:
                        # Guardo datos en dataframe
                        rows_ite = {
                            'n_iteration': cont_iter, 
                            'comp_to_select': comp_to_select,
                            'n_last_matches': n_last_matches, 'n_anios_hist': n_years_h2h, 'segun_localia': segun_localia, 'calculate_dif': calculate_dif, 'decay_rate': decay_rate,
                            'thr_corr': thr_corr, 'thr_fs': thr_fs,
                            'n_years_to_select': n_years_to_select, 'fill_na': fill_na, 
                            'bal_type': bal_type,'n_reg_val': n_reg_val, 'n_reg_test': n_reg_test, 
                            'k': k
                            }
                        
                        # Acumular filas en listas
                        rows_train_list.extend(rows_train)  # Agregar todos los elementos de rows_train (si es una lista de diccionarios)
                        rows_test_list.extend(rows_test)    # Agregar todos los elementos de rows_test (si es una lista de diccionarios)
                        rows_ite_list.append(rows_ite)

                        if ((cont_iter % checkpoint == 0) or (cont_iter == n_iter) )and export: # n_iter o n_iter -1 ????
                            logger.critical("Checkpoint. Guardado de datos")
                
                            # Concatenar todas las filas acumuladas en DataFrames
                            df_ite_train = pd.concat([df_ite_train, pd.DataFrame(rows_train_list)], ignore_index=True)
                            df_ite_test = pd.concat([df_ite_test, pd.DataFrame(rows_test_list)], ignore_index=True)
                            df_params_ite = pd.concat([df_params_ite, pd.DataFrame(rows_ite_list)], ignore_index=True)

                            # Exportar
                            df_ite_train.to_excel(f'{BASE_DIR_mod}/df_ite_train.xlsx', index=False)
                            df_ite_test.to_excel(f'{BASE_DIR_mod}/df_ite_test.xlsx', index=False)
                            df_params_ite.to_excel(f'{BASE_DIR_mod}/df_params_ite.xlsx', index=False)
                            if rows_test_by_fold_list:  # detalle por fold, para inspeccionar la dispersion
                                pd.DataFrame(rows_test_by_fold_list).to_excel(f'{BASE_DIR_mod}/df_ite_test_folds.xlsx', index=False)

                            # Limpiar listas después de exportar
                            rows_ite_list.clear()
                            rows_train_list.clear()
                            rows_test_list.clear()

                            if verbose >= 0:
                                current_train = time.time()
                                ritmo = cont_iter / ((current_train - start_train) / 3600)  # iteraciones por hora
                                horas_restantes = (n_iter - cont_iter) / ritmo
                                horas_train = n_iter / ritmo

                                logger.info(f"Ritmo: {ritmo:.1f} iteraciones/hora (ideal >60). Quedan {n_iter - cont_iter} iteraciones.")
                                logger.info(f"Tiempo estimado para finalizar: {horas_restantes:.1f} horas (~{horas_restantes * 60:.1f} minutos).")
                                logger.info(f"Tiempo total proyectado de entrenamiento: {horas_train:.1f} horas.")


    end_train = time.time()
    duration_min = (end_train - start_train) / 60
    if verbose >= 0:
        logger.info(f"Tiempo total de entrenamiento: {duration_min:.1f} minutos")

    # Anoto la corrida en el historial (data/_shared/logs/_training_log.xlsx), sea smoke o entrenamiento real.
    training_log.log_run(
        country=country, date=date, n_iter=n_iter, l_modelos=l_modelos,
        duration_min=duration_min, models_path=ruta_base_modelos,
        df_ite_test=df_ite_test, run_type=run_type, notes=notes,
    )

    return df_params_ite, df_ite_train, df_ite_test

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

def describe_fs_data(df_match, df_match_player, df_match_odds, verbose: int = 0):
    """
    Obtengo datos de Flashscore a usar en el nuevo entrenamiento.
    """    
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

def get_sofifa_data(country, update_sofifa, BASE_DIR_sofifa, n_seasons_update: int = 1, verbose: int = 0):
    """
    Obtengo datos de Sofifa a usar en el nuevo entrenamiento.
    """
    if update_sofifa:
        # Actualizar sofifa con las ultimas seasons
        df_comp = pd.read_excel('./data/_shared/master_tables/df_competencies.xlsx')
        df_comp_country = df_comp[(df_comp['id_country'] == id_country) & (df_comp['is_cup'] == 0)]
        
        # Levanto los datos viejos
        df_player_sofifa_old, df_player_fifa_sofifa_old = update_sofifa_data.read_last_player_data(country)

        # Obtengo ultimas seasons
        df_player, df_player_fifa = update_sofifa_data.get_player_data(id_country, country, df_comp_country, n_seasons_update=n_seasons_update, path_save=f'{BASE_DIR_sofifa}/data_seg')

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

def aggregate_folds(rows: list, std_metrics=('f1_score', 'test_accuracy', 'roi', 'expected_roi', 'error', 'f1_score_train')):
    """
    Colapsa las filas de los folds del walk-forward a UNA fila por modelo.

    Los nombres de las métricas se mantienen (`f1_score`, `roi`, etc.) pero ahora
    el valor es el **promedio entre folds** — así `main_select_model.py` y
    `training_log.py` siguen funcionando sin cambios, y de paso eligen por un
    promedio de 1000 partidos en vez de por un único corte de ~100.

    Además agrega `std_<metrica>` para las métricas clave: es el número que dice
    si una mejora de "+2 de f1" está dentro del ruido entre folds o no. Y
    `n_folds`, para no confundir un promedio de 5 folds con uno de 2.

    # Parameters
        rows: Filas devueltas por `train_and_assess_models`, ya etiquetadas con
            `n_fold`. (list[dict])
        std_metrics: Métricas a las que además se les calcula el desvío. (tuple)

    # Returns
        Una fila por modelo. (list[dict])
    """
    if not rows:
        return []

    df = pd.DataFrame(rows)
    if 'model_name' not in df.columns:
        return rows

    out = []
    for model_name, g in df.groupby('model_name', sort=False):
        row = {}
        for col in g.columns:
            if col == 'n_fold':
                continue
            s = g[col]
            # Numéricas -> promedio entre folds. El resto (hiperparámetros,
            # shapes, listas de columnas) -> el valor del primer fold, a título
            # informativo: puede diferir entre folds (cada uno elige sus features
            # y sus hiperparámetros con su propio train).
            row[col] = s.mean() if pd.api.types.is_numeric_dtype(s) else s.iloc[0]
        row['model_name'] = model_name
        row['n_folds'] = len(g)
        for m in std_metrics:
            if m in g.columns and pd.api.types.is_numeric_dtype(g[m]):
                row[f'std_{m}'] = g[m].std(ddof=0)
        out.append(row)

    return out

def determine_walk_forward_folds(df_match, n_folds: int = None, fold_size: int = None, n_reg_val: int = 100, verbose: int = 0):
    """
    Arma los folds de la evaluación walk-forward: bloques consecutivos de
    partidos ordenados por fecha, del más reciente al más viejo. Reemplaza al
    split único de `determine_rows_for_test_set` (ver predictor/config.py).

    Para el fold k: test = su bloque de `fold_size` partidos; val = los
    `n_reg_val` partidos inmediatamente ANTERIORES (para elegir hiperparámetros);
    train = todo lo anterior a eso. O sea, cada fold se entrena y valida solo
    con pasado respecto de su test — nunca ve el futuro, igual que producción.

    Mismos 2 requisitos que antes para que un partido pueda ir a test:
    competición pública y estar entre los últimos partidos.

    # Parameters
        df_match: Dataframe con `date` e `id_competition`.
        n_folds: Cantidad de folds. Default `config.WALK_FORWARD_N_FOLDS`.
        fold_size: Partidos por fold de test. Default `config.WALK_FORWARD_FOLD_SIZE`.
        n_reg_val: Partidos de validación por fold. (int)

    # Returns
        Lista de dicts `{'n_fold', 'index_test', 'index_val'}`, del fold más
        reciente al más viejo. Si no hay suficientes partidos para los `n_folds`
        pedidos, devuelve los que entran y avisa.
    """
    n_folds = WALK_FORWARD_N_FOLDS if n_folds is None else n_folds
    fold_size = WALK_FORWARD_FOLD_SIZE if fold_size is None else fold_size

    # Ordeno por fecha descendiente (0 = más reciente)
    df_match = df_match.copy()
    df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M')
    df_match = df_match.sort_values(by='date', ascending=False)

    # Requisito 1: solo competiciones públicas (idem determine_rows_for_test_set)
    index_comp = select_league_matches(df_match).index
    df_match_comp = df_match[df_match.index.isin(index_comp)]
    idx = df_match_comp.index

    # Cuántos folds entran: cada fold consume fold_size (test) y necesito además
    # n_reg_val de validación + algo de train para el fold más viejo.
    min_train = fold_size  # piso arbitrario pero explícito: al menos un bloque de train
    max_folds = max(0, (len(idx) - n_reg_val - min_train) // fold_size)
    if max_folds < n_folds:
        logger.warning(
            f"Solo alcanza para {max_folds} folds de {fold_size} partidos (hay {len(idx)} "
            f"partidos de competiciones públicas, y hacen falta {n_reg_val} de val + "
            f"{min_train} de train). Pedidos: {n_folds}."
        )
        n_folds = max_folds
    if n_folds <= 0:
        raise ValueError(
            f"No hay suficientes partidos ({len(idx)}) para armar ni un fold de {fold_size} "
            f"con {n_reg_val} de validación. Bajá fold_size/n_reg_val o usá más años de datos."
        )

    folds = []
    for k in range(n_folds):
        start_test = k * fold_size
        end_test = start_test + fold_size
        index_test = idx[start_test:end_test]
        index_val = idx[end_test:end_test + n_reg_val]

        # Todo lo anterior a la fecha mas vieja de (test + val) puede ser train.
        # Se usa una FECHA de corte (y no "lo que no es test ni val") para que el
        # train del fold k no incluya los folds mas recientes -- que son futuro
        # respecto de k -- y para que las copas (no elegibles como test) entren
        # al train solo si son anteriores.
        idx_eval = index_test.union(index_val)
        date_cutoff = df_match_comp.loc[df_match_comp.index.isin(idx_eval), 'date'].min()
        folds.append({
            'n_fold': k + 1,
            'index_test': index_test,
            'index_val': index_val,
            'date_cutoff': date_cutoff,
        })

        if verbose >= 1:
            fechas_test = df_match_comp.loc[index_test, 'date']
            logger.info(
                f"  Fold {k + 1}: test={len(index_test)} partidos "
                f"({fechas_test.min().date()} → {fechas_test.max().date()}), val={len(index_val)}"
            )

    logger.info(f"Walk-forward: {len(folds)} folds de {fold_size} partidos de test + {n_reg_val} de val cada uno.")
    return folds

def determine_rows_for_test_set(df_match, n_reg_val: int = 100, n_reg_test: int = 100, verbose : int = 0):
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
    
    # Requisito 1: id competition
    df1 = select_league_matches(df_match)
    index_comp = df1.index
    df_match_comp = df_match[df_match.index.isin(index_comp)]  # Dejo solo las ligas / comp publicas

    # Paso 2: Selecciono últimos partidos para test (ya hecho)
    df_test = df_match_comp.head(n_reg_test)
    index_test = df_test.index
    df_match_comp_filt = df_match_comp[~df_match_comp.index.isin(index_test)]

    # Paso 3: Selecciono los siguientes 100 partidos para validation
    df_val = df_match_comp_filt.head(n_reg_val)
    index_val = df_val.index

    logger.info(f"Index val set: {len(index_val)} y Index test set: {len(index_test)}")
    return index_val, index_test

def concat_dataframes_on_iteration(df_params_ite, df_ite_train, df_ite_test):
    # Realizamos un merge por 'n_iteration' para combinar los DataFrames
    df_temp = pd.merge(
        df_ite_train,
        df_ite_test,
        on=['n_iteration', 'model_name'],
        how='outer',
        suffixes=('_train', '_test')  # Evita conflictos de columnas duplicadas
    )

    # df_iteration = pd.merge(df_params_ite, df_temp, on=['n_iteration'], how='outer')
    df_iteration = pd.merge(
        df_temp,
        df_params_ite,
        on='n_iteration',
        how='left'  # LEFT asegura que se repita la info de params sin perder filas
    )
    return df_iteration

def define_params_space(id_country):

    # Defino hiperparametros a probar
    d_comps = determine_country_competitions(id_country)
    # random_state=SEED para que dos corridas de la misma config den el mismo
    # resultado (RandomForest y XGBoost son aleatorios; ver predictor/config.py).
    l_modelos = [LogisticRegression(random_state=SEED), XGBClassifier(random_state=SEED), RandomForestClassifier(random_state=SEED)] # Pruebo Modelos no lineales #  MLPClassifier() (distrib de probas rara)

    l_comp = [d_comps['all_comp']]
    l_comp_sin_duplicados = list(map(list, set(map(tuple, l_comp))))
    print(l_comp_sin_duplicados)

    d_params = {  
        'clean_post_integrate': {
            'competencies_to_select': l_comp_sin_duplicados, 
            'n_years_to_select': [5, 10],
        },
        'construct': {
            'n_last_matches': [[60], [120]], # [30] [60, 180] # Variables historicas en ultimos n partidos,
            'n_years_h2h': [2],
            'segun_localia': [True, False], 
            'calculate_dif': [True], # False
            'decay_rate': [0.1, 0.5],
        },
        'select': {
            'thr_corr': [0.7, None], #  0.85
            'thr_fs': [0.05, 0.15], # 0.9 para ver metricas con la variable mas importante. Si no le gano a eso, es porque las otras variables son una verga.
            'fill_na': [None, "0"] # en realidad es clean_post_select
        },
        'modeling': {
            'n_reg_val': [100],
            'n_reg_test': [200],  # tamaño de cada fold del walk-forward (ver predictor/config.py)
            'bal_type': ['under'], 
            'k': [10],
            # 'refit': ['mean_test_cross_entropy_loss', 'mean_test_f1_score'] # A futuro...
        }
    }

    logger.info(f"Parametros para entrenar: {d_params}")
    return d_params, l_modelos

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
        
    # Parametros de ejecucion
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [-1]

    data_unders = True 
    update_sofifa = False if data_unders else False
    data_prep_int = True
    data_prep_int_miss = False
    
    d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}

    for id_country in l_countries:

        country = d_countries[id_country]

        # Determino date  
        date = datetime.datetime.now().date() # Si queres usar fecha en especifico: datetime.datetime.strptime('2025-03-16', '%Y-%m-%d').date()
        logger.info(f"Country: {country} Date: {date}".center(120, "#"))
        
        # Defino hiperparametros a probar
        d_params, l_modelos = define_params_space(id_country)
        
        # Exportar un archivo .txt con los hiperparametros probados. --> Asi tengo que hiper probe en cada entrenamiento...
        df_params = pd.DataFrame.from_dict(
            {(cat, param): values for cat, params in d_params.items() for param, values in params.items()},
            orient="index"
        )

        # Preparo y entreno modelos para todas las combinaciones de hiper posibles 
        df_params_ite, df_ite_train, df_ite_test = comprehensive_search(
            country=country, date=date, 
            data_unders=data_unders, update_sofifa=update_sofifa, 
            data_prep_int=data_prep_int, data_prep_int_miss=data_prep_int_miss,
            d_params=d_params, l_modelos=l_modelos
            )
        
        # concatenar y hacer un df_iteration
        df_iteration = concat_dataframes_on_iteration(df_params_ite, df_ite_train, df_ite_test)

        # Exportar resultados
        df_params.to_csv(f'data/{country}/modeling/{date}/hyperparameters.csv')
        df_iteration.to_excel(f'./data/{country}/modeling/{date}/df_iteration.xlsx', index=False)

