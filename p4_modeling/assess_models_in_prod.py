# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
from set_up_logging import logger
import pandas as pd
import numpy as np
## Data preparation
from p3_data_preparation import construct_data
from p6_deployment.main_next_matches import DataPreparationNew, filter_dataframe_by_date
# Modeling
import pickle
import joblib
from p4_modeling import asses_model
import directories
import datetime

# Data understanding
def save_assess(ruta_base):
    """
    Muevo assess a old_assess_iterations para no sobreescribirlo con el nuevo assess.
    """
    date_con_hora = datetime.datetime.now()
    date = date_con_hora.date()

    l_dir_origen = [f'{ruta_base}/assess_models_in_prod', f'{ruta_base}/df_iteration_test_prod.xlsx',  f'{ruta_base}/df_best_model.xlsx', f'{ruta_base}/df_iteration.xlsx']
    directorio_destino = f'{ruta_base}/old_assess_iterations/{date}'

    # Creo directorio de destino
    directories.make_directories(l_directorios=directorio_destino)

    # Por directorio de origen
    for direc in l_dir_origen:
        # Muevo directorio a destino
        directories.mover_archivo(direc, directorio_destino)

    # Creo directorios de proximo assess
    l_dirs = [f'{ruta_base}/assess_models_in_prod/data_preparation', f'{ruta_base}/assess_models_in_prod/modeling']
    directories.make_directories(l_directorios=l_dirs)

def select_league_matches(df):
    """
    Filtra partidos seleccionado solo aquellos que son de liga (eliminando partidos de copa)
    """
    # Levanto df_competencies
    df_comp = pd.read_excel('data/df_competencies.xlsx')

    # Selecciono solo las ligas del pais
    l_leagues = list(df_comp[(df_comp['is_cup']==0) & (df_comp['is_second_division']==0)]['id_competition'].values) 
    print("Ligas: ", l_leagues)

    df = df[df['id_competition'].isin(l_leagues)]
    print(f"Shape sin copas: {df.shape}")
    return df

def load_preparation_hyperparameters(row_hiper, verbose: int = 1):

    d = {}

    # Guardo hiperparametros en diccionario
    ## Construct_data
    d['n_dias_ult_part'] = load_as_list(row_hiper['n_dias_ult_part'].values[0])    # d['n_dias_ult_part'] =  int(row_hiper['n_dias_ult_part'].values[0])
    d['n_years_h2h'] = int(row_hiper['n_anios_hist'].values[0])
    d['segun_localia'] = row_hiper['segun_localia'].values[0]
    d['dif_con_against'] = row_hiper['dif_con_against'].values[0]
    ## Clean_data_2
    n_years_to_select = row_hiper['n_years_to_select'].values[0]    
    if pd.isna(n_years_to_select):
        d['n_years_to_select'] = None
    else:
        # Verificar si el valor tiene decimales
        d['n_years_to_select'] = int(n_years_to_select) if n_years_to_select.is_integer() else float(n_years_to_select)
    d['comp_to_select'] = load_as_list(row_hiper['comp_to_select'].values[0])
    d['selected_columns'] = load_as_list(row_hiper['X_columns'].values[0])

    if verbose >= 1:
        print("\nHiperparametros cargados:")
        for key, value in d.items():
            print(f'\t {key}: {value}')
    return d

def load_as_list(lista):
    """
    Convierte un elemento con forma de lista pero con otro formato a una lista.
    Garantiza que el resultado siempre sea una lista.
    """
    try:
        # Intenta evaluar la expresión y asegurarte de que es una lista
        result = eval(lista)
        if not isinstance(result, list):
            return [result]  # Si no es una lista, lo convierte en una lista con un solo elemento
        return result
    except (TypeError, SyntaxError):  # Captura posibles errores de eval() o tipo
        return list(lista)  # Si eval falla, intenta convertir a lista usando list()

def load_models(ruta_base_dp, d):

    # Levanto hiperparametros de DataPreparation de la iteracion 
    n_ult_part, n_years_h2h, segun_localia, dif_con_against, n_years_sel, comp = d['n_dias_ult_part'], d['n_years_h2h'], d['segun_localia'], d['dif_con_against'], d['n_years_to_select'], d['comp_to_select']
    path_construct = f'{n_ult_part}_{n_years_h2h}_{segun_localia}_{dif_con_against}'
    path_clean = f'{n_years_sel}_{comp}'

    # Cargo modelos segun hiperparametros
    tager_loaded = pd.read_excel(f'{ruta_base_dp}/df_etiquetas_{path_construct}.xlsx')
    scaler, columns_scaled = joblib.load(f'{ruta_base_dp}/scaler_model_{path_construct}_{path_clean}.pkl')
    return tager_loaded, scaler, columns_scaled

def load_trained_models(n_ite, ruta_base_mod):
    """
    Carga todos los modelos entrenados para la iteracion
    """
    l_models, l_names = [], []

    # Para entrenamientos anteriores a 2024-10-10
    try:
        model = pickle.load(open(f"{ruta_base_mod}/models/{n_ite}_model.pkl", "rb"))
        l_models.append(model)
        l_names.append('neural_networ') # da igual el model name..    
    
    except FileNotFoundError:
        for model_name in ['LogisticRegression', 'neural_networ', 'XGBClassifier', 'GradientBoostingClassifier', 'MLPClassifier', 'SVC']:  # Podria hacer unique a df_iteration_test['model_name']:
            try:
                loaded_model = pickle.load(open(f"{ruta_base_mod}/models/{n_ite}_{model_name}.pkl", "rb"))

                # Agrego modelo a lista
                l_models.append(loaded_model)
                l_names.append(model_name)

            except FileNotFoundError:
                pass

    return l_models, l_names

################################################### MAIN ###################################################
def main(df_iteration, country, iteration_date, ruta_base_dp, ruta_base_mod, export: bool = True, relleno_formaciones: bool = True, 
         n_days_to_fill:int = 60, strategy = 'general', verbose: int = 0):
    """
    Levanta los datos missing, los prepara y predice con modelo ya entrenado. 
    """
    logger.critical("ASSESS MODELS IN PRODUCTION...")
    # Evito sobreescribir assess actual y lo muevo. Ademas, creo directorio para el nuevo assess.
    save_assess(ruta_base_mod)   # Cuidado al correr este progrma, sobreescribis el assess que esta hoy actualmente. Si lo queres evitar, guarda el assess en carpeta "old_assess_iterations" 

    # Definicion de variables
    df_iteration_prod = pd.DataFrame()
    dp = DataPreparationNew(country=country, export=False)  # Creo objeto de clase DataPreparationNew

    #______________________________________________ DATA UNDERSTANDING ______________________________________________#  # --> Levanto dfs missing de p6_deployment
    if verbose >= 0:
        print("\n", "#"*120, "\n", "DATA UNDERSTANDING".center(120), "\n", "#"*120, "\n")

    # Levanto datos
    df_match_odds = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/all/df_match_odds_miss.xlsx', index_col=0)
    df_teams = pd.read_excel(f'data/{country}/p3_data_preparation/integrate_data/df_teams.xlsx', index_col=0)
    df_int_missing = pd.read_excel(f'data/{country}/p6_deployment/missing/data_preparation/all/df_integrated_missing.xlsx', index_col=0)  # tienen que ser /all... Solo partidos missing.
    
    # Determino la fecha del partido missing mas "viejo"
    df_int_missing['date'] = pd.to_datetime(df_int_missing['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime --> estoy casi seguro que no hace falta.
    initial_date = df_int_missing['date'].min()
    if verbose >= 1:
        print(f"Initial_date (es decir, la del partido missing mas viejo): {initial_date}")

    # Filtro por competencias. No quiero partidos de copas (e.g. FA cup) solo de la liga
    df_int_missing = select_league_matches(df_int_missing)
    
    # Construyo la variable "result"
    df_int = construct_data.determine_result(df_int_missing, 'result')  # Es necesaria? Creo que si porque en main_next_matches.py no le construyo result...

    # Por iteracion
    for idx, row in df_iteration.iterrows():

        # Definicion de variables
        n_ite = row['n_iteration']
        row_hiper = df_iteration[df_iteration['n_iteration'] == n_ite]
        if verbose >= 0:
            print("\n", "#"*120, "\n", f"ITERATION Nº {n_ite}".center(120), "\n", "#"*120, "\n")

        # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
        d_hiper = load_preparation_hyperparameters(row_hiper)
        tager, scaler, columns_scaled = load_models(ruta_base_dp, d_hiper)
        
        #______________________________________________ DATA PREPARATION ______________________________________________#
        if verbose >= 0:
            print("DATA PREPARATION".center(120, "-"))

        # Levanto datos ya construidos
        path_cons = f'{ruta_base_mod}/assess_models_in_prod/data_preparation/df_constructed_{d_hiper['n_dias_ult_part']}_{d_hiper['n_years_h2h']}_{d_hiper['segun_localia']}_{d_hiper['dif_con_against']}.xlsx'

        try:
            df_cons = pd.read_excel(path_cons, index_col=0)
            
            if verbose >= 1:
                print("Evito construir datos dado que levanto dataframe ya construido")

        # Levanto df_integrated y construyo datos
        except FileNotFoundError:      
            # Levanto datos viejos  
            df_old_int = pd.read_excel(f'data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)  ## Datos con los que entrenó el modelo
            df_old_int = df_old_int.sort_values(by='date', ascending=False)  # Ordeno por fecha ascendente. Funciona? Es entendida como datetime la columna? Si.

            # Selecciono los ultimos partidos de los ya jugados
            if verbose >= 1:
                logger.info("Seleccion de ultimos partidos para construccion de variables...")
            n_days_max = max(d_hiper['n_dias_ult_part'])
            n_days_period = n_days_max * 2 if d_hiper['segun_localia'] == True else n_days_max
            df_last_old_matches_construct = filter_dataframe_by_date(df=df_old_int, initial_date=initial_date, n_days=n_days_period) # No sirve de nada hacerlo flex dado que construct_data() de main.py usa n_days

            if relleno_formaciones:
                # Selecciono los ultimos partidos de los ya jugados
                if verbose >= 1:
                    logger.info("Seleccion de ultimos partidos para rellenar formaciones...")
                df_last_old_matches_fill = filter_dataframe_by_date(df=df_old_int, initial_date=initial_date, n_days=n_days_to_fill) # Los parates pueden ser de 3 meses o mas. Por eso tomo 5 meses para tener un poco de margen de seguridad.

                # Relleno formaciones
                df_fill, df_c1, df_c2 = dp.fill_data_not_available_yet(df_int, df_last_old_matches_fill, verbose=verbose)
                df_int = df_fill

            # Construyo datos usando partidos viejos
            df_cons = dp.construct_data_new(df_next_matches=df_int, df_last_old_matches=df_last_old_matches_construct, df_old_matches=df_old_int, 
                                            n_days=d_hiper['n_dias_ult_part'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'], 
                                            dif_con_against=d_hiper['dif_con_against'],  columns_used=columns_scaled, verbose=verbose)
            df_cons.to_excel(path_cons, index=True)
        
        # Sigo preparando datos
        df_tag = dp.tag_string_data_to_integer_new(df_cons, tager, verbose=verbose)
        df_clean = dp.clean_data_2_new(df_tag, scaler, columns_scaled, d_hiper['comp_to_select'], verbose=verbose)

        # Por modelo entrenado en iteration (tienen misma preparacion hasta select_data)
        l_models, l_names = load_trained_models(n_ite, ruta_base_mod)
        for loaded_model, model_name in zip(l_models, l_names):

            # Selecciono datos + Treat nan values (c/modelo selecciona ≠ columnas)
            df_sel = dp.select_data_new(df_clean, d_hiper['selected_columns'], verbose=verbose)
            df_treat, df_emer = dp.treat_nan_values_new(df_sel, verbose=verbose)

            if verbose >= 1:
                logger.info(f"\nShape Dataframe antes de Modeling(): {df_treat.shape}")
                if len(df_sel) != len(df_treat):
                    logger.warning(f"WARNING! De los {len(df_sel)} proximos partidos, quedan {len(df_treat)} luego de la preparacion")

            #______________________________________________ MODELING ______________________________________________#
            if verbose >= 0:
                print("MODELING".center(120, "-"))
                logger.info(f"model_name: {model_name}. loaded_model: {loaded_model}")

            # Realizo predicciones sobre los nuevos partidos
            try:
                y_pred_prob = loaded_model.predict_proba(df_treat) # Te da las probabilidad de cada clase. Funciona para todos los modelos? # AttributeError: predict_proba is not available when probability=False
                classes = loaded_model.classes_
            except AttributeError: # AttributeError: 'Sequential' object has no attribute 'predict_proba'
                y_pred_prob = loaded_model.predict(df_treat)
                classes = [0, 1, 2]
            # logger.info(classes)

            y_pred = np.argmax(y_pred_prob, axis=1)  # Obtengo la clase predicha segun la que tenga mayor probabilidad 
            df_pred_proba = pd.DataFrame({
                    'predicted_result': y_pred,
                    f'prob_class_{classes[1]}': y_pred_prob[:, 1],  # Probabilidad de la clase 1
                    f'prob_class_{classes[0]}': y_pred_prob[:, 0],  # Probabilidad de la clase 0
                    f'prob_class_{classes[2]}': y_pred_prob[:, 2]   # Probabilidad de la clase 2 (si hay 3 clases)
                }, index=df_treat.index)
        
            # Agrego resultado y cuotas a df_match
            df_match_odds_2 = df_match_odds[df_match_odds.index.isin(df_treat.index)]
            df_match_odds_2 = df_match_odds.reindex(df_treat.index)  # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test
            df_match = df_int.loc[df_int.index.isin(df_treat.index), ['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition', 'result']]

            # Concateno conjunto de datos
            df_match_odds_2 = asses_model.calculate_result_probabilities_by_bookmaker(df_match_odds_2) # Caculo probabilidades segun casa de apuesta
            if relleno_formaciones:
                df_predicciones = pd.concat([df_match, df_match_odds_2, df_pred_proba, df_c1['copiado_formaciones'], df_emer['emergency_fill']], axis=1)
            else:
                df_predicciones = pd.concat([df_match, df_match_odds_2, df_pred_proba, df_emer['emergency_fill']], axis=1)
 
            # Ordeno partidos predichos por fecha decreciente (x si no estaba)
            df_predicciones['date'] = pd.to_datetime(df_predicciones['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
            df_predicciones = df_predicciones.sort_values(by='date', ascending=True)  # Ordeno por fecha de menos reciente a mas reciente para calcular ROI bien.

            # Evaluo predicciones del modelo
            df_predicciones, d_roi = asses_model.calculate_roi_by_betting_strategy(df_predicciones, strategy=strategy, verbose=verbose)
            if verbose >= 1:
                logger.info(f"Metricas: {d_roi}")
                asses_model.confusion_matrix(y_real=df_predicciones['result'], y_pred=df_predicciones['predicted_result'], verbose=1)

            # Revierto etiquetas para tener nombres de equipos en vez de ids
            d_mapeo = dict(zip(df_teams.index, df_teams['team_name']))        
            df_predicciones['id_team_home'] = df_predicciones['id_team_home'].replace(d_mapeo)
            df_predicciones['id_team_away'] = df_predicciones['id_team_away'].replace(d_mapeo)
            
            # Concateno datos
            d_roi['param1'], d_roi['param2'] = str(d_roi['param1']), str(d_roi['param2'])
            df_roi = pd.DataFrame(d_roi, index=[row['n_iteration']])
            df_roi['X_shape_missing'] = [df_treat.shape]  # Me interesa saber el largo del df_missing
            df_roi['model_name'] = model_name
            df_iteration_prod = pd.concat([df_iteration_prod, df_roi], axis=0)

            # Exporto datos
            if export:
                df_predicciones.to_excel(f'{ruta_base_mod}/assess_models_in_prod/modeling/{row['n_iteration']}__{model_name}_pred.xlsx', index=True)
                df_iteration_prod.to_excel(f'{ruta_base_mod}/assess_models_in_prod/df_iteration_prod_seg_{iteration_date}.xlsx')

    if export:
        df_iteration_prod.to_excel(f'{ruta_base_mod}/df_iteration_test_prod.xlsx')

    return df_iteration_prod

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    import main_best_model as mbm
    logger.warning("Asegurate de haber extraido nuevos partidos missing respecto del anterior assess puesto que sino será igual.")

    # Defino condiciones del analisis
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [77]
    only_select_best_model = False # False
    bet_strategy = 'general' # Si queres saber el ROI de la realidad, usar 'reality'
    d = {
        6: ["argentina", "2024-05-07"],
        48: ["england", '2024-10-02'],
        55: ["france", "2024-10-03"], 
        59: ["germany", "2024-10-13"], # 13
        77: ["italy", "2024-10-03"], # 15
        148: ["spain", "2024-10-13"],  # 13
        167: ["usa", "2024-10-06"]
    }

    for id_country in l_countries:

        # Seleccionar pais
        n_days_fill_data = 360 if id_country == 55 else 60  # Para FRA uso 360 porque no llega a minimos para integrar var de jugadores.
        country, date = d[id_country]
        logger.info(f"Country: {country}, Date: {date}")

        # Defino rutas
        ruta_base_dp = f"./data/{country}/p4_modeling/{date}/p3_data_preparation"
        ruta_base_mod = f"./data/{country}/p4_modeling/{date}" 
        ruta_assess_2 = f'{ruta_base_mod}/assess_models_in_prod/modeling'

        # Levanto datos de modelos entrenados
        df_ite_train = pd.read_excel(f'{ruta_base_mod}/df_iteration_train.xlsx')
        logger.info(df_ite_train)
        try:
            df_ite_test = pd.read_excel(f'{ruta_base_mod}/df_iteration_test.xlsx')
            logger.info(df_ite_test)
        except FileNotFoundError:
            pass

        # Evaluo modelos en produccion
        if only_select_best_model:
            df_ite_test_prod = pd.read_excel(f'{ruta_base_mod}/df_iteration_test_prod.xlsx', index_col=0)
            logger.info(df_ite_test_prod)
        else:
            df_ite_test_prod = main(df_ite_train, country, date, ruta_base_dp, ruta_base_mod, relleno_formaciones=True, n_days_to_fill=n_days_fill_data, strategy=bet_strategy)

        # Selecciono el mejor modelo (mayor roi por partido en produccion)
        best_model, df_best = mbm.select_best_model(df_ite_test_prod, ruta_assess_2, country, thr_distrib=0.35)
        
        # Concateno dataframes en un solo dataframe.
        ##  test y prod
        # Saco n_iteration de indice y la hago una columna normal
        try:
            df_ite_test_prod = df_ite_test_prod.reset_index() # Convertir el índice en una columna normal
            df_ite_test_prod.rename(columns={'index': 'n_iteration'}, inplace=True)
            df_concat = pd.merge(df_ite_test, df_ite_test_prod, on=['n_iteration', 'model_name'], how='outer', suffixes=('_test', '_prod'))
            logger.info(df_concat)
            # df_concat.to_excel('/Users/nachomondino/Desktop/df_iteration_test_ct.xlsx')
            # test + prod y train
            df_merged = pd.merge(df_ite_train, df_concat, on='n_iteration', how='outer')     # Realizamos un merge por 'n_iteration' para combinar los DataFrames
        
        # En entrenamientos anteriores a 2024-10-10, no separaba df_ite_test de df_ite_train, lo ponia junto en df_ite_train.
        except NameError: 
            # df_merged = pd.concat([df_ite_train, df_ite_test_prod], axis=1)
            df_merged = pd.merge(df_ite_train, df_ite_test_prod, on='n_iteration', how='outer', suffixes=('_test', '_prod'))     # Realizamos un merge por 'n_iteration' para combinar los DataFrames
        logger.info(df_merged)

        # Exporto datos
        ## Seleccion de modelo
        df_best.to_excel(f'{ruta_base_mod}/df_best_model.xlsx') # Los mejores modelos
        pickle.dump(best_model, open(f"{ruta_base_mod}/best_model.pkl", "wb"))
        ## df_iteration concatenando train, test y test_prod
        df_merged.to_excel(f'{ruta_base_mod}/df_iteration.xlsx')

