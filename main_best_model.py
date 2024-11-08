# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
from set_up_logging import logger
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
from p3_data_preparation.select_data import determine_country_competitions
from p4_modeling import assess_models_in_prod, train_models
import directories
import pickle


# Guardado de assess actual y creacion de directorio para nuevo assess
def save_old_assess():
    """
    Muevo assess a old_assess_iterations para no sobreescribirlo con el nuevo assess.
    """
    directorio_origen = f"./data/{country}/p4_modeling/"
    directorio_destino = f"./data/{country}/old/"
    directories.make_directories([directorio_destino])
    directories.mover_archivo(directorio_origen, directorio_destino)

def select_best_model(df, ruta_assess_2, country, thr_distrib=0.35):
    """
    Selecciona el mejor modelo
    """
    first = True
    logger.info(df)

    # Analisis de rentabilidad
    ## 1) Selecciono Top 20% modelos (Pareto)
    df = df.sort_values(by='roi_por_partido', ascending=False)
    percentile_value = df['roi_por_partido'].quantile(0.8)
    df_top_rois = df[df['roi_por_partido'] >= percentile_value]
    logger.info(df_top_rois)

    # 2) Selecciono modelos con ROI creciente
    df_top_rois_filt = roi_creciente(df_top_rois, country)

    # Analisis de distribucion
    df_best = df_top_rois_filt.copy()

    # Por modelo
    for idx, row in df_top_rois_filt.iterrows():

        # Leo sus predicciones
        n_ite, model_name = idx, row['model_name']
        idx_real = (df_best.index == idx) & (df_best['model_name'] == model_name)
        logger.info(f"Model: {idx} - {model_name}")
        df_pred = read_model_predictions(ruta_assess_2, n_ite, model_name)

        # Calculo distribucion segun sus predicciones
        n_loc, n_emp, n_vis = calculate_distribucion(df_pred)
        logger.info(f"Distribucion: {n_loc}-{n_emp}-{n_vis}")

        if first:
            n_loc_real, n_emp_real, n_vis_real = calculate_distribucion(df_pred, col_to_sum='result')
            df_best['# result 1'] = n_loc_real
            df_best['# result 0'] = n_emp_real
            df_best['# result 2'] = n_vis_real
            logger.info(f"Distribucion real: {n_loc_real}-{n_emp_real}-{n_vis_real}")
            first = False

        if not check_similar_distribution(thr_distrib, n_loc, n_emp, n_vis, n_loc_real, n_emp_real, n_vis_real):
            # Elimino modelo con mala distribucion
            df_best = df_best.drop(df_best[(df_best.index == idx) & (df_best['model_name'] == model_name)].index)
        else:
            # Calcula precision
            df_best.loc[idx_real, '# pred 1'] = n_loc
            df_best.loc[idx_real, '# pred 0'] = n_emp
            df_best.loc[idx_real, '# pred 2'] = n_vis
            df_best.loc[idx_real, '# aciertos'] = df_pred['acerte'].sum()
            df_best.loc[idx_real, 'precision'] = df_pred['acerte'].mean() * 100

    # De los mejores, el que mas ROIpp tiene
    logger.info(df_best)
    best_model = df_best[df_best['roi_por_partido']==df_best['roi_por_partido'].max()]
    return best_model, df_best

def roi_creciente(df_top_rois, country):
    """
    Calcular % ROI
    """
    # Determino desde cuando calcular el % ROI
    col_to_use = 'roi_50' if country == 'france' else 'roi_100'  # Para FRA: col_to_use = 'roi_50'
    new_col = f'% {col_to_use}'
    logger.info(f"Columna usada: {col_to_use}")

    # Calculo % ROI
    df_top_rois[new_col] = (df_top_rois['roi_por_partido'] - df_top_rois[col_to_use]) / abs(df_top_rois[col_to_use])
    df_top_rois_filt = df_top_rois[df_top_rois[new_col] >= 0]

    # ROIpp 50 < ROIpp 100 < ROIpp 150 y asi. --> ROI creciente (en la realidad no es tan asi... no siempre son lineales...)
    '''
    if 'roi_150' in df_top_rois.columns: # Tendria que automatizarlo...
        # Seleccionar filas donde la condición de múltiples columnas se cumpla
        df_top_rois_filt = df_top_rois[(df_top_rois['roi_50'] < df_top_rois['roi_100']) &
                                (df_top_rois['roi_100'] < df_top_rois['roi_150']) &
                                (df_top_rois['roi_150'] < df_top_rois['roi_por_partido'])]
    else:
        # Seleccionar filas sin 'roi_150'
        df_top_rois_filt = df_top_rois[(df_top_rois['roi_50'] < df_top_rois['roi_100']) &
                                (df_top_rois['roi_100'] < df_top_rois['roi_por_partido'])]
    '''
    return df_top_rois_filt


def read_model_predictions(ruta_assess_2, n_ite, model_name):
    df_pred = pd.read_excel(f'{ruta_assess_2}/{n_ite}__{model_name}_pred.xlsx')
    return df_pred

def calculate_distribucion(df, col_to_sum: str = 'predicted_result'):

    n_local = len(df[df[f'{col_to_sum}'] == 1])
    n_empate = len(df[df[f'{col_to_sum}'] == 0])
    n_vis = len(df[df[f'{col_to_sum}'] == 2])
    return n_local, n_empate, n_vis

def check_similar_distribution(thr, n_loc, n_emp, n_vis, n_loc_real, n_emp_real, n_vis_real):

    # Calculo variacion por resultado
    var_loc = calculate_variation(n_loc, n_loc_real)
    var_emp = calculate_variation(n_emp, n_emp_real)
    var_vis = calculate_variation(n_vis, n_vis_real)
    
    for var in [var_loc, var_emp, var_vis]:
        if abs(var) >= thr:
            return False
    return True

def calculate_variation(end, ini):
    return (end - ini) / abs(ini)


def main(l_modelos, d_params, rows_to_features_min: int = 10, retrain: bool = False, continue_old_train: bool = False, export: bool = True):
    """
    Entrena modelos segun las combinaciones de hiperparametros deseadas. Luego los evalua en produccion y selecciona el mejor.
    """
    # Evito sobreescribir assess actual y lo muevo. Ademas, creo directorio para el nuevo assess.
    # save_old_assess()      # Cuidado al correr este progrma, sobreescribis el assess que esta hoy actualmente. Si lo queres evitar, guarda el assess en carpeta "old_assess_iterations" 

    ruta_base_mod = f"./data/{country}/p4_modeling/{date}" 
    ruta_base_dp = f"./data/{country}/p4_modeling/{date}/p3_data_preparation"
    ruta_base_modelos = f"./data/{country}/p4_modeling/{date}/models" 
    ruta_assess = f'{ruta_base_mod}/assess_models_in_prod/data_preparation'
    ruta_assess_2 = f'{ruta_base_mod}/assess_models_in_prod/modeling'

    if not continue_old_train: 
        directories.make_directories(l_directorios=[ruta_base_dp, ruta_base_modelos, ruta_assess, ruta_assess_2])
    
    # Preparao datos, entreno modelos y evaluo en df_test
    df_ite_train, df_ite_test = train_models.main(country, ruta_base_dp, ruta_base_mod, ruta_base_modelos, d_params, l_modelos, rows_to_features_min=rows_to_features_min, retrain=retrain, continue_old_train=continue_old_train, export=export)

    # Preparo datos missing y evaluo modelos en produccion
    df_ite_test_prod = assess_models_in_prod.main(df_ite_train, country, date, ruta_base_dp, ruta_base_mod, export=export)

    # Selecciono el mejor modelo (mayor roi por partido en produccion)
    best_model, df_best = select_best_model(df_ite_test_prod, ruta_assess_2, country)
    df_best.to_excel(f'{ruta_base_mod}/df_best_model.xlsx') # Los mejores modelos
    pickle.dump(best_model, open(f"{ruta_base_mod}/best_model.pkl", "wb"))

    # Concateno dataframes en un solo dataframe.
    ##  test y prod
    # Saco n_iteration de indice y la hago una columna normal
    df_ite_test_prod = df_ite_test_prod.reset_index() # Convertir el índice en una columna normal
    df_ite_test_prod.rename(columns={'index': 'n_iteration'}, inplace=True)
    df_concat = pd.merge(df_ite_test, df_ite_test_prod, on=['n_iteration', 'model_name'], how='outer', suffixes=('_test', '_prod'))
    # df_concat.to_excel('/Users/nachomondino/Desktop/df_iteration_test_ct.xlsx')
    # test + prod y train
    df_ite = pd.merge(df_ite_train, df_concat, on='n_iteration', how='outer')     # Realizamos un merge por 'n_iteration' para combinar los DataFrames
    df_ite.to_excel(f'{ruta_base_mod}/df_iteration.xlsx')
    # logger.info(df_merged)

    # Actualizar df_best_models.xlsx autoamticamente
    # ...
    return df_ite

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
        
    # Parametros de ejecucion
    id_country = 77
    retrain = True
    continue_old_train = False

    # Determina date de la iteracion
    if continue_old_train:
        date = '2024-10-30'
    else:
        date_con_hora = datetime.datetime.now()  # + datetime.timedelta(days=1) --> Si queres correr 2 el mismo dia. No funciona aun.
        date = date_con_hora.date()

    # Defino hiperparametros a probar
    d_comps = determine_country_competitions(id_country)
    l_modelos = [LogisticRegression(), 'neural_network', SVC(), XGBClassifier()] #  GradientBoostingClassifier(), MLPClassifier()]
    # l_modelos = [SVC(), XGBClassifier()] #  GradientBoostingClassifier(), MLPClassifier()]

    # 1728 iteraciones
    d_params = {  
        'construct': {
            'n_dias_ult_part': [[30, 180], [60, 240]], # [90], [30]
            'n_years_h2h': [3],
            'segun_localia': [True, False], # False
            'dif_con_against': [True, False] # False
        },
        'clean_data_2': {
            'competencies_to_select': [d_comps['comp_sin_b'], d_comps['all_comp']], # d_comps['comp_sin_b'] solo para USA 
            'n_years_to_select': [3, 5, 10], #, None --> no tiene sentido porque el fifa arranca en 2007 (hace 17 años). Tampoco tiene sentido usar 15 años si elimino los datos de antes de 2012
        },
        'select': {
            'thr_corr': [0.7, 0.85, None],
            'thr_fs': [None, 0.25, 0.5, 0.75],
        },
        'treat_nan': {
            'fill_na': [None, 'ml'],
        },
        'modeling': {
            'val_size': [0.10],
            'test_size': [0.15], 
            'bal_type': ['under'], # None (ni con f1_score..)
            'k': [5] 
        }
    }
    
    rows_to_features_min = 10      # Idealmente mayor a 10. En caso de redes neuronales entre 30 y 100 veces mas.
    logger.info(f"Parametros para entrenar: {d_params}")
        
    ## Obtengo el nombre del pais segun su id
    if id_country > 0:
        df_countries = pd.read_excel('./data/df_countries.xlsx')
        country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0]
    else:
        country = 'all'

    # Entreno modelos y los evaluo en produccion.
    df = main(l_modelos, d_params, rows_to_features_min=rows_to_features_min, continue_old_train=continue_old_train, retrain=retrain)