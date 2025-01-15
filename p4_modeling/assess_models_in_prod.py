import sys
sys.path.append('.')  # Fallaba el import de main
from utils import directories
from utils.set_up_logging import logger
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from p3_data_preparation import format_data
from p3_data_preparation.construct_data import determine_result, determine_expected_result
from p4_modeling import asses_model, betting_strategy, compare_assess_prod
from p6_deployment import main_next_matches


def update_test_with_missing(df_ite, id_country, country, iteration_date, path_save):
    """
    Creo un df_iteration actualizado con las predicciones de test y las de missing. Actualiza el input para la seleccion de modelos o estrategia de apuesta.

    # Parameters
        df_ite
        country
        iteration_date

    # Return
        df_ite_updated: Train mas el test actualizado con nuevas metricas segun test y missing. (DataFrame)
    """
    # Defino variables
    df_test = pd.DataFrame()    
    bs = betting_strategy.BettingStrategy() # no le paso iteration_date para que no guarde datos
    progress_bar = tqdm(total=len(df_ite), ncols=80)  # Inicializo barra de progreso
    base_path_dp = f'./data/{country}/p3_data_preparation/{iteration_date}'

    # Defino que n_model use en prod (para comparar assess y prod)
    n_model_prod, _, _ = main_next_matches.read_data_of_best_model(id_country)

    # Por modelo
    for idx, row in df_ite.iterrows():

        n_model = row['n_iteration'] if 'n_iteration' in df_ite.columns else idx
        model_name = row['model_name']
        logger.info(f'n_model: {n_model} model_name: {model_name}')

        # Levanto df_predicciones de test
        df_pred = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx", index_col=0)
        logger.info(df_pred.shape)

        # Predict missing
        df_pred_missing = predict_missing(id_country, n_model, model_name, iteration_date)

        '''
        # DIF ASSESS Y PROD (ACA ESTA OK? FUNCIONA?)
        # Si el modelo fue utilizado en produccion:
        if n_model == n_model_prod:
            compare_assess_prod.compare_preparation(n_model, country, export=True)
            compare_assess_prod.compare_modeling(n_model, country, export=True)
        '''

        #  Agrego columnas result y expected result
        df_pred_missing = determine_results(df_pred_missing, country)  

        # Concateno df_pred y df_pred missing.
        df_predicciones = pd.concat([df_pred, df_pred_missing], axis=0)
        df_predicciones = df_predicciones.loc[:, ['result', 'predicted_result', 'prob_class_1', 'prob_class_0', 'prob_class_2']]  # Elimino metricas del df_test viejo (dejo el df_pred_proba raso...)

        # Recalculo metricas con test + missing
        df_filled = pd.read_excel(f'{base_path_dp}/treat_nan/df_filled_columns.xlsx', index_col=0) #  para calcular relleno..
        df_predicciones, d_metrics = asses_model.calculate_metrics(df_predicciones, country=country, df_filled=df_filled, retrain=True, export=False)  # --> Sobreescribe metricas de df_pred...
        df_predicciones = determine_expected_result(df_predicciones, goals_to_xg_ratio=0.42) # Intento hacerlo antes con df_match pero rompia.
        
        d_params = bs.define_hiperparameters(strategy='train')
        df_predicciones, _, d_roi = bs.calculate_roi_in_combinations(df_predicciones, d_params=d_params) # Chequear que funciona...
        d_metrics.update(d_roi)
        d_metrics.update(asses_model.calculate_advanced_metrics(df_predicciones=df_predicciones))

        # Convierto ids de equipos a nombres
        df_teams = pd.read_excel(f'{base_path_dp}/integrate_data/df_teams.xlsx', index_col=0)
        df_predicciones = format_data.map_teams(df_predicciones, df_teams=df_teams)

        # Guardo datos + Exporto
        row_test = {'n_iteration': n_model, 'model_name': model_name, **d_metrics}
        df_row_test = pd.DataFrame([row_test])  # 1. Convertir el diccionario d_metrics en un DataFrame de una fila
        df_test = pd.concat([df_test, df_row_test], ignore_index=True)  # 2. Concatenar este nuevo DataFrame con df_metrics existente
        ## Exporto datos del modelo
        df_test.to_excel(f'{path_save}/df_iteration_test.xlsx', index=False)
        df_predicciones.to_excel(f'{path_save}/{n_model}__{model_name}_predicciones.xlsx', index=True)
        
        progress_bar.update(1)

    progress_bar.close()

    # Concateno df_test actualizado con df_ite
    df_ite = df_ite.rename(columns={col: f"{col}_train" for col in df_ite.columns if col != 'n_iteration'})
    df_ite_updated = pd.merge(df_ite, df_test, on='n_iteration', how='outer')  
    
    # Creo columna 'dif_roi_pp'
    df_ite_updated['dif_roi_pp'] = (df_ite_updated['roi_por_partido'] - df_ite_updated['roi_por_partido_train']) / df_ite_updated['roi_por_partido_train']
    ave_dif = df_ite_updated['dif_roi_pp'].mean() * 100
    if ave_dif > 0:
        logger.critical(f"La variacion del ROIpp de prod respecto de prod+asses es de {ave_dif:.0f}%.")
    else:
        logger.error(f"La variacion del ROIpp de prod respecto de prod+asses es de {ave_dif:.0f}%. Hay un declive general en el ROI tras los nuevos partidos assess. Puede ser por tener mucho nan en produccion aunque tal vez fueron pocos partidos aun.")

    # Exporto df_iteration actualizado
    df_ite_updated.to_excel(f'{path_save}/df_iteration.xlsx', index=False)

    return df_ite_updated
        
def predict_missing(id_country, n_model, model_name, iteration_date): # No se si funciona ok el run_missing
    """
    Recolecta predicciones de modelos en partidos "missing"

    :return: df_predicciones del modelo en partidos missing
    """
    # Defino variables (no tocar)
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True}  # No puedo correr data_unders = False si los proximos partidos ya estan en missing.
    d_model = {'n_model': n_model, 'model_name': model_name, 'iteration_date': iteration_date} # XGBClassifier, neural_networ, SVC, LogisticRegression, MLPClassifier

    # Usar mnm.py con predict_missing=True y data_unders=False.
    df = main_next_matches.main(d_run, id_country, d_model=d_model, predict_missing=True, export=False, verbose=0) 

    if not isinstance(df, pd.DataFrame):
        raise ValueError("No se generó un dataframe.")

    elif len(df) == 0:
        logger.warning("No hay predicciones de partidos missing...")
        raise ValueError

    return df

def determine_results(df, country):  # Ponerlo como funcion dentro de BettingStrategy????
    """
    Agregar columnas result y acerte en df_pred. Determino result y expected result para poder determinar "acerte"
    """
    ## Levanto df_match_miss para obtener goals? ??
    df_match_miss = pd.read_excel(f"data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx", index_col=0)
    l_columns_to_copy = ['goals_home', 'goals_away', 'expected_goals_(xg)_home', 'expected_goals_(xg)_away']   # Columnas a copiar
    
    # Ordeno df por date
    df = df.sort_values(by='date', ascending=False)

    # Asignar valores de df_match_miss a df solo en las columnas y filas correspondientes
    for idx, row in df.iterrows():

        for column in l_columns_to_copy:

            df.loc[idx, column] = df_match_miss.loc[idx, column] 

    ## Determino result y expected result segun goals
    df = determine_result(df) # Intento hacerlo antes con df_match pero rompia.
    df = determine_expected_result(df, goals_to_xg_ratio=0.42) # Intento hacerlo antes con df_match pero rompia.
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    id_country = 77
    iteration_date = '2024-12-23'

    # Defino variables
    d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Directorios
    BASE_PATH = f"data/{country}/p4_modeling/{iteration_date}"
    BASE_PATH_2 = f"data/{country}/p4_modeling/{iteration_date}/assess"
    directories.make_directories(l_directorios=[BASE_PATH_2])

    # Levanto df_iteration
    df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")

    # Filtro df_ite seleccionado top x% de registros.
    ## Ordenar los registros por 'metric' en orden descendente
    df_ite = df_ite.sort_values(by='roi_por_partido', ascending=False)
    ## Seleccionar el 20% de los registros con los valores más altos de 'metric'
    cutoff = int(len(df_ite) * 0.01)  # Calcular el 20% superior
    df_ite_filt = df_ite.iloc[:cutoff]

    print(df_ite_filt)

    # Evaluo modelos en test y missing
    df_ite_updated = update_test_with_missing(df_ite_filt, id_country, country, iteration_date)