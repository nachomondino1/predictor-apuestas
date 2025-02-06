import sys
sys.path.append('.')  # Fallaba el import de main
from utils import directories
from utils.set_up_logging import logger
import pandas as pd
from dotenv import load_dotenv
from p3_data_preparation import construct_data
from p6_deployment import main_next_matches
from main import Modeling
from p3_data_preparation import format_data
from p4_modeling import asses_model

def get_model_predictions_with_missing(df_pred_test, n_model, model_name, id_country, country, iteration_date):
    """
    Obtengo df_probabilities test + missing.
    """
    # Predict missing
    df_pred_missing = predict_missing(id_country, n_model, model_name, iteration_date)

    # Concateno df_pred_test y df_pred missing.
    df_predicciones = pd.concat([df_pred_test, df_pred_missing], axis=0)
    l_cols = ['date', 'id_team_home', 'id_team_away', 'predicted_result', 'prob_class_1', 'prob_class_2', 'prob_class_0', 'odds_home', 'odds_draw', 'odds_away']
    df_predicciones = df_predicciones.loc[:, l_cols]

    #  Agrego columnas result y expected result
    df_predicciones = get_goals(df_predicciones, country)  

    # Determino result y expected result segun goals
    df_predicciones = construct_data.determine_result(df_predicciones) # Intento hacerlo antes con df_match pero rompia.
    df_predicciones = construct_data.determine_expected_result(df_predicciones, goals_to_xg_ratio=0.42) # Intento hacerlo antes con df_match pero rompia.
    return df_predicciones
        
def predict_missing(id_country, n_model, model_name, iteration_date): # No se si funciona ok el run_missing
    """
    Recolecta predicciones de modelos en partidos "missing"

    :return: df_predicciones del modelo en partidos missing
    """
    # Defino variables (no tocar)
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True}  # No puedo correr data_unders = False si los proximos partidos ya estan en missing.
    d_model = {'n_model': n_model, 'model_name': model_name} # XGBClassifier, neural_networ, SVC, LogisticRegression, MLPClassifier

    # Usar mnm.py con predict_missing=True y data_unders=False.
    df = main_next_matches.main(d_run, id_country, iteration_date=iteration_date, d_model=d_model, predict_missing=True, export=False, verbose=0) 

    if not isinstance(df, pd.DataFrame):
        raise ValueError("No se generó un dataframe.")

    elif len(df) == 0:
        logger.warning("No hay predicciones de partidos missing...")
        raise ValueError

    return df

def get_goals(df, country):  # Ponerlo como funcion dentro de BettingStrategy????
    """
    Agregar columnas result y acerte en df_pred. Determino result y expected result para poder determinar "acerte"
    """
    ## Levanto df_match_miss para obtener goals? ??
    df_match_miss = pd.read_excel(f"data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx", index_col=0)
    l_columns_to_copy = ['goals_home', 'goals_away', 'expected_goals_(xg)_home', 'expected_goals_(xg)_away']   # Columnas a copiar
    
    # Ordeno df por date
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
    df = df.sort_values(by='date', ascending=True)

    # Asignar valores de df_match_miss a df solo en las columnas y filas correspondientes
    for idx, row in df.iterrows():

        for column in l_columns_to_copy:

            df.loc[idx, column] = df_match_miss.loc[idx, column] 

    return df

def determine_metrics(df_predicciones, country, iteration_date):
    mo = Modeling(country, iteration_date)

    # Recalculo metricas con test + missing
    df_predicciones, d_metrics = mo.calculate_metrics(df_predicciones)

    # Reformateo teams
    df_predicciones = format_data.map_teams(df_predicciones, df_teams=mo.df_teams) # Convierto ids de equipos a nombres
    return df_predicciones, d_metrics

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