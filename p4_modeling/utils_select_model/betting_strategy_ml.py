import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from p3_data_preparation.construct_data import determine_expected_result
from p4_modeling.asses_model import determine_confidence_margin, calculate_roi, determine_roi, drop_old_metrics, normalize_column
from utils import directories
import datetime
from itertools import product
from tqdm import tqdm
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def read_predictions(country, iteration_date, n_model, model_name, assess: bool = False, date_assess: str = None):

    if assess and date_assess is None:
        date_assess = datetime.datetime.now().date()

    if assess:
        logger.warning("Debe ser test + assess concatenado")
        path = f"data/{country}/p4_modeling/{iteration_date}/best_model/2_assess/{date_assess}/{n_model}__{model_name}_test_assess_.xlsx"  # Debe ser test + assess concatenado
    else:
        path = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"

    df_pred_test = pd.read_excel(path, index_col=0)
    return df_pred_test

def main(df_pred):

   # Seleccionar variables predictoras
    features = ['odds_home', 'odds_draw', 'odds_away', 'overround', 'prob_class_0', 'prob_class_1', 'prob_class_2', 'bookmaker_result', 'predicted_result', 'prob_result_to_bet', 'confidence_margin', 'odd_to_bet',	'kelly_criterion', 'result_to_bet'] # 'id_country', 'id_competition', 
    features = ['overround', 'bookmaker_result', 'predicted_result', 'odds_home', 'odds_draw', 'odds_away', 'confidence_margin', 'prob_result_to_bet', 'odd_to_bet', 'kelly_criterion'] # 'id_country', 'id_competition', 

    df_pred['ganancia_unitaria'] = np.where(
        df_pred['acerte'] == 1, 
        df_pred['odd_to_bet'] - 1,  
        -1
        )
    
    X = df_pred[features]
    y = df_pred['ganancia_unitaria']

    # Separo en train y test
    # deberia separar por partidos (para evitar el mismo partido en train y test x tener ≠ modelos) --> sino es DATA LEAKAGE
    unique_partidos = df_pred.index.unique()  

    # 2. Dividir los partidos (no las filas) en train/test
    partidos_train, partidos_test = train_test_split(unique_partidos, test_size=0.2, random_state=42)

    # 3. Seleccionar filas por índice (que es el id_partido)
    df_train = df_pred.loc[df_pred.index.isin(partidos_train)]
    df_test = df_pred.loc[df_pred.index.isin(partidos_test)]
    print(df_train.shape, df_test.shape)
    print(len(df_train.index.unique()), len(df_test.index.unique()))


    X_train = df_train[features]
    y_train = df_train['ganancia_unitaria']
    X_test = df_test[features]
    y_test = df_test['ganancia_unitaria']

    # Entreno modelo
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Predicción de stake
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}, R²: {r2:.4f}")  # Evalúa precisión y generalización    

    # Crear un DataFrame con las predicciones y valores reales
    df_results = pd.DataFrame({
        'y_real': y_test,  # Valores reales
        'y_pred': y_pred   # Predicciones del modelo
    })

    # Guardar Excel para análisis posterior
    df_results.to_excel('./data/model_predictions.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    l_countries = [48, 55, 59, 77, 148] 
    l_countries = [48] 
    one_model = True
    assess, date_assess = False, '2025-04-29' # datetime.datetime.now().date() 

    df_best_models = pd.read_excel("./data/df_best_models.xlsx")

    d_countries = {
        48: ["england"],
        55: ["france"], 
        59: ["germany"],
        77: ["italy"],
        148: ["spain"]
        }
        
    for id_country in l_countries:
        country = d_countries[id_country][0]
        
        row = df_best_models[df_best_models['id_country'] == id_country]
        iteration_date_dt = row['iteration_date'].values[0]
        iteration_date = pd.to_datetime(iteration_date_dt, format='%Y-%m-%d').date()
       
        # Levanto iteraciones del pais
        df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/best_model/3_bet_strategy/df_ite_bs.xlsx')
        df_ite = df_ite.head(1)

        df_ct = pd.DataFrame()

        # Leo predicciones
        for idx, row_ in df_ite.iterrows():

            n_model = int(row_['n_iteration'])
            model_name = str(row_['model_name'])
            print(f"N_model: {n_model} Iteration date: {iteration_date}")

            # Levanto df_test
            df_pred_test = read_predictions(country, iteration_date, n_model, model_name, assess=assess, date_assess=date_assess)
            logger.info(df_pred_test.shape)

            df_ct = pd.concat([df_ct, df_pred_test], axis=0)

        print(df_ct)
        main(df_ct)