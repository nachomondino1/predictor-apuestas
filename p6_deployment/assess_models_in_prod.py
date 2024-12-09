import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import os
from utils.directories import make_directories
import main_next_matches
from p3_data_preparation.construct_data import determine_result, determine_expected_result
from p4_modeling.asses_model import determine_winning_bets, calculate_roi

# El objetivo es evaluar las predicciones de los mejores modelos de un pais en los ultimos partidos jugados sin tener que hacerlo manualmente.

# Defino parametros
id_country = 148
n_models = 20
d_countries = {48: ['england', "2024-12-04"], 55: ['france', "2024-12-05"], 59: ['germany', "2024-12-05"], 77: ['italy', "2024-12-04"], 148: ['spain', "2024-12-03"]}
d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': False}  # No puedo correr data_unders = False si los proximos partidos ya estan en missing.
directorio = os.getenv('BASE_DIR_LOCAL')
verbose = 0

# Defino variables
country = d_countries[id_country][0]
iteration_date = d_countries[id_country][1]
print(country, iteration_date)
df_final = pd.DataFrame()

path_country = f'{directorio}/{country}/{iteration_date}'
make_directories(l_directorios=[path_country])


# Levantar los mejores x modelos del pais
df_iteration = pd.read_excel(f"./data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
if verbose >= 1:
    print(df_iteration)

# Determino modelos a probar
df_iteration = df_iteration.sort_values(by='roi_por_partido', ascending=False)  # Ordenar por roi por partido decreciente
df_iteration_filt = df_iteration.head(n_models) # Seleccionar los primeros n registros
if verbose >= 1:
    print(df_iteration_filt)

# Levanto resultados
df_match_miss = pd.read_excel(f"./data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx", index_col=0) 
if verbose >= 1:
    print("Df_match_miss:", df_match_miss)


# Por pais
# for id_country in l_countries:
for idx, row in df_iteration_filt.iterrows():

    d_model = {'n_model': row['n_iteration'], 'model_name': row['model_name'], 'iteration_date': iteration_date} # XGBClassifier, neural_networ, SVC, LogisticRegression, MLPClassifier
    print(d_model)

    # Usar mnm.py con predict_missing=True y data_unders=False.
    df = main_next_matches.main(d_run, id_country, d_model=d_model, predict_missing=True, export=d_run['export']) # Probar un modelo

    # Determino resultado
    df_match_miss_filt = df_match_miss[df_match_miss.index.isin(df.index)]
    df_match_miss_filt = determine_result(df_match_miss_filt)
    df_match_miss_filt = determine_expected_result(df_match_miss_filt)
    if verbose >= 1:
        print("A", df_match_miss_filt)

    df = pd.concat([df, df_match_miss_filt.loc[:, ['goals_home', 'goals_away', 'result', 'expected_goals_(xg)_home', 'expected_goals_(xg)_away', 'expected_result']]], axis=1)
    if verbose >= 1:
        print("B", df)
        print("C", df.index)

    # Calculo metricas
    ## Expected ROI
    df = determine_winning_bets(df, name_extension='expected_')
    df_predicciones, d_metrics = calculate_roi(df, name_extension='expected_')
    ## ROI
    df = determine_winning_bets(df)
    df_predicciones_2, d_metrics_2 = calculate_roi(df)     # df_predicciones, d_metrics = mo.calculate_metrics(df_pred_proba=df, retrain=True)  # Ya tengo definida la estrategia de apuesta... Solo me queda calcular el ROI.

    missing_columns = [col for col in df_predicciones_2.columns if col not in df_predicciones.columns]
    df_pred = pd.concat([df_predicciones, df_predicciones_2[missing_columns]], axis=1) # Concatenar únicamente las columnas que faltan

    # Guardo metricas del modelo
    d_pred = {'n_model': row['n_iteration']} 
    d_pred.update(d_metrics)
    d_pred.update(d_metrics_2)
    df_metrics = pd.DataFrame(data=d_pred, index=[0])
    df_final = pd.concat([df_final, df_metrics], axis=0)

    # Export
    df_pred.to_excel(f"{path_country}/predicciones_{d_model['n_model']}_{d_model['model_name']}.xlsx") # df_predicciones???
    df_final.to_excel(f"{path_country}/df_final.xlsx")  #Por seguridad

    if verbose >= 1:
        print(d_metrics)
        print(df_predicciones)
        print()
        print()


df_final.to_excel(f"{path_country}/df_final.xlsx")
