import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
from utils.set_up_logging import logger
from p6_deployment import main_next_matches
import os
from main import Modeling

def predict_missing_data(n_model, model_name, id_country, country, iteration_date, verbose: int = 0):
    """
    No uso mnm porque el enfoque de predecir missing es como el de test y train en el entrenamiento y no como el de prod (en cuando a rellenar y construir con ultimos partidos)
    """
    logger.critical(f"n_model: {n_model} model_name: {model_name} iteration_date: {iteration_date}")

    # Creo objeto de clase
    iteration_date_dt = pd.to_datetime(iteration_date, format='%Y-%m-%d').date()  # con .date() saco hora y minutos
    lo = main_next_matches.TrainingDataLoader(country=country, n_model=n_model, model_name=model_name, iteration_date=iteration_date_dt)
    dp = main_next_matches.DataPreparationNew(id_country, country, iteration_date)
    mo = Modeling(country=country, date=iteration_date_dt) # Creo objeto de clase DataPreparation

    # Determino con que partidos entrené (puede haber missing) 
    df_match_train = pd.read_excel(f"./data/{country}/p3_data_preparation/{iteration_date}/df_integrated.xlsx", index_col=0)
    idxs_train = df_match_train.index

    # Levanto hiperparametros y modelos utilizados en los datos con los que se entreno el modelo
    d_hiper = lo.load_data_preparation_hyperparameters()
    df_etiquetas = lo.load_df_etiquetas()
    scaler, columns_scaled = lo.load_scaler_model()


    # DATA PREP
    # Levanto datasets
    df_match = pd.read_excel(f'./data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx', index_col=0)
    df_match_odds = pd.read_excel(f'./data/{country}/p6_deployment/missing/data_understanding/all/df_match_odds_miss.xlsx', index_col=0)
    df_integrated_updated = pd.read_excel(f'./data/{country}/p6_deployment/missing/old_updated/df_integrated.xlsx', index_col=0)

    # Filtro df para no construir todo el set por tiempo y por competencia
    ## Tiempo
    df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') 
    initial_date = df_match['date'].min()  # Obtiene la fecha mínima (para filtrar dfs para rellenar y construir) 
    df_integrated_updated_filt = main_next_matches.filter_dataframe_by_date(df_integrated_updated, initial_date=initial_date, n_days=d_hiper['n_years_h2h'] * 365 + 30)    # Dejo los missing con los que NO entrené (serian como mis "next matches") ---> ACA DEBERIA PONER MISSING + N_DAYS ANTES.    
    ## Competencia
    df_integrated_updated_filt = df_integrated_updated_filt[df_integrated_updated_filt['id_competition'].isin(d_hiper['comp_to_select'])]
    logger.warning(df_integrated_updated_filt.shape)
    # df_integrated_updated_filt.to_excel('/Users/nachomondino/Desktop/1.xlsx')

    # Construccion de datos (no hace falta fill_data pues ya tengo las formaciones)
    dp.determine_stats_to_use()
    df = dp.construct_data(df_integrated_updated_filt, n_last_matches=d_hiper['n_last_matches'], n_years_h2h=d_hiper['n_years_h2h'], segun_localia=d_hiper['segun_localia'], calculate_dif=d_hiper['calculate_dif'], prod=False, export=False)    
    # df.to_excel('/Users/nachomondino/Desktop/2.xlsx')

    # Selecciono solo los partidos missing para predecir
    df = df[~df.index.isin(idxs_train)]
    # df.to_excel('/Users/nachomondino/Desktop/3.xlsx')

    # Preparacion desde tag a treat_nan (--> como en prod pues debo usar lo que ya tengo del entrenamiento)
    df = dp.tag_string_data_to_integer_new(df, df_etiquetas, columns_scaled=columns_scaled)
    df, df_fill = dp.clean_data_2_new(df=df, scaler_loaded=scaler, columns_scaled=columns_scaled, comp_to_select=d_hiper['comp_to_select'], columns_selected=d_hiper['selected_columns']) # Antes usaba comp_to_select pero me quedaban los partidos de todas las comp en predicciones.xlsx
    df = dp.select_data_new(df, d_hiper['selected_columns'])
    # df.to_excel('/Users/nachomondino/Desktop/4.xlsx')

    # MODELING     
    # Predigo
    df_predicciones = mo.assess_model(model=lo.load_model(), X_test=df, y_test=None, df_match=df_match, df_match_odds=df_match_odds, df_filled=df_fill, prod=True)

    if not isinstance(df_predicciones, pd.DataFrame):
        raise ValueError("No se generó un dataframe.")

    elif len(df_predicciones) == 0:
        logger.warning("No hay predicciones de partidos missing...")
        raise ValueError
    
    if verbose > 1:
        df_predicciones.to_excel('/Users/nachomondino/Desktop/df_pred_missing.xlsx')

    return df_predicciones

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [48]

    d_countries = {
        # Train nuevos
        6: ["argentina", '2025-02-06'], 
        48: ["england", '2025-02-05'],
        55: ["france", '2025-02-05'], 
        59: ["germany", '2025-02-05'],
        77: ["italy", '2025-02-05'],
        148: ["spain", '2025-02-05'], 
        }

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]

        predict_missing_data(
            n_model=145, model_name='LogisticRegression',
            id_country=id_country, country=country, iteration_date=iteration_date, 
            )
        