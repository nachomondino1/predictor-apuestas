
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
from utils.set_up_logging import logger
from p3_data_preparation import construct_data
from p4_modeling import betting_strategy, asses_model
from p6_deployment import main_next_matches
from utils import directories
import datetime

def assess_models_in_prod(
        df_ite,
        id_country,
        country,
        iteration_date,
        update_missing: bool = False,
        concat_with_test: bool = False,
        xlsx_name: str = 'df_ite_bs.xlsx',
        export: bool = True,
        verbose: int = 1
):
    date = datetime.datetime.now().date()
    path = f"data/{country}/p4_modeling/{iteration_date}/best_model/2_assess/{date}" # _con_ea
    directories.make_directories(l_directorios=[path])    
    bs = betting_strategy.BettingStrategy(country, iteration_date, verbose=0)
    rows = []

    # Actualizo missing (1 sola vez para todos los modelos)
    if update_missing:
        d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True} 
        main_next_matches.main(d_run, id_country, iteration_date=iteration_date, export=d_run['export']) 

    # Por modelo
    for idx, row in df_ite.iterrows():
        col1 = 'n_model' if 'n_model' in df_ite.columns else 'n_iteration'
        col2 = 'model_name' if 'model_name' in df_ite.columns else 'model_name_x'
        n_model, model_name = row[col1], row[col2]
        logger.info(f'{n_model} {model_name}')
    
        # 1. Predict missing
        try:
            df_pred_missing = pd.read_excel(f'{path}/{n_model}__{model_name}_predicciones.xlsx', index_col=0)
        except FileNotFoundError:
            df_pred_missing = assess_model_in_prod(id_country, iteration_date, n_model, model_name)
            if export: # Exportamos para no tener que volver a hacer el assess
                df_pred_missing.to_excel(f"{path}/{n_model}__{model_name}_predicciones.xlsx", index=True) # Cuando haces assess

        # 2. Concat test + missing 
        if concat_with_test:
            # Levanto predicciones del modelo (test o test + assess)
            path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
            df_pred_test = pd.read_excel(path_test, index_col=0)
            df_pred = pd.concat([df_pred_test, df_pred_missing], axis=0)
        else:
            df_pred = df_pred_missing.copy()

        # 3. Agrego columnas 'result' y 'expected_result' --> Lo podria implementar en betting strategy no?
        df_pred = construct_data.determine_result(df_pred) # Intento hacerlo antes con df_match pero rompia.
        df_pred = construct_data.determine_expected_result(df_pred) # Intento hacerlo antes con df_match pero rompia.
                    
        # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = asses_model.drop_old_metrics(df_pred)

        # 📌 Aplicar estrategia "sin_ea" --> predict_missing ya tiene la estrategia aplicada cuando corri mnm.py. Solo seria para test que tiene la de train.
        # d_params = bs.define_hiperparameters(strategy='train')
        # d_params = {'prob_dp': None, 'curva': 'kelly', 'm': 5, 'b': 0, 'k': 10} 
        # df_pred = bs.apply_strategy(df_pred, d_params)

        ## Calculo metricas (precision, f1_score, etc)
        df_pred_met, d_rois = bs.calculate_roi_in_combination(df_pred) # Ver si hago solo el yield en vez del ROI --> Deberia separar la aplicacion de la strategia del calculo del roi
        d_metric_sin_ea = asses_model.calculate_metrics(df_pred_met, var_resp='result')
        d_metric_sin_ea_ex = asses_model.calculate_metrics(df_pred_met, var_resp='expected_result', prefix='expected_')

        # Guardo datos
        new_row = {'n_model': n_model, 'model_name': model_name, 'n_reg_assess': len(df_pred_met),  **d_rois, **d_metric_sin_ea, **d_metric_sin_ea_ex}
        rows.append(new_row)
        df_ite_bs = pd.DataFrame(data=rows)

        # Exporto datos
        if export:
            path1 = f"{path}/{n_model}__{model_name}_predicciones_met.xlsx"
            path2 = f"{path}/{n_model}__{model_name}_test_assess_.xlsx"
            path_final = path2 if concat_with_test else path1
            
            df_pred_met.to_excel(path_final, index=True) # Cuando haces assess
            df_ite_bs.to_excel(f'{path}/bis_{xlsx_name}.xlsx', index=False)
    
    return df_ite_bs

def assess_model_in_prod(id_country, iteration_date, n_model, model_name):
    """
    Assess de un modelo en especifico
    """
    # Parameters
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': True} 
    d_model = {'n_model': n_model, 'model_name': model_name}

    # Obtengo predicciones en partidos "missing"
    df_pred_missing = main_next_matches.main(d_run, id_country, iteration_date=iteration_date, predict_missing=True, d_model=d_model, export=False) 
            
    if len(df_pred_missing) == 0:
        logger.error("No hay partidos missing que predecir. Entrenaste con los ultimos missing.")
        raise ValueError
    
    return df_pred_missing

if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [48]
    one_model = False
    n_models = 10

    # Levanto df_best_models
    df_best_models = pd.read_excel('data/df_best_models.xlsx')
    
    # Por pais
    for id_country in l_countries:

        country = df_best_models['country'][df_best_models['id_country'] == id_country].values[0]
        iteration_date = df_best_models['iteration_date'][df_best_models['id_country'] == id_country].values[0]
        iteration_date_dt = pd.to_datetime(iteration_date, format='%Y-%m-%d').date()  # con .date() saco hora y minutos
        print(f"--- {country} {iteration_date_dt} ---")

        # Levanto df_ite
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date_dt}/best_model/3_bet_strategy/df_ite_bs.xlsx") # pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
     
        if one_model:

            # Eligo el modelo de prod
            n_model = df_best_models['n_model'][df_best_models['id_country'] == id_country].values[0]
            model_name = df_best_models['model_name'][df_best_models['id_country'] == id_country].values[0]
            print(f"--- {n_model} {model_name} ---")
            
            # Filtro df_ite
            df_ite = df_ite[df_ite['n_iteration'].isin([n_model])]
            df_ite = df_ite[df_ite['model_name'].isin([model_name])]
    
            df_ite_bs = assess_models_in_prod( # df_ite_bs, df_pred_met
                df_ite=df_ite,
                id_country=id_country, 
                country=country, 
                iteration_date=iteration_date_dt, 
                export=True
                )
            
            df_ite_bs.to_excel(f'data/{country}/p4_modeling/metrics_{n_model}_{model_name}.xlsx')

        else:
            df_ite = df_ite.head(n_models)
            print(df_ite)

            xlsx_name = 'expected_f1_score'

            df_ite_bs = assess_models_in_prod(
                df_ite=df_ite,
                id_country=id_country, 
                country=country, 
                iteration_date=iteration_date_dt, 
                xlsx_name=xlsx_name
                )
    
            df_ite_bs.to_excel(f"data/{country}/p4_modeling/{iteration_date_dt}/best_model/2_assess/{xlsx_name}.xlsx")