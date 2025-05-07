
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
        predict_missing: bool = True,
        concat_with_test: bool = True,
        export: bool = True
):
    date = datetime.datetime.now().date()
    path = f"data/{country}/p4_modeling/{iteration_date}/best_model/2_assess/{date}"
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
        
        if predict_missing:
            logger.warning("Se estan concatenando las predicciones de TEST y ASSESS...")

            # 1. Predict missing
            try:
                df_pred_missing = pd.read_excel(f'{path}/{n_model}__{model_name}_predicciones.xlsx', index_col=0)
            except FileNotFoundError:
                df_pred_missing = assess_model_in_prod(id_country, iteration_date, n_model, model_name)
                
                if export:
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
                    
        else:
            df_pred = pd.read_excel(f'{path}/{n_model}__{model_name}_predicciones.xlsx', index_col=0)

        # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = asses_model.drop_old_metrics(df_pred)

        # 📌 Aplicar estrategia "sin_ea"
        d_params = bs.define_hiperparameters(strategy='train')
        df_pred_met, d_rois = bs.calculate_roi_in_combination(df_pred, d_params)
        
        ## Calculo metricas
        d_metric_sin_ea = asses_model.calculate_metrics(df_pred_met, var_resp='result')
        d_metric_sin_ea_ex = asses_model.calculate_metrics(df_pred_met, var_resp='expected_result', prefix='expected_')
            
        # Guardo datos
        new_row = {'n_model': n_model, 'model_name': model_name, **d_rois, **d_metric_sin_ea, **d_metric_sin_ea_ex}
        rows.append(new_row)
        df_ite_bs = pd.DataFrame(data=rows)

        # Exporto datos
        if export:
            df_pred_met.to_excel(f"{path}/{n_model}__{model_name}_predicciones_met.xlsx", index=True) # Cuando haces assess
            df_ite_bs.to_excel(f'{path}/df_ite_bs.xlsx', index=False)
        else:
            df_pred_met.to_excel(f"{path}/{n_model}__{model_name}_test_assess_.xlsx", index=True)
    
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
    one_model, n_model = False, 328
    n_models = 10
    concat_with_test = False

    d_countries = {
        48: ["england", '2025-05-07'],
        55: ["france", '2025-05-07'], 
        59: ["germany", '2025-05-07'],
        77: ["italy", '2025-05-07'],
        148: ["spain", '2025-05-07']
        }
    
    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]

        # Levanto df_ite
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/best_model/3_bet_strategy/df_ite_bs.xlsx") # pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
     
        if one_model:

            df_ite = df_ite[df_ite['n_iteration'].isin([n_model])]
    
            df_ite_bs, df_pred_met = assess_models_in_prod(
                df_ite=df_ite,
                id_country=id_country, 
                country=country, 
                iteration_date=iteration_date, 
                concat_with_test=concat_with_test,
                export=False
                )
            
            df_ite_bs.to_excel(f'/Users/nachomondino/Desktop/metrics_{n_model}.xlsx')
            df_pred_met.to_excel(f'/Users/nachomondino/Desktop/{n_model}.xlsx')

        else:
            df_ite = df_ite.head(n_models)
            print(df_ite)

            df_ite_bs = assess_models_in_prod(
                df_ite=df_ite,
                id_country=id_country, 
                country=country, 
                iteration_date=iteration_date, 
                concat_with_test=concat_with_test,
                )
    
            df_ite_bs.to_excel(f"data/{country}/p4_modeling/{iteration_date}/best_model/2_assess/df_ite_bs.xlsx")