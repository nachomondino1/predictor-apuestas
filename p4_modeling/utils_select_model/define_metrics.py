
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from p3_data_preparation.select_data import delete_correlated_columns
import p4_modeling.main_select_model as msm
from p4_modeling import betting_strategy, asses_model
from tqdm import tqdm


def determine_metrics_by_model(df_ite, country, iteration_date, n_matches_test: int = 50):
    """
    Automatizo el experimento para definir metricas segun correlacion con ROI prod y Ex ROI prod.
    Es un experimento retroactivo. Tengo el ROI de cada modelos en los partidos futuros y veo como seleccionar a los modelos que mejor les fue.
    """
    rows = []
    bs = betting_strategy.BettingStrategy(country, iteration_date, verbose=0)

    progress_bar = tqdm(total=len(df_ite), ncols=80)  # Inicializo barra de progreso

    # Itero sobre cada modelo
    for idx, row in df_ite.iterrows():

        n_model, model_name = row['n_iteration'], row['model_name']
        # print(n_model)

        # Obtengo predicciones
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred = pd.read_excel(path_test, index_col=0)
        # print(df_pred.shape)

        # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = msm.drop_old_metrics(df_pred)

        # Separo 75% como test y 25% como ROI en prod
        n_matches_prod = len(df_pred) - n_matches_test
        df_first_matches = df_pred.head(n_matches_test)
        df_last_matches = df_pred.tail(n_matches_prod)
        # print(df_first_matches.shape, df_last_matches.shape)

        # Aplico estrategia
        d_params_sin_ea = bs.define_hiperparameters(strategy='train') 

        # Aplico estrategia a "TEST" (first_matches)
        # 📌 Sin ea 
        df_pred_fm_met_sin_ea, _ = bs.calculate_roi_in_combination(df_first_matches, d_params_sin_ea)
        d_metrics_test_sin_ea = asses_model.calculate_metrics(df_pred_fm_met_sin_ea, var_resp='result', advanced_metrics=True)
        d_metrics_test_sin_ea_ex = asses_model.calculate_metrics(df_pred_fm_met_sin_ea, var_resp='expected_result', advanced_metrics=True)

        # Aplico estrategia a "PROD" o "ASSESS" (last_matches) 
        df_pred_lm_met_sin_ea, _ = bs.calculate_roi_in_combination(df_last_matches, d_params_sin_ea)
        d_metrics_prod_sin_ea = asses_model.calculate_metrics(df_pred_lm_met_sin_ea, var_resp='result', advanced_metrics=False)
        d_metrics_prod_sin_ea_ex = asses_model.calculate_metrics(df_pred_lm_met_sin_ea, var_resp='expected_result', advanced_metrics=False)

        # Renombro metricas para evitar sobreescribirlas
        d_metrics_test_sin_ea = asses_model.rename_dict_keys(d_metrics_test_sin_ea)
        d_metrics_test_sin_ea_ex = asses_model.rename_dict_keys(d_metrics_test_sin_ea_ex, prefix="expected_")
        d_metrics_prod_sin_ea = asses_model.rename_dict_keys(d_metrics_prod_sin_ea, suffix='prod')
        d_metrics_prod_sin_ea_ex = asses_model.rename_dict_keys(d_metrics_prod_sin_ea_ex, prefix="expected_", suffix='prod')

        # Guardo métricas del modelo en un solo diccionario
        row_dict = {
            "n_model": n_model,
            "model_name": model_name,
            **d_metrics_test_sin_ea,  # Métricas de test sin ea
            **d_metrics_test_sin_ea_ex,
            **d_metrics_prod_sin_ea,  # Agrego métricas de producción (prod)
            **d_metrics_prod_sin_ea_ex
        }

        # Agrego la fila a la lista
        rows.append(row_dict)
        progress_bar.update(1)

    progress_bar.close()
    df_ite = pd.DataFrame(data=rows)
    merged_dict = {**d_metrics_test_sin_ea, **d_metrics_test_sin_ea_ex}

    df_ite.to_excel(f'/Users/nachomondino/Desktop/{country}/df_ite.xlsx', index=True)

    return df_ite, merged_dict

def calculate_correlation(df_ite, merged_dict):
    """
    Calculo de correlacion de metricas con roi_prod
    """
    # Crear un DataFrame vacío para la correlación
    df_corr = pd.DataFrame(index=merged_dict.keys(), columns=['corr_roi', 'corr_ex_roi']) 

    # Columnas de ROI de "PROD" o "ASSESS"
    roi_col, ex_roi_col = 'roi_prod', 'expected_roi_prod'

    # Calculo correlación entre métricas de init y prod con los ROIs
    for col in merged_dict.keys():
        if col in df_ite.columns:  # Verifico que la columna exista en df_ite
            df_corr.loc[col, 'corr_roi'] = df_ite[col].corr(df_ite[roi_col])
            df_corr.loc[col, 'corr_ex_roi'] = df_ite[col].corr(df_ite[ex_roi_col])

    # Ordeno por correlación con ROI
    df_corr = df_corr.sort_values(by='corr_roi', ascending=False)

    # Exporto a Excel
    df_corr.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr.xlsx', index=True)
    return df_corr

def eliminate_corr_metrics(df_ite):

    # Determino columnas a eliminar
    l_strings_to_avoid = ["_prod", '_last_', '_filled_', '%_gp_', '_bm_', 'dif_']
    cols_to_avoid = [col for col in df_ite.columns if any(substring in col for substring in l_strings_to_avoid)]

    # Evito eliminar ciertas columnas
    cols_to_keep = ['roi_prod']   
    cols_to_avoid = [col for col in cols_to_avoid if col not in cols_to_keep]

    # Filtro columns (solo float y solo las que quiero)
    df_ite.drop(columns=cols_to_avoid, inplace=True)
    print(df_ite.shape)
    
    # Determino correlacion
    l_cols_to_elim, df = delete_correlated_columns(df_ite, var_resp='roi_prod', verbose=0)
    cols_selected = [col for col in df_ite.columns if col not in l_cols_to_elim]
    cols_selected.remove('roi_prod')
    
    # df_corr_y.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr_y.xlsx')
    df.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr_metrics.xlsx')
    return cols_selected

def main(df_ite, calc: bool = True):

    if calc:
        # 1. Calculo metricas por modelo
        df_ite, metrics = determine_metrics_by_model(df_ite, country, iteration_date, n_matches_test=50)

        # 2. Calculo correlacion de metricas y roi_prod
        df_corr = calculate_correlation(df_ite, metrics)
    else:
        df_ite = pd.read_excel(f"/Users/nachomondino/Desktop/{country}/df_ite.xlsx")
        df_corr = pd.read_excel(f"/Users/nachomondino/Desktop/{country}/df_corr.xlsx", index_col=0)

    # 3. Seleccionar las metricas de mayor correlacion
    thr = df_corr['corr_roi'].quantile(0.75)
    l_cols = list(df_corr[df_corr['corr_roi'] >= thr].index)
    l_cols.append('roi_prod')
    df_ite_filt = df_ite.loc[:, l_cols]
    print(thr, l_cols)

    # 4. Elimino metricas x correlacion
    metrics_selected = eliminate_corr_metrics(df_ite_filt)            
    d = {metric: df_corr.loc[metric, 'corr_roi'] for metric in metrics_selected}
    sum_peso = sum(d.values())

    for key, value in d.items():
        print(key, value / sum_peso)
            
if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]

    d_countries = {
        # train viejos
        48: ["england", '2025-02-05'],
        55: ["france", '2025-02-05'], 
        59: ["germany", '2025-02-05'],
        77: ["italy", '2025-02-05'],
        148: ["spain", '2025-02-05'], 
        # Train nuevos
        # 6: ["argentina", '2025-02-06'], 
        # 48: ["england", '2025-03-03'],
        # 55: ["france", '2025-03-03'], 
        # 59: ["germany", '2025-03-04'],
        # 77: ["italy", '2025-03-04'],
        # 148: ["spain", '2025-03-04'], 
        }
    
    for id_country in l_countries:

        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        
        # 1: Levanto df_ite_test (test) --> NUNCA REDUCIR EL NRO DE MODELOS PUES LOS RDOS PUEDEN SER MUY ≠ A LOS QUE REALMENTE SON.
        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
        print(df_ite)

        main(df_ite)