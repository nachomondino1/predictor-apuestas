
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import p4_modeling.main_select_model as msm
from p4_modeling import betting_strategy

def main(df_ite, country, iteration_date, num_matches: int = 25):
    """
    Automatizo el experimento para definir metricas segun correlacion con ROI prod y Ex ROI prod.
    Es un experimento retroactivo. Tengo el ROI de cada modelos en los partidos futuros y veo como seleccionar a los modelos que mejor les fue.
    """
    rows = []
    bs = betting_strategy.BettingStrategy(country, iteration_date, verbose=0)

    # Itero sobre cada modelo
    for idx, row in df_ite.iterrows():
        n_model, model_name = row['n_iteration'], row['model_name']
        print(n_model)

        # Obtengo predicciones
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred = pd.read_excel(path_test, index_col=0)
        print(df_pred.shape)

        # Separo 75% como test y 25% como ROI en prod
        df_last_matches = df_pred.tail(num_matches)
        df_first_matches = df_pred.head(len(df_pred) - num_matches)
        print(df_first_matches.shape, df_last_matches.shape)

        # Aplico sin ea o con ea?
        # 📌 Aplicar estrategia a first_matches
        d_params = bs.define_hiperparameters(strategy='train')  
        df_pred_fm_met, _, __ = bs.calculate_roi_in_combinations(df_first_matches, d_params=d_params)
        d_metrics_init = msm.calculate_all_metrics(df_pred_fm_met, suffix='sin_ea', advanced_metrics=True)

        # 📌 Aplicar estrategia a last_matches
        # df_pred_lm_met, _, __ = bs.calculate_roi_in_combinations(df_last_matches, d_params=d_params)
        df_strat, df_pred_lm_met = msm.apply_betting_strategy(df_pred, bs, per_res=True, vary_dp=True, vary_m=True)
        d_metrics_prod = msm.calculate_all_metrics(df_pred_lm_met, suffix='prod')

        # Guardo métricas del modelo en un solo diccionario
        row_dict = {
            "n_model": n_model,
            "model_name": model_name,
            **d_metrics_init,  # Agrego métricas de test (init)
            **d_metrics_prod   # Agrego métricas de producción (prod)
        }

        # Agrego la fila a la lista
        rows.append(row_dict)

    # Convertir la lista de diccionarios en un DataFrame
    df_ite = pd.DataFrame(data=rows)

    # Columnas de ROI
    roi_col, ex_roi_col = 'roi_prod', 'expected_roi_prod'

    # Crear un DataFrame vacío para la correlación
    df_corr = pd.DataFrame(index=d_metrics_init.keys(), columns=['corr_roi', 'corr_ex_roi']) 

    # Calculo correlación entre métricas de init y prod con los ROIs
    for col in d_metrics_init.keys():
        if col in df_ite.columns:  # Verifico que la columna exista en df_ite
            df_corr.loc[col, 'corr_roi'] = df_ite[col].corr(df_ite[roi_col])
            df_corr.loc[col, 'corr_ex_roi'] = df_ite[col].corr(df_ite[ex_roi_col])

    # Ordeno por correlación con ROI
    df_corr = df_corr.sort_values(by='corr_roi', ascending=False)

    # Exporto a Excel
    # df_pred_fm_met.to_excel('/Users/nachomondino/Desktop/df_init.xlsx', index=True)
    # df_pred_lm_met.to_excel('/Users/nachomondino/Desktop/df_prod.xlsx', index=True)
    df_ite.to_excel(f'/Users/nachomondino/Desktop/{country}/df_ite.xlsx', index=True)
    df_corr.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr.xlsx', index=True)


if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    
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

        # 1: Levanto df_ite_test (test)
        df_cand = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
        df_cand = df_cand.sort_values(by='roi', ascending=False)
        df_cand = df_cand.head(100)
        print(df_cand)

        main(df_cand, country, iteration_date)