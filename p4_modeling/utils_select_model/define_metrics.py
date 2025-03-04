
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import p4_modeling.main_select_model as msm
from p4_modeling import betting_strategy
from tqdm import tqdm


def main(df_ite, country, iteration_date, n_matches_test: int = 50, n_matches_prod: int = 50):
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
        df_first_matches = df_pred.head(n_matches_test)
        df_last_matches = df_pred.tail(len(df_pred)-n_matches_test).head(n_matches_prod)
        # print(df_first_matches.shape, df_last_matches.shape)

        # Aplico estrategia
        d_params_sin_ea = bs.define_hiperparameters(strategy='train') 
        d_params_con_ea = bs.define_hiperparameters(strategy='kelly', vary_dp=False)

        # Aplico estrategia a "TEST" (first_matches)
        # 📌 Aplicar estrategia a first_matches ("test") sin ea 
        df_strat, df_pred_fm_met_sin_ea = bs.define_model_betting_strategy(df_first_matches, d_params=d_params_sin_ea)
        d_metrics_test_sin_ea = msm.calculate_all_metrics(df_pred_fm_met_sin_ea, suffix='sin_ea')
    
        # 📌 Aplicar estrategia a first_matches ("test") con ea --> con ea pues selecciono con metricas con ea
        df_strat, df_pred_fm_met = bs.define_model_betting_strategy(df_first_matches, d_params=d_params_con_ea) # no por res porque 25 part es muy poco...
        d_metrics_test_con_ea = msm.calculate_all_metrics(df_pred_fm_met, suffix='con_ea')

        # Aplico estrategia a "PROD" o "ASSESS" (last_matches) --> con o sin ea? Sin ea Por que con ea no? con ea creo que no tiene sentido porque no usa la ea del test sino que elige una nueva...
        df_strat, df_pred_lm_met_sin_ea = bs.define_model_betting_strategy(df_last_matches, d_params=d_params_sin_ea)
        d_metrics_prod_sin_ea = msm.calculate_all_metrics(df_pred_lm_met_sin_ea, suffix='prod_sin_ea')

        # Guardo métricas del modelo en un solo diccionario
        row_dict = {
            "n_model": n_model,
            "model_name": model_name,
            **d_metrics_test_sin_ea,  # Métricas de test sin ea
            **d_metrics_test_con_ea,  # Métricas de test con ea
            **d_metrics_prod_sin_ea,  # Agrego métricas de producción (prod)
        }

        # Agrego la fila a la lista
        rows.append(row_dict)
        progress_bar.update(1)

    progress_bar.close()
    # Convertir la lista de diccionarios en un DataFrame
    df_ite = pd.DataFrame(data=rows)

    # Columnas de ROI de "PROD" o "ASSESS"
    roi_col, ex_roi_col = 'roi_prod_sin_ea', 'expected_roi_prod_sin_ea'

    # Crear un DataFrame vacío para la correlación
    d_metrics_test_sin_ea.update(d_metrics_test_con_ea)
    d_metrics_test = d_metrics_test_sin_ea
    df_corr = pd.DataFrame(index=d_metrics_test.keys(), columns=['corr_roi_sin_ea', 'corr_ex_roi_sin_ea']) 

    # Calculo correlación entre métricas de init y prod con los ROIs
    for col in d_metrics_test.keys():
        if col in df_ite.columns:  # Verifico que la columna exista en df_ite
            df_corr.loc[col, 'corr_roi_sin_ea'] = df_ite[col].corr(df_ite[roi_col])
            df_corr.loc[col, 'corr_ex_roi_sin_ea'] = df_ite[col].corr(df_ite[ex_roi_col])

    # Ordeno por correlación con ROI
    df_corr = df_corr.sort_values(by='corr_roi_sin_ea', ascending=False)

    # Exporto a Excel
    # df_pred_fm_met.to_excel('/Users/nachomondino/Desktop/df_init.xlsx', index=True)
    # df_pred_lm_met.to_excel('/Users/nachomondino/Desktop/df_prod.xlsx', index=True)
    df_ite.to_excel(f'/Users/nachomondino/Desktop/{country}/df_ite.xlsx', index=True)
    df_corr.to_excel(f'/Users/nachomondino/Desktop/{country}/df_corr.xlsx', index=True)


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

        main(df_ite, country, iteration_date)