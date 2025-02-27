
import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import p4_modeling.main_select_model as msm

def main(id_country, country, iteration_date, n_max_candidates: int = 30):
    """
    Automatizo el experimento para definir metricas segun correlacion con ROI prod y Ex ROI prod
    """
    # 1: Levanto df_ite_test (test)
    df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
    print(df_ite)

    # 2: Determino candidatos? y metricas ?
    df_ite_test = msm.main(
        df_ite=df_ite,
        id_country=id_country, country=country, iteration_date=iteration_date,
        select_candidates=True,
        n_max_candidates=n_max_candidates,
        assess=False,
        update_missing=False,
        predict_missing=False,
        select_model=False,
        export=False
        )
    df_ite_test.to_excel(f'/Users/nachomondino/Desktop/df_ite_test.xlsx', index=False)

    # 2: Levanto df_ite_bs (assess)
    df_ite_assess = msm.main(
        df_ite=df_ite,
        id_country=id_country, country=country, iteration_date=iteration_date, 
        select_candidates=True,
        n_max_candidates=n_max_candidates,
        assess=True,
        update_missing=False,
        predict_missing=True,
        select_model=False,
        export=True
        )
    df_ite_assess.to_excel(f'/Users/nachomondino/Desktop/df_ite_assess.xlsx', index=False)
    
    # 3: calcular ROIs en partidos assess --> ya lo calculo en df_ite_assess
    roi_assess, ex_roi_assess = 'roi_sin_ea_last_50', 'ex_roi_sin_ea_last_50'

    # 4: Concatenar df_ite_test con ROIs en assess
    df_ite_assess = df_ite_assess.rename(columns={'n_model': 'n_iteration'})
    df_ite_assess_filt = df_ite_assess[['n_iteration', 'roi', 'expected_roi', roi_assess, ex_roi_assess]]
    df_final = pd.merge(df_ite_test, df_ite_assess_filt, on='n_iteration', how='outer')  
    df_final.to_excel(f'/Users/nachomondino/Desktop/df_final.xlsx', index=False)
    
    # 5: Calcular correlacion entre metricas de df_ite_test y ROIs en assess
    df_corr = pd.DataFrame()

    for col in df_ite_test.columns:

        try:
            corr_roi = df_final[col].corr(df_final[roi_assess])
            corr_ex_roi = df_final[col].corr(df_final[ex_roi_assess])   

            df_corr.loc[col, 'corr_roi'] = corr_roi
            df_corr.loc[col, 'corr_ex_roi'] = corr_ex_roi

        except:
            pass

    # 6: Definir metricas con mayor correlacion con ROIs
    df_corr = df_corr.sort_values(by='corr_roi', ascending=False)
    df_corr.to_excel(f'/Users/nachomondino/Desktop/df_corr.xlsx', index=True)


if __name__ == "__main__":
    # Defino parametros
    l_countries = [55, 59, 77, 148]
    
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

        main(id_country, country, iteration_date)



'''
## (0) Determino metricas para seleccionar modelos candidatos (1) y modelos en prod (3)
# expected_weight = df_ite['roi'].corr(df_ite['expected_roi'])
# roi_weight = 1 - expected_weight
# l_metrics = ['roi_con_ea', 'roi_con_ea_last_50', 'ex_roi_con_ea', 'ex_roi_con_ea_last_50'] # 50 y/o 25?
# l_weights = [roi_weight/2, roi_weight/2, expected_weight/2, expected_weight/2]
# logger.info(f'ROI weight: {roi_weight} Expected Weight: {expected_weight}')

# l_metrics = ['ex_roi_con_ea', 'ex_roi_con_ea_last_50'] # ROI test+assess poca corr con roi en prod. Encima usa ea mas agresiva.
# l_metrics = ['expected_roi', 'ex_roi_last_50'] # ROI test+assess poca corr con roi en prod. Encima usa ea mas agresiva.
# l_weights = [0.5, 0.5]
'''