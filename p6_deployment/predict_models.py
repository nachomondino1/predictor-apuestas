# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p6_deployment.main_next_matches import main
import datetime


def predict_models(l_countries, n_models_predict: int = 5):
    """
    Para obtener predicciones en prox partidos de los modelos candidatos
    """       
    d_run = {'run_missing': False, 'data_unders': False, 'data_prep': True, 'modeling': True, 'export': False}  

    # Defino country, iteration date y modelo
    d_countries = {
        48: ["england", '2025-08-26'], 
        55: ["france", '2025-08-26'], 
        59: ["germany", '2025-08-26'], 
        77: ["italy", '2025-08-26'],
        148: ["spain", '2025-08-26'], 
        }
    
    # Por pais
    for id_country in l_countries: 

        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        df_country = pd.DataFrame()

        # Determino mejores n modelos
        df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/best_model/3_bet_strategy/df_ite_bs.xlsx')
        df_best_models = df_ite.head(n_models_predict)
        print(df_ite.head(n_models_predict))

        # Por modelo
        for idx, row in df_best_models.iterrows():

            col_name = 'n_iteration' if 'n_iteration' in df_best_models.columns else 'n_model'
            n_model = row[col_name]
            model_name = row['model_name']
            print(n_model, model_name)
            d_model = {'n_model': n_model, 'model_name': model_name}

            # Obtengo predicciones en proximos partidos
            df = main(d_run, id_country, iteration_date=iteration_date, d_model=d_model, export=False) 

            # Selecciono id_match y predicted_result
            df_filt = df.loc[:, ['id_team_home', 'id_team_away', 'odds_home', 'odds_draw', 'odds_away', 'predicted_result']]
            df_filt.rename(columns={'predicted_result': f'{n_model}_{model_name}'}, inplace=True)
            new_cols = [col for col in df_filt.columns if col not in df_country.columns]

            # Concateno predicciones de modelos
            df_country = pd.concat([df_country, df_filt[new_cols]], axis=1)
            df_country.to_excel(f"/Users/nachomondino/Desktop/df_{country}.xlsx")

        # Luego contamos cuántas veces aparece cada valor
        cols_to_check = df_country.columns
        for val in [0, 1, 2]:
            df_country[str(val)] = df_country[cols_to_check].apply(lambda row: sum(row == val), axis=1)

        df_country.to_excel(f"/Users/nachomondino/Desktop/df_{country}.xlsx")

if __name__ == "__main__":    

    # Defino hiperparametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [48, 59, 77, 148]
    n_models_predict=10

    predict_models(l_countries, n_models_predict=n_models_predict)