import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from utils.set_up_logging import logger


def check_preparation(country):
    """
    Verificar diferencia en la preparacion de datos entre assess y produccion. Podria haber diferencia 
    solo en las variables jugadores porque una es con formaciones y la otra sin.
    """
    df_selected_assess = pd.read_excel(f'data/{country}/p6_deployment/data_preparation/df_selected_MISS.xlsx', index_col=0)
    df_selected_prod = pd.read_excel(f'data/{country}/p6_deployment/data_preparation/df_selected.xlsx', index_col=0)

    # Shapes (deben ser iguales)
    print(df_selected_assess)
    print(df_selected_prod)
    print(df_selected_assess.shape, df_selected_prod.shape)

    # df_dif = pd.DataFrame(columns=df_selected_assess.columns, index=df_selected_prod.index)
    df_dif = pd.DataFrame(index=df_selected_prod.index)

    # Por partido
    for idx, row in df_selected_assess.iterrows():

        if idx in df_selected_prod.index:

            # Por columna
            for col in df_selected_assess.columns:

                print(f"Columna: {col}")

                val_assess = df_selected_assess.loc[idx, col]
                val_prod = df_selected_prod.loc[idx, col] # .values[0]
                print(val_assess, val_prod)

                if isinstance(val_assess, (int, float)) and isinstance(val_prod, (int, float)):

                    if round(val_assess, 3) != round(val_prod, 3):
                        logger.error("ERROR! Valor diferente")

                        df_dif.loc[idx, f'{col}_prod'] = val_prod         
                        df_dif.loc[idx, f'{col}_assess'] = val_assess         
                        df_dif.loc[idx, f'dif_{col}'] = val_prod - val_assess         

                else:
                    logger.error(f"En el partido {idx}, la variable {col} no es float o int: {val_assess} {val_prod}")
                    raise ValueError
        else:
            logger.error(f"El partido {idx}, no se encuentra en produccion.")


        # break

    df_dif.to_excel("/Users/nachomondino/Desktop/df_dif_assess_prod.xlsx")


def calculate_dif_proba(country):
    """
    Calcular la diferencia de probabilidad por partido...
    
    Mejoras:
    - Ver si cambia result_to_bet
    - Diferencias probas de assess y prod con extension.
    """

    df_pred_assess = pd.read_excel(f'./data/{country}/p6_deployment/df_predicciones_missing.xlsx', index_col=0)
    try:
        df_pred_prod = pd.read_excel(f'./data/{country}/p6_deployment/predicciones.xlsx', index_col=0)
    except FileNotFoundError:
        df_pred_prod = pd.read_excel(f'./data/{country}/p6_deployment/predicciones_prod.xlsx', index_col=0)

    
    print(df_pred_assess.shape, df_pred_prod.shape)
    print(df_pred_assess)
    print(df_pred_prod)

    l_cols_prob = ['prob_class_1', 'prob_class_0', 'prob_class_2', 'result_to_bet']

    # Definir extensiones para cada DataFrame
    # suffix_prod = '_prod'
    suffix_assess = '_assess'

    # Renombrar las columnas en cada DataFrame
    # df_pred_prod_renamed = df_pred_prod[l_cols_prob].add_suffix(suffix_prod)
    df_pred_assess_renamed = df_pred_assess[l_cols_prob].add_suffix(suffix_assess)

    # Concatenar los DataFrames renombrados
    df_dif = pd.concat([df_pred_prod, df_pred_assess_renamed], axis=1)

    # df_dif = df_pred_assess.loc[:, l_cols_prob]
    # df_dif = pd.concat([df_dif, df_pred_prod[l_cols_prob]], axis=1)
    # df_dif = pd.concat([df_pred_prod[l_cols_prob], df_pred_assess[l_cols_prob]], axis=1)

    # Por partido
    for idx, row in df_pred_prod.iterrows():
        
        if idx in df_pred_assess.index:
            
            for col in l_cols_prob:

                prob_assess = df_pred_assess.loc[idx, col]
                prob_prod = df_pred_prod.loc[idx, col]
                dif_prob = prob_assess - prob_prod
                print(prob_assess, prob_prod)

                df_dif.loc[idx, f'dif_{col}'] = dif_prob
    
    df_dif.to_excel("/Users/nachomondino/Desktop/df_dif_modeling_assess_prod_.xlsx")



if __name__ == "__main__":

    id_country = 77

    d_countries = {
        6: ["argentina", '2024-12-05'], 
        48: ["england", '2024-12-23'], 
        55: ["france", '2024-12-26'], 
        59: ["germany", '2024-12-26'], 
        77: ["italy", '2024-12-23'], 
        148: ["spain", '2024-12-25'], 
        167: ["usa", '2024-12-05']
        }
    
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]
    n_model = 14

    check_preparation(country)

    calculate_dif_proba(country)