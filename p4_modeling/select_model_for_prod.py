import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p4_modeling.betting_strategy import BettingStrategy
from sklearn.preprocessing import MinMaxScaler
from utils.set_up_logging import logger


def normalize_metrics(df, columns_to_normalize):
    """
    Normalizo columnas de dataframe
    """
    # Crear el escalador
    scaler = MinMaxScaler()    

    # Normalizar las columnas
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def main(df, BASE_PATH, cutoff:float = 0.02, verbose: int = 0):

    # Defino variables
    bs = BettingStrategy(verbose=0)
    df_final = pd.DataFrame()

    # PASO 1: DESCARTE POR ROI
    # Normalizar las columnas roi_por_partido y expected_roi_por_partido
    df = normalize_metrics(df, columns_to_normalize=['roi_por_partido', 'expected_roi_por_partido'])
    
    # Sumar las columnas normalizadas
    df['metric'] = df['roi_por_partido'] + df['expected_roi_por_partido']    

    # Eliminar registros con 'metric' < 0
    df = df[df['metric'] >= 0]

    # Ordenar los registros por 'metric' en orden descendente
    df = df.sort_values(by='metric', ascending=False)

    # Seleccionar el 20% de los registros con los valores más altos de 'metric'
    pareto_cutoff = int(len(df) * cutoff)  # Calcular el 20% superior
    df_pareto = df.iloc[:pareto_cutoff]

    if verbose >= 0:
        logger.info(f"Descarte por ROI: {len(df)} --> {len(df_pareto)}")
        logger.info(df_pareto.head())
    
    # PASO 2: Aplicar estrategia de apuesta a modelos que quedan       
    # Por modelo
    for idx, row in df_pareto.iterrows():

        n_model = row['n_iteration']
        model_name = row['model_name']
        if verbose >= 0:
            logger.info(f'n_model: {n_model} model_name: {model_name}')

        # Levanto df_predicciones --> Aqui deberia ser capaz de levantar las predicciones sobre los missing tambien y evaluar todo junto (test + missing)
        df_pred = pd.read_excel(f"{BASE_PATH}/models/{n_model}__{model_name}_predicciones.xlsx")
        # Falta levantar las predicciones de los missing y concatenerlas (si hubiera) --> no haria falta el assess_models_in_prod.py????
        
        # Determino la mejor estrategia de apuesta
        best_df_pred, best_d_rois = bs.calculate_roi_by_betting_strategy(df_pred, strategy = "general", save_strategy=True)

        df_ct = pd.DataFrame(best_d_rois, index=[n_model])
        df_ct.loc[n_model, 'model_name'] = model_name

        df_final = pd.concat([df_final, df_ct], axis=0)

        # Guardar los mejores hiperparametros de apuesta en df_best_models.xlsx
        # best_d_rois['thr_prob_min_best'] # 'curva', 'param1', 'param2' 'normalized', 'odd_weight', 'dif_prob_sup_cap'

        # Exporto datos
        # best_df_pred.to_excel(f'{BASE_PATH}/df_iteration_with_strategy.xlsx')
        df_final.to_excel(f'{BASE_PATH}/df_iteration_with_strategy.xlsx')
    
    # PASO 3: DETERMINAR QUE MODELO ES EL MEJOR CON LA ESTRATEGIA (QUEREMOS MAXIMIZAR EL ROI PERO TAMBIEN EL EXPECTED ROI)
    # habria que calcular los roi nuevamente pero ahora aplicando la estrategia de apuesta...
    # ... Ordenar segun metric y tomar el que la maximiza? En SPA me gusto mas el 1001 que tiene mayor ROI pero no asi expected. Con el tiempo sabré cual fue mejor.


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 148
    iteration_date = '2024-12-10'

    # Defino variables
    d_countries = {6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]
    BASE_PATH = f'data/{country}/p4_modeling/{iteration_date}'

    # Obtengo listado de todos los modelos entrenados
    df_iteration = pd.read_excel(f'{BASE_PATH}/df_iteration.xlsx')

    # Determino el modelo a usar en produccion
    n_model = main(df_iteration, BASE_PATH)




