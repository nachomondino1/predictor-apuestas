import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.set_up_logging import logger
from utils.directories import make_directories
from p4_modeling.betting_strategy import BettingStrategy
from p6_deployment.assess_models_in_prod import assess_model_in_prod


class SelectBestModel():

    def __init__(self, id_country, country, iteration_date, verbose: int = 1):
        self.id_country = id_country
        self.country = country
        self.iteration_date = iteration_date
        self.BASE_PATH = f'data/{self.country}/p4_modeling/{self.iteration_date}'
        self.verbose = verbose

    # Paso 1
    def select_models_best_roi(self, df, perc_cutoff:float = 0.02, verbose: int = 0):
        """
        Seleccionar los modelos con mejor retorno de inversión (ROI) basado en una métrica combinada de ROI y ROI esperado (expected ROI)

        # Parameters
        df: Un DataFrame que contiene información sobre los modelos, incluyendo las columnas roi_por_partido y expected_roi_por_partido.
        cutoff: Proporción de los mejores registros a seleccionar (por defecto, 0.02 o el 2% superior).
        verbose: Nivel de detalle en los mensajes de salida.
            0 (por defecto): Salida básica.
            Valores mayores producen más detalles.

        # Return 
        La función devuelve un DataFrame (df_filt) que contiene solo los registros seleccionados con los valores más altos en la métrica combinada.
        """
        # Normalizar las columnas roi_por_partido y expected_roi_por_partido
        df = self.normalize_metrics(df, columns_to_normalize=['roi_por_partido', 'expected_roi_por_partido'])
        
        # Sumar las columnas normalizadas
        df['metric'] = df['roi_por_partido'] + df['expected_roi_por_partido']    

        # Eliminar registros con 'metric' < 0
        df = df[df['metric'] >= 0]

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by='metric', ascending=False)

        # Seleccionar el 20% de los registros con los valores más altos de 'metric'
        cutoff = int(len(df) * perc_cutoff)  # Calcular el 20% superior
        df_filt = df.iloc[:cutoff]

        if verbose >= 0:
            logger.info(f"Descarte por ROI: {len(df)} --> {len(df_filt)}")
            logger.info(df_filt.head())
            
        return df_filt

    def normalize_metrics(self, df, columns_to_normalize):
        """
        Normalizo columnas de dataframe
        """
        # Crear el escalador
        scaler = MinMaxScaler()    

        # Normalizar las columnas
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df

    # Paso 2
    def define_betting_strategy_per_model(self, df, with_assess: bool = False, verbose: int = 0):

        # Defino variables
        bs = BettingStrategy(verbose=verbose)
        df_final = pd.DataFrame()

        # Por modelo
        for idx, row in df.iterrows():

            n_model, model_name = row['n_iteration'], row['model_name']
            if verbose >= 0:
                logger.info(f'n_model: {n_model} model_name: {model_name}')

            # Levanto df_predicciones --> Aqui deberia ser capaz de levantar las predicciones sobre los missing tambien y evaluar todo junto (test + missing).             # Falta levantar las predicciones de los missing y concatenerlas (si hubiera) --> no haria falta el assess_models_in_prod.py????
            df_pred = pd.read_excel(f"{self.BASE_PATH}/models/{n_model}__{model_name}_predicciones.xlsx")
            
            # (opcional) Hacer assess --> Lo haria solo para los pocos modelos que pasan el filtro inicial.
            if with_assess: # Hay que ver si funciona...

                # Recolecto predicciones en ultimos partidos
                df_pred_assess = assess_model_in_prod(self.id_country, n_model, model_name, iteration_date)

                # # Calculo ROI y expected ROI --> No deberia pues ahora calculo metricas del df_concatenado..
                # df_pred_assess = calculate_metrics(df_pred_assess)

                # Concateno df
                len_inic = len(df_pred)
                df_pred = pd.concat([df_pred, df_pred_assess], axis=0)
                logger.info(f"Se concatenó las predicciones de los partidos missing: {len_inic} --> {len(df_pred)}")

                # Exporto predicciones concatenado
                path_country = f'{self.BASE_PATH}/best_model'
                make_directories(l_directorios=[path_country])
                df_pred.to_excel(f"{path_country}/predicciones_{n_model}_{model_name}.xlsx") # df_predicciones???

            # Determino la mejor estrategia de apuesta
            best_df_pred, best_d_rois = bs.calculate_roi_by_betting_strategy(df_pred, strategy="general", save_strategy=True)

            # Concateno datos y guardo
            df_ct = pd.DataFrame(best_d_rois, index=[n_model])
            df_ct.loc[n_model, 'model_name'] = model_name
            df_final = pd.concat([df_final, df_ct], axis=0)
            # Guardar los mejores hiperparametros de apuesta en df_best_models.xlsx
            # best_d_rois['thr_prob_min_best'] # 'curva', 'param1', 'param2' 'normalized', 'odd_weight', 'dif_prob_sup_cap'
            # Exporto datos
            # best_df_pred.to_excel(f'{BASE_PATH}/df_iteration_with_strategy.xlsx')
            df_final.to_excel(f'{self.BASE_PATH}/df_iteration_with_strategy.xlsx')

        return df_final

    # Paso 3? 
    def algo():
        # DETERMINAR QUE MODELO ES EL MEJOR CON LA ESTRATEGIA (QUEREMOS MAXIMIZAR EL ROI PERO TAMBIEN EL EXPECTED ROI)
        # habria que calcular los roi nuevamente pero ahora aplicando la estrategia de apuesta...
        # ... Ordenar segun metric y tomar el que la maximiza? En SPA me gusto mas el 1001 que tiene mayor ROI pero no asi expected. Con el tiempo sabré cual fue mejor.
        pass

    def main(self, df, verbose: int = 0):
        """
        Determino el modelo a usar en produccion
        """
        # INPUT? --> ASSESS ? --> Lo estaria haciendo dentro del paso 2... Aun no se si funciona.
        
        # PASO 1: DESCARTE POR ROI
        df_filt = self.select_models_best_roi(df, verbose=verbose)

        # PASO 2: Definir mejor combinación de hiperparametros de apuesta por modelo
        df_filt_strategy = self.define_betting_strategy_per_model(df_filt, with_assess=False)
        
        # PASO 3: Seleccionar el modelo que maximiza ROI y expected ROI con la estrategia de apuesta
        # self.algo()

        # return n_model ?

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 148
    iteration_date = '2024-12-10'

    # Defino variables
    d_countries = {6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]
    sbm = SelectBestModel(id_country=id_country, country=country, iteration_date=iteration_date)

    # Obtengo listado de todos los modelos entrenados
    df = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

    # Selecciono el mejor modelo
    sbm.main(df)





