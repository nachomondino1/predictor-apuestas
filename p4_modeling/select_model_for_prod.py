import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.set_up_logging import logger
from utils.directories import make_directories
from p4_modeling.betting_strategy import BettingStrategy
from p6_deployment.assess_models_in_prod import assess_model_in_prod
from tqdm import tqdm


class SelectBestModel():

    def __init__(self, id_country, country, iteration_date, verbose: int = 1):
        self.id_country = id_country
        self.country = country
        self.iteration_date = iteration_date
        self.BASE_PATH = f'data/{self.country}/p4_modeling/{self.iteration_date}'
        self.PATH_sbm = f'data/{self.country}/p4_modeling/{self.iteration_date}/best_models'
        make_directories(l_directorios=[self.PATH_sbm])
        self.verbose = verbose
        self.bs = BettingStrategy()

    # Paso 1
    def select_models_best_roi(self, df, perc_cutoff, verbose: int = 0):
        """
        Selecciona los mejores modelos (sin tener en cuenta la estrategia de apuesta aun).

        # Parameters
        df: Un DataFrame que contiene información sobre los modelos, incluyendo las columnas roi_por_partido y expected_roi_por_partido.
        cutoff: Proporción de los mejores registros a seleccionar (por defecto, 0.02 o el 2% superior).
        verbose: Nivel de detalle en los mensajes de salida.
            0 (por defecto): Salida básica.
            Valores mayores producen más detalles.

        # Return 
        La función devuelve un DataFrame (df_filt) que contiene solo los registros seleccionados con los valores más altos en la métrica combinada.
        """
        logger.info("Paso 1: Descartando modelos con bajo ROI en df_test")

        # Calculo metrica combinada
        df = self.bs.calculate_metric(df)

        # Eliminar registros con 'metric' < 0
        df = df[df['metric'] >= 0]

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by='metric', ascending=False)

        # Seleccionar el 20% de los registros con los valores más altos de 'metric'
        cutoff = int(len(df) * perc_cutoff)  # Calcular el 20% superior
        df_filt = df.iloc[:cutoff]

        if verbose >= 1:
            logger.info(f"Descarte por ROI: {len(df)} --> {len(df_filt)}")
            logger.info(df_filt.head())
        
        df_filt.to_excel(f'{self.PATH_sbm}/df_p1.xlsx', index=False)
        return df_filt

    # Paso 2
    def define_betting_strategy_per_model(self, df, with_assess: bool = False, verbose: int = 0):
        """
        Determina la estrategia de apuesta optima para cada modelo.
        """
        # Lista para almacenar los resultados
        results = []
        logger.info("Paso 2: Definiendo la estrategia de apuesta optima por modelo...")
        progress_bar = tqdm(total=len(df), ncols=80)  # Inicializo barra de progreso

        path_predic = f'{self.PATH_sbm}/predicciones/'
        make_directories(l_directorios=[path_predic])

        # Por modelo
        for idx, row in df.iterrows():

            n_model, model_name = row['n_iteration'], row['model_name']
            if verbose >= 1:
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
                df_pred.to_excel(f"{path_predic}/predicciones_raw_{n_model}_{model_name}.xlsx") # df_predicciones???

            # Determino la mejor estrategia de apuesta
            d_hiper, best_df_pred, best_d_rois = self.bs.calculate_roi_by_betting_strategy(df_pred, strategy="general")

            # Concateno datos y guardo
            d_ct = {**d_hiper, **best_d_rois}  # Combinar los dos diccionarios
            d_ct['model_name'] = model_name
            d_ct['n_model'] = n_model

            # Añadir el resultado al DataFrame final
            results.append(d_ct)
            progress_bar.update(1)

            # Exporto datos
            best_df_pred.to_excel(f'{path_predic}/predicciones_{n_model}_{model_name}.xlsx')

        progress_bar.close()

        # Convertir la lista de resultados en un DataFrame
        df_final = pd.DataFrame(results)
        df_final.set_index('n_model', inplace=True)
        
        # Exportar el DataFrame final a un archivo Excel
        df_final.to_excel(f'{self.PATH_sbm}/df_p2.xlsx', index=True)
    
        return df_final

    # Paso 3
    def select_best_model_with_strategy(self, df):
        """
        Selecciona el mejor modelo ya teniendo la estrategia de apuesta optima para cada uno.
        """
        # DETERMINAR QUE MODELO ES EL MEJOR CON LA ESTRATEGIA (QUEREMOS MAXIMIZAR EL ROI PERO TAMBIEN EL EXPECTED ROI)
        # habria que calcular los roi nuevamente pero ahora aplicando la estrategia de apuesta...
        # ... Ordenar segun metric y tomar el que la maximiza? En SPA me gusto mas el 1001 que tiene mayor ROI pero no asi expected. Con el tiempo sabré cual fue mejor.
        logger.info("Paso 3: Seleccionando mejor modelo ya habiendo aplicado la estrategia de apuesta optima a cada uno.")

        # Calculo metrica combinada (con ROIpp y ExpectedRoipp habianedo aplicado la estrategia de apuesta)
        df = self.bs.calculate_metric(df)

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by='metric', ascending=False)
        df.to_excel(f'{self.PATH_sbm}/df_p3.xlsx', index=True)

        return df

    # Main
    def main(self, df, perc_cutoff:float = 0.05, with_assess=False, verbose: int = 0):
        """
        Determino el modelo a usar en produccion
        """
        # INPUT? --> ASSESS ? --> Lo estaria haciendo dentro del paso 2... Aun no se si funciona.
        
        # PASO 1: DESCARTE POR ROI (sin estrategia de apuesta)
        df_filt = self.select_models_best_roi(df, perc_cutoff=perc_cutoff,verbose=verbose)

        # PASO 2: Definir mejor combinación de hiperparametros de apuesta por modelo
        df_filt_strategy = self.define_betting_strategy_per_model(df_filt, with_assess=with_assess)
        
        # PASO 3: Seleccionar el modelo que maximiza ROI y expected ROI con la estrategia de apuesta
        df = self.select_best_model_with_strategy(df_filt_strategy)

        # Imprimo por pantalla el mejor modelo
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]}")

        # Concateno dataframes y exporto
        # df_concat = 

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 55
    iteration_date = '2024-12-12'

    # Defino variables
    d_countries = {6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]
    sbm = SelectBestModel(id_country=id_country, country=country, iteration_date=iteration_date)

    # Obtengo listado de todos los modelos entrenados
    df = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

    # Selecciono el mejor modelo
    sbm.main(df)





