import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.set_up_logging import logger
from utils import directories
from p4_modeling.betting_strategy import BettingStrategy, calculate_metric
from p6_deployment.assess_models_in_prod import assess_model_in_prod
from tqdm import tqdm
import datetime

class SelectBestModel():

    def __init__(self, id_country, iteration_date: str, roi_weight: float = 0.75, verbose: int = 1):
        self.id_country = id_country
        self.iteration_date = iteration_date
        self.roi_weight = roi_weight
        self.inicialize_directories()
        self.verbose = verbose

    def inicialize_directories(self):
        """
        Inicializo paths donde guardar los datos generados durante la seleccion del mejor modelo
        """
        d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
        country = d_countries[id_country]

        self.BASE_PATH = f'data/{country}/p4_modeling/{self.iteration_date}'
        self.PATH_sbm = f'data/{country}/p4_modeling/{self.iteration_date}/best_models' 
        self.path_old_sbm = f'{self.PATH_sbm}/{datetime.datetime.now().date()}' 
        self.path_predic = f'{self.PATH_sbm}/predicciones/'
        directories.copy_directory(origen=self.PATH_sbm, destino=self.path_old_sbm)
        directories.make_directories(l_directorios=[self.PATH_sbm, self.path_predic])

    # Paso 1
    def filter_models_by_roi(self, df, perc_cutoff):
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
        df = calculate_metric(df, self.roi_weight)

        # Eliminar registros con 'metric' < 0
        df = df[df['metric'] >= 0]

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by='metric', ascending=False)

        # Seleccionar el 20% de los registros con los valores más altos de 'metric'
        cutoff = int(len(df) * perc_cutoff)  # Calcular el 20% superior
        df_filt = df.iloc[:cutoff]

        if self.verbose >= 1:
            logger.info(f"Descarte por ROI: {len(df)} --> {len(df_filt)}")
            logger.info(df_filt.head())
        
        df_filt.to_excel(f'{self.PATH_sbm}/df_p1.xlsx', index=False)
        return df_filt

    # Paso 2
    def filter_models_by_distribution(self, df, diff_max=0.3, diff_min=0.05):
        """
        Selecciono solo los modelos con una distribución de predicted_result similar 
        a la distribución de resultados en la realidad.
        """
        logger.info("Paso 2: Descartando modelos según distribución en df_test")

        # Defino variables
        l_idx_to_remove = []

        # Por modelo
        for idx, row in df.iterrows():
            l_difs = [abs(row['dif_loc']), abs(row['dif_vis'])]  # No considero la diff de empate.

            # Si alguna diferencia es menor o igual a diff_min, no eliminar el modelo
            if any(diff <= diff_min for diff in l_difs):
                continue  # Salta este modelo y no lo elimina

            # Si alguna diferencia es mayor o igual a diff_max, eliminar el modelo
            if any(diff >= diff_max for diff in l_difs):
                l_idx_to_remove.append(idx)

        # Filtrar DataFrame eliminando los índices a remover
        df_filtered = df.drop(index=l_idx_to_remove)
        df_filtered.to_excel(f'{self.PATH_sbm}/df_p2.xlsx', index=False)

        logger.warning(f"Se eliminaron {len(l_idx_to_remove)} modelos por distribucion muy distinta a la de results.")

        if self.verbose >= 1:
            print(f'{df.shape} --> {df_filtered.shape}' )

        return df_filtered

    # Paso 3
    def define_betting_strategy_per_model(self, df, with_assess: bool = False):
        """
        Determina la estrategia de apuesta optima para cada modelo.
        """
        # Lista para almacenar los resultados
        logger.info("Paso 3: Definiendo la estrategia de apuesta optima por modelo...")
        bs = BettingStrategy(strategy='general', verbose=-1)
        progress_bar = tqdm(total=len(df), ncols=80)  # Inicializo barra de progreso
        results = []

        # Por modelo
        for idx, row in df.iterrows():

            n_model, model_name = row['n_iteration'], row['model_name']
            if self.verbose >= 1:
                logger.info(f'n_model: {n_model} model_name: {model_name}')

            # Levanto df_predicciones --> Aqui deberia ser capaz de levantar las predicciones sobre los missing tambien y evaluar todo junto (test + missing).             # Falta levantar las predicciones de los missing y concatenerlas (si hubiera) --> no haria falta el assess_models_in_prod.py????
            df_pred = pd.read_excel(f"{self.BASE_PATH}/models/{n_model}__{model_name}_predicciones.xlsx", index_col=0)
            logger.info(df_pred.shape)
            
            # (opcional) Hacer assess --> Lo haria solo para los pocos modelos que pasan el filtro inicial.
            if with_assess: # Hay que ver si funciona...

                # Recolecto predicciones en ultimos partidos
                df_pred_assess = assess_model_in_prod(self.id_country, n_model, model_name, iteration_date)
                logger.info(df_pred_assess.shape)

                # Concateno df
                len_inic = len(df_pred)
                df_pred = pd.concat([df_pred, df_pred_assess], axis=0)
                logger.info(f"Se concatenó las predicciones de los partidos missing: {len_inic} --> {len(df_pred)}")

                if self.verbose >= 2:
                    # Exporto predicciones concatenado
                    df_pred.to_excel(f"{self.path_predic}/predicciones_raw_{n_model}_{model_name}.xlsx") # df_predicciones???
            
            # Determino la mejor estrategia de apuesta
            if self.verbose >= 1:
                logger.info("Calculando la mejor estrategia de apuesta...")
            d_hiper, best_df_pred, best_d_rois = bs.calculate_roi_by_betting_strategy(df_pred, roi_weight=self.roi_weight)  # no esta calculando el ROI con los nuevos partidos assess... no calcula ok las winning bets ni nada.

            # Concateno datos y guardo
            d_ct = {**d_hiper, **best_d_rois}  # Combinar los dos diccionarios
            d_ct['model_name'] = model_name
            d_ct['n_model'] = n_model

            # Añadir el resultado al DataFrame final
            results.append(d_ct)
            progress_bar.update(1)

            # Exporto datos
            best_df_pred.to_excel(f'{self.path_predic}/predicciones_{n_model}_{model_name}.xlsx')

        progress_bar.close()

        # Convertir la lista de resultados en un DataFrame
        df_final = pd.DataFrame(results)
        df_final.set_index('n_model', inplace=True)
        
        # Exportar el DataFrame final a un archivo Excel
        df_final.to_excel(f'{self.PATH_sbm}/df_p3.xlsx', index=True)
    
        return df_final

    # Paso 4
    def select_best_model_with_strategy(self, df):
        """
        Selecciona el mejor modelo ya teniendo la estrategia de apuesta optima para cada uno.
        """
        # DETERMINAR QUE MODELO ES EL MEJOR CON LA ESTRATEGIA (QUEREMOS MAXIMIZAR EL ROI PERO TAMBIEN EL EXPECTED ROI)
        # habria que calcular los roi nuevamente pero ahora aplicando la estrategia de apuesta...
        # ... Ordenar segun metric y tomar el que la maximiza? En SPA me gusto mas el 1001 que tiene mayor ROI pero no asi expected. Con el tiempo sabré cual fue mejor.
        logger.info("Paso 4: Seleccionando mejor modelo ya habiendo aplicado la estrategia de apuesta optima a cada uno.")

        # Calculo metrica combinada (con ROIpp y ExpectedRoipp ya habiendo aplicado la estrategia de apuesta)
        df = calculate_metric(df, roi_weight=self.roi_weight)

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by='metric', ascending=False)
        df.to_excel(f'{self.PATH_sbm}/df_p4.xlsx', index=True)

        return df

    # Main
    def main(self, df, perc_cutoff:float = 0.2, with_assess: bool = False):
        """
        Determino el modelo a usar en produccion
        """
        # PASO 1: DESCARTE POR ROI (sin estrategia de apuesta)
        df_filt = self.filter_models_by_roi(df, perc_cutoff=perc_cutoff)

        # PASO 2: DESCARTE POR DISTRIBUCION
        df_filt = self.filter_models_by_distribution(df_filt)

        # PASO 3: Definir mejor combinación de hiperparametros de apuesta por modelo
        df_filt_strategy = self.define_betting_strategy_per_model(df_filt, with_assess=with_assess)
        
        # PASO 4: Seleccionar el modelo que maximiza ROI y expected ROI con la estrategia de apuesta
        df = self.select_best_model_with_strategy(df_filt_strategy)

        # Imprimo por pantalla el mejor modelo
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]}")

        # Concateno dataframes y exporto
        # df_concat = 

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 48
    iteration_date = '2024-12-16'

    # Defino variables
    d_countries = {6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Determino roi_weight
    roi_weight = 1  # Pues expected presumo que mete ruido x no tener bien definido el threshold. # if id_country == 55 else 0.8 # Uso roi_weight de 1 en GER porque no hay correl entre roi y expected roi.

    # Creo objeto de clase select_best_model
    sbm = SelectBestModel(id_country=id_country, iteration_date=iteration_date, roi_weight=roi_weight)

    # Obtengo listado de todos los modelos entrenados
    df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

    # Selecciono el mejor modelo
    sbm.main(df_ite, perc_cutoff=0.01, with_assess=False)





