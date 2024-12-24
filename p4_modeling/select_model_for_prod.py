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
        self.PATH_sbm = f'{self.BASE_PATH}/best_models' 
        self.path_old_sbm = f'{self.BASE_PATH}/best_models_old/{datetime.datetime.now().date()}' 

        directories.make_directories(l_directorios=[self.PATH_sbm, self.path_old_sbm])
        directories.mover_archivo(origen=self.PATH_sbm, destino=self.path_old_sbm)

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
            logger.info(df_filt.head())
            logger.warning(f"Descarte por ROI: {len(df)} --> {len(df_filt)}")

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
            l_difs = [abs(row['dif_loc']), abs(row['dif_emp']), abs(row['dif_vis'])]  # No considero la diff de empate.

            # # Si alguna diferencia es menor o igual a diff_min, no eliminar el modelo
            # if any(diff <= diff_min for diff in l_difs):
            #     continue  # Salta este modelo y no lo elimina

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
    def filter_models_by_fill_nan(self, df):

        # Lista para almacenar los resultados
        logger.info("Paso 3: Descartando modelos según relleno de nan values en df_test...")
        progress_bar = tqdm(total=len(df), ncols=80)  # Inicializo barra de progreso
        
        # Por modelo
        for idx, row in df.iterrows():

            n_model, model_name = row['n_iteration'], row['model_name']
            # if self.verbose >= 1:
            #     logger.info(f'n_model: {n_model} model_name: {model_name}')

            # Levanto df_predicciones --> Aqui deberia ser capaz de levantar las predicciones sobre los missing tambien y evaluar todo junto (test + missing).             # Falta levantar las predicciones de los missing y concatenerlas (si hubiera) --> no haria falta el assess_models_in_prod.py????
            df_pred = pd.read_excel(f"{self.BASE_PATH}/models/{n_model}__{model_name}_predicciones.xlsx", index_col=0)
            logger.info(df_pred.shape)

            # Calculo metricas sobre relleno de nan
            rows_filled = df_pred[df_pred['player_emergency_fill'] == 1].index
            rows_not_filled = df_pred[df_pred['player_emergency_fill'] != 1].index
            # print(len(rows_filled), len(rows_not_filled))
            
            gp_filled = df_pred.loc[rows_filled, 'G/P_sin_bank'].sum()
            gp_not_filled = df_pred.loc[rows_not_filled, 'G/P_sin_bank'].sum()
            average_col_filled = df_pred['n_col_filled'].sum() / len(df_pred)

            # Agrego columnas sobre relleno de nan en df_test en df
            ## Cantidad de registros rellenados y average de columnas rellenadas
            df.loc[idx, 'n_matches_filled'] = len(rows_filled)
            df.loc[idx, 'average_col_filled'] = average_col_filled
            ## G/P cuando relleno y G/P cuando no relleno
            df.loc[idx, 'sum_gp_filled'] = gp_filled
            df.loc[idx, 'sum_gp_not_filled'] = gp_not_filled
            df.loc[idx, '%_gp_filled'] = int(gp_filled / (gp_filled + gp_not_filled) * 100)
            df.loc[idx, '%_gp_not_filled'] = int(gp_not_filled / (gp_filled + gp_not_filled) * 100)
            progress_bar.update(1)  
        
        progress_bar.close()

        
        # Eliminar modelos con G/P provenientes de relleno nan...
        import numpy as np
        median_gp_filled = np.percentile(df['%_gp_filled'], 75)
        mean_n_cols_filled =  np.percentile(df['average_col_filled'], 75) 

        df_filt = df[(df['%_gp_filled'] <= median_gp_filled) & (df['average_col_filled'] <= mean_n_cols_filled)]
        logger.warning(f"Eliminar modelos con G/P filled >= {median_gp_filled} o n_cols_filled >= {mean_n_cols_filled}. {len(df)} --> {len(df_filt)}")
        
        # df_filt = df.copy()

        # Exportar el DataFrame final a un archivo Excel
        df_filt.to_excel(f'{self.PATH_sbm}/df_p3.xlsx', index=True)
        return df_filt

    # Paso 4
    def select_model(self, df):
        """
        Selecciona el mejor modelo (aun sin estrategia de apuesta)
        """
        logger.info("Paso 4: Seleccionando mejor modelo...")

        # Calculo metrica combinada (con ROIpp y ExpectedRoipp ya habiendo aplicado la estrategia de apuesta)
        df = calculate_metric(df, roi_weight=self.roi_weight)

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by='metric', ascending=False)
        df.set_index('n_iteration', inplace=True)

        # Imprimo por pantalla el mejor modelo
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]:.1f}")

        df.to_excel(f'{self.PATH_sbm}/df_p4.xlsx', index=True)
        return row

    # Paso 5
    def define_betting_strategy(self, row, strategy: str = 'general', with_assess: bool = False):
        """
        Determina la estrategia de apuesta optima para cada modelo.
        """
        # Lista para almacenar los resultados
        logger.info("Paso 5: Definiendo la estrategia de apuesta optima para el modelo seleccionado...")
        bs = BettingStrategy(strategy=strategy, verbose=-1)
        results = []

        print(row)
        logger.warning(row)
        n_model, model_name = row.index[0], row['model_name'].values[0]
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

        # Convertir la lista de resultados en un DataFrame
        df_final = pd.DataFrame(results)
        df_final.set_index('n_model', inplace=True)
        
        # Exportar el DataFrame final a un archivo Excel
        best_df_pred.to_excel(f'{self.PATH_sbm}/predicciones_{n_model}_{model_name}.xlsx')
        df_final.to_excel(f'{self.PATH_sbm}/df_strategy.xlsx', index=True)
        return df_final

    # Main
    def main(self, df, perc_cutoff:float = 0.2, strategy: str = 'general', with_assess: bool = False):
        """
        Determino el modelo a usar en produccion
        """
        # PASO 1: DESCARTE POR ROI (sin estrategia de apuesta)
        df_filt_1 = self.filter_models_by_roi(df, perc_cutoff=perc_cutoff)

        # PASO 2: DESCARTE POR DISTRIBUCION
        df_filt_2 = self.filter_models_by_distribution(df_filt_1)

        # PASO 3: DESCARTE POR RELLENO DE NAN EN TEST
        df_filt_3 = self.filter_models_by_fill_nan(df_filt_2)
        
        # PASO 4: Seleccionar el modelo que maximiza ROI y expected ROI (sin estrategia)
        row_model = self.select_model(df_filt_3)

        # PASO 5: Determinar estrategia de apuesta optima para el modelo seleccionado
        self.define_betting_strategy(row=row_model, strategy=strategy, with_assess=with_assess)
        
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 77
    iteration_date = '2024-12-23'

    # Defino variables
    d_countries = {6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Parametros de ejecucion
    roi_weight = 0.75  # Pues expected presumo que mete ruido x no tener bien definido el threshold. # if id_country == 55 else 0.8 # Uso roi_weight de 1 en GER porque no hay correl entre roi y expected roi.
    strategy = 'general' # 'general_0' if id_country == 77 else 'general'
    with_assess = False

    # Creo objeto de clase select_best_model
    sbm = SelectBestModel(id_country=id_country, iteration_date=iteration_date, roi_weight=roi_weight)

    # Obtengo listado de todos los modelos entrenados
    df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

    # Selecciono el mejor modelo
    sbm.main(df_ite, perc_cutoff=0.05, strategy=strategy, with_assess=with_assess)





