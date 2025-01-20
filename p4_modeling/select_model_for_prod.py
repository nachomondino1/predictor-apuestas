import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
from tqdm import tqdm
import datetime


class SelectBestModel():

    def __init__(self, id_country, path_save: str = None, verbose: int = 1):
        self.id_country = id_country
        self.path_save = path_save
        self.verbose = verbose
 
    # Paso 1
    def filter_models_by_accuracy(self, df, dif_prob_bet_min=1):
        """
        Selecciono solo los modelos con una distribución de predicted_result similar 
        a la distribución de resultados en la realidad.
        """
        logger.info("Descartando modelos según precision...")

        df_filtered = df[df['dif_prec_bm'] > dif_prob_bet_min]

        if self.verbose >= 0:
            logger.warning(f'Descarte por accuracy: {len(df)} --> {len(df_filtered)}' )
            self.error_empty_dataframe(df_filtered)

        df_filtered.set_index('n_iteration', inplace=True)

        if self.path_save is not None:
            df_filtered.to_excel(f'{self.path_save}/df_filt_by_acc.xlsx', index=True)

        return df_filtered
    
    def filter_models_by_distribution(self, df, l_variables, l_values):
        """
        Filtra el DataFrame eliminando los índices que no cumplen con el valor mínimo 
        especificado para cada variable en l_variables y l_values.

        Args:
            df (pd.DataFrame): DataFrame a filtrar.
            l_variables (list): Lista de variables a evaluar.
            l_values (list): Lista de valores mínimos correspondientes a cada variable.

        Returns:
            pd.DataFrame: DataFrame filtrado.
        """
        logger.info("Descartando modelos según distribución en df_test")
        len_inic = len(df)
        # Asegurarse de que l_variables y l_values tengan la misma longitud
        if len(l_variables) != len(l_values):
            raise ValueError("Las listas l_variables y l_values deben tener la misma longitud.")

        # Crear un DataFrame filtrado
        for var, val in zip(l_variables, l_values):
            df = df[df[var] >= val]

        if self.verbose >= 0:
            logger.warning(f'Descarte por distribucion: {len(len_inic)} --> {len(df)}' )
            self.error_empty_dataframe(df)

        df.set_index('n_iteration', inplace=True)

        if self.path_save is not None:
            df.to_excel(f'{self.path_save}/df_filt_by_distrib.xlsx', index=True)

        return df

    # Paso 2
    def filter_models_by_fill_nan(self, df, type_: str = 'percentile', gp_fill_max: float = 0.1, n_col_fill_max: int = 5):

        # Lista para almacenar los resultados
        logger.info("Paso 3: Descartando modelos según relleno de nan values en df_test...")

        # Determino thresholds 
        if type_ == 'percentile': 
            gp_max = np.percentile(df['%_gp_filled'], 75) 
            max_average_cols =  np.percentile(df['average_col_filled'], 75) 

        elif type_ == 'fixed':
            gp_max = gp_fill_max 
            max_average_cols = n_col_fill_max

        # Eliminar modelos con G/P provenientes de relleno nan...
        df_filt = df[(df['%_gp_filled'] <= gp_max) & (df['average_col_filled'] <= max_average_cols)]

        if self.verbose >= 0:
            logger.warning(f'Descarte por relleno de nan en test: {len(df)} --> {len(df_filt)}')
            logger.info(f"Eliminar modelos con G/P filled >= {gp_max} o n_cols_filled >= {max_average_cols}. {len(df)} --> {len(df_filt)}")
            self.error_empty_dataframe(df_filt)

        if self.path_save is not None:
            df_filt.to_excel(f'{self.path_save}/df_filt_by_nan.xlsx', index=True)

        return df_filt

    # Paso 3
    def filter_models_by_metric(self, df, prop_to_max: float = 0.6, perc_cutoff: float = 1, metric_col: str = 'metric_sin_ea'):
        """
        Selecciona los mejores modelos (sin tener en cuenta la estrategia de apuesta aun).

        # Parameters
        df: Un DataFrame que contiene información sobre los modelos, incluyendo las columnas roi_por_partido y expected_roi_por_partido.
        cutoff: Proporción de los mejores registros a seleccionar (por defecto, 0.02 o el 2% superior).
        method: Metodo para filtrar. 
            'percentile' o 'prop_to_max'
        verbose: Nivel de detalle en los mensajes de salida.
            0 (por defecto): Salida básica.
            Valores mayores producen más detalles.

        # Return 
        La función devuelve un DataFrame (df_filt) que contiene solo los registros seleccionados con los valores más altos en la métrica combinada.
        """
        logger.info("Paso 1: Descartando modelos segun metrica en df_test")

        # Determino roi cut segun roi proporcional al max
        max_metric = df[metric_col].max()
        metric_cut_prop = max_metric * prop_to_max

        # Determino roi cut segun percentil 
        metric_cut_perc = np.percentile(df[metric_col], (100-perc_cutoff)) # Seleccionar el 20% de los registros con los valores más altos de 'metric'
        
        # Determino el minimo
        roi_cut = max(metric_cut_prop, metric_cut_perc)
        logger.info(f"\n(1) Metric Max: {max_metric} --> Metric cut: {max_metric * prop_to_max} \n(2) Metric cut percentile {perc_cutoff}: {metric_cut_perc}")
        logger.info(f"Metric cut: {metric_cut_prop} y {metric_cut_perc} --> {roi_cut}")

        # Filtro modelos segun roi to cut
        df_filt = df[df[metric_col] >= roi_cut]

        if self.verbose >= 1:
            logger.info(df_filt.head())
            logger.warning(f"Descarte por metrica: {len(df)} --> {len(df_filt)}")
            self.error_empty_dataframe(df_filt)

        if self.path_save is not None:
            df_filt.to_excel(f'{self.path_save}/df_filt_by_metric.xlsx', index=False)

        return df_filt

    # Paso 4
    def select_model(self, df, metric_col):
        """
        Selecciona el mejor modelo (aun sin estrategia de apuesta)
        """
        logger.info("Paso 4: Seleccionando mejor modelo...")

        if "n_iteration" in df.columns:
            df.set_index('n_iteration', inplace=True)

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by=metric_col, ascending=False)

        # Imprimo por pantalla el mejor modelo
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con metrica {row[metric_col].values[0]} y ROIpp {row['roi_por_partido'].values[0]:.1f}")

        if self.path_save is not None:
            df.to_excel(f'{self.path_save}/df_selected.xlsx', index=True)
            
        return row
    
    def error_empty_dataframe(self, df):
        if len(df) == 0:
            logger.error("El dataframe esta vació. Probablemente uno de los filtros eliminó todos los modelos que quedaban.")
            raise ValueError
        
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 55

    # Defino variables
    d_countries = {6: ["argentina", '2024-12-05'], 48: ["england", '2024-12-23'], 55: ["france", '2024-12-26'], 59: ["germany", '2024-12-26'], 77: ["italy", '2024-12-23'], 148: ["spain", '2024-12-25'], 167: ["usa", '2024-12-05']}
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]

    # Creo objeto de clase select_best_model
    sbm = SelectBestModel(id_country=id_country)

    # Obtengo listado de todos los modelos entrenados
    df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

    # PASO 1: DESCARTE POR DISTRIBUCION
    df_filt_1 = sbm.filter_models_by_distribution(df_ite)

    # PASO 2: DESCARTE POR RELLENO DE NAN EN TEST
    if ['%_gp_filled', 'average_col_filled'] in df_filt_1.columns:
        df_filt_2 = sbm.filter_models_by_fill_nan(df_filt_1)
    else:
        logger.warning("Evito eliminacion de modelos por relleno de nan values en test puesto que no le medí lo cuando entrené")
        df_filt_2 = df_filt_1.copy()

    # PASO 3: Seleccionar el modelo que maximiza ROI y expected ROI (sin estrategia)
    row = sbm.select_model(df_filt_2)





