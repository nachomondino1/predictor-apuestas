import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from utils import directories
from p4_modeling.asses_model import calculate_combined_metric, asignar_roi_weight
from tqdm import tqdm
import datetime


class SelectBestModel():

    def __init__(self, id_country, iteration_date: str, path_save: str = None, verbose: int = 1):
        self.id_country = id_country
        self.iteration_date = iteration_date
        self.path_save = path_save
        self.verbose = verbose
 
    # Paso 1
    def filter_models_by_distribution(self, df, diff_max=0.3, diff_max_draw=0.3):
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
            l_difs_2 =  [abs(row['dif_emp'])]

            # # Si alguna diferencia es menor o igual a diff_min, no eliminar el modelo
            # if any(diff <= diff_min for diff in l_difs):
            #     continue  # Salta este modelo y no lo elimina

            if any(diff >= diff_max_draw for diff in l_difs_2):
                l_idx_to_remove.append(idx)

            # Si alguna diferencia es mayor o igual a diff_max, eliminar el modelo
            if any(diff >= diff_max for diff in l_difs):
                l_idx_to_remove.append(idx)

        # Filtrar DataFrame eliminando los índices a remover
        df_filtered = df.drop(index=l_idx_to_remove)

        if self.verbose >= 0:
            logger.warning(f'Descarte por distribucion: {len(df)} --> {len(df_filtered)}' )
            logger.info(f"Se eliminaron {len(l_idx_to_remove)} modelos por distribucion muy distinta a la de results.")
            self.count_models(df_filtered)

        df_filtered.set_index('n_iteration', inplace=True)

        if self.path_save is not None:
            df_filtered.to_excel(f'{self.path_save}/df_distrib.xlsx', index=True)

        return df_filtered

    # Paso 2
    def filter_models_by_fill_nan(self, df):

        # Lista para almacenar los resultados
        logger.info("Paso 3: Descartando modelos según relleno de nan values en df_test...")

        # Determino thresholds 
        median_gp_filled = np.percentile(df['%_gp_filled'], 75)
        mean_n_cols_filled =  np.percentile(df['average_col_filled'], 75) 

        # Eliminar modelos con G/P provenientes de relleno nan...
        df_filt = df[(df['%_gp_filled'] <= median_gp_filled) & (df['average_col_filled'] <= mean_n_cols_filled)]

        if self.verbose >= 0:
            logger.warning(f'Descarte por relleno de nan en test: {len(df)} --> {len(df_filt)}')
            logger.info(f"Eliminar modelos con G/P filled >= {median_gp_filled} o n_cols_filled >= {mean_n_cols_filled}. {len(df)} --> {len(df_filt)}")
            self.count_models(df_filt)

        if self.path_save is not None:
            df_filt.to_excel(f'{self.path_save}/df_nan.xlsx', index=True)
            
        return df_filt

    # Paso 3
    def select_model(self, df, metric_col):
        """
        Selecciona el mejor modelo (aun sin estrategia de apuesta)
        """
        logger.info("Paso 4: Seleccionando mejor modelo...")

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by=metric_col, ascending=False)

        # Imprimo por pantalla el mejor modelo
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]:.1f}")

        if self.path_save is not None:
            df.to_excel(f'{self.path_save}/df_selected.xlsx', index=True)
            
        return row
    
    def count_models(self, df):
        if len(df) == 0:
            logger.error("Tras el descarte, se han eliminado todos los modelos. Revisar descartes.")
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
    sbm = SelectBestModel(id_country=id_country, iteration_date=iteration_date)

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





