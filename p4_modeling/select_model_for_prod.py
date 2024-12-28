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

    def __init__(self, id_country, iteration_date: str, verbose: int = 1):
        self.id_country = id_country
        self.iteration_date = iteration_date
        self.inicialize_directories()
        self.verbose = verbose

    def inicialize_directories(self):
        """
        Inicializo paths donde guardar los datos generados durante la seleccion del mejor modelo
        """
        d_countries = {-1: "all", 6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
        country = d_countries[self.id_country]

        self.BASE_PATH = f'data/{country}/p4_modeling/{self.iteration_date}'
        self.PATH_sbm = f'{self.BASE_PATH}/best_models' 
        self.path_old_sbm = f'{self.BASE_PATH}/best_models_old/{datetime.datetime.now().date()}' 

        directories.mover_archivo(origen=self.PATH_sbm, destino=self.path_old_sbm)
        directories.make_directories(l_directorios=[self.PATH_sbm])

    # Paso 1
    def filter_models_by_roi(self, df, perc_cutoff, roi_weight):
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
        name_extension='_sin_ea'
        self.metric_col = f'metric{name_extension}'

        # Calculo metrica combinada
        if roi_weight is None:
            self.roi_weight = asignar_roi_weight(df)  # Segun correlacion entre ROI y Expected ROI
        else:
            self.roi_weight = roi_weight

        df = calculate_combined_metric(df, roi_weight=self.roi_weight, name_extension=name_extension)

        # Eliminar registros con 'metric' < 0
        df = df[df[self.metric_col] >= 0]

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by=self.metric_col, ascending=False)

        # Seleccionar el 20% de los registros con los valores más altos de 'metric'
        cutoff = int(len(df) * perc_cutoff)  # Calcular el 20% superior
        df_filt = df.iloc[:cutoff]

        if self.verbose >= 1:
            logger.info(df_filt.head())
            logger.warning(f"Descarte por ROI: {len(df)} --> {len(df_filt)}")
            self.count_models(df_filt)

        df_filt.to_excel(f'{self.PATH_sbm}/df_p1.xlsx', index=False)
        return df_filt

    # Paso 2
    def filter_models_by_distribution(self, df, diff_max=0.3, diff_max_draw=0.4):
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

        df_filtered.to_excel(f'{self.PATH_sbm}/df_p2.xlsx', index=False)

        return df_filtered

    # Paso 3
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

        # Exportar el DataFrame final a un archivo Excel
        df_filt.set_index('n_iteration', inplace=True)
        df_filt.to_excel(f'{self.PATH_sbm}/df_p3.xlsx', index=True)

        return df_filt

    # Paso 4
    def select_model(self, df):
        """
        Selecciona el mejor modelo (aun sin estrategia de apuesta)
        """
        logger.info("Paso 4: Seleccionando mejor modelo...")

        # Ordenar los registros por 'metric' en orden descendente
        df = df.sort_values(by=self.metric_col, ascending=False)

        # Imprimo por pantalla el mejor modelo
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]:.1f}")

        df.to_excel(f'{self.PATH_sbm}/df_p4.xlsx', index=True)
        return row
    
    def count_models(self, df):
        if len(df) == 0:
            logger.error("Tras el descarte, se han eliminado todos los modelos. Revisar descartes.")
            raise ValueError

    # Main
    def main(self, df, perc_cutoff:float = 0.2, roi_weight: float = None):
        """
        Determino el modelo a usar en produccion
        """
        # PASO 1: DESCARTE POR ROI (sin estrategia de apuesta)
        df_filt_1 = self.filter_models_by_roi(df, perc_cutoff=perc_cutoff, roi_weight=roi_weight)

        # PASO 2: DESCARTE POR DISTRIBUCION
        df_filt_2 = self.filter_models_by_distribution(df_filt_1)

        # PASO 3: DESCARTE POR RELLENO DE NAN EN TEST
        df_filt_3 = self.filter_models_by_fill_nan(df_filt_2)
        
        # PASO 4: Seleccionar el modelo que maximiza ROI y expected ROI (sin estrategia)
        row = self.select_model(df_filt_3)
        return row
        
        
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 48

    # Defino variables
    d_countries = {6: ["argentina", '2024-12-05'], 48: ["england", '2024-12-23'], 55: ["france", '2024-12-26'], 59: ["germany", '2024-12-26'], 77: ["italy", '2024-12-23'], 148: ["spain", '2024-12-25'], 167: ["usa", '2024-12-05']}
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]

    # Creo objeto de clase select_best_model
    sbm = SelectBestModel(id_country=id_country, iteration_date=iteration_date)

    # Obtengo listado de todos los modelos entrenados
    df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

    # Selecciono el mejor modelo
    row = sbm.main(df_ite, perc_cutoff=0.05)





