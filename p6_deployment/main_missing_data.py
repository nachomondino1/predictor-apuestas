# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import datetime
from utils.set_up_logging import logger
from utils import directories
import os
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_data
## Data preparation
from main import DataPreparation
from p3_data_preparation.integrate_sofifa_to_flashscore import *
from p3_data_preparation.select_data import determine_country_competitions


class DataUnderstandingMissing():

    def __init__(self, l_countries, export: bool = True):
        self.l_countries = l_countries
        # self.date = date
        self.export = export
        self.make_directories()

    def make_directories(self):
        base_path = f'./data/p6_deployment'
        self.path_missing = f'{base_path}/missing'
        self.path_unders = f'{base_path}/data_understanding'
        
        l_directorios = [
            self.path_unders,
            f'{self.path_missing}/data_understanding/all',
            f'{self.path_missing}/old_updated',   
            f'{self.path_missing}/data_preparation/all',
        ]

        directories.make_directories(l_directorios=l_directorios)
        
    def collect_missing_data(self, df_match: pd.DataFrame, df_comp_country: pd.DataFrame, n_seasons_max: int = 1, _print: bool = False):
        """
        Extraccion de varias competencias de un mismo country.
        """
        logger.info("Collecting data...")
        # Definicion de variables
        df_match_concat, df_match_player_concat, df_match_odds_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Solo extriago las competencias que tengo en los datos viejos
        l_ids_extracted = list(df_match.index) 
        l_competencies = df_match['id_competition'].unique()
        print("Competencias extraidas: ", l_competencies)

        # POR COMPETITION (solo las que hay en df_match)
        for id_competition in l_competencies:

            # evito competencias que extraje en df_match pero no quiero recolectar missing
            if id_competition in [1672, 1673]:
                continue

            # Obtengo nombre de competicion y is_cup
            df_comp_filt = df_comp_country[df_comp_country['id_competition'] == id_competition] 
            competition, is_cup = df_comp_filt['competition_flashscore'].values[0], df_comp_filt['is_cup'].values[0]
            if _print:
                print(f" Competition: {competition} ".center(120, '+'))

            # Actualizo df_match y df_match_player con los partidos faltantes
            df_match_miss, df_match_player_miss, df_match_odds_miss = extract_data(self.id_country, self.country, id_competition, competition, is_cup, n_seasons_max=n_seasons_max, l_ids_already_collected=l_ids_extracted, export=False)
            if _print:
                print(f"Cantidad de partidos faltantes en df_match: {df_match_miss.shape[0]}")

            # Concateno dfs
            df_match_concat = pd.concat([df_match_concat, df_match_miss], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_miss], axis=0)
            df_match_odds_concat =  pd.concat([df_match_odds_concat, df_match_odds_miss], axis=0)
            
        # Verificaciones
        if len(df_match_concat) > 0:

            ## Df_match_player
            if len(df_match_player_concat.columns) == 0:
                logger.error(f"No se tiene las formaciones de ninguno de los {len(df_match_player_concat.columns)} partido ya jugado. Esto es correcto solo si realmente no existe el dato de las formaciones para estos partidos.")
                raise ValueError

            ## Df_match_odds
            porcentaje_nan = df_match_odds_concat.isna().mean().mean()
            umbral = 0.5
            if porcentaje_nan > umbral: 
                logger.error(f"El DataFrame df_match_odds_concat tiene {porcentaje_nan:.2%} valores NaN, lo cual supera el umbral de {umbral:.2%}. Esto no es posible una vez jugado el partido, se deben tener las cuotas.")
                raise ValueError

            if len(df_match_odds_concat.columns) != 3:
                logger.error("No se recolectaron todas las odds en df_match_odds. Esto no es posible una vez jugado el partido, se debe tener las cuotas.")
                raise ValueError

        # Exporto datasets
        if self.export:
            df_match_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_miss.xlsx', index=True)
            df_match_player_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_player_miss.xlsx', index=True)
            df_match_odds_concat.to_excel(f'{self.path_missing}/data_understanding/df_match_odds_miss.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_match_odds_concat
    
class MissingData:
    def __init__(self, country, iteration_date, verbose: int = 1):
        self.country = country
        self.iteration_date = iteration_date
        self.verbose = verbose
        self.construct_directories()

    def construct_directories(self):
        self.BASE_DIR_du = f"./data/{self.country}/p2_data_understanding"
        self.BASE_DIR_dp = f"./data/{self.country}/p3_data_preparation/{self.iteration_date}"
        self.BASE_DIR_mod = f"./data/{self.country}/p4_modeling/{self.iteration_date}"
        
        path_missing = f'./data/{self.country}/p6_deployment/missing'
        self.BASE_DIR_MISSING_AND_OLD = f'{path_missing}/old_updated'
        self.BASE_DIR_MISSING_DP = f"{path_missing}/data_preparation"
        self.BASE_DIR_MISSING_ALL_du = f"{path_missing}/data_understanding/all"
        self.BASE_DIR_MISSING_ALL_dp = f"{path_missing}/data_preparation/all"
    
    def read_last_flashscore_data(self):
        """
        Obtengo ultima version de df_match, df_match_player y df_match odds (con missing)
        No lo pongo en la clase puesto que no son datos usados durante el entrenamiento.
        """
        try:
            df_match = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match.xlsx', index_col=0)
            df_match_player = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_player.xlsx', index_col=0)
            df_match_odds = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_odds.xlsx', index_col=0)
            logger.info('Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')

        except FileNotFoundError:
            
            user_input = input("No se encontraron los dfs con missing concatenados. ¿Quiere levantar los dataframes de partidos viejos? (y/n): ")
            if user_input.strip().lower()  == "y":
                df_match = pd.read_excel(f'data/{self.country}/p2_data_understanding/df_match.xlsx', index_col=0) 
                df_match_player = pd.read_excel(f'data/{self.country}/p2_data_understanding/df_match_player.xlsx', index_col=0) 
                df_match_odds = pd.read_excel(f'data/{self.country}/p2_data_understanding/df_match_odds.xlsx', index_col=0) 
            else:
                raise SystemExit("⛔ Predicción cancelada por el usuario.")
        
        logger.info(f"Shapes: \t df_match: {df_match.shape} \t df_match_player:{df_match_player.shape} \t df_match_odds: {df_match_odds.shape}")
        return df_match, df_match_player, df_match_odds

    def read_last_integrate_data(self):

        # Levanto df_missing o corro la extraccion con el concat de df_match y df_match_missing (en vez de df_match solo pues sino siempre levanta los mismos partidos y cada vez mas...)
        try:
            df_integrated = pd.read_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_integrated.xlsx', index_col=0)
            logger.info('Se levantó el dataframe de partidos viejos concatenado con algunos partidos missing concatenados.')

        # Si no existe un df_integrated concatenado entre old y missing
        except FileNotFoundError:

            warning_msg = f"No se pudo levantar el df_integrated con old + missing. Esto es correcto solo si nunca se ha extraido / integrado missing. Desea levantar el df_integrated con el que se entrenó? (y/n): "
            user_input = input(warning_msg).strip().lower() 
            if user_input == "y":
                df_integrated = pd.read_excel(f'{self.BASE_DIR_dp}/df_integrated.xlsx', index_col=0)  # Tiene missing hasta el dia en el que entrené (por no desde ese dia en adelante)
                logger.warning('Se levantó el dataframe de partidos viejos puesto que no se encontró con missing concatenados.')
            else: 
                raise SystemExit("⛔ Predicción cancelada por el usuario.")            
        
        logger.info(f"df_integrated: {df_integrated.shape}")
        return df_integrated

    def read_last_missing_data(self):
        """
        Leer todos los datos missing ya extraidos
        """
        try:
            df_match_miss = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_miss.xlsx', index_col=0)
            df_match_player_miss = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_player_miss.xlsx', index_col=0)
            df_match_odds_miss = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_odds_miss.xlsx', index_col=0)

        # Si es la primera vez que extraigo partidos missing
        except FileNotFoundError:

            warning_msg = f"No se pudo levantar datos missing ya extraidos. Esto es correcto solo si nunca se ha extraido missing. Desea inicializar crear los dataframes? (y/n): "
            user_input = input(warning_msg).strip().lower() 
            if user_input == "y":
                df_match_miss, df_match_player_miss, df_match_odds_miss = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            else: 
                raise SystemExit("⛔ Predicción cancelada por el usuario.")            

        return df_match_miss, df_match_player_miss, df_match_odds_miss
    
    def read_last_integrate_missing_data(self):
        try:
            df_integrated_missing_all = pd.read_excel(f'{self.BASE_DIR_MISSING_ALL_dp}/df_integrated_missing.xlsx', index_col=0)

        # Si es la primera vez que extraigo partidos missing
        except FileNotFoundError:
            logger.error("Nunca se ha integrado missing")
            warning_msg = f"No se pudo levantar el df_integrated_missing. Esto es correcto solo si nunca se ha integrado missing. Desea inicializar crear el dataframe? (y/n): "
            user_input = input(warning_msg).strip().lower() 
            if user_input == "y":
                df_integrated_missing_all = pd.DataFrame()  # Es importante para que se guarde por primera vez df_integrated_missing en /all 
            else: 
                raise SystemExit("⛔ Predicción cancelada por el usuario.")
        
        return df_integrated_missing_all

    def concat_old_with_missing(self, df_match, df_match_player, df_match_odds, df_match_miss, df_match_player_miss, df_match_odds_miss):
        """
        Exporta datos de partidos con los que se entrena el modelo y los partidos missing,
        evitando duplicar datos ya existentes en los DataFrames originales.
        """
        len_inicial = len(df_match)
        len_inicial_miss = len(df_match_miss)

        # Función auxiliar para evitar duplicados antes de concatenar
        def avoid_duplicate_concat(df_original, df_miss):
            return pd.concat(
                [df_original, df_miss.loc[~df_miss.index.isin(df_original.index)]],
                axis=0
            )
        
        # Concatenar evitando duplicados
        df_concat_match = avoid_duplicate_concat(df_match, df_match_miss)
        df_concat_match_player = avoid_duplicate_concat(df_match_player, df_match_player_miss)
        df_concat_match_odds = avoid_duplicate_concat(df_match_odds, df_match_odds_miss)

        len_final = len(df_concat_match)
        logger.warning(f"Old: {df_match.shape} + Missing: {df_match_miss.shape} = {df_concat_match.shape}")

        # Verificación
        verif = (len_inicial + len_inicial_miss) == len_final
        if not verif:
            logger.error("Fallo la concatenación de partidos missing a los datos viejos")

        # Exportar datos
        df_concat_match.to_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match.xlsx')
        df_concat_match_player.to_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_player.xlsx')
        df_concat_match_odds.to_excel(f'{self.BASE_DIR_MISSING_AND_OLD}/df_match_odds.xlsx')
        logger.info(f"Shape de df_match concatenado con missing: {len_inicial} --> {len_final}")

    def concat_with_missing_already_extracted(self, df_match_miss, df_match_player_miss, df_match_odds_miss, df_match_miss_comp, df_match_player_miss_comp, df_match_odds_miss_comp):
        """
        Guarda los nuevos partidos missing con los que ya tenía, evitando duplicados.

        Funciona pero hay que ver cuando hay repetidos si los evita... (si corres bien, nunca deberia siquiera tener que evitarlo... pero bueno).
        """
        # Identificar las claves primarias únicas en los datos previos
        keys_match = df_match_miss_comp.index if df_match_miss_comp.index.is_unique else df_match_miss_comp['id_match']
        keys_player = df_match_player_miss_comp.index if df_match_player_miss_comp.index.is_unique else df_match_player_miss_comp['id_player']
        keys_odds = df_match_odds_miss_comp.index if df_match_odds_miss_comp.index.is_unique else df_match_odds_miss_comp['id_match']

        # Filtrar nuevos registros que no estén ya en los datos previos
        df_match_miss_new = df_match_miss[~df_match_miss.index.isin(keys_match)]
        df_match_player_miss_new = df_match_player_miss[~df_match_player_miss.index.isin(keys_player)]
        df_match_odds_miss_new = df_match_odds_miss[~df_match_odds_miss.index.isin(keys_odds)]

        # Concatenar solo los registros nuevos
        df_match_miss_comp_ct = pd.concat([df_match_miss_comp, df_match_miss_new], axis=0)
        df_match_player_miss_comp_ct = pd.concat([df_match_player_miss_comp, df_match_player_miss_new], axis=0)
        df_match_odds_miss_comp_ct = pd.concat([df_match_odds_miss_comp, df_match_odds_miss_new], axis=0)

        logger.info(f"Se han añadido {len(df_match_miss_new)} nuevos registros a df_match_miss.")
        logger.info(f"Se han añadido {len(df_match_player_miss_new)} nuevos registros a df_match_player_miss.")
        logger.info(f"Se han añadido {len(df_match_odds_miss_new)} nuevos registros a df_match_odds.")

        # Exporto datos
        df_match_miss_comp_ct.to_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_miss.xlsx', index=True)
        df_match_player_miss_comp_ct.to_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_player_miss.xlsx', index=True)
        df_match_odds_miss_comp_ct.to_excel(f'{self.BASE_DIR_MISSING_ALL_du}/df_match_odds_miss.xlsx', index=True)

########################################################################## MAIN #######################################################################
def main(l_countries: list, iteration_date: str, extract_missing: bool = True, n_seasons_missing : int = 1, verbose: int = 1, export: bool = True):
    """
    Recoleccion de partidos "missing" (partidos ya jugados pero que no se usaron para entrenar el modelo).
    """
    # Determino competencias a extraer
    d_comps = determine_country_competitions(l_countries)
    df_comp = pd.read_excel('./data/df_competencies.xlsx')
    df_comp_country = df_comp[df_comp['id_competition'].isin(d_comps['all_comp'])] 
    print(d_comps['all_comp'])

    # Levanto datos: old + los ultimos missing extraidos
    df_match_upd, df_match_player_upd, df_match_odds_upd = mis.read_last_flashscore_data() # Last df_integrated con missing + old

    # _____________________________________________________________ EXTRACT MISSING DATA _____________________________________________________________ #
    du = DataUnderstandingMissing(l_countries, export=export) # Creo objeto de clase DataUnderstanding
    if extract_missing:
        # Extraer partidos missing teniendo en cuenta df_match + df_match_missing
        df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new = du.collect_missing_data(df_match_upd, df_comp_country=df_comp_country, n_seasons_max=n_seasons_missing)
        logger.info(f"Cantidad de partidos missing extraidos: {len(df_match_miss_new)}")
    else:
        # Unicamente util para cuando falla la preparacion de missing pero ya extrajiste...
        df_match_miss_new = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/df_match_miss.xlsx', index_col=0)
        df_match_player_miss_new = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/df_match_player_miss.xlsx', index_col=0)
        df_match_odds_miss_new = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/df_match_odds_miss.xlsx', index_col=0)
        print("Shape:", df_match_miss_new.shape, df_match_player_miss_new.shape, df_match_odds_miss_new.shape)

    # _____________________________________________________________ PREPARE MISSING DATA _____________________________________________________________ #
    iteration_date_dt = pd.to_datetime(iteration_date, format='%Y-%m-%d').date()  # con .date() saco hora y minutos

    dp = DataPreparation(l_countries=l_countries, date=iteration_date_dt, export=export) # Creo objeto de clase DataPreparation

    # Paths
    DIR = f"./data/{iteration_date_dt}/{l_countries}"
    DIR_sofifa = f"./data/sofifa"

    # Read data
    mis = MissingData(country=country, iteration_date=iteration_date_dt)

    ## Last missing data
    df_match_miss, df_match_player_miss, df_match_odds_miss = mis.read_last_missing_data()
    df_integrated_missing = mis.read_last_integrate_missing_data()
    df_integrated_upd = mis.read_last_integrate_data() # Last df_integrated con missing + old   

    ## SOFIFA
    df_player_sofifa = pd.read_excel(f"{DIR_sofifa}/df_player_sofifa.xlsx", index_col=0)
    df_player_fifa_sofifa = pd.read_excel(f"{DIR_sofifa}/df_player_fifa_sofifa.xlsx")  

    # Si extrajo missing
    if len(df_match_miss_new) > 0:
        
        # Preparo datos missing            
        df_match_miss_new_f, df_match_player_miss_new_f, df_match_odds_miss_new_f, df_player_fifa_sofifa = dp.format_data(df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new, df_player_fifa_sofifa, reformat=True, export=False)            
        df_match_miss_new_c, df_match_player_miss_new_c, df_player_sofifa, df_player_fifa_sofifa = dp.clean_data(df_match_miss_new_f, df_match_player_miss_new_f, df_player_sofifa, df_player_fifa_sofifa, export=False)
        # df_match_miss_new_vf, df_match_player_miss_new_vf, df_match_odds_miss_new_vf, df_player_sofifa, df_player_fifa_sofifa = dp.verify_format(df_match_miss_new_c, df_match_player_miss_new_c, df_match_odds_miss_new_f, df_player_sofifa, df_player_fifa_sofifa, prod=False) # prod=False pues los partidos ya se jugaron..
        df_integrated_missing_new = dp.integrate_data(df_match_miss_new_c, df_match_player_miss_new_c, df_player_sofifa, df_player_fifa_sofifa, prod=True, export=False) 

        # Concateno missing y old (que puede tener algunos missing ya)
        df_integrated_updated = pd.concat([df_integrated_upd, df_integrated_missing_new], axis=0)
        df_integrated_updated = df_integrated_updated[~df_integrated_updated.index.duplicated(keep='first')]  # El df_integrated tiene missing hasta el dia en que entrené
        logger.warning(f"Concatenación old + missing: {df_integrated_upd.shape} + {df_integrated_missing_new.shape} --> {df_integrated_updated.shape} (si nunca preparaste, missing no se agrega pues ya está)")

        # Guardo registro de todos los partidos missing juntos (los recien recolectados y los que ya tenia)
        df_integrated_missing_all = pd.concat([df_integrated_missing, df_integrated_missing_new], axis=0)
        
        if export:
            if extract_missing:    
                # Exporto datos extraidos una vez que la integracion funcionó (sino lo extrae pero no lo integra) --> # Mucho cuidado si falla la preparacion pues los missing estaran en old_updated pero no integrados correctamente. (deberias exportar si la prep funciona o algo asi)
                mis.concat_with_missing_already_extracted(df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new, df_match_miss, df_match_player_miss, df_match_odds_miss)  # missing all --> NO HACERLO CUANDO SOLO QUIERO PREPARAR... Deberia evitar que concatene si los partidos missing ya estan...
                mis.concat_old_with_missing(df_match_upd, df_match_player_upd, df_match_odds_upd, df_match_miss_new, df_match_player_miss_new, df_match_odds_miss_new) # Old + missing # # No lo quiero cuando ya extraje missing y solo quiero preparar...
            df_integrated_missing_new.to_excel(f'{mis.BASE_DIR_MISSING_DP}/df_integrated_missing.xlsx', index=True)
            df_integrated_updated.to_excel(f'{mis.BASE_DIR_MISSING_AND_OLD}/df_integrated.xlsx', index=True)
            df_integrated_missing_all.to_excel(f'{mis.BASE_DIR_MISSING_ALL_dp}/df_integrated_missing.xlsx', index=True)

    else:
        df_integrated_updated = df_integrated_upd.copy()
        logger.warning(f"Ya se habian extriado todos los partidos missing. Aun no hay partidos nuevos. {df_integrated_updated.shape}")
       
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":    
    directorio = os.getenv('BASE_DIR_LOCAL')

    # Defino country
    d_countries = {
        48: ["england", '2025-08-13'], 
        55: ["france", '2025-05-07'], 
        59: ["germany", '2025-05-08'], 
        77: ["italy", '2025-05-08'],
        148: ["spain", '2025-05-07'], 
        167: ["usa", '2025-05-29'], 
        }

    l_countries= [48, 55]
    n_days = 1

    # iteration date y modelo
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]
    d_model = {'n_model': 5, 'model_name': "XGBClassifier"}

    
    logger.warning("Extract and prepare missing matches")
    df = main(l_countries, iteration_date=iteration_date, export=True) 

    if isinstance(df, pd.DataFrame):
        df.to_excel(f"{directorio}/predicciones.xlsx")