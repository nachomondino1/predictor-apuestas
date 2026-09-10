# Importo librerias
import os
from utils import directories
from utils.set_up_logging import logger
from dotenv import load_dotenv
import pandas as pd
from p2_data_understanding.collect_initial_data import scraper_sofifa
from p3_data_preparation.format_data import verify_format


def get_player_data(id_country, country, df_comp_country, n_seasons_update, path_save):
    """
    Levanta datos de sofifa ya extraidos y actualiza las ultimas temporadas
    """
    directories.make_directories(l_directorios=[path_save])
    df_player_sofifa, df_player_fifa_sofifa = pd.DataFrame(), pd.DataFrame()

    # POR COMPETITION
    for i, row in df_comp_country.iterrows():
        print(f' Competition: {row["competition_flashscore"]} '.center(120, '+'))

        if row['id_competition'] in [62, 1673]:
            logger.warning(f"Evito extraccion para {row["competition_flashscore"]}")
            continue

        # Extraigo datos de Sofifa 
        ## Player
        df_player_sofifa_comp, df_player_fifa_sofifa_comp = scraper_sofifa.extract_players(id_country, country, row['id_competition'], row['competition_sofifa'], n_seasons=n_seasons_update, export=True)

        # Concateno datos de competiciones del pais
        df_player_sofifa = pd.concat([df_player_sofifa, df_player_sofifa_comp], axis=0)
        df_player_fifa_sofifa = pd.concat([df_player_fifa_sofifa, df_player_fifa_sofifa_comp], axis=0)
 
        # Exporto datos x seguridad
        df_player_sofifa_comp.to_excel(f'{path_save}/df_player_{country}_{row['competition_sofifa']}.xlsx', index=True)
        df_player_fifa_sofifa_comp.to_excel(f'{path_save}/df_player_fifa_{country}_{row['competition_sofifa']}.xlsx', index=False)
        df_player_sofifa.to_excel(f'{path_save}/df_player_{country}.xlsx', index=True)
        df_player_fifa_sofifa.to_excel(f'{path_save}/df_player_fifa_{country}.xlsx', index=False)

    ## Exporto datos (ya listos para usar en main.py)
    df_player_sofifa.to_excel(f'{path_save}/df_player_sofifa.xlsx', index=True)
    df_player_fifa_sofifa.to_excel(f'{path_save}/df_player_fifa_sofifa.xlsx', index=True)
    print(f"Shape final: {df_player_sofifa.shape} {df_player_fifa_sofifa.shape}")

    # Verificar formato de datos
    df_player_sofifa = format_df_player_sofifa(df_player_sofifa)
    df_player_fifa_sofifa = format_df_player_fifa_sofifa(df_player_fifa_sofifa)

    return df_player_sofifa, df_player_fifa_sofifa

def read_last_player_data(country, verbose: int = 0):
    
    base_path = f'data/{country}/p2_data_understanding'

    # Levanto datos viejos
    df_player_sofifa_old = pd.read_excel(f'{base_path}/df_player_sofifa.xlsx', index_col=0)
    df_player_fifa_sofifa_old = pd.read_excel(f'{base_path}/df_player_fifa_sofifa.xlsx') # index_col=0
    print(f"Shape inicial: {df_player_sofifa_old.shape} {df_player_fifa_sofifa_old.shape}")
    print(df_player_sofifa_old)
    print(df_player_fifa_sofifa_old)

    if 'Unnamed: 0' in df_player_fifa_sofifa_old.columns:
        df_player_fifa_sofifa_old = df_player_fifa_sofifa_old.drop(columns=['Unnamed: 0'])

    if verbose >= 2:
        logger.info(df_player_sofifa_old)
        logger.info(df_player_fifa_sofifa_old)
    
    return df_player_sofifa_old, df_player_fifa_sofifa_old

def concat_player_data(df_player, df_player_fifa, df_player_sofifa_old, df_player_fifa_sofifa_old, path_save):
    """
    Reemplazo fifas que se han extraido en los datos viejos. En caso de ser un nuevo fifa, lo "agrega" en vez de "reemplazar".
    
    # Parameters:
        df_player_sofifa: Datos de Sofifa recien extraidos (DataFrame)
        df_player_fifa_sofifa: Datos de Sofifa recien extraidos (DataFrame)
        base_path: Ruta de donde obtener los datos de sofifa viejos.

    # Return
        df_player_sofifa_ct: Datos de sofifa viejos con los ultimos fifas extraidos. (DataFrame)
        df_player_fifa_sofifa_ct: Datos de sofifa viejos con los ultimos fifas extraidos. (DataFrame)
    """    
    # Determinar que fifas a actualizar
    l_fifas_extracted = df_player_fifa['fifa'].unique()
    print(f"Fifas a actualizar: {l_fifas_extracted}")
    print(f'Shapes new_data: \n df_player_sofifa {df_player.shape} \n df_player_fifa_sofifa: {df_player_fifa.shape}')

    # Elimino fifas actualizados en los datos viejos
    df_player_fifa_sofifa_old_filt = df_player_fifa_sofifa_old[~df_player_fifa_sofifa_old['fifa'].isin(l_fifas_extracted)]
    l_ids_que_quedan = df_player_fifa_sofifa_old_filt['id_player'].unique()
    df_player_sofifa_old_filt = df_player_sofifa_old[df_player_sofifa_old.index.isin(l_ids_que_quedan)]
    print(f'Shapes luego de eliminar fifas a actualizar: \n df_player_sofifa {df_player_sofifa_old.shape} --> {df_player_sofifa_old_filt.shape} \n df_player_fifa_sofifa: {df_player_fifa_sofifa_old.shape} --> {df_player_fifa_sofifa_old_filt.shape}')

    # Concateno datos 1) viejos sin fifas actualziados + 2) datos recien extraidos
    df_player_ct = pd.concat([df_player, df_player_sofifa_old_filt], axis=0)
    df_player_fifa_ct = pd.concat([df_player_fifa, df_player_fifa_sofifa_old_filt], axis=0)
    print(f"Shape concat: \n {df_player_ct.shape} \n {df_player_fifa_ct.shape}")

    # Elimino duplicados (Por que no funciona como el de clean_data de main.py? Deja duplicados... y en clean los elimina bien. Tal vez x formato del idx?)
    df_player_sofifa_filt = df_player_ct[~df_player_ct.index.duplicated(keep='first')]  # Esta es la que uso en clean_data
    df_player_fifa_sofifa_filt = df_player_fifa_ct.drop_duplicates(subset=['id_player', 'fifa', 'date'])  # No uso id_competition? Creo que no pues ya mapié por competicion antes... y no lo hare a futuro pues usare este df
    print(f"Shape concat sin duplicados: \n {df_player_ct.shape} --> {df_player_sofifa_filt.shape} \n {df_player_fifa_ct.shape} --> {df_player_fifa_sofifa_filt.shape}")

    # Exporto datos
    df_player_sofifa_filt.to_excel(f'{path_save}/df_player_sofifa.xlsx', index=True)
    df_player_fifa_sofifa_filt.to_excel(f'{path_save}/df_player_fifa_sofifa.xlsx', index=True)

    return df_player_sofifa_filt, df_player_fifa_sofifa_filt

def format_df_player_sofifa(df):

    # Formateo columnas int
    # df.index = df.index.astype(str)
    # df['height'] = df['height'].astype(int)

    column_specs = {
        'player_name': {'dtype': str},
        'player_name_short': {'dtype': str},
        'nationality': {'dtype': str},
        'height': {'dtype': int, 'rango': [100, 250]},
        'preferred_foot': {'dtype': str},
        'url_player': {'dtype': str},
        }
    
    return verify_format(df, column_specs)


def format_df_player_fifa_sofifa(df):
    """
    Debo reformatear campos de sofifa extraidos nuevos. Esta fallando 'age' porque ahora es string en vez de int?
    """
    # Formateo columnas int
    # df['id_player'] = df['id_player'].astype(str)
    # df['age'] = df['age'].astype(int)
    # df['overall_rating'] = df['overall_rating'].astype(int)
    # df['potential'] = df['potential'].astype(int)
    # df['int_reputation'] = df['int_reputation'].astype(int)
    
    column_specs = {
        # Verifico formato
        'id_player': {'dtype': str},
        'date': {'dtype': 'datetime64[ns]'},
        'id_country': {'dtype': int, 'rango': [0, 300]},
        'id_competition': {'dtype': int, 'rango': [0, 10000]},
        'id_team': {'dtype': int},
        'age': {'dtype': int, 'rango': [14, 50]},
        'overall_rating': {'dtype': int, 'rango': [20, 100]},
        'potential': {'dtype': int, 'rango': [20, 100]},
        'value': {'dtype': str},  # Se extrae como str
        'wage': {'dtype': str}, # Se extrae como str
        'int_reputation': {'dtype': int, 'rango': [0, 5]},
        'fifa': {'dtype': str},
        'fifa_year': {'dtype': int, 'rango': [6, 30]},
        }
    
    return verify_format(df, column_specs)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # try Format
    df1 = pd.read_excel('data/spain/p2_data_understanding/sofifa_update/2025-01-19/data_seg/df_player_sofifa.xlsx')
    df2 = pd.read_excel('data/spain/p2_data_understanding/sofifa_update/2025-01-19/data_seg/df_player_fifa_sofifa.xlsx')

    format_df_player_sofifa(df1)
    format_df_player_fifa_sofifa(df2)

    '''

    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    # Seleccionar pais  
    id_country = 59
    n_seasons_to_extract = 2

    # Levanto dataframes
    df_countries = pd.read_excel('./data/df_countries.xlsx')
    df_comp = pd.read_excel('./data/df_competencies.xlsx')
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0]
    df_comp_country = df_comp[(df_comp['id_country'] == id_country) & (df_comp['is_cup'] == 0)]
    print(f' COUNTRY: {country} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_country['competition_flashscore']}")
    
    date = datetime.datetime.now().date()
    path_save = f'data/{country}/p2_data_understanding/sofifa_update/{date}'
    path_save_seg = f'{path_save}/data_seg'

    # Levanto los datos viejos
    df_player_sofifa_old, df_player_fifa_sofifa_old = read_last_player_data()

    # Obtener ultimas seasons de Sofifa
    df_player, df_player_fifa = get_player_data(id_country, df_comp_country, n_seasons_update=n_seasons_to_extract, path_save=path_save_seg, verbose=1)

    # Actualizar sofifa con las ultimas seasons
    concat_player_data(df_player, df_player_fifa, df_player_sofifa_old, df_player_fifa_sofifa_old, path_save=path_save)
    '''