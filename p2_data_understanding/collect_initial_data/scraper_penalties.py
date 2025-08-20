import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from tqdm import tqdm
from p2_data_understanding.collect_initial_data.scraper_flashscore import FlashscoreCrawler
from p3_data_preparation.format_data import convert_goals_to_int

def extract_penalties(df, export: bool = True):
    """
    # Parameters:
        df: Dataframe 

    # Returns:
        df_match: Dataframe.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    crawler = FlashscoreCrawler(headless=True)
    country = df['id_country'].unique()
    print(country)

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.driver.get("https://www.flashscore.com/")  # hasta que no se carga toda la pagina, no sigue...
    crawler.accept_cookies()

    # Filtro partidos por copa y dif_goals = 1
    df_filt = df[df['is_cup'] == 1]
    print(df.shape)

    df_aux = df_filt.copy()
    df_aux['dif_goals'] = df_aux['goals_home'] - df_aux['goals_away']
    df_aux['dif_goals_abs'] = df_aux['dif_goals'].abs()
    df_aux = df_aux[df_aux['dif_goals_abs'] == 1]    # Ahora, puedes filtrar basándote en esta nueva columna
    df_filt = df_filt[df_filt.index.isin(df_aux.index)]
    print(df_filt.shape)

    progress_bar = tqdm(total=len(df_filt), ncols=80)  # Inicializo barra de progreso
    list_match_data = []

    # POR MATCH (c/u identificado con un id)
    for id_match, row in df_filt.iterrows():

        # Ingreso a pagina de informacion del match
        url_match = f'https://www.flashscore.com/match/{id_match}/#/match-summary'
        crawler.driver.get(url_match)
 
        d_row_match = {'id_match': id_match}
        d_row_match.update(crawler.extract_disclaimer_data())
        print()
        print(d_row_match)
        
        # Agregar el diccionario a la lista
        list_match_data.append(d_row_match)
        progress_bar.update(1)

    progress_bar.close()

    # Convertir la lista de diccionarios en un DataFrame
    df_match = pd.DataFrame(list_match_data)

    # Establecer 'id_match' como el índice del DataFrame
    if not df_match.empty:
        df_match.set_index('id_match', inplace=True)

    print(df_match.head())

    if export:
        # Guardo partidos de la competition
        df_match.to_excel(f'./{country}.xlsx', index=True)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_match

def add_neutralidad(df, df_match):
    # Crear la columna 'neutral' en el DataFrame principal
    df['neutral'] = 0

    # Obtener una lista de los 'id_match' de los partidos neutrales. Esto es mucho más seguro que usar el índice
    neutral_match_ids = df_match[df_match['neutral'] == 1].index.tolist()

    # Usar .isin() para crear una máscara booleana que identifique las filas en `df` con un 'id_match' que esté en la lista de partidos neutrales
    mask = df.index.isin(neutral_match_ids)

    # Asignar el valor 1 a la columna 'neutral' para esas filas
    df.loc[mask, 'neutral'] = 1

    return df

def corregir_goals(df, df_match):

    # Creo la columna
    df['penalties'] = "FINISHED"

    # Identifica los índices de los partidos con penales
    idxs_extra_time = df_match[df_match['penalties'] == "AFTER EXTRA TIME"].index.tolist()
    idxs_penalties = df_match[df_match['penalties'] == "AFTER PENALTIES"].index.tolist()

    # Usar .isin() para crear una máscara booleana que identifique las filas en `df` con un 'id_match' que esté en la lista de partidos neutrales
    mask2 = df.index.isin(idxs_extra_time)
    mask = df.index.isin(idxs_penalties)

    # Reemplazo valor 
    df.loc[mask2, 'penalties'] = "AFTER EXTRA TIME"
    df.loc[mask, 'penalties'] = "AFTER PENALTIES"
    combined_mask = mask | mask2


    # For matches that went to penalties, make goals_home and goals_away equal to the minimum.
    # First, calculate the minimum goals for the relevant rows.
    min_goals = df.loc[combined_mask, ['goals_home', 'goals_away']].min(axis=1)

    # Now, assign this `min_goals` series to both columns in a single line.
    df.loc[combined_mask, ['goals_home', 'goals_away']] = min_goals
    return df
 

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    l_countries = [55, 59, 77, 148, 48]
    l_countries = [48]

    d_countries = {
        48: "england", 
        55: "france",
        59: "germany",
        77: "italy",
        148: "spain",
        167: "usa"
        }
    
    for id_country in l_countries:

        country = d_countries[id_country]

        df = pd.read_excel(f'data/{country}/p6_deployment/missing/old_updated/df_match_old.xlsx', index_col=0)
        print(df.head(5))
        print(df.shape)

        df = convert_goals_to_int(df)

        # Extraigo neutralidad y penalties
        df_match = extract_penalties(df)
        # df_match = pd.read_excel('[55].xlsx', index_col=0)

        # Corrigo data
        df = add_neutralidad(df, df_match)
        df = corregir_goals(df, df_match)
        df.to_excel(f'data/{country}/p6_deployment/missing/old_updated/df_match.xlsx', index=True)
