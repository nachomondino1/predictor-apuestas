# Juntar entidades jugadores, partido y atrib_playeradores
import pandas as pd
from fuzzywuzzy import fuzz
import warnings
import time
from tqdm import tqdm
import math


def match_players_by_name(df_match_player, df_player):
    """
    Match entre jugadores de Flashscore y de Sofifa mediante name de jugadores
    """
    # Obtengo listado de jugadores unicos tanto en df_player (Sofifa) como en df_match_player (Flashscore) para agilizar vinculacion
    df_match_player_unique_players = unique_players_df_match_player(df_match_player)
    df_player_unique_players = df_player.drop_duplicates(subset=['id_player']).loc[:, ['id_player', 'name']]

    # Vinculo con "id_player" a df_player (Sofifa) y df_match_player (Flashscore) utilizando los nombres de los jugadores
    df_map_players_name_id = match_unique_players(df_match_player_unique_players, df_player_unique_players)
    return df_map_players_name_id

def unique_players_df_match_player(df_match_player):
    """
    Obtencion de listado de jugadores unicos de df_match_player. Un mismo jugador se repite varias veces porque esta en mas
    de un partido.
    :param df_match_player: Dataframe. Unidad de analisis: partido. Columnas: una por jugador segun formaciones.
    :return: Dataframe. Unidad de analisis: jugador. Una sola columna con nombres de los jugadores en df_match_player
    (sin repetidos).
    """
    # Definicion de variables
    set_unique_players = set()

    # Por columna (e.g. player_start_home_7)
    for col in list(df_match_player.columns):

        # Obtengo jugadores unicos y agrego al set
        l_unique_players = df_match_player[col].unique()
        set_unique_players.update(l_unique_players)  #  Uso set puesto que un jugador puede estar en mas de una columna

    # Creo dataframe con listado de jugadores unicos
    df = pd.DataFrame(list(set_unique_players), columns=['player_name'])
    df = df.dropna() # No se porque le queda un na
    return df

def match_unique_players(df_match_player_up, df_player_up): # Es la que hay que eficientizar (Le agregue progress bar para ver velocidad)
    """
    Vinculo datasets de los jugadores de Sofifa y los jugadores de Flashscore segun name de jugador.
    :param df_match_player_up: Dataframe. Unidad de analisis: jugador. Una sola columna con name de los jugadores en
    df_match_player (sin repetidos).
    :param df_player_up: Dataframe. Unidad de analisis: jugador. Columnas id_player y name. La columna "name" tiene los
    nombres de los jugadores en df_player (sin repetidos).
    :return: Dataframe. Unidad de analisis: jugador. df_match_player pasado como parametro con columna "id_player" de
    df_player gracias a vincular nombres de jugadores de sendos dataframes.
    """
    print("Matching players from Sofifa and Flashscore by name...")

    # Definicion de variables
    df_match_player_with_id = df_match_player_up.copy()  # Creo copia del dataframe df_match_player en el que agregar la columna "id_player"
    l_umbrales = [95, 90, 85, 80, 75]
    n_pos_matchs, n_matchs = len(df_match_player_up), 0

    # Funcion que hace una busqueda aproximada de un string en una columna
    def buscar_coincidencias(row, palabra, columna, umbral):
        return fuzz.token_set_ratio(palabra, row[columna]) >= umbral

    # Inicializo barra de progreso
    progress_bar = tqdm(total=len(df_match_player_up), ncols=80)

    # Por jugador en df_match_player
    for i, row in df_match_player_up.iterrows():
        progress_bar.update(1)

        # Filtro inicial. Me quedo con los jugadores con name mas parecido (agiliza enormemente la funcion)
        df_player_filt_ini = df_player_up[df_player_up.apply(buscar_coincidencias, args=(row['player_name'], 'name', min(l_umbrales)), axis=1)]

        # Por umbral
        for umbral in l_umbrales:

            # Selecciono los jugadores con name mas parecido al buscado
            df_player_filt = df_player_filt_ini[df_player_filt_ini.apply(buscar_coincidencias, args=(row['player_name'], 'name', umbral), axis=1)]

            # Si hay al menos un posible match
            if len(df_player_filt) >= 1:

                # Agrego id_player de Sofifa como columna en df_match_player
                df_match_player_with_id.loc[i, 'id_player'] = df_player_filt.id_player.values[0]
                df_match_player_with_id.loc[i, 'nombre_sofifa'] = df_player_filt.name.values[0]  # temporalmente para analizar calidad de match

                # Elimino jugador de df_player que hizo match para agilizar la busqueda
                df_player_up = df_player_up.drop(df_player_filt.index[0])
                n_matchs += 1
                break    
    progress_bar.close()

    try:
        print(f"De los {n_pos_matchs} jugadores en df_match_player, hizo match para {n_matchs/n_pos_matchs*100:.2f}% de ellos, es decir, para {n_matchs}.")
    except ZeroDivisionError:
        print("No se cuenta con las formaciones de ningun partido de df_match, por ende, df_match_player no tiene que integrar a df_match.")

    return df_match_player_with_id

def replace_players_name_with_id(df_match_player, df_map_players_name_id):

    print("Replacing player's names by id in df_match_player...")
    progress_bar = tqdm(total=len(df_map_players_name_id), ncols=80)

    # Por jugador
    for i, row in df_map_players_name_id.iterrows():

        # Reemplazo su name por su id en df_match_player
        df_match_player = df_match_player.replace(row['player_name'], row['id_player'])
        progress_bar.update(1)

    progress_bar.close()
    return df_match_player

def integrate_player_data_in_match(df_match, df_match_player, df_player): 
    """
    Integra la entidad jugador en la entidad partido. Es decir, sintetiza los datos de los jugadores a cada partido en
    particular. Se determinan los promedios de age, overall rating, value de mercado y height del equipo titular,
    suplente y los ausentes para cada equipo.
    :param df_match: Dataframe. Unidad de analisis: partido. Columnas: equipos, arbitros, estadisticas del partido, etc.
    :param df_match_player: Dataframe. Unidad de analisis: partido. Columnas: id_match y una por jugador segun formaciones.
    Celdas: id de jugador (en vez de name).
    :param df_player: Dataframe. Unidad de analisis: jugador. Columnas: id_player y datos del jugador como age y overall
    rating.
    :return: Dataframe. Dataframe con los datos de todos los dataframes pasados como parametro. Tod@ en un solo
    dataframe para poder entrenar un modelo con ellos.
    """
    print("\nIntegrating all dataframes in just one dataframe...")
    # Definicion de variables
    warnings.filterwarnings('ignore')  # Ver el ignore, y solucionarlo en vez de ignorarlo...
    l_titularidad = ['start', 'sub', 'miss']  # tendria que agregar 'sup_ing' pero se debe procesar con sup...
    l_condicion = ['home', 'away']

    # Agrego columna "fifa_year" quedandome solo con el año del fifa (e.g. "22" en vez de "FIFA 22")
    df_player['fifa_year'] = df_player['fifa'].str.split(' ').str[-1]

    # Por titularidad (Titular, suplente o ausente)
    for titularidad in l_titularidad:

        # Por condicion (Local o visitante)
        for condicion in l_condicion:

            # Defino pattern y con el, selecciono las variables a procesar
            pattern = f'player_{titularidad}_[a-z]*[_]*{condicion}_[0-9]+'  # CAMBIE EL PATTERN PARA QUE SUP Y SUP_ING SEAN PROCESADOS JUNTOS. VERIFICAR QUE FUNCIONA..
            l_col_to_preprocess = df_match_player.filter(regex=pattern, axis=1).columns.tolist()  # LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...
            print(f'Columnas a procesar: {l_col_to_preprocess}')

            # Inicializo barra de progreso
            progress_bar = tqdm(total=len(df_match_player), ncols=80)

            # Por partido
            for i, row in df_match.iterrows():

                progress_bar.update(1)
                # print(f' Partido Nº: {i} '.center(120, '#'))

                # Busco el fifa correspondiente segun la fecha del partido
                year_fifa = search_fecha_fifa(row['date'])
                # print(f"Fecha partido: {row['fecha']} --> Fifa a buscar: {fecha_part_fifa}")

                # Reinicio variables
                l_mean_age, l_mean_hei, l_mean_rating, l_mean_val = [], [], [], []

                # Por jugador
                for col_player in l_col_to_preprocess:

                    # Busco id del jugador en df_match_player
                    id_player_ent_part = df_match_player.loc[i, col_player]
                    # print(f"\t Id jugador a buscar en Sofifa: {id_player_ent_part}")

                    # Si el id_player no es nan
                    if not math.isnan(id_player_ent_part):  # hay mucho nan sobretodo columnas de jugadores ausentes (e.g. player_aus_vis_12)

                        # Busco el id y la fecha en df_player (Sofifa)
                        df_player_filt = df_player[(df_player['id_player'] == id_player_ent_part) & (df_player['fifa_year'] == year_fifa)]

                        # Guardo datos del jugador
                        if len(df_player_filt) > 0:
                            l_mean_age.append(df_player_filt.age.values[0])
                            l_mean_hei.append(df_player_filt.height.values[0])
                            l_mean_rating.append(df_player_filt.overall_rating.values[0])
                            l_mean_val.append( df_player_filt.value.values[0])
                            # print("Ejemplo de lista promedio de age: ", l_mean_age)

                # Guardo promedios de age, height, overall_rating y market value
                try:
                    df_match.loc[i, f'mean_age_player_{titularidad}_{condicion}'] = sum(l_mean_age) / len(l_mean_age)
                    df_match.loc[i, f'mean_hei_player_{titularidad}_{condicion}'] = sum(l_mean_hei) / len(l_mean_hei)
                    df_match.loc[i, f'mean_rat_player_{titularidad}_{condicion}'] = sum(l_mean_rating) / len(l_mean_rating)
                    df_match.loc[i, f'mean_val_player_{titularidad}_{condicion}'] = sum(l_mean_val) / len(l_mean_val)

                    # Calculo nro de jugadores lesionados
                    if titularidad == 'miss':
                        df_match.loc[i, f'n_player_{titularidad}_{condicion}'] = len(l_mean_rating)
                        # print("Numero de ausentes: ", len(l_mean_rating))

                except ZeroDivisionError:
                    # print("Aparentemente no hay datos de jugadores para el partido")
                    pass

            # Cerrar la barra de progreso al finalizar
            progress_bar.close()

    return df_match

def search_fecha_fifa(fecha_part):
    """
    Dado la fecha de un partido, busco el fifa que le corresponde.
    :param fecha_part: Datetime. Fecha del partido.
    :return: String. Año del fifa que corresponde segun la fecha pasada como parametro (e.g. partido del 23/12/2023 le
    corresponde FIFA 24)
    """
    # Definicion de variables
    year_part = fecha_part.year  # e.g. "2021"

    # Si el partido se jugo de Julio a Diciembre (post mercado de pases de invierno)
    if fecha_part.month >= 7:

        year_part_str = str(year_part + 1)[-2:]  # Ultimos dos "22"
        return year_part_str

    # Si el partido se jugo de Enero a Julio (post mercado de pases de verano)
    else:
        year_part_str = str(year_part)[-2:]  # Ultimos dos "21"
        return year_part_str

def prueba():
    start = time.time()
    print("\nIntegrando los datos...")
    country = 'England'

    # Levanto datasets para pruebas
    df_match = pd.read_excel(f"./p3_data_preparation/data/{country}/df_match_formated.xlsx")
    df_match_player = pd.read_excel(f"./p2_data_understanding/data/{country}/df_match_player_formated.xlsx")
    df_player = pd.read_excel(f"./p3_data_preparation/data/{country}/df_player_formated.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace fheia el index_col=0
    print(f"df_match: \n{df_match.head(1)} \n\ndf_match_player: \n{df_match_player.head(1)} \n\n df_player: \n{df_player.head(1)}")

    # Mapeo jugadores por nombre
    df_map_players_name_id = match_players_by_name(df_match_player, df_player)
    df_map_players_name_id.to_excel("/Users/nachomondino/Desktop/df_map_players_name_id.xlsx")

    # Reemplazo nombre de jugadores por id en df_match_player
    df_match_player = replace_players_name_with_id(df_match_player, df_map_players_name_id)
    df_match_player.to_excel('/Users/nachomondino/Desktop/df_match_player_1.xlsx', index=True)

    # Sintetizar la data de df_player (Sofifa) en df_match (Flashscore) gracias al vinculo con df_match_player (Flashscore) -->   Aca dentro hago esto:  # Traer fecha, equipo y no se que mas de df_match (Flashscore) y agregar a df_match_player (Flashscore) para poder saber en que momento traer la info del jugador (Sofifa tiene varias veces un mismo jugador porque es el jugador en ≠ fifas)
    df = integrate_player_data_in_match(df_match, df_match_player, df_player)
    df.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=False)

    end = time.time()
    print(f"Integracion de datos en {(end - start) / 60:.1f} minutos")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()