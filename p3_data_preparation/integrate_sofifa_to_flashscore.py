# Juntar entidades jugadores, partido y atrib_playeradores
import pandas as pd
from fuzzywuzzy import fuzz
import warnings
import time
from tqdm import tqdm
import math
from p3_data_preparation import clean_data


def match_dataframes_by_str_column(df1, df2, column_to_relation, column_to_integrate, thr_coincidence_min: int, _print: bool = False):
    """
    Vinculo datasets mediante columna string.
    En el caso de teams: column_to_relation = 'team_name' y column_to_integrate="id_team" (index de df2)
    df1 tiene team_name como columna (la unica)
    df2 tiene team_name como columnas y id_team como index
    """
    print("Matching data from Sofifa and Flashscore...")

    # Definicion de variables
    df1_with_id = df1.copy()  # Creo copia del dataframe df_match_player en el que agregar la columna "id_player"
    n_matchs, n_pos_matchs = 0, len(df1)

    # Funcion que hace una busqueda aproximada de un string en una columna
    def calculate_coincidence(palabra1, palabra2):
        coincidencia = fuzz.token_set_ratio(palabra1, palabra2)
        return coincidencia

    # Inicializo barra de progreso
    progress_bar = tqdm(total=n_pos_matchs, ncols=80)

    # Por jugador en df_match_player
    for i, row in df1.iterrows():

        str_to_fetch = row[column_to_relation]  # (e.g. Manchester City)
        df2_filt = df2.copy()
        if _print:
            print(f"\n VALUE TO FETCH: {str_to_fetch}")
            print("Shape df2_filt: ", df2_filt.shape)

        # Calculo el porcentaje de coincidencia con cada posible string en df2
        df2_filt['porcentaje_coincidencia'] = df2_filt.apply(lambda row: calculate_coincidence(str_to_fetch, row[column_to_relation]), axis=1)
        # for j, fila in df2_filt.iterrows():        
            # df2_filt.loc[j, 'porcentaje_coincidencia'] = calculate_coincidence(str_to_fetch, fila[column_to_relation])

        # Selecciono la opcion con mayor coincidencia
        df2_filt = df2_filt.sort_values(by='porcentaje_coincidencia', ascending=False) 
        row_best_coincidende = df2_filt.iloc[0] # Selecciono la primera fila
        if _print:
            pd.set_option("display.max.columns", None)  # para ver todas las columnas del df y no que las colapse
            print("Mejores coincidencias: \n", df2_filt.head(5))
            print("Mejor coincidencia: ", row_best_coincidende.values)

        # Si la mejor opcion tiene mayor coincidencia que la minima deseada (mayor a thr_coincidence_min)
        if row_best_coincidende['porcentaje_coincidencia'] >= thr_coincidence_min:

            # Hago match
            # Agrego column_to_integrate a df1_with_id
            df1_with_id.loc[i, column_to_integrate] = row_best_coincidende.name
            df1_with_id.loc[i, f'{column_to_relation}_sofifa'] =  row_best_coincidende[column_to_relation]  # team_name
            df1_with_id.loc[i, f'porcentaje_coincidencia'] = row_best_coincidende['porcentaje_coincidencia']

            # Elimino string que ya hizo match en df2 para agilizar la busqueda y evitar Falsos positivos
            df2 = df2.drop(row_best_coincidende.name)
            n_matchs += 1
            if _print:
                print(df1_with_id.loc[i].values)
                print(df2.shape)
                print(f"MATCH: '{str_to_fetch}' <--> '{row_best_coincidende['team_name']}'. Coincidencia: {row_best_coincidende['porcentaje_coincidencia']}")
        elif _print:
            print(f'No hizo match puesto que la opcion con mas coincidencia fue {row_best_coincidende['porcentaje_coincidencia']} (menor a {thr_coincidence_min}). La mejor coincidencia para "{str_to_fetch}" fue "{row_best_coincidende['team_name']}".')
        progress_bar.update(1)
    progress_bar.close()

    if n_pos_matchs > 0:
        print(f"De los {n_pos_matchs} strings en df1, hizo match para {n_matchs}, es decir para el {n_matchs/n_pos_matchs*100:.2f}% de ellos.")
    else:
        print("Warning! No se cuenta con las formaciones de ningun partido de df_match, por ende, df_match_player no tiene que integrar a df_match.")
    return df1_with_id

# DF_TEAMS TO DF_MATCH
def match_teams_by_name(df_teams: pd.DataFrame, df_teams_sofifa: pd.DataFrame, thr_coincidence_min: int = 90):
    """
    Matchea equipos entre df_teams de Flashcore y df_teams de Sofifa mediante "team_name".
    """
    print("Integrating df_teams to df_match...")
    # Preparo datasets para integrar
    ## Dataframe teams:
    df_teams_sofifa = clean_data.prepare_text_columns(df_teams_sofifa, l_cols_to_process=['team_name'])
    df_teams_only = pd.DataFrame(df_teams_sofifa['team_name'], index=df_teams_sofifa.index)

    # Matcheo df_teams y df_match por nombre de equipo
    df_map_teams_name_id = match_dataframes_by_str_column(df_teams, df_teams_only, column_to_relation='team_name', column_to_integrate='id_team', thr_coincidence_min=thr_coincidence_min)
    return df_map_teams_name_id

def integrate_team_data_in_match(df_match, df_etiquetas, df_teams_sofifa, df_map_teams_name_id):  # LO HARE A DF_ETIQUETAS...
    
    # integrar id_team, rival team id y algo mas? 
    l_col_teams = ['id_team_home', 'id_team_away']
    
    # Por columna equipo
    for col_team in l_col_teams:

        # Obtengo valores unicos en df_match
        l_unique_values = df_match[col_team].unique()

        # Por valor unico
        for value in l_unique_values:

            # Obtengo filas donde team=value
            df_match_val = df_match[df_match[col_team] == value]

            # Busco id en df_etiquetas
            row_etiquetas = df_etiquetas[(df_etiquetas['variable'] == col_team) & (df_etiquetas['int_value'] == value)]
            id_team_flash = row_etiquetas['str_value'].values[0]
            
            # Busco 
            row_map = df_map_teams_name_id[df_map_teams_name_id.index == id_team_flash]
            id_team_sofifa = row_map['id_team'].values[0]

            row_teams = df_teams_sofifa[df_teams_sofifa.index == id_team_sofifa]

            # Si el valor es un id    
            if not pd.isna(id_team_sofifa):
                df_match.loc[df_match_val.index, f'{col_team}_int_prestige'] =  row_teams['international_prestige'].values[0]
                df_match.loc[df_match_val.index, f'{col_team}_domestic_prestige'] =  row_teams['domestic_prestige'].values[0]
                df_match.loc[df_match_val.index, f'{col_team}_rival_team'] =  row_teams['id_rival_team'].values[0]

    return df_match


# DF_PLAYER, DF_PLAYER_FIFA_SOFIFA Y DF_MATCH_PLAYER TO DF_MATCH
def replace_players_with_sofifa_id(df_match_player, df_map_players_name_id):  # Creo que es mas rapida...
    """
    Reemplaza los id de jugadores en df_match_player por el id que corresponda al jugador en Sofifa. Esto facilitará la integracion posterior.
    """
    print("\nReplacing player's names by id in df_match_player...")
    progress_bar = tqdm(total=len(df_map_players_name_id), ncols=80)

    # Por jugador
    for idx, row in df_map_players_name_id.iterrows():

        # Reemplazo su name por su id en df_match_player
        df_match_player = df_match_player.replace(idx, row['id_player'])  # Si no hizo match, asigna nan pues id_player es nan.
        progress_bar.update(1)

    progress_bar.close()
    return df_match_player

def integrate_player_data_in_match(df_match, df_match_player, df_player_sofifa, df_player_fifa_sofifa, _print: bool = False): 
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
    l_titularidad = ['start', 'sub', 'miss']  # tendria que agregar 'sup_ing' pero se lo proceso con sup.
    l_condicion = ['home', 'away']

    # Agrego columna "fifa_year" quedandome solo con el año del fifa (e.g. "22" en vez de "FIFA 22")
    df_player_fifa_sofifa['fifa_year'] = df_player_fifa_sofifa['fifa'].str.split(' ').str[-1]
    
    # Por titularidad (Titular, suplente o ausente)
    for titularidad in l_titularidad:

        # Por condicion (Local o visitante)
        for condicion in l_condicion:

            # Defino pattern y con el, selecciono las variables a procesar
            pattern = f'id_player_{titularidad}_[a-z]*[_]*{condicion}_[0-9]+'
            l_col_to_preprocess = df_match_player.filter(regex=pattern, axis=1).columns.tolist()
            print(f'Columnas a procesar: {l_col_to_preprocess}')
            if _print:
                print(f'Columnas a procesar: {l_col_to_preprocess}')

            # Inicializo barra de progreso
            progress_bar = tqdm(total=len(df_match_player), ncols=80)

            # Por partido
            for i, row in df_match.iterrows():

                l_mean_age, l_mean_hei, l_mean_rating, l_mean_val = [], [], [], []
                
                # Busco el fifa correspondiente segun la fecha del partido
                year_fifa = search_fecha_fifa(row['date'])
                if _print:
                    print(f' Partido Nº: {i} '.center(120, '#'))
                    print(f"Fecha partido: {row['fecha']} --> Fifa a buscar: {fecha_part_fifa}")

                # Por jugador
                for col_player in l_col_to_preprocess:

                    # Busco id del jugador en df_match_player
                    id_player_ent_part = df_match_player.loc[i, col_player]
                    if _print:
                        print(f"\t Id jugador a buscar en Sofifa: {id_player_ent_part}")

                    # Si el id_player no es nan
                    if not math.isnan(id_player_ent_part):  # hay mucho nan sobretodo columnas de jugadores ausentes (e.g. player_aus_vis_12)

                        # Busco el id y la fecha en df_player (Sofifa)
                        df_player_filt = df_player_fifa_sofifa[(df_player_fifa_sofifa['id_player'] == id_player_ent_part) & (df_player_fifa_sofifa['fifa_year'] == year_fifa)]
                        height = df_player_sofifa.loc[id_player_ent_part, 'height']

                        # Guardo datos del jugador
                        if len(df_player_filt) > 0:
                            l_mean_age.append(df_player_filt.age.values[0])
                            l_mean_hei.append(height)  # l_mean_hei.append(df_player_filt.height.values[0])
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
                progress_bar.update(1)
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