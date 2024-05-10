import pandas as pd
from fuzzywuzzy import fuzz
import warnings
import time
from tqdm import tqdm
import datetime

def create_df_teams(df: pd.DataFrame):
    """
    Obtiene los valores unicos en las columnas especificadas
    """
    # Combinar las columnas id_team_home y team_home
    df_home = df[['id_team_home', 'team_home']]
    df_home.columns = ['id_team', 'team_name']

    # Combinar las columnas id_team_away y team_away
    df_away = df[['id_team_away', 'team_away']]
    df_away.columns = ['id_team', 'team_name']

    # Concatenar ambos DataFrames y eliminar duplicados
    df_teams = pd.concat([df_home, df_away]).drop_duplicates()
  
    # Establecer los valores únicos como índice
    df_teams.set_index('id_team', inplace=True)
    return df_teams

def create_df_coaches(df: pd.DataFrame):
    """
    Obtiene los valores unicos en las columnas especificadas
    """
    # Combinar las columnas id_team_home y team_home
    df_home = df[['id_coach_home', 'coach_home']]
    df_home.columns = ['id_coach', 'coach_name']

    # Combinar las columnas id_team_away y team_away
    df_away = df[['id_coach_away', 'coach_away']]
    df_away.columns = ['id_coach', 'coach_name']

    # Concatenar ambos DataFrames y eliminar duplicados
    df_coaches = pd.concat([df_home, df_away]).drop_duplicates()
  
    # Establecer los valores únicos como índice
    df_coaches.set_index('id_coach', inplace=True)
    return df_coaches

def create_df_stadiums(df: pd.DataFrame):
    """
    Obtiene los valores unicos en las columnas 
    # Cuidado. Tendria que ver obtener el estadio del team home de acuerdo al estadio que jugo como local la mayoria de veces (y no por un solo partido)
    """
    # Combinar las columnas id_team_home y team_home
    df = df[df['is_cup'] == 0]
    df_home = df[['venue', 'id_team_home']].reset_index()
    df_home = df_home.drop_duplicates(subset='id_team_home')
    return df_home

def create_df_player(df: pd.DataFrame):
    """
    Obtiene los valores unicos en las columnas especificadas
    """
    # Identificar las columnas que contienen "id_player" y "player_name"
    id_player_cols = [col for col in df.columns if 'id_player' in col]
    player_name_cols = [col.replace("id_player", "player_name") for col in id_player_cols]

    # Extraer los valores únicos de "id_player" y sus correspondientes "player_name"
    unique_values = set()
    for id_col, name_col in zip(id_player_cols, player_name_cols):
        unique_values.update(zip(df[id_col], df[name_col]))

    # Crear DataFrame con los valores únicos
    df_unique_values = pd.DataFrame(list(unique_values), columns=['id_player', 'player_name'])

    # Establecer 'id_player' como índice
    df_unique_values.set_index('id_player', inplace=True)

    return df_unique_values

def match_dataframes_by_str_column(df1, df2, column_to_match1, column_to_match2, column_to_integrate, thr_coincidence_min: int, column_to_match2_aux: str = None, _print: bool = False):
    """
    Vinculo datasets df1 y df2 mediante columnas strings.
    
    # Parameters
        df1: Dataframe de fuente 1 (Dataframe)
        df2: Dataframe de fuente 2 (Dataframe)
        column_to_match1: Nombre de columna en df1 con la cual relacionar ambos dataframes. (e.g.'player_name') (String) 
        column_to_match2: Nombre de columna en df2 con la cual relacionar ambos dataframes. (e.g.'player_name') (String) 
        column_to_match2_aux: Nombre de columna en df2 con la cual relacionar ambos dataframes en caso que no haga match con column_to_match2. (e.g.'player_name_short') (String) 
        column_to_integrate: por default es el indice de ambos dataframes pasados como parametro (e.g. 'id_player') (String)
        thr_coincidence_min: Porcentaje de coincidencia minimo entre strings (integer)

    # Returns
        Dataframe con columnas column_to_match1, column_to_match2 y column_to_integrate de ambos dataframes. (DataFrame)
    """
    # Definicion de variables
    df_map = pd.DataFrame(columns=[f'{column_to_integrate}_fs', f'{column_to_match1}_fs', f'{column_to_integrate}_so', f'{column_to_match2}_so', 'porcentaje_coincidencia', 'tipo'])
    n_matchs, n_pos_matchs = 0, len(df1)
    print(f"Mapping Sofifa and Flashscore by {column_to_match1}...")
    progress_bar = tqdm(total=n_pos_matchs, ncols=80)

    # Por fila en df1
    for id_df1, row in df1.iterrows():

        df2_filt = df2.copy()
        if _print:
            print(f"\n VALUE TO FETCH: {row[column_to_match1]}")
            print("Shape df2_filt: ", df2_filt.shape)

        # Calculo el porcentaje de coincidencia con cada posible string en df2
        func = lambda row_df2: calculate_coincidence(row[column_to_match1], row_df2[column_to_match2])        
        df2_filt['porcentaje_coincidencia'] = df2_filt.apply(func, axis=1)
        df2_filt = df2_filt.sort_values(by='porcentaje_coincidencia', ascending=False) 
    
        if _print:
            pd.set_option("display.max.columns", None)  # para ver todas las columnas del df y no que las colapse
            print("Mejores coincidencias: \n", df2_filt.head(5))
            print("Mejor coincidencia: ", row_best_coincidende.values)

        # En caso que haya otra variable para matchear en df2
        if column_to_match2_aux is not None:
            func2 = lambda row_df2: calculate_coincidence(row[column_to_match1], row_df2[column_to_match2_aux])        
            df2_filt['porcentaje_coincidencia_2'] = df2_filt.apply(func2, axis=1)
            df2_filt_2 = df2_filt.sort_values(by='porcentaje_coincidencia_2', ascending=False) 
            row_best_coincidende_2 = df2_filt_2.iloc[0] # Selecciono la primera fila

        
        if len(df2_filt) > 0:
            # Selecciono la opcion con mayor coincidencia
            row_best_coincidende = df2_filt.iloc[0]
            
            # Si la mejor opcion tiene mayor coincidencia que la minima deseada (mayor a thr_coincidence_min)
            if row_best_coincidende['porcentaje_coincidencia'] >= thr_coincidence_min:

                # Hago match
                id_dataframe_2 = row_best_coincidende.name
                col_to_match_2 = row_best_coincidende[column_to_match2]
                col_porcentaje_coincidencia = row_best_coincidende['porcentaje_coincidencia']
                tipo = "long"
                
                # Elimino string que ya hizo match en df2 para agilizar la busqueda y evitar Falsos positivos
                df2 = df2.drop(id_dataframe_2)
                n_matchs += 1
                if _print:
                    print(df_map.loc[id_df1].values)
                    print(df2.shape)
                    print(f"MATCH: '{row[column_to_match1]}' <--> '{row_best_coincidende[column_to_match2]}'. Coincidencia: {row_best_coincidende['porcentaje_coincidencia']}")
            
            # Probar a hacer match con columna alternativa.
            elif column_to_match2_aux is not None:
                
                if row_best_coincidende_2['porcentaje_coincidencia_2'] >= 100:

                    id_dataframe_2 = row_best_coincidende_2.name
                    col_to_match_2 = row_best_coincidende_2[column_to_match2_aux]
                    col_porcentaje_coincidencia = row_best_coincidende_2['porcentaje_coincidencia_2']
                    tipo = "short"
                    
                    # Elimino string que ya hizo match en df2 para agilizar la busqueda y evitar Falsos positivos
                    df2 = df2.drop(id_dataframe_2)
                    n_matchs += 1
                    if _print:
                        print(df_map.loc[id_df1].values)
                        print(df2.shape)
                        print(f"MATCH: '{row[column_to_match1]}' <--> '{row_best_coincidende_2[column_to_match2_aux]}'. Coincidencia: {row_best_coincidende_2['porcentaje_coincidencia']}")
                
                else:
                    id_dataframe_2 = None
                    col_to_match_2 = None
                    col_porcentaje_coincidencia = None
                    tipo = None
            
            # Si no hizo match  
            else:
                id_dataframe_2 = None
                col_to_match_2 = None
                col_porcentaje_coincidencia = None
                tipo = None
                if _print:
                    print(f'No hizo match puesto que la opcion con mas coincidencia fue {row_best_coincidende['porcentaje_coincidencia']} (menor a {thr_coincidence_min}). La mejor coincidencia para "{row[column_to_match1]}" fue "{row_best_coincidende[column_to_match2]}".')
        
            # Guardo datos
            d_data = {f'{column_to_integrate}_fs': id_df1, f'{column_to_match1}_fs': row[column_to_match1], f'{column_to_integrate}_so': id_dataframe_2, f'{column_to_match2}_so': col_to_match_2, 'porcentaje_coincidencia': col_porcentaje_coincidencia, 'tipo': tipo}
            df_map = pd.concat([df_map, pd.DataFrame(d_data, index=[len(df_map)])])
        progress_bar.update(1)
    progress_bar.close()

    if n_pos_matchs > 0:
        print(f"De los {n_pos_matchs} strings en df1, hizo match para {n_matchs}, es decir para el {n_matchs/n_pos_matchs*100:.2f}% de ellos.")
    else:
        print("Warning! No se cuenta con las formaciones de ningun partido de df_match, por ende, df_match_player no tiene que integrar a df_match.")
    return df_map

def calculate_coincidence(str1, str2):
    """
    Calcula porcentaje de coincidencia entre dos strings pasados como parametro
    :param str1: Primer string a comparar. (String)
    :param str2: Segundo string a comparar. (String)
    """
    coincidencia = fuzz.token_set_ratio(str1, str2)
    return coincidencia
    
# DF_TEAMS TO DF_MATCH
def integrate_team_data_in_match(df_match, df_map_teams_fs_so, df_teams_sofifa):
    """
    Integra datos de equipos desde Sofifa hasta df_match.
    """
    print("\n Integrating team's data to df_match using mapping...")
    
    # Por columna de equipos en df_match
    for col_team in ['id_team_home', 'id_team_away']:

        # Obtengo valores unicos
        l_id_teams = df_match[col_team].unique()

        # Por equipo
        for id_team_fs in l_id_teams:

            # Obtengo partidos del equipo
            l_idxs = df_match[df_match[col_team] == id_team_fs].index

            # Busco el mapeo del equipo con sofifa
            row_map = df_map_teams_fs_so[df_map_teams_fs_so['id_team_fs'] == id_team_fs]

            if len(row_map) > 0:

                # Busco id_team_sofifa
                id_team_sofifa = row_map['id_team_so'].values[0]

                # Busco datos en df_teams_sofifa usando id_team_sofifa
                row_team_sofifa = df_teams_sofifa[df_teams_sofifa.index == id_team_sofifa]

                # Si el valor es un id    
                if (not pd.isna(id_team_sofifa)) and (len(row_team_sofifa) > 0):
                    # df_match.loc[l_idxs, f'{col_team}_int_prestige'] =  row_team_sofifa['international_prestige'].values[0]
                    # df_match.loc[l_idxs, f'{col_team}_dom_prestige'] =  row_team_sofifa['domestic_prestige'].values[0]

                    # Rival team (# no quiero guardar el id_team_rival de sofifa sino el de Flashscore...)
                    id_sofifa_rival_team = row_team_sofifa['id_rival_team'].values[0] 
                    row_fs_rival_team = df_map_teams_fs_so[df_map_teams_fs_so['id_team_so'] == id_sofifa_rival_team]
                    if len(row_fs_rival_team)==1:
                        id_fs_rival_team = row_fs_rival_team['id_team_fs'].values[0]
                        df_match.loc[l_idxs, f'{col_team}_rival_team'] =  id_fs_rival_team

    return df_match

# DF_PLAYER, DF_PLAYER_FIFA_SOFIFA Y DF_MATCH_PLAYER TO DF_MATCH
def integrate_player_data_in_match(df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa, _print: bool = False):  # Mejorar. Agregar potential
    """
    Integra la entidad jugador en la entidad partido. Es decir, sintetiza los datos de los jugadores a cada partido en
    particular. Se determinan los promedios de age, overall rating, value de mercado y height del equipo titular,
    suplente y los ausentes para cada equipo.
   
    # Parameters
        df_match: Dataframe. Unidad de analisis: partido. Columnas: equipos, arbitros, estadisticas del partido, etc.
        df_match_player: Dataframe. Unidad de analisis: partido. Columnas: id_match y una por jugador segun formaciones. Celdas: id de jugador (en vez de name).
        df_player: Dataframe. Unidad de analisis: jugador. Columnas: id_player y datos del jugador como age y overall rating.

    # Returns
        Dataframe. Dataframe con los datos de todos los dataframes pasados como parametro. Tod@ en un solo dataframe para poder entrenar un modelo con ellos.
    """
    print("\n Integrating players's data to df_match using mapping...")
    # Definicion de variables
    warnings.filterwarnings('ignore')  # Ver el ignore, y solucionarlo en vez de ignorarlo...
    l_titularidad = ['start', 'sub', 'miss']  # tendria que agregar 'sup_ing' pero se lo proceso con sup.
    l_condicion = ['home', 'away']
    d_n_reg_min = {'start': 7, 'sub': 4, 'miss': 0}

    # Por titularidad (Titular, suplente o ausente)
    for titularidad in l_titularidad:
        n_reg_min = d_n_reg_min[titularidad]

        # Por condicion (Local o visitante)
        for condicion in l_condicion:

            # Defino pattern y con el, selecciono las variables a procesar
            pattern = f'id_player_{titularidad}_[a-z]*[_]*{condicion}_[0-9]+'
            l_col_to_preprocess = df_match_player.filter(regex=pattern, axis=1).columns.tolist()
            print(f"Integrating players: {titularidad} {condicion}")
            if _print:
                print(f'Columnas a procesar: {l_col_to_preprocess}')

            # Inicializo barra de progreso
            progress_bar = tqdm(total=len(df_match_player), ncols=80)

            # Por partido
            for id_match, row_match in df_match.iterrows():

                l_age, l_height, l_rating, l_market_value, l_potential, l_int_reputation = [], [], [], [], [], []
                
                # Busco el fifa correspondiente segun la fecha del partido
                year_fifa = search_fecha_fifa(row_match['date'])                
                if _print:
                    print(f' Partido Nº: {id_match} '.center(120, '#'))
                    print(f"Fecha partido: {row_match['date']} --> Fifa a buscar: {year_fifa}")

                # Por columna jugador en df_match_player
                for col_player in l_col_to_preprocess:

                    # Busco id del jugador en df_match_player
                    # id_player_fs = df_match_player.loc[id_match, col_player]  # fallo en assess_model_in_prod de Argentina
                    try:
                        id_player_fs = df_match_player.loc[id_match, col_player].values[0]  # fallo en assess_model_in_prod de Argentina
                    except:
                        id_player_fs = df_match_player.loc[id_match, col_player]  # fallo en assess_model_in_prod de Argentina
                    if _print:
                        print(f"\t Id jugador Flashscore: {id_player_fs}")

                    # Si el id_player no es nan
                    if not pd.isna(id_player_fs):  # hay mucho nan sobretodo columnas de jugadores ausentes (e.g. player_aus_vis_12)

                        # Busco el mapeo del equipo con sofifa
                        row_map = df_map_fs_so[df_map_fs_so['id_player_fs'] == id_player_fs]

                        if len(row_map) > 0:

                            # Busco id_team_sofifa
                            id_player_sofifa = row_map['id_player_so'].values[0]

                            # Busco el id y la fecha en df_player (Sofifa)
                            df_player_filt = df_player_fifa_sofifa[(df_player_fifa_sofifa['id_player'] == id_player_sofifa)]
                            df_player_filt = df_player_filt[(df_player_filt['fifa_year'].astype(int) == int(year_fifa))]
                            if _print:
                                print(f"\t\t Hizo match para este jugador! Id jugador en Sofifa: {id_player_sofifa}")       
                                print(f"\t\t Shape df_player_filt (debe ser 1 o 2): {df_player_filt.shape[0]}")       

                            # Guardo datos del jugador
                            if len(df_player_filt) > 0:
                                try:
                                    height = df_player_sofifa.loc[id_player_sofifa, 'height'].values[0]
                                except:
                                    height = df_player_sofifa.loc[id_player_sofifa, 'height']
                                l_age.append(df_player_filt.age.values[0])
                                l_height.append(height)  # l_height.append(df_player_filt.height.values[0])
                                l_rating.append(df_player_filt.overall_rating.values[0])
                                l_market_value.append(df_player_filt.value.values[0])
                                l_potential.append(df_player_filt.potential.values[0])
                                l_int_reputation.append(df_player_filt.int_reputation.values[0])
                                if _print:
                                    print(f"\t\t DATOS DEL JUGADOR: Age: {df_player_filt.age.values[0]}; Height: {height}; Rating: {df_player_filt.overall_rating.values[0]}; Market value: {df_player_filt.value.values[0]}; Potencial: {df_player_filt.potential.values[0]}; Int rep: {df_player_filt.int_reputation.values[0]}")       

                # Guardo promedios de age, height, overall_rating y market value
                if len(l_age) > n_reg_min:
                    if _print:
                        print(f"Promedio age: {sum(l_age) / len(l_age)}; Promedio height: {sum(l_height) / len(l_height)}; Promedio int rep: {sum(l_int_reputation) / len(l_int_reputation)}")

                    df_match.loc[id_match, f'mean_age_player_{titularidad}_{condicion}'] = sum(l_age) / len(l_age)
                    df_match.loc[id_match, f'mean_hei_player_{titularidad}_{condicion}'] = sum(l_height) / len(l_height)
                    df_match.loc[id_match, f'mean_int_rep_player_{titularidad}_{condicion}'] = sum(l_int_reputation) / len(l_int_reputation)

                    # Calculo nro de jugadores lesionados
                    if titularidad == 'miss':
                        df_match.loc[id_match, f'n_player_{titularidad}_{condicion}'] = len(l_rating)

                        df_match.loc[id_match, f'sum_rat_player_{titularidad}_{condicion}'] = sum(l_rating)
                        df_match.loc[id_match, f'sum_val_player_{titularidad}_{condicion}'] = sum(l_market_value)
                        df_match.loc[id_match, f'sum_pot_player_{titularidad}_{condicion}'] = sum(l_potential)

                    else:
                        df_match.loc[id_match, f'mean_rat_player_{titularidad}_{condicion}'] = sum(l_rating) / len(l_rating)
                        df_match.loc[id_match, f'mean_val_player_{titularidad}_{condicion}'] = sum(l_market_value) / len(l_market_value)
                        df_match.loc[id_match, f'mean_pot_player_{titularidad}_{condicion}'] = sum(l_potential) / len(l_potential)


                progress_bar.update(1)
            progress_bar.close()
    return df_match

def search_fecha_fifa(fecha_part):
    """
    Dado la fecha de un partido, busco el fifa que le corresponde.

    # Parameters:
        fecha_part: Fecha del partido. (datetime)
    
    # Returns:
        Año del fifa que corresponde segun la fecha pasada como parametro (e.g. partido del 23/12/2023 le corresponde FIFA 24) (string)
    """
    # Definicion de variables
    year_part = fecha_part.year  # e.g. "2021"
    
    # Obtener el año actual
    año_actual = datetime.datetime.now().year

    # Si el partido se jugo de Julio a Diciembre (post mercado de pases de invierno)
    if fecha_part.month >= 7:  # Puedo poner 9? (Usaria 9 porque el fifa sale en Septiembre (mes 9)) Creo que no porque si un jugador no esta en el fifa anterior, quedaria NaN.

        # Si el partido es de la nueva temporada y el fifa aun no salió (Con esto creo que arreglaria lo de que el fifa aun no salió)
        if (year_part == año_actual) and (fecha_part.month <= 9):
            year_part_str = str(year_part)[-2:]  # Ultimos dos "21"
            return year_part_str
        
        year_part_str = str(year_part + 1)[-2:]  # Ultimos dos "22"
        return year_part_str

    # Si el partido se jugo de Enero a Julio (post mercado de pases de verano)
    else:
        year_part_str = str(year_part)[-2:]  # Ultimos dos "21"
        return year_part_str

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    start = time.time()
    print("\nIntegrando los datos...")
    country = 'spain'

    # Levanto datasets para pruebas
    df_match = pd.read_excel(f"./p3_data_preparation/data/{country}/clean_data/df_match_cleaned.xlsx", index_col=0)
    df_match_player = pd.read_excel(f"./p3_data_preparation/data/{country}/clean_data/df_match_player_cleaned.xlsx",  index_col=0)
    df_player = pd.read_excel(f"./p3_data_preparation/data/{country}/clean_data/df_player_cleaned.xlsx", index_col=0)
    df_player_sofifa = pd.read_excel(f"./p3_data_preparation/data/{country}/clean_data/df_player_sofifa_cleaned.xlsx", index_col=0) 
    df_player_fifa_sofifa = pd.read_excel(f"./p3_data_preparation/data/{country}/clean_data/df_player_fifa_sofifa_cleaned.xlsx")
    df_teams_sofifa = pd.read_excel(f"./p3_data_preparation/data/{country}/clean_data/df_teams_sofifa_cleaned.xlsx", index_col=0)  
    print(f"df_match: \n{df_match.head(1)} \n\ndf_match_player: \n{df_match_player.head(1)} \n\n df_player: \n{df_player.head(1)}")

    df_match = df_match.head(100)
    df_match_player = df_match_player.head(100)
    df_player = df_player.head(200)
    # df_player_sofifa = df_player_sofifa.head(200)
    
    # TEAMS --> MATCH  (Mapeo df_teams_sofifa con df_teams e integro a df_match)
    # print("\nIntegrating team's data to df_match...")
    # df_map_teams_fs_so = match_dataframes_by_str_column(df_teams, df_teams_sofifa, column_to_relation='team_name', column_to_integrate='id_team', thr_coincidence_min=90)
    # df_match = integrate_team_data_in_match(df_match, df_map_teams_fs_so, df_teams_sofifa)
    # df_map_teams_fs_so.to_excel("/Users/nachomondino/Desktop/df_map_teams_fs_so.xlsx")

    # PLAYERS --> MATCH (Mapeo df_player_sofifa con df_player e integro a df_match)
    print("\nIntegrating player's data to df_match...")
    df_map_players_fs_so = match_dataframes_by_str_column(df_player, df_player_sofifa, column_to_relation="player_name", column_to_integrate='id_player', thr_coincidence_min=90)
    df_map_players_fs_so.to_excel('/Users/nachomondino/Desktop/df_map_players_fs_so.xlsx', index=True)
    df = integrate_player_data_in_match(df_match, df_match_player, df_map_players_fs_so, df_player_sofifa, df_player_fifa_sofifa,  _print=True)

    df.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=True)

    end = time.time()
    print(f"Integracion de datos en {(end - start) / 60:.1f} minutos")