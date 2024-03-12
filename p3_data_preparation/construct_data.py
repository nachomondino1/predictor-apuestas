import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import re
from p3_data_preparation.clean_data import replace_nan_with_zero

# Main.py
def determine_result(df: pd.DataFrame, var_resp: str):
    """
    Se determina el 'result' a partir de los goles que hizo cada team
    :param df: Dataframe. Unidad de analisis: match. Columnas: entre ellas goals_home y goals_away
    :return: Dataframe pasado por parametro con nueva columna, 'result', que detalla el resultado del match.
    """
    # Condiciones para determine el ganador
    condiciones = [
        df['goals_home'] > df['goals_away'],
        df['goals_home'] < df['goals_away'],
    ]

    # Valores correspondientes a las condiciones
    valores = [1, 2]

    # Usar numpy.select para aplicar las condiciones
    df[var_resp] = pd.Series(np.select(condiciones, valores, default=0), index=df.index)
    return df

def determine_points(df: pd.DataFrame):
    """
    Determina los points obtenidos por cada team segun el resultado del juego.

    :param df:
    :return:
    """
    # Inicializo las columnas "points_home" y "points_away"
    df['points_home'] = 0
    df['points_away'] = 0

    df.loc[df['result'] == 1, 'points_home'] = 3
    df.loc[df['result'] == 1, 'points_away'] = 0

    df.loc[df['result'] == 0, 'points_home'] = 1
    df.loc[df['result'] == 0, 'points_away'] = 1

    df.loc[df['result'] == 2, 'points_home'] = 0
    df.loc[df['result'] == 2, 'points_away'] = 3
    return df

def determine_number_matches_last_days(df: pd.DataFrame, n_days, _print: bool = False):

    # Ordeno por fecha ascendente
    df = df.sort_values(by='date', ascending=False)

    # Por team
    for team in df['id_team_home'].unique():
    
        # Obtengo los matchs que jugó el team
        df_match_team = df[(df['id_team_home'] == team) | (df['id_team_away'] == team)]

        # Por match del team
        for id_match, row in df_match_team.iterrows(): # for id_match, row in df_match_team_2.iloc[:3].iterrows():

            home_or_away = 'home' if row['id_team_home'] == team else 'away'
            limit_date = row['date'] - timedelta(days=n_days)

            # Selecciono los ultimos matchs del team
            df_match_team_filt = df_match_team.loc[(df_match_team['date'] >= limit_date) & (df_match_team['date'] < row['date'])]

            if len(df_match_team_filt) > 0:
                df.loc[id_match, f'n_matches_last_days_{home_or_away}'] = len(df_match_team_filt)

    return df

def construct_percentaje_column(df: pd.DataFrame, col_num: str, col_den: str):
    df[f'perc_{col_num}_of_{col_den}_home'] = round(df[f'{col_num}_home'] / df[f'{col_den}_home'], 2)
    df[f'perc_{col_num}_of_{col_den}_away'] = round(df[f'{col_num}_away'] / df[f'{col_den}_away'], 2)
    return df

def determine_stats_columns(df: pd.DataFrame):
    """
    Determina automáticamente las variables que deben ser promediadas en los últimos partidos.
    :param df: DataFrame.
    :return: Lista de variables a ser promediadas.
    """
    # Definicion variables
    keywords_prohibidas = ['_player_', 'team_', 'odds_', 'coach_']  # Definir palabras clave prohibidas
    pattern = r'[a-z_\(\)%]+\_(home|away)' # Patrón regex para encontrar columnas relevantes

    # Obtener nombres de columnas relevantes
    relevant_columns = df.filter(regex=pattern, axis=1).columns

    # Filtrar columnas relevantes excluyendo las palabras clave prohibidas
    relevant_columns = [col for col in relevant_columns if not any(keyword in col for keyword in keywords_prohibidas)]

    # Extraer los nombres de las estadísticas
    stats = set()
    for col in relevant_columns:
        stat = re.sub(r'_(home|away)$', '', col)
        stats.add(stat)

    return list(stats)

def h2h_by_date(df: pd.DataFrame, n_years: int, segun_localia: bool = False, _print: bool = False):
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    print("\n Constructing Head to Head...")
    # Definicion de variables
    n_days = 365 * n_years
    l_equipos = df['id_team_home'].unique()
    h2h_col_name = 'h2h_segun_loc' if segun_localia else 'h2h'

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo

    # Por equipo 1
    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]

        # Por equipo 2
        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]

            # Determino df_historial para eq1 y eq2
            df_historial = df[((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2)) | (df['id_team_home'] == eq2) & (df['id_team_away'] == eq1)]
            l_dfs = [df_historial] if not segun_localia else [df_historial[df_historial['id_team_home'] == eq1], df_historial[df_historial['id_team_home'] == eq2]]
            if _print:
                print(f"\n Equipo 1: {eq1} Equipo 2: {eq2}. \n".center(120, "-"))
                print("\nDF_HISTORIAL: \n", df_historial.loc[:, ['date', 'id_team_home', 'id_team_away', 'result']])

            for df_hist in l_dfs:
                if _print and segun_localia:
                    print("\nDF_HISTORIAL LOCALIA: \n", df_hist.loc[:, ['date', 'id_team_home', 'id_team_away', 'result']])

                # Por partido entre equipos
                for idx, row in df_hist.iterrows():

                    h2h = 0
                    limit_date = row['date'] - timedelta(days=n_days)

                    # Selecciono los ultimos matchs
                    df_hist_filt = df_hist.loc[(df_hist['date'] >= limit_date) & (df_hist['date'] < row['date'])]
                    if _print:
                        print(f"\nPartido al cual construir historial: 'id_match': {idx} 'Date': {row['date']} 'team_home': {row['id_team_home']} 'team_away': {row['id_team_away']} ")
                        print("Nºpartidos para construir historial: ", len(df_hist_filt))
                        print(df_hist_filt.loc[:, ['date', 'id_team_home', 'id_team_away', 'result']])

                    # Por ultimos matchs
                    for _, fila in df_hist_filt.iterrows():

                        if fila['result'] == 1:
                            h2h += +1 if fila['id_team_home'] == row['id_team_home'] else -1

                        elif fila['result'] == 2:
                            h2h += -1 if fila['id_team_home'] == row['id_team_home'] else +1
                        if _print:
                            print(i, h2h)

                    # Guardo h2h
                    if len(df_hist_filt) > 0:  # Para evitar guardar h2h = 0 en matchs donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                        df.loc[idx, h2h_col_name] = h2h / len(df_hist_filt)
                        if _print:
                            print(f"Historial a agregar: {h2h/len(df_hist_filt)}")

    return df

def determine_mean_in_last_matches(df: pd.DataFrame, n_days: int, variable: str, segun_localia: bool, tipo: str = 'mean', _print: bool = False):
    """
     Obtiene el promedio de las stats en los ultimos matchs

     :param df: DataFrame.
     :param n_days: Integer. Numero de dias de los cuales obtener los datos.
     :param variable: String. Nombre de la variable a promediar.
     :param tipo: String. Tipo de cálculo a realizar ('mean' para promedio, 'sum' para suma).
     :return: DataFrame con stats promediadas
     """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='date', ascending=False)

    # Por team
    for team in df['id_team_home'].unique():
    
        # Obtengo los matchs que jugó el team
        df_match_team = df[(df['id_team_home'] == team) | (df['id_team_away'] == team)]
        l_dfs = [df_match_team] if not segun_localia else [df_match_team[df_match_team['id_team_home'] == team], df_match_team[df_match_team['id_team_away'] == team]]
        if _print:
            print(f"Team: {team}")
            print(f"Cantidad total de matchs del team: {df_match_team.shape[0]}")
            print('DF_MATCH_TEAM: \n', df_match_team.loc[:, ['date', 'id_team_home', 'id_team_away', variable]].head(5))
        
        for df_match_team_2 in l_dfs:
            if _print and segun_localia:
                print(f'\nDF_MATCH_TEAM_LOCALIA: {df_match_team_2.shape} \n', df_match_team_2.loc[:, ['date', 'id_team_home', 'id_team_away', variable]].head(5))

            # Por match del team
            for id_match, row in df_match_team_2.iterrows(): # for id_match, row in df_match_team_2.iloc[:3].iterrows():

                home_or_away = 'home' if row['id_team_home'] == team else 'away'
                limit_date = row['date'] - timedelta(days=n_days)

                # Selecciono los ultimos matchs del team
                df_match_team_filt = df_match_team_2.loc[(df_match_team_2['date'] >= limit_date) & (df_match_team_2['date'] < row['date'])]

                # Necesito la posesion segun si fue home o away en cada uno de esos matchs...
                s_valores_home = df_match_team_filt.loc[df_match_team_filt['id_team_home'] == team, variable]
                s_valores_away = df_match_team_filt.loc[df_match_team_filt['id_team_away'] == team, variable] * -1  # -1 puesto que valores positivos en dif_variable es para el home y valores negativos es favor del away
                s_valores = pd.concat([s_valores_home, s_valores_away], ignore_index=True)
                if _print:
                    print(f"\nMatch Nº: {id_match}")
                    print(f"Cantidad de partidos en ultimos {n_days} dias: {df_match_team_filt.shape[0]}")
                    print(f"Estadistica {variable} en dichos partidos: {s_valores.values}")

                if len(s_valores) > 0:
                    value = s_valores.mean() if tipo == "mean" else (s_valores.sum() if tipo == "sum" else None)
                    df.loc[id_match, f'{tipo}_last_match_{variable}_{home_or_away}'] = value
                    # df.loc[id_match, f'{tipo}_last_match_{variable}_{home_or_away}_pond'] = value / len(s_valores)
                    if _print:
                        print(f"Valor a agregar: {value} y {value / len(s_valores)} \n")

    return df

def suma_rat_player_missing(df: pd.DataFrame):  # Ojo falla en calculo cuando uno de los dos equipos no tiene players missing (o sea, las variables missing son nan) --> en ese caso tiene que hacer la diferencia igual...
    """
    Calculo la suma del rating de los players missing dado que cada team tiene distinto numero de missing.
    :param df: Dataframe. Unidad de analisis: match.
    :return:
    """
    # Creo una copia del df para evitar sum=nan (nan*0=nan) cdo uno de los dos equipos no tiene jugadores ausentes.
    df_copia = replace_nan_with_zero(df, 'mean_rat_player_miss_home', 'mean_rat_player_miss_away')  # Reemplazo mean=nan por mean=0 cdo uno de los dos equipos no tiene jugadores ausentes
    # Si haria el replace en construt (fuera de esta funcion), calcularia mal la diferencia de promedio de rating. Cdo uno de los dos equipos no tiene jug ausentes, dif_prom seria exageradamente alta. (e.g. dif_prom = mean_rat_player_miss_home - mean_rat_player_miss_home = 0 (deberia ser nan) - 80 = -80 (en vez de nan))
  
    df['sum_rat_player_miss_home'] = df_copia['mean_rat_player_miss_home'] * df_copia['n_player_miss_home']  # Lo guardo en la misma porque sino la cago con calculate_dif_col_players()
    df['sum_rat_player_miss_away'] = df_copia['mean_rat_player_miss_away'] * df_copia['n_player_miss_away']
    return df

def calculate_dif_col_players(df: pd.DataFrame):
    """
    Calcula la diferencia entre home y away
    :param df: Dataframe. Unidad de analisis: match
    :return:
    """
    # Defincion de variables
    l_var_jug = ['mean_age', 'mean_hei', 'mean_rat', 'mean_val']
    l_titularidad = ['start', 'sub', 'miss']

    for titularidad in l_titularidad:

        for var in l_var_jug:

            # Si es el rating de los jugadores ausentes 
            if titularidad == 'miss' and var == 'mean_rat':
                var_2 = 'sum_rat'
                # Calculo diferencia entre home y away
                df[f'dif_{var_2}_player_{titularidad}'] = df[f'{var_2}_player_{titularidad}_home'] - df[f'{var_2}_player_{titularidad}_away']

                # Elimino variables utilizadas para calcular la diferencia
                df = df.drop([f'{var_2}_player_{titularidad}_home', f'{var_2}_player_{titularidad}_away'], axis=1)
            
            # Calculo diferencia entre home y away
            df[f'dif_{var}_player_{titularidad}'] = df[f'{var}_player_{titularidad}_home'] - df[f'{var}_player_{titularidad}_away']

            # Elimino variables utilizadas para calcular la diferencia
            df = df.drop([f'{var}_player_{titularidad}_home', f'{var}_player_{titularidad}_away'], axis=1)
    
    df['dif_n_player_miss'] = df['n_player_miss_home'] - df['n_player_miss_away']
    df = df.drop(["n_player_miss_home", "n_player_miss_away"], axis=1) 
    return df

# Main_next_matches.py
def h2h_by_date_new_matches(df_new: pd.DataFrame, df: pd.DataFrame, n_years: int):
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    # Definicion de variables
    n_days = 365 * n_years

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo

    # Por partido nuevo
    for i, row in df_new.iterrows():

        # Obtener equipos
        eq1, eq2 = row['id_team_home'], row['id_team_away']
        limit_date = row['date'] - timedelta(days=n_days)
        h2h = 0

        # Filtrar df_old con partidos de estos equipos en ultimos n años
        filter1 = ((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2)) | ((df['id_team_home'] == eq2) & (df['id_team_away'] == eq1))
        filter2 = (df['date'] >= limit_date) & (df['date'] < row['date'])
        df_filtered = df[filter1 & filter2]

        # Por ultimos matchs
        for idx, fila in df_filtered.iterrows():

            if fila['result'] == 1:
                h2h += +1 if fila['id_team_home'] == row['id_team_home'] else -1

            elif fila['result'] == 2:
                h2h += -1 if fila['id_team_home'] == row['id_team_home'] else +1

            else:
                h2h += 0

        # Guardo h2h
        if len(df_filtered) > 0:  # Para evitar guardar h2h = 0 en matchs donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
            df_new.loc[i, 'h2h'] = h2h

    return df_new

def determine_mean_in_last_matches_next_matches(df_new: pd.DataFrame, df: pd.DataFrame, n_days: int, variable: str, tipo: str = "mean"):
    """
     Obtiene el promedio de las stats en los ultimos matchs

     :param df: DataFrame.
     :param n_days: Integer. Numero de dias de los cuales obtener los datos.
     :param variable: String. Nombre de la variable a promediar.
     :param tipo: String. Tipo de cálculo a realizar ('mean' para promedio, 'sum' para suma).
     :return: DataFrame con stats promediadas
     """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='date', ascending=False)  #ignore_index=True

    # Por partido nuevo
    for idx, row in df_new.iterrows():
        # print(f"Partido: {idx}")

        # Si no tiene un valor asignado aun (A veces ya calcule la variable correctamente al tener disponible las formaciones del partido)
        if not pd.notna(row[variable]):
   
            # Obtengo id de equipos local y visitante
            d_teams = {row['id_team_home']: 'home', row['id_team_away']: 'away'}
            # print(f"Team home: {row['id_team_home']}, Team away: {row['id_team_away']}")

            # Por equipos
            for team, home_or_away in d_teams.items():

                limit_date = row['date'] - timedelta(days=n_days)

                # Filtrar df_old: solo partidos del equipo en ultimos dias
                filter1 = (df['id_team_home'] == team) | (df['id_team_away'] == team)
                filter2 = (df['date'] >= limit_date) & (df['date'] < row['date'])
                df_filtered = df[filter1 & filter2]
                # print(df_filtered)  # Presumo que puede dar siempre 0 el filter...

                # Necesito la posesion segun si fue home o away en cada uno de esos matchs...
                s_valores_home = df_filtered.loc[df_filtered['id_team_home'] == team, variable]
                s_valores_away = df_filtered.loc[df_filtered['id_team_away'] == team, variable] * -1  # -1 puesto que valores positivos en dif_variable es para el home y valores negativos es favor del away
                s_valores = pd.concat([s_valores_home, s_valores_away], ignore_index=True)
                # print(len(s_valores))

                if len(s_valores) > 0:
                    value = s_valores.mean() if tipo == "mean" else (s_valores.sum() if tipo == "sum" else None)
                    df_new.loc[idx, f'{tipo}_last_match_{variable}_{home_or_away}'] = value
            # df_new.to_excel('/Users/nachomondino/Desktop/df_new_with_mean_last_match_columns.xlsx', index=True)

            # Determinar la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            df_new.loc[idx, variable] = df_new.loc[idx, f'mean_last_match_{variable}_home'] - df_new.loc[idx, f'mean_last_match_{variable}_away'] # (e.g. mean_last_match_dif_mean_player_start_home)

    try:
        df_new = df_new.drop(columns=[f'mean_last_match_{variable}_home', f'mean_last_match_{variable}_away'], axis=1)
    except:
        pass
    return df_new

# Prueba
def prueba():
    # Definicion de variables
    country = 'England'
    var_resp = 'result'
    n_days = 30  # 30 es como N_LAST_MATCH igual a 5...
    n_years_h2h = 2
    segun_localia = True

    # Levanto dataset
    df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
    print(df.head())

    start = time.time()
    # df.info()

    # Construyo variables: "equipo_gandor", diferencia de goles y points obtenidos
    # df = determine_result(df, var_resp)
    # df = determine_points(df)

    # Variables historicas
    # df = h2h_by_date(df, n_years=n_years_h2h, segun_localia=segun_localia, _print=True)

    # Por estadistica del match
    l_stats = determine_stats_columns(df)
    print(f"Stats a promediar en ultimos matchs: {l_stats}")

    # Por estadistica del partido
    for var in l_stats: # e.g. shots_on_goal
        print(f"\tEstadistica a promediar: {var}", df[f"{var}_home"].dtype, df[f"{var}_away"].dtype)
        
        if var in ['shots_on_goal']:
    
            # Determine la diferencia de la estadistica entre equipo local y visitante de cada partido
            df[f'dif_{var}'] = df[f'{var}_home'] - df[f'{var}_away']  # (e.g. dif_goles = goles_home - goles_away)
            df = df.drop([f'{var}_home', f'{var}_away'], axis=1)  # (e.g. borro goles_home y goles_away)

            # Determine para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
            df = determine_mean_in_last_matches(df, n_days=n_days, variable=f'dif_{var}', segun_localia=segun_localia, tipo='mean', _print=True)  # mean_last_match_dif_points_home
            df = df.drop([f'dif_{var}'], axis=1)

            # Determine la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            df[f'dif_mean_last_match_dif_{var}'] = df[f'mean_last_match_dif_{var}_home'] - df[f'mean_last_match_dif_{var}_away']  # KeyError: 'mean_last_match_dif_points_home'
            df = df.drop(columns=[f'mean_last_match_dif_{var}_home', f'mean_last_match_dif_{var}_away'], axis=1)

    # Construyo variables de diferencias para las variables promedio de los players
    # df = suma_rat_player_missing(df)
    # df = calculate_dif_col_players(df)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()