import pandas as pd
import numpy as np
import time
from datetime import timedelta
import re
from p3_data_preparation.clean_data import replace_nan_with_zero

# Construyo variables en dataframe "match"
def determine_result(df, var_resp):
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

def determine_points(df):
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

def construct_percentaje_column(df, col_num, col_den):
    df[f'perc_{col_num}_of_{col_den}_home'] = df[f'{col_num}_home'] / df[f'{col_den}_home']
    df[f'perc_{col_num}_of_{col_den}_away'] = df[f'{col_num}_away'] / df[f'{col_den}_away']
    return df

def determine_mean_in_last_matches(df, n_days, variable, tipo):
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

    # Por team
    for team in df['id_team_home'].unique():

        # Obtengo los matchs que jugó el team
        df_equipo = df[(df['id_team_home'] == team) | (df['id_team_away'] == team)]
        # print(f"Team: {team}")
        # print(f"Las filas son la cantidad total de matchs del team: {df_equipo.shape}")
        # print(df_equipo.loc[:, ['id_team_home', 'id_team_away']].head(5))

        # Por match del team
        for idx, row in df_equipo.iterrows():

            home_or_away = 'home' if row['id_team_home'] == team else 'away'
            limit_date = row['date'] - timedelta(days=n_days)
            # print(f"Match Nº: {idx}")

            # Selecciono los ultimos matchs del team
            df_equipo_last_matches = df_equipo.loc[(df_equipo['date'] >= limit_date) & (df_equipo['date'] < row['date'])]
            # print(f"Las filas son las cantidad de match en ultimos {n_days} dias: {df_equipo_last_matches.shape}")
            # df_equipo_last_matches.to_excel('/Users/nachomondino/Desktop/df_equipo_last_matches.xlsx')

            # Necesito la posesion segun si fue home o away en cada uno de esos matchs...
            s_valores_home = df_equipo_last_matches.loc[df_equipo_last_matches['id_team_home'] == team, variable]
            s_valores_away = df_equipo_last_matches.loc[df_equipo_last_matches['id_team_away'] == team, variable] * -1  # -1 puesto que valores positivos en dif_variable es para el home y valores negativos es favor del away
            s_valores = pd.concat([s_valores_home, s_valores_away], ignore_index=True)
            # print(f"Valores del team en estadistica dif_{variable}: {s_valores}")

            if len(s_valores) > 0:
                value = s_valores.mean() if tipo == "mean" else (s_valores.sum() if tipo == "sum" else None)
                # print(f"Valor: {value} \n")
                df.loc[idx, f'{tipo}_last_match_{variable}_{home_or_away}'] = value

    return df

def h2h_by_date(df, n_years, segun_localia: bool = False):
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

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo

    # Por equipo 1
    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]

        # Por equipo 2
        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]

            # Obtengo partidos entre los equipos 1 y 2
            if segun_localia:
                df_historial = df[((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2))] # --> POR LOCALIA
            else:
                df_historial = df[((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2)) | (df['id_team_home'] == eq2) & (df['id_team_away'] == eq1)] 
            # print(f"\n Equipo 1: {eq1} Equipo 2: {eq2}")
            # print(df_historial.loc[:, ['id_team_home', 'id_team_away']])

            # Por partido entre equipos
            for idx, row in df_historial.iterrows():

                h2h = 0
                limit_date = row['date'] - timedelta(days=n_days)

                # Selecciono los ultimos matchs
                df_sel = df_historial.loc[(df_historial['date'] >= limit_date) & (df_historial['date'] < row['date'])]
                # print("Partidos para construir historial: ", len(df_sel))
                # print(df_sel.loc[:, ['id_team_home', 'id_team_away']])

                # Por ultimos matchs
                for index, fila in df_sel.iterrows():

                    if fila['result'] == 1:
                        h2h += +1 if fila['id_team_home'] == row['id_team_home'] else -1

                    elif fila['result'] == 2:
                        h2h += -1 if fila['id_team_home'] == row['id_team_home'] else +1

                    else:
                        h2h += 0
                    # print(i, h2h)

                # Guardo h2h
                if len(df_sel) > 0:  # Para evitar guardar h2h = 0 en matchs donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                    df.loc[idx, 'h2h_date'] = h2h
    return df

def suma_rat_player_missing(df):  # Ojo falla en calculo cuando uno de los dos equipos no tiene players missing (o sea, las variables missing son nan) --> en ese caso tiene que hacer la diferencia igual...
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

def calculate_dif_col_players(df):
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

def determine_stats_columns(df):
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

# Para proximos partidos
def h2h_by_date_new_matches(df_new, df, n_years):
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
            df_new.loc[i, 'h2h_date'] = h2h

    return df_new

def determine_mean_in_last_matches_next_matches(df_new, df, n_days, variable, tipo):
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
    country = 'Italia'
    n_days = 30  # 30 es como N_LAST_MATCH igual a 5...
    n_years_h2h = 2

    # Levanto dataset
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{country}/df_integrated.xlsx')
    print(df.head())

    start = time.time()

    # Construyo variables: "equipo_gandor", diferencia de goles y points obtenidos
    df = determine_result(df)
    df = determine_points(df)

    # Variables historicas
    df = h2h_by_date(df, n_years=n_years_h2h)

    # Por estadistica del match
    l_stats = determine_stats_columns(df)
    print(f"Stats a promediar en ultimos matchs: {l_stats}")

    df.info()

    for est in l_stats:

        # Determine la diferencia de la estadistica entre team home y away de cada match
        df[f'dif_{est}'] = df[f'{est}_home'] - df[f'{est}_away']  # (e.g. dif_goles = goals_home - goals_away)
        df = df.drop([f'{est}_home', f'{est}_away'], axis=1)  # (e.g. borro goals_home y goals_away)

        # Determine para cada team de un match, el promedio en los ultimos matchs de dicha diferencia de la estadistica
        df = determine_mean_in_last_matches(df, n_days=n_days, variable=est, tipo='mean')
        df = df.drop([f'dif_{est}'], axis=1)

        # Determine la diferencia entre promedio del home y del away (por ej, diferencia entre mean_dif_goals_home y mean_dif_goals_away)
        df[f'dif_mean_last_match_dif_{est}'] = df[f'mean_last_match_dif_{est}_home'] - df[f'mean_last_match_dif_{est}_away']
        df = df.drop(columns=[f'mean_last_match_dif_{est}_home', f'mean_last_match_dif_{est}_away'], axis=1)

    # Construyo variables de diferencias para las variables promedio de los players
    # df = suma_rat_player_missing(df)
    df = calculate_dif_col_players(df)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()