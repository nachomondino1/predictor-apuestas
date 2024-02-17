import pandas as pd
import numpy as np
import time
import math
from datetime import timedelta

# Construyo variables en dataframe "match"
def determine_result(df):
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
    valores = ['Home', 'Away']

    # Usar numpy.select para aplicar las condiciones
    df['result'] = pd.Series(np.select(condiciones, valores, default='Draw'), index=df.index)
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

    df.loc[df['result'] == 'Home', 'points_home'] = 3
    df.loc[df['result'] == 'Home', 'points_away'] = 0

    df.loc[df['result'] == 'Draw', 'points_home'] = 1
    df.loc[df['result'] == 'Draw', 'points_away'] = 1

    df.loc[df['result'] == 'Away', 'points_home'] = 0
    df.loc[df['result'] == 'Away', 'points_away'] = 3
    return df

def determine_mean_in_last_match(df, n_days, variable, tipo):
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
    for team in df['team_home'].unique():

        # Obtengo los matchs que jugó el team
        df_equipo = df[(df['team_home'] == team) | (df['team_away'] == team)]
        # print(f"Team: {team}")
        # print(f"Las filas son la cantidad total de matchs del team: {df_equipo.shape}")

        # Por match del team
        for idx, row in df_equipo.iterrows():

            home_or_away = 'home' if row['team_home'] == team else 'away'
            date_limite = row['date'] - timedelta(days=n_days)
            # print(f"Match Nº: {idx}")

            # Selecciono los ultimos matchs del team
            df_equipo_last_matches = df_equipo.loc[(df_equipo['date'] >= date_limite) & (df_equipo['date'] < row['date'])]
            # print(f"Las filas son las cantidad de match en ultimos {n_days} dias: {df_equipo_last_matches.shape}")
            # df_equipo_last_matches.to_excel('/Users/nachomondino/Desktop/df_equipo_last_matches.xlsx')

            # Necesito la posesion segun si fue home o away en cada uno de esos matchs...
            s_valores_home = df_equipo_last_matches.loc[df_equipo_last_matches['team_home'] == team, variable]
            s_valores_away = df_equipo_last_matches.loc[df_equipo_last_matches['team_away'] == team, variable] * -1  # -1 puesto que valores positivos en dif_variable es para el home y valores negativos es favor del away
            s_valores = pd.concat([s_valores_home, s_valores_away], ignore_index=True)
            # print(f"Valores del team en estadistica dif_{variable}: {s_valores}")

            if len(s_valores) > 0:
                if tipo == "mean":
                    df.loc[idx, f'mean_last_match_{variable}_{home_or_away}'] = s_valores.mean()
                    # print(f"Promedio: {s_valores.mean()} \n")

                elif tipo == "sum":
                    df.loc[idx, f'sum_last_match_{variable}_{home_or_away}'] = s_valores.sum()

    return df

def h2h_by_date(df, n_years):  # Borra el indice
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, team_home, team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    # Definicion de variables
    n_days = 365 * n_years
    l_equipos = df['team_home'].unique()
    # print(f"Lista de equipos: {l_equipos}")

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo

    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]
        # print(f"Team 1: {l_equipos[i]}")

        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]
            # print(f"Team 2: {l_equipos[j]}")

            df_historial = df[((df['team_home'] == eq1) & (df['team_away'] == eq2)) | (
                            df['team_home'] == eq2) & (df['team_away'] == eq1)]
            # print(df_historial)

            # Por match del h2h
            for idx, row in df_historial.iterrows():

                team_home = row['team_home']
                h2h = 0
                date_match = row['date']
                date_limite = date_match - timedelta(days=n_days)
                # print(f"Team home en match {idx}: {team_home}".center(120))
                # print(f"Fecha: {date_match} ; Fecha limite: {date_limite}")

                # Selecciono los ultimos matchs
                df_sel = df_historial.loc[(df_historial['date'] >= date_limite) & (df_historial['date'] < date_match)]
                # print(df_sel)

                # Por ultimos matchs
                for index, fila in df_sel.iterrows():

                    if fila['result'] == "Home":
                        h2h += +1 if fila['team_home'] == team_home else -1

                    elif fila['result'] == "Away":
                        h2h += -1 if fila['team_home'] == team_home else +1

                    else:
                        h2h += 0
                    # print(f"Index: {index} ; Team ganador: {fila['result']} ; Team home: {fila['team_home']}")
                    # print(h2h)

                # Guardo h2h
                if len(df_sel) > 0:  # Para evitar guardar h2h = 0 en matchs donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                    df.loc[idx, 'h2h_date'] = h2h
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

            # Calculo diferencia entre home y away
            df[f'dif_{var}_player_{titularidad}'] = df[f'{var}_player_{titularidad}_home'] - df[f'{var}_player_{titularidad}_away']

            # Elimino variables utilizadas para calcular la diferencia
            df = df.drop([f'{var}_player_{titularidad}_home', f'{var}_player_{titularidad}_away'], axis=1)
    return df

def suma_rat_player_missing(df):  # Ojo falla en calculo cuando uno de los dos equipos no tiene players missing (o sea, las variables missing son nan) --> en ese caso tiene que hacer la diferencia igual...
    """
    Calculo la suma del rating de los players missing dado que cada team tiene distinto numero de missing.
    :param df: Dataframe. Unidad de analisis: match.
    :return:
    """
    df['mean_rat_player_miss_home'] = df['mean_rat_player_miss_home'] * df['n_player_miss_home']  # Lo guardo en la misma porque sino la cago con calculate_dif_col_players()
    df['mean_rat_player_miss_away'] = df['mean_rat_player_miss_away'] * df['n_player_miss_away']

    # Elimino columnas que contaban players missing en cada team
    df = df.drop(["n_player_miss_home", "n_player_miss_away"], axis=1)
    return df

def determine_l_stats(df): # Es muy ineficiente creo.
    """
    Determina automaticamente las variables que deben ser promediadas en los ultimos matchs
    :param df: DataFrame. (DataFrame)
    :return: List. Variables a ser promediadas. (List)
    """
    # Definicion de variables
    keywords_prohibidas = ['_player_', 'team_', 'odds_', 'coach_']
    pattern = f'[a-z_]+\_(home|away)'  # posesion_home

    # Filtro inicial (me quedo con algo_home y algo_away)
    l_stats_raw = df.filter(regex=pattern, axis=1).columns.tolist()
    # print(l_stats_raw)

    def validar_palabras_prohibidas(cadena):
        for keyword in keywords_prohibidas:
            if keyword in cadena:
                return False
        return True

    l_stats = [cadena for cadena in l_stats_raw if validar_palabras_prohibidas(cadena)]

    set_est = set()
    for est in l_stats:
        est_filt = est.replace('_home', "").replace('_away', "")
        set_est.add(est_filt)
    return list(set_est)

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
    l_stats = determine_l_stats(df)
    print(f"Stats a promediar en ultimos matchs: {l_stats}")

    df.info()

    for est in l_stats:

        # Determine la diferencia de la estadistica entre team home y away de cada match
        df[f'dif_{est}'] = df[f'{est}_home'] - df[f'{est}_away']  # (e.g. dif_goles = goals_home - goals_away)
        df = df.drop([f'{est}_home', f'{est}_away'], axis=1)  # (e.g. borro goals_home y goals_away)

        # Determine para cada team de un match, el promedio en los ultimos matchs de dicha diferencia de la estadistica
        df = determine_mean_in_last_match(df, n_days=n_days, variable=est, tipo='mean')
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