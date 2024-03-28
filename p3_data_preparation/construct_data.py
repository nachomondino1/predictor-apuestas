import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import re
from p3_data_preparation.clean_data import replace_nan_with_zero

# MAIN.PY
## Variable respuesta y otras
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

## Rendimiento del equipo
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

## Teams
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
    if n_years == -1:
        n_years = (max(df['date']) - min(df['date'])).days / 365
        n_years = int(-(-n_years // 1)) # redondeo hacia arriba numero de años
    n_days =  365 * n_years
    l_equipos = df['id_team_home'].unique()
    h2h_col_name = f'h2h_{n_years}_segun_loc' if segun_localia else f'h2h_{n_years}'

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
                        df.loc[idx, h2h_col_name] = h2h
                        if _print:
                            print(f"Historial a agregar: {h2h/len(df_hist_filt)}")

    return df

## Stats
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

def construct_percentaje_column(df: pd.DataFrame, col_num: str, col_den: str):
    n_col_1, n_col_2 = f'perc_{col_num}_of_{col_den}_home', f'perc_{col_num}_of_{col_den}_away'
    
    df[f'perc_{col_num}_of_{col_den}_home'] = round(df[f'{col_num}_home'] / df[f'{col_den}_home'], 2)
    df[f'perc_{col_num}_of_{col_den}_away'] = round(df[f'{col_num}_away'] / df[f'{col_den}_away'], 2)

    # Reemplazo los porcentajes mayores a uno o menores a cero por None (puesto que no pueden ser mayores a 1 o menores que 0)
    func = lambda x: x if (x <= 1 and x >= 0) else None
    df[n_col_1] = df[n_col_1].apply(func)
    df[n_col_2] = df[n_col_2].apply(func)
    return df

def calculate_dif_col_stats(df, l_stats, n_days, segun_localia): # ver que estan bien los cambios que hice... creo que si igual
    """
    Construye variables diferencia de estadisticas entre local y visitante.
    
    # Parameters
        df:
        n_days:
        segun_localia: 

    # Returns
        ...
    """
    # Por estadistica del partido
    for var in l_stats: # e.g. shots_on_goal
        print(f"Estadistica a promediar: {var}")
        column_home, column_away = f'{var}_home', f'{var}_away'

        # Si la estadistica fue recolectada para estos partidos
        if (column_home in df.columns) and (column_away in df.columns):

            # PROMEDIANDO LA ESTADISTICA Y LUEGO CALCULANDO LA DIFERENCIA       
            # Determine para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
            df = determine_mean_in_last_matches(df, n_days=n_days, variable=var, segun_localia=segun_localia)  # mean_last_match_dif_points_home
            df = df.drop([f'{var}_home', f'{var}_away'], axis=1)  # (e.g. borro goles_home y goles_away)
            
            # Determino la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            not_none_condition = (df[f'mean_last_match_{var}_home'].notnull()) & (df[f'mean_last_match_{var}_away'].notnull())
            df[f'dif_mean_last_match_{var}'] = np.where(not_none_condition, df[f'mean_last_match_{var}_home'] - df[f'mean_last_match_{var}_away'], np.nan)
            df = df.drop(columns=[f'mean_last_match_{var}_home', f'mean_last_match_{var}_away'], axis=1)

            # Determino la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            not_none_condition_2 = (df[f'mean_last_match_{var}_home_against'].notnull()) & (df[f'mean_last_match_{var}_away_against'].notnull())
            df[f'dif_mean_last_match_{var}_against'] = np.where(not_none_condition_2, df[f'mean_last_match_{var}_home_against'] - df[f'mean_last_match_{var}_away_against'], np.nan)
            df = df.drop(columns=[f'mean_last_match_{var}_home_against', f'mean_last_match_{var}_away_against'], axis=1)
    
    return df

def determine_mean_in_last_matches(df: pd.DataFrame, n_days: int, variable: str, segun_localia: bool, _print: bool = False):
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
    n_days = int(n_days*2) if segun_localia else n_days

    # Por partido
    for id_match, row in df.iterrows():

        # Selecciono los ultimos partidos
        limit_date = row['date'] - timedelta(days=n_days)
        df_last_matches = df.loc[(df['date'] >= limit_date) & (df['date'] < row['date'])]

        # Por equipo
        d_teams = {'id_team_home': 'home', 'id_team_away': 'away'}

        # Por equipo
        for col_team, home_or_away in d_teams.items():

            variable_form = f'{variable}_{home_or_away}'
            other = 'away' if home_or_away == 'home' else 'home'
            team = row[col_team]

            # Si es por localia
            if segun_localia: # --> Verificar que calcula bien... lo hice estando cansado...
                
                # Busco promedio en ultimos partidos
                df_matches_team = df_last_matches[df_last_matches[col_team] == team]
  
                # Obtener los valores de la variable para los partidos en casa y fuera de casa
                values_team = df_matches_team[variable_form].values
                values_against_team = df_matches_team[f'{variable}_{other}'].values

                # Remover los valores NaN
                values_team_clean = values_team[~np.isnan(values_team)]
                values_against_team_clean = values_against_team[~np.isnan(values_against_team)]

                # Calcular el número total de partidos
                total_partidos = len(values_team_clean)
                total_partidos_against = len(values_against_team_clean)
                suma = np.sum(values_team_clean)
                suma_against = np.sum(values_against_team_clean)
            
            # Si no es por localia
            else:
                # Busco promedio en ultimos partidos
                df_matches_home_team = df_last_matches[df_last_matches['id_team_home'] == team]
                df_matches_away_team = df_last_matches[df_last_matches['id_team_away'] == team]
                if _print:
                    print("\n DF_MATCH_TEAM_HOME \n", df_matches_home_team.loc[:, ['date', 'id_team_home', 'id_team_away', f'{variable}_home']].head(5))
                    print("\n DF_MATCH_TEAM_AWAY \n", df_matches_away_team.loc[:, ['date', 'id_team_home', 'id_team_away', f'{variable}_away']].head(5))

                # Obtener los valores de la variable para los partidos en casa y fuera de casa
                values_home = df_matches_home_team[f'{variable}_home'].values
                values_away = df_matches_away_team[f'{variable}_away'].values
                values_against_home = df_matches_home_team[f'{variable}_away'].values
                values_against_away = df_matches_away_team[f'{variable}_home'].values

                # Remover los valores NaN
                values_home_clean = values_home[~np.isnan(values_home)]
                values_away_clean = values_away[~np.isnan(values_away)]
                values_against_home_clean = values_against_home[~np.isnan(values_against_home)]
                values_against_away_clean = values_against_away[~np.isnan(values_against_away)]

                # Calcular el número total de partidos
                total_partidos = (len(values_home_clean) + len(values_away_clean))
                total_partidos_against = (len(values_against_home_clean) + len(values_against_away_clean))
                suma = np.sum(values_home_clean) + np.sum(values_away_clean)
                suma_against =  np.sum(values_against_home_clean) + np.sum(values_against_away_clean)

            # Si hay al menos un valor que promediar, guardo promedio
            if total_partidos > 0:
                # Calcular el promedio
                df.loc[id_match, f'mean_last_match_{variable_form}'] = suma / total_partidos
                if _print:
                    print(f"Valor a rellenar: {suma / total_partidos} en {variable_form}")
            
            if total_partidos_against > 0:
                df.loc[id_match, f'mean_last_match_{variable_form}_against'] = suma_against / total_partidos_against
             

    return df

## Player
def calculate_dif_col_players(df: pd.DataFrame):
    """
    Calcula la diferencia entre home y away
    :param df: Dataframe. Unidad de analisis: match
    :return:
    """
    # Determinar columnas players (e.g.sum_rat_player_miss)
    pattern = r'_player_[a-z_\(\)%]+_(home|away)' # Patrón regex para encontrar columnas relevantes
    relevant_columns = df.filter(regex=pattern, axis=1).columns
    l_var_sin_suffix = [re.sub(r'_(home|away)$', '', col) for col in relevant_columns]
    print(f"Variables jugadores a calcular diferencia entre local y visitante: {l_var_sin_suffix}")

    for var in l_var_sin_suffix:
        print(f"Variable: {var}")
        columna_home, columna_away = f'{var}_home', f'{var}_away'

        if (columna_home in df.columns) and (columna_away in df.columns):
            if (('sum_' in var) or ('n_player' in var)) and ('_miss' in var):
                # Creo una copia del df para evitar sum=nan (nan*0=nan) cdo uno de los dos equipos no tiene jugadores ausentes.
                print("\t Reemplazo NaN values por cero.")
                df = replace_nan_with_zero(df, columna_home, columna_away)  # Reemplazo sum_rat_player_miss=nan por sum_rat_player_miss=0 cdo uno de los dos equipos no tiene jugadores ausentes y el otro si

            # Calculo diferencia entre home y away cuando ambos equipos no tienen NaN
            not_none_condition = (df[columna_home].notnull()) & (df[columna_away].notnull())
            df[f'dif_{var}'] = np.where(not_none_condition, df[columna_home] - df[columna_away], np.nan)

            # Elimino variables utilizadas para calcular la diferencia
            df = df.drop([columna_home, columna_away], axis=1)
    
    return df

# MAIN_NEXT_MATCHES.PY
def h2h_by_date_new_matches(df_new: pd.DataFrame, df: pd.DataFrame, n_years: int, segun_localia: bool): # Me gustaria juntarla con h2h de main.py
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    # Definicion de variables
     # Definicion de variables
    if n_years == -1:
        n_years = (max(df['date']) - min(df['date'])).days / 365
        n_years = int(-(-n_years // 1)) # redondeo hacia arriba numero de años
    n_days =  365 * n_years
    h2h_col_name = f'h2h_{n_years}_segun_loc' if segun_localia else f'h2h_{n_years}'

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo

    # Por partido nuevo
    for i, row in df_new.iterrows():

        # Obtener equipos
        eq1, eq2 = row['id_team_home'], row['id_team_away']
        limit_date = row['date'] - timedelta(days=n_days)
        h2h = 0

        # Determino df_historial para eq1 y eq2
        df_historial = df[((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2)) | (df['id_team_home'] == eq2) & (df['id_team_away'] == eq1)]
        l_dfs = [df_historial] if not segun_localia else [df_historial[df_historial['id_team_home'] == eq1], df_historial[df_historial['id_team_home'] == eq2]]

        for df_hist in l_dfs:

            # Filtrar df_old con partidos de estos equipos en ultimos n años
            # filter1 = ((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2)) | ((df['id_team_home'] == eq2) & (df['id_team_away'] == eq1))
            filter2 = (df_hist['date'] >= limit_date) & (df_hist['date'] < row['date'])
            df_hist_filt = df_hist[filter2]  # df_hist_filt = df[filter1 & filter2]

            # Por ultimos matchs
            for idx, fila in df_hist_filt.iterrows():

                if fila['result'] == 1:
                    h2h += +1 if fila['id_team_home'] == row['id_team_home'] else -1

                elif fila['result'] == 2:
                    h2h += -1 if fila['id_team_home'] == row['id_team_home'] else +1

            # Guardo h2h
            if len(df_hist_filt) > 0:  # Para evitar guardar h2h = 0 en matchs donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                df_new.loc[i, h2h_col_name] = h2h

    return df_new

# PRUEBA
def prueba():
    # Definicion de variables
    country = 'Argentina'
    var_resp = 'result'
    n_days = 30  # 30 es como N_LAST_MATCH igual a 5...
    n_years_h2h = 2
    segun_localia = True

    # Levanto dataset
    df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
    df = df.sort_values(by='date', ascending=False)
    df = df.head(200)
    print(df.head())

    start = time.time()
    # df.info()

    # Construyo variables: "equipo_gandor"
    df = determine_result(df, var_resp)

    # Rendimiento del equipo
    df = determine_points(df)
    df = h2h_by_date(df, n_years=-1, segun_localia=True) # QUIERO USAR EL MAXIMO HISTORIAL, NO QUIERO TENER QUE DECIRLE...
    df = h2h_by_date(df, n_years=-1, segun_localia=False) # QUIERO USAR EL MAXIMO HISTORIAL, NO QUIERO TENER QUE DECIRLE...
    df = h2h_by_date(df, n_years=n_years_h2h, segun_localia=True)
    df = h2h_by_date(df, n_years=n_years_h2h, segun_localia=False)

    # STATS
    stats_columns = determine_stats_columns(df)
    print(f"Stats a promediar en ultimos partidos: {stats_columns}")
    
    # Construyo variables porcentajes (funciona ok!) No tira error de division ni nada. Es nan solo cuando es 0/0 (sin tiirar error).
    # df = construct_percentaje_column(df, col_num="shots_on_goal", col_den="goal_attempts")
    # df = construct_percentaje_column(df, col_num="goals", col_den="goal_attempts")

    # Determino cuales son las variables stats automaticamente
    df = calculate_dif_col_stats(df, stats_columns, n_days, segun_localia)

    # PLAYER 
    # Construyo variables de diferencias para las variables promedio de los players
    df = determine_mean_in_last_matches(df, n_days, variable='mean_rat_player_start', segun_localia=segun_localia)
    df = df.drop(columns=['mean_last_match_mean_rat_player_start_home', 'mean_last_match_mean_rat_player_start_away'], axis=1)
    df = calculate_dif_col_players(df)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()