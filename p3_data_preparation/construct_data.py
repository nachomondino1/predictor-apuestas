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

def determine_expected_result(df: pd.DataFrame, col_name="expected_result"):
    """
    Se determina el 'result' a partir de los expected goals que hizo cada team
    :param df: Dataframe. Unidad de analisis: match. Columnas: entre ellas expected_goals_(xg)_home y expected_goals_(xg)_away
    :return: Dataframe pasado por parametro con nueva columna, 'result', que detalla el resultado del match.
    """
    # Calcular dif_expected_goals solo dentro de las condiciones, sin crear una nueva columna
    dif_expected_goals = df['expected_goals_(xg)_home'] - df['expected_goals_(xg)_away']

    # Condiciones para determinar el ganador
    condiciones = [
        dif_expected_goals > 1,
        dif_expected_goals < -1,
    ]

    # Valores correspondientes a las condiciones
    valores = [1, 2]

    # Usar numpy.select para aplicar las condiciones y asignar directamente a col_name
    df[col_name] = pd.Series(np.select(condiciones, valores, default=0), index=df.index)
    return df

def determine_number_matches_last_days(df: pd.DataFrame, n_days):

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
                df.loc[id_match, f'n_matches_last_{n_days}_days_{home_or_away}'] = len(df_match_team_filt)
            else:
                df.loc[id_match, f'n_matches_last_{n_days}_days_{home_or_away}'] = np.nan

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

def determine_expected_points(df: pd.DataFrame):
    """
    Determina los points obtenidos por cada team segun el resultado del juego.

    :param df:
    :return:
    """
    # Inicializo las columnas "points_home" y "points_away"
    df['expected_points_home'] = 0
    df['expected_points_away'] = 0

    df.loc[df['expected_result'] == 1, 'expected_points_home'] = 3
    df.loc[df['expected_result'] == 1, 'expected_points_away'] = 0

    df.loc[df['expected_result'] == 0, 'expected_points_home'] = 1
    df.loc[df['expected_result'] == 0, 'expected_points_away'] = 1

    df.loc[df['expected_result'] == 2, 'expected_points_home'] = 0
    df.loc[df['expected_result'] == 2, 'expected_points_away'] = 3
    return df

## Teams
def h2h_by_date(df: pd.DataFrame, n_years: int, _print: bool = False):
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    print("\n Constructing Head to Head...")
    # Definicion de variables
    start = time.time()

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo

    if n_years == -1:
        n_years = (max(df['date']) - min(df['date'])).days / 365
        n_years = int(-(-n_years // 1)) # redondeo hacia arriba numero de años
    n_days =  365 * n_years
    h2h_col_name = f'h2h_{n_years}'
    print(f"Historial a construir: {h2h_col_name} para {n_years}")

    # Determinar equipos a calcular el historial
    frecuencias = df['id_team_home'].value_counts()
    threshold = int(0.001 * len(df))
    l_equipos = frecuencias[frecuencias > threshold].index.tolist()
    if _print:
        print(f"Integer {threshold} ; Cantidad de equipos: {len(l_equipos)}")

    # Por equipo 1
    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]

        # Por equipo 2
        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]

            # Determino df_historial para eq1 y eq2
            df_hist = df[((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2)) | (df['id_team_home'] == eq2) & (df['id_team_away'] == eq1)]

            # Por partido entre equipos
            for idx, row in df_hist.iterrows():

                # Selecciono los ultimos matchs
                limit_date = row['date'] - timedelta(days=n_days)
                df_hist_filt = df_hist.loc[(df_hist['date'] >= limit_date) & (df_hist['date'] < row['date'])]
                
                # Por ultimos matchs
                h2h = 0
                for _, fila in df_hist_filt.iterrows():
                    if fila['result'] == 1:
                        h2h += +1 if fila['id_team_home'] == row['id_team_home'] else -1
                    elif fila['result'] == 2:
                        h2h += -1 if fila['id_team_home'] == row['id_team_home'] else +1

                # Guardo h2h
                if len(df_hist_filt) > 0:  # Para evitar guardar h2h = 0 en matchs donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                    df.loc[idx, h2h_col_name] = h2h

    end = time.time()
    print(f"Construccion de historiales in {(end - start) / 60:.1f} minutes")
    return df

def h2h_by_date_by_localia(df: pd.DataFrame, n_years: int, _print: bool = False):
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    print("\n Constructing Head to Head...")
    # Definicion de variables
    start = time.time()

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo

    if n_years == -1:
        n_years = (max(df['date']) - min(df['date'])).days / 365
        n_years = int(-(-n_years // 1)) # redondeo hacia arriba numero de años
    n_days =  365 * n_years
    h2h_col_name = f'h2h_{n_years}_segun_loc'
    print(f"Historial a construir: {h2h_col_name} para {n_years}")

    # Determinar equipos a calcular el historial
    frecuencias = df['id_team_home'].value_counts()
    threshold = int(0.001 * len(df))
    l_equipos = frecuencias[frecuencias > threshold].index.tolist()
    if _print:
        print(f"Integer {threshold} ; Cantidad de equipos: {len(l_equipos)}")
    
    # Por equipo 1
    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]

        # Por equipo 2
        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]

            # Determino df_historial para eq1 y eq2
            df_historial = df[((df['id_team_home'] == eq1) & (df['id_team_away'] == eq2)) | (df['id_team_home'] == eq2) & (df['id_team_away'] == eq1)]
            l_dfs = [df_historial[df_historial['id_team_home'] == eq1], df_historial[df_historial['id_team_home'] == eq2]]

            # Segun localia
            for df_hist in l_dfs:

                # Por partido entre equipos
                for idx, row in df_hist.iterrows():

                    # Selecciono los ultimos matchs
                    limit_date = row['date'] - timedelta(days=n_days)
                    df_hist_filt = df_hist.loc[(df_hist['date'] >= limit_date) & (df_hist['date'] < row['date'])]

                    # Por ultimos matchs
                    h2h = 0
                    for _, fila in df_hist_filt.iterrows():
                        if fila['result'] == 1:
                            h2h += +1 if fila['id_team_home'] == row['id_team_home'] else -1
                        elif fila['result'] == 2:
                            h2h += -1 if fila['id_team_home'] == row['id_team_home'] else +1

                    # Guardo h2h
                    if len(df_hist_filt) > 0:  # Para evitar guardar h2h = 0 en matchs donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                        df.loc[idx, h2h_col_name] = h2h

    end = time.time()
    print(f"Construccion de historiales in {(end - start) / 60:.1f} minutes")
    return df

## Stats
def determine_stats_columns(df: pd.DataFrame):
    """
    Determina automáticamente las variables que deben ser promediadas en los últimos partidos.
    
    # Parameters:
        df: DataFrame.

    # Returns:
        Lista de variables a ser promediadas.
    """
    # Definicion variables
    keywords_prohibidas = ['_player_', 'team_', 'odds_', 'coach_']  # Definir palabras clave prohibidas 
    pattern = r'[a-zA-Z_\(\)%]+\_(home|away)' # Patrón regex para encontrar columnas relevantes

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

def construct_sum_columns(df: pd.DataFrame, l_columns: list, column_name: str = None):
    """
    Crea dos nuevas columnas en el DataFrame, una con la suma de las columnas que contienen '_home' 
    y otra con la suma de las columnas que contienen '_away'.
    
    :param df: DataFrame donde se aplicará la suma de columnas.
    :param l_columns: Lista de nombres de columnas base, a las que se les agrega el sufijo '_home' y '_away'.
    :param column_name: Nombre base de las nuevas columnas. Si no se especifica, se asigna uno por defecto.
    :return: DataFrame con las nuevas columnas agregadas.
    """
    # Generar nombres por defecto si no se especifican
    if column_name is None:
        column_name = f'sum_{"_".join(l_columns)}'
    
    # Crear las listas de columnas '_home' y '_away'
    home_columns = [f"{col}_home" for col in l_columns if f"{col}_home" in df.columns]
    away_columns = [f"{col}_away" for col in l_columns if f"{col}_away" in df.columns]
    
    # Sumar las columnas con sufijo '_home'
    df[f'{column_name}_home'] = df[home_columns].sum(axis=1) if home_columns else None
    
    # Sumar las columnas con sufijo '_away'
    df[f'{column_name}_away'] = df[away_columns].sum(axis=1) if away_columns else None
    
    return df

def construct_percentaje_column(df: pd.DataFrame, col_num: str, col_den: str, column_name:str=None, laplace: bool = False):
    """
    Nueva columna siendo el porcentaje resultante de la division de otras dos columnas.
    """

    col_name = f'perc_{col_num}_of_{col_den}' if column_name is None else column_name

    num1 = df[f'{col_num}_home'] + 1 if laplace else df[f'{col_num}_home']
    num2 = df[f'{col_num}_away'] + 1 if laplace else df[f'{col_num}_away']

    # Home
    df[f'{col_name}_home'] = np.where(
        df[f'{col_den}_home'].notna(), 
        num1 / df[f'{col_den}_home'], 
        None 
    )

    # Away
    df[f'{col_name}_away'] = np.where(
        df[f'{col_den}_away'].notna(),  
        num2 - df[f'{col_den}_away'], 
        None  
    )

    return df

def determine_mean_in_last_matches(df: pd.DataFrame, n_days: int, variable: str, segun_localia: bool, calculate_dif: bool = True, _print: bool = False):
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
    n_days = int(n_days*2) if segun_localia else n_days  # no me gusta esto... o si? Tecnicamente tambien tiene efecto en main_next_matches.py porque uso esta funcion... asique no habria problema.

    # inicializo diccionarios (para evitar Performance Warning)
    mean_last_matches = {}
    mean_last_matches_against = {}

    # Por partido
    for id_match, row in df.iterrows():

        # Selecciono los ultimos partidos
        limit_date = row['date'] - timedelta(days=n_days)
        df_last_matches = df.loc[(df['date'] >= limit_date) & (df['date'] < row['date'])]

        # Por equipo
        d_teams = {'id_team_home': 'home', 'id_team_away': 'away'}
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

                # Convertir los valores a numéricos y forzar errores como NaN
                values_team = pd.to_numeric(values_team, errors='coerce')
                values_against_team = pd.to_numeric(values_against_team, errors='coerce')

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

                # Convertir los valores a numéricos y forzar errores como NaN
                values_home = pd.to_numeric(values_home, errors='coerce')
                values_away = pd.to_numeric(values_away, errors='coerce')
                values_against_home = pd.to_numeric(values_against_home, errors='coerce')
                values_against_away = pd.to_numeric(values_against_away, errors='coerce')

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

            # Calcular el promedio  (COMO EVITAR WARNING?)
            if total_partidos > 0 and total_partidos_against > 0:
                # Guardar en los diccionarios
                mean_last_matches[(id_match, variable_form)] = suma / total_partidos
                mean_last_matches_against[(id_match, variable_form)] = suma_against / total_partidos_against

                # df.loc[id_match, f'mean_last_{n_days}_matches_{variable_form}'] = suma / total_partidos
                # df.loc[id_match, f'mean_last_{n_days}_matches_{variable_form}_against'] = suma_against / total_partidos_against
                # if _print:
                #     print(f"Valor a rellenar: {suma / total_partidos} en {variable_form}")

    # Convertir los diccionarios a columnas del DataFrame
    for home_or_away in ['home', 'away']:
        variable_form = f'{variable}_{home_or_away}'
        
        # Asignar la columna para cada combinación de id_match y variable_form
        df[f'mean_last_{n_days}_matches_{variable_form}'] = df.apply(
            lambda row: mean_last_matches.get((row.name, variable_form), None), axis=1
        )
        
        df[f'mean_last_{n_days}_matches_{variable_form}_against'] = df.apply(
            lambda row: mean_last_matches_against.get((row.name, variable_form), None), axis=1
        )

    # Calculo diferencia entre local y visitante
    if calculate_dif:
        not_none_condition = (df[f'mean_last_{n_days}_matches_{variable}_home'].notnull()) & (df[f'mean_last_{n_days}_matches_{variable}_away'].notnull())
        df[f'dif_mean_last_{n_days}_matches_{variable}'] = np.where(not_none_condition, df[f'mean_last_{n_days}_matches_{variable}_home'] - df[f'mean_last_{n_days}_matches_{variable}_away'], np.nan)
        df = df.drop(columns=[f'mean_last_{n_days}_matches_{variable}_home', f'mean_last_{n_days}_matches_{variable}_away'], axis=1)

        # Calculo diferencia entre local y visitante against
        not_none_condition_2 = (df[f'mean_last_{n_days}_matches_{variable}_home_against'].notnull()) & (df[f'mean_last_{n_days}_matches_{variable}_away_against'].notnull())
        df[f'dif_mean_last_{n_days}_matches_{variable}_against'] = np.where(not_none_condition_2, df[f'mean_last_{n_days}_matches_{variable}_home_against'] - df[f'mean_last_{n_days}_matches_{variable}_away_against'], np.nan)
        df = df.drop(columns=[f'mean_last_{n_days}_matches_{variable}_home_against', f'mean_last_{n_days}_matches_{variable}_away_against'], axis=1)

    return df

## Player
def calculate_dif_col_players(df: pd.DataFrame):
    """
    Calcula la diferencia entre home y away

    # Parameters:
        df: Dataframe con columnas de jugadores

    # Returns
        Dataframe pasado como parametro habiendo construido columnas "diferencias" entre local y visitante.
    """
    # Determinar columnas players (e.g.sum_rat_player_miss)
    pattern = r'_player_[a-z_\(\)%]+_(home|away)' # Patrón regex para encontrar columnas relevantes
    relevant_columns = df.filter(regex=pattern, axis=1).columns
    l_var_sin_suffix = list({re.sub(r'_(home|away)$', '', col) for col in relevant_columns})
    # print(f"Variables jugadores a calcular diferencia entre local y visitante: {l_var_sin_suffix}")
    
    # Por columna de jugadores
    for var in l_var_sin_suffix:
        columna_home, columna_away = f'{var}_home', f'{var}_away'

        if  "_against" not in var:  # Temporalmente. Porque falla sin con la columna player que es against.
            # print(f"\nVariable: {var}")
            # print(f"Columna home: {columna_home} ; Columna away: {columna_away}")
            
            if (('sum_' in var) or ('n_player' in var)) and ('_miss' in var):  # Si quiero reemplazar tambien las 'mean' --> if ('_miss' in var):
                df = replace_nan_with_zero(df, columna_home, columna_away)  # Reemplazo sum_rat_player_miss=nan por sum_rat_player_miss=0 cdo uno de los dos equipos no tiene jugadores ausentes y el otro si
                # print("\t Reemplazo NaN values por cero.")

            try:
                # Calculo diferencia entre home y away cuando ambos equipos no tienen NaN
                not_none_condition = (df[columna_home].notnull()) & (df[columna_away].notnull())
                df[f'dif_{var}'] = np.where(not_none_condition, df[columna_home] - df[columna_away], np.nan)

                # Elimino variables utilizadas para calcular la diferencia
                df = df.drop([columna_home, columna_away], axis=1)
            except KeyError:
                pass
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    # Definicion de variables
    country = 'england'
    var_resp = 'result'
    n_days = 30  # 30 es como N_LAST_MATCH igual a 5...
    n_years_h2h = 10
    segun_localia = True

    # Levanto dataset
    df = pd.read_excel(f'./data/{country}/p3_data_preparation/df_integrated.xlsx', index_col=0)
    df = df.sort_values(by='date', ascending=False)
    # df = df.head(5000)
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
    df = h2h_by_date(df, n_years=n_years_h2h, segun_localia=True)

    # STATS
    # stats_columns = determine_stats_columns(df)
    # print(f"Stats a promediar en ultimos partidos: {stats_columns}")
    
    # Construyo variables porcentajes (funciona ok!) No tira error de division ni nada. Es nan solo cuando es 0/0 (sin tiirar error).
    # df = construct_percentaje_column(df, col_num="shots_on_goal", col_den="goal_attempts")
    # df = construct_percentaje_column(df, col_num="goals", col_den="goal_attempts")

    # Determino cuales son las variables stats automaticamente
    # df = calculate_dif_col_stats(df, stats_columns, n_days, segun_localia)

    # PLAYER 
    # Construyo variables de diferencias para las variables promedio de los players
    # df = determine_mean_in_last_matches(df, n_days, variable='mean_rat_player_start', segun_localia=segun_localia)
    # df = df.drop(columns=['mean_last_match_mean_rat_player_start_home', 'mean_last_match_mean_rat_player_start_away'], axis=1)
    # df = calculate_dif_col_players(df)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel(f'{BASE_DIR_LOCAL}/df_constructed_prueba.xlsx')