import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import re
from p3_data_preparation.clean_data import replace_nan_with_zero

# MAIN.PY
## Variable respuesta y otras
def determine_result(df: pd.DataFrame, var_resp: str = 'result'):
    """
    Determina el 'result' a partir de los goles que hizo cada equipo.
    
    :param df: DataFrame con columna 'goals_home' y 'goals_away'.
    :param var_resp: Nombre de la nueva columna de resultado.
    :return: DataFrame con la nueva columna 'result'.
    """
    df = df.copy()  # Evitar modificar el original

    df[var_resp] = df.apply(lambda row: 1 if row['goals_home'] > row['goals_away'] 
                            else 2 if row['goals_home'] < row['goals_away'] 
                            else 0, axis=1)

    return df

def compare_distributions(
    df: pd.DataFrame,
    col_expected: str = "expected_result",
    col_result: str = "result",
    verbose: int = 1
) -> float:
    """
    Compara la distribución de valores en 'col_expected' y 'col_result' por valor individual.
    Devuelve la diferencia máxima entre proporciones.
    
    :param df: DataFrame con las columnas a comparar.
    :param col_expected: Nombre de la columna de resultados esperados.
    :param col_result: Nombre de la columna de resultados reales.
    :param tolerance: Tolerancia máxima permitida para la diferencia en las proporciones.
    :param exclude_classes: Lista de valores a excluir de la comparación.
    :return: Diferencia máxima entre las proporciones.
    """
    diffs = {}

    # Normalizar los conteos de valores
    expected_counts = df[col_expected].value_counts(normalize=True, dropna=True).sort_index()
    result_counts = df[col_result].value_counts(normalize=True, dropna=True).sort_index()

    # Imprimir las distribuciones
    if verbose >= 2:
        print("Distribución de valores en expected_result:")
        print(expected_counts)
        print("\nDistribución de valores en result:")
        print(result_counts)

    for value in sorted(set(expected_counts.index).union(result_counts.index)):
        expected_ratio = expected_counts.get(value, 0)
        result_ratio = result_counts.get(value, 0)
        diffs[value] = abs(expected_ratio - result_ratio)

    return diffs

def adjust_ratio_to_match_distributions(
    df: pd.DataFrame,
    initial_ratio: float = 0.3,
    step: float = 0.01,
    max_iterations: int = 50,
    col_expected="expected_result",
    col_actual="result",
    verbose: int = 0
) -> float:
    """
    Ajusta el ratio para que las distribuciones de las columnas sean similares.
    :param df: DataFrame con las columnas 'expected_goals_(xg)_home' y 'expected_goals_(xg)_away'.
    :param initial_ratio: Umbral inicial.
    :param tolerance: Diferencia máxima permitida entre distribuciones (en porcentaje).
    :param max_iterations: Número máximo de iteraciones.
    :param col_expected: Nombre de la columna de resultados esperados.
    :param col_actual: Nombre de la columna de resultados reales.
    :return: Ratio ajustado.
    """
    ratio = initial_ratio
    rows = []

    for _ in range(max_iterations):

        if verbose >= 1:
            print(f"Threshold a probar: {ratio:.2f}%")

        # Calcular la columna de resultados esperados
        df = determine_expected_result(df, goals_to_xg_ratio=ratio, col_name=col_expected, verbose=0)
        
        # Comparar las distribuciones y calcular diferencia por resultado
        diff = compare_distributions(df, col_expected, col_actual, verbose=0)
        total_diff = sum(diff.values())  # o diff.sum() si usás una Serie
        if verbose >= 1:
            print(diff)

        # Guardar los resultados
        rows.append({'ratio': ratio, 'diff': total_diff})

        # Ajustar el ratio
        ratio += step

    # Convertir a DataFrame
    df_results = pd.DataFrame(rows)

    # Seleccionar el ratio que minimiza la diferencia
    best_row = df_results.loc[df_results['diff'].idxmin()]
    best_ratio = best_row['ratio']
    best_diff = best_row['diff']

    if verbose >= 0:
        logger.critical(f"Ratio final tras {max_iterations} iteraciones: {best_ratio:.2f}, Diferencia: {best_diff:.2f}%")

    return best_ratio

def determine_expected_result(df: pd.DataFrame, goals_to_xg_ratio_sup: float = 0.46, goals_to_xg_ratio_inf: float = -0.35, col_name="expected_result", verbose: int = 0):
    """
    Determina el 'expected_result' a partir de los expected goals de cada equipo.
    :param df: DataFrame con columnas 'expected_goals_(xg)_home' y 'expected_goals_(xg)_away'.
    :param goals_to_xg_ratio: Ratio entre diferencia de goals y diferencia de expected goals. 
        Si usas 0.3, significa que la diferencia de 0.3 en expected goals, equivale a la diferencia de 1 goal. 
        Por lo tanto, si los expected goals en un partido son 1 y 0.6, la dif de expected seria de 0.4, por ende, tomo que hay la diferencia de 1 goal y, por ende, 
        el ganador deberia haber sido el local.
    :param col_name: Nombre de la columna de resultado esperado.
    :return: DataFrame con la nueva columna 'expected_result'.
    """
    # Calcular la diferencia de expected goals
    dif_expected_goals = df['expected_goals_(xg)_home'] - df['expected_goals_(xg)_away']

    # Seleccionar filas válidas (ignorar NaN en expected_goals)
    filas_validas = dif_expected_goals.notna()

    # Definir condiciones y valores
    condiciones = [
        dif_expected_goals[filas_validas] > goals_to_xg_ratio_sup,
        dif_expected_goals[filas_validas] < goals_to_xg_ratio_inf,
    ]
    valores = [1, 2]  # 1: Local, 2: Visitante

    # Crear una columna con valores por defecto para todas las filas
    df = df.copy() # Evita SettingWithCopyWarning
    df[col_name] = np.nan
    df.loc[filas_validas, col_name] = np.select(condiciones, valores, default=0)

    if verbose >= 1:
        compare_distributions(df)

    return df

def determine_expected_result_ml(df):

    ## Op 2: ML (Funciona? --> si pero peor)
    import pickle

    # Load model
    loaded_model = pickle.load(open("./data/expected_result.pkl", "rb")) 

    # Seleccion columnas predictoras + variable respuesta
    X = df.loc[:, ['id_country', 'id_competition', 'expected_goals_(xg)_home', 'expected_goals_(xg)_away']]

    # Treat nan: Elimino nan values en varibles predictoras
    X.dropna(subset=['expected_goals_(xg)_home', 'expected_goals_(xg)_away'], inplace=True)

    # Predecir expected result
    y_pred = loaded_model.predict(X)
    df.loc[X.index, 'expected_result'] = y_pred  # Guardar y_pred como 'expected_result'
    return df

# ELO
def assign_elo_before_match(df, k=30, base_rating=1500):
    ratings = {}  # ELO actual de cada equipo
    elo_home_list = []
    elo_away_list = []

    df = df.sort_values('date')  # si tenés una columna de fecha

    # Por partido
    for idx, row in df.iterrows():
        team_home = row['id_team_home']
        team_away = row['id_team_away']
        result = row['result']

        r_home = ratings.get(team_home, base_rating)
        r_away = ratings.get(team_away, base_rating)

        # Guardar ELO antes del partido
        elo_home_list.append(r_home)
        elo_away_list.append(r_away)

        # Calcular resultado esperado
        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        expected_away = 1 - expected_home

        # Actualizar ELO después del partido
        ratings[team_home] = r_home + k * (result - expected_home)
        ratings[team_away] = r_away + k * ((1 - result) - expected_away)

    # Agregar columnas al DataFrame
    df['ELO_home'] = elo_home_list
    df['ELO_away'] = elo_away_list

    return df

def assign_elo_home_away(df, k=30, base_rating=1500):
    ratings_local = {}  # ELO como local
    ratings_away = {}  # ELO como visitante
    elo_home_list = []
    elo_away_list = []

    df = df.sort_values('date')  # Ordenar por fecha

    for idx, row in df.iterrows():
        team_home = row['id_team_home']
        team_away = row['id_team_away']
        result = row['result']

        r_home = ratings_local.get(team_home, base_rating)
        r_away = ratings_away.get(team_away, base_rating)

        # Guardar ELO antes del partido
        elo_home_list.append(r_home)
        elo_away_list.append(r_away)

        # Calcular resultado esperado
        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        expected_away = 1 - expected_home

        # Actualizar ELO según el resultado
        ratings_local[team_home] = r_home + k * (result - expected_home)
        ratings_away[team_away] = r_away + k * ((1 - result) - expected_away)

    # Agregar columnas al DataFrame
    df['ELO_de_local'] = elo_home_list
    df['ELO_de_vis'] = elo_away_list

    return df

def determine_number_matches_last_days(df: pd.DataFrame, n_days): # Ver si funciona
    """
    Determinar numero de partidos jugados en los ultimos dias. 
    
    Posibles mejoras:
        - Hacerlo por localia usando "segun_localia"
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='date', ascending=False).copy()  # Hacer copia para evitar modificaciones sobre el original
    l_teams = df['id_team_home'].unique()

    # Diccionario para acumular valores antes de asignarlos
    results_dict = {
        f'n_matches_last_{n_days}_days_home': [],
        f'n_matches_last_{n_days}_days_away': []
    }

    # Por team
    for team in l_teams:
        # Filtrar los partidos del equipo
        df_match_team = df[(df['id_team_home'] == team) | (df['id_team_away'] == team)]

        # Por match del team
        for idx, row in df_match_team.iterrows():
            home_or_away = 'home' if row['id_team_home'] == team else 'away'
            limit_date = row['date'] - timedelta(days=n_days)

            # Selecciono los últimos partidos del equipo dentro del rango de días
            df_match_team_filt = df_match_team.loc[
                (df_match_team['date'] >= limit_date) & (df_match_team['date'] < row['date'])
            ]
            n_games = len(df_match_team_filt)

            # Acumular valores en el diccionario
            results_dict[f'n_matches_last_{n_days}_days_{home_or_away}'].append((idx, n_games))

    # Convertir listas en Series y asignarlas de una vez
    for col, values in results_dict.items():
        df[col] = pd.Series(dict(values))  # Crea la columna usando un diccionario de índices

    # Calculo diferencia entre local y visitante
    df[f'dif_n_matches_last_{n_days}_days'] = (
        df[f'n_matches_last_{n_days}_days_home'] - df[f'n_matches_last_{n_days}_days_away']
    )

    # Elimino columnas temporales
    df = df.drop(columns=[f'n_matches_last_{n_days}_days_home', f'n_matches_last_{n_days}_days_away'])

    return df

def determine_number_results_last_matches(df: pd.DataFrame, n_matches, segun_localia: bool = False, var_resp: str = 'result'): # Ver si funciona
    """
    Determinar numero de triunfos, empates y derrotas en los ultimos n partidos por equipo.
    
    Posibles mejoras:
        - Hacerlo por localia usando "segun_localia"
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='date', ascending=True).copy()  # Hacer copia para evitar fragmentación
    pf = '' if var_resp == 'result' else 'x_'

    team_matches = {}
    # Construyo df por equipo
    for team in pd.concat([df['id_team_home'], df['id_team_away']]).unique():
        if segun_localia:
            team_matches[f"{team}_local"] = df[df['id_team_home'] == team]
            team_matches[f"{team}_visitante"] = df[df['id_team_away'] == team]
        else:
            team_matches[team] = df[(df['id_team_home'] == team) | (df['id_team_away'] == team)]
        
    # Diccionarios para acumular valores y evitar asignaciones repetitivas con `.at[]`
    results_dict = {
        f'{pf}n_wins_last_{n_matches}_matches_home': [],
        f'{pf}n_draws_last_{n_matches}_matches_home': [],
        f'{pf}n_loss_last_{n_matches}_matches_home': [],
        f'{pf}n_wins_last_{n_matches}_matches_away': [],
        f'{pf}n_draws_last_{n_matches}_matches_away': [],
        f'{pf}n_loss_last_{n_matches}_matches_away': []
    }

    # Por equipo
    for team_key, df_team in team_matches.items():
        team = team_key.split('_')[0] if segun_localia else team_key

        # Por partido
        for idx, row in df_team.iterrows():
            match_date = row['date']
            home_or_away = 'home' if row['id_team_home'] == team else 'away'

            # Filtrar últimos n partidos antes del actual
            df_last_matches = df_team[df_team['date'] < match_date].tail(n_matches)
            
            # Eliminar registros con NaN en la columna "expected_result"
            if var_resp == 'expected_result':
                df_last_matches = df_last_matches.dropna(subset=['expected_result'])

            n_games = len(df_last_matches)

            # Obtener los valores de la variable considerando si fue home o away
            df_match_team_filt_home = df_last_matches[df_last_matches['id_team_home'] == team]
            df_match_team_filt_away = df_last_matches[df_last_matches['id_team_away'] == team]

            if n_games != (len(df_match_team_filt_home) + len(df_match_team_filt_away)):
                logger.error(f"Error en filtrado de partidos en la construcción... {len(df_match_team_filt_home)} + {len(df_match_team_filt_away)} != {len(df_last_matches)}")
                raise ValueError

            # Construyo variables
            n_wins = len(df_match_team_filt_home[df_match_team_filt_home[var_resp] == 1]) + len(df_match_team_filt_away[df_match_team_filt_away[var_resp] == 2])
            n_draws = len(df_last_matches[df_last_matches[var_resp] == 0])
            n_loss = len(df_match_team_filt_home[df_match_team_filt_home[var_resp] == 2]) + len(df_match_team_filt_away[df_match_team_filt_away[var_resp] == 1])

            if n_games == (n_wins + n_draws + n_loss):
                # Acumular valores en listas
                results_dict[f'{pf}n_wins_last_{n_matches}_matches_{home_or_away}'].append((idx, n_wins))
                results_dict[f'{pf}n_draws_last_{n_matches}_matches_{home_or_away}'].append((idx, n_draws))
                results_dict[f'{pf}n_loss_last_{n_matches}_matches_{home_or_away}'].append((idx, n_loss))
                
            else:
                logger.error(f"Error en determinación de resultados en últimos días {n_wins} + {n_draws} + {n_loss} != {n_games}")
                raise ValueError # en el caso de expected_result este puede ser nan... y por ende fallar --> ahora elimino partidos con expected_result none.

    # Convertir listas en Series y asignarlas de una vez para evitar fragmentación
    for col, values in results_dict.items():
        df[col] = pd.Series(dict(values))  # Crea la columna usando un diccionario de índices

    for elem in ['wins', 'draws', 'loss']:
        dif_col = f'dif_{pf}n_{elem}_last_{n_matches}_matches'
        col1 = f'{pf}n_{elem}_last_{n_matches}_matches_home'
        col2 = f'{pf}n_{elem}_last_{n_matches}_matches_away'
        
        df[dif_col] = df[col1] - df[col2] 
        df.drop(columns=[col1, col2], inplace=True) # eliminar columnas col1 o col2 o dejar que las elimine si asi fuera necesario el feature_selection? --> al parecer no tienen mucha impor

    return df

## Rendimiento del equipo
def determine_points(df: pd.DataFrame, suffix: str = ''):
    """
    Determina los points obtenidos por cada team según el resultado del juego.

    :param df: DataFrame con la columna 'expected_result'.
    :return: DataFrame con 'expected_points_home' y 'expected_points_away'.
    """
    # Inicializo las columnas "expected_points_home" y "expected_points_away"
    col1, col2, col3 = f'{suffix}points_home', f'{suffix}points_away', f'{suffix}result'
    df[col1] = 0
    df[col2] = 0

    df.loc[df[col3] == 1, [col1, col2]] = [3, 0]
    df.loc[df[col3] == 0, [col1, col2]] = [1, 1]
    df.loc[df[col3] == 2, [col1, col2]] = [0, 3]

    # Asignar NaN donde expected_result es NaN
    df.loc[df[col3].isna(), [col1, col2]] = np.nan

    return df

## Teams
def h2h_by_date(df: pd.DataFrame, n_years: int, prod: bool = False, idxs_to_construct: list = None, _print: bool = False):
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    prod: Permite construir todos los equipos o solo los mas frecuentes. En prod necesito todos. En train los mas frecuentes (x tiempo). (bool)
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    print("\n Constructing Head to Head...")
    # Definicion de variables
    start = time.time()

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo
    n_days =  365 * n_years
    h2h_col_name = f'h2h_{n_years}'
    print(f"Historial a construir: {h2h_col_name} para {n_years}")

    if prod:
        df_next_matches = df[df.index.isin(idxs_to_construct)]
        l_equipos = list(set(df_next_matches['id_team_home']).union(set(df_next_matches['id_team_away'])))
    else:
        idxs_to_construct = df.index
        l_equipos = determine_most_frequent_teams(df)

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

                if idx in idxs_to_construct:

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

def determine_most_frequent_teams(df, _print: bool = False):
    # Determinar equipos a calcular el historial
    frecuencias = df['id_team_home'].value_counts()
    threshold = int(0.001 * len(df))
    l_equipos = frecuencias[frecuencias > threshold].index.tolist()
    if _print:
        print(f"Integer {threshold} ; Cantidad de equipos: {len(l_equipos)}")
    return l_equipos

def h2h_by_date_by_localia(df: pd.DataFrame, n_years: int, prod: bool = False, idxs_to_construct: list = None, _print: bool = False):
    """
    Determina el h2h entre los equipos que disputan el match según los resultados en los últimos matchs entre ellos.

    :param df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
    :param n_years: Integer. Número de años a tener en cuenta para determine el h2h entre dos equipos.
    prod: Permite construir todos los equipos o solo los mas frecuentes. En prod necesito todos. En train los mas frecuentes (x tiempo). (bool)
    :return: DataFrame pasado por parámetro con nueva columna, 'h2h_date', que permite determine a cuál de los dos
    equipos de un match le favorece más el h2h entre ellos.
    """
    print("\n Constructing Head to Head...")
    # Definicion de variables
    start = time.time()

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='date', ascending=False)  # Mas reciente a mas antiguo
    n_days =  365 * n_years
    h2h_col_name = f'h2h_{n_years}_segun_loc'
    print(f"Historial a construir: {h2h_col_name} para {n_years}")

    if prod:
        df_next_matches = df[df.index.isin(idxs_to_construct)]
        l_equipos = list(set(df_next_matches['id_team_home']).union(set(df_next_matches['id_team_away'])))
    else:
        idxs_to_construct = df.index
        l_equipos = determine_most_frequent_teams(df)

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

                    if idx in idxs_to_construct:

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
    keywords_prohibidas = ['_player_', 'team_', 'odds_', 'coach_']  # Definir palabras clave prohibidas. 
    
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
    
    # Verificar y convertir las columnas '_home' a tipo numérico
    if home_columns:
        df[home_columns] = df[home_columns].apply(pd.to_numeric, errors='coerce')
    
    # Verificar y convertir las columnas '_away' a tipo numérico
    if away_columns:
        df[away_columns] = df[away_columns].apply(pd.to_numeric, errors='coerce')
    
    # Sumar las columnas con sufijo '_home'
    df[f'{column_name}_home'] = df[home_columns].sum(axis=1) if home_columns else None
    
    # Sumar las columnas con sufijo '_away'
    df[f'{column_name}_away'] = df[away_columns].sum(axis=1) if away_columns else None
    
    return df

def construct_percentaje_column(df: pd.DataFrame, col_num: str, col_den: str, column_name: str = None, laplace: bool = False):
    """
    Nueva columna que representa el porcentaje resultante de la división de dos columnas.
    """

    col_name = f'perc_{col_num}_of_{col_den}' if column_name is None else column_name

    num_home = df[f'{col_num}_home'] + 1 if laplace else df[f'{col_num}_home']
    num_away = df[f'{col_num}_away'] + 1 if laplace else df[f'{col_num}_away']

    den_home = df[f'{col_den}_home'] + 1 if laplace else df[f'{col_den}_home']
    den_away = df[f'{col_den}_away'] + 1 if laplace else df[f'{col_den}_away']

    # Evitar división por 0
    den_home = np.where(den_home == 0, 1, den_home)
    den_away = np.where(den_away == 0, 1, den_away)

    # Home
    df[f'{col_name}_home'] = num_home / den_home

    # Away
    df[f'{col_name}_away'] = num_away / den_away

    return df

def determine_mean_last_matches_difference(df, n_days, variable, segun_localia, decay_rate: float = 0.1, diff: bool = True):
    """
    Calcula la media en los ultimos partidos a partir de una columna de diferencias ("dif_") (e.g. dif goals). 
    Usa diferencia previa antes del promedio.
    
    # Parameters:
        df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
        n_days: Integer. Número de días a tener en cuenta para determinar la media.
        variable: String. Nombre de la variable a promediar (e.g. goals).
        segun_localia: Boolean. Si True, calcula la media según localía.
        decay_rate: Float. Tasa de decaimiento para el promedio ponderado.
        diff: Boolean. Si True, calcula dif entre home y away del promedio en last matches.

    # Returns:
        DataFrame con nuevas columnas que contienen la media de los últimos partidos.
    """
    df = df.sort_values(by='date', ascending=False)
    team_matches = {}
    results = {}

    # Construyo df por equipo
    for team in pd.concat([df['id_team_home'], df['id_team_away']]).unique():
        if segun_localia:
            team_matches[f"{team}_local"] = df[df['id_team_home'] == team]
            team_matches[f"{team}_visitante"] = df[df['id_team_away'] == team]
            name_ext = "loc_"
        else:
            team_matches[team] = df[(df['id_team_home'] == team) | (df['id_team_away'] == team)]
            name_ext = ""

    # Por equipo
    for team_key, df_team in team_matches.items():

        team = team_key.split('_')[0] if segun_localia else team_key

        # Por partido
        for idx, row in df_team.iterrows():
            match_date = row['date']
            home_or_away = 'home' if row['id_team_home'] == team else 'away'

            # Filtrar por fecha
            limit_date = match_date - timedelta(days=n_days)
            df_last_matches = df_team[(df_team['date'] >= limit_date) & (df_team['date'] < match_date)] 

            # Obtengo valores de la variable.
            s_values = df_last_matches[variable]  # Obtiene los valores de la variable

            # Convertir a numérico y eliminar NaN
            s_values = pd.to_numeric(s_values, errors="coerce").dropna()
            df_last_matches = df_last_matches.loc[s_values.index]  # <-- alineación necesaria con s_values

            # En los visitantes, invierto la diferencia pues es positiva para el visitante.
            s_values = np.where(df_last_matches['id_team_away'] == team, -s_values, s_values)
            s_values = pd.Series(s_values, index=df_last_matches.index)

            # Calcular promedio ponderado (priorizando registros recientes)
            if not s_values.empty:
                weights = np.exp(-decay_rate * np.arange(len(s_values)))  # Pesos exponenciales
                weights /= weights.sum()  # Normalizar pesos
                weighted_mean = np.dot(s_values, weights)  # Promedio ponderado

                # Guardar resultado en el diccionario
                results.setdefault(idx, {})[f"{name_ext}mean_last_{n_days}_days_{variable}_{home_or_away}"] = weighted_mean
            
    # Convertir a DataFrame y hacer join con el original
    if results:
        df_update = pd.DataFrame.from_dict(results, orient="index")
        df = df.join(df_update)
    
    if diff:
        dif_col = f'dif_{name_ext}mean_last_{n_days}_days_{variable}'
        col1 = f'{name_ext}mean_last_{n_days}_days_{variable}_home'
        col2 = f'{name_ext}mean_last_{n_days}_days_{variable}_away'
        
        df[dif_col] = df[col1] - df[col2] 
        df.drop(columns=[col1, col2], inplace=True) # eliminar columnas col1 o col2 o dejar que las elimine si asi fuera necesario el feature_selection? --> al parecer no tienen mucha impor

    return df

def determine_mean_last_matches_home_away(df: pd.DataFrame, n_days: int, variable: str, segun_localia: bool, decay_rate: float = 0.1, diff: bool = True): 
    """
    Calcula la media en los ultimos partidos a partir de valores separados en columnas "home" y "away" (e.g. goals_home y goals_away). 
    No usa diferencia previa.

    # Parameters:
        df: DataFrame. Unidad de análisis: match. Columnas: al menos fecha, id_team_home, id_team_away y result.
        n_days: Integer. Número de días a tener en cuenta para determinar la media.
        variable: String. Nombre de la variable a promediar (e.g. goals).
        segun_localia: Boolean. Si True, calcula la media según localía.
        decay_rate: Float. Tasa de decaimiento para el promedio ponderado.
        diff: Boolean. Si True, calcula dif entre home y away del promedio en last matches.

    # Returns:
        DataFrame con nuevas columnas que contienen la media de los últimos partidos.
    """
   # Ordeno por fecha descendente para iterar correctamente sobre los partidos
    df = df.sort_values(by='date', ascending=False)
    d_teams = {'id_team_home': 'home', 'id_team_away': 'away'}
    results = {}

    # Inicializo diccionarios (para evitar Performance Warning)
    name_ext = "loc_" if segun_localia else ""

    # Por partido
    for id_match, row in df.iterrows():
        
        # Obtener los últimos partidos antes de la fecha actual
        match_date = row['date']
        limit_date = match_date - timedelta(days=n_days)
        df_past_matches = df[(df['date'] < match_date) & (df['date'] >= limit_date)]

        # Por equipo
        for col_team, home_or_away in d_teams.items():

            variable_form = f'{variable}_{home_or_away}'
            team = row[col_team]

            if segun_localia:
                # Selecciono los últimos n partidos del equipo en esa localía
                df_team_matches = df_past_matches.loc[df_past_matches[col_team] == team]
                values = df_team_matches[variable_form]

            else:
                # Selecciono los últimos n partidos del equipo en cualquier localía
                df_team_matches = df_past_matches.loc[(df_past_matches["id_team_home"] == team) | (df_past_matches["id_team_away"] == team)]

                # Extraer los valores de la variable correspondiente dependiendo de si el equipo fue local o visitante
                values = np.where(
                    df_team_matches["id_team_home"] == team,  
                    df_team_matches[f"{variable}_home"],  
                    df_team_matches[f"{variable}_away"]
                )
                values = pd.Series(values, index=df_team_matches.index)

            # Convertir a numérico y eliminar NaN
            values = pd.to_numeric(values, errors="coerce").dropna()

            # Calcular los pesos exponenciales solo si hay datos
            if not values.empty:
                weights = np.exp(-decay_rate * np.arange(len(values)))
                weights /= weights.sum()
                weighted_mean = np.dot(values, weights)

                results.setdefault(id_match, {})[f"{name_ext}mean_last_{n_days}_days_{variable_form}"] = weighted_mean

    # Convertir los diccionarios a DataFrames y actualizar el original
    if results:
        df_update = pd.DataFrame.from_dict(results, orient="index")
        df = df.join(df_update)  # Mucho más eficiente que usar df.loc en cada iteración

    if diff:
        dif_col = f'dif_{name_ext}mean_last_{n_days}_days_{variable}'
        col1 = f'{name_ext}mean_last_{n_days}_days_{variable}_home'
        col2 = f'{name_ext}mean_last_{n_days}_days_{variable}_away'
        
        df[dif_col] = df[col1] - df[col2] 
        df.drop(columns=[col1, col2], inplace=True) # eliminar columnas col1 o col2 o dejar que las elimine si asi fuera necesario el feature_selection? --> al parecer no tienen mucha impor

    return df

def determine_mean_last_matches_home_away_against(df: pd.DataFrame, n_days: int, variable: str, segun_localia: bool, decay_rate: float = 0.1, diff: bool = True): 
    """
    Calcula la media en los ultimos partidos a partir de valores separados en columnas "home" y "away" (e.g. goals_home y goals_away). No usa diferencia previa.

    Mejoras:
        - Revisar calculo de dif_con_against. Lo implemente rapido mirando como lo tenia antes. Pero por las. 
    """
   # Ordeno por fecha descendente para iterar correctamente sobre los partidos
    df = df.sort_values(by='date', ascending=False)
    d_teams = {'id_team_home': 'home', 'id_team_away': 'away'}
    results_against = {}

    # Inicializo diccionarios (para evitar Performance Warning)
    name_ext = "loc_" if segun_localia else ""

    # Por partido
    for id_match, row in df.iterrows():
        
        # Obtener los últimos partidos antes de la fecha actual
        match_date = row['date']
        limit_date = match_date - timedelta(days=n_days)
        df_past_matches = df[(df['date'] < match_date) & (df['date'] >= limit_date)]

        # Por equipo
        for col_team, home_or_away in d_teams.items():

            home_or_away_against =  f'away' if home_or_away == 'home' else f'home'
            variable_form = f'{variable}_{home_or_away}'
            variable_against = f'{variable}_{home_or_away_against}' 
            team = row[col_team]

            if segun_localia:
                # Selecciono los últimos n partidos del equipo en esa localía
                df_team_matches = df_past_matches.loc[df_past_matches[col_team] == team]
                values_against = df_team_matches[variable_against]

            else:
                # Selecciono los últimos n partidos del equipo en cualquier localía
                df_team_matches = df_past_matches.loc[(df_past_matches["id_team_home"] == team) | (df_past_matches["id_team_away"] == team)]

                values_against = np.where(
                    df_team_matches["id_team_home"] == team,  
                    df_team_matches[f"{variable}_away"],  
                    df_team_matches[f"{variable}_home"]
                )
                values_against = pd.Series(values_against, index=df_team_matches.index)

            # Convertir a numérico y eliminar NaN
            values_against = pd.to_numeric(values_against, errors="coerce").dropna()

            if not values_against.empty:
                weights_ag = np.exp(-decay_rate * np.arange(len(values_against)))
                weights_ag /= weights_ag.sum()
                weighted_mean_ag = np.dot(values_against, weights_ag)

                # Guardar media en el diccionario de resultados
                results_against.setdefault(id_match, {})[f"{name_ext}mean_last_{n_days}_days_against_{variable_form}"] = weighted_mean_ag

    if results_against:
        df_update_against = pd.DataFrame.from_dict(results_against, orient="index")
        df = df.join(df_update_against)

    if diff:
        dif_col = f'dif_{name_ext}mean_last_{n_days}_days_against_{variable}'
        col1 = f'{name_ext}mean_last_{n_days}_days_against_{variable}_home'
        col2 = f'{name_ext}mean_last_{n_days}_days_against_{variable}_away'
        
        df[dif_col] = df[col1] - df[col2] 
        df.drop(columns=[col1, col2], inplace=True) # eliminar columnas col1 o col2 o dejar que las elimine si asi fuera necesario el feature_selection? --> al parecer no tienen mucha impor

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
                logger.warning("Falló el calculo de diferencia entre local y visitante")
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
    # df = determine_mean_last_matches_difference(df, n_days, variable='mean_rat_player_start', segun_localia=segun_localia)
    # df = df.drop(columns=['mean_last_match_mean_rat_player_start_home', 'mean_last_match_mean_rat_player_start_away'], axis=1)
    # df = calculate_dif_col_players(df)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel(f'{BASE_DIR_LOCAL}/df_constructed_prueba.xlsx')