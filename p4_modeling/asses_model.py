import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score
from p3_data_preparation import construct_data


# CALCULO DE METRICAS BASICAS (ACCURACY, F1_SCORE, ETC)
def calculate_metrics(
        df_pred_proba, var_pred: str = 'predicted_result', var_resp: str = 'result',
        metrics_by_result: bool = True, prefix: str = None, suffix: str = None, verbose: int = 0
        ):
    """
    Calculo de metricas que no requieren mas que y_pred e y_test
    """
    # Defino los array para calc metrics
    y_test = df_pred_proba[var_resp].values         # Etiquetas reales
    y_pred = df_pred_proba[var_pred].values         # Predicciones del modelo

    # Calculo métricas básicas
    d_metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred) * 100,
        'recall': recall_score(y_test, y_pred, average='macro') * 100,
        'f1_score': f1_score(y_test, y_pred, average='macro') * 100,
    }

    if metrics_by_result:
        d_metrics.update ({
            'accuracy_home': accuracy_score(y_test == 1, y_pred == 1) * 100,  # Accuracy para la clase "Local"
            'accuracy_draw': accuracy_score(y_test == 0, y_pred == 0) * 100,  # Accuracy para la clase "Empate"
            'accuracy_away': accuracy_score(y_test == 2, y_pred == 2) * 100,   # Accuracy para la clase "Visitante"
            'f1_score_home': f1_score(y_test, y_pred, labels=[1], average='macro', zero_division=0) * 100,
            'f1_score_draw': f1_score(y_test, y_pred, labels=[0], average='macro', zero_division=0) * 100,
            'f1_score_away': f1_score(y_test, y_pred, labels=[2], average='macro', zero_division=0) * 100
        })

    # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
    if verbose >= 1:
        df_conf_mat = confusion_matrix(y_test, y_pred)
        # df_conf_mat.to_excel(f'/data/{country}/p4_modeling/modeling/df_conf_matrix.xlsx')

    d_metrics.update(determine_distribution(df_pred_proba, var_resp=var_resp))

    if prefix or suffix:
        d_metrics = rename_dict_keys(d_metrics)

    if verbose > 1:
        print(d_metrics)

    return d_metrics

def confusion_matrix(y_real, y_pred):
    """
    Calcula matriz de confusion del modelo.

    # Parameters:
        y_real: Resultados reales de los partidos. (numpy.ndarray)
        y_pred: Resultados preddichos de los partidos segun modelo. (numpy.ndarray)

    # Returns
        Matriz de confusion con la prediccion del modelo en el eje x y el resultado real en el eje y. (DataFrame)
    """
    # Calculo los numeros para la matriz de confusion
    confusion_matrix = metrics.confusion_matrix(y_real, y_pred)  # numpy.ndarray

    # Convertir el array a un DataFrame de pandas
    df_cm = pd.DataFrame(confusion_matrix)
    df_cm.index.name = "Resultado real"
    print(f"\n\nMatriz de confusion:\n {df_cm}")
    return df_cm

# Bookies
def calculate_bookie_metrics(df_pred_proba, var_resp: str = 'result', var_pred_bm: str = 'bookmaker_result'):
    """
    Requiere df_match_odds
    """
    # Defino los array para calc metrics
    y_test = df_pred_proba[var_resp].values         # Etiquetas reales
    y_pred_bm = df_pred_proba[var_pred_bm].values   # Predicciones del BET

    # Calculo metricas de bookie
    test_accuracy_bm = accuracy_score(y_test, y_pred_bm) * 100  # Calcula bien tras el reindex()
    d_metrics = {
        'test_accuracy_bm': test_accuracy_bm,
        }
    return d_metrics

def determine_result_by_bookmaker(df, col_name):
    """
    Determina el resultado del partido predicho segun la casa de apuestas. 

    # Paramaters:
        df: Dataframe con cuotas de la casa de apuestas. (DataFrame)
        col_name: Nombre de la nueva variable con el resultado predicho segun la casa de apuestas. (str)

    # Returns
        Dataframe pasado como parametro con nueva columna con el resultado predicho segun la casa de apuestas. (DataFrame)
    """
    class_home, class_draw, class_away = 1, 0, 2

    # Por partido
    for id_match, row in df.iterrows():

        # Determino la cuota minima de las 3 posibles
        odds_min = min(row['odds_home'], row['odds_draw'], row['odds_away'])

        # Determino resultado predicho segun casa de apuestas (el de la cuota minima) y lo guardo
        result_pred_bm = class_home if row['odds_home'] == odds_min else (class_away if row['odds_away'] == odds_min else class_draw)
        df.loc[id_match, col_name] = result_pred_bm

    return df

def calculate_result_probabilities_by_bookmaker(df_match_odds):
    """
    Calcula las probabilidades de cada resultado (Home, Draw y Away) segun la casa de apuestas.

    # Paramaters:
        df_match_odds: Dataframe con partidos y las respectivas cuotas de la casa de apuestas. (DataFrame)

    # Returns
        Dataframe pasado como parametro con cuatro nuevas columnas: la probabilidad de cada resultado (Home, Draw y Away) segun la casa de apuestas y el overround. (DataFrame)
    """
    df_match_odds = df_match_odds.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])

    # Por partido
    for idx, row in df_match_odds.iterrows():

        odds_home = float(row['odds_home'])
        odds_draw = float(row['odds_draw'])
        odds_away = float(row['odds_away'])

        # Calcular probabilidades a partir de invertir las cuotas
        prob_home_with_over = 1 / odds_home
        prob_draw_with_over = 1 / odds_draw
        prob_away_with_over = 1 / odds_away

        # Sumo las probabilidades (deberia ser >1 por el margen de ganancia de la casa de apuesta)
        sum_prob_with_over = prob_home_with_over + prob_draw_with_over + prob_away_with_over # Si no habria overround seria 100%
        overround = sum_prob_with_over - 1

        # Calculo probabilidades sin el margen
        prob_home = prob_home_with_over / sum_prob_with_over
        prob_draw = prob_draw_with_over / sum_prob_with_over
        prob_away = prob_away_with_over / sum_prob_with_over

        df_match_odds.loc[idx, ['prob_home_bm', 'prob_draw_bm', 'prob_away_bm', 'overround']] = [prob_home, prob_draw, prob_away, overround]

    return df_match_odds

# Distribucion de resultados
def determine_confidence_margin(df):
    """
    Calcula el confidence_margin como la diferencia entre la probabilidad más alta 
    y la segunda más alta. 

    Parameters:
        df (pd.DataFrame): DataFrame con las columnas de probabilidades.

    Returns:
        pd.DataFrame: DataFrame con confidence_margin calculado y stake ajustado.
    """
    
    # Obtener las dos probabilidades más altas por fila
    probs_sorted = np.sort(df[['prob_class_0', 'prob_class_1', 'prob_class_2']], axis=1)
    
    # Calcular confidence_margin
    df['confidence_margin'] = probs_sorted[:, -1] - probs_sorted[:, -2]
    return df

def determine_distribution(df, var_resp: str = 'result', var_pred: str = 'predicted_result'):
    """
    Determina la cantidad de predicciones por resultado y las compara con la distribucion de resultados reales.

    # Parameters
        df: Nuestas predicciones y el resultado real. (DataFrame)

    # Return
        d: Diccionario con la distribucion de nuestras predicciones, las de los resultados reales y la variacion.
    """
    # Calculo distribucion de nuestros resultados
    n_loc, n_emp, n_vis = count_results(df, col=var_pred)
    n_loc_r, n_emp_r, n_vis_r = count_results(df, col=var_resp)

    # Calculo variacion por resultado
    var_loc = calculate_variation(end=n_loc, ini=n_loc_r)
    var_emp = calculate_variation(end=n_emp, ini=n_emp_r)
    var_vis = calculate_variation(end=n_vis, ini=n_vis_r)

    # Calcular variaciones absolutas
    var_loc_abs = abs(var_loc)
    var_emp_abs = abs(var_emp)
    var_vis_abs = abs(var_vis)

    # Promedio de variaciones absolutas
    var = (var_loc_abs + var_emp_abs + var_vis_abs) / 3

    # Crear el diccionario con los resultados
    d = {
        'n_loc': n_loc, 'n_emp': n_emp, 'n_vis': n_vis,
        'n_loc_r': n_loc_r, 'n_emp_r': n_emp_r, 'n_vis_r': n_vis_r,
        'dif_loc': var_loc, 'dif_emp': var_emp, 'dif_vis': var_vis, '%_dif': var
    }
    return d

def count_results(df, col):
    """
    Cuenta la cantidad de cada resultado.
    """
    n_local = len(df[df[col] == 1])
    n_empate = len(df[df[col] == 0])
    n_vis = len(df[df[col] == 2])
    return n_local, n_empate, n_vis

def calculate_variation(end, ini):
    return (end - ini) / abs(ini)

def rename_dict_keys(d, prefix=None, suffix=None):
    return {f"{prefix or ''}{k}{suffix if suffix else ''}": v for k, v in d.items()}

# CALCULO DE METRICAS DE ROI
def calculate_roi_metrics(df_pred_proba, 
                      var_pred: str = 'predicted_result', var_resp: str = 'result', var_pred_bm: str = 'bookmaker_result', 
                      roi_metrics: bool = True, advanced_metrics: bool = False, 
                      prefix: str = None, suffix: str = None,
                      verbose: int = 0
                      ):
    """
    Calculo de metricas
    """
    # roi = determine_roi(df_pred_proba, var_resp=var_resp) # hago el rename con expected por fuera de la funcion...
    # d_metrics.update({'roi': roi})

    # Otras métricas
    if advanced_metrics:
        # d_metrics.update(calculate_nan_metrics(df_pred_proba))
        d_metrics.update(calculate_gp_by_result(df_pred_proba, var_resp=var_resp))

    if prefix or suffix:
        d_metrics = rename_dict_keys(d_metrics)

    if verbose > 1:
        print(d_metrics)

    return d_metrics

def calculate_roi(df: pd.DataFrame, name_extension=''):
    """
    Calcula ROI obtenido segun las predicciones del modelo y el resultado real de los partidos. Para poder seleccionar el mejor modelo.

    # Parameters:
        df: Dataframe (DataFrame)
        stake_base: Stake base sobre el cual aplicar el multiplicador para obtener el stake variable. (int)
        _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

    # Returns
        ROI del modelo. (float)
    """
    # Converito date a datetime y ordeno por fecha
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
    df = df.sort_values(by='date', ascending=True)  # Mas antiguo a mas reciente
     
    # Definicion de variables
    bank_inicial = 100 # CUIDADO! NO ES sum(df['stake_mod'])
    bank_final = bank_inicial
    n_apuestas = len(df)
    d_rois = {}
    cont = 0

    # Por partido
    for idx, row in df.iterrows():
        cont += 1

        # Defino stake a apostar en pesos (a partir del stake como porcentaje del bank)
        stake_a_apostar = bank_final * row['stake_to_bet'] / 100
        df.loc[idx, f'{name_extension}bank_inicial'] = bank_final
        df.loc[idx, f'{name_extension}stake_to_bet_en_$'] = stake_a_apostar

        # Determino ganancias / perdidas 
        ingresos = stake_a_apostar * row['odd_to_bet'] if row[f'{name_extension}acerte'] == 1 else 0
        ganancia = ingresos - stake_a_apostar
        bank_final += ganancia
        df.loc[idx, f'{name_extension}G/P'] = ganancia
        df.loc[idx, f'{name_extension}bank_final'] = bank_final

        # Determino ganancias / perdidas (sin bank)
        ingresos_sin_bank = row['stake_to_bet'] * row['odd_to_bet'] if row[f'{name_extension}acerte'] == 1 else 0
        ganancia_sin_bank = ingresos_sin_bank - row['stake_to_bet']
        df.loc[idx, f'{name_extension}G/P_sin_bank'] = ganancia_sin_bank

        # Raise error si perdi todo el dinero de las apuestas
        if bank_final <= 0:
            logger.warning("El dinero tras apuestas se hizo negativo y esto no es posible puesto que el stake siempre es un % del bank.")
            break

    # Calculo el ROI
    roi = (bank_final - bank_inicial) / bank_inicial * 100
    roi_por_partido = roi / n_apuestas  # No quiero el ROI mas alto sino el ROI / partido mas alto
    d_rois[f'{name_extension}roi'] = roi
    d_rois[f'{name_extension}roi_por_partido'] = roi_por_partido
    return df, d_rois

def calculate_reality_roi(df: pd.DataFrame):
    """
    Calcula ROI simulando la forma de apostar de la realidad, de manera de saber que ROI esperar en la realidad.

    # Parameters:
        df: Dataframe (DataFrame)
        stake_base: Stake base sobre el cual aplicar el multiplicador para obtener el stake variable. (int)
        _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

    # Returns
        ROI del modelo. (float)
    """
    # Definicion de variables
    bank_inicial = 100 # CUIDADO! NO ES sum(df['stake_mod'])
    bank_final = bank_inicial
    n_apuestas = len(df)
    d_rois = {}
    l_rois_partido = [50, 100, 150, 200, 250, 300, 400, 500]
    cont = 0
    exit_loops = False
    # df_correct = df[df['acerte'] == 1]   # Filtrar el dataframe solo a las filas donde el modelo predijo correctamente

    # Obtengo dias unicos de partidos
    df['date_only'] = pd.to_datetime(df['date']).dt.date # Suponiendo que tienes un DataFrame con una columna de fecha y hora llamada 'date'
    unique_dates = df['date_only'].unique()  # Luego puedes obtener las fechas únicas sin la hora

    # Por fecha
    for date in unique_dates:

        if exit_loops:
            break

        # Filtro partidos por fecha
        df_date = df[df['date_only'] == date]
        
        # Determino el bank inicial
        bank_inicial_dia = bank_final
        sum_stakes_dia = 0

        # Por partido
        for idx, row in df_date.iterrows():
            cont += 1
            sum_stakes_dia += row['stake_to_bet']

            # Defino stake a apostar en pesos (a partir del stake como porcentaje del bank)
            stake_a_apostar =  bank_inicial_dia * row['stake_to_bet'] / 100
            df.loc[idx, 'bank_inicial_r'] = bank_inicial_dia
            df.loc[idx, 'stake_to_bet_en_$_r'] = stake_a_apostar

            # Determino ganancias / perdidas 
            ingresos = stake_a_apostar * row['odd_to_bet'] if row['acerte'] == 1 else 0
            ganancia = ingresos - stake_a_apostar
            bank_final += ganancia
            df.loc[idx, 'G/P_r'] = ganancia
            df.loc[idx, 'bank_final_r'] = bank_final

            # Guardo ROI en partidos especificados
            if cont in l_rois_partido:
                roi_partido = (bank_final - bank_inicial) / bank_inicial * 100
                d_rois[f'roi_{cont}_r'] = roi_partido / cont
                df.loc[idx, 'roi_partido_r'] = roi_partido / cont

            # Raise error si perdi todo el dinero de las apuestas
            if bank_final <= 0:
                logger.warning("El dinero tras apuestas se hizo negativo y esto no es posible puesto que el stake siempre es un % del bank.")
                exit_loops = True
                break
            
            # Raise error si apuesta mas del 100% del bank en un dia dado
            if sum_stakes_dia > 100:
                bank_final = -100
                exit_loops = True
                logger.warning("El stake to bet superó el 100% del bank para un mismo dia. ")
                break            
            
    # Calculo el ROI
    if not exit_loops:
        roi = (bank_final - bank_inicial) / bank_inicial * 100
        roi_por_partido = roi / n_apuestas  # No quiero el ROI mas alto sino el ROI / partido mas alto
        d_rois['roi_r'] = roi
        d_rois['roi_por_partido_r'] = roi_por_partido
        # cuota_media_ganada = sum(df_correct['odd_to_bet'] / len(df_correct))
        # cuota_media_apostada = sum(df['odd_to_bet'] / len(df))  # --> no tengo la cuota apostada en cada partido... la eestoy calculando en el ciclo for...
    else:
        d_rois['roi_por_partido_r'] = -100

    return df, d_rois

def determine_roi(df_pred, var_resp):

    if var_resp == 'result':
        col_inic, col_fin = 'bank_inicial', 'bank_final'
    elif var_resp == 'expected_result':
        col_inic, col_fin = 'expected_bank_inicial', 'expected_bank_final'
    else:
        logger.error("No se puede determinar el roi porque no se encontró la variable predicha.")

    # Tomar el primer valor de la columna
    bank_inicial = df_pred[col_inic].iloc[0]

    # Tomar el último valor de la columna
    bank_final = df_pred[col_fin].iloc[-1]

    return (bank_final - bank_inicial) / bank_inicial

def calculate_nan_metrics(df_predicciones):
    """
    Calcula las métricas relacionadas con el relleno de NaN en el DataFrame.
    """
    # Cálculo del promedio de columnas rellenadas
    if 'n_col_filled' in df_predicciones.columns:
        average_col_filled = df_predicciones['n_col_filled'].sum() / len(df_predicciones)

    # Filtrar registros con y sin relleno de NaN
    if 'player_emergency_fill' in df_predicciones.columns:
        rows_player_filled = df_predicciones[df_predicciones['player_emergency_fill'] == 1].index
        rows_player_not_filled = df_predicciones[df_predicciones['player_emergency_fill'] != 1].index

        # G/P por estado de relleno de NaN
        gp_filled = df_predicciones.loc[rows_player_filled, 'G/P_sin_bank'].sum()
        gp_not_filled = df_predicciones.loc[rows_player_not_filled, 'G/P_sin_bank'].sum()
        gp_total = df_predicciones['G/P_sin_bank'].sum()

        perc_gp_filled = calculate_perc_gp(gp_filled, gp_total)
        perc_gp_not_filled = calculate_perc_gp(gp_not_filled, gp_total)

    else:
        average_col_filled = -1
        rows_player_filled = []
        gp_filled, gp_not_filled = 0, df_predicciones['G/P_sin_bank'].sum()
        perc_gp_filled, perc_gp_not_filled = 0, 1
        
    d = {
        'average_col_filled': average_col_filled,
        'n_emer_player_filled': len(rows_player_filled),
        'gp_filled': gp_filled, 
        'gp_not_filled': gp_not_filled,
        '%_gp_filled': perc_gp_filled,
        '%_gp_not_filled': perc_gp_not_filled,
    }
    return d

def calculate_gp_by_result(df_predicciones, var_resp):
    """
    Calcula el G/P por resultado (local, empate, visitante).
    """
    if var_resp == 'result':
        col_gp = 'G/P_sin_bank'
    elif var_resp == 'expected_result':
        col_gp = 'expected_G/P_sin_bank'

    # Dividir DataFrame por tipo de resultado
    df_pred_home = df_predicciones[df_predicciones[var_resp] == 1]
    df_pred_draw = df_predicciones[df_predicciones[var_resp] == 0]
    df_pred_away = df_predicciones[df_predicciones[var_resp] == 2]
    
    # Calcular G/P por resultado
    gp_home = df_pred_home[col_gp].sum()
    gp_draw = df_pred_draw[col_gp].sum()
    gp_away = df_pred_away[col_gp].sum()
    gp_total = gp_home + gp_draw + gp_away

    perc_gp_home = calculate_perc_gp(gp_home, gp_total)
    perc_gp_draw = calculate_perc_gp(gp_draw, gp_total)
    perc_gp_away = calculate_perc_gp(gp_away, gp_total)

    # Crear el diccionario de resultados
    d = {
        # Resultados por tipo
        'gp_home': gp_home,
        'gp_draw': gp_draw,
        'gp_away': gp_away,
        'gp_total': gp_total,
        '%_gp_home': perc_gp_home,
        '%_gp_draw': perc_gp_draw,
        '%_gp_away': perc_gp_away,
    }
    return d

def calculate_perc_gp(gp, gp_total):
    if gp > gp_total and gp_total > 0:
        return 1
    elif gp > 0 and gp_total > 0:
        return gp / gp_total
    else:
        return 0

# METRICA COMBINADA
def calculate_combined_metric(df, l_metrics: list, l_weights: list, metric_name: str = 'metric'):
    """
    Calcula una métrica combinada según las columnas de 'l_metrics' y los pesos de 'l_weights'.
    Normaliza las columnas en 'l_metrics' antes del cálculo y agrega el resultado como una nueva columna.
    """
    # Validar que el número de métricas y pesos coincida
    if len(l_metrics) != len(l_weights):
        raise ValueError("El número de métricas debe coincidir con el número de pesos.")
    
    norm_metrics = []
    
    # Normalizar cada columna en l_metrics
    for metric in l_metrics:
        df = normalize_column(df, col=metric)
        norm_metrics.append(metric + '_norm')
    
    # Definir función para calcular la métrica por fila
    def calculate_row_metric(row):
        return sum(row[norm_metric] * weight for norm_metric, weight in zip(norm_metrics, l_weights))
    
    # Aplicar la función fila por fila
    df[metric_name] = df.apply(calculate_row_metric, axis=1)
    return df

def normalize_column(df, col, norm_extension: str = '_norm', verbose : int = 0):
    
    # Determino puntos minimo y maximo de la columna
    p_min = df[col].min()
    p_max = df[col].max()
    if verbose >= 2:
        logger.info(f"Punto minimo: {p_min}. Punto maximo: {p_max}")

    # Normalizo columna
    df[f'{col}{norm_extension}'] = (df[col] - p_min) / (p_max - p_min)
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    pass