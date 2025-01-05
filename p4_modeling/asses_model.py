import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score
from p3_data_preparation import construct_data


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
def determine_result_by_bookmaker(df, col_name):
    """
    Determina el resultado del partido predicho segun la casa de apuestas. 

    # Paramaters:
        df: Dataframe con cuotas de la casa de apuestas. (DataFrame)
        col_name: Nombre de la nueva variable con el resultado predicho segun la casa de apuestas. (str)

    # Returns
        Dataframe pasado como parametro con nueva columna con el resultado predicho segun la casa de apuestas. (DataFrame)
    """
    # Por partido
    for id_match, row in df.iterrows():

        # Determino la cuota minima de las 3 posibles
        odds_min = min(row['odds_home'], row['odds_draw'], row['odds_away'])

        # Determino resultado predicho segun casa de apuestas (el de la cuota minima) y lo guardo
        result_pred_bm = 1 if row['odds_home'] == odds_min else (2 if row['odds_away'] == odds_min else 0)
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
    # Por partido
    for idx, row in df_match_odds.iterrows():

        # Calcular probabilidades a partir de invertir las cuotas
        prob_home_with_over = 1 / float(row['odds_home'])
        prob_draw_with_over = 1 / float(row['odds_draw'])
        prob_away_with_over = 1 / float(row['odds_away'])

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

# ROI
def calculate_roi(df: pd.DataFrame, name_extension='', save_roi: bool = False):
    """
    Calcula ROI obtenido segun las predicciones del modelo y el resultado real de los partidos. Para poder seleccionar el mejor modelo.

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

    # Por partido
    for idx, row in df.iterrows():
        cont += 1

        # Defino stake a apostar en pesos (a partir del stake como porcentaje del bank)
        stake_a_apostar =  bank_final * row['stake_to_bet'] / 100
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

        # Guardo ROI en partidos especificados
        if save_roi:
            if cont in l_rois_partido:
                roi_partido = (bank_final - bank_inicial) / bank_inicial * 100
                d_rois[f'{name_extension}roi_{cont}'] = roi_partido / cont
                df.loc[idx, f'{name_extension}roi_partido'] = roi_partido / cont

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

def calculate_combined_metric(df, roi_weight: float = 0.75, name_extension: str = '', normalize: bool = True):
    """
    Calcula la métrica combinada según 'roi_por_partido' y 'expected_roi_por_partido'.
    Normaliza las columnas antes del cálculo, asigna el resultado a una nueva columna llamada 'metric' y retorna el DataFrame.
    """
    col1, col2 = f'roi_por_partido', f'expected_roi_por_partido'
      
    if normalize:
        norm_extension = '_norm' 
        
        # Normalizo columnas por separado (cada una segun su escala)
        df = normalize_column(df, col=col1, norm_extension=norm_extension)
        df = normalize_column(df, col=col2, norm_extension=norm_extension)
        col1, col2 = f'{col1}{norm_extension}', f'{col2}{norm_extension}'

    def calculate_row_metric(row):
        roi_pp = row[col1]
        expected_roi_pp = row[col2]
        return ((roi_weight * roi_pp) + ((1 - roi_weight) * expected_roi_pp))

    # Aplicar la función fila por fila
    df[f'metric{name_extension}'] = df.apply(calculate_row_metric, axis=1)
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

def asignar_roi_weight(df, roi_col='roi_por_partido', expected_roi_col='expected_roi_por_partido'):
    """
    Asigna valores de roi_weight basados en la correlación entre ROI y Expected ROI por modelo.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        roi_col (str): Nombre de la columna de ROI. Default 'roi_por_partido'.
        expected_roi_col (str): Nombre de la columna de Expected ROI. Default 'expected_roi_por_partido'.
        
    Returns:
        'roi_weight' asignado segun correlacion entre ROI y Expected ROI.
    """       
    correlacion = df[roi_col].corr(df[expected_roi_col])
    logger.info(f"La correlacion entre {roi_col} y {expected_roi_col} es de {correlacion}")

    min_high_corr = 0.7
    max_low_corr = 0.5
    max_no_corr = 0.2

    # Si la correlacion es alta
    if correlacion >= min_high_corr:
        value = 0.25

    # Si la correlacion es media
    elif correlacion >= max_low_corr:
        value = 0.5

    # Si la correlacion es baja
    elif correlacion >= max_no_corr:
        value = 0.75

    # Si no hay correlacion
    else:
        value = 1
    
    logger.critical(f"El roi_weight a usar es {value}. Es decir, un peso de {value*100:.0f}% para el ROI y de {(1-value)*100:.0f}% para el Expected ROI")
    return value


# Simplificar...
def calculate_metrics( 
        df_pred_proba: pd.DataFrame,
        country: str,
        var_resp: str = 'result',
        var_pred: str = 'predicted_result',
        var_pred_bm: str = 'bookmaker_result',
        retrain: bool = False,
        verbose: int = 0,
        export: bool = False):
    """
    Calculo metricas como precision y ROI de las predicciones del modelo entrenado.
    """
    # Defino variables
    y_test = df_pred_proba[var_resp].values  # Etiquetas reales
    y_pred = df_pred_proba[var_pred].values  # Predicciones del modelo
    base_path = f'./data/{country}/p4_modeling'

    # Calculo métricas básicas
    d_metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred) * 100,
        'recall': recall_score(y_test, y_pred, average='macro') * 100,
        'f1_score': f1_score(y_test, y_pred, average='macro') * 100,
    }

    if verbose >= 1:
        # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
        df_conf_mat = confusion_matrix(y_test, y_pred)
        if export:
            df_conf_mat.to_excel(f'{base_path}/modeling/df_conf_matrix.xlsx')

    # Procesar df_match y df_match_odds
    df_match = load_file_by_condition(country=country, retrain=retrain, file_name="df_match.xlsx")
    df_match_odds = load_file_by_condition(country=country, retrain=retrain, file_name="df_match_odds.xlsx")
    if verbose >= 2:
        logger.info(df_match_odds)

    indices_to_use = df_pred_proba.index  # Selecciono los partidos que estan en df_test
    df_match = df_match[df_match.index.isin(indices_to_use)].reindex(indices_to_use)
    df_match_odds = df_match_odds[df_match_odds.index.isin(indices_to_use)].reindex(indices_to_use) # Reordeno df_match_odds el orden de X_test (X_test sufrió un shuffle) --> sino lo haces, la precision del bookmaker se calcula mal dado que y_pred tiene un orden ≠ al de y_test

    # Calculo metricas de bookie
    df_match_odds = calculate_result_probabilities_by_bookmaker(df_match_odds) # Caculo probabilidades segun casa de apuesta
    df_match_odds = determine_result_by_bookmaker(df_match_odds, var_pred_bm)  # Determino resultado predicho segun cuota minima (e.g. "Home")
    y_pred_bm = df_match_odds[var_pred_bm].values

    d_metrics.update({
        'test_accuracy_bm': accuracy_score(y_test, y_pred_bm) * 100,  # Calcula bien tras el reindex()
        'dif_prec_bm': d_metrics['test_accuracy'] - accuracy_score(y_test, y_pred_bm) * 100,
    })

    # Concatenación selectiva
    df_filled = pd.read_excel(f'./data/{country}/p3_data_preparation/treat_nan/df_filled_columns.xlsx', index_col=0)
    l_cols = [col for col in ['emergency_fill', 'player_emergency_fill', 'n_col_filled_sin_player', 'n_col_filled', 'perc_col_filled', 'l_col_filled'] if col in df_filled.columns]
    columns_to_concat = [
        df_match[['date', 'id_team_home', 'id_team_away', 'id_country', 'id_competition', 'goals_home', 'goals_away',  'expected_goals_(xg)_home', 'expected_goals_(xg)_away']],
        df_match_odds,
        df_pred_proba,
        df_filled[l_cols]
    ]
    df_predicciones = pd.concat(columns_to_concat, axis=1)

    if verbose >=1:
        print(d_metrics)

    return df_predicciones, d_metrics

def calculate_advanced_metrics(df_predicciones):
    """
    Calculo metricas mas avanzadas que precision o recall.

    Posibles mejoras:
        - l_col_filled = # Listado de columnas que rellena....
        - Mas o mejores metricas
    """
    # RELLENO DE NAN
    # Calculo metricas sobre relleno de nan
    rows_filled = df_predicciones[df_predicciones['player_emergency_fill'] == 1].index
    rows_not_filled = df_predicciones[df_predicciones['player_emergency_fill'] != 1].index
    average_col_filled = df_predicciones['n_col_filled'].sum() / len(df_predicciones)
    # l_col_filled = # Listado de columnas que rellena....
    # print(len(rows_filled), len(rows_not_filled))

    # G/P segun relleno de NaN
    gp_filled = df_predicciones.loc[rows_filled, 'G/P_sin_bank'].sum()
    gp_not_filled = df_predicciones.loc[rows_not_filled, 'G/P_sin_bank'].sum()
    gp_total = df_predicciones['G/P_sin_bank'].sum()
    perc_gp_filled = calculate_perc_gp(gp_filled, gp_total)
    perc_gp_not_filled = calculate_perc_gp(gp_not_filled, gp_total)

    # POR RESULTADO
    # Calculo numero de predicciones por resultado
    d_distrib = determine_distribution(df_predicciones)
    df_pred_home = df_predicciones[df_predicciones['predicted_result'] == 1]
    df_pred_draw = df_predicciones[df_predicciones['predicted_result'] == 0]
    df_pred_away = df_predicciones[df_predicciones['predicted_result'] == 2]
    ## G/P por resultado
    gp_home = df_pred_home['G/P_sin_bank'].sum()
    gp_draw = df_pred_draw['G/P_sin_bank'].sum()
    gp_away = df_pred_away['G/P_sin_bank'].sum()
    gp_total = gp_home + gp_draw + gp_away
    # G/P por resultado %
    perc_gp_home = calculate_perc_gp(gp_home, gp_total)
    perc_gp_draw = calculate_perc_gp(gp_draw, gp_total)
    perc_gp_away = calculate_perc_gp(gp_away, gp_total)
    ## Precision por resultado
    prec_home = int( df_pred_home['acerte'].sum() / len(df_pred_home) * 100) if len(df_pred_home) > 0 else 0
    prec_draw = int( df_pred_draw['acerte'].sum() / len(df_pred_draw) * 100) if len(df_pred_draw) > 0 else 0
    prec_away = int( df_pred_away['acerte'].sum() / len(df_pred_away) * 100) if len(df_pred_away) > 0 else 0

    d = {
        # RELLENO DE NAN
        # Cantidad de registros rellenados y average de columnas rellenadas
        'n_matches_filled': len(rows_filled),
        'average_col_filled': average_col_filled,
        # G/P cuando relleno y G/P cuando no relleno
        'gp_filled': gp_filled, 
        'gp_not_filled': gp_not_filled,
        '%_gp_filled': perc_gp_filled,
        '%_gp_not_filled': perc_gp_not_filled,

        # POR RESULTADO
        **d_distrib,
        # Precision por resultado
        'acc_home': prec_home,
        'acc_draw': prec_draw,
        'acc_away': prec_away,
        # G/P por resultado
        'gp_home': gp_home,
        'gp_draw': gp_draw,
        'gp_away': gp_away,
        'gp_total': gp_total,
        '%_gp_home': perc_gp_home,
        '%_gp_draw': perc_gp_draw,
        '%_gp_away': perc_gp_away
        }
    return d

def calculate_perc_gp(gp, gp_total):
    if gp > gp_total and gp_total > 0:
        return 1
    elif gp > 0 and gp_total > 0:
        return gp / gp_total
    else:
        return 0
    
def load_file_by_condition(country: str, retrain: bool, file_name: str) -> pd.DataFrame:
    subpath = f"data/{country}/p6_deployment/missing/old_updated" if retrain else f'data/{country}/p2_data_understanding'
    return  pd.read_excel(f'{subpath}/{file_name}', index_col=0) # --> missing no lo necesita y el otro si?

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    pass