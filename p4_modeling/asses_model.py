import pandas as pd
import numpy as np
from sklearn import metrics
import warnings

def calculate_precision(df_result, var_resp, var_pred):
    """
    Calcula precision del modelo comparando sus predicciones y lo real
    :param df_result: Dataframe test. Con variable respuesta y con la prediccion del modelo
    :param var_resp: String. Nombre de la variable respuesta
    :param var_pred: String. Nombre de la variable con la prediccion del modelo
    :return: Float. Precision del modelo en sus predicciones
    """
    n_aciertos = 0

    # Por registro
    for i in range(len(df_result)):

        # Si el modelo predijo bien, sumo 1
        if df_result.loc[i, var_resp] == df_result.loc[i, var_pred]:
            n_aciertos += 1
    return n_aciertos / len(df_result) * 100

def confusion_matrix(y_real, y_pred):
    """
    Muestra matriz de confusion del modelo
    :param df_result: Dataframe test. Con variable respuesta y con la prediccion del modelo
    :param var_resp: String. Nombre de la variable respuesta
    :param var_pred: String. Nombre de la variable con la prediccion del modelo
    """
    # Calculo los numeros para la matriz de confusion
    confusion_matrix = metrics.confusion_matrix(y_real, y_pred)  # numpy.ndarray

    # Convertir el array a un DataFrame de pandas
    df_cm = pd.DataFrame(confusion_matrix)
    df_cm.index.name = "Resultado real"
    return df_cm

def determine_bookmaker_result(df, var_pred):
    
    # Por fila
    for i, row in df.iterrows():

        # Determino la cuota minima de las 3 posibles
        odds_min = min(row['odds_home'], row['odds_draw'], row['odds_away'])

        # Si la cuota minima es la del team home
        if row['odds_home'] == odds_min:
            df.loc[i, var_pred] = 1

        # Si la cuota minima es la del team away
        elif row['odds_away'] == odds_min:
            df.loc[i, var_pred] = 2

        # Si la cuota minima es la del draw
        else:
            df.loc[i, var_pred] = 0

    return df

def calculate_probas_bookmarker(df_match_odds): # Funciona bien. Comprobado.
    """
    Calcula las probabilidades de cada resultado (Home, Draw y Away) segun la casa de apuesta
    """
    # Por partido
    for idx, row in df_match_odds.iterrows():

        # Calcular probabilidades a partir de invertir las cuotas
        prob_home = 1 / row['odds_home']
        prob_draw = 1 / row['odds_draw']
        prob_away = 1 / row['odds_away']

        # Sumo las probabilidades (deberia ser >1 por el margen de ganancia de la casa de apuesta)
        sum_prob = prob_home + prob_draw + prob_away

        # Calculo probabilidades sin el margen
        prob_home = prob_home / sum_prob
        prob_draw = prob_draw / sum_prob
        prob_away = prob_away / sum_prob

        df_match_odds.loc[idx, ['prob_home_bm', 'prob_draw_bm', 'prob_away_bm']] = [prob_home, prob_draw, prob_away]

    return df_match_odds

def calculate_roi_by_betting_strategy(df: pd.DataFrame, stake_base: int = 100, _print: bool = False):
    """
    Determine the ROI for different betting strategies.
    Cosas a agregar:
    - Que no necesite df_etiquetas previo?
    - Desarrollo de tipo de stake poly
    """
    d = {}

    # Calculo cuotas segun probabilidades del modelo
    df = calculate_dif_probas(df)

    # Stake fijo # es como linear con m=0 y b=0
    df_roi1 = construct_stake_modified(df, stake_base, type_relation='equal')
    d['roi_stake_fijo'] = calculate_roi(df_roi1, _print=_print)

    df_roi5 = construct_stake_modified(df, stake_base, type_relation="linear", m=15, b=-2) # relacion m/b = -7.5
    d['roi_m15_b-2'] = calculate_roi(df_roi5, _print=_print)

    df_roi5 = construct_stake_modified(df, stake_base, type_relation="linear", m=20, b=-3) # relacion m/b = -6
    d['roi_m20_b-3'] = calculate_roi(df_roi5, _print=_print)

    df_roi5 = construct_stake_modified(df, stake_base, type_relation="linear", m=50, b=-8.33) # relacion m/b = -6
    d['roi_m50_b-8.3'] = calculate_roi(df_roi5, _print=_print)
    
    df_roi5 = construct_stake_modified(df, stake_base, type_relation="linear", m=50, b=-1.66)  # relacion m/b = -30
    d['roi_m50_b-1.66'] = calculate_roi(df_roi5, _print=_print)

    df_roi5 = construct_stake_modified(df, stake_base, type_relation="linear", m=50, b=-1)  # relacion m/b = -50
    d['roi_m50_b-1'] = calculate_roi(df_roi5, _print=_print)

    # Selecciono el mejor ROI
    filtered_values = [value for value in d.values() if not pd.isna(value)] # Quito ROI que puedan ser nan
    d['best_roi'] = max(filtered_values)
    return d

def calculate_dif_probas(df: pd.DataFrame, _print: bool = False):  # Mejor uso dif_prob_mod_bm como x (en vez de dif_cuotas_mod_bm)
    """
    Calcula las probabilidades de cada resultado (Home, Draw y Away) segun la casa de apuesta
    """
    # Por partido
    for idx, row in df.iterrows():

        # Obtengo probabilidad del resultado predicho por el modelo
        prob_max = max(row['prob_class_1'], row['prob_class_0'], row['prob_class_2'])

        # Obtengo probabilidad de la casa de apuesta para el resultado predicho por el modelo
        pred_mod = row['predicted_result']
        prob_bm_in_pred_result = row['prob_home_bm'] if pred_mod == 1 else row['prob_draw_bm'] if pred_mod == 0 else row['prob_away_bm']
     
        # Calculo diferencia de probabilidad entre mi modelo y bm para el predicted_result 
        dif_prob_mod_bm = prob_max - prob_bm_in_pred_result
        df.loc[idx, 'dif_prob_mod_bm'] = dif_prob_mod_bm
        if _print:
            print(f"Partido {idx}")
            print(f"Probabilidad resultado predicho: {prob_max}")
            print(f"Probabilidad bm para el resultado predicho: {prob_bm_in_pred_result}")
            print(f"Diferencia de probabilidad para el resultado predicho: {dif_prob_mod_bm}")

    return df

def construct_stake_modified(df: pd.DataFrame, stake_base, type_relation: str = 'equal', x1: float = -1, y1: float = -1, x2: float = 1, y2: float = 1,  m: float = None, b: float = None):
    """
    Construye multiplier y luego se lo aplica al stake base para construir la columna "stake_mod".
    """
    df = calculate_multiplier(df, type_relation=type_relation, x1=x1, y1=y1, x2=x2, y2=y2, m=m, b=b)

    # Por partido
    for i, row in df.iterrows():
        
        # Calculo stake mod
        stake_mod = stake_base * (1 + row['multiplier'])

        # Si el stake se vuelve negativo tras aplicar el multiplier (multiplier < -1)
        if stake_mod < 0:
            stake_mod = 0

        df.loc[i, 'stake_mod'] = stake_mod
    return df

def calculate_multiplier(df: pd.DataFrame, type_relation, x1, y1, x2, y2, m, b):
    """
    Construye columna multiplier. Cada fila tiene su multiplier segun las probabilidades del modelo y de la casa de apuesta para ese partido.
    """
    # Linear
    if type_relation == "equal":  
        df['multiplier'] = 0

    elif type_relation == "linear":  # y= m * x + b

        # Si la pendiente no fue pasada como parametro, calculo la pendiente y ordenada al origen
        if m is None:
            m = (y2-y1) / (x2-x1)
            b = y1 - m*x1            
            
        df['multiplier'] = df['dif_prob_mod_bm'] * m + b

    elif type_relation == "poly":  # y = b + b1 * x1 + b2 * x2 + ... + bn * xn # a desarrollar en un futuro
        pass

    elif type_relation == "exponential":  # y=a * b^x

        # Transformación logarítmica de los valores de x e y  (no pueden ser valores negativos)
        x2 = x2 - x1 + 3
        y2 = y2 - y1 + 3
        x1 = 3
        y1 = 3
        # print(x2, y2, x1, y1)

        # Resolver el sistema de ecuaciones
        A = np.array([[1, np.log(x1)], [1, np.log(x2)]])
        b = np.array([np.log(y1), np.log(y2)])
        # print("Z: ", A, b)

        a, log_b = np.linalg.solve(A, b)
        # print("X: ", a, log_b)

        # Calcular b a partir de su logaritmo
        b = np.exp(log_b)     
        # print("W: ", a, b)

        df['multiplier'] = a * (b ** df['dif_prob_mod_bm'])
        
    else:
        pass
    return df

def calculate_roi(df: pd.DataFrame, _print: bool = False):
    """
    Calcula ROI comparando las predicciones del modelo y los resultados reales.
    :param df_result: Dataframe de prueba con la variable respuesta y la predicción del modelo. (DataFrame)
    :param var_resp: Nombre de la variable respuesta. (str)
    :param var_pred: Nombre de la variable con la predicción del modelo. (str)
    :return: ROI del modelo. (float)
    """
    # Definicion de variables
    ingresos = 0

    # Elimino partidos con odds NaN
    df_sin_odds_nan = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
    if len(df) != len(df_sin_odds_nan):
        warnings.warn(f'Se eliminaron {len(df)-len(df_sin_odds_nan)} partidos de {len(df)} por tener odds=NaN')
    dinero_a_invertir = sum(df_sin_odds_nan['stake_mod'])
    
    # Filtrar el dataframe solo a las filas donde el modelo predijo correctamente
    df_correct = df_sin_odds_nan[df_sin_odds_nan['result'] == df_sin_odds_nan['predicted_result']]
    if len(df_correct) == len(df_sin_odds_nan):
        warnings.warn(f"Considera que acertó todos los partidos (es decir, 100% de precision). Es muy probable que no este filtrando bien los partidos que acierta de los que no.")
    if _print:
        print(f"De {df.shape[0]} partidos, acerté {df_correct.shape[0]}")
        print(f"Dinero a invertir: {dinero_a_invertir}")
    
    # Por partido acertado
    for idx, row in df_correct.iterrows():

        # Obtengo el ingreso obtenido segun la etiqueta
        cuota_ganada = row['odds_home'] if row['result'] == 1 else (row['odds_draw'] if row['result'] == 0 else row['odds_away'])  # Vefificada
        
        # Calculo ingresos por acertar el resultado
        ingresos += row['stake_mod'] * cuota_ganada
        if _print:
            print(f"Cuota ganada: {cuota_ganada} Stake: {row['stake_mod']} --> Ingresos: {ingresos}")

    # Calculo el ROI
    roi = (ingresos - dinero_a_invertir) / dinero_a_invertir * 100
    # cuota_media_ganada = ingresos / dinero_a_invertir
    # cuota_media_apostada = sum() --> no tengo la cuota apostada en cada partido... la eestoy calculando en el ciclo for...
    if _print:
        print(f"\t ROI: {roi:.1f}%. ${dinero_a_invertir:.0f} --> ${ingresos:.0f}")
    return roi

def prueba():
    var_resp = 'result'
    var_pred = 'predict_result'

    df_predicciones = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p4_modeling/data/england/modeling/df_predicciones.xlsx', index_col=0)
    d_roi = calculate_roi_by_betting_strategy(df_predicciones, _print=True)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()