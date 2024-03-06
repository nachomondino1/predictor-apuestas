import pandas as pd
import numpy as np
from sklearn import metrics

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

def calculate_roi_by_betting_strategy(df: pd.DataFrame, stake_base: int = 100):
    """
    Determine the ROI for different betting strategies.
    Cosas a agregar:
    - Que no necesite df_etiquetas previo?
    - Desarrollo de tipo de stake poly
    """
    d = {}
    # Calculo cuotas segun probabilidades del modelo
    df = calculate_odds_model(df)

    # Stake fijo # es como linear con m=0 y b=0
    print("Estrategia 1: Stake fijo")
    df_roi1 = construct_stake_modified(df, stake_base, type_stake='equal')
    roi1 = calculate_roi(df_roi1)
    d['roi_stake_fijo'] = roi1

    # Stake lineal 1
    print("\n Estrategia 2: Stake lineal, casi todos los partidos el mismo stake. (m=0.2 y b=0)")
    df_roi2 = construct_stake_modified(df, stake_base, type_stake="linear",  m=0.2, b=0)
    roi2 = calculate_roi(df_roi2)
    d['roi_stake_var_m.2_b0'] = roi2

    # Stake lineal 2
    print("\n Estrategia 3: Stake lineal, relacion 1:1 stake y dif_cuotas. (m=1 y b=0)")
    df_roi3 = construct_stake_modified(df, stake_base, type_stake="linear", m=1, b=0)
    roi3 = calculate_roi(df_roi3)
    d['roi_stake_var_m1_b0'] = roi3

    # Stake lineal 3 
    print("\n Estrategia 4: Stake lineal, relacion 3:1 stake y dif_cuotas. (m=3 y b=0)")
    df_roi4 = construct_stake_modified(df, stake_base, type_stake="linear",  m=3, b=0)
    roi4 = calculate_roi(df_roi4)
    d['roi_stake_var_m3_b0'] = roi4

    # Stake lineal 4
    print("\n Estrategia 5: Stake lineal, relacion 3:1 stake y dif_cuotas y apuesto cuando esta mas seguro. (m=3 y b=-1)")
    df_roi5 = construct_stake_modified(df, stake_base, type_stake="linear", m=3, b=-1)
    roi5 = calculate_roi(df_roi5)
    d['roi_stake_var_m3_b-1'] = roi5

    # Stake lineal 5
    print("\n Estrategia 6: Stake lineal, relacion 3:1 stake y dif_cuotas y apuesto cuando esta menos seguro. (m=3 y b=1)")
    df_roi6 = construct_stake_modified(df, stake_base, type_stake="linear", m=3, b=1)
    roi6 = calculate_roi(df_roi6)
    d['roi_stake_var_m3_b1'] = roi6

    # Stake exponencial  # aun tengo problema con los x1 e y1 negativos...
    # df_roi4 = construct_stake_modified(df, stake_base, type_stake="exponential")
    # roi4 = calculate_roi(df_roi4)
    
    # Selecciono el mejor ROI
    d['best_roi'] = max(d.values())
    return d

def calculate_odds_model(df: pd.DataFrame):  # Mejor uso dif_prob_mod_bm como x (en vez de dif_cuotas_mod_bm)
    """
    Calcula las probabilidades de cada resultado (Home, Draw y Away) segun la casa de apuesta
    """
    # Por partido
    for idx, row in df.iterrows():

        # Calcular probabilidades a partir de invertir las cuotas
        cuota_home = 1 / row['prob_class_1']
        cuota_draw = 1 / row['prob_class_0']
        cuota_away = 1 / row['prob_class_2']
        cuota_min = min(cuota_home, cuota_draw, cuota_away)

        # Determino diferencia entre cuotas de mi modelo y de casa de apuesta
        pred_mod = df.loc[idx, 'predicted_result']
        cuota_bm = row['odds_home'] if pred_mod == 1 else row['odds_draw'] if pred_mod == 0 else row['odds_away']
        dif_cuota_bm_mod = cuota_bm - cuota_min

        df.loc[idx, 'dif_cuota_bm_mod'] = dif_cuota_bm_mod
        # df.loc[idx, ['odds_home_mod', 'odds_draw_mod', 'odds_away_mod', 'cuota_min', 'cuota_bm', 'dif_cuota_bm_mod']] = [cuota_home, cuota_draw, cuota_away, cuota_min, cuota_bm, dif_cuota_bm_mod]
    return df

def construct_stake_modified(df: pd.DataFrame, stake_base, type_stake, m: float = None, b: float = None, x1: float = -1, y1: float = -1, x2: float = 1, y2: float = 1):
    """
    Construye multiplier y luego se lo aplica al stake base para construir la columna "stake_mod".
    """
    df = calculate_multiplier(df, m, b, x1, y1, x2, y2, type_stake)

    # Por partido
    for i, row in df.iterrows():
        
        # Calculo stake mod
        stake_mod = stake_base * (1 + row['multiplier'])

        # Si el stake se vuelve negativo tras aplicar el multiplier (multiplier < -1)
        if stake_mod < 0:
            stake_mod = 0

        df.loc[i, 'stake_mod'] = stake_mod
    return df

def calculate_multiplier(df: pd.DataFrame, m: float, b: float, x1: float, y1: float, x2: float, y2: float, type_stake: str = "equal"):
    """
    Construye columna multiplier. Cada fila tiene su multiplier segun las probabilidades del modelo y de la casa de apuesta para ese partido.
    """
    # Linear
    if type_stake == "equal":  
        df['multiplier'] = 0

    elif type_stake == "linear":  # y= m * x + b

        # Si la pendiente no fue pasada como parametro, calculo la pendiente y ordenada al origen
        if m is None:
            m = (y2-y1) / (x2-x1)
            b = y1-m*x1            

        df['multiplier'] = df['dif_cuota_bm_mod'] * m + b

    elif type_stake == "poly":  # y = b + b1 * x1 + b2 * x2 + ... + bn * xn # a desarrollar en un futuro
        pass

    elif type_stake == "exponential":  # y=a * b^x

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

        df['multiplier'] = a * (b ** df['dif_cuota_bm_mod'])
        
        df.to_excel('/Users/nachomondino/Desktop/AAAA.xlsx')
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
    dinero_a_invertir = sum(df['stake_mod'])

    # Filtrar el dataframe solo a las filas donde el modelo predijo correctamente
    df_correct = df[df['result'] == df['predicted_result']]
    if _print:
        print(df_correct.shape)
     
    # Por partido acertado
    for idx, row in df_correct.iterrows():

        result_etiqueta = row['result']

        # Obtengo el ingreso obtenido segun la etiqueta
        cuota = row['odds_home'] if result_etiqueta == 1 else row['odds_draw'] if result_etiqueta == 0 else row['odds_away']  # Vefificada
        
        # Calculo ingresos por acertar el resultado
        ingresos += row['stake_mod'] * cuota
 
    # Calculo el ROI
    roi = (ingresos - dinero_a_invertir) / dinero_a_invertir * 100
    cuota_media_ganada = ingresos / dinero_a_invertir
    # cuota_media_apostada = sum() --> no tengo la cuota apostada en cada partido... la eestoy calculando en el ciclo for...
    if _print:
        print(f"\t ROI: {roi:.1f}%. ${dinero_a_invertir:.0f} --> ${ingresos:.0f}")
    return roi

def prueba():
    var_resp = 'result'
    var_pred = 'predict_result'

    df = pd.read_excel('/Users/nachomondino/Desktop/df_results.xlsx')

    calculate_precision(df, var_resp=var_resp, var_pred=var_pred)

    # confusion_matrix(df, var_resp, var_pred)
    # df_cm.to_excel('/Users/nachomondino/Desktop/cm.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()