import pandas as pd
import numpy as np
from sklearn import metrics
from utils.set_up_logging import logger

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

# Nosotros
def calculate_roi_by_betting_strategy(df: pd.DataFrame, strategy: str = "general", verbose: int = 0):
    """
    Determine the ROI for different betting strategies.

    # Paramaters:
        df: Dataframe con predicciones del modelo, cuotas de la casa de apuestas y el resultado real del partido. (DataFrame)
        stake_base: Stake base sobre el cual aplicar el multiplicador para obtener el stake variable. (int)
        _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

    # Returns
        Diccionario con ROI para las distintas estrategias de apuesta. (dict)
    """
    # Definicion de variables
    l_thr_dif_prob, d_rectas = define_hiperparameters(strategy)
    best_roi = -100000
    if verbose >= 1:
        logger.info(f"Calculating ROI...")
        logger.info(f"Hiperparametros estrategia de apuesta: \n- Doble oportunidad: {l_thr_dif_prob} \n- Rectas: {d_rectas}")
    
    # Eliminate rows with NaN odds or missing predictions
    df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away', 'predicted_result'])
    df = calculate_dif_proba_in_predicted_result(df)  # Calculo la diferencia de probabilidad entre el modelo y la casa de apuestas para el resultado predicho por el modelo (Columna 'dif_prob_mod_bm')

    # Por combinacion de hiperparametros 
    for prob in l_thr_dif_prob:

        # Determinamos el resultado a apostar (no necesariamente el resultado predicho)
        df2 = df.copy()  # esto parece boludo pero es clave sino df2 se le agrega las columnas de variacion de stake y los rdos son falsos...
        df2 = determine_result_to_bet(df2, thr_prob_min=prob)
        df2 = determine_winning_bets(df2)

        # Por recta con la cual variar el stake
        for key, value in d_rectas.items():
            normalized = True if key == 'kelly' else False  # Solo lo aplico a Kelly dado que puede tener stakes negativos y este debe ser mayor a 0.

            for a1, a2 in value:

                # Definir parámetros según la clave
                if key in ['linear', 'kelly']:
                    m, b = a1, a2
                    p1, p2 = None, None
                else:
                    m, b = None, None
                    p1, p2 = a1, a2
                
                if verbose >= 1:
                    logger.info(f"{a1} {a2} --> {m} {b} {p1} {p2}")

                # for normalized in [True, False]:
                for odd_weight in [0, 1, 2, 4]: # 0 significa no afectar stake con cuotas.  # 0.5,

                    for lim_sup in [0, 1]:

                        # Determino stake a apostar segun curva
                        df_aux = determine_stake_to_bet(df2, type_relation=key, m=m, b=b, p1=p1, p2=p2, odd_weight=odd_weight, dif_prob_sup_cap=lim_sup, normalized=normalized)

                        # Calculo ROI
                        df_pred, d_metrics = (calculate_reality_roi(df_aux)) if strategy == 'reality' else (calculate_roi(df_aux))
                        roi = d_metrics['roi_por_partido_r'] if strategy == 'reality' else d_metrics['roi_por_partido']            
                        # logger.info(f"Porcentaje de cuota {porc_cuota} --> ROI: {roi*100:.0f}")

                        # Verificación si es el mejor ROI y actualización en una sola línea
                        if roi > best_roi:
                            best_roi = roi
                            best_df_pred = df_pred.copy()
                            best_d_rois = d_metrics
                            best_d_rois.update({
                                'thr_prob_min_best': prob, 
                                'curva': key, 
                                'param1': a1, 
                                'param2': a2, 
                                'normalized': normalized,
                                'odd_weight': odd_weight,
                                'dif_prob_sup_cap': lim_sup
                            })

                            if verbose >= 1:
                                logger.info(f"Se encontró una mejor estrategia con ROI: {best_roi*100:.0f}. Metricas {best_d_rois}")

    return best_df_pred, best_d_rois

def define_hiperparameters(strategy):
    """
    Defino hiperparametros de estrategia de apuesta a probar segun si apuesto como la realidad o no.
    """
    if strategy=="general":
        l_thr_dif_prob = [-0.5, -0.35, -0.25]  # tengo varios valores porque cambia mucho si el modelo es under o no.
        d_rectas = {
            # "equal": [[(0, 0), (1, 0)]],
            # 'kelly': [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], # le sumo b pues la casa esta desbalanceada y yo no... y muchas veces conviene aunque paguen "poco"
            'linear': [[10, 0], [15, 0], [20, 0], [25, 0], [30, 0], [40, 0], [50, 0], [70, 0]],  # [1, 0], --> para que hay mas dif entre ROIpp de modelos en test..
            # 'exponential': [[(0.33, 4), (1, 10)], [(0.33, 6), (1, 10)], [(0.33, 4), (1, 30)], [(0.33, 2), (1, 30)]] # no entiendo la curva. Se resuelve con matrices.
        }

    elif strategy=="reality":
        l_thr_dif_prob = [-0.5, -0.35, -0.25]  # tengo varios valores porque cambia mucho si el modelo es under o no.
        d_rectas = {
            "equal": [[(0, 0), (1, 0)]],
            'linear': [[1, 0], [5, 0], [10, 0], [15, 0],[20, 0], [25, 0],[30, 0], [40, 0]],
            # 'exponential': [[(0.33, 3), (1, 15)], [(0.33, 5), (1, 20)], [(0.33, 5), (1, 30)]]
        }
    
    return l_thr_dif_prob, d_rectas

def calculate_dif_proba_in_predicted_result(df: pd.DataFrame):
    """
    Calcula la diferencia de probabilidad entre el modelo y la casa de apuestas para el resultado predicho por el modelo.
    
    # Parameters:
        df: Dataframe con probabilidades de cada resultado tanto para mi modelo como para la casa de apuestas. (DataFrame)

    # Returns:
        Dataframe pasado por parametro con nueva columna, 'dif_prob_mod_bm', siendo ésta la diferencia de probabilidad entre el modelo y la casa de apuestas para el resultado predicho por el modelo. (DataFrame)
    """
    # Por partido
    for idx, row in df.iterrows():

        # Obtengo probabilidad del resultado predicho por el modelo
        prob_max = max(row['prob_class_1'], row['prob_class_0'], row['prob_class_2'])

        # Obtengo probabilidad de la casa de apuesta para el resultado predicho por el modelo
        predicted_result_mod = row['predicted_result']
        prob_bm_in_pred_result = row['prob_home_bm'] if predicted_result_mod == 1 else row['prob_draw_bm'] if predicted_result_mod == 0 else row['prob_away_bm']
     
        # Calculo diferencia de probabilidad entre mi modelo y bm para el predicted_result 
        dif_prob_mod_bm = prob_max - prob_bm_in_pred_result
        df.loc[idx, 'dif_prob_mod_bm'] = dif_prob_mod_bm

    return df

def determine_result_to_bet(df: pd.DataFrame, thr_prob_min):
    """
    Determina el/los resultado/s a apostar (no necesariamente coincide con el resultado predicho).
    
    # Parameters:
        df: Dataframe con probabilidades de mi modelo y con predicciones y cuotas de la casa de apuestas. (DataFrame)
        _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

    # Returns:
        Dataframe pasado como parametro con resultado a apostar, la cuota a apostar, la estrategia utiilizada y la probabilidad del resultado al que se apuesta. (DataFrame)
    """
    # Por partido
    for id_match, row in df.iterrows():

        # Si el modelo esta MAS seguro del resultado predicho que la casa de apuestas
        if row['dif_prob_mod_bm'] >= thr_prob_min:

            # Apuesto al resultado predicho
            result_to_bet = row['predicted_result']
            dif_prob_result_to_bet = row['dif_prob_mod_bm']
            prob_result_to_bet = max(row['prob_class_1'], row['prob_class_0'], row['prob_class_2'])
            odd_to_bet = row['odds_home'] if row['predicted_result'] == 1 else (row['odds_draw'] if row['predicted_result'] == 0 else row['odds_away'])  # Verificada
            strategy = f"dif_prob_mod_bm > {thr_prob_min}"

        # Si el modelo esta MENOS seguro del resultado predicho que la casa de apuestas
        else:
            # Apuesto doble oportunidad sin el resultado predicho
            result_to_bet = -1 if row['predicted_result'] == 1 else (-2 if row['predicted_result'] == 2 else -0)
            dif_prob_result_to_bet = row['dif_prob_mod_bm'] * -1
            prob_result_to_bet = 1 - max(row['prob_class_1'], row['prob_class_0'], row['prob_class_2'])
            odd_to_bet = calculate_odd_double_chance(row, result_to_bet)
            strategy = f"dif_prob_mod_bm < {thr_prob_min}"

        # Guardo el resultado a apostar
        df.loc[id_match, 'result_to_bet'] = result_to_bet
        df.loc[id_match, 'dif_prob_result_to_bet'] = dif_prob_result_to_bet
        df.loc[id_match, 'prob_result_to_bet'] = prob_result_to_bet
        df.loc[id_match, 'odd_to_bet'] = odd_to_bet
        df.loc[id_match, 'strategy'] = strategy

    return df

def calculate_odd_double_chance(row, result_to_bet):
    """
    Calculo de cuota cuando se realiza apuesta doble oportunidad (es decir, a dos de los tres resultados).

    # Parameters:
        row: Partido con sus cuotas. (pd.Series)?
        result_to_bet: Resultado doble oportunidad al cual apostar. (integer)

    # Returns:
        Cuota a apostar cuando se hace doble oportunidad. (float)
    """
    odds_home, odds_draw, odds_away = float(row['odds_home']), float(row['odds_draw']), float(row['odds_away'])

    # Si el resultado a apostar es doble oportunidad sin Home
    if result_to_bet == -1:
        proporcion = odds_draw / (odds_draw + odds_away)  # 6,5 / (6,5 + 12) = 0,35
        odd_to_bet = odds_away * proporcion

    # Si el resultado a apostar es doble oportunidad sin Away
    elif result_to_bet == -2:
        proporcion = odds_draw / (odds_draw + odds_home)
        odd_to_bet = odds_home * proporcion

    # Si el resultado a apostar es doble oportunidad sin Draw
    elif result_to_bet == -0:
        proporcion = odds_away / (odds_away + odds_home)
        odd_to_bet = odds_home * proporcion

    return odd_to_bet

def determine_winning_bets(df: pd.DataFrame):
    """
    Determina si el resultado apostado fue el resultado real del partido o no.
    
    # Parameters
        df: Dataframe con partidos en los que se indica tanto el resultado a apostar como el resultado real del partido.

    # Returns
        Dataframe pasado como parametro con una nueva columna, 'acerte' indicando si se acertó el resultado apostado o no.
    """
    # inicializo columna "acerte"
    df['acerte'] = 0

    # Por partido
    for id_match, row in df.iterrows():

        # Si el resultado a apostar es Home, Draw o Away
        if row['result_to_bet'] >= 0:
            if row['result'] == row['result_to_bet']:
                df.loc[id_match, 'acerte'] = 1

        # Si el resultado a apostar es doble oportunidad sin Home
        elif row['result_to_bet'] == -1:
            if (row['result'] == 0) or (row['result'] == 2):
                df.loc[id_match, 'acerte'] = 1

        # Si el resultado a apostar es doble oportunidad sin Away
        elif row['result_to_bet'] == -2:
            if (row['result'] == 0) or (row['result'] == 1):
                df.loc[id_match, 'acerte'] = 1
        
        # Si el resultado a apostar es doble oportunidad sin Draw
        elif row['result_to_bet'] == -0:
            if (row['result'] == 2) or (row['result'] == 1):
                df.loc[id_match, 'acerte'] = 1

        # Si fallo la prediccion
        else:
            df.loc[id_match, 'acerte'] = np.nan

    # Imprimo warning si supuestamente acerté el 100% de partidos
    if len(df[df['acerte']==1]) == len(df):
        logger.warning(f"Considera que acertó todos los partidos (es decir, 100% de precision). Es muy probable que no este filtrando bien los partidos que acierta de los que no.")

    return df

def determine_stake_to_bet(df, type_relation: str = 'equal', p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None, 
                           odd_weight: float = 1, dif_prob_inf_cap: int = -1, dif_prob_sup_cap: int = 1, normalized: bool = False):
    """
    Construye multiplicador para variar el stake y poder apostar difentes cantidades en diferentes partidos. 
    Cuanto mayor es la probabilidad del modelo para el resultado a apostar, mas dinero apuesto.

    # Parameters
        df: Dataframe (DataFrame)
        type_relation: Tipo de relacion entre el multiplicador y la probabilidad del modelo para el resultado a apostar. (str)
        p1: Primer punto (x, y) para construir curva. (float)
        p2: Segundo punto (x, y) para construir curva. (float)
        m: Pendiente de la recta. Solo cuando type_relation = 'linear'. (float)
        b: Ordenada al origen de la recta. Solo cuando type_relation = 'linear'. (float)
        odd_weight: Peso de dif_prob_result_to_bet en el stake.
            (e.g. 0 entonces no tenemos en cuenta cuotas en stake, con 0.5 tenemos en cuenta las cuotas pero no tanto y asi. ) 
        dif_prob_inf_cap: Minima diferencia de probabilidad con casa de apuesta para afectar el stake con cuotas. Va de -1 a 1. 
            (e.g. si es -0.25, si la dif de proba es menor a -25% de todas maneras afecto el stake como si esta fuera de un -25%).
        dif_prob_sup_cap: Maxima diferencia de probabilidad  con casa de apuesta para afectar el stake con cuotas. Va de -1 a 1. 
            (e.g. si es 0.1, si la dif de proba es mayor a 10% de todas maneras afecto el stake como si esta fuera de un 10%).

    # Returns
        Dataframe pasado como parametro con nueva columna 'stake_to_bet'
    """
    # Separo puntos en x e y
    if p1 is not None and p2 is not None:
        x1, y1 = p1
        x2, y2 = p2

    # Si la pendiente no fue pasada como parametro, calculo la pendiente y ordenada al origen
    if m is None:
        m = (y2-y1) / (x2-x1)
        b = y1 - m*x1

    if type_relation == "equal":  
        df['stake_to_bet'] = 1

    elif type_relation == 'kelly':
        df['stake_to_bet'] = ((df['prob_result_to_bet'] + b) - (1 - df['prob_result_to_bet'])) * 100 / (df['odd_to_bet'] - 1)  #  (= (((df['odd_to_bet'] - 1) * df['prob_result_to_bet'] + b) - (1 - df['prob_result_to_bet'])) / (df['odd_to_bet'] - 1) * 100)

        # Puntos para normalizar
        p_min, p_max = 0, 50 
        
    # Linear
    elif type_relation == "linear": # Vario stake con prob_result_to_bet y cuotas de la casa

        if dif_prob_inf_cap != dif_prob_sup_cap:
            df['stake_to_bet'] = (df['prob_result_to_bet'] + np.clip(df['dif_prob_result_to_bet'], dif_prob_inf_cap, dif_prob_sup_cap) * odd_weight) * m + b  # NO usar np.where() pues descarta los valores fuera del rango. En cambio np.clip() los ajusta dentro del rango. # Limita los valores de 'dif_prob_result_to_bet' a un rango de -0.15 a 0.15

        else:
            df['stake_to_bet'] = df['prob_result_to_bet'] * m + b
        
        # Puntos para normalizar --> no tiene mucho sentido. Para eso esta exponential (modificar mas el stake ante un menor cambio en proba). Incluso con el b de linear tambien puedo lograr algo parecido.
        # p_min, p_max = (0.55 * m + b), (0.7 * m + b) # El punto min usa un stake de m_to_bet / 2. Si queres que prob=0.33 use un stake mas bajo, no tiene sentido usarlo como p_min.

    elif type_relation == "poly":  # y = b + b1 * x1 + b2 * x2 + ... + bn * xn # a desarrollar en un futuro
        pass

    elif type_relation == "exponential":  # y=a * b^x

        # Resolver el sistema de ecuaciones
        A = np.array([[1, np.log(x1)], [1, np.log(x2)]])
        B = np.array([np.log(y1), np.log(y2)])

        a, log_b = np.linalg.solve(A, B)

        # Calcular b a partir de su logaritmo
        b = np.exp(log_b)   

        # Calculo stake_to_bet
        if dif_prob_inf_cap != dif_prob_sup_cap:
            df['stake_to_bet'] = a * (b ** (df['prob_result_to_bet'] + np.clip(df['dif_prob_result_to_bet'], dif_prob_inf_cap, dif_prob_sup_cap)))
        else:
            df['stake_to_bet'] = a * (b ** df['prob_result_to_bet'])

    # Capa de funcion sigmoide (creo que sirve nada mas para Kelly. Para linear seria muy parecico a la variacion exponencial puesto que buscas agrandar las diferencias entre probas de 0.33 y 1)
    if normalized:
        df = df.rename(columns={'stake_to_bet': 'stake_to_bet_raw'})

        # Normalización: Escalar los valores de kelly_raw entre 0 y 1
        df['stake_to_bet_norm'] = (df['stake_to_bet_raw'] - p_min) / (p_max - p_min)

        # Usar la función sigmoide para reducir la variabilidad y escalar entre 0 y 30
        num = m if m > 0 else 1
        df['stake_to_bet'] = num / (1 + np.exp(-df['stake_to_bet_norm']))
        # df = df.drop(['stake_to_bet_raw', 'stake_to_bet_normalized'], axis=1)
    
    # Si existen la columna 'emergency_fill' (pues para X_test no existe. Es solo para stakes en produccion). --> Ver si falla cuando hago main_best_model.py (ni deberia entrar)
    if 'emergency_fill' in df.columns:
        logger.warning("Disminución de stakes por copiado de emergencia.")
        # Reducir el stake al 50% solo para las filas donde 'emergency_fill' es igual a 1
        df.loc[df['emergency_fill'] == 1, 'stake_to_bet'] *= 0.5

    # Restringo stake de 0 a 99 (e.g. evito que el stake a apostar sea mayor al 100% del bank)
    val_min, val_max = 0, 99  # 100 no pues sino el bank es negativo.
    func = lambda x: val_min if x < val_min else (val_max if x>val_max else x)
    df['stake_to_bet'] = df['stake_to_bet'].apply(func)

    return df

# Metricas
def calculate_roi(df: pd.DataFrame):
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
        df.loc[idx, 'bank_inicial'] = bank_final
        df.loc[idx, 'stake_to_bet_en_$'] = stake_a_apostar

        # Determino ganancias / perdidas 
        ingresos = stake_a_apostar * row['odd_to_bet'] if row['acerte'] == 1 else 0
        ganancia = ingresos - stake_a_apostar
        bank_final += ganancia
        df.loc[idx, 'G/P'] = ganancia
        df.loc[idx, 'bank_final'] = bank_final

        # Guardo ROI en partidos especificados
        if cont in l_rois_partido:
            roi_partido = (bank_final - bank_inicial) / bank_inicial * 100
            d_rois[f'roi_{cont}'] = roi_partido / cont
            df.loc[idx, 'roi_partido'] = roi_partido / cont

        # Raise error si perdi todo el dinero de las apuestas
        if bank_final <= 0:
            logger.warning("El dinero tras apuestas se hizo negativo y esto no es posible puesto que el stake siempre es un % del bank.")
            break

    # Calculo el ROI
    roi = (bank_final - bank_inicial) / bank_inicial * 100
    roi_por_partido = roi / n_apuestas  # No quiero el ROI mas alto sino el ROI / partido mas alto
    d_rois['roi'] = roi
    d_rois['roi_por_partido'] = roi_por_partido
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


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    from random import randint
    import os
    from dotenv import load_dotenv
    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    # Levanto dataframe con predicciones de un modelo ejemplo
    # df_predicciones = pd.read_excel('p4_modeling/data/england/modeling/df_predicciones.xlsx', index_col=0)
    df_predicciones = pd.read_excel('main_find_best_hyper/data/england/2024-04-22/assess_model_in_prod/modeling/1_df_pred_metrics.xlsx', index_col=0)
    print(df_predicciones.head(5))
    
    l_cols = ['stake_to_bet', 'stake_to_bet', 'odd_to_bet', 'strategy', 'acerte', 'G/P', 'dinero_tras_apuestas', 'prob_result_to_bet', 'dif_prob_result_to_bet', 'result_to_bet']
    for col in l_cols:
        try:
            df_predicciones = df_predicciones.drop(col, axis=1)
        except:
            pass
    # df_predicciones = df_predicciones.drop(['stake_to_bet', 'multiplier', 'odd_to_bet', 'strategy', 'acerte', 'G/P', 'dinero_tras_apuestas', 'prob_result_to_bet', 'dif_prob_result_to_bet', 'result_to_bet'], axis=1)
    shuffled_df = df_predicciones.sample(frac=1, random_state=140)  # Usa random_state para reproducibilidad

    # Evaluo predicciones
    df, d_roi = calculate_roi_by_betting_strategy(shuffled_df, _print=True)
    df.to_excel(f"{BASE_DIR_LOCAL}/df_pred_best.xlsx", index=True)
