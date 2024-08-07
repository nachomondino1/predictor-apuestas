import pandas as pd
import numpy as np
from sklearn import metrics
from set_up_logging import logger

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
    return df_cm

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

def calculate_roi_by_betting_strategy(df: pd.DataFrame, stake_base: int = 1, _print: bool = False):
    """
    Determine the ROI for different betting strategies.

    # Paramaters:
        df: Dataframe con predicciones del modelo, cuotas de la casa de apuestas y el resultado real del partido. (DataFrame)
        stake_base: Stake base sobre el cual aplicar el multiplicador para obtener el stake variable. (int)
        _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

    # Returns
        Diccionario con ROI para las distintas estrategias de apuesta. (dict)
    """
    # Hiperparametros
    l_thr_dif_prob = [-0.5, -0.3, -0.2, -0.15, -0.1, -0.05, 0]  # tengo varios valores porque cambia mucho si el modelo es under o no.
    d_rectas = {"equal": [[(0, 0), (1, 0)]], 'linear': [[10, 0], [20, 0], [30, 0], [50, -5], [50, 0], [70, 0], [80, 0]], 'exponential': [[(0.5, 4), (1, 10)], [(0.33, 5), (1, 50)], [(0.33, 10), (1, 50)], [(0.33, 10), (1, 80)]]}  
    
    # Definicion de variables
    best_roi = -100000
    roi_max = -100000

    # Elimino partidos con odds NaN y sin predicciones
    df_sin_odds_nan = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away', 'predicted_result'])  # Con 'predicted_result' elimino tambien los partidos que no predijo el modelo, por si me llegan a pasar eso
    if len(df) != len(df_sin_odds_nan):
        logger.info(f'Se eliminaron {len(df)-len(df_sin_odds_nan)} partidos de {len(df)} por tener odds=NaN o no haber sido predicho')
        df = df_sin_odds_nan.copy()

    # Calculo la diferencia de probabilidad entre el modelo y la casa de apuestas para el resultado predicho por el modelo (Columna 'dif_prob_mod_bm')
    df = calculate_dif_proba_in_predicted_result(df)

    # Por combinacion de hiperparametros 
    for prob in l_thr_dif_prob:
        d = {}
        if _print:
            print(f"\n - thr_prob_min: {prob}\n")

        # Determinamos el resultado a apostar (no necesariamente el resultado predicho)
        df2 = df.copy()  # esto parece boludo pero es clave sino df2 se le agrega las columnas de variacion de stake y los rdos son falsos...
        df2 = determine_result_to_bet(df2, thr_prob_min=prob)
        # print(df2.shape)
        df2 = determine_winning_bets(df2)

        # Por recta con la cual variar el stake
        for key, value in d_rectas.items():
            if _print:
                print(f"Key: {key} Value: {value}")

            for a1, a2 in value:

                m = a1 if key == 'linear' else None  # m, b = a1, a2 if key == 'linear' else None, None
                b = a2 if key == 'linear' else None
                p1 = a1 if key != 'linear' else None
                p2 = a2 if key != 'linear' else None
                if _print:
                    print(f'\t a1={a1} ; a2={a2}')
                    print(f'\t m={m} ; b={b} ; p1={p1}; p2={p2}')

                # Determino stake a apostar segun curva
                df_aux = determine_stake_to_bet(df2, stake_base=stake_base, type_relation=key, m=m, b=b, p1=p1, p2=p2)

                # Calculo roi stake a apostar segun curva
                df_no_se, d_rois, = calculate_roi(df_aux, _print=False)
                roi = d_rois['roi_por_partido']                    
                d[f'roi_stake_{key}_{a1}_{a2}'] = roi
                if _print:
                    print("ROI: ", roi)

                if roi > roi_max:
                    if _print:
                        print(f"ROI MAX: {roi_max} --> {roi}")
                    roi_max = roi
                    d_rois_best = d_rois
                    best_prob = prob
                    best_key = key
                    best_a1, best_a2 = a1, a2
                    df_pred_best = df_no_se.copy()

            # Si es el mejor ROI
            if roi_max > best_roi:
                best_roi = roi_max
                best_df_pred = df_pred_best.copy()
                best_d_rois = d_rois_best
                if _print:
                    print(f"BEST ROI: {best_roi}")

                # Guardar hiperparametros de estrategia...
                d_best = {'thr_prob_min_best': best_prob, 'curva': best_key, 'param1': best_a1, 'param2': best_a2}
                best_d_rois.update(d_best)

    if _print:
        print(f"\n La mejor estrategia es: \n - thr_prob_min: {d_best['thr_prob_min_best']}\n - curva: {d_best['curva']}\n - param1: {d_best['param1']}; \n - param2: {d_best['param2']}")
        print(best_d_rois)
    return best_df_pred, best_d_rois

def calculate_dif_proba_in_predicted_result(df: pd.DataFrame, _print: bool = False):
    """
    Calcula la diferencia de probabilidad entre el modelo y la casa de apuestas para el resultado predicho por el modelo.
    
    # Parameters:
        df: Dataframe con probabilidades de cada resultado tanto para mi modelo como para la casa de apuestas. (DataFrame)
        _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

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
        if _print:
            print(f"Partido {idx}")
            print(f"Probabilidad resultado predicho: {prob_max}")
            print(f"Probabilidad bm para el resultado predicho: {prob_bm_in_pred_result}")
            print(f"Diferencia de probabilidad para el resultado predicho: {dif_prob_mod_bm}")

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
        # print(f"\nPartido: {id_match}")
        # print(f"MI MODELO: Predicted_result: {row['predicted_result']}. Probabilidades modelo: Local:{row['prob_class_1']:.2f} Empate: {row['prob_class_0']:.2f} Visitante: {row['prob_class_2']:.2f}")
        # print(f"BOOKIE: Probabilidades Bookie: Local:{row['prob_home_bm']:.2f} Empate: {row['prob_draw_bm']:.2f} Visitante: {row['prob_away_bm']:.2f}")
        # print(f"Diferencia de probabilidad sobre resultado predicho entre MI MODELO y BOOKIE: {row['dif_prob_mod_bm']}")

        # Si el modelo esta MAS seguro del resultado predicho que la casa de apuestas
        if row['dif_prob_mod_bm'] >= thr_prob_min:

            # Apuesto al resultado predicho
            result_to_bet = row['predicted_result']
            dif_prob_result_to_bet = row['dif_prob_mod_bm']
            prob_result_to_bet = max(row['prob_class_1'], row['prob_class_0'], row['prob_class_2'])
            odd_to_bet = row['odds_home'] if row['predicted_result'] == 1 else (row['odds_draw'] if row['predicted_result'] == 0 else row['odds_away'])  # Verificada
            strategy = f"dif_prob_mod_bm > {thr_prob_min}"
            # print(f"La diferencia de probabilidad entre mi modelo y la casa de apuesta sobre el resultado predicho por mi modelo es MAYOR al 0%. Resultado a apostar: {result_to_bet} con probabilidad {prob_result_to_bet}")

        # Si el modelo esta MENOS seguro del resultado predicho que la casa de apuestas
        else:
            # Apuesto doble oportunidad sin el resultado predicho
            result_to_bet = -1 if row['predicted_result'] == 1 else (-2 if row['predicted_result'] == 2 else -0)
            dif_prob_result_to_bet = row['dif_prob_mod_bm'] * -1
            prob_result_to_bet = 1 - max(row['prob_class_1'], row['prob_class_0'], row['prob_class_2'])
            odd_to_bet = calculate_odd_double_chance(row, result_to_bet)
            strategy = f"dif_prob_mod_bm < {thr_prob_min}"
            # print(f"La diferencia de probabilidad entre mi modelo y la casa de apuesta sobre el resultado predicho por mi modelo es MENOR al 0%. Resultado a apostar: {result_to_bet} con probabilidad {prob_result_to_bet}")
    
        # Guardo el resultado a apostar
        df.loc[id_match, 'result_to_bet'] = result_to_bet
        df.loc[id_match, 'dif_prob_result_to_bet'] = dif_prob_result_to_bet
        df.loc[id_match, 'prob_result_to_bet'] = prob_result_to_bet
        df.loc[id_match, 'odd_to_bet'] = odd_to_bet
        df.loc[id_match, 'strategy'] = strategy
        # print(result_to_bet, dif_prob_result_to_bet)

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

def determine_stake_to_bet(df, stake_base, type_relation: str = 'equal', p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None):

    df = calculate_multiplier(df, type_relation=type_relation, p1=p1, p2=p2, m=m, b=b)
    df['stake_to_bet'] =  df['multiplier']  * stake_base

    # Ajusto valores de stake_to_bet segun valor minimo y valor maximo
    val_min, val_max = 0, 100  # Evito que el stake a apostar sea mayor al 100% del bank
    func = lambda x: val_min if x < val_min else (val_max if x>val_max else x)
    df['stake_to_bet'] = df['stake_to_bet'].apply(func)
    return df

def calculate_multiplier(df: pd.DataFrame,  type_relation: str = 'equal', p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None):
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

    # Returns
        Dataframe pasado como parametro con nueva columna 'multiplier', el multiplicador para variar el stake.
    """
    # Separo puntos en x e y
    if p1 is not None and p2 is not None:
        x1, y1 = p1
        x2, y2 = p2

    # Linear
    if type_relation == "equal":  
        df['multiplier'] = 1

    elif type_relation == "linear":  # y= m * x + b

        # Si la pendiente no fue pasada como parametro, calculo la pendiente y ordenada al origen
        if m is None:

            m = (y2-y1) / (x2-x1)
            b = y1 - m*x1
        
        df['multiplier'] = df['prob_result_to_bet'] * m + b  #  df['multiplier'] = (df['prob_result_to_bet'] * df['odd_to_bet']) * m + b

    elif type_relation == "poly":  # y = b + b1 * x1 + b2 * x2 + ... + bn * xn # a desarrollar en un futuro
        pass

    elif type_relation == "exponential":  # y=a * b^x

        # Resolver el sistema de ecuaciones
        A = np.array([[1, np.log(x1)], [1, np.log(x2)]])
        b = np.array([np.log(y1), np.log(y2)])

        a, log_b = np.linalg.solve(A, b)

        # Calcular b a partir de su logaritmo
        b = np.exp(log_b)     

        df['multiplier'] = a * (b ** df['prob_result_to_bet'])
        
    return df

def calculate_roi(df: pd.DataFrame, _print: bool = False):
    """
    Calcula ROI obtenido segun las predicciones del modelo y el resultado real de los partidos.

    # Parameters:
        df: Dataframe (DataFrame)
        stake_base: Stake base sobre el cual aplicar el multiplicador para obtener el stake variable. (int)
        _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

    # Returns
        ROI del modelo. (float)
    """
    # Definicion de variables
    dinero_tras_apuestas_ini = 100 # CUIDADO! NO ES sum(df['stake_mod'])
    dinero_tras_apuestas = 100
    n_apuestas = len(df)
    d_rois = {}
    l_rois_partido = [50, 100, 150, 200, 250, 300, 400, 500]
    cont = 0
    df_correct = df[df['acerte'] == 1]   # Filtrar el dataframe solo a las filas donde el modelo predijo correctamente

    # Por partido acertado
    for idx, row in df.iterrows():
        cont += 1
        
        # Defino stake a apostar 
        stake_a_apostar =  dinero_tras_apuestas * row['stake_to_bet'] / 100  # Stake como porcentaje del bank
        df.loc[idx, 'stake_to_bet_%bank'] = stake_a_apostar

        if _print:
            print(f"Stake [% bank]: {row['stake_to_bet']} Stake [$]: {stake_a_apostar}")

        if row['acerte'] == 1:
            # Calculo ingresos por acertar el resultado
            ingresos = stake_a_apostar * row['odd_to_bet']
            ganancia = ingresos - stake_a_apostar
            dinero_tras_apuestas += ganancia
            df.loc[idx, 'G/P'] = ganancia
            if _print:
                print(f"\tAcerte! Cuota ganada: {row['odd_to_bet']} Ingresos: ${ingresos:.0f} --> Dinero tras apuesta: ${dinero_tras_apuestas:.0f}")

        else:
            dinero_tras_apuestas -= stake_a_apostar
            df.loc[idx, 'G/P'] = -1 * stake_a_apostar
            if _print:
                print(f"\tFallé. Cuota apostada: {row['odd_to_bet']} Perdidas: ${stake_a_apostar:.0f} --> Dinero tras apuesta: ${dinero_tras_apuestas:.0f}")

        if cont in l_rois_partido:
            roi_partido = (dinero_tras_apuestas - dinero_tras_apuestas_ini) / dinero_tras_apuestas_ini * 100
            d_rois[f'roi_{cont}'] = roi_partido / cont
            df.loc[idx, 'roi_partido'] = roi_partido / cont
            # print(f"Dinero tras apuestas: {dinero_tras_apuestas} ; Dinero a apostar: {dinero_a_apostar} ; ROI: {roi_partido}")

        df.loc[idx, 'dinero_tras_apuestas'] = dinero_tras_apuestas

        # Si perdi todo el dinero de las apuestas --> ESTO NUNCA DEBERIA SUCEDER AL USAR STAKE COMO % DEL BANK...
        if dinero_tras_apuestas <= 0:
            dinero_tras_apuestas = 0
            raise ValueError("El dinero tras apuestas se hizo negativo y esto no es posible.")

    # Calculo el ROI
    roi = (dinero_tras_apuestas - dinero_tras_apuestas_ini) / dinero_tras_apuestas_ini * 100
    roi_por_partido = roi / n_apuestas  # No quiero el ROI mas alto sino el ROI / partido mas alto
    d_rois['roi'] = roi
    d_rois['roi_por_partido'] = roi_por_partido
    cuota_media_ganada = sum(df_correct['odd_to_bet'] / len(df_correct))
    cuota_media_apostada = sum(df['odd_to_bet'] / len(df))  # --> no tengo la cuota apostada en cada partido... la eestoy calculando en el ciclo for...
    if _print:
        print(f"Cuota media apostada {cuota_media_apostada} ; Cuota media ganada: {cuota_media_ganada} ; Precision minima para ganar dinero: {1/cuota_media_apostada:.2f}")
        print(f"De {df.shape[0]} partidos, acerté {df_correct.shape[0]}. Precision: {len(df_correct) / len(df)}")
        print(f"Cantidad de partidos en los que se apostó: {n_apuestas}, es decir, el {n_apuestas / len(df)*100:.1f}%")
        print(f"\t ROI: {roi:.1f}%. ${dinero_tras_apuestas_ini:.0f} --> ${dinero_tras_apuestas:.0f}")
        print(f"\t ROI por partido: {roi_por_partido:.1f}%")

    return df, d_rois

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
    
    l_cols = ['stake_to_bet', 'multiplier', 'odd_to_bet', 'strategy', 'acerte', 'G/P', 'dinero_tras_apuestas', 'prob_result_to_bet', 'dif_prob_result_to_bet', 'result_to_bet']
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
