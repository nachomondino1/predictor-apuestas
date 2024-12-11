import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from p4_modeling import asses_model


class BettingStrategy:

    def __init__(self):
        # self.child_driver = self.driver
        pass

    def calculate_roi_by_betting_strategy(self, df: pd.DataFrame, strategy: str = "train", save_strategy: bool = True, verbose: int = 0):
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
        l_thr_dif_prob, d_rectas, l_odd_weight, l_lim_sup = self.define_hiperparameters(strategy)
        best_roi = -100000
        if verbose >= 1:
            logger.info(f"Calculating ROI...")
            logger.info(f"Hiperparametros estrategia de apuesta: \n- Doble oportunidad: {l_thr_dif_prob} \n- Rectas: {d_rectas}")
        
        # Eliminate rows with NaN odds or missing predictions
        df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away', 'predicted_result'])
        df = self.calculate_dif_proba_in_predicted_result(df)  # Calculo la diferencia de probabilidad entre el modelo y la casa de apuestas para el resultado predicho por el modelo (Columna 'dif_prob_mod_bm')

        # Por combinacion de hiperparametros 
        for prob in l_thr_dif_prob:

            # Determinamos el resultado a apostar (no necesariamente el resultado predicho)
            df2 = df.copy()  # esto parece boludo pero es clave sino df2 se le agrega las columnas de variacion de stake y los rdos son falsos...
            df2 = self.determine_result_to_bet(df2, thr_prob_min=prob)
            df2 = self.determine_winning_bets(df2)
            df2 = self.determine_winning_bets(df2, name_extension='expected_')

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
                    for odd_weight in l_odd_weight: # 0 significa no afectar stake con cuotas.  # 0.5,

                        for lim_sup in l_lim_sup:

                            # Determino stake a apostar segun curva
                            df_aux = self.determine_stake_to_bet(df2, type_relation=key, m=m, b=b, p1=p1, p2=p2, odd_weight=odd_weight, dif_prob_sup_cap=lim_sup, normalized=normalized)

                            # Calculo ROI
                            df_pred, d_metrics = asses_model.calculate_roi(df_aux)   # (calculate_reality_roi(df_aux)) if strategy == 'reality' else (calculate_roi(df_aux))
                            roi = d_metrics['roi_por_partido']  # d_metrics['roi_por_partido_r'] if strategy == 'reality' else d_metrics['roi_por_partido']       

                            # Calculo expected ROI
                            df_pred_2, d_expected_roi = asses_model.calculate_roi(df_aux, name_extension='expected_')     
                            missing_columns = [col for col in df_pred_2.columns if col not in df_pred.columns]
                            df_pred = pd.concat([df_pred, df_pred_2[missing_columns]], axis=1) # Concatenar únicamente las columnas que faltan
                            d_metrics.update(d_expected_roi)
                            # logger.info(f"Porcentaje de cuota {porc_cuota} --> ROI: {roi*100:.0f}")

                            # Verificación si es el mejor ROI y actualización en una sola línea
                            if roi > best_roi:
                                best_roi = roi
                                best_df_pred = df_pred.copy()
                                best_d_rois = d_metrics

                                if save_strategy:
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

    def define_hiperparameters(self, strategy):
        """
        Defino hiperparametros de estrategia de apuesta a probar segun si apuesto como la realidad o no.
        """
        if strategy == "train":
            l_thr_dif_prob = [-1] 
            d_rectas = {
                'linear': [[10, 0]], 
            }
            l_odd_weight = [0]
            l_lim_sup = [0]

        elif strategy == "general":
            l_thr_dif_prob = [-0.5, -0.3] # [-0.5, -0.35, -0.25]  # tengo varios valores porque cambia mucho si el modelo es under o no.
            d_rectas = {
                # "equal": [[(0, 0), (1, 0)]],
                # 'kelly': [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], # le sumo b pues la casa esta desbalanceada y yo no... y muchas veces conviene aunque paguen "poco"
                'linear': [[10, 0], [15, 0], [20, 0], [25, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0], [80, 0]],  # [1, 0], --> para que hay mas dif entre ROIpp de modelos en test..
                # 'exponential': [[(0.33, 4), (1, 10)], [(0.33, 6), (1, 10)], [(0.33, 4), (1, 30)], [(0.33, 2), (1, 30)]] # no entiendo la curva. Se resuelve con matrices.
            }
            l_odd_weight = [0, 1, 2, 4]
            l_lim_sup = [0, 1]

        elif strategy=="reality":
            l_thr_dif_prob = [-0.5, -0.35, -0.25]  # tengo varios valores porque cambia mucho si el modelo es under o no.
            d_rectas = {
                "equal": [[(0, 0), (1, 0)]],
                'linear': [[1, 0], [5, 0], [10, 0], [15, 0],[20, 0], [25, 0],[30, 0], [40, 0]],
                # 'exponential': [[(0.33, 3), (1, 15)], [(0.33, 5), (1, 20)], [(0.33, 5), (1, 30)]]
            }
        
        return l_thr_dif_prob, d_rectas, l_odd_weight, l_lim_sup

    def calculate_dif_proba_in_predicted_result(self, df: pd.DataFrame):
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

    def determine_result_to_bet(self, df: pd.DataFrame, thr_prob_min):
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
                odd_to_bet = self.calculate_odd_double_chance(row, result_to_bet)
                strategy = f"dif_prob_mod_bm < {thr_prob_min}"

            # Guardo el resultado a apostar
            df.loc[id_match, 'result_to_bet'] = result_to_bet
            df.loc[id_match, 'dif_prob_result_to_bet'] = dif_prob_result_to_bet
            df.loc[id_match, 'prob_result_to_bet'] = prob_result_to_bet
            df.loc[id_match, 'odd_to_bet'] = odd_to_bet
            df.loc[id_match, 'strategy'] = strategy

        return df

    def calculate_odd_double_chance(self, row, result_to_bet):
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

    def determine_winning_bets(self, df: pd.DataFrame, name_extension=''):
        """
        Determina si el resultado apostado fue el resultado real del partido o no.
        
        # Parameters
            df: Dataframe con partidos en los que se indica tanto el resultado a apostar como el resultado real del partido.

        # Returns
            Dataframe pasado como parametro con una nueva columna, 'acerte' indicando si se acertó el resultado apostado o no.
        """
        # inicializo columna "acerte"
        var_result=f'{name_extension}result'
        var_acerte=f'{name_extension}acerte'

        df[var_acerte] = 0

        # Por partido
        for id_match, row in df.iterrows():

            # Si el resultado a apostar es Home, Draw o Away
            if row['result_to_bet'] >= 0:
                if row[var_result] == row['result_to_bet']:
                    df.loc[id_match, var_acerte] = 1

            # Si el resultado a apostar es doble oportunidad sin Home
            elif row['result_to_bet'] == -1:
                if (row[var_result] == 0) or (row[var_result] == 2):
                    df.loc[id_match, var_acerte] = 1

            # Si el resultado a apostar es doble oportunidad sin Away
            elif row['result_to_bet'] == -2:
                if (row[var_result] == 0) or (row[var_result] == 1):
                    df.loc[id_match, var_acerte] = 1
            
            # Si el resultado a apostar es doble oportunidad sin Draw
            elif row['result_to_bet'] == -0:
                if (row[var_result] == 2) or (row[var_result] == 1):
                    df.loc[id_match, var_acerte] = 1

            # Si fallo la prediccion
            else:
                df.loc[id_match, var_acerte] = np.nan

        # Imprimo warning si supuestamente acerté el 100% de partidos
        if len(df[df[var_acerte]==1]) == len(df):
            logger.warning(f"Considera que acertó todos los partidos (es decir, 100% de precision). Es muy probable que no este filtrando bien los partidos que acierta de los que no.")

        return df

    def determine_stake_to_bet(self, df, type_relation: str = 'equal', p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None, 
                            porc_emergency: float = 0.75,
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
            df.loc[df['emergency_fill'] == 1, 'stake_to_bet'] *= porc_emergency

        # Restringo stake de 0 a 99 (e.g. evito que el stake a apostar sea mayor al 100% del bank)
        val_min, val_max = 0, 99  # 100 no pues sino el bank es negativo.
        func = lambda x: val_min if x < val_min else (val_max if x>val_max else x)
        df['stake_to_bet'] = df['stake_to_bet'].apply(func)

        return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 148
    iteration_date = '2024-12-10'
    n_models = 20  # Ver cuantos. suma > 0? Pareto?

    # Defino variables
    bs = BettingStrategy()
    d_countries = {6: "argentina", 48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa"}
    country = d_countries[id_country]

    # Obtengo listado de todos los modelos entrenados
    df_iteration = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

    # Determino metrica para seleccionar modelos
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()    # Crear el escalador
    columns_to_normalize = ['roi_por_partido', 'expected_roi_por_partido']    # Seleccionar las columnas a normalizar
    df_iteration[columns_to_normalize] = scaler.fit_transform(df_iteration[columns_to_normalize])    # Normalizar las columnas
    df_iteration['metric'] = df_iteration['roi_por_partido'] + df_iteration['expected_roi_por_partido']    # Sumar las columnas normalizadas

    # Selecciono los mejors n modelos segun roi_por_partido
    df_iteration = df_iteration.sort_values(by='metric', ascending=False)  # Ordenar por roi por partido decreciente
    df_iteration_filt = df_iteration.head(n_models) # Seleccionar los primeros n registros

    # Por modelo
    for idx, row in df_iteration_filt.iterrows():

        n_model = row['n_iteration']
        model_name = row['model_name']

        # Levanto df_predicciones
        df = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx")

        # Determino la mejor estrategia de apuesta
        best_df_pred, best_d_rois = bs.calculate_roi_by_betting_strategy(df, strategy = "general", save_strategy=True)

        # Guardar los mejores hiperparametros de apuesta en df_best_models.xlsx
        # best_d_rois['thr_prob_min_best'] # 'curva', 'param1', 'param2' 'normalized', 'odd_weight', 'dif_prob_sup_cap'

        # Exporto datos
        best_df_pred.to_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration_with_strategy.xlsx')