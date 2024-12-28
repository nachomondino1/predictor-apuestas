import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from p4_modeling.asses_model import calculate_combined_metric, calculate_roi
from p4_modeling import select_model_for_prod
from utils import directories


class BettingStrategy:

    def __init__(self, country: str = None, iteration_date: str = None, strategy: str = "general", verbose: int = 0):
        self.country = country
        self.iteration_date = iteration_date
        self.strategy = strategy
        self.verbose = verbose

        if self.iteration_date is not None:
            self.initialize_directories()

    def initialize_directories(self):
        
        self.BASE_PATH = f'data/{self.country}/p4_modeling/{self.iteration_date}'
        self.BASE_PATH_sbm = f'data/{self.country}/p4_modeling/{self.iteration_date}/best_models'

        directories.make_directories(l_directorios=[self.BASE_PATH_sbm])
        
    def define_hiperparameters(self):
        """
        Defino hiperparametros de estrategia de apuesta a probar segun si apuesto como la realidad o no.
        """
        if self.strategy == "train":
            l_thr_dif_prob = [-1] 
            d_rectas = {
                'linear': [[10, 0]], 
            }
            l_odd_weight = [0]
            l_lim_sup = [0]

        elif self.strategy == "general":
            l_thr_dif_prob = [-1] # , -0.3] # [-0.5, -0.35, -0.25]  # tengo varios valores porque cambia mucho si el modelo es under o no.
            d_rectas = {
                # "equal": [[(0, 0), (1, 0)]],
                # 'kelly': [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], # le sumo b pues la casa esta desbalanceada y yo no... y muchas veces conviene aunque paguen "poco"
                'linear': [[10, 0], [15, 0], [20, 0], [25, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0], [80, 0]],  # [1, 0], --> para que hay mas dif entre ROIpp de modelos en test..
                # 'exponential': [[(0.33, 4), (1, 10)], [(0.33, 6), (1, 10)], [(0.33, 4), (1, 30)], [(0.33, 2), (1, 30)]] # no entiendo la curva. Se resuelve con matrices.
            }
            l_odd_weight = [0, 1, 2, 4]
            l_lim_sup = [0, 1] # no mas 1.

        elif self.strategy == "all":
            l_thr_dif_prob = [-0.5, -0.3] # [-0.5, -0.35, -0.25]  # tengo varios valores porque cambia mucho si el modelo es under o no.
            d_rectas = {
                "equal": [[(0, 0), (1, 0)]],
                'kelly': [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], # le sumo b pues la casa esta desbalanceada y yo no... y muchas veces conviene aunque paguen "poco"
                'linear': [[10, 0], [15, 0], [20, 0], [25, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0], [80, 0]],  # [1, 0], --> para que hay mas dif entre ROIpp de modelos en test..
                # 'exponential': [[(0.33, 4), (1, 10)], [(0.33, 6), (1, 10)], [(0.33, 4), (1, 30)], [(0.33, 2), (1, 30)]] # no entiendo la curva. Se resuelve con matrices.
            }
            l_odd_weight = [0, 1, 2, 3, 4]
            l_lim_sup = [0, 1]

        elif self.strategy=="general_0":
            l_thr_dif_prob = [-0.5]  # tengo varios valores porque cambia mucho si el modelo es under o no.
            d_rectas = {
                # "equal": [[(0, 0), (1, 0)]],
                'linear': [[10, 0], [15, 0], [20, 0], [25, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0], [80, 0]],  # [1, 0], --> para que hay mas dif entre ROIpp de modelos en test..
                # 'exponential': [[(0.33, 3), (1, 15)], [(0.33, 5), (1, 20)], [(0.33, 5), (1, 30)]]
            }
            l_odd_weight = [0, 1, 2, 4]
            l_lim_sup = [0]

        if self.verbose >= 1:
            logger.info(f"Hiperparametros estrategia de apuesta: \n- Doble oportunidad: {l_thr_dif_prob} \n- Rectas: {d_rectas} \n- Pesos cuotas: {l_odd_weight} \n- Limite para afectar stake con cuotas: {l_lim_sup}")
        
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
                            porc_emergency: float = 0.5,
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
        ''' Creo qu funciona mejor sin esto...
        # Limito caps segun odd_weight (para evitar stake=99 x inflado de stake con cuotas sobretodo cuando odd_weight=4). Esto pasaba en el 437 de GER. Basicamente evito overfitting de hiper de apuesta.
        if odd_weight > 0:
            dif_prob_inf_cap = dif_prob_inf_cap / odd_weight
            dif_prob_sup_cap = dif_prob_sup_cap / odd_weight

            if self.verbose >= 1:
                logger.info(f"dif_prob_result_to_bet capped: [{dif_prob_inf_cap}, {dif_prob_sup_cap}]")
        '''

        filled, rows_not_filled = False, []
        if 'player_emergency_fill' in df.columns:
            filled = True
            rows_player_filled = df[df['player_emergency_fill'] == 1].index
            rows_not_filled = df[df['player_emergency_fill'] != 1].index

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

                if filled and len(rows_not_filled) > 0:
                    # Stake para filas NO rellenadas
                    df.loc[rows_not_filled, 'stake_to_bet'] = (df['prob_result_to_bet'] + np.clip(df['dif_prob_result_to_bet'], dif_prob_inf_cap, dif_prob_sup_cap) * odd_weight) * m + b  # NO usar np.where() pues descarta los valores fuera del rango. En cambio np.clip() los ajusta dentro del rango. # Limita los valores de 'dif_prob_result_to_bet' a un rango de -0.15 a 0.15

                    # Stake para filas rellenadas
                    df.loc[rows_player_filled, 'stake_to_bet'] = (df['prob_result_to_bet'] + np.clip(df['dif_prob_result_to_bet'], dif_prob_inf_cap, 0)) * m + b  # uso lim_sup=0 (no puedo agrandar stake con cuotas) y sin odd_weight para no influenciar en el valor que pueda tomar.
                    
                else:
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
        if ('player_emergency_fill' in df.columns) and (len(rows_player_filled) > 0):

            # Reducir el stake al 50% solo para las filas donde 'emergency_fill' es igual a 1
            df.loc[rows_player_filled, 'stake_to_bet'] *= porc_emergency
            
            # Contar los registros donde 'player_emergency_fill' es igual a 1
            if self.verbose >= 0:
                # Agregar el conteo al warning
                logger.warning(f"Disminución de stakes por copiado de emergencia de variables START y/o SUB en partido. Se afectaron los stakes de {len(rows_player_filled)} de {len(df)} registros.")
            
        # Restringo stake de 0 a 99 (e.g. evito que el stake a apostar sea mayor al 100% del bank)
        val_min, val_max = 0, 99  # 100 no pues sino el bank es negativo.
        func = lambda x: val_min if x < val_min else (val_max if x>val_max else x)
        df['stake_to_bet'] = df['stake_to_bet'].apply(func)

        return df

    def calculate_roi_by_betting_strategy(self, df: pd.DataFrame, roi_weight: float = 0.5):
        """
        Determine the ROI for different betting strategies.

        # Paramaters:
            df: Dataframe con predicciones del modelo, cuotas de la casa de apuestas y el resultado real del partido. (DataFrame)
            strategy: Combinacion de hiperparametos de estrategia de apuesta a probar. (str)

        # Returns
            d_hiper: Diccionario con los hiperparametros de la estrategia de apuesta ganadora. (dict)
            best_df_pred: Predicciones con estrategia de apuesta ganadora. (DataFrame)
            best_d_rois: Diccionario con las metricas (roi, expected_roi y metric) de la estrategia de apuesta ganadora. (dict)
        """
        # Definicion de variables
        records, d_predicciones = [], {}  # Inicializa una lista para acumular los datos
        l_thr_dif_prob, d_rectas, l_odd_weight, l_lim_sup = self.define_hiperparameters()

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
                    
                    if self.verbose >= 2:
                        logger.info(f"{a1} {a2} --> {m} {b} {p1} {p2}")

                    # for normalized in [True, False]:
                    for odd_weight in l_odd_weight: # 0 significa no afectar stake con cuotas.  # 0.5,

                        for lim_sup in l_lim_sup:

                            # Determino stake a apostar segun curva
                            df_aux = self.determine_stake_to_bet(df2, type_relation=key, m=m, b=b, p1=p1, p2=p2, odd_weight=odd_weight, dif_prob_sup_cap=lim_sup, normalized=normalized)

                            # Calculo ROI
                            df_pred, d_metrics = calculate_roi(df_aux)   # (calculate_reality_roi(df_aux)) if strategy == 'reality' else (calculate_roi(df_aux))

                            # Calculo expected ROI
                            df_pred_2, d_expected_roi = calculate_roi(df_aux, name_extension='expected_')     

                            # Concateno datos de ROI y Expected ROI
                            missing_columns = [col for col in df_pred_2.columns if col not in df_pred.columns]
                            df_pred = pd.concat([df_pred, df_pred_2[missing_columns]], axis=1) # Concatenar únicamente las columnas que faltan
                            d_metrics.update(d_expected_roi)

                            # Guardo resultados
                            id_str = f'{prob}_{key}_{m}_{b}_{odd_weight}_{lim_sup}'
                            data = {
                                'id': id_str,
                                'thr_prob_min': prob,
                                'curva': key,
                                'param1': m,
                                'param2': b,
                                'odd_weight': odd_weight,
                                'dif_prob_sup_cap': lim_sup,
                                'normalized': normalized
                            }
                            data.update(d_metrics)
                            records.append(data)
                            d_predicciones.update({id_str: df_pred})
        
        df_bs = pd.DataFrame(records)
        if self.verbose >= 2:
            print(df_bs)
            print(d_predicciones.keys())

        # Determino mejor estrategia
        if self.strategy == "train":
            d_hiper = data
            best_d_rois = d_metrics
            best_df_pred = df_pred
        else:
            d_hiper, best_d_rois, best_df_pred = self.select_best_combination(df_bs, d_predicciones, roi_weight=roi_weight)

        if self.verbose >= 1:
            logger.info(f"Estrategia de apuesta ganadora: {d_hiper}")
            logger.info(f'Metricas: {best_d_rois}.')

        return d_hiper, best_df_pred, best_d_rois

    def select_best_combination(self, df, d_predicciones, roi_weight: float):
        """
        Selecciona la fila con la máxima métrica y retorna su ID.
        También crea un diccionario `d_hiper` basado en dicha fila, excluyendo columnas específicas.

        :param df_bs: DataFrame con la columna 'metric' y una columna identificadora.
        :param id_column: Nombre de la columna que actúa como identificador único.
        :return: (id_max, d_hiper) - ID de la fila seleccionada y el diccionario `d_hiper`.
        """
        name_extension = "_con_ea"
        metric_col = f'metric{name_extension}'

        # Calcular metrica combinada para determinar mejor estrategia
        df = calculate_combined_metric(df, roi_weight=roi_weight, name_extension=name_extension)
        # df.to_excel("/Users/nachomondino/Desktop/prueba.xlsx", index=True)
        
        # Seleccionar mejor estrategia
        if metric_col not in df.columns:
            raise ValueError(f"La columna '{metric_col}' no existe en el DataFrame.")

        # Encontrar la fila con el valor máximo de 'metric'
        row = df.loc[df[metric_col].idxmax()]  # # Convertir a serie
        id_max = row['id']
        if self.verbose >= 2:
            print("Id max", id_max)
            print("Row", row)

        # excluded_columns = {'id', 'roi', 'roi_por_partido', 'expected_roi', 'expected_roi_por_partido', 'metric'}
        excluded_columns = {'id', 'roi', 'roi_por_partido', 'expected_roi', 'expected_roi_por_partido', 'roi_por_partido_norm', 'expected_roi_por_partido_norm', metric_col}

        # Construyo variables a retornar
        d_hiper = row.drop(labels=excluded_columns.intersection(df.columns)).to_dict() # Crear d_hiper excluyendo las columnas específicas
        best_d_rois = row[list(excluded_columns.intersection(df.columns))].to_dict()
        best_df_pred = d_predicciones[id_max]  # Con el id obtengo su df_pred y su df_metrics....
        
        return d_hiper, best_d_rois, best_df_pred

    def define_betting_strategy(self, row, roi_weight: float, with_assess: bool = False):
        """
        Determina la estrategia de apuesta optima para cada modelo.
        """
        # Lista para almacenar los resultados
        logger.info("Paso 5: Definiendo la estrategia de apuesta optima para el modelo seleccionado...")
        results = []

        print(row)
        logger.warning(row)
        n_model, model_name = row.index[0], row['model_name'].values[0]
        if self.verbose >= 1:
            logger.info(f'n_model: {n_model} model_name: {model_name}')

        # Levanto df_predicciones --> Aqui deberia ser capaz de levantar las predicciones sobre los missing tambien y evaluar todo junto (test + missing).             # Falta levantar las predicciones de los missing y concatenerlas (si hubiera) --> no haria falta el assess_models_in_prod.py????
        df_pred = pd.read_excel(f"{self.BASE_PATH}/models/{n_model}__{model_name}_predicciones.xlsx", index_col=0)
        logger.info(df_pred.shape)
        
        '''
        # (opcional) Hacer assess --> Lo haria solo para los pocos modelos que pasan el filtro inicial.
        if with_assess: # Hay que ver si funciona...

            # Recolecto predicciones en ultimos partidos
            df_pred_assess = assess_model_in_prod(self.id_country, n_model, model_name, iteration_date)
            logger.info(df_pred_assess.shape)

            # Concateno df
            len_inic = len(df_pred)
            df_pred = pd.concat([df_pred, df_pred_assess], axis=0)
            logger.info(f"Se concatenó las predicciones de los partidos missing: {len_inic} --> {len(df_pred)}")

            if self.verbose >= 2:
                # Exporto predicciones concatenado
                df_pred.to_excel(f"{self.path_predic}/predicciones_raw_{n_model}_{model_name}.xlsx") # df_predicciones???
        '''

        # Determino la mejor estrategia de apuesta
        if self.verbose >= 1:
            logger.info("Calculando la mejor estrategia de apuesta...")
        d_hiper, best_df_pred, best_d_rois = self.calculate_roi_by_betting_strategy(df_pred, roi_weight=roi_weight)

        # Concateno datos y guardo
        d_ct = {**d_hiper, **best_d_rois}  # Combinar los dos diccionarios
        d_ct['model_name'] = model_name
        d_ct['n_model'] = n_model

        # Añadir el resultado al DataFrame final
        results.append(d_ct)

        # Convertir la lista de resultados en un DataFrame
        df_final = pd.DataFrame(results)
        df_final.set_index('n_model', inplace=True)
        
        if self.verbose >= 0:
            logger.critical(f"La mejor estrategia de apuesta: {d_hiper}")

        # Exportar el DataFrame final a un archivo Excel
        best_df_pred.to_excel(f'{self.BASE_PATH_sbm}/predicciones_{n_model}_{model_name}.xlsx')
        df_final.to_excel(f'{self.BASE_PATH_sbm}/df_strategy.xlsx', index=True)
        return df_final


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    
    # Defino parametros
    id_country = 77

    # Defino hiperparametros
    select_best_model = True
    with_assess = False
    strategy = 'general'

    # Defino variables
    d_countries = {6: ["argentina", '2024-12-05'], 48: ["england", '2024-12-23'], 55: ["france", '2024-12-26'], 59: ["germany", '2024-12-26'], 77: ["italy", '2024-12-23'], 148: ["spain", '2024-12-25'], 167: ["usa", '2024-12-05']}
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]
    bs = BettingStrategy(country, iteration_date, strategy=strategy)


    # Actualizar df_prediccion test con missing?
    # ... 

    # Seleccion del modelo
    if select_best_model:
        perc_cutoff = 0.05

        # Creo objeto de clase select_best_model
        sbm = select_model_for_prod.SelectBestModel(id_country=id_country, iteration_date=iteration_date)

        # Obtengo listado de todos los modelos entrenados
        df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')

        # Selecciono el mejor modelo
        row = sbm.main(df_ite, perc_cutoff=perc_cutoff)
        roi_weight_strategy = sbm.roi_weight

    else:
        roi_weight_strategy = 0.5
        df = pd.read_excel(f'{bs.BASE_PATH_sbm}/df_p4.xlsx', index_col=0)
        row = df.head(1) # Selecciono la primera fila
        logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]:.1f}")

    # Determinar estrategia de apuesta optima para el modelo seleccionado
    bs.define_betting_strategy(row=row, roi_weight=roi_weight_strategy, with_assess=with_assess)


    '''
    # Obtengo listado de todos los modelos entrenados
    # df_iteration = pd.read_excel(f'/df_iteration.xlsx')

    # Obtengo predicciones del modelo
    df_predicciones = pd.read_excel(f"{BASE_PATH}/models/{n_model}__{model_name}_predicciones.xlsx")

    # Construyo variable expected result
    d_hiper, df_predicciones, d_roi = bs.calculate_roi_by_betting_strategy(df_predicciones, strategy='general')
    print(d_roi)
    '''