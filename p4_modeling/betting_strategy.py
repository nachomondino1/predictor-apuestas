import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from p4_modeling.asses_model import calculate_combined_metric, calculate_roi
from utils import directories
from itertools import product


class BettingStrategy:

    def __init__(self, country: str = None, iteration_date: str = None, d_paths: dict = None, verbose: int = 0):
        self.country = country
        self.iteration_date = iteration_date
        self.verbose = verbose
        self.d_paths = d_paths
        self.initialize_directories()

    def initialize_directories(self):
        
        # Si se quiere guardar los datos:
        if self.iteration_date is not None:

            # Si no pasaron d_paths
            if self.d_paths is None:

                self.BASE_PATH = f'data/{self.country}/p4_modeling/{self.iteration_date}'
                self.BASE_PATH_sbm = f'data/{self.country}/p4_modeling/{self.iteration_date}/best_model'
                directories.make_directories(l_directorios=[self.BASE_PATH_sbm])
        
            else:
                # Si pasaron d_paths
                self.BASE_PATH = self.d_paths['base_path']
                self.BASE_PATH_sbm = self.d_paths['base_path_sbm']

    def define_hiperparameters(self, strategy):
        """
        Defino hiperparametros de estrategia de apuesta a probar segun si apuesto como la realidad o no.
        """
        if strategy == "train":
            dic = {
                'prob_dp': [-1],
                'curva': ['linear'],
                'm': [10],
                'b': [0],
                'odd_weight': [0],
                'lim_sup': [0]
            }
        
        elif strategy == "no_odds":
            dic = {
                'prob_dp': [-1], #  -0.5, -0.35, -0.25
                'curva': ['linear'],
                'm': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250],
                'b': [0],
                'odd_weight': [0],
                'lim_sup': [0]
            }

        elif strategy == "general":
            dic = {
                'prob_dp': [-1],  # tengo varios valores porque cambia mucho si el modelo es under o no.
                'curva': ['linear'], # ['linear',  'kelly'],  #
                'm': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250],
                'b': [0, -10, -20],
                'odd_weight': [0, 1, 2],
                'lim_sup': [0] # no dar la posibilidad de inflar
            }

        elif strategy == "all":
               
            dic = {
                'l_thr_dif_prob': [-0.5, -0.3], # [-0.5, -0.35, -0.25]  # tengo varios valores porque cambia mucho si el modelo es under o no.

                'curva': ['linear'], # 'equal', 'kelly', 'exponential'
                'm': [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                'b': [0],
                'odd_weight': [0, 1, 2, 3, 4],
                'lim_sup': [0, 1]
            }

        if self.verbose >= 0:
            logger.info(f"Hiperparametros estrategia de apuesta: {dic}")
            # logger.info(f"Hiperparametros estrategia de apuesta: \n- Doble oportunidad: {l_thr_dif_prob} \n- Rectas: {d_rectas} \n- Pesos cuotas: {l_odd_weight} \n- Limite para afectar stake con cuotas: {l_lim_sup}")
    
        return dic

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

    # RESULT TO BET
    def determine_result_to_bet(self, df: pd.DataFrame, thr_prob_min):
        """
        Determina el/los resultado/s a apostar (no necesariamente coincide con el resultado predicho).
        
        # Parameters:
            df: Dataframe con probabilidades de mi modelo y con predicciones y cuotas de la casa de apuestas. (DataFrame)
            _print: True para imprimir por pantalla el procesamiento de la funcion, en caso contrario, False. (bool)

        # Returns:
            Dataframe pasado como parametro con resultado a apostar, la cuota a apostar, la estrategia utiilizada y la probabilidad del resultado al que se apuesta. (DataFrame)
        """
        df = df.copy()  # Crea una copia explícita del DataFrame antes de modificarlo (evita warnings por no saber si trabajas en un vista o en una copia)

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

    # STAKE TO BET
    def determine_stake_to_bet(self, 
                               df, type_relation: str = 'equal',                                                                            # Estrategia
                               p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None,                                   # Puntos de rectas
                               odd_weight: float = 1, dif_prob_inf_cap: int = -1, dif_prob_sup_cap: int = 1,      # Cuotas en stake
                               porc_emergency: float = 0.5                                                                                  # Disminucion por relleno de emergencia
                               ):
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

        Mejoras:
            - Separar variacion de stake por cuotas en otra funcion. 
        """
        # Limito caps segun odd_weight (para evitar stake=99 x inflado de stake con cuotas sobretodo cuando odd_weight=4). Esto pasaba en el 437 de GER. Basicamente evito overfitting de hiper de apuesta.
        if odd_weight > 0:
            dif_prob_inf_cap = dif_prob_inf_cap / odd_weight
            dif_prob_sup_cap = dif_prob_sup_cap / odd_weight

            if self.verbose >= 1:
                logger.info(f"dif_prob_result_to_bet capped: [{dif_prob_inf_cap}, {dif_prob_sup_cap}]")
        
        # Separo puntos en x e y
        if p1 is not None and p2 is not None:
            x1, y1 = p1
            x2, y2 = p2

        # Si la pendiente no fue pasada como parametro, calculo la pendiente y ordenada al origen
        if m is None:
            m = (y2-y1) / (x2-x1)
            b = y1 - m*x1

        # STRATEGY: EQUAL
        if type_relation == "equal":  
            df['stake_to_bet'] = 1

        # STRATEGY: KELLY
        elif type_relation == 'kelly':

            # df['stake_to_bet'] = ((df['prob_result_to_bet'] + b) - (1 - df['prob_result_to_bet'])) * 100 / (df['odd_to_bet'] - 1)  #  (= (((df['odd_to_bet'] - 1) * df['prob_result_to_bet'] + b) - (1 - df['prob_result_to_bet'])) / (df['odd_to_bet'] - 1) * 100)
            df['stake_to_bet'] = ((df['odd_to_bet'] - 1) * df['prob_result_to_bet'] - (1 - df['prob_result_to_bet'])) * 100 / (df['odd_to_bet'] - 1) 

            # Normalizo stake  # si o si sino el stake es negativo.
            p_min, p_max = 0, m
            df = self.normalize_stake(df, p_min, p_max, m) 
            
        # STRATEGY: LINEAR
        elif type_relation == "linear": # Vario stake con prob_result_to_bet y cuotas de la casa

            if dif_prob_inf_cap != dif_prob_sup_cap:
                df['stake_to_bet'] = (df['prob_result_to_bet'] + np.clip(df['dif_prob_result_to_bet'], dif_prob_inf_cap, dif_prob_sup_cap) * odd_weight) * m + b  # NO usar np.where() pues descarta los valores fuera del rango. En cambio np.clip() los ajusta dentro del rango. # Limita los valores de 'dif_prob_result_to_bet' a un rango de -0.15 a 0.15
            else:
                df['stake_to_bet'] = df['prob_result_to_bet'] * m + b
            
            # Puntos para normalizar --> no tiene mucho sentido. Para eso esta exponential (modificar mas el stake ante un menor cambio en proba). Incluso con el b de linear tambien puedo lograr algo parecido.
            # p_min, p_max = (0.55 * m + b), (0.7 * m + b) # El punto min usa un stake de m_to_bet / 2. Si queres que prob=0.33 use un stake mas bajo, no tiene sentido usarlo como p_min.

        # STRATEGY: POLY
        elif type_relation == "poly":  # y = b + b1 * x1 + b2 * x2 + ... + bn * xn # a desarrollar en un futuro
            pass

        # STRATEGY: EXPONENTIAL
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
        
        # Disminuyo stake por rellenado de emergencia
        df = self.stake_reduction_emergency_fill(df, porc_emergency=porc_emergency)

        # Limito stake a entre 0 y 100 
        df = self.cap_stake(df)
        return df
    
    def cap_stake(self, df):
        # Restringo stake de 0 a 99 (e.g. evito que el stake a apostar sea mayor al 100% del bank)
        val_min, val_max = 0, 99  # 100 no pues sino el bank es negativo.
        func = lambda x: val_min if x < val_min else (val_max if x>val_max else x)
        df.loc[:, 'stake_to_bet'] = df['stake_to_bet'].apply(func)         # df['stake_to_bet'] = df['stake_to_bet'].apply(func)
        return df
    
    def normalize_stake(self, df, p_min, p_max, m):

        # Capa de funcion sigmoide (creo que sirve nada mas para Kelly. Para linear seria muy parecico a la variacion exponencial puesto que buscas agrandar las diferencias entre probas de 0.33 y 1)
        df = df.rename(columns={'stake_to_bet': 'stake_to_bet_raw'})

        # Normalización: Escalar los valores de kelly_raw entre 0 y 1
        df['stake_to_bet_norm'] = (df['stake_to_bet_raw'] - p_min) / (p_max - p_min)

        # Usar la función sigmoide para reducir la variabilidad y escalar entre 0 y 30
        num = m if m > 0 else 1
        df['stake_to_bet'] = num / (1 + np.exp(-df['stake_to_bet_norm']))
        df = df.drop(['stake_to_bet_raw'], axis=1)
        return df
        
    def stake_reduction_emergency_fill(self, df, porc_emergency: float = 0.5):

        # Si existen la columna 'emergency_fill' (pues para X_test no existe. Es solo para stakes en produccion). --> Ver si falla cuando hago main_best_model.py (ni deberia entrar)
        if ('player_emergency_fill' in df.columns):
            
            rows_player_filled = df[df['player_emergency_fill'] == 1].index

            if len(rows_player_filled) > 0:

                # Reducir el stake al 50% solo para las filas donde 'emergency_fill' es igual a 1
                df.loc[rows_player_filled, 'stake_to_bet'] *= porc_emergency
                
                # Contar los registros donde 'player_emergency_fill' es igual a 1
                if self.verbose >= 2:
                    # Agregar el conteo al warning
                    logger.warning(f"Disminución de stakes por copiado de emergencia de variables START y/o SUB en partido. Se afectaron los stakes de {len(rows_player_filled)} de {len(df)} registros.")
        return df

    ## Determinar la mejor estrategia
    def calculate_roi_in_combinations(self, df, d_params):
        """
        Determinar un ROI para cada combinacion de hiperparametros de la estrategia de apuesta. 

        # Parameters
            df: El df puede ser de un solo resultado y por ende, definir una estrategia para dicho resultado...

        Mejoras:
            - Estrategia por resultado...
        """
        # Definicion de variables
        d_predic, d_hiper, d_metricas = {}, {}, {}

        # Eliminate rows with NaN odds or missing predictions
        df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away', 'predicted_result'])
        df = self.calculate_dif_proba_in_predicted_result(df)  # Calculo la diferencia de probabilidad entre el modelo y la casa de apuestas para el resultado predicho por el modelo (Columna 'dif_prob_mod_bm')

        # Generar combinaciones de parámetros automáticamente
        param_combinations = list(product(*d_params.values()))
        if self.verbose >= 1:
            print(f'Las {len(param_combinations)} combinaciones: {param_combinations}')
        cont = 0
        
        # Iterar sobre cada combinación
        for params in param_combinations:
            cont += 1

            # Emparejar cada parámetro con su nombre desde `param_grid`
            param_dict = dict(zip(d_params.keys(), params))
            if self.verbose >= 1:
                print(f"\nNº Combinacion: {cont}")
                print(param_dict)

            if param_dict['odd_weight'] == 0 and param_dict['lim_sup'] != 0:
                if self.verbose >= 1:
                    logger.warning("Se evitó calcular combinacion porque si odd_weight es 0, no tiene sentido lim_sup != 0.")
                continue

            # Aplicar estrategia a df_pred
            df_aux = self.apply_strategy(df, param_dict, prod=False)

            # Calculo de metricas (ROI y expected roi)
            df_pred_with_metrics, d_metrics = self.calculate_roi_in_combination(df_aux)
            if self.verbose >= 0:
                print(f"Params: {params} \n Metrics: {d_metrics} \n")
                # df_pred_with_metrics.to_excel("/Users/nachomondino/Desktop/bs.xlsx")

            # Guardo resultados
            d_predic[cont] = df_pred_with_metrics
            d_hiper[cont] = param_dict
            d_metricas[cont] = d_metrics

        if len(param_combinations) == 1: # strategy == 'train':
            d_predic, d_hiper, d_metricas = df_pred_with_metrics, param_dict, d_metrics

        return d_predic, d_hiper, d_metricas

    def apply_strategy_by_result(self, df, df_hiper):
        """
        Es para usar en produccion.

        df_hiper: Parametros a probar por result
        """
        df_comp = pd.DataFrame()
        print(df.shape)

        # Por resultado
        for pred in [1, 0, 2]:
            print(f"Resultado: {pred}")

            df_pred = df[df['predicted_result'] == pred] 
            print(df_pred.shape)

            # Si no hay registros falla...
            if len(df_pred) > 0:
                row_pred = df_hiper.loc[pred]
                prob = row_pred['prob_dp']
                curva = row_pred['curva']
                m = row_pred['m']
                b = row_pred['b']
                odd_weight = row_pred['odd_weight']
                lim_sup = row_pred['lim_sup']
        
                # Determino result to bet
                df_pred = self.determine_result_to_bet(df_pred, thr_prob_min=prob)

                # Determino stake to bet
                d_params_stake = {'type_relation': curva, 'm': m, 'b': b}
                d_params_odds = {'odd_weight': odd_weight, 'dif_prob_sup_cap': lim_sup}
                df_pred = self.determine_stake_to_bet(df_pred, **d_params_stake, **d_params_odds)

                df_comp = pd.concat([df_comp, df_pred], axis=0)

            else:
                logger.warning(f"No hay partidos para el resultado {pred}, por lo que, no se aplica la estrategia a dicho resultado.")

        return df_comp

    def apply_strategy(self, df, param_dict, prod: bool = True):
        
        if len(df) == 0:
            logger.error("El dataframe a aplicar la estrategia esta vacio...")
            raise ValueError
        
        # Determino result to bet
        df = self.determine_result_to_bet(df, thr_prob_min=param_dict['prob_dp'])
  
        # Determino acierto de prediccion
        if not prod:
            df = self.determine_winning_bets(df)
            df = self.determine_winning_bets(df, name_extension='expected_')

        # Determino stake to bet
        d_params_stake = {'type_relation': param_dict['curva'], 'm': param_dict['m'], 'b': param_dict['b']}
        d_params_odds = {'odd_weight': param_dict['odd_weight'], 'dif_prob_sup_cap': param_dict['lim_sup']}
        df = self.determine_stake_to_bet(df, **d_params_stake, **d_params_odds)
        return df
    
    def calculate_roi_in_combination(self, df):
   
        # Calculo ROI
        df_pred, d_metrics = calculate_roi(df)

        # Calculo expected ROI
        df_pred_2, d_expected_roi = calculate_roi(df, name_extension='expected_')     

        # Concateno datos de ROI y Expected ROI
        missing_columns = [col for col in df_pred_2.columns if col not in df_pred.columns]
        df_pred = pd.concat([df_pred, df_pred_2[missing_columns]], axis=1) # Concatenar únicamente las columnas que faltan
        d_metrics.update(d_expected_roi)

        return df_pred, d_metrics

    def select_best_parameters(self, data, verbose: int = 0):
        
        name_extension = "_con_ea"
        metric_col = f'metric{name_extension}'

        # df = pd.DataFrame(data=data)
        df = pd.DataFrame.from_dict(data, orient='index')
        if self.verbose >= 1:
            print(df)

        # Calcular metrica combinada para determinar mejor estrategia
        df = calculate_combined_metric(df, l_metrics=['roi_por_partido'], l_weights=[1], name_extension=name_extension)
        if verbose >= 0:
            df.to_excel("/Users/nachomondino/Desktop/prueba.xlsx", index=True)
        
        # Seleccionar mejor estrategia
        if metric_col not in df.columns:
            raise ValueError(f"La columna '{metric_col}' no existe en el DataFrame.")

        # Encontrar la fila con el valor máximo de 'metric'
        n_comb = df[metric_col].idxmax()
        if self.verbose >= 1:
            logger.critical(f"Nº combination: {n_comb}")
            row = df.loc[n_comb]  # # Convertir a serie
            print("Row", row)

        return n_comb
         
    # Main
    def define_model_betting_strategy_by_result(self, df_pred, d_params: dict = None, verbose: int = 0):
        """
        Determina la estrategia de apuesta optima para un modelo.

        Mejoras:
            - simplificar codigo cuando hago el if by_result
        """
        # Lista para almacenar los resultados
        logger.info("Definiendo la estrategia de apuesta optima para el modelo...")
        d_metrics_res, d_hiper_res = {}, {}
        best_df_pred = pd.DataFrame()

        if d_params is None:
            d_params = self.define_hiperparameters(strategy='general')

        # Por resultado
        for pred in [1, 0, 2]:

            # Filtro predicciones por result
            df_result = df_pred[df_pred['predicted_result'] == pred] 
            if verbose >= 0:
                logger.info(f"Resultado: {pred}. Shape {df_result.shape}")

            # Si hay predicciones del modelo para ese result
            if len(df_result) > 0:
                # Calculo roi por cada set de hiper de apuesta
                d_predic, d_hiper, d_metricas = self.calculate_roi_in_combinations(df_result, d_params=d_params)

                # Determinar mejor estrategia para el resultado            
                n_comb = self.select_best_parameters(d_metricas)
            
                if pred == 0 and d_params['lim_sup'] == [0]:  # El empate no cambia con las cuotas y tira error el select porque todas las combinaciones tienen el mismo ROI
                    n_comb = 1
                   
                # Guardo datos
                d_hiper_res[pred] = d_hiper[n_comb]
                d_metrics_res[pred] = d_metricas[n_comb]
                best_df_pred = pd.concat([best_df_pred, d_predic[n_comb]], axis=0)

            else:
                logger.warning(f"No hay predicciones con el resultado {pred} (o sea, el modelo no lo predice). Asigno strategy de train.")
                d_hiper_res[pred] = {
                    'prob_dp': -1,
                    'curva': 'linear',
                    'm': 10,
                    'b': 0,
                    'odd_weight': 0,
                    'lim_sup': 0
                    }
                d_metrics_res[pred] = {}

        # Concateno datos y guardo
        df_metrics = pd.DataFrame.from_dict(d_metrics_res, orient='index')
        df_hiper = pd.DataFrame.from_dict(d_hiper_res, orient='index')
        df_final = pd.concat([df_hiper, df_metrics], axis=1)

        # Calculo G/P por resultado (para comparar con G/P sin estrategia)
        gp_total = sum(df_final['roi'])
        df_final.loc[1, '%_G/P'] = df_final.loc[1, 'roi'] / gp_total * 100
        df_final.loc[0, '%_G/P'] = df_final.loc[0, 'roi'] / gp_total * 100
        df_final.loc[2, '%_G/P'] = df_final.loc[2, 'roi'] / gp_total * 100

        if self.verbose >= 1:
            logger.critical(f"La mejor estrategia de apuesta: {d_hiper}")

        # Exportar el DataFrame final a un archivo Excel
        return df_final, best_df_pred

    def define_model_betting_strategy_general(self, df_pred, d_params: dict = None):
        """
        Determina la estrategia de apuesta optima para un modelo. Todos los resultados con la misma estrategia
        """
        # Lista para almacenar los resultados
        logger.info("Definiendo la estrategia de apuesta optima para el modelo...")
        results = []

        # Defino hiperparametros a probar
        if d_params is None:
            d_params = self.define_hiperparameters(strategy='general')

        # Determino ROI por combinacion de hiper de apuesta
        d_predic, d_hiper, d_metricas = self.calculate_roi_in_combinations(df_pred, d_params=d_params)
        n_comb = self.select_best_parameters(d_metricas)

        d_hiper = d_hiper[n_comb]
        best_df_pred = d_predic[n_comb]
        best_d_rois = d_metricas[n_comb]

        # Concateno datos y guardo
        ## Combinar los dos diccionarios
        d_ct = {**d_hiper, **best_d_rois}  

        ## Añadir el resultado al DataFrame final
        results.append(d_ct)

        ## Convertir la lista de resultados en un DataFrame
        df_final = pd.DataFrame(results)
        # df_final.set_index('n_model', inplace=True)

        if self.verbose >= 1:
            logger.critical(f"La mejor estrategia de apuesta: {d_hiper}")

        # Exportar el DataFrame final a un archivo Excel
        return df_final, best_df_pred

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Defino parametros
    id_country = 77
    n_model = 1339
    model_name = 'LogisticRegression'
    strategy = 'general'

    # Defino hiperparametros
    asssess_models_in_prod = False
    avoid_assess = True
    select_best_model = True

    # Defino variables
    d_countries = {
        6: ["argentina", '2024-12-05'], 
        48: ["england", '2024-12-23'],  # '2024-12-23'  '2025-01-02'
        55: ["france", '2024-12-26'], 
        59: ["germany", '2024-12-26'], 
        77: ["italy", '2024-12-23'], 
        # 77: ["italy", '2025-01-01'],
        148: ["spain", '2024-12-25'], 
        167: ["usa", '2024-12-05']
        }
    country = d_countries[id_country][0]
    iteration_date = d_countries[id_country][1]
    

    bs = BettingStrategy(country, iteration_date)

    # Levanto el df del ultimo paso de la seleccion y obtengo el mejor modelo
    df = pd.read_excel(f'{bs.BASE_PATH_sbm}/2_select_model/df_selected_model.xlsx', index_col=0)
    row = df.head(1) # Selecciono la primera fila
    logger.critical(f"El mejor modelo es el {row.index[0]} con ROIpp {row['roi_por_partido'].values[0]:.1f}")

    from main_select_model import read_predicciones
    df_pred = read_predicciones(n_model=n_model, model_name=model_name, assess=True, )

    df = bs.define_model_betting_strategy_by_result(df_pred=df_pred, strategy="general", roi_weight=0.755)
