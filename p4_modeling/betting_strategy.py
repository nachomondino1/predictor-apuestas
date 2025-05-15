import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from p3_data_preparation.format_data import value_nan_to_none
from p3_data_preparation.construct_data import determine_expected_result
from p4_modeling.asses_model import calculate_roi, determine_roi, drop_old_metrics, normalize_column
from utils import directories
import datetime
from itertools import product
from tqdm import tqdm


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

    # HIPER SPACE
    def define_hiperparameters(self, strategy, vary_dp: bool = False, vary_k: bool = False, val_min: int = 10, val_max: int = 100, step_m: int = 10):
        """
        Defino hiperparametros de estrategia de apuesta a probar segun si apuesto como la realidad o no.

        Mejoras:
            - lista de estrategias (e.g. kelly, linear, etc)
        """
        list_dp = [None, -1, -0.75, -0.5, -0.25] if vary_dp else [None] 
        list_m = list(range(val_min, val_max + 1, step_m))
        list_strat = strategy if isinstance(strategy, list) else [strategy]
        list_b = [0]
        list_k = [1, 1.5] if vary_k else [1] # [1, 2, 4, 8]
        
        if strategy == "train": # "Sin estrategia"
            dic = {
                'prob_dp': None,
                'curva': 'linear',
                'm': 10,
                'b': 0,
            }
        else:
            dic = {
                'prob_dp': list_dp,
                'curva': list_strat,
                'm': list_m,
                'b': list_b,
                'k': list_k # Cuanto mayor es k, mas favorece los stakes en 0
            }
    
        if self.verbose >= 1:
            logger.info(f"Hiperparametros estrategia de apuesta: {dic}")
    
        return dic

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

            prob_result_to_bet = max(row['prob_class_1'], row['prob_class_0'], row['prob_class_2'])
            odd_to_bet = row['odds_home'] if row['predicted_result'] == 1 else (row['odds_draw'] if row['predicted_result'] == 0 else row['odds_away'])  # Verificada
            kelly_crit = ((odd_to_bet - 1) * prob_result_to_bet - (1 - prob_result_to_bet)) / (odd_to_bet - 1)  # creo que esta bien
            
            # Calculo la confianza de la prediccion
            probs_sorted = np.sort([row['prob_class_1'], row['prob_class_0'], row['prob_class_2']], axis=0)  # axis=0 para vertical sorting si trabajas filas como columnas
            confidence_margin = probs_sorted[2] - probs_sorted[1]  # La diferencia entre la probabilidad más alta y la segunda más alta

            # Si el riesgo-beneficio es malo
            if thr_prob_min is not None and row['predicted_result'] != 0 and kelly_crit < thr_prob_min:
                if self.verbose >= 1:
                    logger.warning(f"Aplicamos doble oportunidad por kelly_crit = {kelly_crit} < {thr_prob_min}. ")

                # Apuesto doble oportunidad sin el resultado predicho
                result_to_bet = -1 if row['predicted_result'] == 1 else (-2 if row['predicted_result'] == 2 else -0)
                prob_result_to_bet = 1 - prob_result_to_bet
                odd_to_bet = self.calculate_odd_double_chance(row, result_to_bet)
                strategy = f" kelly_crit < {thr_prob_min}"
                kelly_crit = ((odd_to_bet - 1) * prob_result_to_bet - (1 - prob_result_to_bet)) / (odd_to_bet - 1)  # Lo recalculo pues ahora apuesto a otro rdo

            # Si el riesgo-beneficio es alto
            else:
                # Apuesto al resultado predicho
                result_to_bet = row['predicted_result']
                strategy = f" kelly_crit > {thr_prob_min}"

            # Guardo el resultado a apostar
            df.loc[id_match, 'result_to_bet'] = result_to_bet
            df.loc[id_match, 'prob_result_to_bet'] = prob_result_to_bet
            df.loc[id_match, 'odd_to_bet'] = odd_to_bet
            df.loc[id_match, 'strategy'] = strategy
            df.loc[id_match, 'kelly_criterion'] = kelly_crit # Podria calcular kelly_crit por rdo...
            df.loc[id_match, 'confidence_margin'] = confidence_margin # Podria calcular kelly_crit por rdo...

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
                               p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None, k: float = 1,                                  # Puntos de rectas
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
            k: Constante que controla la sensibilidad para disminuir stake con kelly negativo. Cuanto mayor de 1 es, mas decrecerá el stake

        # Returns
            Dataframe pasado como parametro con nueva columna 'stake_to_bet'

        Mejoras:
            - Separar variacion de stake por cuotas en otra funcion. 
        """
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

            # Usar la función sigmoide para reducir la variabilidad y escalar entre p_min y p_max
            num = m if m > 0 else 1
            df['stake_to_bet'] = num / (1 + np.exp(-k * df['kelly_criterion']))

        # STRATEGY: LINEAR
        elif type_relation == "linear": # Vario stake con prob_result_to_bet y cuotas de la casa

            df['stake_to_bet'] = df['prob_result_to_bet'] * m + b 
            
        # STRATEGY: LINEAR + kelly
        elif type_relation == 'kelly_linear':

            # Calcular m_ajustado para asegurar continuidad en kelly_criterion = 0
            m_ajustado = 2 * (df['prob_result_to_bet'] * m + b)
                      
            df['stake_to_bet'] = np.where(
                df['kelly_criterion'] > 0, 
                df['prob_result_to_bet'] * m + b, # aplicar linear si kelly_crit > 0 para no inflar 
                m_ajustado / (1 + np.exp(-k * df['kelly_criterion'])) # aplicar kelly si kelly_crit < 0 para reducir 
            )

            # df['lin_stake_to_bet'] = df['prob_result_to_bet'] * m + b

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

            df['stake_to_bet'] = a * (b ** df['prob_result_to_bet'])
        
        # Aumento stake en dp (forma manual de usar un stake mas alto en dp)
        df.loc[df['result_to_bet'].isin([-1, -2]), 'stake_to_bet'] *= 10

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
    
    def normalize_stake(self, df, p_min, p_max):
        """
        Capa de funcion sigmoide (creo que sirve nada mas para Kelly. 
        Para linear seria muy parecico a la variacion exponencial puesto que buscas agrandar las diferencias entre probas de 0.33 y 1)
        """
        if p_min >= p_max:
            raise ValueError("p_min debe ser menor que p_max para una normalización correcta.")

        # Normalización: Escalar los valores de kelly_raw entre 0 y 1
        df['stake_to_bet_norm'] = (df['kelly_criterion'] - p_min) / (p_max - p_min)

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

    # DETERMINE BETTING STRATEGY
    def calculate_roi_in_combinations(self, df, d_params):
        """
        Determinar un ROI para cada combinacion de hiperparametros de la estrategia de apuesta. 

        # Parameters
            df: El df puede ser de un solo resultado y por ende, definir una estrategia para dicho resultado...

        Mejoras:
            - Evitar doble oportunidad para empate. No quiero usar -0.
        """
        # Definicion de variables
        d_predic, d_metricas = {}, {}

        # Eliminate rows with NaN odds or missing predictions
        df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away', 'predicted_result'])

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

            # Aplicar estrategia a df_pred
            df_aux = self.apply_strategy(df, param_dict, prod=False)

            # Calculo de metricas (ROI y roi_pp)
            df_pred_with_metrics_roi, d_metrics = calculate_roi(df_aux)
            df_pred_with_metrics__ex, d_metrics_2 = calculate_roi(df_aux, name_extension="expected_")

            mean_stake = df_aux['stake_to_bet'].mean()
            std_stake = df_aux['stake_to_bet'].std()

            # # Concateno datos de ROI y Expected ROI (no calculo metric aqui para poder normalizar dsp)
            missing_columns = [col for col in df_pred_with_metrics__ex.columns if col not in df_pred_with_metrics_roi.columns]
            df_pred_with_metrics = pd.concat([df_pred_with_metrics_roi, df_pred_with_metrics__ex[missing_columns]], axis=1) # Concatenar únicamente las columnas que faltan
            
            # Agrego metricas de cada param
            param_dict.update(d_metrics)
            param_dict.update(d_metrics_2)
            param_dict.update({'mean_stake': mean_stake, 'std_stake': std_stake})

            if self.verbose >= 1:
                print(f"Params: {params} \n Metrics: {d_metrics} \n")

            # Guardo resultados
            d_predic[cont] = df_pred_with_metrics
            d_metricas[cont] = param_dict

        if len(param_combinations) == 1: 
            d_predic[cont] = df_pred_with_metrics
            d_metricas[cont] = param_dict
            # d_predic, d_metricas = df_pred_with_metrics, d_metrics  # Si hay una sola, evitar seleccion y listo pero no cambiar el formato..

        return d_predic, d_metricas

    def calculate_roi_in_combination(self, df, param_dict):

        if 'expected_result' not in df.columns:
            df = determine_expected_result(df, verbose=0)

        # Aplicar estrategia a df_pred
        df_aux = self.apply_strategy(df, param_dict, prod=False)

        # Calculo de metricas (ROI y roi_pp)
        df_pred_with_metrics, d_metrics = calculate_roi(df_aux)
        df_pred_with_metrics_2, d_metrics_2 = calculate_roi(df_aux, name_extension="expected_")
        
        # # Concateno datos de ROI y Expected ROI
        missing_columns = [col for col in df_pred_with_metrics_2.columns if col not in df_pred_with_metrics.columns]
        df_pred_with_metrics = pd.concat([df_pred_with_metrics, df_pred_with_metrics_2[missing_columns]], axis=1) # Concatenar únicamente las columnas que faltan
        d_metrics.update(d_metrics_2)
        return df_pred_with_metrics, d_metrics

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
        d_params_stake = {
            'type_relation': param_dict.get('curva', None),  # None o algún valor por defecto
            'm': param_dict.get('m', 10),  # 1 es un ejemplo de valor por defecto
            'b': param_dict.get('b', 0),  # 0 por defecto si no está en el diccionario
            'k': param_dict.get('k', 1)   # 1 por defecto si falta 'k'
        }
        df = self.determine_stake_to_bet(df, **d_params_stake)
        return df

    def select_best_parameters(self, data, roi_weight: float = 1, normalize: bool = False):
        """
        Seleccion de la combinacion de hiper de apuesta que maximizan el roi y expected roi minimizando el stake.
        
        # Parameters:
            data: diferentes alternativas con sus roi y expected roi.
            roi_weight: Peso del roi para seleccionar la alternativa. Su complemento es el peso del expected_roi. (float)
            normalize: Normalizando antes de sumar roi y expected_roi--> Con True funciona mal + Realmente creo que no es correcto. La diferente escala es info valiosa y que creo que debo usar.

        # Return
            n_comb: Numero de combinacion de hiper de apuesta ganador.
        """
        
        # Convierto diccionario a dataframe para facilitar manejo
        df = pd.DataFrame.from_dict(data, orient='index')

        # Calcular metrica a maximizar (lo hago aqui para poder normalizar): Max Expected roi con minimo stake
        ex_weight =  float(1 - roi_weight)   
        
        ## Sin normalizar (prefiero esta actualmente, con la dif de escala)
        if normalize:
            df = normalize_column(df, col='roi')
            df = normalize_column(df, col='expected_roi')
            col1, col2 = 'roi_norm', 'expected_roi_norm'
        else:
            col1, col2 = 'roi', 'expected_roi'

        df['metric'] = ((roi_weight * df[col1]) + (ex_weight * df[col2])) / df['mean_stake'] # + 0.5 * df['std_stake']

        # Maximizar ROI y minimizar stake
        n_comb = df['metric'].idxmax()            

        if self.verbose >= 1:
            print(roi_weight, ex_weight)
            logger.info(df)
            logger.critical(f"Nº combination: {n_comb}")

        return n_comb

    # Main
    def define_model_betting_strategy_by_result(self, df_pred, d_params, roi_weight: int, verbose: int = 0):
        """
        Determina la estrategia de apuesta optima para un modelo.

        # Parameters:
            df_pred
            col_to_max: Columna a maximizar para determinar la mejor estrategia de apuesta (str)
            d_params: 
        
        # Return

        """
        # Lista para almacenar los resultados
        logger.info("Definiendo la estrategia de apuesta optima para el modelo...")
        d_metrics_res, d_hiper_res = {}, {}
        best_df_pred = pd.DataFrame()

        # Por resultado
        for pred in [1, 0, 2]:

            # Filtro predicciones por result
            df_result = df_pred[df_pred['predicted_result'] == pred] 
            if verbose >= 0:
                logger.info(f"Resultado: {pred}. Shape {df_result.shape}")

            # Si hay predicciones del modelo para ese result
            if len(df_result) > 0:

                # si se paso un set de parametros por rdo
                if len(d_params.values()) > 1:
                    d_params_aux = d_params[pred]

                '''
                # Evito kelly en empate (aumenta stake en misma proporcion que g/p pues kelly_crit siempre es positivo)
                if pred == 0:
                    d_params_aux['curva'] = ['linear']
                # En 1 y 2, evito 'linear' (xq tal vez gana x buena racha en test y lleva a stakes altos) --> el tema es si hago dp lo elimina tmb y no quiero..
                else:
                    d_params_aux = d_params.copy()
                '''

                # Calculo roi por cada set de hiper de apuesta
                d_predic, d_metricas = self.calculate_roi_in_combinations(df_result, d_params=d_params_aux)

                # Determinar mejor estrategia para el resultado
                n_comb = self.select_best_parameters(d_metricas, roi_weight=roi_weight)
                
                if pd.isna(n_comb):
                    logger.error("La metrica es nan en todas las combinaciones. Eso puede ser porque todas las alternativas tienen el mismo roi y/o expected roi.")
                    raise ValueError # Solucion: n_comb = 1

                # Guardo datos
                d_metrics_res[pred] = d_metricas[n_comb] # Son sin ea ???
                best_df_pred = pd.concat([best_df_pred, d_predic[n_comb]], axis=0)

            else:
                logger.warning(f"No hay predicciones con el resultado {pred} (o sea, el modelo no lo predice). Asigno strategy de train.")
                d_metrics_res[pred] = {}

        # Concateno datos y guardo
        df_strat = pd.DataFrame.from_dict(d_metrics_res, orient='index')
 
        # Para tener mismo bank across all results.
        df_pred_with_stra_roi, _ = calculate_roi(best_df_pred) 
        df_pred_with_stra_ex, _ = calculate_roi(best_df_pred, name_extension='expected_') 
        # Seleccionar solo las columnas que contienen "expected_"
        expected_cols = [col for col in df_pred_with_stra_ex.columns if "expected_" in col] #  missing_columns = [col for col in df_pred_with_stra_ex.columns if col not in df_pred_with_stra_roi.columns]
        df_pred_with_stra_roi.drop(columns=expected_cols, inplace=True)
        best_df_pred = pd.concat([df_pred_with_stra_roi, df_pred_with_stra_ex[expected_cols]], axis=1)

        # Calculo G/P por resultado (para comparar con G/P sin estrategia)
        gp_total = sum(df_strat['roi'])
        df_strat.loc[1, '%_G/P'] = df_strat.loc[1, 'roi'] / gp_total * 100
        df_strat.loc[0, '%_G/P'] = df_strat.loc[0, 'roi'] / gp_total * 100
        df_strat.loc[2, '%_G/P'] = df_strat.loc[2, 'roi'] / gp_total * 100

        if self.verbose >= 1:
            logger.critical("La mejor estrategia de apuesta")
            for k, v in d_hiper_res.items():
                print(f"{k} --> {v}")

        # Exportar el DataFrame final a un archivo Excel
        return df_strat, best_df_pred

    def define_model_betting_strategy(self, df_pred, d_params, roi_weight: int, verbose: int = 0):
        """
        Determina la estrategia de apuesta óptima para un modelo aplicando la misma estrategia a todos los resultados.

        # Parameters:
            df_pred: DataFrame con las predicciones.
            d_params: Diccionario de parámetros para calcular el ROI.
            verbose: Nivel de detalle de logs.

        # Return
            df_final: DataFrame con los mejores parámetros y métricas.
            best_df_pred: DataFrame con las predicciones y resultados usando la mejor estrategia.
        """
        if verbose >= 1:
            logger.info("Definiendo la estrategia de apuesta óptima para el modelo...")

        # Calculo ROI para todas las predicciones juntas
        d_predic, d_metricas = self.calculate_roi_in_combinations(df_pred, d_params=d_params)

        if not isinstance(d_predic, pd.DataFrame):
            # Selecciono la mejor combinación
            n_comb = self.select_best_parameters(d_metricas, roi_weight)
                
            # Guardo resultados
            best_df_pred = d_predic[n_comb]
            df_strat = pd.DataFrame({**d_metricas[n_comb]}, index=[0])

            if verbose >= 1:
                logger.critical(f"La mejor estrategia de apuesta: {d_metricas[n_comb]}")

        else:
            best_df_pred = d_predic
            df_strat = pd.DataFrame([{**d_metricas}])

        return df_strat, best_df_pred

    # Prod
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
                prob, curva, m, b, k = row_pred['prob_dp'], row_pred['curva'], row_pred['m'], row_pred['b'], row_pred['k']
                prob = value_nan_to_none(prob)

                # Determino result to bet
                df_pred = self.determine_result_to_bet(df_pred, thr_prob_min=prob)

                # Determino stake to bet
                d_params_stake = {'type_relation': curva, 'm': m, 'b': b, 'k': k}
                df_pred = self.determine_stake_to_bet(df_pred, **d_params_stake)

                df_comp = pd.concat([df_comp, df_pred], axis=0)

            else:
                logger.warning(f"No hay partidos para el resultado {pred}, por lo que, no se aplica la estrategia a dicho resultado.")

        return df_comp
    
def scale_values(values: int, old_min: int, old_max: int, new_min: int, new_max: int):
    return new_min + ((values - old_min) / (old_max - old_min)) * (new_max - new_min)

def determine_bs_all_models(df_ite, country, iteration_date, date_assess, roi_weight):

    d_rows = []
    progress_bar = tqdm(total=len(df_ite), ncols=80)  # Inicializo barra de progreso

    for idx, row in df_ite.iterrows():
        n_model = row['n_iteration']
        model_name = row['model_name']

        # Leo predicciones
        df_pred_test = read_predictions(date_assess, country, iteration_date, n_model, model_name)

        # Determino ea
        df_strat, df_pred_with_stra = determine_bs_for_model(df_pred_test, roi_weight=roi_weight)
        
        # Calculo roi_sin_ea y multplicador --> Funciona mal cuando hacemos ea por rdo...
        df_strat['roi_sin_ea'] = determine_roi(df_pred_test) * 100
        df_strat['expected_roi_sin_ea'] = determine_roi(df_pred_test, var_resp='expected_result') * 100
        df_strat['mean_stake_sin_ea'] = df_pred_test['stake_to_bet'].mean()
        df_strat['mean_stake'] = df_pred_with_stra['stake_to_bet'].mean()
        df_strat['x_roi'] = (df_strat['roi'] - df_strat['roi_sin_ea']) / df_strat['roi_sin_ea']
        df_strat['x_exroi'] = (df_strat['expected_roi'] - df_strat['expected_roi_sin_ea']) / df_strat['expected_roi_sin_ea']
        df_strat['x_stake'] = (df_strat['mean_stake'] - df_strat['mean_stake_sin_ea']) / df_strat['mean_stake_sin_ea']

        # Guardo resultados
        df_strat["idx"] = n_model
        d_rows.append(df_strat)
        progress_bar.update(1)

    progress_bar.close()

    # Concateno todos los resultados en un DataFrame
    df_result = pd.concat(d_rows).set_index('idx')

    return df_result
        
def read_predictions(country, iteration_date, n_model, model_name, assess: bool = False, date_assess: str = None):

    if assess and date_assess is None:
        date_assess = datetime.datetime.now().date()

    if assess:
        logger.warning("Debe ser test + assess concatenado")
        path = f"data/{country}/p4_modeling/{iteration_date}/best_model/2_assess/{date_assess}/{n_model}__{model_name}_test_assess_.xlsx"  # Debe ser test + assess concatenado
    else:
        path = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"

    df_pred_test = pd.read_excel(path, index_col=0)
    return df_pred_test

def determine_bs_for_model(
        df_pred_test, 
        d_params,
        bs_per_res: bool = False, 
        roi_weight: int = 1, 
        rescale: bool = False,
        rescale_values: list = [5, 15],
        verbose: int = 0
        ):
    
    bs = BettingStrategy(verbose=0)

    # Imprimo prob_result_to_bet promedio
    if verbose >= 1:
        mean_prob = df_pred_test['prob_result_to_bet'].mean()
        print(f"Prob result to bet promedio: {mean_prob}")
    
    df_pred = drop_old_metrics(df_pred_test)

    if bs_per_res:
        logger.warning("WARNING! Mucho cuidado con usar bs por rdo puesto que es muy facil caer en overfitting ya sea por 1) pocos datos en el test 2) muchos parametros de bs y/o espacio de valores de cada uno. Esto le permite ajustarse mucho al test.")
        func = bs.define_model_betting_strategy_by_result
    else:
        func = bs.define_model_betting_strategy
    df_strat, df_pred_with_stra = func(df_pred, d_params=d_params, roi_weight=roi_weight, verbose=verbose)
    
    # Reescalo m (para reducir amplitud y evitar overfitting)
    if rescale:
        # Reescalar valores de m
        df_strat.rename(columns={'m': 'm_old'}, inplace=True)
        for idx, row in df_strat.iterrows():
            df_strat.loc[idx, 'm'] = scale_values(row['m_old'], old_min=min(d_params['m']), old_max=max(d_params['m']), new_min=rescale_values[0], new_max=rescale_values[1])
 
    return df_strat, df_pred_with_stra

def strategy_metrics(df_pred_test, df_pred_with_stra, dp_met: bool = True):

    def calculate_metrics(df_pred, col_name, df_metrics):
        df_metrics.loc['roi', col_name] = determine_roi(df_pred) * 100
        df_metrics.loc['expected_roi', col_name] = determine_roi(df_pred, var_resp='expected_result') * 100
        df_metrics.loc['mean_stake', col_name] = df_pred['stake_to_bet'].mean()

        df_metrics.loc['mean_prob_acerte', col_name] = df_pred[df_pred['acerte'] == 1]['prob_result_to_bet'].mean()
        df_metrics.loc['mean_prob_falle', col_name] = df_pred[df_pred['acerte'] == 0]['prob_result_to_bet'].mean()
        df_metrics.loc['mean_kc_acerte', col_name] = df_pred[df_pred['acerte'] == 1]['kelly_criterion'].mean()
        df_metrics.loc['mean_kc_falle', col_name] = df_pred[df_pred['acerte'] == 0]['kelly_criterion'].mean()

        df_metrics.loc['gp_min', col_name] = df_pred['G/P_sin_bank'].min()
        df_metrics.loc['gp_median', col_name] = df_pred['G/P_sin_bank'].median()
        df_metrics.loc['gp_mean', col_name] = df_pred['G/P_sin_bank'].mean()
        df_metrics.loc['gp_max', col_name] = df_pred['G/P_sin_bank'].max()
        df_metrics.loc['gp_std', col_name] = df_pred['G/P_sin_bank'].std()

        df_metrics.loc['stake_min', col_name] = df_pred['stake_to_bet'].min()
        df_metrics.loc['stake_median', col_name] = df_pred['stake_to_bet'].median()
        df_metrics.loc['stake_max', col_name] = df_pred['stake_to_bet'].max()
        df_metrics.loc['stake_std', col_name] = df_pred['stake_to_bet'].std()

        df_metrics.loc['gp_1', col_name] = df_pred[df_pred['predicted_result'] == 1]['G/P_sin_bank'].sum()
        df_metrics.loc['gp_0', col_name] = df_pred[df_pred['predicted_result'] == 0]['G/P_sin_bank'].sum()
        df_metrics.loc['gp_2', col_name] = df_pred[df_pred['predicted_result'] == 2]['G/P_sin_bank'].sum()
        df_metrics.loc['stake_mean_1', col_name] = df_pred[df_pred['predicted_result'] == 1]['stake_to_bet'].mean()
        df_metrics.loc['stake_mean_0', col_name] = df_pred[df_pred['predicted_result'] == 0]['stake_to_bet'].mean()
        df_metrics.loc['stake_mean_2', col_name] = df_pred[df_pred['predicted_result'] == 2]['stake_to_bet'].mean()
        df_metrics.loc['kc_mean_1', col_name] = df_pred[df_pred['predicted_result'] == 1]['kelly_criterion'].mean()
        df_metrics.loc['kc_mean_0', col_name] = df_pred[df_pred['predicted_result'] == 0]['kelly_criterion'].mean()
        df_metrics.loc['kc_mean_2', col_name] = df_pred[df_pred['predicted_result'] == 2]['kelly_criterion'].mean()

        if dp_met:
            df_metrics.loc['dp_number', col_name] = len(df_pred[df_pred['result_to_bet'].isin([-1, -2])])
            df_metrics.loc['dp_precision', col_name] = df_pred[df_pred['result_to_bet'].isin([-1, -2])]['acerte'].mean() * 100
            df_metrics.loc['dp_gp', col_name] = df_pred[df_pred['result_to_bet'].isin([-1, -2])]['G/P_sin_bank'].sum()

        return df_metrics

    # Crear DataFrame vacío con índice definido
    df_metrics = pd.DataFrame(index=[
        'roi', 'expected_roi', 'mean_stake',
        'gp_1', 'gp_0', 'gp_2', 'stake_mean_1', 'stake_mean_0', 'stake_mean_2', 'kc_mean_1', 'kc_mean_0', 'kc_mean_2',
        'mean_prob_acerte', 'mean_prob_falle', 'mean_kc_acerte', 'mean_kc_falle',
        'gp_min', 'gp_median', 'gp_mean', 'gp_max', 'gp_std',
        'stake_min', 'stake_median', 'stake_max', 'stake_std',
        'dp_number', 'dp_precision', 'dp_gp'
    ], columns=['sin_ea', 'con_ea', 'mult'])

    # Calcular métricas para cada estrategia
    df_metrics = calculate_metrics(df_pred_test, 'sin_ea', df_metrics)
    df_metrics = calculate_metrics(df_pred_with_stra, 'con_ea', df_metrics)

    # Calcular multiplicadores
    # df_metrics['mult'] = (df_metrics['con_ea'] - df_metrics['sin_ea'])/ df_metrics['sin_ea'] # falla por division error x dp
    for idx, row in df_metrics.iterrows():
        try:
            df_metrics.loc[idx, 'mult'] = (row['con_ea'] - row['sin_ea']) / row['sin_ea']
        except ZeroDivisionError:
            pass
        
    return df_metrics

def roi_to_m(df, option, m_max):
    """
    # Traducir roi a m
    """
    # Op1: Escalado lineal
    if option == 1:
        p_min, p_max = 0, df['roi'].max() # df_rois['roi'].min()
        df[f'roi_norm'] = (df['roi'] - p_min) / (p_max - p_min)

        # Calcular m por resultado
        df['m'] = df['roi_norm'] * m_max

        # Reemplazo los m negativos por 0
        df[df['m'] <= 0] = 0

    # Op2: Binning en categorias de stake --> me gusta pq tiene en cuenta dif relativas en roi.
    elif option == 2:
        df['m'] = pd.qcut(df['roi'], q=5, labels=range(1, m_max+1))

    # Op3: Escalado por percentiles (rank scaling) --> segun el ranking te da el m. No tiene en cuenta dif relativas en roi.
    elif option == 3:
        df['percentil'] = df['roi'].rank(pct=True)
        df['m'] = df['percentil'] * m_max

    # Op4: Escalado no lineal (√)
    elif option == 4:
        # Transformación (evita ceros)
        df['roi_sqrt'] = np.sqrt(df['roi'].clip(lower=0.1)) #

        # Luego normalizas y escalas igual que antes
        df['roi_norm_sqrt'] = df['roi_sqrt'] / df['roi_sqrt'].max()
        df['m'] = df['roi_norm_sqrt'] * m_max

    # Op4: Escalado no lineal (log) # no me gustó. Stakes altos para todos.
    elif option == 5:
        df['roi_log'] = np.log1p(df['roi'].clip(lower=0))
        df['roi_norm_log'] = df['roi_log'] / df['roi_log'].max()
        df['m'] = df['roi_norm_log'] * m_max

    return df

def main_2(df_best_models, d_countries, assess, date_assess):
    
    # FORMA 2 DE DEFINIR BS: 1º determino m con roi_sin_ea. 2º aplico bs con m del paso 1.
    # Problema: No puedo usar un m mas alto en doble oportunidad pues define el m con el roi sin ea y yo uso el mismo m entre rdo y su dp.
    # Por ahora no uso dp... Esta forma seria la ideal si logro separar el m de dp y rdo. O meterle un boost al stake del dp.

    # (1) Junto el roi_sin_ea por rdo de cada pais
    df_rois = pd.DataFrame()

    for id_country in l_countries:
        country = d_countries[id_country][0]
        
        row = df_best_models[df_best_models['id_country'] == id_country]
        iteration_date_dt = row['iteration_date'].values[0]
        iteration_date = pd.to_datetime(iteration_date_dt, format='%Y-%m-%d').date()
        
        n_model = int(row['n_model'].values[0])
        model_name = str(row['model_name'].values[0])
        print(f"N_model: {n_model} Iteration date: {iteration_date}")

        # Levanto df_test
        df_pred_test = read_predictions(country, iteration_date, n_model, model_name, assess=assess, date_assess=date_assess)
        logger.info(df_pred_test.shape)
        
        # Calculo rois por rdo
        df_rois.loc[f'{country}_1', 'roi'] = df_pred_test[df_pred_test['predicted_result'] == 1]['G/P_sin_bank'].sum()
        df_rois.loc[f'{country}_0', 'roi'] = df_pred_test[df_pred_test['predicted_result'] == 0]['G/P_sin_bank'].sum()
        df_rois.loc[f'{country}_2', 'roi'] = df_pred_test[df_pred_test['predicted_result'] == 2]['G/P_sin_bank'].sum()
    
    df_rois = roi_to_m(df_rois, option=4, m_max=20)
    df_rois.to_excel('/Users/nachomondino/Desktop/aksrgaksr.xlsx')


    # (2) Aplico bs para definir curva y k (pero no el m)
    for id_country in l_countries:
        country = d_countries[id_country][0]
        
        row = df_best_models[df_best_models['id_country'] == id_country]
        iteration_date_dt = row['iteration_date'].values[0]
        iteration_date = pd.to_datetime(iteration_date_dt, format='%Y-%m-%d').date()
       
        n_model = int(row['n_model'].values[0])
        model_name = str(row['model_name'].values[0])
        print(f"N_model: {n_model} Iteration date: {iteration_date}")

        # Levanto df_test
        df_pred_test = read_predictions(country, iteration_date, n_model, model_name, assess=assess, date_assess=date_assess)
        logger.info(df_pred_test.shape)

        # Defino estrategia
        bs = BettingStrategy(verbose=0)

        # Defino parametros a probar y aplico bs
        d_params = {
            1: {
                'prob_dp': [None, -0.75, -0.5, -0.25], 
                'curva': ['kelly_linear', 'kelly'], 
                'm': [df_rois.loc[f'{country}_1', 'm']], 
                'b': [0], 
                'k': [1, 2, 4]
                },
            0: {
                'prob_dp': [None], 
                'curva': ['linear'], 
                'm': [df_rois.loc[f'{country}_0', 'm']], 
                'b': [0], 
                'k': [1]
                },
            2: {
                'prob_dp': [None, -0.75, -0.5, -0.25], 
                'curva': ['kelly_linear', 'kelly'], 
                'm': [df_rois.loc[f'{country}_2', 'm']], 
                'b': [0], 
                'k': [1, 2, 4]
                }
            }
        
        df_strat, df_pred_with_stra = bs.define_model_betting_strategy_by_result(df_pred_test, d_params, roi_weight=roi_weight)

        # Calculo metricas de estrategia
        df_metrics_strat = strategy_metrics(df_pred_test, df_pred_with_stra)

        # Exporto datos
        path = f"data/{country}/p4_modeling/{iteration_date}/best_model/3_bet_strategy"
        df_strat.to_excel(f"{path}/df_strategy_{n_model}_{model_name}.xlsx", index=True)
        df_metrics_strat.to_excel(f"{path}/strat_metrics_{n_model}_{model_name}.xlsx", index=True)
        df_pred_with_stra.to_excel(f"{path}/predicciones_{n_model}_{model_name}.xlsx", index=True)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    l_countries = [48, 55, 59, 77, 148] 
    one_model = True
    assess, date_assess = False, '2025-04-29' # datetime.datetime.now().date() 

    df_best_models = pd.read_excel("./data/df_best_models.xlsx")

    # Defino hiperparametros de apuesta
    bs_per_res = True
    ## Result to bet
    vary_dp = False # Doble oportunidad --> te recomendaria que fuerzes dp con curvas linear o kelly_linear (sin kelly) para no inflar su stake desmedidamente
    ## Stake to bet
    strat = ['kelly', 'kelly_linear'] # 'linear' no usar linear para que no pueda usar mas stake en 1 o 2 si justo hubo una buena racha como en SPA
    space_m = [10, 11, 10] # Problema al variar: muev
    vary_k = True
    ## Seleccion de bs
    roi_weight = 1 # Expected tiene mas razon a largo plazo que roi (segun libro). Puede que coincida.

    d_countries = {
        48: ["england"],
        55: ["france"], 
        59: ["germany"],
        77: ["italy"],
        148: ["spain"]
        }

    main_2(df_best_models, d_countries, assess, date_assess)