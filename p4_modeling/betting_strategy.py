import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from p3_data_preparation.construct_data import determine_expected_result
from p4_modeling.asses_model import calculate_combined_metric, calculate_roi, determine_roi, drop_old_metrics
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

    # HIPER SPACE
    def define_hiperparameters(self, strategy, big_space_m: bool = True, vary_dp: bool = False):
        """
        Defino hiperparametros de estrategia de apuesta a probar segun si apuesto como la realidad o no.
        """
        list_dp = [0, 0.45, 0.6, 0.75] if vary_dp else [0]  # el 0.4 esta muy cerca del cambio de result to bet entre assess y prod.
        list_m = [10, 15, 20, 25, 30, 35, 40, 45, 50, 65, 80, 95, 110, 140, 170, 200] if big_space_m else [60, 80, 100, 120, 140] # [10, 20, 40, 60, 80, 100, 120] # no uso 200 por si acertó todo en ese rdo.
        
        if strategy == "train": # "Sin estrategia"
            dic = {
                'prob_dp': 0,
                'curva': 'linear',
                'm': 10,
                'b': 0,
            }
        else:
            dic = {
                'prob_dp': list_dp,
                'curva': [strategy], 
                'm': list_m,
                'b': [0],
                'k': [1, 1.5, 2, 2.5, 3, 5, 10] if strategy == 'kelly_linear' else [None]
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

            # Si el modelo esta POCO seguro del resultado predicho
            if (prob_result_to_bet < thr_prob_min):
                
                # Apuesto doble oportunidad sin el resultado predicho
                result_to_bet = -1 if row['predicted_result'] == 1 else (-2 if row['predicted_result'] == 2 else -0)
                prob_result_to_bet = 1 - prob_result_to_bet
                odd_to_bet = self.calculate_odd_double_chance(row, result_to_bet)
                strategy = f"dif_prob_mod_bm < {thr_prob_min}"

            # Si nuestro modelo esta seguro del rdo
            else:
                # Apuesto al resultado predicho
                result_to_bet = row['predicted_result']
                strategy = f"dif_prob_mod_bm > {thr_prob_min}"

            # Guardo el resultado a apostar
            df.loc[id_match, 'result_to_bet'] = result_to_bet
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
                               p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None, k: float = None,                                  # Puntos de rectas
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

        # Calculo kelly criterion
        df['kelly_criterion'] = ((df['odd_to_bet'] - 1) * df['prob_result_to_bet'] - (1 - df['prob_result_to_bet'])) / (df['odd_to_bet'] - 1) 

        # STRATEGY: EQUAL
        if type_relation == "equal":  
            df['stake_to_bet'] = 1

        # STRATEGY: KELLY
        elif type_relation == 'kelly':

            # Usar la función sigmoide para reducir la variabilidad y escalar entre p_min y p_max
            num = m if m > 0 else 1
            df['stake_to_bet'] = num / (1 + np.exp(-df['kelly_criterion']))

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
        d_predic, d_hiper, d_metricas = {}, {}, {}

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
            if self.verbose >= 0:
                print(f"\nNº Combinacion: {cont}")
                print(param_dict)

            # Aplicar estrategia a df_pred
            df_aux = self.apply_strategy(df, param_dict, prod=False)

            # Calculo de metricas (ROI y roi_pp)
            df_pred_with_metrics_roi, d_metrics = calculate_roi(df_aux)
            df_pred_with_metrics__ex, d_metrics_2 = calculate_roi(df_aux, name_extension="expected_")
            
            # # Concateno datos de ROI y Expected ROI (no calculo metric aqui para poder normalizar dsp)
            missing_columns = [col for col in df_pred_with_metrics__ex.columns if col not in df_pred_with_metrics_roi.columns]
            df_pred_with_metrics = pd.concat([df_pred_with_metrics_roi, df_pred_with_metrics__ex[missing_columns]], axis=1) # Concatenar únicamente las columnas que faltan
            d_metrics.update(d_metrics_2)

            if self.verbose >= 1:
                print(f"Params: {params} \n Metrics: {d_metrics} \n")

            # Guardo resultados
            d_predic[cont] = df_pred_with_metrics
            d_hiper[cont] = param_dict
            d_metricas[cont] = d_metrics

        if len(param_combinations) == 1: 
            d_predic, d_hiper, d_metricas = df_pred_with_metrics, param_dict, d_metrics

        return d_predic, d_hiper, d_metricas

    def calculate_roi_in_combination(self, df, param_dict):

        if 'expected_result' not in df.columns:
            df = determine_expected_result(df, goals_to_xg_ratio=0.42, verbose=0)

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
        d_params_stake = {'type_relation': param_dict['curva'], 'm': param_dict['m'], 'b': param_dict['b'], 'k': param_dict['k']}
        df = self.determine_stake_to_bet(df, **d_params_stake)
        return df

    def select_best_parameters(self, data, roi_weight: int = 1):
        
        # Convierto diccionario a dataframe para facilitar manejo
        df = pd.DataFrame.from_dict(data, orient='index')
        
        # Calcular metrica a maximizar (lo hago aqui para poder normalizar)
        ex_weight = 1 - roi_weight
        df = calculate_combined_metric(df, l_metrics=['roi', 'expected_roi'], l_weights=[roi_weight, ex_weight])

        # Encontrar la fila con el valor máximo de 'metric'
        n_comb = df['metric'].idxmax()
        if self.verbose >= 1:
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

        Mejoras:
            - simplificar codigo cuando hago el if by_result
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
                # Calculo roi por cada set de hiper de apuesta
                d_predic, d_hiper, d_metricas = self.calculate_roi_in_combinations(df_result, d_params=d_params)

                # Determinar mejor estrategia para el resultado      
                n_comb = self.select_best_parameters(d_metricas, roi_weight=roi_weight)
                
                if pd.isna(n_comb):
                    logger.error("La metrica es nan en todas las combinaciones. Eso puede ser porque todas las alternativas tienen el mismo roi y/o expected roi.")
                    n_comb = 1
                print("sefiqwiefjqiowfj", n_comb)
                
                # Guardo datos
                d_hiper_res[pred] = d_hiper[n_comb]
                d_metrics_res[pred] = d_metricas[n_comb]
                best_df_pred = pd.concat([best_df_pred, d_predic[n_comb]], axis=0)

            else:
                logger.warning(f"No hay predicciones con el resultado {pred} (o sea, el modelo no lo predice). Asigno strategy de train.")
                d_hiper_res[pred] = self.define_hiperparameters(strategy="train")
                d_metrics_res[pred] = {}

        # Concateno datos y guardo
        df_metrics = pd.DataFrame.from_dict(d_metrics_res, orient='index')
        df_hiper = pd.DataFrame.from_dict(d_hiper_res, orient='index')
        df_final = pd.concat([df_hiper, df_metrics], axis=1)

        # Para tener mismo bank across all results.
        df_pred_with_stra_roi, _ = calculate_roi(best_df_pred) 
        df_pred_with_stra_ex, _ = calculate_roi(best_df_pred, name_extension='expected_') 
        # Seleccionar solo las columnas que contienen "expected_"
        expected_cols = [col for col in df_pred_with_stra_ex.columns if "expected_" in col] #  missing_columns = [col for col in df_pred_with_stra_ex.columns if col not in df_pred_with_stra_roi.columns]
        df_pred_with_stra_roi.drop(columns=expected_cols, inplace=True)
        best_df_pred = pd.concat([df_pred_with_stra_roi, df_pred_with_stra_ex[expected_cols]], axis=1)

        # Calculo G/P por resultado (para comparar con G/P sin estrategia)
        gp_total = sum(df_final['roi'])
        df_final.loc[1, '%_G/P'] = df_final.loc[1, 'roi'] / gp_total * 100
        df_final.loc[0, '%_G/P'] = df_final.loc[0, 'roi'] / gp_total * 100
        df_final.loc[2, '%_G/P'] = df_final.loc[2, 'roi'] / gp_total * 100

        if self.verbose >= 1:
            logger.critical(f"La mejor estrategia de apuesta: {d_hiper}")

        # Exportar el DataFrame final a un archivo Excel
        return df_final, best_df_pred

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
        d_predic, d_hiper, d_metricas = self.calculate_roi_in_combinations(df_pred, d_params=d_params)

        if not isinstance(d_predic, pd.DataFrame):
            # Selecciono la mejor combinación
            n_comb = self.select_best_parameters(d_metricas, roi_weight)

            # Guardo resultados
            best_df_pred = d_predic[n_comb]
            df_final = pd.DataFrame({**d_hiper[n_comb], **d_metricas[n_comb]}, index=[0])
        else:
            best_df_pred = d_predic
            df_final = pd.DataFrame([{**d_hiper, **d_metricas}])

        if verbose >= 1:
            logger.critical(f"La mejor estrategia de apuesta: {d_hiper[n_comb]}")

        return df_final, best_df_pred

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

                # Determino result to bet
                df_pred = self.determine_result_to_bet(df_pred, thr_prob_min=prob)

                # Determino stake to bet
                d_params_stake = {'type_relation': curva, 'm': m, 'b': b, 'k': k}
                df_pred = self.determine_stake_to_bet(df_pred, **d_params_stake)

                df_comp = pd.concat([df_comp, df_pred], axis=0)

            else:
                logger.warning(f"No hay partidos para el resultado {pred}, por lo que, no se aplica la estrategia a dicho resultado.")

        return df_comp
    
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    l_countries = [48, 55, 59, 77, 148]
    l_countries = [59, 77, 148]

    d_countries = {
        # 6: ["argentina", '2025-02-06'], 
        48: ["england", '2025-03-23', 1071],
        55: ["france", '2025-03-23', 918], 
        59: ["germany", '2025-03-23', 385],
        77: ["italy", '2025-03-23', 955],
        148: ["spain", '2025-03-24', 1135]
        }
    
    model_name = "LogisticRegression"
    
    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        n_model = d_countries[id_country][2]
    
        # (4) ESTRATEGIA DE APUESTA
        bs = BettingStrategy(country, iteration_date, verbose=0)
        path_test = f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx"
        df_pred_test = pd.read_excel(path_test, index_col=0)
        
        # Dropeo old metrics (sino calcula mal las nuevas)
        df_pred = drop_old_metrics(df_pred_test)

        per_res = True
        d_params = bs.define_hiperparameters(strategy='kelly_linear', big_space_m=True, vary_dp=False) # Defino hiperparametros de estrategia de apuesta a probar. Con linear no tiene en cuenta cuotas y puede llegar a apostar mucho en cuota baja.
        d_params['m'] = [10]
        if per_res:
            func = bs.define_model_betting_strategy_by_result
        else:
            func = bs.define_model_betting_strategy
        df_strat, df_pred_with_stra = func(df_pred, d_params=d_params, roi_weight=1)

        ## Calculo metricas  ## Solo calculo el roi que es lo unico que cambia.. o que me interesa medir
        roi_con_ea = determine_roi(df_pred_with_stra, var_resp='result')

        path = f"data/{country}/p4_modeling/{iteration_date}/best_model//3_bet_strategy"
        df_strat.to_excel(f"{path}/df_strategy_{n_model}_{model_name}.xlsx", index=True)
        df_pred_with_stra.to_excel(f"{path}/predicciones_{n_model}_{model_name}.xlsx", index=True)
