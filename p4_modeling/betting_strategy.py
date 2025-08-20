import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
from p3_data_preparation.format_data import value_nan_to_none
from p3_data_preparation.construct_data import determine_expected_result
from p4_modeling.asses_model import calculate_roi
from utils import directories


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

            # Si el riesgo-beneficio es malo, doble oportunidad
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

    # STAKE TO BET
    def determine_stake_to_bet(self, df, type_relation: str = 'equal', p1: tuple = (0, 0), p2: tuple = (1, 1),  m: float = None, b: float = None, k: float = 1):
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

        # Limito stake a entre 0 y 100 
        df = self.cap_stake(df)
        return df
    
    def stake_reduction(self, df):
        """
        Aplico reducciones a stake en PROD.
        """
        logger.warning("Aplicando modificadores de stake para PROD...")

        # No apostamos en local
        df.loc[df['result_to_bet'] == 1, 'stake_to_bet'] *= 0

        # Apostamos el doble en empate
        # df.loc[df['result_to_bet'] == 2, 'stake_to_bet'] *= 2

        # Confidence margin
        # df.loc[(df['confidence_margin'] < 0.025) & (df['result_to_bet'] != 0), 'stake_to_bet'] *= 0.3 # Reducir stake si confidence_margin < threshold

        # Disminuyo stake por rellenado de emergencia
        df.loc[df['player_emergency_fill'] == 1, 'stake_to_bet'] *= 0
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
    
    # Calculo de ROI en predicciones
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

    def calculate_roi_in_combination(self, df, param_dict):
        """
        Para calcular ROI de las prediciones de un modelo

        ### Parameters:
            df: Predicciones de un modelo (DataFrame)
            param_dict: Estrategia de apuesta (dict)
        """
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
    
    # Prod
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
        if prod:
            df = self.stake_reduction(df)
            
        return df

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
    roi_weight = 0.5 # Expected tiene mas razon a largo plazo que roi (segun libro). Puede que coincida.

    d_countries = {
        48: ["england"],
        55: ["france"], 
        59: ["germany"],
        77: ["italy"],
        148: ["spain"]
        }