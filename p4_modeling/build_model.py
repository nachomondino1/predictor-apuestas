import pandas as pd
import numpy as np
import time
from itertools import product
from utils.set_up_logging import logger
import warnings
# Grid y Bayes
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import log_loss, make_scorer, f1_score, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


def select_best_hiperparameters(
        model, X_train: pd.DataFrame, y_train:  pd.DataFrame, X_val:  pd.DataFrame, y_val: pd.DataFrame, 
        k: int, params: dict = None, bayes: bool = True, n_iter: int = None, 
        refit: str = None, verbose: int = 1):
    """
    Selecciona los mejores hiperparametros para un modelo.

    # Parameters
        model: Modelo de Machine Learning. (sklearn.ensemble)
        X: Dataframe de validacion con variables predictoras. (DataFrame)
        y: Dataframe de validacion solo con variable respuesta. (DataFrame)
        k: Numero de folds. (int)
        params: Parametros a evaluar (dict)
        bayes: True para usar BayesSearchCV y false para usar GridSearchCV (bool)
        scoring: Métrica de evaluación para definir mejor combinacion de hiperparametros (e.g. 'accuracy', 'precision', 'recall', 'roc_auc', 'f1', etc) (str)

    # Return
        best_search: Elemento Search con el mejor estimador, los mejores hiperparametros, las metricas de cada combinacion, etcetera.
    """
    # Defino el set de val como test en CV de GridSearchCV / BayesSearchCV
    X_val_train = pd.concat([X_train, X_val], axis=0) 
    y_val_train = pd.concat([y_train, y_val], axis=0) 
    split_index = [-1 if x in X_train.index else 0 for x in X_val_train.index] # Create a list where train data indices are -1 and validation data indices are 0
    pds = PredefinedSplit(test_fold = split_index) # Use the list to create PredefinedSplit
    if verbose >= 2:
        logger.warning(f'X_train: {X_train.shape} + X_val: {X_val.shape} = {X_val_train.shape}')
        logger.warning(pds)

    # Definicion de variables
    model_name = str(model)[:str(model).find('(')]   # Obtengo el nombre del modelo para poder buscar sus hiperparametros
    # Seteo Bayes a False cuando es Logistic. Evito Bayes para Logistic
    if model_name == 'LogisticRegression':
        bayes = False
    params = space_params(model_name, bayes) if params is None else params

    # Determino numero de clases
    num_classes = len(np.unique(y_val_train))
    var_resp = 'categorical' if num_classes <= 5 else 'continuous'
    scoring = default_scoring(target_type=var_resp)
    if refit is None:
        logger.warning("No se paso un refit a usar, por lo que, se asigna el default.")
        refit = lambda cv_results: custom_refit(cv_results, target_type=var_resp)

    if verbose >= 1:
        logger.info(f"Seleccionando mejores hiperparametros para {model_name} con k={k}")

    # BayesSearch
    if bayes:
                
        if verbose >= 1: 
            logger.warning("BayesSearchCV...")
        start_bayes = time.time()

        if n_iter is None:
            # Calcular iteraciones basadas en el tamaño del dataset
            n_iter = 30 # Puedo usar estimate_bayes_iterations_2 o estimate_bayes_iterations 
            
        # Crear el objeto BayesSearchCV
        bayes_search = BayesSearchCV(
            estimator=model, 
            search_spaces=params, 
            cv=pds, 
            n_iter=n_iter, 
            scoring=scoring,
            refit="f1_score" if var_resp == 'categorical' else 'neg_mean_squared_error',  # cross_entropy_loss # O la métrica que prefieras
            n_jobs=-1
            )

        # Ajustar el objeto BayesSearchCV a los datos de entrenamiento
        bayes_search.fit(X_val_train, y_val_train)
        best_search = bayes_search

        end_bayes = time.time()
        if verbose >= 1:
            logger.info(f"Seleccion de hiperparametros optimos con Bayes en {(end_bayes - start_bayes) / 60:.1f} minutos")

    # GridSearch
    else:    
        if verbose >= 1: 
            logger.warning("GridSearchCV...")

        start_grid = time.time()
        params_grid = space_params(model_name, bayes=False)

        # Crear el objeto GridSearchCV
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=params_grid, 
            cv=pds,
            scoring=scoring, 
            refit=refit,
            n_jobs=-1  # el n_jons -1 evitaria el error  "warnings.warn(f"resource_tracker: {name}: {e!r}")"" (AUN NO LO PROBÉ, NO SE SI FUNCIONA)
            )

        # Ajustar el objeto GridSearchCV a los datos de entrenamiento
        grid_search.fit(X_val_train, y_val_train)  # Esta ok X_val_train y y_val_train
        best_search = grid_search         

        # Obtener los mejores hiperparámetros
        end_grid = time.time()
        if verbose >= 1: 
            logger.info(f"Seleccion de hiperparametros optimos con Grid en {(end_grid - start_grid) / 60:.1f} minutos")

    best_model = best_search.best_estimator_
    best_params = best_search.best_params_
    best_index = best_search.best_index_ # Metricas de la mejor combinacion
    results = pd.DataFrame(best_search.cv_results_)
    
    # Metricas de validacion
    if var_resp == 'categorical':
        d_metrics = {
            'cv_accuracy': best_search.cv_results_['mean_test_accuracy'][best_index],
            'cv_f1_score_wei': best_search.cv_results_['mean_test_f1_score_wei'][best_index],
            'cv_f1_score': best_search.cv_results_['mean_test_f1_score'][best_index],
            'cv_f1_score_draw': best_search.cv_results_['mean_test_f1_score_draw'][best_index],
            'cv_cross_entropy_loss': -best_search.cv_results_['mean_test_cross_entropy_loss'][best_index]  # Negar porque invertimos el log_loss
            # Agregar medidas de desviacion estandar. --> Solo mirar el promedio de la validación cruzada puede ocultar problemas de inconsistencia entre pliegues. Incluye siempre la desviación estándar.
        }
    else:
        d_metrics = {
            'mean_test_neg_mean_squared_error': best_search.cv_results_['mean_test_score'][best_index]
        }

    # Mejores hiperparámetros
    if verbose >= 0:
        logger.info(f"Mejor combinación de parameters: {best_params} \n Metricas de la mejor comb (scoring): {d_metrics} ")

    return best_model, best_params, d_metrics, results

# Metricas a calcular y maximizar para selec hiper
def default_scoring(target_type: str, verbose: int = 0):
    """
    Asigna un valor default a scoring según la variable respuesta. Metricas a medir en GridSeachCV.

    # Parameters
        target_type: 'categorical' para clasificación, 'continuous' para regresión.
        verbose: Nivel de detalle en los logs (0 = silencioso, 1 = muestra información).

    # Return
        scoring: Métrica a utilizar en evaluación para determinar la mejor combinación de hiperparámetros.
    """

    if target_type == 'categorical': 
        
        scoring = {
            'accuracy': 'accuracy',
            'f1_score': make_scorer(f1_score, average='macro'),
            'f1_score_wei': make_scorer(f1_score, average='weighted'),
            'f1_score_draw': make_scorer(f1_score_class_0),
            'cross_entropy_loss': make_scorer(log_loss, greater_is_better=False, response_method="predict_proba"),
            'f1_logloss_combo': combined_scorer(alpha=0.7),
        }

    elif target_type == 'continuous':
        scoring = 'neg_mean_squared_error'  # Regresión con MSE para variables continuas

    else:
        raise ValueError("target_type debe ser 'categorical' o 'continuous'.")

    if verbose >= 1:
        print(f"Target type: {target_type} --> Métrica por default: {scoring}")

    return scoring

# Función personalizada para medir solo la clase 0 en problemas multiclase
def f1_score_class_0(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[0], average="micro")  # 'micro' cuenta solo los positivos en la clase 0

def combined_f1_logloss(y_true, y_pred, y_proba, alpha=0.7):
    """
    alpha: peso del f1_score (entre 0 y 1). El complemento se asigna a log_loss.
    """
    f1 = f1_score(y_true, y_pred, average='macro')

    # Evitar log_loss infinita
    eps = 1e-15
    y_proba = np.clip(y_proba, eps, 1 - eps)

    # Log loss normalizada (0 = perfecta, 1 = pésima o baseline)
    ll = log_loss(y_true, y_proba)
    
    # Normalización: asumimos que el máximo log_loss posible ≈ log(n_classes)
    n_classes = y_proba.shape[1]
    max_ll = np.log(n_classes)
    ll_normalized = ll / max_ll

    # Convertimos log_loss a score (mayor es mejor)
    log_score = np.clip(1 - ll_normalized, 0, 1)

    # Combinamos ambos
    return alpha * f1 + (1 - alpha) * log_score

def combined_scorer(alpha=0.7):
    def score_func(estimator, X, y):
        y_pred = estimator.predict(X)
        y_proba = estimator.predict_proba(X)
        return combined_f1_logloss(y, y_pred, y_proba, alpha=alpha)
    return score_func

def custom_refit(cv_results, target_type='categorical', verbose: int = 0):
    """
    Selecciona el mejor modelo según la métrica adecuada. La metrica de las de "scoring" que usaremos para seleccionar
    la mejor combinacion de hiperparametros.
    
    Parámetros:
    - cv_results: dict con los resultados de la validación cruzada.
    - target_type: 'categorical' para clasificación, 'continuous' para regresión.

    Retorna:
    - Índice del mejor modelo según la métrica correspondiente.
    """
    metrics = {
        'categorical': 'mean_test_f1_logloss_combo',
        'continuous': 'mean_test_score'  # En lugar de 'mean_test_neg_mean_squared_error' 
    }

    key = metrics.get(target_type)

    if key not in cv_results:
        raise ValueError(f"La métrica {key} no está en los resultados: {cv_results.keys()}")

    if verbose >= 1:
        print(f"Metrica a usar para selec la mejor comb {target_type}: {key}")

    return cv_results[key].argmax() # # Mayor valor negativo (menor pérdida real) # CUIDADO con la metrica que uses para ver si usar argmax() o argmin()

# Espacio de hiperparametros
def check_best_params_limits(best_params, space):
    for param, value in best_params.items():
        # Obtiene el rango del parámetro desde el espacio
        param_space = space.get(param, None)
        
        if param_space:
            # Solo verifica límites para parámetros continuos (Integer, Real)
            if isinstance(param_space, Integer) or isinstance(param_space, Real):
                min_val, max_val = param_space.bounds  # Límite inferior y superior del espacio

                # Genera un warning si el valor del parámetro es igual al límite inferior o superior
                if value == min_val:
                    warnings.warn(f"El parámetro '{param}' ha tomado su valor mínimo {min_val}. "
                                  f"Considera ampliar el espacio inferior.")
                elif value == max_val:
                    warnings.warn(f"El parámetro '{param}' ha tomado su valor máximo {max_val}. "
                                  f"Considera ampliar el espacio superior.")
            # Para Categorical, no hay un límite "numérico" pero puedes agregar alguna lógica si es necesario.

def space_params(model_name, bayes, verbose: int = 0):
    """
    Defino hiperparametros a probar por modelo.

    # Parameters
        model_name: Nombre del modelo (e.g. LogisticRegression, SVC, etc) (str)
        bayes: True para usar BayesSearchCV y false para usar GridSearchCV (bool)

    # Return
        params: Parametros a evaluar para el modelo dado (list o dict)
    """
    # Defino hiperparametros de arboles de decision
    l_n_estimators = [31, 51, 101]
    l_learning_rate = [0.001, 0.01, 0.1]
    l_max_depth = [2, 3, 5] # None, 10
    l_min_samples_leaf = [21, 51] # [21, 51]
    l_min_samples_split = [(elem * 2) + 1 for elem in l_min_samples_leaf] # min_samples_split≥2×min_samples_leaf. P
    l_max_features = ["sqrt", None] # "log2",  # None juega sobreotdo cdo quedan pocas variables predictoras...
    l_bootstrap = [True]

    # Defino hiperparametros de LogisticRegression
    l_max_it = [10000] # aumentar max_iter no causa overfitting en LogisticRegression()
    param_c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    l_tol = [1e-4, 1e-3, 1e-2]

    d_params = {
        # ARBOLES DE DECISION
        'DecisionTreeClassifier': {
            'criterion': Categorical(['entropy', 'gini']) if bayes else ['entropy', 'gini'],
            'splitter': Categorical(['random', 'best']) if bayes else ['best', 'random'],
            'max_depth': Categorical(l_max_depth) if bayes else [None, 3, 5, 10, 15],
            'min_samples_split': Integer(10, 50) if bayes else l_min_samples_split, # Mayor o igual a 2
            'min_samples_leaf': Integer(10, 50) if bayes else l_min_samples_leaf, # Si usas 1 las probas van a ser 1-0-0, 0-1-0, 0-0-1, si usas 4 0.5-0.5-0 y asi.
            'max_features': Categorical(l_max_features) if bayes else l_max_features, # Real(0.1, 1.0)
        },
        'RandomForestClassifier': {
            'n_estimators': Integer(100, 200) if bayes else l_n_estimators,
            'criterion': Categorical(['entropy', 'gini']) if bayes else ['entropy', 'gini'],
            'max_depth': Categorical(l_max_depth) if bayes else l_max_depth,
            'min_samples_split': Integer(30, 51) if bayes else l_min_samples_split, # Mayor o igual a 2
            'min_samples_leaf': Integer(20, 51) if bayes else l_min_samples_leaf,
            'max_features': Categorical(l_max_features) if bayes else l_max_features, # Real(0.1, 1.0)
            'bootstrap': Categorical(l_bootstrap) if bayes else l_bootstrap # False
        },
        'RandomForestRegressor': {
            'n_estimators': Integer(100, 500) if bayes else l_n_estimators,        
            'criterion': Categorical(["friedman_mse"]) if bayes else ["friedman_mse"], # "squared_error", "absolute_error"
            'max_depth': Categorical(l_max_depth) if bayes else l_max_depth,  # 30 # 3
            'min_samples_split': Integer(2, 21) if bayes else l_min_samples_split, # Mayor o igual a 2 
            'min_samples_leaf': Integer(2, 21) if bayes else l_min_samples_leaf,
            'bootstrap': Categorical(l_bootstrap) if bayes else l_bootstrap
            # 'verbose': Categorical([0]) if bayes else [0]
        },
        'XGBClassifier': {
            'booster': Categorical(['gbtree']) if bayes else ['gbtree'], # 'gbtree', 'dart'
            'n_estimators': Integer(10, 30) if bayes else [10, 20, 30],  # Suele ganar con 100
            'max_depth': Integer(3, 6) if bayes else l_max_depth, # 3, 
            'learning_rate': Real(0.01, 0.1, prior="log-uniform") if bayes else l_learning_rate,
            'min_child_weight': Integer(3, 10) if bayes else [3, 10], #5
            'grow_policy': Categorical(['lossguide']) if bayes else ['depthwise', 'lossguide'], # 'depthwise'
            'subsample': Real(0.6, 0.9) if bayes else [0.6, 0.9],
            'colsample_bytree': Real(0.6, 0.9) if bayes else [0.6, 0.9],
            # Regularizacion. --> Demasiada regularizacion puede llevar a predicciones uniformes 33-33-33.
            'gamma': Real(0.3, 1.0) if bayes else [0.3, 0.65, 1],
            'alpha': Real(0.5, 2.0, prior="log-uniform") if bayes else [0.5, 2.0],
            'lambda': Real(1.0, 5.0, prior="log-uniform")if bayes else [1.0, 5.0]
            # 'verbosity': Categorical([1]) if bayes else [1] # 0 (silent), 1 (warning), 2 (info), and 3 (debug). Por default es 1.
        },
        'GradientBoostingClassifier': { # Tarda muchisimo en entrenar
            'n_estimators': Integer(100, 300) if bayes else l_n_estimators, 
            'learning_rate': Real(0.001, 0.1) if bayes else l_learning_rate,
            'max_depth': Integer(3, 12) if bayes else l_max_depth,
            'min_samples_split': Integer(2, 8) if bayes else l_min_samples_split,  # Minimo 2.
            'min_samples_leaf': Integer(2, 5) if bayes else l_min_samples_leaf, 
            # 'subsample': Real(0.7, 1.0) if bayes else [0.8, 1.0],
            # 'max_features': ['auto'],
            # 'loss': ['deviance']
        },
        
        # Modelos lineales
        'LogisticRegression': [
            {
                'penalty': Categorical([None]) if bayes else [None], 
                'solver': Categorical(['newton-cg', 'lbfgs']) if bayes else ['newton-cg', 'lbfgs', 'sag'], # 'sag' --> ConvergenceWarning (talvez por la escala)
                'max_iter': Categorical(l_max_it) if bayes else l_max_it,
                'tol': Real(1e-6, 1e-2, prior='log-uniform') if bayes else l_tol,
                'warm_start': Categorical([True, False]) if bayes else [True, False]
                }, 
            {
                'penalty': Categorical(['l2']) if bayes else ['l2'], 
                'solver': Categorical(['newton-cg', 'lbfgs']) if bayes else ['newton-cg', 'lbfgs', 'sag'], 
                'C': Real(0.001, 100, prior='log-uniform') if bayes else param_c, 
                'max_iter': Categorical(l_max_it)  if bayes else l_max_it,
                'tol': Real(1e-6, 1e-2, prior='log-uniform') if bayes else l_tol,
                'warm_start': Categorical([True, False]) if bayes else [True, False]
                },
            {
                'penalty': Categorical(['l1']) if bayes else ['l1'], 
                'solver': Categorical(['liblinear', 'saga']) if bayes else ['liblinear', 'saga'], 
                'C': Real(0.001, 100, prior='log-uniform') if bayes else param_c, 
                'max_iter': Categorical(l_max_it) if bayes else l_max_it,
                'tol': Real(1e-6, 1e-2, prior='log-uniform') if bayes else l_tol,
                'warm_start': Categorical([True, False]) if bayes else [True, False]
                },
            {
                'penalty': Categorical(['elasticnet']) if bayes else ['elasticnet'], 
                'solver': Categorical(['saga']) if bayes else ['saga'], 
                'C': Real(0.001, 100, prior='log-uniform') if bayes else param_c, 
                'l1_ratio': Real(0, 1) if bayes else [0.5], 
                'max_iter': Categorical(l_max_it) if bayes else l_max_it,
                'tol': Real(1e-6, 1e-2, prior='log-uniform') if bayes else l_tol,
                'warm_start': Categorical([True, False]) if bayes else [True, False]
                }
        ],
        'Lasso': {
            'alpha': [0.1, 1.0, 10.0],  # Parámetro de regularización que controla la fuerza de la penalización L1. Un valor más alto de alpha produce una mayor regularización y puede conducir a una selección más agresiva de características.
            'fit_intercept': [True, False],  # Indica si se debe ajustar un intercepto (término independiente) en el modelo.
            'precompute': [True, False],  # Indica si se deben precalcular las matrices de productos internos para acelerar el ajuste del modelo.
            'max_iter': [100, 500, 1000],  # Número máximo de iteraciones para converger durante el ajuste del modelo.
            'positive': [True, False],  # Indica si se deben restringir los coeficientes a ser solo valores no negativos.
            'selection': ['cyclic', 'random']  # Método de selección de características. 'cyclic' utiliza el orden cíclico de las características para ajustar el modelo, mientras que 'random' selecciona aleatoriamente características en cada iteración.
        },

        # Otros / Hibridos
        'SVC': {
            'C': Real(0.01, 3) if bayes else [0.1, 0.5, 1], # C=3 genera overf # Controls strictness of the missclassifications. +, + over
            'kernel': Categorical(['rbf', 'sigmoid']) if bayes else ['rbf', 'sigmoid'], # Tipo de plano separador de clases.
            'gamma': Categorical(['scale', 'auto']) if bayes else ['scale', 'auto'], # Controls the influence of a single training point. +, + over
            'coef0': Real(0, 1) if bayes else [0.0, 0.5],
            'shrinking': Categorical([True, False]) if bayes else [True, False],
            'probability': Categorical([True]) if bayes else [True],
            'decision_function_shape': Categorical(['ovo', 'ovr']) if bayes else ['ovo', 'ovr']
        },
        'PCA': {
            'n_components': [None, 2, 3, 4, 5, 8, 10, 15],  # Si gana None, elimina 1 sola variable... # Número de componentes principales a mantener
            'whiten': [False, True],  # Indica si aplicar blanqueamiento de los datos
            'svd_solver': ['auto', 'full', 'arpack', 'randomized'], # Algoritmo de descomposición SVD a utilizar
            'iterated_power': [0, 1, 2],  # Número de veces que se aplica el método de la potencia iterada
            'tol': [0.0, 0.001, 0.01],  # Tolerancia para la convergencia del algoritmo
            'copy': [True, False]  # Copiar los datos de entrada o modificarlos en su lugar
        },

        # REDES NEURONALES
        'MLPClassifier': {
            # 'hidden_layer_sizes': Categorical([(50,), (100,)]) if bayes else [(50,), (100,)], # Tiene problema.
            'hidden_layer_sizes': Categorical([50, 100]) if bayes else [50, 100], # 50 va bien
            'activation': Categorical(['logistic', 'relu']) if bayes else ['logistic', 'relu'], # logistic va.
            'solver': Categorical(['adam']) if bayes else ['adam', 'lbfgs'],
            'alpha': Real(0.0001, 0.001) if bayes else [0.00001, 0.0001, 0.001],
            # 'learning_rate': ['constant', 'invscaling', 'adaptive'], # Only used when solver='sgd'.
            'learning_rate_init': Real(0.01, 0.1) if bayes else [0.01, 0.1],
            'max_iter': Integer(500, 1000) if bayes else [1000],
            'early_stopping': Categorical([True, False]) if bayes else [True]
        },

    }

    # Busco hiperpamateros default a probar
    params = d_params[model_name]

    if verbose >= 1:
        logger.info(f"Hiperparametros a probar: {params}")

    return params

# Iteraciones de BayesSearchCV
def estimate_bayes_iterations(param_space, scaling_factor=0.01, min_iters=10, max_iters=500):
    """
    Estima el número óptimo de iteraciones para BayesSearchCV en función del espacio de búsqueda.
    - param_space: diccionario con hiperparámetros en formato skopt (Integer, Real, Categorical)
    - scaling_factor: fracción de combinaciones a probar (ajustable según el problema)
    - min_iters: número mínimo de iteraciones
    - max_iters: número máximo de iteraciones
    """
    total_combinations = 1
    
    for param in param_space.values():
        if isinstance(param, Categorical):
            total_combinations *= len(param.categories)
        elif isinstance(param, Integer):
            total_combinations *= (param.high - param.low + 1)
        elif isinstance(param, Real):
            # Aproximamos la cantidad de valores que puede tomar usando 100 divisiones
            total_combinations *= min(100, int((param.high - param.low) / param.prior))
    
    # Aplicar la fórmula con el factor de escala
    estimated_iters = int(total_combinations * scaling_factor)
    
    return max(min_iters, min(estimated_iters, max_iters))

def estimate_bayes_iterations_2(num_samples, num_hyperparameters, mult_iter: float = 0.01, mult_hip: float = 200, min_iter: int = 20, max_iter: int = 60, verbose: int = 0):
    """
    Determina el número de iteraciones para BayesSearchCV basado en el tamaño del conjunto de datos
    y el número de hiperparámetros a optimizar.

    # Parameters
        num_samples: Número de muestras en el conjunto de datos. (int)
        num_hyperparameters: Número de hiperparámetros a optimizar. (int)

    # Return
        n_iter: Número sugerido de iteraciones (n_iter).
    """
    # Determinar n_iter basado en el tamaño del conjunto de datos
    n_iter_reg = int(mult_iter * num_samples)

    # Ajustar n_iter según el número de hiperparámetros
    n_iter_hip = int(mult_hip / num_hyperparameters)
    n_iter = n_iter_reg + n_iter_hip

    if verbose >= 1:
        logger.info(f"n_reg: {n_iter_reg} (={num_samples} * {mult_iter}) + n_hip: {n_iter_hip} (={mult_hip} / {num_hyperparameters} ) = {n_iter}")

    return min(max(n_iter, min_iter), max_iter) # Asegurarse de que n_iter sea al menos 50

# Cross validation manual
def manual_cross_validation(model, X_train, y_train, k=5):  # Funciona igual que la libreria (podria utilizar la libreria si quiero o no) # antes recibia X e y --> lo saque para hacer la division en train y test en generate test design
    """
    Realiza cross validation para evaluar el rendimiento del modelo entrenado.

    # Parameters
        model: Modelo de Machine Learning. (sklearn.ensemble)
        X_train: Dataframe de entrenamiento con variables predictoras. (DataFrame)
        y_train: Dataframe de entrenamiento solo con variable respuesta. (DataFrame)
        k: Numero de folds. (int)

    # Return
        cv_accuracy: Precision promedio de la validación cruzada. (float)
    """
    # Definicion de variables
    scores, rois = [], []
    fold_size = len(X_train) // k  # e.g 3000 / 5 = 600

    # Por k
    for i in range(k):

        # Dividir los datos en conjuntos de entrenamiento y test (confirme experimentalmente y por Chat GPT que esta division de folds no importa si el indice no es de 0 a len(df). Sin embargo, si importa que el df no tenga algun orden especifico puesto que si no los folds quedan desbalanceados)
        start, end = i * fold_size, (i + 1) * fold_size
        X_train_fold = np.concatenate((X_train[:start], X_train[end:]), axis=0)
        y_train_fold = np.concatenate((y_train[:start], y_train[end:]), axis=0)
        X_test_fold = X_train[start:end]
        y_test_fold = y_train[start:end]

        # Entrenar el modelo con el conjunto de entrenamiento de la iteración actual
        model.fit(X_train_fold, y_train_fold)

        # Realizar predicciones en el conjunto de test
        y_pred = model.predict(X_test_fold)

        # Calcular la precisión en el conjunto de test
        accuracy = accuracy_score(y_test_fold, y_pred) * 100
        scores.append(accuracy)
        # print(f'Fold {i+1} --> Precision: {accuracy:.1f}%')

    # Calcular la precisión promedio de la validación cruzada
    cv_accuracy = np.mean(scores)
    return cv_accuracy

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    pass