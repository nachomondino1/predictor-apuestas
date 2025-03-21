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

def select_best_hiperparameters(model, X_train, y_train, X_val, y_val, k, params: dict = None, bayes: bool = True, n_iter:int = None, 
                                scoring: bool = None, verbose: int = 1):
    """
    Selecciona los mejores hiperparametros para un modelo.

    # Parameters
        model: Modelo de Machine Learning. (sklearn.ensemble)
        X: Dataframe de validacion con variables predictoras. (DataFrame)
        y: Dataframe de validacion solo con variable respuesta. (DataFrame)
        k: Numero de folds. (int)
        params: Parametros a evaluar (dict)
        bayes: True para usar BayesSearchCV y false para usar GridSearchCV (bool)
        all_tunning: True para usar tanto BayesSearchCV como GridSearchCV y elegir el mejor (bool)
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
    scoring = default_scoring(target_type=var_resp) #  if scoring is None else scoring

    if verbose >= 1:
        logger.info(f"Seleccionando mejores hiperparametros para {model_name} con k={k}")

    # BayesSearch
    if bayes:
                
        if verbose >= 1: 
            logger.warning("BayesSearchCV...")
        start_bayes = time.time()

        if n_iter is None:
            # Calcular iteraciones basadas en el tamaño del dataset
            n_iter = 50 # Puedo usar estimate_bayes_iterations_2 o estimate_bayes_iterations 
            
        with warnings.catch_warnings():  # Logistic te vuelve loco --> no funciona.
            warnings.simplefilter("ignore")  # Ignora todas las advertencias

            # Crear el objeto BayesSearchCV
            bayes_search = BayesSearchCV(
                estimator=model, 
                search_spaces=params, 
                cv=pds, 
                n_iter=n_iter, 
                scoring=scoring,
                refit="cross_entropy_loss" if var_resp == 'categorical' else 'neg_mean_squared_error',  # O la métrica que prefieras
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
        with warnings.catch_warnings():  # Logistic te vuelve loco
            warnings.simplefilter("ignore")  # Ignora todas las advertencias
            grid_search = GridSearchCV(
                estimator=model, 
                param_grid=params_grid, 
                cv=pds,
                scoring=scoring, 
                refit=lambda cv_results: custom_refit(cv_results, target_type=var_resp), 
                n_jobs=-1  # el n_jons -1 evitaria el error  "warnings.warn(f"resource_tracker: {name}: {e!r}")"" (AUN NO LO PROBÉ, NO SE SI FUNCIONA)
                )

            # Ajustar el objeto GridSearchCV a los datos de entrenamiento
            grid_search.fit(X_val_train, y_val_train)  # Esta ok X_val_train y y_val_train
            best_search = grid_search         

        # Obtener los mejores hiperparámetros
        end_grid = time.time()
        if verbose >= 1: 
            logger.info(f"Seleccion de hiperparametros optimos con Grid en {(end_grid - start_grid) / 60:.1f} minutos")

    # Obtengo el mejor modelo (ya entrenado), los mejores hiper, la mejor metrica y los resultados de todas las combinaciones de hiper
    best_model = best_search.best_estimator_ 
    best_params = best_search.best_params_
    best_indice = best_search.best_index_ # Metricas de la mejor combinacion
    results = pd.DataFrame(data=best_search.cv_results_)  # Metricas de cada combinacion de params

    if var_resp == 'categorical':
        d_metrics = {
            'cv_accuracy': best_search.cv_results_['mean_test_accuracy'][best_indice],
            'cv_f1_score_wei': best_search.cv_results_['mean_test_f1_score_wei'][best_indice],
            'cv_f1_score': best_search.cv_results_['mean_test_f1_score'][best_indice],
            'cv_cross_entropy_loss': -best_search.cv_results_['mean_test_cross_entropy_loss'][best_indice]  # Negar porque invertimos el log_loss
            # Agregar medidas de desviacion estandar. --> Solo mirar el promedio de la validación cruzada puede ocultar problemas de inconsistencia entre pliegues. Incluye siempre la desviación estándar.
        }
    else:
        d_metrics = {
            'mean_test_neg_mean_squared_error':  best_search.cv_results_['mean_test_score'][best_indice]
        }

    # Mejores hiperparámetros
    if verbose >= 0:
        # print("Índice seleccionado por GridSearchCV:", best_search.best_index_)
        # print("Mejor pérdida de entropía cruzada encontrada:", best_search.cv_results_['mean_test_cross_entropy_loss'][grid_search.best_index_])
        logger.info(f"Mejor combinación de parameters: {best_params} \n Metricas de la mejor comb (scoring): {d_metrics} ")

    return best_model, best_params, d_metrics, results

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
        'categorical': 'mean_test_cross_entropy_loss',
        'continuous': 'mean_test_score'  # En lugar de 'mean_test_neg_mean_squared_error' 
    }

    key = metrics.get(target_type)

    if key not in cv_results:
        raise ValueError(f"La métrica {key} no está en los resultados: {cv_results.keys()}")

    if verbose >= 1:
        print(f"Metrica a usar para selec la mejor comb {target_type}: {key}")

    return cv_results[key].argmax() # # Mayor valor negativo (menor pérdida real) # CUIDADO con la metrica que uses para ver si usar argmax() o argmin()

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
            # 'f1_score_0': make_scorer(custom_scorer),
            'cross_entropy_loss': make_scorer(log_loss, greater_is_better=False, response_method="predict_proba")
        }

    elif target_type == 'continuous':
        scoring = 'neg_mean_squared_error'  # Regresión con MSE para variables continuas

    else:
        raise ValueError("target_type debe ser 'categorical' o 'continuous'.")

    if verbose >= 1:
        print(f"Target type: {target_type} --> Métrica por default: {scoring}")

    return scoring

# Definir un scorer que priorice la clase 0
def custom_scorer(y_true, y_pred):
    # F1-Score para la clase 0 (pos_label=0 para priorizar esa clase)
    f1_class_0 = f1_score(y_true, y_pred, labels=[0], average='micro')  # Solo clase 0
    
    # F1-Score general (macro promedio para todas las clases)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # Combinar los resultados favoreciendo a la clase 0
    # Puedes ajustar los pesos (ej., 70% clase 0, 30% macro)
    return 0.7 * f1_class_0 + 0.3 * f1_macro

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
    # Arboles de decision
    l_n_estimators = [10, 50, 100, 300]
    l_learning_rate = [0.001, 0.01, 0.1]
    l_max_depth = [3, 5, 10]
    l_min_samples_leaf = [2, 11, 21, 51] # [21, 51]
    l_min_samples_split = [(elem * 2) + 1 for elem in l_min_samples_leaf] # min_samples_split≥2×min_samples_leaf. P
    l_max_features = ["sqrt", "log2"]
    l_bootstrap = [True]

    # Logistic
    l_max_it = [10000] # aumentar max_iter no causa overfitting en LogisticRegression()
    param_c = [0.0001, 0.001, 0.01, 0.1, 1, 10] # [0.01, 1, 10]

    d_params = {

        # ARBOLES DE DECISION
        'DecisionTreeClassifier': {
            'criterion': Categorical(['entropy', 'gini']) if bayes else ['entropy', 'gini'],
            'splitter': Categorical(['random', 'best']) if bayes else ['random', 'best'], 
            'max_depth': Categorical(l_max_depth) if bayes else l_max_depth,
            'min_samples_split': Integer(2, 20) if bayes else l_min_samples_split, # Mayor o igual a 2
            'min_samples_leaf': Integer(2, 20) if bayes else l_min_samples_leaf, # Si usas 1 las probas van a ser 1-0-0, 0-1-0, 0-0-1, si usas 4 0.5-0.5-0 y asi.
            'max_features': Categorical(l_max_features) if bayes else l_max_features, # Real(0.1, 1.0)
        },
        'RandomForestClassifier': {
            'n_estimators': Integer(100, 500) if bayes else l_n_estimators,
            'criterion': Categorical(['entropy', 'gini']) if bayes else ['entropy', 'gini'],
            'max_depth': Categorical(l_max_depth) if bayes else l_max_depth,
            'min_samples_split': Integer(2, 21) if bayes else l_min_samples_split, # Mayor o igual a 2
            'min_samples_leaf': Integer(2, 21) if bayes else l_min_samples_leaf,
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
            'booster': Categorical(['gbtree', 'dart']) if bayes else ['gbtree'], # 'gbtree',
            'n_estimators': Integer(5, 120) if bayes else l_n_estimators,  # Suele ganar con 100
            'max_depth': Categorical(l_max_depth) if bayes else l_max_depth, # 3, 
            'learning_rate': Real(0.001, 1) if bayes else l_learning_rate,
            # 'min_child_weight': Integer(1, 10) if bayes else [1, 5], #5
            'grow_policy': Categorical(['depthwise', 'lossguide']) if bayes else ['depthwise', 'lossguide'],
            # 'verbosity': Categorical([1]) if bayes else [1] # 0 (silent), 1 (warning), 2 (info), and 3 (debug). Por default es 1.

            # Evitar si los datos ya estan balanceados
            # 'subsample': Real(0.8, 1.0) if bayes else [1.0], #  sample of the training data prior to growing trees
            # 'colsample_bylevel': Real(0.3, 1.0) if bayes else [1.0],
            # 'colsample_bytree': Real(0.3, 1.0) if bayes else [1.0],

            # Demasiada regularizacion puede llevar a predicciones uniformes 33-33-33.
            'gamma': Real(0, 1) if bayes else [0],
            'alpha': Real(0.001, 1) if bayes else [0.001],
            'lambda': Real(0.001, 1) if bayes else [0.001],
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
                'max_iter': Integer(100, max(l_max_it)) if bayes else l_max_it
                }, 
            {
                'penalty': Categorical(['l2']) if bayes else ['l2'], 
                'solver': Categorical(['newton-cg', 'lbfgs']) if bayes else ['newton-cg', 'lbfgs', 'sag'], 
                'C': Real(0.001, 10, prior='log-uniform') if bayes else param_c, 
                'max_iter': Integer(100, max(l_max_it)) if bayes else l_max_it
                },
            {
                'penalty': Categorical(['l1']) if bayes else ['l1'], 
                'solver': Categorical(['liblinear', 'saga']) if bayes else ['liblinear', 'saga'], 
                'C': Real(0.001, 10, prior='log-uniform') if bayes else param_c, 
                'max_iter': Integer(100, max(l_max_it)) if bayes else l_max_it
                },
            {
                'penalty': Categorical(['elasticnet']) if bayes else ['elasticnet'], 
                'solver': Categorical(['saga']) if bayes else ['saga'], 
                'C': Real(0.001, 10, prior='log-uniform') if bayes else param_c, 
                'l1_ratio': Real(0, 1) if bayes else [0.5], 
                'max_iter': Integer(100, max(l_max_it)) if bayes else l_max_it
                }
        ],
        'SVC': {
            'C': Real(0.01, 3) if bayes else [0.1, 0.5, 1, 3],
            'kernel': Categorical(['rbf', 'sigmoid']) if bayes else ['rbf', 'sigmoid'],
            'gamma': Categorical(['scale', 'auto']) if bayes else ['scale', 'auto'],
            'coef0': Real(0, 1) if bayes else [0.0, 0.5],
            'shrinking': Categorical([True, False]) if bayes else [True, False],
            'probability': Categorical([True]) if bayes else [True],
            'decision_function_shape': Categorical(['ovo', 'ovr']) if bayes else ['ovo', 'ovr']
        },

        # REDES NEURONALES
        'MLPClassifier': {
            # 'hidden_layer_sizes': Categorical([(50,), (100,)]) if bayes else [(50,), (100,)], # Tiene problema.
            'hidden_layer_sizes': Categorical([50, 100]) if bayes else [50, 100], # 50 va bien
            'activation': Categorical(['logistic', 'relu']) if bayes else ['logistic', 'relu'], # logistic va.
            'solver': Categorical(['adam']) if bayes else ['adam'],
            'alpha': Real(0.0001, 0.001) if bayes else [0.0001, 0.001],
            'learning_rate_init': Real(0.01, 0.1) if bayes else [0.01, 0.1],
            'max_iter': Integer(500, 1000) if bayes else [500, 1000],
            'early_stopping': Categorical([True, False]) if bayes else [True, False]
        },

        # OTROS 2
        'PCA': {
            'n_components': [None, 2, 3, 4, 5, 8, 10, 15],  # Si gana None, elimina 1 sola variable... # Número de componentes principales a mantener
            'whiten': [False, True],  # Indica si aplicar blanqueamiento de los datos
            'svd_solver': ['auto', 'full', 'arpack', 'randomized'], # Algoritmo de descomposición SVD a utilizar
            'iterated_power': [0, 1, 2],  # Número de veces que se aplica el método de la potencia iterada
            'tol': [0.0, 0.001, 0.01],  # Tolerancia para la convergencia del algoritmo
            'copy': [True, False]  # Copiar los datos de entrada o modificarlos en su lugar
        },
        'Lasso': {
            'alpha': [0.1, 1.0, 10.0],  # Parámetro de regularización que controla la fuerza de la penalización L1. Un valor más alto de alpha produce una mayor regularización y puede conducir a una selección más agresiva de características.
            'fit_intercept': [True, False],  # Indica si se debe ajustar un intercepto (término independiente) en el modelo.
            'precompute': [True, False],  # Indica si se deben precalcular las matrices de productos internos para acelerar el ajuste del modelo.
            'max_iter': [100, 500, 1000],  # Número máximo de iteraciones para converger durante el ajuste del modelo.
            'positive': [True, False],  # Indica si se deben restringir los coeficientes a ser solo valores no negativos.
            'selection': ['cyclic', 'random']  # Método de selección de características. 'cyclic' utiliza el orden cíclico de las características para ajustar el modelo, mientras que 'random' selecciona aleatoriamente características en cada iteración.
        },
    }

    # Busco hiperpamateros default a probar
    params = d_params[model_name]

    if verbose >= 1:
        logger.info(f"Hiperparametros a probar: {params}")

    return params

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

'''
# Comparacion de BayesSearchCV y GridSearchCV
def compare_tunners(bayes_search, grid_search, verbose: int = 0):
    """
    Comparacion de mejores modelos de cada search y seleccion del mejor.

    # Parameters:
        bayes_search: Elemento BayesSearchCV con mejor modelo, mejores hiperparametros, metrics por combinacion de hiper, etc. (BayesSearchCV)
        grid_search: Elemento GridSearchCV con mejor modelo, mejores hiperparametros, metrics por combinacion de hiper, etc. (GridSearchCV)
    
    # Return
        best_search: Elemento GridSearchCV / BayesSearchCV segun el que sea mejor. (BayesSearchCV o GridSearchCV)
    """
    # Hago Bayes y Grid
    metric1 = bayes_search.best_score_
    metric2 = grid_search.best_score_

    # Comparacion y seleccion del mejor
    if metric1 > metric2:
        dif = (metric1 - metric2) / abs(metric2) * 100
        ganador, best_search = "BayesSearch", bayes_search
        if verbose >= 1:
            logger.critical(f"M1: {metric1} y M2: {metric2} --> Ganador: {ganador} por {dif:.0f}%.")
    else:
        dif = (metric2 - metric1) / abs(metric1) * 100
        ganador, best_search= "GridSearch", grid_search
        if verbose >= 1:
            logger.error(f"M1: {metric1} y M2: {metric2} --> Ganador: {ganador} por {dif:.0f}%.")

    return best_search

def compare_scoring_methods(model, X_train, y_train, X_val, y_val, k, params: dict = None, bayes: bool = True, n_iter:int = None, #
                                scoring: bool = None, all_tuning: bool = False, verbose: int = 1):
    
    """
    Determinar mejor scoring method para seleccionar los hiperparametros optimos
    """
    results, models = [], []
    if len(np.unique(y_train)) <= 5:
        scoring_methods = {
            "f1_macro": "f1_macro",
            "f1_micro": "f1_micro",
            "f1_weighted": "f1_weighted",
            "log_loss": "neg_log_loss",
            "accuracy": "accuracy",
        }
    else:
        scoring_methods = {"neg_mean_squared_error": "neg_mean_squared_error"}

    # Por metrica
    for name, scoring in scoring_methods.items():
        logger.info(f"Metrica: {scoring}")

        # Selecciono mejor combinacion de hiperparametros segun metrica
        best_estim, best_params, best_score, cv_res = select_best_hiperparameters(model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k, bayes=bayes, scoring=scoring, verbose=1)

        # Evaluar en el conjunto de validación
        y_val_pred = best_estim.predict(X_val)
        y_val_pred_proba = best_estim.predict_proba(X_val)

        # Calcular métricas en el conjunto de validación
        val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
        val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted")
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_log_loss = log_loss(y_val, y_val_pred_proba)
        
        # Registro de resultados
        results.append({
            "scoring": name,
            'val_f1': val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
            "val_accuracy": val_accuracy,
            "val_loss": val_log_loss,
            # "best_score_val": best_score,
            # 'best_model': best_estim,
            # "best_params": best_params
        })

        models.append({
            "scoring": name,
            "best_score_val": best_score,
            'best_model': best_estim,
            "best_params": best_params
        })

    # Calculo metrica combinada entre f1_score y val_loss
    results = combined_metric(results)

    # Aquí podrías normalizar los valores o ponderar las métricas, si es necesario
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='combined_metric', ascending=True)
    models_df = pd.DataFrame(models)
    logger.info("\nComparación de Métricas:\n" + results_df.to_string())

    # Buscar la mejor combinación de hiperparámetros según la métrica (por ejemplo, accuracy)
    best_scoring_row = results_df.loc[results_df["combined_metric"].idxmin()]
    best_scoring = best_scoring_row['scoring']
    logger.critical(f"Mejor Scoring: {best_scoring}")

    # Obtener el mejor modelo y parámetros
    row = models_df[models_df['scoring'] == best_scoring].iloc[0]  # Usa iloc[0] para obtener la primera (y única) fila
    best_model = row['best_model']
    best_score = row['best_score_val']
    best_params = row['best_params']
    logger.info(f"Model: {best_model} Score: {best_score} Params: {best_params}")

    # results_df.to_excel(f"/Users/nachomondino/Desktop/hiperparametros.xlsx")
    return best_model, best_params, best_score, results_df # best_scoring

def combined_metric(results):
    # Obtener los valores mínimo y máximo de cada métrica para escalar y evitar divisiones por cero
    min_loss, max_loss = min(r['val_loss'] for r in results), max(r['val_loss'] for r in results)
    min_f1, max_f1 = min(r['val_f1'] for r in results), max(r['val_f1'] for r in results)
    
    # Si max_loss == min_loss, el rango sería cero; en tal caso, forzamos el denominador a 1
    loss_range = max_loss - min_loss if max_loss > min_loss else 1
    f1_range = max_f1 - min_f1 if max_f1 > min_f1 else 1

    # Crear una métrica combinada normalizada en results
    for result in results:
        normalized_loss = (result['val_loss'] - min_loss) / loss_range
        normalized_f1 = (result['val_f1'] - min_f1) / f1_range
        result['combined_metric'] = normalized_loss - normalized_f1  # Minimizar esta métrica
    
    return results
'''