import pandas as pd
import numpy as np
import time
from itertools import product
from set_up_logging import logger
import warnings
# Grid y Bayes
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, log_loss
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
# Red neuronal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# from scikeras.wrappers import KerasClassifier


class TrainNeuralNetwork():
    
    def __init__(self) -> None:
        pass

    def create_neural_network(self, input_shape, output_shape, activation='relu', hidden_layer_sizes=[50, 100], 
                          optimizer='adam', learning_rate=0.001, kernel_regularizer=0.001, 
                          batch_normalization=False, dropout_rate=None, metrics=['accuracy', Precision(), Recall()], 
                          verbose: int = 0):
        """
        Creación de la arquitectura de la red neuronal y del modelo.

        # Parameters
            input_shape: Número de neuronas en la capa de entrada, igual al número de variables predictoras (int).
            output_shape: Número de neuronas en la capa de salida, generalmente 1 para regresión o igual al número de clases en clasificación (int).
            activation: Función de activación para las capas ocultas (str, default='relu').
                Ejemplos: 'relu', 'sigmoid', 'tanh', 'softmax', etc.
            hidden_layer_sizes: Lista que define la cantidad de neuronas en cada capa oculta (list).
                Ejemplo: [50, 100] crea dos capas ocultas con 50 y 100 neuronas respectivamente.
            optimizer: Optimizador utilizado para entrenar la red (str o keras.optimizers).
                Ejemplo: 'adam', 'sgd', 'rmsprop', o un objeto de optimizador de Keras.
            learning_rate: Tasa de aprendizaje para el optimizador (float, default=0.001).
            kernel_regularizer: Coeficiente de regularización L2 para reducir el sobreajuste (float, default=0.001).
            batch_normalization: Si es True, añade capas de normalización por lotes después de cada capa oculta (bool, default=False).
            dropout_rate: Proporción de neuronas a desactivar en cada capa durante el entrenamiento para evitar sobreajuste (float, default=None).
                Ejemplo: dropout_rate=0.5 mantendrá el 50% de las neuronas activas en cada paso.
            metrics: Lista de métricas para evaluar el rendimiento del modelo (list, default=['accuracy', Precision(), Recall()]).

        # Return
            model: El modelo de red neuronal compilado y listo para entrenarse.
        """
        model = Sequential()
        if verbose >= 1:
            logger.info(f"Parametros a probar: {input_shape} {output_shape} {activation} {hidden_layer_sizes} {optimizer} {learning_rate} {kernel_regularizer} {batch_normalization} {dropout_rate} {metrics}")
        
        # First layer
        model.add(Input(shape=(input_shape,)))
  
        # Por Hidden layers
        for neurons in hidden_layer_sizes:
            model.add(Dense(neurons, activation=activation, kernel_regularizer=l2(kernel_regularizer)))
            
            # Dropout opcional en capas ocultas
            if dropout_rate:
                model.add(Dropout(dropout_rate))
            
            if batch_normalization:
                model.add(BatchNormalization())

        # Output layer (Capa de salida con softmax para clasificación multiclase)
        model.add(Dense(output_shape, activation='softmax'))

        # Configuración del optimizador
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported")

        # Compilación del modelo
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
        
        return model
     
    def select_best_arquitecture(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose: int = 0):
        """
        Entrena múltiples arquitecturas de redes neuronales y selecciona la mejor según su desempeño en los datos de validación.

        # Parameters
            X_train: Conjunto de entrenamiento para las variables predictoras (DataFrame o array).
            y_train: Conjunto de entrenamiento para la variable respuesta (DataFrame o array).
            X_val: Conjunto de validación para las variables predictoras, usado para evaluar el rendimiento (DataFrame o array).
            y_val: Conjunto de validación para la variable respuesta (DataFrame o array).
            epochs: Número de épocas para entrenar cada arquitectura de red (int, default=20).
            batch_size: Tamaño de los lotes para el entrenamiento (int, default=32).
            verbose: Nivel de detalle de la salida durante el entrenamiento; 0 = silencioso, 1 = detallado, 2 = una barra de progreso por época (int, default=0).

        # Return
            best_model: Modelo de red neuronal con el mejor rendimiento en los datos de validación.
            best_params: Diccionario con los parámetros de la arquitectura seleccionada.
            best_score: Mejor puntaje obtenido en los datos de validación (ej. accuracy, F1, etc., dependiendo de la métrica definida).
        """
        start = time.time()

        # Lista para guardar los resultados y modelos
        results = []

        # Hiperparametros de arquitectura
        param_grid = { 
            'hidden_layer_sizes': [[50], [64, 32], [128, 64], [128, 64, 32]], # , [100, 50], [64, 32], [100, 100], [256, 128, 64], [1024, 512, 256],  [512, 256, 128, 64] (no gana y encima creo que es la causa del kill...)
            'learning_rate': [0.01, 0.1],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam'],
            'kernel_regularizer': [None, 0.01],
            'batch_normalization': [False],
            'dropout_rate': [0.2, None]
        }

        # Generar combinaciones de parámetros automáticamente
        param_combinations = list(product(*param_grid.values()))

        # Convertir etiquetas a formato one-hot --> Evita error target y output con different shape. 
        input_shape, output_shape = X_train.shape[1], 3  # Estaria bueno que sea automatico
        patience = int(epochs * 0.25)  # Por ejemplo, 20% de las épocas totales
        y_val_categorical = to_categorical(y_val, num_classes=output_shape)
        y_train_categorical = to_categorical(y_train, num_classes=output_shape)

        # Definir un callback de EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        # timeout_callback = TimeoutCallback(max_seconds=100)  # Definir el límite de tiempo (por ejemplo, 300 segundos) # No se si funciona y tampoco creo que esta sea la causa de que tarde mucho tiempo.

        # Iterar sobre cada combinación
        for params in param_combinations:
            
            # Emparejar cada parámetro con su nombre desde `param_grid`
            param_dict = dict(zip(param_grid.keys(), params))
            # logger.warning(param_dict)

            # Creo red neuronal
            model = self.create_neural_network(
                input_shape=input_shape,
                output_shape=output_shape,
                **param_dict  # Desempaqueta el diccionario como argumentos nombrados
            )

            # Entrenar el modelo (usa el validation como test en vez de hacer cross val entre X_train)
            history = model.fit(X_train, y_train_categorical, validation_data=(X_val, y_val_categorical), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stopping]) # timeout_callback
            
            # Evaluar en el set de validación
            val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val_categorical, verbose=verbose)
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0  # Accuracy NO.

            results.append({
                'params': params,
                **param_dict,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1, 
                'model': model,
                # 'history': history # &lt;keras.src.callbacks.history.History object at 0x34f4a3e30&gt;
            })

        # Calculo metrica combinada entre f1_score y val_loss
        results = combined_metric(results)

        # Buscar la mejor combinación de hiperparámetros según la métrica (por ejemplo, accuracy)
        best_result = min(results, key=lambda x: x['combined_metric'])  # best_result = min(results, key=lambda x: x['val_loss'])  # best_result = max(results, key=lambda x: x['val_f1']) 

        # Imprimo rdos
        end = time.time()

        if verbose >= 0:
            logger.info(f"Best arquitecture: {best_result['params']}")
            logger.info(f"Metrics: Val loss: {best_result['val_loss']}  Val Accuracy: {best_result['val_acc']} Val f1: {best_result['val_f1']}")
            logger.info(f"\tSeleccion de hiperparametros optimos en {(end - start) / 60:.1f} minutos")

        return best_result['model'], best_result['params'], best_result['val_acc'], pd.DataFrame(results)
    
class TimeoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds):
        super(TimeoutCallback, self).__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        # Registrar el tiempo de inicio del entrenamiento
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Verificar el tiempo transcurrido
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_seconds:
            print(f'\nEntrenamiento detenido: tiempo máximo de {self.max_seconds} segundos alcanzado.')
            self.model.stop_training = True


def select_best_hiperparameters(model, X_train, y_train, X_val, y_val, k, params: dict = None, bayes: bool = True, n_iter:int = None, 
                                scoring: bool = None, all_tuning: bool = False, verbose: int = 1):
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
    bayes, all_tuning = (False, False) if model_name == 'LogisticRegression' else (bayes, all_tuning) # Seteo Bayes a False cuando es Logistic. Evito Bayes para Logistic
    params = space_params(model_name, bayes) if params is None else params
    scoring = default_scoring(num_classes=len(np.unique(y_val_train))) if scoring is None else scoring
    logger.info(f"Seleccionando mejores hiperparametros para {model_name} con k={k}")

    # BayesSearch
    if bayes or all_tuning:
                
        if verbose >= 1: 
            logger.warning("BayesSearchCV...")
        start_bayes = time.time()

        if n_iter is None:
            # Calcular iteraciones basadas en el tamaño del dataset
            num_hyperparameters = len(params.keys())
            n_iter = determine_n_iter(num_samples=len(X_val_train), num_hyperparameters=num_hyperparameters, verbose=verbose)

        with warnings.catch_warnings():  # Logistic te vuelve loco --> no funciona.
            warnings.simplefilter("ignore")  # Ignora todas las advertencias

            # Crear el objeto BayesSearchCV
            bayes_search = BayesSearchCV(estimator=model, search_spaces=params, cv=pds, n_iter=n_iter, n_jobs=-1, scoring=scoring)

            # Ajustar el objeto BayesSearchCV a los datos de entrenamiento
            bayes_search.fit(X_val_train, y_val_train)
            best_search = bayes_search

        end_bayes = time.time()
        if verbose >= 1:
            logger.info(f"Seleccion de hiperparametros optimos con Bayes en {(end_bayes - start_bayes) / 60:.1f} minutos")

    # GridSearch
    if (not bayes) or all_tuning:
        
        if verbose >= 1: 
            logger.warning("GridSearchCV...")

        start_grid = time.time()
        params_grid = space_params(model_name, bayes=False)

        # Crear el objeto GridSearchCV
        with warnings.catch_warnings():  # Logistic te vuelve loco
            warnings.simplefilter("ignore")  # Ignora todas las advertencias
            grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=pds, scoring=scoring)

            # Ajustar el objeto GridSearchCV a los datos de entrenamiento
            grid_search.fit(X_val_train, y_val_train)  # Esta ok X_val_train y y_val_train
            best_search = grid_search
 
        # Obtener los mejores hiperparámetros
        end_grid = time.time()
        if verbose >= 1: 
            logger.info(f"Seleccion de hiperparametros optimos con Grid en {(end_grid - start_grid) / 60:.1f} minutos")

        # Selecciono mejor modelo de todos los tunner
        if all_tuning:
            best_search = compare_tunners(bayes_search, grid_search, verbose=verbose)

    # Obtengo el mejor modelo (ya entrenado), los mejores hiper, la mejor metrica y los resultados de todas las combinaciones de hiper
    best_model = best_search.best_estimator_ 
    best_params = best_search.best_params_
    best_metric = best_search.best_score_
    results = pd.DataFrame(data=best_search.cv_results_)

    # Expansion de space de params
    # if verbose >= 0:
    #     check_best_params_limits(best_params, params) # probar, no funca aun.

    # Mejores hiperparámetros
    if verbose >= 1:
        logger.info(f"Best score: {best_metric*100:.1f}. Best parameters: {best_params}")

    #  Metricas para cada combinacion de hiperparametros
    if verbose >= 2:
        results.to_excel(f"/Users/nachomondino/Desktop/hiperparametros.xlsx")    

    return best_model, best_params, best_metric, results

def default_scoring(num_classes, verbose: int = 0):
    """
    Asigna un valor default a scoring según la variable respuesta.

    # Parameters
        num_classes: Cantidad de clases de variable respuesta.
    
    # Return
        scoring: Métrica a utilizar en evaluación para determinar la mejor combinación de hiperparámetros.
    """
    # Variable respuesta discreta
    if num_classes <= 5:
        scoring = 'f1_macro'  # Usar métrica combinada en clasificación multiclase con <= 5 clases
    # Variable respuesta continua
    else:
        scoring = 'neg_mean_squared_error'  # Regresión con MSE para variables continuas
    
    if verbose >= 1:
        logger.info(f"Nº clases: {num_classes} --> Metrica por default: {scoring}")

    return scoring

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
    # Logistic
    min_it, max_it = 100, 2000

    d_params = {
        'DecisionTreeClassifier': {
            'criterion': ['entropy'],
            'splitter': ['random', 'best'],
            'max_depth': [None, 5, 7, 8, 10, 12, 14],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto'],
        },
        'RandomForestClassifier': {
            'n_estimators': Integer(100, 500) if bayes else [100, 500],
            'criterion': Categorical(['entropy', 'gini']) if bayes else ['entropy', 'gini'],
            'max_depth': Integer(3, 20) if bayes else [3, 5, 7, 10],
            # 'min_samples_split': Integer(2, 10) if bayes else [2, 10], # Mayor o igual a 2
            # 'min_samples_leaf': Integer(5, 50) if bayes else [1, 4],
            'max_features': Categorical(['sqrt', 'log2']) if bayes else ['sqrt', 'log2'], # Real(0.1, 1.0)
            'bootstrap': Categorical([True]) if bayes else [True] # False
        },
        'XGBClassifier': {
            'booster': Categorical(['gbtree', 'dart']) if bayes else ['gbtree'], # 'gbtree',
            'n_estimators': Integer(50, 120) if bayes else [100],  # Suele ganar con 100
            'learning_rate': Real(0.001, 1) if bayes else [0.001, 0.1],                 # 'learning_rate': Real(0.0001, 0.1) if bayes else [0.001, 0.01, 0.1],
            'max_depth': Integer(3, 20) if bayes else [5, 10], # 3, 
            'gamma': Real(0, 1) if bayes else [0],
            'min_child_weight': Integer(1, 10) if bayes else [1, 5], #5
            'subsample': Real(0.8, 1.0) if bayes else [1.0], #  sample of the training data prior to growing trees
            'alpha': Real(0, 10) if bayes else [0.001],
            'lambda': Real(0, 10) if bayes else [0.001],
            'colsample_bylevel': Real(0.3, 1.0) if bayes else [1.0],
            'colsample_bytree': Real(0.3, 1.0) if bayes else [1.0],
            'grow_policy': Categorical(['depthwise', 'lossguide']) if bayes else ['depthwise', 'lossguide'],
            # 'verbosity': Categorical([1]) if bayes else [1] # 0 (silent), 1 (warning), 2 (info), and 3 (debug). Por default es 1.
        },
        'GradientBoostingClassifier': { # Tarda muchisimo en entrenar a pesar de usar Bayes optimazation
            'n_estimators': Integer(100, 300) if bayes else [100], 
            'learning_rate': Real(0.001, 0.1) if bayes else [0.001, 0.01, 0.1],
            'max_depth': Integer(3, 12) if bayes else [3, 5],
            'min_samples_split': Integer(2, 8) if bayes else [2, 5, 10],  # Minimo 2.
            'min_samples_leaf': Integer(2, 5) if bayes else [2, 4], 
            'subsample': Real(0.7, 1.0) if bayes else [0.8, 1.0],
            # 'max_features': ['auto'],
            # 'loss': ['deviance']
        },
        'LogisticRegression': [
            {'penalty': Categorical([None]) if bayes else [None], 'solver': Categorical(['newton-cg', 'lbfgs']) if bayes else ['newton-cg', 'lbfgs', 'sag'], 'max_iter': Integer(min_it, max_it) if bayes else [1000]}, # 'sag' --> ConvergenceWarning (talvez por la escala)
            {'penalty': Categorical(['l2']) if bayes else ['l2'], 'solver': Categorical(['newton-cg', 'lbfgs']) if bayes else ['newton-cg', 'lbfgs', 'sag'], 'C': Real(0.01, 10, prior='log-uniform') if bayes else [0.1, 1, 10], 'max_iter': Integer(min_it, max_it) if bayes else [1000]}, # , 'sag' --> ConvergenceWarning (talvez por la escala)
            {'penalty': Categorical(['l1']) if bayes else ['l1'], 'solver': Categorical(['liblinear', 'saga']) if bayes else ['liblinear', 'saga'], 'C': Real(0.01, 10, prior='log-uniform') if bayes else [0.1, 1, 10], 'max_iter': Integer(min_it, max_it) if bayes else [1000]},
            {'penalty': Categorical(['elasticnet']) if bayes else ['elasticnet'], 'solver': Categorical(['saga']) if bayes else ['saga'], 'C': Real(0.01, 10, prior='log-uniform') if bayes else [0.1, 1, 10], 'l1_ratio': Real(0, 1) if bayes else [0.5], 'max_iter': Integer(min_it, max_it) if bayes else [1000]}
        ],
        'SVC': {
            'C': Real(0.01, 3) if bayes else [0.1, 0.5, 1],
            'kernel': Categorical(['rbf', 'sigmoid']) if bayes else ['rbf', 'sigmoid'],
            'gamma': Categorical(['scale', 'auto']) if bayes else ['scale', 'auto'],
            'coef0': Real(0, 1) if bayes else [0.0, 0.5],
            'shrinking': Categorical([True, False]) if bayes else [True, False],
            'probability': Categorical([True]) if bayes else [True],
            # 'class_weight': Categorical(['balanced', None]) if bayes else ['balanced', None],
            'decision_function_shape': Categorical(['ovo', 'ovr']) if bayes else ['ovo', 'ovr']
        },
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
        'RandomForestRegressor': {
            'bootstrap': Categorical([True]) if bayes else [True], # False
            'criterion': Categorical(["friedman_mse"]) if bayes else ["friedman_mse"], # "squared_error", "absolute_error"
            'max_depth': Integer(3, 30) if bayes else [5, 10],  # 30 # 3
            'n_estimators': Integer(100, 300) if bayes else [100, 300],
            # 'min_samples_split': Integer(5, 50) if bayes else [10, 50] # Mayor o igual a 2 
            # 'verbose': Categorical([0]) if bayes else [0]
        }
    }

    # Busco hiperpamateros default a probar
    params = d_params[model_name]

    if verbose >= 1:
        logger.info(f"Hiperparametros a probar: {params}")

    return params

def determine_n_iter(num_samples, num_hyperparameters, min_iter: int = 20, max_iter:int = 100, verbose: int = 0):
    """
    Determina el número de iteraciones para BayesSearchCV basado en el tamaño del conjunto de datos
    y el número de hiperparámetros a optimizar.

    # Parameters
        num_samples: Número de muestras en el conjunto de datos. (int)
        num_hyperparameters: Número de hiperparámetros a optimizar. (int)

    # Return
        n_iter: Número sugerido de iteraciones (n_iter).
    """
    mult_iter, mult_hip = 0.01, 200

    # Determinar n_iter basado en el tamaño del conjunto de datos
    n_iter_reg = int(mult_iter * num_samples)

    # Ajustar n_iter según el número de hiperparámetros
    n_iter_hip = int(mult_hip / num_hyperparameters)
    n_iter = n_iter_reg + n_iter_hip

    if verbose >= 1:
        logger.info(f"n_reg: {n_iter_reg} (={num_samples} * {mult_iter}) + n_hip: {n_iter_hip} (={mult_hip} / {num_hyperparameters} ) = {n_iter}")

    return min(max(n_iter, min_iter), max_iter) # Asegurarse de que n_iter sea al menos 50

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