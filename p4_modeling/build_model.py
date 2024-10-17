import numpy as np
from sklearn.model_selection import GridSearchCV  # Seleccion de hiperparametros
from sklearn.metrics import accuracy_score  # Metrica de precision
import time
# Red neuronal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.metrics import Recall
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# from scikeras.wrappers import KerasClassifier
from itertools import product
from set_up_logging import logger

class NeuralNetwork():
    
    def __init__(self) -> None:
        pass

    def create_neural_network(self, input_shape, output_shape, activation='relu', hidden_layer_sizes=[50, 100], 
                          optimizer='adam', learning_rate=0.001, kernel_regularizer=0.001, 
                          batch_normalization=False, dropout_rate=None, metrics=['accuracy']):
        """
        Creación de la arquitectura de la red neuronal y del modelo.
        """
        model = Sequential()
        
        # First layer
        model.add(Input(shape=(input_shape,)))
        model.add(Dense(hidden_layer_sizes[0], activation=activation, kernel_regularizer=l2(kernel_regularizer)))
        
        # Dropout opcional en la primera capa
        if dropout_rate:
            model.add(Dropout(dropout_rate))

        # Hidden layers
        for neurons in hidden_layer_sizes[1:]:
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
            opt = SGD(learning_rate=learning_rate, momentum=0.9)  # Agregamos momentum
        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported")

        # Compilación del modelo
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
        
        return model
     
    def select_best_arquitecture(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        Entrenamiento de redes neuronales y seleccion de la mejor
        """
        start = time.time()

        # Lista para guardar los resultados y modelos
        results = []

        # Hiperparametros de arquitectura
        param_grid = { 
                    'hidden_layer_sizes': [[100], [128, 64], [128, 64, 32], [256, 128, 64]], # [100], [50], [100, 50], [64, 32], [100, 100], [1024, 512, 256],  [512, 256, 128, 64] (no gana y encima creo que es la causa del kill...)
                    'learning_rate': [0.01, 0.1], # 0.001,
                    'activation': ['relu'], #  'tanh'
                    'optimizer': ['adam'], #  'sgd']
                    'kernel_regularizer': [None, 0.01], # 0.001,
                    'batch_normalization': [False], # True
                    'dropout_rate': [0.2, None]
                }
        
        # Generar todas las combinaciones de hiperparámetros
        param_combinations = list(product(
            param_grid['hidden_layer_sizes'],
            param_grid['learning_rate'],
            param_grid['activation'],
            param_grid['optimizer'],
            param_grid['kernel_regularizer'],
            param_grid['batch_normalization'],
            param_grid['dropout_rate']
        ))

        # Convertir etiquetas a formato one-hot --> Evita error target y output con different shape. 
        input_shape = X_train.shape[1]
        output_shape = 3  # Estaria bueno que sea automatico
        y_val_categorical = to_categorical(y_val, num_classes=output_shape)
        y_train_categorical = to_categorical(y_train, num_classes=output_shape)

        # Iterar sobre cada combinación
        for params in param_combinations:
            hidden_layers, learning_rate, activation, optimizer, regularizer, batch_norm, dropout_rate = params
            
            # Creo red neuronal
            model = self.create_neural_network(input_shape=input_shape, output_shape=output_shape, hidden_layer_sizes=hidden_layers, learning_rate=learning_rate, activation=activation, optimizer=optimizer, kernel_regularizer=regularizer, batch_normalization=batch_norm, dropout_rate=dropout_rate)

            # Definir un callback de EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Entrenar el modelo (usa el validation como test en vez de hacer cross val entre X_train)
            history = model.fit(X_train, y_train_categorical, validation_data=(X_val, y_val_categorical), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stopping]) # verbose=0 para no imprimir epochs
            
            # Evaluar en el set de validación
            val_loss, val_acc = model.evaluate(X_val, y_val_categorical, verbose=0)
            
            # Guardar el resultado
            results.append({
                'params': params,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'model': model
            })
            
            # print(f"Params: {params} => Val Loss: {val_loss}, Val Accuracy: {val_acc}")

        # Buscar la mejor combinación de hiperparámetros según la métrica (por ejemplo, accuracy)
        best_result = min(results, key=lambda x: x['val_loss'])

        # Imprimo rdos
        end = time.time()
        logger.info(f"Params: {best_result['params']} =>  Val loss: {best_result['val_loss']}  Val Accuracy: {best_result['val_acc']}")
        logger.info(f"\tSeleccion de hiperparametros optimos en {(end - start) / 60:.1f} minutos")

        return best_result['model'], best_result['params'], best_result['val_acc']

def select_best_hiperparameters(model, X, y, k, params: dict = None, _print: bool = False):
    """
    Selecciona los mejores hiperparametros para un modelo.

    :param model: Modelo de Machine Learning. (sklearn.ensemble)
    :param X: Dataframe de validacion con variables predictoras. (DataFrame)
    :param y: Dataframe de validacion solo con variable respuesta. (DataFrame)
    :param k: Numero de folds. (int)
    :return: Modelo actualizado con los hiperparametros optimos pero aun sin ajustar. (sklearn.ensemble)  # sklearn.ensemble._forest.RandomForestClassifier
    """
    if _print:
        start = time.time()
        print(f"\nSeleccionando mejores hiperparametros para {model} con k={k}")

    # Definicion de hiperparametros a considerar para cada modelo
    if params is None:
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
                'n_estimators': [100, 500],
                'criterion': ['entropy'],
                'max_depth': [3, 5, 7], 
                'min_samples_split': [2, 10], 
                'min_samples_leaf': [1, 4],
                'max_features': ['sqrt'],
                'bootstrap': [True],
            },
            'XGBClassifier': {
                'n_estimators': [100], # suele ganar 100 
                'learning_rate': [0.001, 0.01, 0.1], 
                'max_depth': [3, 4, 5, 6, 10], #  15, 20
                'gamma': [0, 0.5],
                'min_child_weight': [1, 5, None],        # Peso mínimo del niño
                'subsample': [0.8, 1.0],                 # Tasa de muestreo
                'colsample_bytree': [0.8, 1.0]           # Fracción de columnas por árbol
                # 'reg_alpha': [0, 0.1], # 0.01,
                # 'reg_lambda': [0, 0.1], # 0.01,
            },
            'GradientBoostingClassifier': { # Tarda muchisimo en entrenar.
                'n_estimators': [100], # 200
                'learning_rate': [0.01], #  0.1
                'max_depth': [3, 5],
                # 'min_samples_split': [1, 5, 10],
                # 'min_samples_leaf': [2, 4],
                # 'subsample': [0.8, 1.0],
                # 'max_features': ['auto'],
                # 'loss': ['deviance']
            },
            'LogisticRegression': {
                'penalty': ['l1', 'l2'], # 'l1' solo usa solver 'saga'
                'C': [0.1, 0.5, 1], # 5
                'solver': ['saga', 'liblinear'], #  'newton-cg', 'sag', 'lbfgs'
                'fit_intercept': [True, False], 
                'max_iter': [2000],
                # 'multi_class': ['auto'],
                # 'class_weight': ['balanced', None], # hace un under basicamente... no tiene sentido cuando hago under creo.
            },
            'SVC': {
                'C': [0.1, 0.5, 1],
                'kernel': ['rbf', 'sigmoid'], # 'poly',
                'gamma': ['scale', 'auto'], 
                # 'degree': [3, 5],
                'coef0': [0.0,  0.5], # , 1.0
                'shrinking': [True, False], # False
                'probability': [True], # Tiene que ser True para poder usar predict.proba()
                # 'tol': [1e-3, 1e-4, 1e-5],
                'class_weight': ['balanced', None], # No deberia usarlo porque ya balanceo pero es que tal vez es diferente...?
                'decision_function_shape': ['ovo', 'ovr'],
            },
            'MLPClassifier': {
                'hidden_layer_sizes': [(50,), (100,)], 
                'activation': ['logistic',  'relu'],  #  'tanh'
                'solver': ['adam'], # 'sgd', 'lbfgs'
                'alpha': [0.0001, 0.001], #  0.01
                # 'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.01, 0.1],  # 0.001
                'max_iter': [500],
                'early_stopping': [True] # False
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
                'n_estimators': [100, 200, 500],  # Número de árboles en el bosque
                # 'criterion': ['friedman_mse', 'absolute_error'],  # Criterio de división de los nodos del árbol (Mean Squared Error o Mean Absolute Error)
                'max_depth': [3, 5, 10],  # Profundidad máxima de los árboles
                # 'min_samples_split': [2, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
                # 'min_samples_leaf': [1, 4],  # Número mínimo de muestras requeridas en cada hoja del árbol
                'bootstrap': [True],  # Si se utiliza o no bootstrap para muestreo de datos
            }
        }
        
        # Obtengo el nombre del modelo para poder buscar sus hiperparametros
        model_name = str(model)[:str(model).find('(')]
        # print(f"Hiperparametros a probar: {params[model_name]}")

        # Busco hiperpamateros default a probar
        params = d_params[model_name]

    # Crear el objeto GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=k)

    # Ajustar el objeto GridSearchCV a los datos de entrenamiento
    grid_search.fit(X, y)

    # Obtener los mejores hiperparámetros
    best_params = grid_search.best_params_

    # Actualizar los hiperparámetros de model con los mejores hiperparámetros encontrados
    model.set_params(**best_params)
    if _print:
        end = time.time()
        print("\tMejores hiperparametros:", model)
        print(f"\tSeleccion de hiperparametros optimos en {(end - start) / 60:.1f} minutos")
    return model

def bayer_optimization_hiperparameters(model, X, y, k): # No probada
    # import optuna
    # from sklearn.model_selection import train_test_split

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.3)
        }

        # Inicializa el modelo con los hiperparámetros sugeridos por Optuna
        model_instance = model(**params)
        model_instance.fit(X_train, y_train)
        
        # Realiza predicciones en el conjunto de prueba
        y_pred = model_instance.predict(X_test)
        
        # Calcula la métrica de evaluación (en este caso, precisión)
        score = accuracy_score(y_test, y_pred)
        return score

    # Divide los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializa el estudio de Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Número de iteraciones de búsqueda

    # Imprime los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros:", study.best_params)

     # Obtiene los mejores hiperparámetros encontrados
    best_params = study.best_params

    # Inicializa el modelo con los mejores hiperparámetros
    best_model = model(**best_params)
    return best_model

def manual_cross_validation(model, X_train, y_train, k=5):  # Funciona igual que la libreria (podria utilizar la libreria si quiero o no) # antes recibia X e y --> lo saque para hacer la division en train y test en generate test design
    """
    Realiza cross validation para evaluar el rendimiento del modelo entrenado.

    :param model: Modelo de Machine Learning. (sklearn.ensemble)
    :param X_train: Dataframe de entrenamiento con variables predictoras. (DataFrame)
    :param y_train: Dataframe de entrenamiento solo con variable respuesta. (DataFrame)
    :param k: Numero de folds. (int)
    :return: Precision promedio de la validación cruzada. (float)
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