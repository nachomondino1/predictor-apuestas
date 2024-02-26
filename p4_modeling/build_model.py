import numpy as np
from sklearn.model_selection import GridSearchCV  # Seleccion de hiperparametros
from sklearn.metrics import accuracy_score  # Metrica de precision
import time


def select_best_hiperparameters(model, X, y, k, _print: bool = False):
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
            'n_estimators': [100, 500],  # Número de árboles en el bosque.
            'criterion': ['entropy'],  # Función para medir la calidad de una división.
            'max_depth': [3, 5, 7], # Podria reemplazar 10 y 15 por 8 # Profundidad máxima de los árboles.
            'min_samples_split': [2, 10],  # Número mínimo de muestras requeridas para realizar una división en un nodo interno.
            'min_samples_leaf': [1, 4], # Número mínimo de muestras requeridas para estar en un nodo hoja.
            'max_features': ['sqrt'],  # Número máximo de características a considerar al buscar la mejor división.
            'bootstrap': [True],  # Indica si se deben realizar muestras bootstrap al construir árboles.
        },
        'XGBClassifier': {  # ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical fill_type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:dt_loc: object
            'n_estimators': [100, 500, 1000],  # Número de árboles en el ensamblado.
            'learning_rate': [0.01, 0.1],  # Tasa de aprendizaje que controla la contribución de cada árbol.
            'max_depth': [3, 7, 10, 15, 20],  # Profundidad máxima de cada árbol.
            # 'min_child_weight': [1, 3, 5],  # Peso mínimo requerido en una hoja del árbol.
            'subsample': [0.8, 1.0],  # Proporción de muestras utilizadas para entrenar cada árbol.
            'colsample_bytree': [0.8, 1.0],  # Proporción de características utilizadas para entrenar cada árbol.
            # 'gamma': [0, 0.1, 0.5],  # Reducción mínima de la función de pérdida requerida para realizar una partición adicional en un nodo del árbol.
            # 'reg_alpha': [0, 0.01, 0.1],  # Término de regularización L1 en los pesos del árbol.
            # 'reg_lambda': [0, 0.01, 0.1],  # Término de regularización L2 en los pesos del árbol.
        },
        'GradientBoostingClassifier': {  # Puede performar mejor que Random pero con estos hiper tarda 39.9 minutos --> saco 1000 de n_estim y pongo 200 y saco max depth de 7
            'n_estimators': [100, 200, 500], # 1000 # Número de árboles en el ensamblado.
            'learning_rate': [0.01, 0.1],  # Tasa de aprendizaje que controla la contribución de cada árbol.
            'max_depth': [3, 5],  # 7  # Profundidad máxima de cada árbol.
            # 'min_samples_split': [1, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno.
            # 'min_samples_leaf': [2, 4],  # Número mínimo de muestras requeridas en cada hoja del árbol.
            'subsample': [0.8, 1.0],  # Proporción de muestras utilizadas para entrenar cada árbol.
            # 'max_features': ['auto'],  # Número máximo de características consideradas al buscar la mejor división.
            # 'loss': ['deviance']  # Función de pérdida a optimizar.
        },
        'LogisticRegression': {
            'penalty': ['l1', 'l2'],  # Tipo de regularización a aplicar.
            'C': [0.1, 1.0, 5.0],  # Podria probar un 3.0 en vez de 5  # Inverso de la fuerza de regularización.
            'solver': ['saga', 'liblinear', 'lbfgs'], # Algoritmo a utilizar en la optimización del problema.
            'fit_intercept': [True, False],  # Especifica si se debe ajustar o no el intercepto.  # Mas del 75% de las veces es True
            'max_iter': [100, 1000, 2000],  # Podria prescindir de 1000 # Número máximo de iteraciones para la convergencia del algoritmo.
            'multi_class': ['auto'],  # Esquema de clasificación multiclase.
        },
        'SVC': {
            'C': [0.1, 1.0, 5.0],  # Parámetro de regularización.
            'kernel': ['poly', 'rbf'], # 'linear', 'sigmoid' # Función kernel utilizada para transformar los datos de entrada.
            'gamma': ['scale', 'auto'],  # Coeficiente para el kernel RBF, 'poly' y 'sigmoid'.
            'degree': [3, 5],  # Grado del kernel polinomial.
            'coef0': [0.0,  0.5, 1.0],  # Término independiente en funciones kernel polinomiales y sigmoide.
            'shrinking': [True, False],  # Activa o desactiva el uso de la heurística de encogimiento.
            'probability': [True],  # Habilita o deshabilita la estimación de probabilidades. --> no tiene sentido probarlo aqui
            # 'tol': [1e-3, 1e-4, 1e-5],  # Siempre gana 1e-3 (y es el valor default) y ChatGPT no me lo dio como hiper tipico
            'decision_function_shape': ['ovo'],  # Siempre le gana ovo (One Vs One) a ovr (One Vs Rest)
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(10,), (50,), (100,)], # Número de neuronas en las capas ocultas.
            # 'activation': ['logistic',  'relu', 'tanh'],  # Función de activación utilizada en las capas ocultas.
            'solver': ['lbfgs', 'sgd'],  # Algoritmo utilizado para la optimización de pesos.
            'alpha': [0.0001, 0.001, 0.01],  # Parámetro de regularización para controlar la penalización de los pesos.
            'learning_rate': ['constant', 'adaptive'], # 'invscaling' # Tasa de aprendizaje utilizada en la actualización de los pesos.
            'learning_rate_init': [0.001, 0.01, 0.1],  # Tasa de aprendizaje inicial.
            # 'max_iter': [100, 200, 500],  # Número máximo de iteraciones.
            # 'early_stopping': [True, False]  # Opción para detener el entrenamiento tempranamente si no hay mejoras en la métrica de validación.
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
    # print(f"Hiperparametros a probar: {d_params[model_name]}")

    # Crear el objeto GridSearchCV
    grid_search = GridSearchCV(model, param_grid=d_params[model_name], cv=k)

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

def prueba():

    modelo = RandomForestClassifier()
    X_val = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p4_modeling/data/england/X_val.xlsx')
    y_val = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p4_modeling/data/england/X_val.xlsx')
 
    best_model = bayer_optimization_hiperparameters(RandomForestClassifier, X, y, 5)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()