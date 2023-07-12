import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV  # Seleccion de hiperparametros
from sklearn.metrics import accuracy_score  # Metrica de precision
import time


def select_best_hiperparameters(model, X, y, k):
    """
    Selecciona los mejores hiperparametros para un modelo.

    :param model: Modelo de Machine Learning. (sklearn.ensemble)
    :param X: Dataframe de validacion con variables predictoras. (DataFrame)
    :param y: Dataframe de validacion solo con variable respuesta. (DataFrame)
    :param k: Numero de folds. (int)
    :return: Modelo con mejores hiperparametros. (sklearn.ensemble?)
    """
    start = time.time()

    # Definicion de variables
    d_params = {
        'DecisionTreeClassifier': {
            'criterion': ['entropy'],  # 'gini'
            'splitter': ['random', 'best'],
            'max_depth': [None, 5, 7, 8, 10, 12, 14],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', None, 'sqrt', 'log2'],
        },
        'RandomForestClassifier': {
            'n_estimators': [100, 200, 500],  # Número de árboles en el bosque. Puedes probar diferentes valores como 100, 200, 500.
            'criterion': ['entropy'],  # 'gini'  # Función para medir la calidad de una división. Puedes probar 'gini' para el índice de Gini y 'entropy' para la ganancia de información.
            'max_depth': [5, 7, 8, 10, 15],  # Profundidad máxima de los árboles. Puedes probar None para que los árboles se expandan hasta que todas las hojas sean puras, o un número como 5 o 10 para limitar la profundidad máxima.
            'min_samples_split': [5, 10],  # Número mínimo de muestras requeridas para realizar una división en un nodo interno. Puedes probar diferentes valores como 2, 5, 10.
            'min_samples_leaf': [2, 4], # Evitaria valores bajos como 1 para evitar overfitting # Número mínimo de muestras requeridas para estar en un nodo hoja. Puedes probar diferentes valores como 1, 2, 4.
            'max_features': ['auto'],  # None no uso porque auto usa todas...# 'sqrt', 'log2'  # Número máximo de características a considerar al buscar la mejor división. Puedes probar 'auto' para considerar todas las características, o 'sqrt' para considerar la raíz cuadrada del número total de características.
            'bootstrap': [True, False],  # False  # Indica si se deben realizar muestras bootstrap al construir árboles. Puedes probar True o False.
            # 'class_weight': [None, 'balanced']  # Pesos para las clases. Puedes probar None para igual peso, o 'balanced' para ajustar automáticamente los pesos inversamente proporcionales a las frecuencias de clase en los datos de entrada.
        },
        'XGBClassifier': {
            'n_estimators': [100, 500, 1000],  # Número de árboles en el ensamblado. Valores típicos: 100, 500, 1000.
            'learning_rate': [0.01, 0.1, 0.3],  # Tasa de aprendizaje que controla la contribución de cada árbol. Valores típicos: 0.01, 0.1, 0.3.
            'max_depth': [7, 8, 10, 15, 20],  # Profundidad máxima de cada árbol. Valores típicos: 3, 5, 7.
            'min_child_weight': [1, 3, 5],  # Peso mínimo requerido en una hoja del árbol. Valores típicos: 1, 3, 5.
            'subsample': [0.8, 1.0],  # Proporción de muestras utilizadas para entrenar cada árbol. Valores típicos: 0.8, 1.0.
            'colsample_bytree': [0.8, 1.0],  # Proporción de características utilizadas para entrenar cada árbol. Valores típicos: 0.8, 1.0.
            'gamma': [0, 0.1, 0.5],  # Reducción mínima de la función de pérdida requerida para realizar una partición adicional en un nodo del árbol. Valores típicos: 0, 0.1, 0.5.
            'reg_alpha': [0, 0.01, 0.1],  # Término de regularización L1 en los pesos del árbol. Valores típicos: 0, 0.01, 0.1.
            'reg_lambda': [0, 0.01, 0.1],  # Término de regularización L2 en los pesos del árbol. Valores típicos: 0, 0.01, 0.1.
            # 'scale_pos_weight': [1, 5, 10]  # WARNING: /Users/runner/work/xgboost/xgboost/python-package/build/temp.macosx-10.9-x86_64-cpython-38/xgboost/src/learner.cc:767: # Parameters: { "scale_pos_weight" } are not used.# Relación de pesos entre la clase positiva y la negativa en el conjunto de datos. Valores típicos: 1, 5, 10.
        },
        'GradientBoostingClassifier': {
            'n_estimators': [100, 500, 1000],  # Número de árboles en el ensamblado. Valores típicos: 100, 500, 1000.
            'learning_rate': [0.01, 0.05, 0.1],  # Tasa de aprendizaje que controla la contribución de cada árbol. Valores típicos: 0.01, 0.1, 0.3.
            'max_depth': [3, 5, 7],  # Profundidad máxima de cada árbol. Valores típicos: 3, 5, 7.
            'min_samples_split': [5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno. Valores típicos: 2, 5, 10.
            'min_samples_leaf': [2, 4],  # Número mínimo de muestras requeridas en cada hoja del árbol. Valores típicos: 1, 2, 4.
            'subsample': [0.8, 1.0],  # Proporción de muestras utilizadas para entrenar cada árbol. Valores típicos: 0.8, 1.0.
            'max_features': ['auto'], # None no uso porque auto usa todas... # 'sqrt' no usaria pues no tengo tantas variables # Número máximo de características consideradas al buscar la mejor división. Valores típicos: 'auto', 'sqrt' (puede ser un entero o una fracción).
            'loss': ['deviance']  # 'exponential' no usaria por no usar AdaBoost?... # Función de pérdida a optimizar. Valores típicos: 'deviance' (para clasificación con probabilidades) o 'exponential' (para clasificación con AdaBoost).
        },
        'LogisticRegression': {
            'penalty': [None, 'l1', 'l2'], # 'elasticnet'  # Tipo de regularización a aplicar. Puedes probar 'l1' para regularización L1 (valor absoluto de los coeficientes), 'l2' para regularización L2 (norma euclidiana al cuadrado de los coeficientes).
            'C': [0.1, 1.0, 10.0],  # Inverso de la fuerza de regularización. Valores típicos: 0.1, 1, 10. Un valor más bajo indica una regularización más fuerte.
            'solver': ['lbfgs', 'liblinear', 'newton-cg'],  # , 'sag', 'saga'  # Algoritmo a utilizar en la optimización del problema. Puedes probar 'liblinear' para problemas pequeños, 'saga' para problemas grandes.
            'fit_intercept': [True, False],  # Especifica si se debe ajustar o no el intercepto. Puedes probar True o False.
            'max_iter': [100, 500, 1000],  # Número máximo de iteraciones para la convergencia del algoritmo. Valores típicos: 100, 500, 1000.
            # 'class_weight': [None, 'balanced'],  # No uso puesto que balanceo o no en modeling y quiero respetar eso. # Pesos para las clases. Puedes probar None para igual peso, 'balanced' para ajustar automáticamente los pesos inversamente proporcionales a las frecuencias de clase en los datos de entrada.
            'multi_class': ['ovr', 'multinomial', 'auto'],  # Esquema de clasificación multiclase. Puedes probar 'auto' para seleccionar automáticamente el enfoque más adecuado según los datos, 'ovr' para clasificación uno contra el resto, 'multinomial' para una clasificación multinomial.
        },
        'SVC': {
            'C': [0.1, 0.5, 1.0],  # # Parámetro de regularización. Valores típicos: 0.1, 1, 10. Un valor más bajo suaviza la frontera de decisión, permitiendo clasificaciones más flexibles, mientras que un valor más alto la hace más estricta.
            # 'kernel': ['linear'],  # 'poly', 'rbf', 'sigmoid'  # Tarda mucho no se por que... inclusive usando solo linear...  # Función kernel utilizada para transformar los datos de entrada. Puedes probar 'linear' para un kernel lineal, 'poly' para un kernel polinomial, 'rbf' para un kernel gaussiano (RBF) o 'sigmoid' para un kernel sigmoide.
            'gamma': ['scale', 'auto'],  # Coeficiente para el kernel RBF, 'poly' y 'sigmoid'. Puedes probar 'scale' para usar 1 / (n_features * X.var()) como valor de gamma, 'auto' para 1 / n_features o valores numéricos, como 0.1, 1, etc.
            'degree': [1, 2],  # Grado del kernel polinomial. Puedes probar 2, 3, 4, etc.
            'coef0': [0.0, 0.1, 1.0],  # Término independiente en funciones kernel polinomiales y sigmoide. Valores típicos: 0.0, 0.5, 1.0.
            'shrinking': [True, False],  # Activa o desactiva el uso de la heurística de encogimiento. Puedes probar True o False.
            'probability': [True, False], # Habilita o deshabilita la estimación de probabilidades. Puedes probar True o False.
            # 'tol': [1e-3, 1e-4, 1e-5],  # Siempre gana 1e-3 y ChatGPT no me lo dio como hiper tipico
            # 'decision_function_shape': ['ovo', 'ovr'],  # Siempre gana ovo y ChatGPT no me lo dio como hiper tipico
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(10,), (50,), (100,)],  # [(64), (128), (64, 32)],  # Número de neuronas en las capas ocultas. Puedes probar diferentes combinaciones, como (10,) para una capa oculta de 10 neuronas, (50,) para una capa oculta de 50 neuronas, etc.
            'activation': ['logistic',  'relu', 'tanh'],  # 'identity', 'sigmoid'  # Función de activación utilizada en las capas ocultas. Puedes probar 'logistic' para la función logística o 'relu' para la unidad lineal rectificada.
            'solver': ['lbfgs', 'sgd'],  # 'adam'  # Algoritmo utilizado para la optimización de pesos. Puedes probar 'adam' para el algoritmo de descenso de gradiente estocástico o 'sgd' para el descenso de gradiente estocástico clásico.
            'alpha': [0.0001, 0.001, 0.01],  # Parámetro de regularización para controlar la penalización de los pesos. Valores típicos: 0.0001, 0.001, 0.01.
            'learning_rate': ['constant', 'adaptive', 'invscaling'],  # 'learning_rate',  # Tasa de aprendizaje utilizada en la actualización de los pesos. Puedes probar 'constant' para una tasa de aprendizaje constante o 'adaptive' para una tasa de aprendizaje adaptativa.
            'learning_rate_init': [0.001, 0.01, 0.1],  # Tasa de aprendizaje inicial. Valores típicos: 0.001, 0.01, 0.1.
            'max_iter': [100, 200, 500],  # Número máximo de iteraciones. Valores típicos: 100, 200, 500.
            'early_stopping': [True, False]  # Opción para detener el entrenamiento tempranamente si no hay mejoras en la métrica de validación. Puedes probar True o False.
        },
        'PCA': {
            'n_components': [2, 5, 10],
            'whiten': [True, False],
            'svd_solver': ['auto', 'full', 'randomized'],
            'iterated_power': [1, 2, 3]
        }
    }

    # Obtengo el nombre del modelo para poder buscar sus hiperparametros
    model_name = str(model)[:str(model).find('(')]
    # print(f"Hiperparametros a probar: {d_params[model_name]}")

    # Crear el objeto GridSearchCV
    grid_search = GridSearchCV(model, param_grid=d_params[model_name], cv=k)

    # Ajustar el objeto GridSearchCV a los datos de entrenamiento
    grid_search.fit(X, y)

    # Obtener los mejores hiperparámetros y el modelo con dichos hiperparametros
    best_params = grid_search.best_params_
    model = grid_search.best_estimator_
    # print(f"Mejores hiperparametros: {model}")  # print(f"Mejores hiperparametros: {best_params}")

    end = time.time()
    print(f"Seleccion de hiperparametros optimos en {(end - start) / 60:.1f} minutos")
    return model, best_params

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

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import xgboost as xgb  # XGBoost
    from sklearn.linear_model import LogisticRegression  # Regresion Logistica
    import lightgbm as lgb  # Gradient Boosting
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
    from sklearn.svm import SVC  # SVM
    from sklearn.neural_network import MLPClassifier

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_selected_manual.xlsx')

    warnings.filterwarnings("ignore")
    best_acurracy = 0
    l_modelos = [DecisionTreeClassifier(max_depth=30),
                 RandomForestClassifier(n_estimators=200, max_depth = None, random_state=42),
                 xgb.XGBClassifier(n_estimators=50, objective='multi:softmax', num_class=len(y.unique()), max_depth=20),
                 LogisticRegression(multi_class='multinomial', penalty='l2', C=0.1, solver='lbfgs', max_iter=500),
                 SVC(kernel='rbf', decision_function_shape='ovo'),
                 MLPClassifier(hidden_layer_sizes=128, activation='tanh', solver='adam', learning_rate='invscaling',
                               max_iter=300),
                 GradientBoostingClassifier(learning_rate= 0.1, n_estimators=200, max_depth=7)
                 # self.red_neuronal(n_folds_cv=10, n_epochs=1000, batches=256),
                ]

    # Por modelo a probar
    for modelo in l_modelos:

        # Entreno modelo
        print(f" Modelo: {str(modelo)[:str(modelo).find('(')]} ".center(120, '#'))
        model, cv_accuracy, test_accuracy = train_model(df, 'equipo_ganador', modelo, best_params=True, k=5) # model = DecisionTreeClassifier()  # model2 = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], random_state=42)

        # Si es el mejor modelo hasta aqui
        if test_accuracy > best_acurracy:
            # Guardo modelo
            best_acurracy = test_accuracy
            best_model = model

    print(f"\nEl mejor modelo es: {best_model}")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()