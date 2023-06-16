import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV  # Seleccion de hiperparametros
from sklearn.metrics import accuracy_score  # Metrica de precision


def select_best_hiperparameters(model, X, y, k):

    # Definicion de variables
    d_params = {"DecisionTreeClassifier":
                    {'max_depth': [None, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]},
                "RandomForestClassifier": {'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
                                           'n_estimators': [50, 100, 150, 200]},
                'XGBClassifier': {'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
                                  'n_estimators': [50, 100, 150, 200]},
                'LogisticRegression': {'penalty': [None, 'l2'], 'C': [0.1, 1.0, 10.0],
                                       'solver': ['lbfgs', 'newton-cg', 'sag', 'saga', 'lbfgs'],
                                       'max_iter': [100, 500, 1000],
                                       'multi_class': ['multinomial']},
                'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'decision_function_shape': ['ovo', 'ovr']},
                'MLPClassifier': {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                  'solver': ['lbfgs', 'sgd', 'adam'],
                                  'learning_rate': ['learning_rate', 'invscaling', 'adaptive'], 'max_iter': [200, 300],
                                  'hidden_layer_sizes': [(64), (128), (64, 32)]},
                'GradientBoostingClassifier': {'learning_rate': [0.1, 0.05, 0.01], 'n_estimators': [100, 200, 300],
                                               'max_depth': [None, 5, 7, 20, 30]},
                'PCA': {'n_components': [2, 5, 10], 'whiten': [True, False],
                        'svd_solver': ['auto', 'full', 'randomized'],
                        'iterated_power': [1, 2, 3], 'random_state': [42]}
                }

    # Obtengo el nombre del modelo para poder buscar sus hiperparametros
    model_name = str(model)[:str(model).find('(')]

    # Crear el objeto GridSearchCV
    grid_search = GridSearchCV(model, param_grid=d_params[model_name], cv=k)

    # Ajustar el objeto GridSearchCV a los datos de entrenamiento
    grid_search.fit(X, y)

    # Obtener los mejores hiperparámetros y el modelo con dichos hiperparametros
    best_params = grid_search.best_params_
    model = grid_search.best_estimator_
    return model, best_params

def manual_cross_validation(model, X_train, y_train, k=5):  # Funciona igual que la libreria (podria utilizar la libreria si quiero o no) # antes recibia X e y --> lo saque para hacer la division en train y test en generate test design
    # Cross validation para evaluar la precision del modelo

    # Realizar validación cruzada manual
    scores, rois = [], []
    fold_size = len(X_train) // k

    for i in range(k):

        # Dividir los datos en conjuntos de entrenamiento y test
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
        print(f'Fold {i+1} --> Precision: {accuracy:.1f}%')

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