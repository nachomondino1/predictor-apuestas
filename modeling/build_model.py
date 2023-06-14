import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV  # Seleccion de hiperparametros
from sklearn.metrics import accuracy_score  # Metrica de precision
from modeling.asses_model import calculate_ROI  # Metrica de roi
# from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report


def train_model(df_train, var_resp, model, df_etiquetas, best_params=False, k=5):  # antes recibia X e y --> lo saque para hacer la division en train y test en generate test design

    # Elimino variables de cuotas puesto que no las usare para entrenar sino que solo para calcular el roi   # Iba en generate test design pero lo traje para ver si puedo calcular el roi
    df_train_without_odds = df_train.copy().drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, y_train = df_train_without_odds.drop(var_resp, axis=1), df_train_without_odds[var_resp]

    # Verificar si se deben buscar los mejores hiperparámetros
    if best_params:
        model = select_best_hiperparameters(X_train, y_train, model, k)

    # Realizar validación cruzada manual
    scores, rois = [], []
    fold_size = len(X_train) // k

    for i in range(k):

        # Dividir los datos en conjuntos de entrenamiento y validación
        start, end = i * fold_size, (i + 1) * fold_size
        X_train_fold = np.concatenate((X_train[:start], X_train[end:]), axis=0)
        y_train_fold = np.concatenate((y_train[:start], y_train[end:]), axis=0)
        X_test_fold = X_train[start:end]
        y_test_fold = y_train[start:end]

        # Entrenar el modelo con el conjunto de entrenamiento de la iteración actual
        model.fit(X_train_fold, y_train_fold)

        # Realizar predicciones en el conjunto de validación
        y_pred = model.predict(X_test_fold)

        # Guardo predicciones  # Podria llegar a haber estado mal antes, veremos si la precision baja...
        df_res = df_train[start:end]  # Necesito y_real y confirmo que es igual a pd.concat([X_test_fold, y_test_fold], axis=1) pero con las odds
        df_res['y_pred'] = y_pred

        # Calcular la precisión y el roi en el conjunto de validación
        accuracy = accuracy_score(y_test_fold, y_pred) * 100
        roi = calculate_ROI(df_res, var_resp, 'y_pred', df_etiquetas) * 100
        scores.append(accuracy)
        rois.append(roi)
        # print(f'Fold {i} --> Precision: {accuracy:.1f}%  ROI: {roi:.1f}%')

    # Calcular la precisión promedio de la validación cruzada
    cv_accuracy, cv_roi = np.mean(scores), np.mean(rois)
    # print(f"Resultados promedios de validación cruzada: \n  - Precision prom: {cv_accuracy:.1f}% \n  - ROI prom: {cv_roi:.1f}%")

    # Entrenar el modelo final con todos los datos de entrenamiento
    model.fit(X_train, y_train)
    return model, cv_accuracy, cv_roi

def select_best_hiperparameters(X_train, y_train, model, k):

    # Definicion de variables
    model_name = str(model)[:str(model).find('(')]
    d = {"DecisionTreeClassifier":
             {'max_depth': [None, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]},
        "RandomForestClassifier": {'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
                                   'n_estimators': [50, 100, 150, 200]},
        'XGBClassifier': {'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35], 'n_estimators': [50, 100, 150, 200]},
        'LogisticRegression': {'penalty': [None, 'l2'], 'C': [0.1, 1.0, 10.0],
                               'solver': ['lbfgs', 'newton-cg', 'sag', 'saga', 'lbfgs'], 'max_iter': [100, 500, 1000],
                               'multi_class': ['multinomial']},
        'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'decision_function_shape': ['ovo', 'ovr']},
        'MLPClassifier': {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['learning_rate', 'invscaling', 'adaptive'], 'max_iter': [200, 300],
                'hidden_layer_sizes': [(64), (128), (64, 32)]},
         'GradientBoostingClassifier': {'learning_rate': [0.1, 0.05, 0.01], 'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 7, 20, 30]}
         }

    # Crear el objeto GridSearchCV
    grid_search = GridSearchCV(model, d[model_name], cv=k)

    # Ajustar el objeto GridSearchCV a los datos de entrenamiento
    grid_search.fit(X_train, y_train)

    # Obtener los mejores hiperparámetros
    best_params = grid_search.best_params_
    print("Mejores hiperparámetros encontrados:", best_params)

    # Obtener el modelo con los mejores hiperparámetros
    model = grid_search.best_estimator_
    return model

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
        model, cv_accuracy, test_accuracy = build_model(X, y, modelo, best_params=True, k=5) # model = DecisionTreeClassifier()  # model2 = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], random_state=42)

        # Si es el mejor modelo hasta aqui
        if test_accuracy > best_acurracy:
            # Guardo modelo
            best_acurracy = test_accuracy
            best_model = model

    print(f"\nEl mejor modelo es: {best_model}")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()