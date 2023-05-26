from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, cross_val_predict, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
from sklearn.linear_model import LogisticRegression  # Regresion Logistica

# from typing import Optional
# import time
# from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report
# import matplotlib.pyplot as plt
# from itertools import cycle
# import plotly.graph_objects as go
# # SVM
# from sklearn.svm import SVC
# # Redes Nueronales
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
# from tensorflow import keras
# # from tensorflow.keras import layers
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.utils import np_utils, to_categorical
# from keras.callbacks import EarlyStopping
# # Gradient Boosting
# import lightgbm as lgb


def generate_test(df, var_resp):

    # Shuffle dataset
    df_mezclado = pd.DataFrame(shuffle(df))
    # df = df.sample(frac=1).reset_index(drop=True)

    # Convertir variables categoricas string a categoricas numericas
    le = LabelEncoder()
    oversampler = RandomOverSampler()
    for col in df_mezclado.select_dtypes(include=['object']).columns:
        df_mezclado[col] = le.fit_transform(df_mezclado[col])

    # Balanceamos segun variable respuesta
    X, y = df_mezclado.drop(var_resp, axis=1), df_mezclado[var_resp]
    X_bal, y_bal = oversampler.fit_resample(X, y)
    # df = clean_data.balance_dataset(df, var_resp='equipo_ganador')

    # Separo conjunto de datos en train y test
    # df_train, df_test = test_design.separate_train_and_test(df, porc_corte=0.8)
    # df_train = df_train.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
    # print(df_train.shape, df_test.shape)
    return X_bal, y_bal

def select_best_hiperparameters(X_train, y_train, model, k):

    # Definicion de variables
    model_name = str(model)[:str(model).find('(')]
    d = {"DecisionTreeClassifier": {
        'max_depth': [None, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]},
        "RandomForestClassifier": {'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
                                     'n_estimators': [50, 100, 150, 200]},
        'XGBClassifier': {'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
                            'n_estimators': [50, 100, 150, 200]},
        'LogisticRegression': {'penalty': [None, 'l2'], 'C': [0.1, 1.0, 10.0],
                               'solver': ['lbfgs', 'newton-cg', 'sag', 'saga', 'lbfgs'], 'max_iter': [100, 500, 1000],
                               'multi_class': ['multinomial']}
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

def build_model(X, y, model, best_params=False, k=5):

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verificar si se deben buscar los mejores hiperparámetros
    if best_params:
        model = select_best_hiperparameters(X_train, y_train, model, k)

    # Realizar validación cruzada manual
    scores = []
    fold_size = len(X_train) // k

    for i in range(k):
        # Dividir los datos en conjuntos de entrenamiento y validación
        start = i * fold_size
        end = (i + 1) * fold_size
        X_train_fold = np.concatenate((X_train[:start], X_train[end:]), axis=0)
        y_train_fold = np.concatenate((y_train[:start], y_train[end:]), axis=0)
        X_val_fold = X_train[start:end]
        y_val_fold = y_train[start:end]

        # Entrenar el modelo con el conjunto de entrenamiento de la iteración actual
        model.fit(X_train_fold, y_train_fold)

        # Realizar predicciones en el conjunto de validación
        y_pred = model.predict(X_val_fold)

        # Calcular la precisión en el conjunto de validación y agregarla a la lista de scores
        accuracy = accuracy_score(y_val_fold, y_pred)
        scores.append(accuracy)

    # Calcular la precisión promedio de la validación cruzada
    cv_accuracy = np.mean(scores)
    print(f"Precisión de la validación cruzada: {cv_accuracy:.3f}")

    # Entrenar el modelo final con todos los datos de entrenamiento
    model.fit(X_train, y_train)

    # Predecir las etiquetas para los datos de prueba
    y_pred = model.predict(X_test)

    # Calcular la precisión del modelo en los datos de prueba
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo en los datos de prueba: {test_accuracy:.3f}")

    # Devolver el modelo entrenado, precisión de la validación cruzada y precisión en los datos de prueba
    return model, cv_accuracy, test_accuracy


def main():

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/df_selected_manual.xlsx')


    # Procesamiento que le falta al df --> no iria aca...
    # Select data
    df = df.drop(['historial_entre_si', 'odds_loc', 'odds_emp', 'odds_vis'], axis = 1)
    print(df.head())
    print(df.shape)
    # Borro NaNs
    df = df.dropna()
    # Genero test design
    X, y = generate_test(df, 'equipo_ganador')


    warnings.filterwarnings("ignore")
    best_acurracy = 0
    l_modelos = [DecisionTreeClassifier(max_depth=30),
                 # RandomForestClassifier(n_estimators=200, max_depth = None, random_state=42),
                 # xgb.XGBClassifier(n_estimators=50, objective='multi:softmax', num_class=len(y.unique()), max_depth=20),
                 # LogisticRegression(multi_class='multinomial', penalty='l2', C=0.1, solver='lbfgs', max_iter=500),

                 # self.svm(n_folds_cv=cv, kernel_type='rbf', ovo_o_ovr='ovo'),
                 # self.red_neuronal(n_folds_cv=10, n_epochs=1000, batches=256),
                 # self.perceptron_multiple(n_folds_cv=cv, activ='tanh', hidden_layer=128, solv='adam', lear_rate='invscaling', max_itera=300),
                 # self.gradient_boosting(n_folds_cv=10, lear_rate=0.1, n_trees=200, max_depth=7)]
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


main()