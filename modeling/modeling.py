###### LIBRERIAS #######
# Generales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Arbol de decision
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
# Random Forest
from sklearn.ensemble import RandomForestClassifier
import statistics as stat
from imblearn.over_sampling import RandomOverSampler


###### FUNCIONES #######

def load_dataset_and_clean(path='data_preparation/df_prepared.xlsx'):
    df = pd.read_excel(path)
    df.drop(['historial_entre_si'], axis = 1, inplace=True) # Elimino esta variable porque tiene muchos nans
    df = df.dropna()  # Elimina filas con al menos un valor nulo
    print(df.head())
    # Codificamos las variables categoricas string en numericas
    labelencoder = LabelEncoder()
    cat_columns = ['equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis']
    for column in cat_columns:
        df[column] = labelencoder.fit_transform(df[column])
    df = df.sample(frac=1, random_state=42) # Hacemos Shuffle
    return df

def train_and_test(num_folds, modelo, df, max_depth_tree,number_tress_in_forest, plot_tree = False, plot_conf_matrix = False):
    dict_metricas = {"accuracy":[], "precision":[] , "recall":[] , "f1":[]}
    prom_metricas = {"accuracy":None, "precision":None , "recall":None , "f1":None}

    # Dividir los datos en k folds
    folds = np.array_split(df, num_folds)
    # Validacion Cruzada: Iterar sobre cada fold y entrenar el modelo
    for i in range(num_folds):
        # Separar los datos de entrenamiento y prueba para el fold actual
        test_data = folds[i]
        train_data = pd.concat([f for j, f in enumerate(folds) if j != i])
        X_train, y_train  = train_data.drop("equipo_ganador", axis=1), train_data["equipo_ganador"]
        X_test, y_test = test_data.drop("equipo_ganador", axis=1), test_data["equipo_ganador"]
        print(y_test.value_counts())

        # Probamos a balancear datos
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        # Entrenar el modelo en los datos de entrenamiento del fold actual
        if modelo == 'arbol':
            modelo = DecisionTreeClassifier(max_depth=max_depth_tree)
            modelo.fit(X_train, y_train)
            if plot_tree == True: # Graficar el árbol
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_tree(modelo, feature_names=X.columns, class_names=y.unique(), filled=True, ax=ax)
        elif modelo == 'random_forest':
            modelo = RandomForestClassifier(n_estimators=number_tress_in_forest, random_state=42, max_depth = max_depth_tree) # max_depth=30
            modelo.fit(X_train, y_train)

        # Predicciones
        y_pred = modelo.predict(X_test)
        classes_names = np.unique(y_pred)
        y_pred_proba = modelo.predict_proba(X_test) # Devuelve 3 columnas, cada una posee la prob de una clase

        # Métricas
        dict_metricas["accuracy"].append(accuracy_score(y_test, y_pred))
        dict_metricas["precision"].append(precision_score(y_test, y_pred, average='weighted'))
        dict_metricas["recall"].append(recall_score(y_test, y_pred, average='weighted'))
        dict_metricas["f1"].append(f1_score(y_test, y_pred, average='weighted'))
        
        matriz_confusion = confusion_matrix(y_test, y_pred, labels=classes_names)
        if plot_conf_matrix == True:
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matriz_confusion,
                                                        display_labels=classes_names)
            cm_display.plot(cmap='Blues')
            plt.show()
        print(matriz_confusion)
        print("Fold %d - Score: %.3f" % (i+1, modelo.score(X_test, y_test)))

        ## AUC y Curva ROC para cada clase 

        # Binarizar las etiquetas de las clases
        y_test_bin = label_binarize(y_test, classes=classes_names)
        n_classes = y_test_bin.shape[1]
        # Calcular la curva ROC y el AUC para cada clase
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Graficar la curva ROC para cada clase
        plt.figure()
        classes=np.unique(y_pred)
        lw = 2
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, 
                    label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {classes[i]}')
        # Graficar la línea de referencia aleatoria
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")

        # plt.show()

    # print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos)/len(l_aciertos)}")
    prom_metricas["accuracy"] = stat.mean(dict_metricas["accuracy"])
    prom_metricas["precision"] = stat.mean(dict_metricas["precision"])
    prom_metricas["recall"] = stat.mean(dict_metricas["recall"])
    prom_metricas["f1"] = stat.mean(dict_metricas["f1"])
    return prom_metricas

def hiper_optimos(df, modelo='arbol'):
    X, y  = df.drop("equipo_ganador", axis=1), df["equipo_ganador"]
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Definir modelo
    if modelo == 'arbol':
        model = DecisionTreeClassifier()
        # Definir parámetros a probar
        params = {'max_depth': [3, 4, 5, 6, 7, 8]}
        
    elif modelo == 'random_forest':
        model = RandomForestClassifier() # max_depth=30
        # Definir parámetros a probar
        params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
            'n_estimators': [50, 100,150, 200]}
            
    # Definir esquema de validación cruzada
    cv = 5
    # Realizar búsqueda de cuadrícula
    grid_search = GridSearchCV(model, params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Imprimir mejores parámetros y score
    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # Calcular precisión en conjunto de prueba
    test_score = grid_search.score(X_test, y_test)
    print("Test set score: {:.2f}".format(test_score))

    # results = grid_search.cv_results_
    # return results 


###### DATASET #######
# Levanto dataset
df = load_dataset_and_clean(path='data_preparation/df_prepared.xlsx')

###### HIPERPARAMETROS #######
num_folds = 10 # Cantidad de divisiones de validacion cruzada
max_depth_tree = 7 # Profundidad del arbol
number_tress_in_forest = 100 # Cantidad de arboles en el bosque de Random Forest

# Eje x: K de validacion cruzada & Eje y: Precision

# Eje x: Max profundidad & Eje y: Precision

# Eje x: Cantidad de arboles & Eje y: Precision


###### MODELOS #######
metricas = train_and_test(num_folds, 'random_forest', df, max_depth_tree,number_tress_in_forest, plot_tree = False, plot_conf_matrix = True)
print(metricas) #  arbol

# hiper_optimos(df)

