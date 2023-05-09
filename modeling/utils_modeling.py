# Generales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# Arbol de decision
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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
# XGBoost
import xgboost as xgb


def load_dataset_and_clean(path='data_preparation/df_prepared.xlsx'):
    """
    Levantamos el dataset, eliminamos los nan y aplicamos un encoder a las variables categoricas
    """
    # Abrimos dataset
    df = pd.read_excel(path)
    df.drop(['historial_entre_si'], axis = 1, inplace=True) # Elimino esta variable porque tiene muchos nans
    df = df.dropna()  # Elimina filas con al menos un valor nulo
    # Codificamos las variables categoricas de string a numericas
    labelencoder = LabelEncoder() 
    cat_columns = ['equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis']
    for column in cat_columns:
        df[column] = labelencoder.fit_transform(df[column])
    # Hacemos Shuffle
    df = shuffle(df) 
    return df

def arbol_decision(X_train, y_train, max_depth_tree, plot_tree_bool='False'):
    """
    Entrenar un arbol de decisión
    """
    modelo = DecisionTreeClassifier(max_depth=max_depth_tree)
    modelo.fit(X_train, y_train)
    if plot_tree_bool == True: # Graficar el árbol
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_tree(modelo, feature_names=X_train.columns, class_names=y_train.unique(), filled=True, ax=ax)
    return modelo


def random_forest(X_train, y_train, number_tress_in_forest, max_depth_tree):
    """
    Entrenar random forest
    """
    modelo = RandomForestClassifier(n_estimators=number_tress_in_forest, random_state=42, max_depth = max_depth_tree) # max_depth=30
    modelo.fit(X_train, y_train)
    return modelo

def xgboost(X_train, y_train):
    """
    Entrenar XGBoost
    """
    # Aplicamos encoder a y_train. Pasamos de tener strings [empate, local, visitante] a por ej: [0,1,2]
    le = LabelEncoder() 
    y_train = le.fit_transform(y_train)

    modelo =  xgb.XGBClassifier(objective='multi:softmax') # multi:softproba
    modelo.fit(X_train, y_train)
    return modelo

def predicciones_metricas(modelo, X_test, y_test, i_cross_val, plot_conf_matrix='False'):
    
    y_pred = modelo.predict(X_test)
    classes_names = np.unique(y_pred)

    # Calculo métricas
    dict_metricas = {"accuracy":[], "precision":[] , "recall":[] , "f1":[]}

    le = LabelEncoder()
    y_test = le.fit_transform(y_test)

    dict_metricas["accuracy"].append(accuracy_score(y_test, y_pred))
    dict_metricas["precision"].append(precision_score(y_test, y_pred, average='weighted'))
    dict_metricas["recall"].append(recall_score(y_test, y_pred, average='weighted'))
    dict_metricas["f1"].append(f1_score(y_test, y_pred, average='weighted'))
    matriz_confusion = confusion_matrix(y_test, y_pred, labels=classes_names)

    # Graficamos la matriz de confusión
    if plot_conf_matrix == True: 
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matriz_confusion,
                                                    display_labels=classes_names)
        cm_display.plot(cmap='Blues')
        plt.show()
    print(matriz_confusion)
    print("Fold %d - Score: %.3f" % (i_cross_val+1, modelo.score(X_test, y_test)))

    return dict_metricas


def curva_roc():
    # y_pred_proba = modelo.predict_proba(X_test) # Devuelve 3 columnas, cada una posee la prob de una clase
            ### CURVA ROC ###  
    """
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
    """
    pass

def train_and_test(num_folds, model_to_train, df, max_depth_tree,number_tress_in_forest, plot_tree = False, plot_conf_matrix = False):
    prom_metricas = {"accuracy":None, "precision":None , "recall":None , "f1":None}
    folds = np.array_split(df, num_folds) # Dividir los datos en k folds

    ### VALIDACION CRUZADA ### Iterar sobre cada fold y entrenar el modelo
    for i in range(num_folds):

        # Separar los datos de entrenamiento y prueba para el fold actual
        test_data = folds[i]
        train_data = pd.concat([f for j, f in enumerate(folds) if j != i])
        X_train, y_train  = train_data.drop("equipo_ganador", axis=1), train_data["equipo_ganador"]
        X_test, y_test = test_data.drop("equipo_ganador", axis=1), test_data["equipo_ganador"]
        print(y_test.value_counts())

        # Balancear datos
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        
        ### ENTRENAMIENTO ### Entrenamos modelo con fold actual
        if model_to_train == 'arbol':
           modelo = arbol_decision(X_train, y_train, plot_tree_bool='False')

        elif model_to_train == 'random_forest':
            modelo = random_forest(X_train, y_train, number_tress_in_forest, max_depth_tree)

        elif model_to_train == 'xgboost': # Deberiamos hacer este mapeo con el resto de las variables categoricas
            modelo = xgboost(X_train, y_train)

        dict_metricas = predicciones_metricas(modelo, X_test, y_test, i, plot_conf_matrix='False')

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
metricas = train_and_test(num_folds, 'xgboost', df, max_depth_tree, number_tress_in_forest, plot_tree = False, plot_conf_matrix = True)
print(metricas) #  arbol xgboost random_forest

# hiper_optimos(df)