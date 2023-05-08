###### LIBRERIAS #######
# Generales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Arbol de decision
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn import metrics
# Random Forest
from sklearn.ensemble import RandomForestClassifier


###### HIPERPARAMETROS #######
num_folds = 10 # Cantidad de divisiones de validacion cruzada
max_depth_tree = 6 # Profundidad del arbol
number_tress_in_forest = 100 # Cantidad de arboles en el bosque de Random Forest


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
    return df

def train_and_test(num_folds, modelo, df, max_depth_tree, plot_tree = False, plot_conf_matrix = False):
    l_aciertos = []
    # Dividir los datos en k folds
    folds = np.array_split(df, num_folds)
    # Iterar sobre cada fold y entrenar el modelo
    for i in range(num_folds):
        # Separar los datos de entrenamiento y prueba para el fold actual
        test_data = folds[i]
        train_data = pd.concat([f for j, f in enumerate(folds) if j != i])
        X_train = train_data.drop("equipo_ganador", axis=1)
        y_train = train_data["equipo_ganador"]
        X_test = test_data.drop("equipo_ganador", axis=1)
        y_test = test_data["equipo_ganador"]
        
        # Entrenar el modelo en los datos de entrenamiento del fold actual
        if modelo == 'arbol':
            modelo = DecisionTreeClassifier(max_depth=max_depth_tree)
            modelo.fit(X_train, y_train)
            if plot_tree == 'True': # Graficar el árbol
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_tree(modelo, feature_names=X.columns, class_names=y.unique(), filled=True, ax=ax)
        elif modelo == 'random_forest':
            modelo = RandomForestClassifier(n_estimators=number_tress_in_forest, random_state=42, max_depth = max_depth_tree) # max_depth=30
            modelo.fit(X_train, y_train)

        # Predicciones
        y_pred = modelo.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        l_aciertos.append(precision)
        matriz_confusion = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))
        if plot_conf_matrix == 'True':
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matriz_confusion,
                                                        display_labels=["Local", "Empate", "Visitante"])
            cm_display.plot(cmap='Blues')

        print("precision: ", precision)
        print(matriz_confusion)
        print("Fold %d - Score: %.3f" % (i+1, modelo.score(X_test, y_test)))
        plt.show()
    print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos)/len(l_aciertos)}")
    return max(l_aciertos), min(l_aciertos), sum(l_aciertos)/len(l_aciertos)

###### DATASET #######
# Levanto dataset
df = load_dataset_and_clean(path='data_preparation/df_prepared.xlsx')

###### MODELOS #######
train_and_test(10, 'arbol', df, max_depth_tree= 5, plot_tree = False, plot_conf_matrix = False)