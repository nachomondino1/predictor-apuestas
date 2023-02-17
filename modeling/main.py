# Importo librerias
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy

def balance_dataset(df):

    # Definicion de variables
    df_aux = pd.DataFrame(columns=df.columns)
    n_ejs_clase_min = 1000000000
    l_clases = list(df['equipo_ganador'].unique())

    # Random shuffle dataframe (por evitar mal balanceo en caso de que el df este ordenado por algun campo)
    df = df.sample(frac=1).reset_index(drop=True)

    # Paso 1: Identifico cuantos ejemplos deberia tener cada clase para que el dataset este balanceado
    # Por clase
    for clase in l_clases:

        # Obtengo cantidad de registros con dicha clase
        n_ejs_clase = len(df[df['equipo_ganador'] == clase])
        print("Clase: {}. Nºejemplos: {}".format(clase, n_ejs_clase))

        # Si tiene menos ejemplos que las otras clases
        if n_ejs_clase < n_ejs_clase_min:

            # Guardo Nº ejemplos min
            n_ejs_clase_min = n_ejs_clase

    print("n_ejs_clase_min", n_ejs_clase_min)

    # Paso 2: Efectuo el balanceo segun la cantidad que debe tener cada clase (n_ejs_clase_min)
    for clase in l_clases:

        print("Clase: ", clase)
        df_clase = df[df['equipo_ganador'] == clase]
        df_clase_bal = df_clase[:n_ejs_clase_min]
        print(df_clase_bal)
        print("Balanceo:", len(df_clase_bal))
        df_aux = pd.concat([df_aux, df_clase_bal])
        print(len(df_aux[df_aux['equipo_ganador'] == clase]))

    # Random shuffle dataframe (pues esta ordenado segun la variable respuesta)
    df_shuf = df_aux.sample(frac=1).reset_index(drop=True)
    print(df_aux.shape)
    return df_shuf


'''
# ARBOL DE DECISION --> tengo que sacar variables string como Fecha, equipo local y equipo visitante.. pero carece de sentido resolver el prbolema
# https://www.aprendemachinelearning.com/arbol-de-decision-en-python-clasificacion-y-prediccion/#:~:text=Los%20arboles%20de%20decisi%C3%B3n%20son,(acr%C3%B3nimo%20del%20ingl%C3%A9s%20CART).
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

cv = KFold(n_splits=10)  # Numero deseado de "folds" que haremos --> CROSS VALIDATION, k = 10
accuracies = list()
max_attributes = len(list(df))
depth_range = range(1, max_attributes + 1)

# Testearemos la profundidad de 1 a cantidad de atributos +1
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth=depth,
                                             class_weight={'l': 1, 'e':1, 'v':1})
    for train_fold, valid_fold in cv.split(df):
        f_train = df.loc[train_fold]
        f_valid = df.loc[valid_fold]

        model = tree_model.fit(X=f_train.drop(['equipo_ganador'], axis=1), y=f_train["equipo_ganador"])
        valid_acc = model.score(X=f_valid.drop(['equipo_ganador'], axis=1), y=f_valid["equipo_ganador"])  # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df_results = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df_results = df_results[["Max Depth", "Average Accuracy"]]
print(df_results.to_string(index=False))


# Crear arrays de entrenamiento y las etiquetas que indican el resultado
y_train = df['equipo_ganador']
x_train = df.drop(['equipo_ganador'], axis=1).values

# Crear Arbol de decision con profundidad = 4
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth=4,
                                            class_weight={'l': 1, 'e': 1, 'v': 1})
decision_tree.fit(x_train, y_train)

# exportar el modelo a archivo .dot
with open(r"tree1.dot", 'w') as f:
    f = tree.export_graphviz(decision_tree,
                             out_file=f,
                             max_depth=7,
                             impurity=True,
                             feature_names=list(df.drop(['equipo_ganador'], axis=1)),
                             class_names=['l', 'e', 'v'],
                             rounded=True,
                             filled=True)

# Convertir el archivo .dot a png para poder visualizarlo
# check_call(['dot', '-Tpng', r'tree1.dot', '-o', r'tree1.png'])
# PImage("tree1.png")
'''