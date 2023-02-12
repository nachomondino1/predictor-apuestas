import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense

df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data understanding/integrate data/df_derived_data.xlsx', index_col=0)
print(df)

# convierto columna "Resultado" en numerica (1 --> Gano el local, 2 --> empate, 3 --> gano el vis)
for i in range(len(df)):
    if df.loc[i, 'Resultado'] == df.loc[i, 'Equipo local']:
        df.loc[i, 'Resultado'] = 'l'
    elif df.loc[i, 'Resultado'] == "Empate":
        df.loc[i, 'Resultado'] = 'e'
    else:
        df.loc[i, 'Resultado'] = 'v'

# Remuevo columnas forma_loc y forma_vis
df.drop('Fecha', inplace=True, axis=1)
df.drop('Equipo local', inplace=True, axis=1)
df.drop('Equipo Visitante', inplace=True, axis=1)
df.drop('forma_loc', inplace=True, axis=1)
df.drop('forma_vis', inplace=True, axis=1)
print(df)

# Remuevo ultimos 10 partidos de cada equipo (pues no puedo calcular bien la forma)
df = df.loc[:600]

'''
# RED NEURONAL
# https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/
# cargamos las 4 combinaciones de las compuertas XOR
# training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
training_data = df.drop(['Resultado'], axis=1).to_numpy()

# y estos son los resultados que se obtienen, en el mismo orden
# target_data = np.array([[0], [1], [1], [0]], "float32")  # <class 'numpy.ndarray'>
target_data = df['Resultado'].to_numpy()  # <class 'numpy.ndarray'>

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)

# evaluamos el modelo
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print(model.predict(training_data).round())
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

        model = tree_model.fit(X=f_train.drop(['Resultado'], axis=1), y=f_train["Resultado"])
        valid_acc = model.score(X=f_valid.drop(['Resultado'], axis=1), y=f_valid["Resultado"])  # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos
df_results = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df_results = df_results[["Max Depth", "Average Accuracy"]]
print(df_results.to_string(index=False))


# Crear arrays de entrenamiento y las etiquetas que indican el resultado
y_train = df['Resultado']
x_train = df.drop(['Resultado'], axis=1).values

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
                             feature_names=list(df.drop(['Resultado'], axis=1)),
                             class_names=['l', 'e', 'v'],
                             rounded=True,
                             filled=True)

# Convertir el archivo .dot a png para poder visualizarlo
# check_call(['dot', '-Tpng', r'tree1.dot', '-o', r'tree1.png'])
# PImage("tree1.png")