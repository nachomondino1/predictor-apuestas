# Importo librerias
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense


# Naive Bayes
# Discretizar variables
def categorize_numeric_columns(df):
    """
    Dado un dataframe, categoriza sus columnas numericas (las no numericas no porque al no haber una "distancia" entre
    strings, no puedo determinar cual se asemeja con cual) continuas (las discretas no pues ya estan categorizadas)
    :param df: Dataframe. Unidad de analisis: cualquiera. Columnas: cualquiera.
    :return: Dataframe. Unidad de analisis: cualquiera. Columnas: cualquiera. Todas sus columnas numericas continuas
    ahora son numericas discretas
    """
    # Defino variables
    # POR COLUMNA DEL DATAFRAME
    for columna in list(df.columns):
        print(columna.upper().center(120))

        # SI LA COLUMNA ES NUMERICA
        if (df[columna].dtype == 'float64') or (df[columna].dtype == 'int64'):

            # Defino variables
            n_clases = len(df[columna].unique())
            n_clases_opt = int(len(df[columna].dropna()) ** 0.5)

            # SI LA COLUMNA ES CONTINUA (toma muchos valores distintos, especificamente, mas que la cantidad optima)
            if n_clases > n_clases_opt:
                print("Sera categorizada pues tiene {} valores unicos cuando, en este caso, lo recomendado es {}.".format(n_clases, n_clases_opt))

                # OBTENGO VALORES MEDIOS Y MAXIMOS DE CADA CLASE
                d = create_classes(valores=df[columna], cant_clases=n_clases_opt)  # key=valor_max y val=valor_med

                # REEMPLAZO VALORES CONTINUOS POR LA MEDIA DE LA CLASE A LA QUE PERTENECE
                # Por valor del atributo
                for i in range(len(df[columna])):

                    # Por valor maximo de las clases
                    for valor_limite in list(d.keys()):

                        # Si el valor es menor al valor maximo de la clase
                        if df[columna].iloc[i] <= valor_limite:

                            # reemplazo valor por el valor medio de la clase
                            df.loc[i, columna] = d[valor_limite]

                            # dejo de comparar el valor con los valores maximos de las clases pues ya encontre su clase
                            break

            # SI LA COLUMNA ES DISCRETA (toma pocos valores distintos)
            else:
                # imprimo mensaje
                print("Es numerica pero discreta pues toma {} valores!".format(n_clases))

        # SI LA COLUMNA NO ES NUMERICA
        else:
            # imprimo mensaje
            print("No es numerica!")
    return df

def create_classes(valores, cant_clases):
    """
    Genera clases o categoria para un conjunto de valores numericos continuos
    :param valores: Lista de valores de una columna numerica continua
    :param cant_clases: Cantidad de clases a generar
    :return: Diccionario con valores maximos de cada clase como key y con valores medios de cada clase como value
    """
    # DEFINO VARIABLES
    # respecto de valores
    valores_unicos = sorted(valores.dropna().unique())
    valor_min, valor_max = min(valores_unicos), max(valores_unicos)
    print("Valores unicos: ", valores_unicos)
    # respecto de clases
    rango = valor_max - valor_min
    amplitud_clase = rango / cant_clases
    cant_clases_perc = int(round(0.65 * cant_clases, 0))  # cantidad de clases utilizando percentiles
    percentiles = 1 / cant_clases_perc  # percentil
    # inicializo variables
    d = {}  # diccionario a retornar (con valores maximos y medios de cada clase)
    PORC_MIN_CLASES_CON_VALOR, PORC_MAX_CLASES_CON_VALOR = 0.3, 0.72  # porcentajes min y max de clases con valores (es decir, no vacias)

    # CREO CLASES CON MISMA AMPLITUD
    print("Creo {} clases con amplitud de {:.2f}".format(cant_clases, amplitud_clase))
    print("{:^10s}\t{:^10s}\t{:^10s}\t{:^10s}".format("Clase Nº","Valor min", "Valor med", "Valor max"))
    # Por clase
    for i in range(cant_clases):
        # Determino valores minimo, medio y maximo de la clase
        valor_min_clase = round(valor_min + amplitud_clase * i, 2)  # valor min para estar en clase i
        valor_max_clase = round(valor_min + amplitud_clase * (i + 1), 2)  # valor max para estar en clase i
        valor_med_clase = round((valor_max_clase + valor_min_clase) / 2, 2)  # valor medio de clase i

        # Guardo valor medio y maximo de la clase
        d[valor_max_clase] = valor_med_clase
        print("{:^10d}\t{:^10.1f}\t{:^10.1f}\t{:^10.1f}".format(i+1, valor_min_clase, valor_med_clase, valor_max_clase))

    # Imprimo resultados de distribucion de valores en clase
    cant_val_por_clase = values_distribution_in_classes(d, valores_unicos)

    # Determino % de clases con al menos un valor
    cant_clases_con_valor = len(cant_val_por_clase) - cant_val_por_clase.count(0)
    porc_clases_con_valor = cant_clases_con_valor / cant_clases

    # SI LA DISTRIBUCION DE VALORES EN CLASES NO ES BUENA
    # Si menos del 50% de las clases tienen valores o mas del 71%
    if (porc_clases_con_valor < PORC_MIN_CLASES_CON_VALOR) or (porc_clases_con_valor > PORC_MAX_CLASES_CON_VALOR):

        # Imprimo razon, por la que, vuelvo a generar clases
        print("No funciono bien la creacion de clases con misma amplitud. Razon: ", end="")
        if porc_clases_con_valor < PORC_MIN_CLASES_CON_VALOR:
            print("Hay pocas clases con valores, es decir, hay una gran concentracion de valores en pocas clases. "
                  "Valores muy distintos tomaran mismo sentiment por estar en misma clase")
        else:
            print("Hay muchas clases con valores. Valores tendran sentiment poco robusto")
        print("Ahora, generare {} clases a partir de tomar percentiles {}".format(cant_clases_perc, percentiles))

        # Defino variables
        d = {}  # reinicio diccionario pues no usare clases de misma amplitud

        # CREO CLASES A PARTIR DE PERCENTILES
        # Por clase
        print("{:^10s}\t{:^10s}\t{:^10s}\t{:^10s}".format("Clase Nº", "Valor min", "Valor med", "Valor max"))
        for i in range(cant_clases_perc):

            # Obtengo indices de valor min y max para la clase
            idx_valor_min_clase = int(len(valores_unicos) * percentiles * i)  # valor min para estar en clase i
            idx_valor_max_clase = int(len(valores_unicos) * percentiles * (i + 1)) - 1  # valor min para estar en clase i. El -1 seria porque el idx de la lista arranca en 0

            # Obtengo valores min y max de la clase a partir de los indices
            valor_min_clase = valores_unicos[idx_valor_min_clase]
            valor_max_clase = valores_unicos[idx_valor_max_clase]
            valor_med_clase = (valor_max_clase + valor_min_clase) / 2  # valor medio de clase i

            # Guardo valor maximo y medio de la clase
            d[valor_max_clase] = valor_med_clase
            print("{:^10d}\t{:^10.1f}\t{:^10.1f}\t{:^10.1f}".format(i + 1, valor_min_clase, valor_med_clase, valor_max_clase))

        # Imprimo resultados de distribucion de valores en clases
        values_distribution_in_classes(d, valores_unicos)

    # SI LA DISTRIBUCION DE VALORES EN CLASES ES BUENA
    else:
        # IMPRIMO MENSAJE
        print("Funciono correctamente la creacion de clases con misma amplitud! ")

    return d

def values_distribution_in_classes(dict, valores_unicos):
    """
    Obtiene la distribucion de los valores unicos en las clases, es decir, la cantidad de valores unicos por clase.
    :param dict: Diccionario con valores maximos de cada clase como key y con valores medios de cada clase como value
    :param valores_unicos: Lista de valores unicos de una columna numerica continua
    :return: Lista de cantidad de valores unicos por clase
    """
    # Inicializo diccionario a retornar
    d = {}
    for key in dict.keys():
        d[key] = 0

    # Por valor unico
    for valor in valores_unicos:

        # Por valor maximo de clase
        for valor_max_clase in list(dict.keys()):

            # si el valor es menor al valor maximo de clase
            if valor <= valor_max_clase:  # si el valor unico estaria en clase

                # sumo 1 a clase a la que pertenece el valor
                d[valor_max_clase] += 1

                break  # para no seguir comparando valor con otros valores maximos de clases

    print("Distribucion de valores unicos en clases: ", list(d.values()))
    return list(d.values())

def train_naive_bayes(df):
    # Definicion de variavles
    d = {}
    l_clases = df['equipo_ganador'].unique()
    k = len(l_clases)

    # Paso 1: Obtener estimacion de P(vj) (en este caso, P(escoces) y P(ingles)
    for clase in l_clases:
        d[clase] = len(df[df['equipo_ganador'] == clase]) / len(df)

    # Paso 2: Por cada valor de cada atributo (e.g. atrib scones toma valor 0 o 1), calcular P(ai/vj)
    # Por atributo
    for atributo in list(df.columns)[:-1]: # No debo incluir la variable respuesta (en este caso, Nacionalidad)

        # Por valor
        for valor in df[atributo].unique():

            # Por clase
            for clase in l_clases:

                # Calculo P(valor atrib / clase)
                name = '{}/{}'.format(atributo+str(valor), clase)
                numerador = len(df[(df[atributo] == valor) & (df['equipo_ganador'] == clase)]) + 1
                denominador = len(df[df['equipo_ganador'] == clase]) + k  # Correccion de Laplace
                d[name] = numerador / denominador
    return d


def predict(df , d, ejemplo_a_pred):

    # Definicion de variables
    prob_max = 0
    prob_den = 0
    l_clases = df['equipo_ganador'].unique()
    l_atrib = list(df.columns)[:-1]  # Sin variable respuesta

    # Paso 3: Multiplicar P(ai/vj) y P(vj)
    for clase in l_clases:

        # Definicion de variables
        prob = d[clase]  # Inicializo variable. Probabilidad de que sea de una clase dados ciertos atributos

        # Por atributo
        for i in range(len(ejemplo_a_pred)):

            valor = ejemplo_a_pred[i]
            atrib = l_atrib[i]
            name = '{}/{}'.format(atrib+str(valor), clase)
            prob *= d[name]
            print("P({}/{}) = {}}".format(atrib+str(valor), clase, d[name]))

        # Guardo probabilidad de que pertenezca a la clase
        print("Prob que sea clase {}: {}".format(clase, prob))
        if prob > prob_max:
            prob_max = prob
            clase_max = clase

        # Calculo prob del denominador para poder calcular la prob de una clase dado ciertos atrib
        prob_den += prob
        print("Prob denominador: ", prob_den)

    # Imprimo resultados
    prob_clase = prob_max / prob_den * 100
    print("Dados los atributos, se infiere que esta persona es {} con una prob de {:.0f}%".format(clase_max, prob_clase))

def main():
    # Levanto el dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data understanding/integrate_data/df_derived_data.xlsx',index_col=0)
    print(df)

    # Remuevo columnas forma_loc y forma_vis
    df.drop('fecha', inplace=True, axis=1)
    # df.drop('equipo_loc', inplace=True, axis=1)
    # df.drop('equipo_vis', inplace=True, axis=1)
    df.drop('forma_loc', inplace=True, axis=1)
    df.drop('forma_vis', inplace=True, axis=1)
    print(df)

    # Remuevo ultimos 10 partidos de cada equipo (pues no puedo calcular bien la forma)
    df = df.loc[:len(df)-150]  # Son 15 partidos por jornada y saco las ultimas 10 jornadas pues la forma la calculo 10 partidos para atras...

    # Categorizo columnas numericas
    df = categorize_numeric_columns(df)

    # Separo conjunto de datos en train y test


    # Implemento Naive Bayes
    d = train_naive_bayes(df)

    # Predigo
    predict(df, d, ['Talleres Córdoba', 'Boca Juniors', , 0])
    # Tengo que DIVIDIR CONJUNTO DE DATOS EN TEST Y TRAIN

main()

'''
# RED NEURONAL
# https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/
# cargamos las 4 combinaciones de las compuertas XOR
# training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
training_data = df.drop(['equipo_ganador'], axis=1).to_numpy()

# y estos son los resultados que se obtienen, en el mismo orden
# target_data = np.array([[0], [1], [1], [0]], "float32")  # <class 'numpy.ndarray'>
target_data = df['equipo_ganador'].to_numpy()  # <class 'numpy.ndarray'>

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