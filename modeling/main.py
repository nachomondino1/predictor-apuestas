# Importo librerias
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy



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
    valores_unicos = sorted(valores.dropna().unique())
    print("Valores unicos: ", valores_unicos)
    cant_clases_perc = int(round(0.2 * cant_clases, 0))  # cantidad de clases utilizando percentiles
    percentiles = 1 / cant_clases_perc  # percentil
    d = {}  # diccionario a retornar (con valores maximos y medios de cada clase)

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
    for atributo in list(df.drop(['equipo_ganador'], axis=1).columns): # No debo incluir la variable respuesta (en este caso, Nacionalidad)

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


def predict(df , d):
    # df es df_test

    # Definicion de variables
    l_clases = df['equipo_ganador'].unique()
    l_atrib = list(df.drop(['equipo_ganador'], axis=1).columns) # Sin variable respuesta
    df_result = pd.DataFrame()
    df_result['y_real'] = df['equipo_ganador']

    # Por registro
    for i in range(len(df)):

        # Definicion de variables
        prob_max = 0  # Probabilidad mas alta respecto de una clase
        prob_den = 0  # Probabilidad del denominador. Para calcular la probabilidad respecto de cada clase de 0 a 1.

        # Paso 3: Multiplicar P(ai/vj) y P(vj)
        for clase in l_clases:

            # Definicion de variables
            prob = d[clase]  # Inicializo variable. Probabilidad de que sea de una clase dados ciertos atributos

            # Por atributo
            for atrib in l_atrib:

                valor = df.loc[i, atrib]
                name = '{}/{}'.format(atrib+str(valor), clase)
                prob *= d[name]
                print("P({}/{}) = {}".format(atrib+str(valor), clase, d[name]))

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
        df_result.loc[i, ['y_pred', 'y_pred_prob']] = clase_max, prob_clase

    return df_result

def balance_dataset(df):

    # Definicion de variables
    df_aux = pd.DataFrame(columns=df.columns)
    n_ejs_clase_min = 1000000000
    l_clases = list(df['equipo_ganador'].unique())
    print("ASFSDAFGSAD", l_clases)

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

def main():
    # Levanto el dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data understanding/integrate_data/df_derived_data.xlsx',index_col=0)
    print(df)
    print(df.shape)

    # Remuevo columnas fecha
    df.drop('fecha', inplace=True, axis=1)
    # df.drop('equipo_loc', inplace=True, axis=1)  # Tiene sentido calcular P(equipo_loc = Banfield / equipo_gan = Local) --> un equipo puede ser mas propenso a ganar de local que otro...
    # df.drop('equipo_vis', inplace=True, axis=1)
    print(df.shape)

    # Remuevo ultimos 10 partidos de cada equipo (pues no puedo calcular bien la forma)
    df = df.dropna(subset='dif_forma').reset_index(drop=True)  # Son 15 partidos por jornada y saco las ultimas 10 jornadas pues la forma la calculo 10 partidos para atras...
    print(df.shape)
    print(df.head)

    # Categorizo columnas numericas
    df = categorize_numeric_columns(df)

    # Corro 100 modelos y promedio resultados
    N_CORRIDAS = 100
    l_aciertos = []
    l_prob = []
    l_prob_2 = []
    for i in range(N_CORRIDAS):
        print(" MODELO Nº{} ".format(i).center(120, '#'))

        # Random shuffle del df
        df_shuf = df.sample(frac=1).reset_index(drop=True)  # Random shuffle dataframe

        # Balanceo dataset  (Para mi en este proeycto no conviene balancear el dataset. Puesto que es importante que es mas probable ganar de local que de visitante). Sin embargo entiendo que el modelo puede parecer que da bien pero en realidad siempre decir que gana el local...
        # df = balance_dataset(df_shuf)

        # Separo conjunto de datos en train y test
        corte = int(0.8 * len(df))
        df_train, df_test = df.loc[:corte].reset_index(drop=True), df.loc[corte:].reset_index(drop=True)

        # Implemento Naive Bayes
        d = train_naive_bayes(df_train)
        print(d)

        # EVALUACION DEL MODELO
        df_result = predict(df_test, d)

        n_aciertos = 0
        prob_certeza = 0
        for i in range(len(df_result)):
            y_real = df_result.loc[i, 'y_real']
            y_pred = df_result.loc[i, 'y_pred']

            if y_real == y_pred:
                n_aciertos += 1
                prob_certeza += df_result.loc[i, 'y_pred_prob']

        # Guardo resultados del modelo
        l_aciertos.append(n_aciertos/len(df_result)*100)
        l_prob.append(prob_certeza/n_aciertos)
        l_prob_2.append(sum(df_result['y_pred_prob'])/len(df_result))
        print(l_aciertos)
        print(l_prob)
        print(l_prob_2)

    print("El % de aciertos es {:.3f}%".format(sum(l_aciertos)/N_CORRIDAS))
    print("La probabilidad de certeza promedio en los aciertos es {:.3f}%".format(sum(l_prob)/N_CORRIDAS))
    print("La probabilidad de certeza promedio es {:.3f}%".format(sum(l_prob_2)/N_CORRIDAS))


    # MATRIZ DE CONFUSION
    confusion_matrix = metrics.confusion_matrix(df_result['y_real'], df_result['y_pred'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Local", "Empate", "Visitante"])
    cm_display.plot()
    plt.show()

    # df.to_excel("./df_results.xlsx")

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