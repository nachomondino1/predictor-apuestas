# Importo librerias
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy


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
    cant_clases_perc = int(round(0.1 * cant_clases, 0))  # cantidad de clases utilizando percentiles
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

    df_aux = pd.DataFrame(columns=['y_real', 'Local', 'Empate', 'Visitante', 'y_pred'])
    # Definicion de variables
    l_clases = df['equipo_ganador'].unique()
    l_atrib = list(df.drop(['equipo_ganador'], axis=1).columns) # Sin variable respuesta
    df_result = pd.DataFrame()
    df_result['y_real'] = df['equipo_ganador']

    # Por registro
    for i in range(len(df)):

        df_aux.loc[i, 'y_real'] = df.loc[i, 'equipo_ganador']

        # Definicion de variables
        prob_max = 0  # Probabilidad mas alta respecto de una clase
        prob_den = 0  # Probabilidad del denominador. Para calcular la probabilidad respecto de cada clase de 0 a 1.
        dic = {}

        # Paso 3: Multiplicar P(ai/vj) y P(vj)
        for clase in l_clases:

            # Definicion de variables
            prob = d[clase]  # Inicializo variable. Probabilidad de que sea de una clase dados ciertos atributos

            # Por atributo
            for atrib in l_atrib:

                valor = df.loc[i, atrib]
                name = '{}/{}'.format(atrib+str(valor), clase)

                # Existe la posibilidad que un valor no tenga una probabildad calculada puesto que no ocurrio. (e.g. Barracas nunca empato de local, por lo que, P(equipoloc = Barracas / empato) no existe en el diccionario d)
                try:
                    prob *= d[name]
                    print("P({}/{}) = {}".format(atrib+str(valor), clase, d[name]))
                except:
                    prob *= 1

            # Guardo probabilidad de que pertenezca a la clase
            print("Prob que sea clase {}: {}".format(clase, prob))
            if prob > prob_max:
                prob_max = prob
                clase_max = clase

            dic[clase] = prob

            # Calculo prob del denominador para poder calcular la prob de una clase dado ciertos atrib
            prob_den += prob
            print("Prob denominador: ", prob_den)

        # Imprimo resultados
        prob_clase = prob_max / prob_den * 100
        print("Dados los atributos, se infiere que esta persona es {} con una prob de {:.0f}% \n".format(clase_max, prob_clase))
        df_result.loc[i, ['y_pred', 'y_pred_prob']] = clase_max, prob_clase

        df_aux.loc[i, 'y_pred'] = clase_max
        df_aux.loc[i, 'Local'] = dic['Local'] / prob_den * 100
        df_aux.loc[i, 'Empate'] = dic['Empate'] / prob_den * 100
        df_aux.loc[i, 'Visitante'] = dic['Visitante'] / prob_den * 100

    return df_result, df_aux

def main():
    # Levanto el dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/integrate_data/df_derived_data.xlsx',index_col=0)
    print(df)
    print(df.shape)

    # Remuevo columnas fecha
    df.drop('id', inplace=True, axis=1)
    df.drop('fecha', inplace=True, axis=1)
    df.drop('goles_loc', inplace=True, axis=1)
    df.drop('goles_vis', inplace=True, axis=1)
    df['dif_gol'] = df['dif_gol_loc'] - df['dif_gol_vis']
    df.drop('dif_gol_loc', inplace=True, axis=1)
    df.drop('dif_gol_vis', inplace=True, axis=1)
    print(df.shape)

    # Remuevo ultimos  partidos de cada equipo (pues no puedo calcular bien la forma)
    df = df.dropna(subset='dif_forma_pond').reset_index(drop=True)  # Son 15 partidos por jornada y saco las ultimas 10 jornadas pues la forma la calculo 10 partidos para atras...
    print(df.shape)
    print(df.head)

    # Categorizo columnas numericas (IRIA EN CLEAN_DATA?)
    df = categorize_numeric_columns(df)

    # Corro 100 modelos y promedio resultados
    N_CORRIDAS = 1
    l_aciertos = []
    l_prob = []
    l_prob_2 = []
    for i in range(N_CORRIDAS):
        print(" MODELO Nº{} ".format(i).center(120, '#'))

        # Balanceo dataset  (Para mi en este proeycto no conviene balancear el dataset. Puesto que es importante que es mas probable ganar de local que de visitante). Sin embargo entiendo que el modelo puede parecer que da bien pero en realidad siempre decir que gana el local...
        # df = balance_dataset(df_shuf)

        # Shuffle dataframe
        df = df.sample(frac=1).reset_index(drop=True)

        # Separo conjunto de datos en train y test
        corte = int(0.8 * len(df))
        df_train, df_test = df.loc[:corte].reset_index(drop=True), df.loc[corte:].reset_index(drop=True)

        # Implemento Naive Bayes
        d = train_naive_bayes(df_train)
        print(d)

        # EVALUACION DEL MODELO
        df_result, df_aux = predict(df_test, d)

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

    df.to_excel("./df_results_naive.xlsx")
    df_aux.to_excel("./df_aux.xlsx")

main()