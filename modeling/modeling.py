import pandocfilters
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import xgboost as xgb

# Para red neuronal
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Embedding, Flatten, Dropout
# from keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder

# Para Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dspy.data_preparation import clean_data
from dspy.modeling import naive_bayes, test_design


def equipo_ganador(df):  # Para red neuronal y xgboost

    for i in range(len(df)):
        str = df.loc[i, 'equipo_ganador']
        nro = 2 if str == 'Local' else 0 if str =='Visitante' else 1
        df.loc[i, 'equipo_ganador'] = nro
    return df


def red_neuronal(data):  # Tira error que desconozco, ni siquiera es muy googleable

    # Preprocesamiento de variables categóricas
    cat_cols = ['equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis']
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Preprocesamiento de variables numéricas
    num_cols = ['dif_gol','historial_entre_si', 'dif_posesion', 'dif_remates']
    data[num_cols] = (data[num_cols] - data[num_cols].mean()) / data[num_cols].std()

    # Creación de los conjuntos de entrenamiento y prueba
    X = data.drop('equipo_ganador', axis=1)
    y = data['equipo_ganador']
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construcción del modelo de red neuronal
    model = Sequential()
    model.add(Embedding(1000, 8, input_length=X_train.shape[1]))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenamiento y evaluación del modelo
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    return scores[1]

def xgboost(df):  # Solo variables numericas y no funciona error "ValueError: Classification metrics can't handle a mix of unknown and multiclass targets"

    df = df.drop(['equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis'], axis=1)  # No acepta categoricas
    X = df.drop('equipo_ganador', axis=1)
    y = df['equipo_ganador']

    # Separación de datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'merror'
    }

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, d_train, num_boost_round=100)

    y_pred = model.predict(d_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return accuracy

def naive_bayes_lib(df):  # No puedo convertir las categoricas a numericas y encima tira error ValueError: could not convert string to float: 'Visitante' hasta cuando uso equipo_ganador()

    df = df.drop(['equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis'], axis=1)  # No acepta categoricas

    X = df.drop('equipo_ganador', axis=1)
    y = df['equipo_ganador']

    # Se normalizan los datos
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ''' # No funciona
    # Seleccionar las variables predictoras categóricas y numéricas
    cat_cols = ['equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis']

    # Definir el transformador para la codificación one-hot
    ct = ColumnTransformer([
        ('onehot', OneHotEncoder(), cat_cols)],
        remainder='passthrough')

    # Combinar el transformador y el modelo en un pipeline
    pipe = Pipeline([
        ('transform', ct),
        ('nb', nb)
    ])
    
    # Entrenar el modelo en los datos de entrenamiento
    pipe.fit(X_train, y_train)
    print('Precisión:', pipe.score(X_test, y_test))
    '''

    # Definir el modelo de Naive Bayes Multinomial
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluar la precisión del modelo en los datos de prueba
    y_pred = model.predict(y_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return accuracy


# Pruebo funcion para implementar cambios tod@ junto
def train_naive_bayes_2(df, var_resp):  # PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    """
    Entrena un modelo de Naive Bayes determinando las probibilades correspondientes.
    :param df: Dataframe train. Columnas: cualquier numero de variables dependientes y, necesiaramente al final, la
    variable respuesta.
    :return: Dataframe. Columnas: la variable respuesta y cada variable dependiente por cada valor que toma. Index:
    valores de la variable respuesta. Celdas: probabilidades de que tal variable dependiente valga tal valor dado que
    la variable respuesta es tal valor.
    """
    # Definicion de variables
    d = {}  # Diccionario donde cargare las columnas para el dataset df_prob
    l_atrib = list(df.drop(var_resp, axis=1).columns)
    l_clases = df[var_resp].unique()  # Valores de la variable respuesta (debe ser categorica)
    k = len(l_clases)  # Cantidad de valores de la variable respuesta (utilizado en correccion de Laplace)

    # Paso 1: Obtener estimacion de P(vj) (en este caso, P(escoces) y P(ingles)
    d[var_resp] = [len(df[df[var_resp] == clase]) / len(df) for clase in l_clases]

    # Paso 2: Por cada valor de cada atributo (e.g. atrib scones toma valor 0 o 1), calcular P(ai/vj)
    # Por atributo (sin incluir la variable respuesta)
    for atributo in l_atrib:
        print(f'Atributo: {atributo}')

        # Por valor del atributo
        for valor_atr in df[atributo].unique():

            print(f'\t Valor del atributo: {valor_atr}', end='. ')
            l = []

            # Por clase (valor de la variable respuesta)
            for clase in l_clases:

                # Calculo P(valor atrib / clase)
                numerador = len(df[(df[atributo] == valor_atr) & (df[var_resp] == clase)]) + 1  # el +1 es por la correccion de Laplace
                denominador = len(df[df[var_resp] == clase]) + k  # el +k es por la correccion de Laplace
                l.append(numerador/denominador)
                print(f'\t Probabilidad para la clase {clase}: {numerador/denominador}')

            # Agrego la columna
            d[f'{atributo}={valor_atr}'] = l

    # Concateno columnas del
    df_prob = pd.DataFrame(d, index=l_clases)  # Convierto diccionario en Dataframe (asi evito error de Highly fragmented)
    return df_prob

def predict_naive_bayes_2(df_prob, df_test, var_resp, col_prob_clase=False):  # PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    """
    Predice la clase de cada nuevo registro usando el modelo entrenado de Naive Bayes.
    :param df_prob: Dataframe. Columnas: la variable respuesta y cada variable dependiente por cada valor que toma.
    Index: valores de la variable respuesta. Celdas: probabilidades de que tal variable dependiente valga tal valor dado
    que la variable respuesta es tal valor.
    :param df_test: Dataframe test. Columnas: cualquier numero de variables dependientes y, necesiaramente al final, la
    variable respuesta.
    :return: Dataframe test mas una columna con la clase predicha por el modelo
    """
    # Definicion de variables
    l_atrib = list(df_test.columns)
    l_atrib.remove(var_resp)
    l_clases = list(df_prob.index)  # Valores que puede tomar la variable respuesta

    # Por registro a predecir
    for i in range(len(df_test)):

        # Reinicio variables
        prob_max, prob_den = 0, 0

        # Paso 3: Multiplicar P(ai/vj) y P(vj)
        # Por clase (valor de la variable respuesta)
        for clase in l_clases:

            # Definicion de variables
            prob = df_prob.loc[clase, var_resp]  # Inicializo variable. Probabilidad de que sea de una clase dados ciertos atributos

            # Por atributo
            for atrib in l_atrib:

                valor_atr = df_test.loc[i, atrib]
                col_name = '{}={}'.format(atrib, valor_atr)
                try:
                    prob *= df_prob.loc[clase, col_name]
                    print(f"P({col_name}/{clase}) = {df_prob.loc[clase, col_name]}")
                except KeyError:
                    print(f"Fallo la extraccion de la probabilidad {col_name}")

            # Guardo probabilidad de que pertenezca a la clase
            print(f"Prob que sea clase {clase}: {prob}")
            if prob > prob_max:
                prob_max = prob
                clase_max = clase

            # Calculo prob del denominador para poder calcular la prob de una clase dado ciertos atrib
            prob_den += prob
            print("Prob denominador: ", prob_den)

        # Guardo resultados
        prob_clase = prob_max / prob_den * 100
        df_test.loc[i, 'y_pred'] = clase_max
        # print("Dados los atributos, se infiere que es {} con una prob de {:.0f}%".format(clase_max, prob_clase))

        # Si el usuario quiere la probabilidad de la clase predicha
        if col_prob_clase:
            # Guardo probabilidad de la clase predicha
            df_test.loc[i, 'y_pred_prob'] = prob_clase

    return df_test

def naive_bayes_propio(df):

    # Separo conjunto de datos en train y test
    df_train, df_test = test_design.separate_train_and_test(df)

    # Implemento Naive Bayes
    modelo_nb = train_naive_bayes_2(df_train, var_resp='equipo_ganador')
    # modelo_nb.to_excel('/Users/nachomondino/Desktop/df_prob.xlsx')

    # 5) EVALUACION DEL MODELO
    df_result = predict_naive_bayes_2(modelo_nb, df_test, var_resp='equipo_ganador', col_prob_clase=True)
    # df_result.to_excel('/Users/nachomondino/Desktop/df_result.xlsx')

    n_aciertos, prob_certeza = 0, 0
    for i in range(len(df_result)):

        # Si el modelo predijo bien
        if df_result.loc[i, 'equipo_ganador'] == df_result.loc[i, 'y_pred']:
            n_aciertos += 1
            prob_certeza += df_result.loc[i, 'y_pred_prob']

    # Guardo resultados del modelo
    return n_aciertos / len(df_result) * 100

def main():
    N_MODELOS = 10

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Desktop/df_prepared.xlsx', index_col=0)
    print(df.head())

    df = df.dropna(subset=['historial_entre_si']).reset_index()  # Elimina filas con al menos un valor nulo

    df = df.drop(['dif_ataques'], axis=1)

    # df = equipo_ganador(df)  # Para red neuronal y xgboost (para naive no)

    # df = clean_data.balance_dataset(df, var_resp='equipo_ganador')

    l_aciertos = []
    # Por modelo
    for i in range(N_MODELOS):

        # Shuffle dataset
        df = df.sample(frac=1).reset_index(drop=True)

        # Entreno modelo y evaluo
        # precision = naive_bayes_propio(df)
        # precision = red_neuronal(df)
        # precision = xgboost(df)
        precision = naive_bayes_lib(df)

        l_aciertos.append(precision)

    print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos)/len(l_aciertos)}")

main()