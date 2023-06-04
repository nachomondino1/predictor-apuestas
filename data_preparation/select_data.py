# Importo librerias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


def select_best_features(features, importance, threshold):
    l_selected_features = []

    # Selecciono caracteristicas mas importantes segun threshold
    for feature, imp in zip(features, importance):

        if imp > threshold:
            l_selected_features.append(feature)
    return l_selected_features


# Opción 1: Buscamos variables mas importantes con un random forest
def random_forest(X, y, threshold=0.1):

    # Crear un clasificador Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Obtener la importancia de las características
    importances = clf.feature_importances_

    graficar_importancia_atrib(clf, X)  # Probe a graficar puesto que es un arbol...

    # Selecciono caracteristicas mas importantes segun threshold
    l_selected_features = select_best_features(features=list(X.columns), importance=importances, threshold=threshold)
    print("Características seleccionadas:\n", l_selected_features)
    return l_selected_features

# Opción 2: Arbol de decision
def arbol(X_train, y_train, threshold):

    # Definicion de variables
    l_selected_features = []

    # Entreno arbol de decision
    dt = DecisionTreeClassifier(max_depth=7)
    dt.fit(X_train, y_train)

    # Selecciono caracteristicas mas importantes segun threshold
    select_best_features(features=X_train.columns, importance=dt.feature_importances_, threshold=threshold)

    graficar_importancia_atrib(dt, X_train)

    print("Características seleccionadas:\n", l_selected_features)
    return l_selected_features

def graficar_importancia_atrib(model, X_train):

    features = X_train.columns
    feature_importances = model.feature_importances_

    # Crear figura
    fig = go.Figure()

    # Agregar barras al gráfico
    fig.add_trace(go.Bar(
        x=feature_importances,
        y=features,
        orientation='h'
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title='Importancia de las características',
        xaxis_title='Importancia',
        yaxis_title='Características',
        yaxis=dict(autorange="reversed")  # Invertir el orden de las características
    )

    # Mostrar el gráfico
    fig.show()

# Opción 3: Análisis univariable con tests estadísticos
def analisis_univariable(X, y, threshold):

    # Definicion de variables
    l_selected_features, l_scores = [], []

    # Selecciono variables numericas y categoricas
    numeric_vars = X.select_dtypes(include='number').columns.tolist()
    categorical_vars = X.select_dtypes(include='object').columns.tolist()

    # Variables predictoras numéricas
    if len(numeric_vars) > 0:
        numeric_X = X[numeric_vars].clip(lower=0)  # Asegurar que los valores sean no negativos
        numeric_selector = SelectKBest(score_func=f_classif, k='all')  # Utiliza ANOVA o f-score, selecciona las 3 mejores características
        numeric_X_selected = numeric_selector.fit_transform(numeric_X, y)
        numeric_selected_features = [numeric_vars[i] for i in range(len(numeric_vars)) if numeric_selector.get_support()[i]]
        numeric_scores = numeric_selector.scores_

        l_selected_features += numeric_selected_features
        l_scores += list(numeric_scores)

    # Variables predictoras categóricas
    if len(categorical_vars) > 0:
        categorical_X = X[categorical_vars]
        categorical_selector = SelectKBest(score_func=chi2, k='all')  # Utiliza chi-cuadrado, selecciona las 3 mejores características
        categorical_X_selected = categorical_selector.fit_transform(categorical_X, y)
        categorical_selected_features = [categorical_vars[i] for i in range(len(categorical_vars)) if categorical_selector.get_support()[i]]
        categorical_scores = categorical_selector.scores_

        l_selected_features += categorical_selected_features
        l_scores += list(categorical_scores)

    graf_imp(l_selected_features, l_scores)

    # Selecciono caracteristicas mas importantes segun threshold --> Implemento seleccion porque devuelve todas las variables...
    l_selected_features = select_best_features(features=l_selected_features, importance=l_scores, threshold=threshold)
    # Imprimir las características seleccionadas y los puntajes de relevancia
    # print(f"Características seleccionadas: \n{l_selected_features} \nPuntajes de relevancia: {l_scores}")
    print(f"Características seleccionadas: \n{l_selected_features}")
    return l_selected_features  # Devuelvo todas?

def graf_imp(selected_features, scores):

    # Crear gráfico de barras
    fig = go.Figure(data=go.Bar(x=selected_features, y=scores))

    # Configurar etiquetas y título del gráfico
    fig.update_layout(
        xaxis=dict(title='Características'),
        yaxis=dict(title='Puntajes de relevancia'),
        title='Puntajes de relevancia de las características'
    )

    # Rotar etiquetas en el eje x
    fig.update_layout(xaxis_tickangle=-45)

    # Mostrar el gráfico
    fig.show()

def feature_selection(df, var_resp):

    df = df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
    X = df.drop([var_resp], axis=1)
    y = df[var_resp]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dimensiones de los conjuntos de entrenamiento:", X_train.shape, y_train.shape)
    print("Dimensiones de los conjuntos de prueba:", X_test.shape, y_test.shape)

    # Opcion 1
    l_selected_features_1 = random_forest(X, y, threshold=0.03)

    # Opcion 2? es parte de la opcion 1?
    l_selected_features_2 = arbol(X_train, y_train, threshold=0.03)

    # Opcion 3
    l_selected_features_3 = analisis_univariable(X, y, threshold=30)


    # Seleccion de variables mas importantes segun 1, 2 y 3
    d = {}
    for col in X_train.columns:
        n_sel = 0

        if col in l_selected_features_1:
           n_sel += 1

        if col in l_selected_features_2:
            n_sel += 1

        if col in l_selected_features_3:
            n_sel += 1

        d[col] = n_sel

    print(d)


def prueba():
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_cleaned.xlsx')

    feature_selection(df, 'equipo_ganador')

    '''
    dt = DecisionTreeClassifier(max_depth=7)
    dt.fit(X_train, y_train)

    graficar_importancia_atrib(dt, X_train)
    '''

prueba()