# Importo librerias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from modeling.build_model import train_model
import warnings
from sklearn.preprocessing import scale

def analisis_univariable(df, var_resp):

    # Definicion de variables
    d = {}
    l_features, l_scores = [], []
    df = df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
    X, y = df.drop([var_resp], axis=1), df[var_resp]

    # Selecciono variables numericas y categoricas
    numeric_vars = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_vars = X.select_dtypes(include='object').columns.tolist()

    # Variables predictoras numéricas
    if len(numeric_vars) > 0:
        numeric_X = X[numeric_vars].clip(lower=0)  # Asegurar que los valores sean no negativos
        numeric_selector = SelectKBest(score_func=f_classif, k='all')  # Utiliza ANOVA o f-score, selecciona las 3 mejores características
        numeric_X_selected = numeric_selector.fit_transform(numeric_X, y)
        numeric_selected_features = [numeric_vars[i] for i in range(len(numeric_vars)) if numeric_selector.get_support()[i]]
        numeric_scores = numeric_selector.scores_

        l_features += numeric_selected_features
        l_scores += list(numeric_scores)

    # Variables predictoras categóricas
    if len(categorical_vars) > 0:
        categorical_X = X[categorical_vars]
        categorical_selector = SelectKBest(score_func=chi2, k='all')  # Utiliza chi-cuadrado, selecciona las 3 mejores características
        categorical_X_selected = categorical_selector.fit_transform(categorical_X, y)
        categorical_selected_features = [categorical_vars[i] for i in range(len(categorical_vars)) if categorical_selector.get_support()[i]]
        categorical_scores = categorical_selector.scores_

        l_features += categorical_selected_features
        l_scores += list(categorical_scores)

    # Guardo resultados
    for feature, score in zip(l_features, l_scores):
        d[feature] = score

    # Grafico variables y su importancia
    graficar_importancia_atrib(l_features, l_scores)
    return d

def machine_learning_model(df, var_resp, modelo, best_params=True, k=10):

    # Definicion de variables
    d = {}

    # Entreno modelo
    model, accuracy, roi = train_model(df, var_resp, modelo, best_params=best_params, k=k)

    # Defino variables y su importancia
    l_features = df.drop(['odds_loc', 'odds_emp', 'odds_vis', var_resp], axis=1).columns
    l_importance = model.feature_importances_

    # Guardo resultados
    for feature, importance in zip(l_features, l_importance):
        d[feature] = importance

    # Grafico variables y su importancia
    graficar_importancia_atrib(l_features, l_importance)
    return d

def graficar_importancia_atrib(l_features, l_importance):

    # Crear gráfico de barras
    # Crear figura
    fig = go.Figure()  # fig = go.Figure(data=go.Bar(x=l_features, y=l_importance, orientation='h'))

    # Agregar barras al gráfico
    fig.add_trace(go.Bar(
        x=l_importance,
        y=l_features,
        orientation='h'
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title='Importancia de las características',
        xaxis_title='Importancia',
        yaxis_title='Características',
        yaxis=dict(autorange="reversed")  # Invertir el orden de las características
    )

    # Rotar etiquetas en el eje x
    fig.update_layout(xaxis_tickangle=-45)

    # Mostrar el gráfico
    fig.show()

def select_best_features_from_all_models(df_importance, threshold):

    # Normalizar cada columna del DataFrame
    # df_normalized = pd.DataFrame(normalize(df_importance), columns=df_importance.columns)
    df_normalized = pd.DataFrame(scale(df_importance), columns=df_importance.columns, index=df_importance.index)

    # Calcular la suma de columnas para cada fila
    df_normalized['Sum'] = df_normalized.sum(axis=1)
    df_normalized.to_excel('./df_normalized.xlsx')

    # Seleccionar los índices donde el valor de la columna "Sum" es mayor al umbral
    l_selected_features = list(df_normalized.loc[df_normalized['Sum'] > threshold].index)
    return l_selected_features

def feature_selection(df, var_resp):

    # Definicion de varibles
    df_importance = pd.DataFrame(columns=['analisis_uni', 'arbol', 'random'], index=df.drop(['odds_loc', 'odds_emp', 'odds_vis', var_resp], axis=1).columns)  # que cada analisis devuelva las features y su importancia y guardarlo en un Dataframe...
    dt = DecisionTreeClassifier(max_depth=7)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

    # Obtengo importancia de cada variable segun distintos analisis
    d1 = analisis_univariable(df, var_resp)  # Opción 1: Análisis univariable con tests estadísticos
    d2 = machine_learning_model(df, var_resp, dt, best_params=True, k=10)  # Opcion 2: Arbol
    d3 = machine_learning_model(df, var_resp, rf,  best_params=True, k=10) # Opcion 3: Random Forest

    # Guardo resultados en DataFrame
    df_importance['analisis_uni'] = df_importance.index.map(d1)
    df_importance['arbol'] = df_importance.index.map(d2)
    df_importance['random'] = df_importance.index.map(d3)
    print(df_importance)

    # Selecciono variables mas importantes
    l_selected_features = select_best_features_from_all_models(df_importance, threshold=-1)
    print(l_selected_features)
    return df.loc[:, l_selected_features]

def prueba():
    warnings.filterwarnings('ignore')

    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_cleaned.xlsx')

    df = feature_selection(df, 'equipo_ganador')

prueba()