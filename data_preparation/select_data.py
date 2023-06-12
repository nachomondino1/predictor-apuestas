# Importo librerias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from modeling.build_model import train_model
import warnings
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def eliminar_columnas_correlacionadas(df, var_resp, umbral):

    columnas_eliminar = set()  # Conjunto para almacenar las columnas a eliminar

    # Calculo matriz de correlacion
    df_correlacion = df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1).corr().abs()

    # Separo matriz de correlacion de las variables predictoras y de ellas con la variable objetivo
    df_corr = df_correlacion.drop(var_resp, axis=1).drop(var_resp, axis=0)
    df_corr_var_obj = df_correlacion[var_resp].drop(var_resp, axis=0)

    # Recorrer las columnas de la matriz de correlación
    for i in range(len(df_corr.columns)):
        for j in range(i+1, len(df_corr.columns)):

            # Si hay alta correlacion
            if df_corr.iloc[i, j] > umbral:

                # Busco correlacion de cada columna con variable objetivo
                col1, col2 = df_corr.columns[i], df_corr.columns[j]
                # print(col1, col2)
                corr_col1, corr_col2 = df_corr_var_obj.loc[col1], df_corr_var_obj.loc[col2]
                # print(corr_col1, corr_col2)

                # Elimino aquella columna con menor correlacion con la variable objetivo
                if corr_col2 > corr_col1:
                    columnas_eliminar.add(col1)
                    # print(f"Variable a eliminar: {col1}")
                else:
                    columnas_eliminar.add(col2)
                    # print(f"Variable a eliminar: {col2}")

    print(f"Columnas a eliminar por correlacion: {columnas_eliminar}")
    return list(columnas_eliminar)

def feature_selection(df, var_resp, percentil):

    fs = FeatureSelection(df, var_resp)

    # Definicion de varibles
    df_importance = pd.DataFrame(columns=['analisis_uni', 'random', 'pca'], index=df.drop(['odds_loc', 'odds_emp', 'odds_vis', var_resp], axis=1).columns)  # que cada analisis devuelva las features y su importancia y guardarlo en un Dataframe...
    rf = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42)

    # Obtengo importancia de cada variable segun distintos analisis
    d1 = fs.analisis_univariable()  # Opción 1: Análisis univariable con tests estadísticos
    d3 = fs.machine_learning_model(rf,  best_params=True, k=10) # Opcion 3: Random Forest # d2 = fs.machine_learning_model(dt, best_params=True, k=10)  # Opcion 2: Arbol
    d4 = fs.pca()  # Opcion 4: pca

    # Guardo resultados en DataFrame
    df_importance['analisis_uni'] = df_importance.index.map(d1)
    df_importance['random'] = df_importance.index.map(d3)  # df_importance['arbol'] = df_importance.index.map(d2)  # El arbol es un subconjunto de random_forest y da muy similar
    df_importance['pca'] = df_importance.index.map(d4)

    # Selecciono variables mas importantes
    l_selected_features = fs.select_best_features_from_all_models(df_importance, percentil)
    print(f"Columnas mas importantes: {l_selected_features}")

    return l_selected_features # pd.concat([df.loc[:, l_selected_features], df.loc[:, ['odds_loc', 'odds_emp', 'odds_vis', var_resp]]], axis=1)

class FeatureSelection():

    def __init__(self, df, var_resp):
        self.df = df
        self.var_resp = var_resp

        # Podria guardar los votos aqui dentro... y crear df_importance...

    def analisis_univariable(self):

        # Definicion de variables
        d = {}
        l_features, l_scores = [], []

        df = self.df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
        X, y = df.drop([self.var_resp], axis=1), df[self.var_resp]

        # Selecciono variables numericas y categoricas
        numeric_vars = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_vars = X.select_dtypes(include='object').columns.tolist()

        # Variables predictoras numéricas
        if len(numeric_vars) > 0:
            numeric_X = X[numeric_vars].clip(lower=0)  # Asegurar que los valores sean no negativos
            numeric_selector = SelectKBest(score_func=f_classif, k='all')  # Utiliza ANOVA o f-score, selecciona las 3 mejores características
            numeric_selector.fit_transform(numeric_X, y)  # numeric_X_selected
            numeric_selected_features = [numeric_vars[i] for i in range(len(numeric_vars)) if numeric_selector.get_support()[i]]
            numeric_scores = numeric_selector.scores_

            l_features += numeric_selected_features
            l_scores += list(numeric_scores)

        # Variables predictoras categóricas
        if len(categorical_vars) > 0:
            categorical_X = X[categorical_vars]
            categorical_selector = SelectKBest(score_func=chi2, k='all')  # Utiliza chi-cuadrado, selecciona las 3 mejores características
            categorical_selector.fit_transform(categorical_X, y)  # categorical_X_selected
            categorical_selected_features = [categorical_vars[i] for i in range(len(categorical_vars)) if categorical_selector.get_support()[i]]
            categorical_scores = categorical_selector.scores_

            l_features += categorical_selected_features
            l_scores += list(categorical_scores)

        # Guardo resultados
        for feature, score in zip(l_features, l_scores):
            d[feature] = score

        # Grafico variables y su importancia
        self.graficar_importancia_atrib(l_features, l_scores)
        return d

    def machine_learning_model(self, modelo, best_params=True, k=10):  # Lo dejo en funcion? Si ya llama a train_model... --> SOLO USARE RANDOM ENCIMA...

        # Definicion de variables
        d = {}

        # Entreno modelo
        model, accuracy, roi = train_model(self.df, self.var_resp, modelo, best_params=best_params, k=k)

        # Defino variables y su importancia
        l_features = self.df.drop(['odds_loc', 'odds_emp', 'odds_vis', self.var_resp], axis=1).columns
        l_importance = model.feature_importances_

        # Guardo resultados
        for feature, importance in zip(l_features, l_importance):
            d[feature] = importance

        # Grafico variables y su importancia
        self.graficar_importancia_atrib(l_features, l_importance)
        return d

    def pca(self):
        # Separar las variables independientes (X) y la variable objetivo (y)
        df = self.df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
        X, y = df.drop([self.var_resp], axis=1), df[self.var_resp]

        # Estandarizar las variables independientes
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Aplicar PCA
        pca = PCA()
        pca.fit_transform(X_scaled)

        # Obtener la importancia de las variables a través de los componentes principales
        importance = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i) for i in range(1, pca.n_components_ + 1)], index=X.columns)

        # Convertir el DataFrame de importancia a un diccionario
        importance_dict = importance.to_dict('index')
        importance_dict = {variable: list(importance_dict[variable].values())[0] for variable in importance_dict}

        # Grafico variables y su importancia
        l_features = list(importance_dict.keys())
        l_scores = list(importance_dict.values())
        self.graficar_importancia_atrib(l_features, l_scores)
        return importance_dict

    def graficar_importancia_atrib(self, l_features, l_importance):

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

    def select_best_features_from_all_models(self, df_importance, percentil):

        # Normalizar cada columna del DataFrame
        df_normalized = pd.DataFrame(scale(df_importance), columns=df_importance.columns, index=df_importance.index)

        # Calcular la suma de columnas para cada fila
        df_normalized['suma_de_imp'] = df_normalized.sum(axis=1)
        # df_normalized.to_excel('./df_normalized.xlsx')

        # Calculo percentil
        valor_percentil = df_normalized['suma_de_imp'].quantile(percentil)

        # Seleccionar los índices donde el valor de la columna "suma_de_imp" es mayor al umbral
        l_selected_features = list(df_normalized.loc[df_normalized['suma_de_imp'] > valor_percentil].index)
        return l_selected_features

def prueba():
    warnings.filterwarnings('ignore')

    from data_preparation import format_data

    var_resp = 'equipo_ganador'
    pais = 'argentina'
    thr_corr = 0.7
    perc_fs = 0.7

    # Levanto dataset de prueba
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_constructed.xlsx')

    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

    # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
    df = format_data.convert_columns_to_int(df)

    # Selecciono las variables con menor correlacion  # No usaré la matriz de correlacion puesto que haré feature selection??
    l_columnas_a_eliminar = eliminar_columnas_correlacionadas(df, var_resp, thr_corr)
    df = df.drop(l_columnas_a_eliminar, axis=1)

    # Selecciono las variables mas importantes (feature selection)
    l_selected_features = feature_selection(df.dropna(), var_resp, percentil=perc_fs)  # Le paso el df sin NaN values para evitar ""ValueError: Input X contains NaN.".  Pero no hago fillna() puesto que introduce sesgo
    df = df.loc[:, l_selected_features + ['odds_loc', 'odds_emp', 'odds_vis', var_resp]]

    df.to_excel('/Users/nachomondino/Desktop/df_selected_prueba.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()