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

# TRATAMIENTO DE NAN VALUES
def fill_nan_values(df, type):
    columnas_con_nan = df.columns[df.isna().any()].tolist()

    # Crear una copia del dataframe original
    df_filled = df.copy()

    # OPCION 1: Llenar los valores faltantes con el valor más frecuente en cada columna
    if type == "mode":
        for col in columnas_con_nan:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

    # OPCION 2: Llenar los valores faltantes con ML
    elif type == "ml":

        # Iterar sobre las columnas con valores faltantes
        for col in columnas_con_nan:

            # Dividir el dataframe en conjunto de entrenamiento y prueba
            X_train = df_filled.loc[df[col].notnull()].drop(columns=columnas_con_nan)
            y_train = df_filled.loc[df[col].notnull(), col]
            X_test = df_filled.loc[df[col].isnull()].drop(columns=columnas_con_nan)

            # Crear un modelo RandomForestRegressor
            model = RandomForestRegressor()

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Predecir los valores faltantes
            predicted_values = model.predict(X_test)

            # Rellenar los valores faltantes en el dataframe
            df_filled.loc[df[col].isnull(), col] = predicted_values

    # Imprimir el dataframe después de la imputación
    return df_filled

def eliminar_filas_nan(df, umbral):
    """
    Elimina las filas de un DataFrame que contienen un porcentaje alto de valores NaN.

    Args:
        df (pandas.DataFrame): DataFrame de entrada.
        umbral (float): Umbral en forma de porcentaje (0-100) para determinar el límite de NaN en una fila.

    Returns:
        pandas.DataFrame: DataFrame resultante después de eliminar las filas con valores NaN.

    """
    # Elimino filas segun umbral
    porcentaje_nan = df.isnull().mean(axis=1)  # Calcula el porcentaje de valores NaN en cada fila
    filas_a_eliminar = porcentaje_nan[porcentaje_nan > umbral].index  # Obtiene las filas que superan el umbral
    print(f"Se eliminó el {len(filas_a_eliminar)/len(df)*100:.0f}% de filas, quedan {len(df) - len(filas_a_eliminar)} filas.")

    # Elimino filas segun umbral
    df_filtrado = df.drop(filas_a_eliminar)  # Elimina las filas con valores NaN
    return df_filtrado

def eliminar_columnas_nan(df, umbral):
    # Calcula la proporción de NaN en cada columna
    prop_nan = df.isna().mean()

    # Identifica las columnas con una proporción de NaN mayor al umbral
    columnas_eliminar = prop_nan[prop_nan > umbral].index

    # Elimina las columnas identificadas del DataFrame
    df_sin_nan = df.drop(columnas_eliminar, axis=1)
    print(f"Columnas a eliminar por mas del {umbral*100:.0f}% de nan: {list(columnas_eliminar)}")
    return df_sin_nan

# CORRELACION
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

# SELECCION DE VARIABLES IMPORTANTES
def feature_selection(df, var_resp, percentil):  # Esto tene que ser el main, es decir, lo que va ir en select_data en main (como representacion de la seleccion de var mas imp)

    fs = FeatureSelection(df, var_resp)

    # Definicion de varibles
    df_importance = pd.DataFrame(columns=['analisis_uni', 'random', 'pca'], index=df.drop(['odds_loc', 'odds_emp', 'odds_vis', var_resp], axis=1).columns)  # que cada analisis devuelva las features y su importancia y guardarlo en un Dataframe...
    rf = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42)

    # Obtengo importancia de cada variable segun distintos analisis
    d1 = fs.modelos_estadisticos()  # Opción 1: Análisis univariable con tests estadísticos
    d2 = fs.machine_learning_model(rf,  best_params=True, k=10) # Opcion 2: Random Forest
    d3 = fs.pca()  # Opcion 3: pca

    # Guardo resultados en DataFrame
    df_importance['analisis_uni'] = df_importance.index.map(d1)
    df_importance['random'] = df_importance.index.map(d2)
    df_importance['pca'] = df_importance.index.map(d3)

    # Selecciono variables mas importantes
    l_selected_features = fs.select_best_features_from_all_models(df_importance, percentil)
    print(f"Columnas mas importantes: {l_selected_features}")

    return l_selected_features # pd.concat([df.loc[:, l_selected_features], df.loc[:, ['odds_loc', 'odds_emp', 'odds_vis', var_resp]]], axis=1)

class FeatureSelection():

    def __init__(self, df, var_resp):
        self.df = df
        self.var_resp = var_resp

        # Podria guardar los votos aqui dentro... y crear df_importance...

    def modelos_estadisticos(self):

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

    def pca(self):  # Podria llamarlo desde ml model?
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

    def select_best_features_from_all_models(self, df_importance, percentil):  # Sistema de ponderacion con peso minimo definido por “Porcentaje de peso máximo"

        # Normalizar cada columna del DataFrame --> para poder sumar las importancias de cada metodo
        df_normalized = pd.DataFrame(scale(df_importance), columns=df_importance.columns, index=df_importance.index)

        # Calcular la suma de columnas para cada fila
        df_normalized['suma_de_imp'] = df_normalized.sum(axis=1)
        # df_normalized.to_excel('./df_normalized.xlsx')

        # Re-escalo la variable "suma_de_imp" para que sea de 0 a 1 y facilitar la seleccion de variables
        max_value = df_normalized['suma_de_imp'].max()
        min_value = df_normalized['suma_de_imp'].min()
        def normalize_value(value):
            return (value - min_value) / (max_value - min_value)
        df_normalized['suma_de_imp_norm'] = df_normalized['suma_de_imp'].apply(normalize_value)

        # Calculo peso minimo de una variable para ser considerada como importante
        peso_maximo = df_normalized['suma_de_imp_norm'].max()
        peso_minimo = peso_maximo * percentil
        # print(f"Peso maximo: {peso_maximo} \nPeso minimo necesario: {peso_minimo}")
        # print(df_normalized)

        # Seleccionar los índices donde el valor de la columna "suma_de_imp" es mayor al umbral
        l_selected_features = list(df_normalized.loc[df_normalized['suma_de_imp_norm'] > peso_minimo].index)
        return l_selected_features

def prueba():
    warnings.filterwarnings('ignore')

    from data_preparation import format_data

    var_resp = 'equipo_ganador'
    pais = 'argentina'
    thr_corr = 0.7  # Correlacion minima entre dos variables para indicar una alta correlacion [0-1] (siendo 1 correlacion maxima y 0 sin correlacion)
    perc_fs = 0.3  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)

    # Levanto dataset de prueba
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_constructed.xlsx')

    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

    # TRATAMIENTO DE NAN VALUES
    # Cuand conviene eliminar NaN values? Antes de el analisis de correelacion, antes de construir datos, antes de cuando?
    # 1º elimino registros con muchos nan --> puesto que quiero preservar variables antes que registros
    df = eliminar_filas_nan(df, umbral=0.5)

    prop_nan = df.isna().mean()
    print(prop_nan)

    # 2º elimino columnas con mucho NaN
    df = eliminar_columnas_nan(df, umbral=0.2)

    prop_nan = df.isna().mean()
    print(prop_nan)
    print(df.shape)

    df_2 = df.copy()

    # 3º Vuelvo a eliminar filas con NaN values puesto que al modelo no le pueden entrar NaN values. Alternativamente, podria rellenar los nans...
    n_filas_antes_drop = df.shape[0]
    df = df.dropna()
    n_filas_dsp_drop = df.shape[0]
    print(f"Se eliminó el {100 - (n_filas_dsp_drop / n_filas_antes_drop * 100):.0f}% de filas, quedan {n_filas_dsp_drop} filas.")

    df_3 = eliminar_filas_nan(df_2, umbral=0)  # 3º Vuelvo a eliminar filas con NaN values puesto que al modelo no le pueden entrar NaN values. Alternativamente, podria rellenar los nans...

    df.to_excel('/Users/nachomondino/Desktop/df_selected_dsp_drop_na.xlsx', index=False)

    # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
    df = format_data.convert_columns_to_int(df)

    # Selecciono las variables con menor correlacion  # No usaré la matriz de correlacion puesto que haré feature selection??
    l_columnas_a_eliminar = eliminar_columnas_correlacionadas(df, var_resp, thr_corr)
    df = df.drop(l_columnas_a_eliminar, axis=1)

    # Selecciono las variables mas importantes (feature selection)
    l_selected_features = feature_selection(df, var_resp, percentil=perc_fs)
    df = df.loc[:, l_selected_features + ['odds_loc', 'odds_emp', 'odds_vis', var_resp]]

    df.to_excel('/Users/nachomondino/Desktop/df_selected_prueba.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()