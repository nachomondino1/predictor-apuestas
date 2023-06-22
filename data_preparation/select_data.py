# Importo librerias
import pandas as pd
import warnings
from modeling.build_model import select_best_hiperparameters
from sklearn.feature_selection import SelectKBest, f_classif, chi2  # modelos estadisticos
from sklearn.feature_selection import f_regression  # via
from sklearn.feature_selection import RFE  # rfe
from sklearn.linear_model import LinearRegression  # rfe
from sklearn.linear_model import Lasso  # Lasso
from sklearn.ensemble import RandomForestClassifier  # Random forest
from sklearn.preprocessing import scale
import plotly.graph_objects as go


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

    print(f"Se eliminaron {len(columnas_eliminar)} columnas por tener una correlacion mayor a  thr_corr={umbral*100:.0f}%: {columnas_eliminar}")
    return list(columnas_eliminar)

# SELECCION DE VARIABLES IMPORTANTES
class FeatureSelection():

    def modelos_estadisticos(self, X, y, graf=False):
        """
        Calculo de importancia de cada variable segun los modelos estadisticos.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Definicion de variables
        l_features, l_scores = [], []

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

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'mod_estadisticos': l_scores}, index=l_features)
        # print("Resultados estadisticos: \n", df_importance)

        # Grafico variables y su importancia
        if graf:
            self.graficar_importancia_atrib(X=df_importance['mod_estadisticos'], y=df_importance.index)

        return df_importance

    def random_forest(self, X, y, k=3, graf=False):
        """
        Calculo de importancia de cada variable segun modelo de random forest.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Verificar si se deben buscar los mejores hiperparámetros
        model = select_best_hiperparameters(RandomForestClassifier(), X, y, k=k)  # Tarda puesto que X no es del tamaño de X_val sino que de X_train

        # Entrenar el modelo final con todos los datos de entrenamiento
        model.fit(X, y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'random_forest': model.feature_importances_}, index=X.columns)

        # Grafico variables y su importancia
        if graf:
            self.graficar_importancia_atrib(X=df_importance['random_forest'], y=df_importance.index)

        return df_importance

    def via(self, X, y, graf=False):
        """
        Calculo de importancia de cada variable segun via.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Entreno modelo
        scores, _ = f_regression(X, y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'via': scores}, index=X.columns)
        # print("Resultados via: \n", df_importance)

        if graf:
            self.graficar_importancia_atrib(X=df_importance['via'], y=df_importance.index)

        return df_importance

    def rfe(self, X, y, graf=False):
        """
        Calculo de importancia de cada variable segun rfe.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Definicion de variables
        n_features = 1  # Número deseado de características seleccionadas hasta que se eliminan las menos relevantes
        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=n_features)

        # Entreno modelo
        X_selected = rfe.fit_transform(X, y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'rfe': rfe.ranking_}, index=X.columns)

        # Convierto ranking en importancia (a mayor ranking, menor importancia)
        df_importance['rfe'] = df_importance['rfe'].apply(lambda x: len(X.columns) - x + 1)
        # print("Resultados rfe: \n", df_importance)

        if graf:
            self.graficar_importancia_atrib(X=df_importance['rfe'], y=df_importance.index)

        return df_importance

    def lasso_selection(self, X, y, graf=False):
        """
        Calculo de importancia de cada variable segun lasso.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Entreno modelo
        lasso = Lasso(alpha=0.01)  # con 0.05: 11 variables son cero # 0.15: 14 var # con 0.01: 4 var
        lasso.fit(X, y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'lasso': lasso.coef_}, index=X.columns)

        # Convierto coeficiente en importancia (a mayor coef en valor abs, mas importancia)
        df_importance['lasso'] = df_importance['lasso'].apply(lambda x: abs(x))  #x es coef
        # print("Resultados lasso: \n", df_importance)

        if graf:
            self.graficar_importancia_atrib(X=df_importance['lasso'], y=df_importance.index)

        return df_importance

    def graficar_importancia_atrib(self, X, y):
        """
        Grafica la importancia de cada variable.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        """
        # Crear figura
        fig = go.Figure()

        # Agregar barras al gráfico
        fig.add_trace(go.Bar(
            x=X,
            y=y,
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

    def sum_and_normalize_importances(self, df_importance):
        """
        Normaliza las importancias de cada variable segun cada metodo, las suma y normaliza dicha suma para tener una
        importancia final para cada variable de 0 a 1.

        :param df_importance: Dataframe con importancia de cada variable segun cada metodo. (DataFrame)
        :return: Dataframe con importancias normalizadas. (DataFrame)
        """
        # Normalizar cada columna del DataFrame --> para poder sumar las importancias de cada metodo
        df_normalized = pd.DataFrame(scale(df_importance), columns=df_importance.columns, index=df_importance.index)

        # Calcular la suma de columnas para cada fila
        df_normalized['suma_de_imp'] = df_normalized.sum(axis=1)

        # Re-escalo la variable "suma_de_imp" para que sea de 0 a 1 y facilitar la seleccion de variables
        df_normalized['suma_de_imp_norm'] = (df_normalized['suma_de_imp'] - df_normalized['suma_de_imp'].min()) / (df_normalized['suma_de_imp'].max() - df_normalized['suma_de_imp'].min())
        return df_normalized

def select_best_features(df, var_resp, thr_fs, graf=True):
    """
    Selecciona las variables mas importantes para un Dataframe.

    :param df: Dataframe. (Dataframe)
    :param var_resp: Nombre de la variable respuesta. (string)
    :param thr_fs: Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la
    variable mas importante y 0 la menos). (float)
    :param graf: Boolean. True para graficar variables y sus importancias. De lo contrario, False.
    :return: Lista de variables mas importantes. (list)
    """
    # Definicion de variables
    fs = FeatureSelection()
    graficar = False

    # Elimino odds
    df = df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)

    # Elimino NaN values puesto que no puedo tener NaN en modelos de ml
    df = df.dropna()  # Es dificil que queden pocos registros porque borro filas y col con muchos nan antes
    # print(f"Largo del dataframe antes de fs: {df.shape}")

    # Separo en X e y
    X = df.drop(var_resp, axis=1)
    y = df[var_resp]
    df_importance = pd.DataFrame(index=X.columns)

    # Detemino importancia de cada variable para cada modelo
    df_importance = df_importance.merge(fs.modelos_estadisticos(X, y, graf=graficar), left_index=True, right_index=True)
    df_importance = df_importance.merge(fs.via(X, y, graf=graficar), left_index=True, right_index=True)
    # df_importance = df_importance.merge(fs.rfe(X, y, graf=graficar), left_index=True, right_index=True) # No me gustan sus resultados
    # df_importance = df_importance.merge(fs.lasso_selection(X, y, graf=graficar), left_index=True, right_index=True)  # No me gustan sus resultados
    df_importance = df_importance.merge(fs.random_forest(X, y, graf=graficar), left_index=True, right_index=True)
    # df_importance.to_excel('/Users/nachomondino/Desktop/df_importance_prueba.xlsx')

    # Normalizo importancias para poder sumarlas
    df_normalized = fs.sum_and_normalize_importances(df_importance)
    # df_normalized.to_excel('/Users/nachomondino/Desktop/df_normalized_prueba.xlsx')

    # Selecciono las variables mas importantes segun umbral e imprimo resultados
    l_selected_features = df_normalized.loc[df_normalized['suma_de_imp_norm'] > df_normalized['suma_de_imp_norm'].max() * thr_fs].index.tolist()
    l_not_important_features = [col for col in X.columns if col not in l_selected_features]
    print(f"Se eliminaron {len(l_not_important_features)} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%: {l_not_important_features}")
    print(f"Las {len(l_selected_features)} columnas mas importantes por peso mayor a thr_fs={thr_fs * 100:.0f}%: {l_selected_features}")

    # Grafico importancias teniendo en cuenta todos los modelos
    if graf:
        fs.graficar_importancia_atrib(X=df_normalized['suma_de_imp_norm'], y=df_normalized.index)

    return l_selected_features

def prueba():
    warnings.filterwarnings('ignore')

    from data_preparation import format_data, clean_data

    var_resp = 'equipo_ganador'
    pais = 'argentina'
    thr_corr = 0.6  # Correlacion minima entre dos variables para indicar una alta correlacion [0-1] (siendo 1 correlacion maxima y 0 sin correlacion)
    thr_fs = 0.3  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)
    thr_nan_col = 0.2
    export = False

    # Levanto dataset de prueba
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_constructed.xlsx')

    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

    # Elimino filas y columnas con alto porcentaje de NaN values
    df = clean_data.eliminar_filas_nan(df, umbral=0.5)  # 1º elimino registros con muchos nan --> puesto que quiero preservar variables antes que registros
    if thr_nan_col is not None:
        df = clean_data.eliminar_columnas_nan(df, umbral=thr_nan_col)  # 2º elimino columnas con mucho NaN

    # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
    df, df_etiquetas = format_data.convert_columns_to_int(df)
    # df_etiquetas.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_etiquetas.xlsx')

    # Elimino variables altamente correlacionadas
    l_columnas_a_eliminar = eliminar_columnas_correlacionadas(df, var_resp, thr_corr)
    df = df.drop(l_columnas_a_eliminar, axis=1)

    # Selecciono las variables mas importantes (feature selection)
    l_selected_features = select_best_features(df, var_resp, thr_fs=thr_fs, graf=export)
    columns_to_select = l_selected_features + ['odds_loc', 'odds_emp', 'odds_vis',var_resp]
    df = df.filter(columns_to_select)
    df.to_excel('/Users/nachomondino/Desktop/df_selected_prueba.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()