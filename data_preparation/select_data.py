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

    print(f"Columnas eliminadas por correlacion mayor a thr_corr={umbral*100:.0f}%: {columnas_eliminar}")
    return list(columnas_eliminar)

# SELECCION DE VARIABLES IMPORTANTES
class FeatureSelection():

    def __init__(self, df, var_resp):
        self.var_resp = var_resp
        self.X = df.drop(['odds_loc', 'odds_emp', 'odds_vis', self.var_resp], axis=1)
        self.y = df[self.var_resp]

    def modelos_estadisticos(self, graf=False):

        # Definicion de variables
        l_features, l_scores = [], []

        # Selecciono variables numericas y categoricas
        numeric_vars = self.X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_vars = self.X.select_dtypes(include='object').columns.tolist()

        # Variables predictoras numéricas
        if len(numeric_vars) > 0:
            numeric_X = self.X[numeric_vars].clip(lower=0)  # Asegurar que los valores sean no negativos
            numeric_selector = SelectKBest(score_func=f_classif, k='all')  # Utiliza ANOVA o f-score, selecciona las 3 mejores características
            numeric_selector.fit_transform(numeric_X, self.y)  # numeric_X_selected
            numeric_selected_features = [numeric_vars[i] for i in range(len(numeric_vars)) if numeric_selector.get_support()[i]]
            numeric_scores = numeric_selector.scores_

            l_features += numeric_selected_features
            l_scores += list(numeric_scores)

        # Variables predictoras categóricas
        if len(categorical_vars) > 0:
            categorical_X = self.X[categorical_vars]
            categorical_selector = SelectKBest(score_func=chi2, k='all')  # Utiliza chi-cuadrado, selecciona las 3 mejores características
            categorical_selector.fit_transform(categorical_X, self.y)  # categorical_X_selected
            categorical_selected_features = [categorical_vars[i] for i in range(len(categorical_vars)) if categorical_selector.get_support()[i]]
            categorical_scores = categorical_selector.scores_

            l_features += categorical_selected_features
            l_scores += list(categorical_scores)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'importance': l_scores}, index=l_features)
        # print("Resultados estadisticos: \n", df_importance)

        # Grafico variables y su importancia
        if graf:
            self.graficar_importancia_atrib(x=df_importance['importance'], y=df_importance.index)

        return df_importance

    def random_forest(self, k=3, graf=False):  # Lo dejo en funcion? Si ya llama a train_model... --> SOLO USARE RANDOM ENCIMA...

        # Verificar si se deben buscar los mejores hiperparámetros
        model, best_params = select_best_hiperparameters(RandomForestClassifier(), self.X, self.y, k=k)

        # Entrenar el modelo final con todos los datos de entrenamiento
        model.fit(self.X, self.y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'importance': model.feature_importances_}, index=self.X.columns)

        # Grafico variables y su importancia
        if graf:
            self.graficar_importancia_atrib(x=df_importance['importance'], y=df_importance.index)

        return df_importance

    def via(self, graf=False):

        # Entreno modelo
        scores, _ = f_regression(self.X, self.y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'importance': scores}, index=self.X.columns)
        # print("Resultados via: \n", df_importance)

        if graf:
            self.graficar_importancia_atrib(x=df_importance['importance'], y=df_importance.index)

        return df_importance

    def rfe(self, graf=False):

        # Definicion de variables
        n_features = 1  # Número deseado de características seleccionadas hasta que se eliminan las menos relevantes
        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=n_features)

        # Entreno modelo
        X_selected = rfe.fit_transform(self.X, self.y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'rank': rfe.ranking_}, index=self.X.columns)

        # Convierto ranking en importancia (a mayor ranking, menor importancia)
        df_importance['importance'] = df_importance['rank'].apply(lambda x: len(self.X.columns) - x + 1)
        # print("Resultados rfe: \n", df_importance)

        if graf:
            self.graficar_importancia_atrib(x=df_importance['importance'], y=df_importance.index)

        return df_importance

    def lasso_selection(self, graf=False):

        # Entreno modelo
        lasso = Lasso(alpha=0.01)  # con 0.05: 11 variables son cero # 0.15: 14 var # con 0.01: 4 var
        lasso.fit(self.X, self.y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'coeficiente': lasso.coef_}, index=self.X.columns)

        # Convierto coeficiente en importancia (a mayor coef en valor abs, mas importancia)
        df_importance['importance'] = df_importance['coeficiente'].apply(lambda x: abs(x))  #x es coef
        # print("Resultados lasso: \n", df_importance)

        if graf:
            self.graficar_importancia_atrib(x=df_importance['importance'], y=df_importance.index)

        return df_importance

    def graficar_importancia_atrib(self, x, y):
        # Crear figura
        fig = go.Figure()

        # Agregar barras al gráfico
        fig.add_trace(go.Bar(
            x=x,
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

    def normalize_importances(self, df_importance):

        # Normalizar cada columna del DataFrame --> para poder sumar las importancias de cada metodo
        df_normalized = pd.DataFrame(scale(df_importance), columns=df_importance.columns, index=df_importance.index)

        # Calcular la suma de columnas para cada fila
        df_normalized['suma_de_imp'] = df_normalized.sum(axis=1)

        # Re-escalo la variable "suma_de_imp" para que sea de 0 a 1 y facilitar la seleccion de variables
        df_normalized['suma_de_imp_norm'] = (df_normalized['suma_de_imp'] - df_normalized['suma_de_imp'].min()) / (df_normalized['suma_de_imp'].max() - df_normalized['suma_de_imp'].min())
        return df_normalized

    def select_best_features(self, umbral, graf=True):

        df_importance = pd.DataFrame(index=self.X.columns)
        graficar = False

        # Detemino importancia de cada variable para cada modelo
        df_importance['mod_estadisticos'] = self.modelos_estadisticos(graf=graficar)['importance']
        df_importance['via'] = self.via(graf=graficar)['importance']
        df_importance['rfe'] = self.rfe(graf=graficar)['importance']
        df_importance['lasso'] = self.lasso_selection(graf=graficar)['importance']
        df_importance['random_forest'] = self.random_forest(graf=graficar)['importance']
        # df_importance.to_excel('/Users/nachomondino/Desktop/df_importance_prueba.xlsx')

        # Normalizo importancias para poder sumarlas
        df_normalized = self.normalize_importances(df_importance)
        # df_normalized.to_excel('/Users/nachomondino/Desktop/df_normalized_prueba.xlsx')

        # Selecciono las variables mas importantes segun umbral
        l_selected_features = df_normalized.loc[df_normalized['suma_de_imp_norm'] > df_normalized['suma_de_imp_norm'].max() * umbral].index.tolist()

        # Grafico importancias teniendo en cuenta todos los modelos
        if graf:
            self.graficar_importancia_atrib(x=df_normalized['suma_de_imp_norm'], y=df_normalized.index)

        print(f"Columnas mas importantes por peso mayor a thr_fs={umbral*100:.0f}%: {l_selected_features}")
        return l_selected_features

def prueba():
    warnings.filterwarnings('ignore')

    from data_preparation import format_data, clean_data

    var_resp = 'equipo_ganador'
    pais = 'argentina'
    thr_corr = 0.6  # Correlacion minima entre dos variables para indicar una alta correlacion [0-1] (siendo 1 correlacion maxima y 0 sin correlacion)
    umbral_fs = 0.3  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)

    # Levanto dataset de prueba
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_constructed.xlsx')

    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

    # TRATAMIENTO DE NAN VALUES
    # 1º elimino registros con muchos nan --> puesto que quiero preservar variables antes que registros
    df = clean_data.eliminar_filas_nan(df, umbral=0.5)
    prop_nan = df.isna().mean()
    print(prop_nan)

    # 2º elimino columnas con mucho NaN
    df = clean_data.eliminar_columnas_nan(df, umbral=0.2)
    prop_nan = df.isna().mean()
    print(prop_nan)
    print(df.shape)

    # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
    df, df_etiquetas = format_data.convert_columns_to_int(df)

    # Selecciono las variables con menor correlacion  # No usaré la matriz de correlacion puesto que haré feature selection??
    l_columnas_a_eliminar = eliminar_columnas_correlacionadas(df, var_resp, thr_corr)
    df = df.drop(l_columnas_a_eliminar, axis=1)

    # Selecciono las variables mas importantes (feature selection)
    fs = FeatureSelection(df.dropna(), var_resp)
    l_selected_features = fs.select_best_features(umbral=umbral_fs)
    columns_to_select = l_selected_features + ['odds_loc', 'odds_emp', 'odds_vis', var_resp]
    df = df.filter(columns_to_select)

    # 3º Vuelvo a eliminar filas con NaN values puesto que al modelo no le pueden entrar NaN values. Alternativamente, podria rellenar los nans...
    df = clean_data.eliminar_filas_nan(df, umbral=0)

    df.to_excel('/Users/nachomondino/Desktop/df_selected_prueba.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()