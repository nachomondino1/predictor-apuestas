# Importo librerias
import sys
sys.path.append('/Users/nachomondino/Documents/GitHub/predictor-apuestas')  # Fallaba el import de p4_modeling
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from p3_data_preparation import clean_data
from p4_modeling.build_model import select_best_hiperparameters
from sklearn.feature_selection import SelectKBest, f_classif, chi2  # modelos estadisticos
from sklearn.feature_selection import f_regression  # via
from sklearn.feature_selection import RFE  # rfe
from sklearn.linear_model import LinearRegression, LogisticRegression  # rfe
from sklearn.linear_model import Lasso  # Lasso
from sklearn.ensemble import RandomForestClassifier  # Random forest
from sklearn.preprocessing import scale
import plotly.graph_objects as go
import numpy as np


def delete_correlated_columns(df, var_resp, umbral):
    """
    Identificacion de las columnas con un correlacion alta (mayor al umbral)
    :param df:
    :param var_resp:
    :param umbral:
    :return: List. Columnas a eliminar por correlacion alta.
    """
    print('\nEliminacion de columnas correlacionadas:')
    # Definicion de variables
    columnas_eliminar = set()  # Conjunto para almacenar las columnas a eliminar

    # Calculo matriz de correlacion
    df_correlacion = df.corr().abs()
    # df_correlacion.to_excel('/Users/nachomondino/Desktop/df_correlacion.xlsx')

    # Separo matriz de correlacion de las variables predictoras y de ellas con la variable objetivo
    df_corr = df_correlacion.drop(var_resp, axis=1).drop(var_resp, axis=0)
    df_corr_y = df_correlacion[var_resp].drop(var_resp, axis=0)

    # Obtener matriz triangular superior de correlacion (pues la matriz de correlacion es una matriz simetrica respecto de la diagonal)
    df_corr_tri = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

    # Buscar columnas con alta correlacion
    columnas_correlacionadas = np.where(df_corr_tri > umbral)
    for i, j in zip(*columnas_correlacionadas):
        col1, col2 = df_corr.columns[i], df_corr.columns[j]
        # print(f"\nColumna 1: {col1} ; Columna 2: {col2} --> Correlacion: {df_corr.loc[col1, col2]*100:.0f}%")

        # Buscar correlacion de cada columna con variable objetivo
        corr_col1, corr_col2 = df_corr_y.loc[col1], df_corr_y.loc[col2]
        # print(f"Corr col 1: {corr_col1} ; Corr col 2: {corr_col2}")

        # Eliminar aquella columna con menor correlacion con la variable objetivo
        if corr_col2 > corr_col1:
            columnas_eliminar.add(col1)
        else:
            columnas_eliminar.add(col2)

    return list(columnas_eliminar)

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

    def random_forest(self, X, y, k: int = 5, graf=False):
        """
        Calculo de importancia de cada variable segun modelo de random forest.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Separo en train y val (para que select_best_hiperparameters() no tarde tanto)
        X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        # Verificar si se deben buscar los mejores hiperparámetros
        model = select_best_hiperparameters(RandomForestClassifier(), X_val, y_val, k=k, _print=True)  # Tarda puesto que X no es del tamaño de X_val sino que de X_train

        # Entrenar el modelo final con todos los datos de entrenamiento
        model.fit(X_train, y_train)

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

    def rfe(self, X, y, k: int = 5, graf=False):
        """
        Calculo de importancia de cada variable segun rfe.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Definicion de variables
        n_features = 1  # Número deseado de características seleccionadas hasta que se eliminan las menos relevantes

        # Separo en train y val
        X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        # Busco los mejores hiperparametros para el modelo
        model = select_best_hiperparameters(LogisticRegression(), X_val, y_val, k=k, _print=True)  # Tarda puesto que X no es del tamaño de X_val sino que de X_train
        rfe = RFE(estimator=model, n_features_to_select=n_features)

        # Entreno modelo
        X_selected = rfe.fit_transform(X_train, y_train)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'rfe': rfe.ranking_}, index=X.columns)
        # print("Resultados rfe: \n", df_importance)

        # Convierto ranking en importancia (a mayor ranking, menor importancia)
        df_importance['rfe'] = df_importance['rfe'].apply(lambda x: len(X.columns) - x + 1)
        # print("Resultados rfe dsp convertir: \n", df_importance)

        if graf:
            self.graficar_importancia_atrib(X=df_importance['rfe'], y=df_importance.index)

        return df_importance

    def lasso_selection(self, X, y, k:int = 5, graf=False):
        """
        Calculo de importancia de cada variable segun lasso.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :param graf: Boolean. True para graficar importancia por variable. (bool)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Separo en train y val
        X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        # Busco los mejores hiperparametros para el modelo
        model = select_best_hiperparameters(Lasso(), X_val, y_val, k=k)  # Tarda puesto que X no es del tamaño de X_val sino que de X_train

        # Entreno modelo
        model.fit(X_train, y_train)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'lasso': model.coef_}, index=X.columns)

        # Convierto coeficiente en importancia (a mayor coef en valor abs, mas importancia)
        df_importance['lasso'] = df_importance['lasso'].apply(lambda x: abs(x))  # x es coef

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

def select_best_features(df: pd.DataFrame, var_resp: str, thr_fs: float, graf: bool = True):
    """
    Selecciona las variables mas importantes para un Dataframe.

    :param df: Dataframe. (Dataframe)
    :param var_resp: Nombre de la variable respuesta. (string)
    :param thr_fs: Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la
    variable mas importante y 0 la menos). (float)
    :param graf: Boolean. True para graficar variables y sus importancias. De lo contrario, False.
    :return: Lista de variables mas importantes. (list)
    """
    print('\n Feature Selection...')
    # Definicion de variables
    fs = FeatureSelection()
    k = 10
    graficar_cada_metodo = False

    # Separo en X e y
    X, y = df.drop(var_resp, axis=1), df[var_resp]

    # Elimino NaN values puesto que no puedo tener NaN en modelos de ml
    print("\tReemplazo y remuevo NaN values (uso ML y no puede tener input NaN)... ", end="")

    ## Determino las columns con mucho NaN (mas de nan_threshold%)
    l_columns_con_poco_nan, l_columns_con_mucho_nan = clean_data.determine_columns_to_fill(X, percentil_nan=75, _print=False)

    ## Elimino registros NaN en las columns con bajo % de NaN (para poder usarlas en X_train)
    X = X.dropna(subset=l_columns_con_poco_nan)
    X = clean_data.delete_columns_nan(X, porc_nan_max=0.99)  # Elimino columnas que quedan nan tras el dropna anterior. Esto evita error al rellenar una columna vacia.
    l_columnas_to_fill = X.drop(l_columns_con_poco_nan, axis=1).columns
    
    ## Relleno filas
    X = clean_data.fill_nan_values(X, l_columnas_to_fill, fill_type='mode')  
    y = y[y.index.isin(X.index)]
    print(f"Tras eliminar y reemplazar nan values, se hara el feature selection con {X.shape[0]} filas y {X.shape[1]} columnas")

    # Detemino importancia de cada variable para cada modelo
    print("\t Calculando importancias de variables segun varios modelos...")
    df_importance = pd.DataFrame(index=X.columns)
    # df_importance = df_importance.merge(fs.modelos_estadisticos(X, y, graf=graficar_cada_metodo), left_index=True, right_index=True)  # Solo levanta dt_loc y dt_vis, el resto da 0...
    df_importance = df_importance.merge(fs.via(X, y, graf=graficar_cada_metodo), left_index=True, right_index=True)
    df_importance = df_importance.merge(fs.random_forest(X, y, k=k, graf=graficar_cada_metodo), left_index=True, right_index=True)
    df_importance = df_importance.merge(fs.rfe(X, y, k=k, graf=graficar_cada_metodo), left_index=True, right_index=True)

    # Normalizo importancias para poder sumarlas
    df_normalized = fs.sum_and_normalize_importances(df_importance)

    # Determino columnas mas importantes
    l_important_features = df_normalized.loc[df_normalized['suma_de_imp_norm'] >= df_normalized['suma_de_imp_norm'].max() * thr_fs].index.tolist()

    # Grafico importancias teniendo en cuenta todos los modelos
    if graf:
        fs.graficar_importancia_atrib(X=df_normalized['suma_de_imp_norm'], y=df_normalized.index)
    
    return l_important_features

def prueba():
    from p3_data_preparation import format_data

    # Definicion de variables
    var_resp = 'result'
    country = 'england'
    export = False
    warnings.filterwarnings('ignore')

    # Definicion de hiperparametros
    thr_corr = None  # Correlacion minima entre dos variables para indicar una alta correlacion [0-1] (siendo 1 correlacion maxima y 0 sin correlacion)
    thr_fs = None  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)
    export = False

    # Levanto dataset de prueba
    df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_constructed.xlsx', index_col=0)
    print(df.head())

    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    # n_col = len(df.columns)
    df = df.drop(['date'], axis=1)  
    # print(f"Se eliminó {n_col - len(df.columns)} de {n_col} columnas puesto que no sirven para el analisis (e.g. id_match, fecha, etc).")

    # Elimino columnas con 100% de nan values (puede que construyas y queden con todo nan...)
    df = clean_data.delete_columns_nan(df, porc_nan_max=0.99)
    print(df.head(2))
    
    # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
    df, df_etiquetas = format_data.convert_columns_to_int(df)
    if export:     
        df_etiquetas.to_excel(f'/Users/nachomondino/Desktop/df_etiquetas.xlsx', index=False)
        df.to_excel(f'/Users/nachomondino/Desktop/df_etiquetado.xlsx', index=False)

    # Elimino variables altamente correlacionadas
    if thr_corr is not None:
        l_columnas_a_eliminar = delete_correlated_columns(df, var_resp, thr_corr)
        df = df.drop(l_columnas_a_eliminar, axis=1)
        print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")
        # df.to_excel(f'/Users/nachomondino/Desktop/df_eliminado_corr.xlsx')

    # Selecciono las variables mas importantes (feature selection)
    if thr_fs is not None:
        n_cols = len(df.columns)
        l_important_features = select_best_features(df, var_resp, thr_fs, graf=export)
        df = df.loc[:, l_important_features + [var_resp]]
        print(f"\tSe eliminaron {n_cols-len(l_important_features)} de {n_cols} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%. Columnas importantes: {l_important_features}")

    print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(var_resp, axis=1).columns)}")

    print(df.shape)
    df = clean_data.verification_no_nan(df) # Funciona espectacular.
    print(df.shape)
    df = clean_data.delete_rows_nan(df, porc_nan_max=0)  # df = df.dropna()
    print(df.shape)

    if export:
        df.to_excel('/Users/nachomondino/Desktop/df_selected_prueba.xlsx', index=False)
    
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()