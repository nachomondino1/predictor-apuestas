# Importo librerias
from predictor.utils.set_up_logging import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from predictor.data_preparation import clean_data
from predictor.modeling.build_model import select_best_hiperparameters
from predictor.modeling.assess_model import normalize_column
from sklearn.feature_selection import SelectKBest, f_classif, chi2  # modelos estadisticos
from sklearn.feature_selection import f_regression  # via
from sklearn.feature_selection import RFE  # rfe
from sklearn.linear_model import LinearRegression, LogisticRegression  # rfe
from sklearn.linear_model import Lasso  # Lasso
from sklearn.ensemble import RandomForestClassifier  # Random forest
from sklearn.preprocessing import scale
import plotly.graph_objects as go
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier

def select_league_matches(df, verbose: int = 0):
    """
    Filtra partidos seleccionado solo aquellos que son de liga (eliminando partidos de copa)
    """
    # Levanto df_competencies
    df_comp = pd.read_excel('data/_shared/master_tables/df_competencies.xlsx')

    # Selecciono solo las ligas del pais
    l_leagues = list(df_comp[(df_comp['is_cup']==0) & (df_comp['is_second_division']==0)]['id_competition'].values) 
    
    # Filtro dataset segun ligas
    df = df[df['id_competition'].isin(l_leagues)]
    
    if verbose >=0:
        print("Ligas: ", l_leagues)
        print(f"Shape sin copas: {df.shape}")

    return df

def delete_correlated_columns(df: pd.DataFrame, var_resp: str, thr_corr: float = 0.7, verbose: int = 0) -> list:
    """
    Identificación de las columnas con una alta correlacion.
   
    # Parameters
    df: Dataframe con columnas numericas no? .(DataFrame)
    var_resp: Nombre de la variable respuesta (String)
    thr_corr: Correlacion umbral encima de la cual se considera que hay correlacion entre dos variables. (Float)
   
    # Returns
    Columnas a eliminar por alta correlacion. (List)
    """
    if verbose >= 1:
        print('\nEliminacion de columnas correlacionadas:')

    # Definicion de variables
    columnas_eliminar = set()  # Conjunto para almacenar las columnas a eliminar

    # Calculo matriz de correlacion
    df_correlacion = df.corr().abs()

    # Separo matriz de correlacion de las variables predictoras y de ellas con la variable objetivo
    df_corr_X = df_correlacion.drop(var_resp, axis=1).drop(var_resp, axis=0) # la borro del eje x e y
    df_corr_y = df_correlacion[var_resp].drop(var_resp, axis=0)

    # Obtener matriz triangular superior de correlacion (pues la matriz de correlacion es una matriz simetrica respecto de la diagonal)
    df_corr_tri_X = df_corr_X.where(np.triu(np.ones(df_corr_X.shape), k=1).astype(bool))
    if verbose >= 1:
        import os
        from dotenv import load_dotenv
        load_dotenv() # Cargar las variables de entorno desde el archivo .env
        BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

        df_correlacion.to_excel(f'{BASE_DIR_LOCAL}/df_correlacion.xlsx')
        df_corr_X.to_excel(f'{BASE_DIR_LOCAL}/df_corr_X.xlsx')
        df_corr_y.to_excel(f'{BASE_DIR_LOCAL}/df_corr_y.xlsx')
        df_corr_tri_X.to_excel(f'{BASE_DIR_LOCAL}/df_corr_tri_X.xlsx')

    # EL OBJETIVO ES NO ELIMINAR COLUMNAS POR CORRELACION CON UNA COLUMNA QUE YA DECIDI ELIMINAR...\
    l_cols = list(df_corr_tri_X.columns)

    # Por fila 
    for col1, row in df_corr_tri_X.iterrows():
        l_cols.remove(col1)  # Elimino columna 1 de l_cols para agilizar el procesamiento 

        # Si la columna 1 aun no fue eliminada por alta correlacion
        if col1 not in columnas_eliminar:

            # Por columna 
            for col2 in l_cols:
                
                # Si la columna 2 aun no fue eliminada por alta correlacion
                if col2 not in columnas_eliminar:

                    corr = row[col2]

                    # Si hay correlacion mayor a la umbral
                    if corr > thr_corr:

                        # Buscar correlacion de cada columna con variable objetivo
                        corr_col1_y, corr_col2_y = df_corr_y.loc[col1], df_corr_y.loc[col2]
            
                        # Eliminar aquella columna con menor correlacion con la variable objetivo
                        col_to_eliminate = col1 if corr_col2_y > corr_col1_y else col2
                        columnas_eliminar.add(col_to_eliminate)
                        if verbose >= 1:
                            print(f"\n Columna 1: {col1} y Columna 2: {col2} Correlacion: {corr*100:.0f}%")
                            print(f"Busco la mayor correlacion con y: Corr col 1 e y: {corr_col1_y*100:.0f}% ; Corr col 2 e y: {corr_col2_y*100:.0f}%")
                            print(f"Columna eliminada: {col_to_eliminate}")
                        
                        # Si eliminé la columna 1
                        if col_to_eliminate == col1:
                            # print(f"Dejo de probar si {col1} tiene correlacion con otras columnas puesto que ya fue eliminada por alta correlacion con {col2}")
                            break

    return list(columnas_eliminar), df_corr_tri_X

class FeatureSelection():

    def __init__(self, graficar_cada_metodo: bool = False) -> None:
        self.k = 5 # Cantidad de Folds
        self.bayes = False # Random tarda banda y Logistic +
        self.verbose = 1
        self.graficar_cada_metodo = graficar_cada_metodo

    # Definir función según tipo de variable respuesta
    def anova(self, X, y):
        """
        Realiza un análisis de importancia de características usando ANOVA (Analysis of Variance).
        Esta función puede usarse tanto para problemas de clasificación (variable respuesta discreta)
        como para problemas de regresión (variable respuesta continua).

        Parámetros:
        - X: DataFrame con las variables predictoras (independientes).
        - y: Variable respuesta (dependiente). Puede ser continua o discreta.

        Retorna:
        - Un DataFrame con las características y sus estadísticas de importancia (mod_estadisticos).
        """
        l_features, l_scores = [], []

        # Divido variables predictoras en categoricas (string) y numericas (int o float)
        numeric_vars = X.select_dtypes(include='number').columns.tolist()
        categorical_vars = X.select_dtypes(include='object').columns.tolist()
        if self.verbose >= 2:
            print(f"Variables numericas: {numeric_vars}")
            print(f"Variables categoricas: {categorical_vars}")

        # Seleccionar pruebas estadísticas según el tipo de variable respuesta (clasificación o regresión)
        if pd.api.types.is_integer_dtype(y):  # Clasificación
            score_func_num = f_classif  # para variables predictoras numéricas (ANOVA F-test para clasificación)
            score_func_cat = chi2  # para variables predictoras categóricas (Chi-cuadrado para clasificación)
        else:  # Regresión
            score_func_num = f_regression  # para variables predictoras numéricas (ANOVA F-test para regresion)
            score_func_cat = None  # No se usa chi2 en regresión

        # Análisis de importancia para variables predictoras numéricas
        if len(numeric_vars) > 0:
            numeric_selector = SelectKBest(score_func=score_func_num, k='all')
            numeric_selector.fit(X[numeric_vars], y)
            numeric_scores = numeric_selector.scores_

            l_features += numeric_vars
            l_scores += list(numeric_scores)

        # Análisis de importancia para variables predictoras categoricas
        if len(categorical_vars) > 0 and score_func_cat:
            categorical_selector = SelectKBest(score_func=score_func_cat, k='all')
            categorical_selector.fit(X[categorical_vars], y)
            categorical_scores = categorical_selector.scores_

            l_features += categorical_vars
            l_scores += list(categorical_scores)

        df_importance = pd.DataFrame({'mod_estadisticos': l_scores}, index=l_features)
        return df_importance

    def random_forest(self, X, y):
        """
        Calculo de importancia de cada variable segun modelo de random forest.

        # Parameters
            X: Dataframe con variables predictoras. (DataFrame)
            y: Dataframe solo con variable respuesta. (DataFrame)
            
        # Returns
            Dataframe. Importancia por variable. (Dataframe)
        """
        # Separo en train y val (para que select_best_hiperparameters() no tarde tanto)
        X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        # Entreno modelo con los mejores hiperparámetros
        model, params, best_metric, results  = select_best_hiperparameters(
            DecisionTreeClassifier(), # RandomForestClassifier(), 
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val, 
            y_val=y_val, 
            k=self.k, 
            bayes=self.bayes, 
            verbose=self.verbose
            )  # Tarda puesto que X no es del tamaño de X_val sino que de X_train

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'random_forest': model.feature_importances_}, index=X.columns)   # AttributeError: 'BayesSearchCV' object has no attribute 'feature_importances_'
        # df_importance = pd.DataFrame({'random_forest': model.best_estimator_.feature_importances_}, index=X.columns)

        # Grafico variables y su importancia
        if self.graficar_cada_metodo:
            visualize_tree(model, X_train)
            self.graficar_importancia_atrib(X=df_importance['random_forest'], y=df_importance.index)

        return df_importance

    def rfe(self, X, y):
        """
        Calculo de importancia de cada variable segun rfe.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Definicion de variables
        n_features = 1  # Número deseado de características seleccionadas hasta que se eliminan las menos relevantes

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)  # X_scaled = scaler.fit_transform(X) --> numpy y falla en la concatenacion en select_best_hyper

        # Separo en train y val
        X_train, X_val, y_train, y_val= train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

        # Entreno modelos buscando los mejores hiperparametros
        model, params, best_metric, results  = select_best_hiperparameters(
            LogisticRegression(), 
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val, 
            y_val=y_val, 
            k=self.k, 
            bayes=self.bayes, 
            verbose=self.verbose)  # Tarda puesto que X no es del tamaño de X_val sino que de X_train

        # Entreno modelo RFE a partir de Logistic
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        rfe.fit_transform(X_train, y_train) # rfe.fit_transform(X_scaled, y)

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'rfe': rfe.ranking_}, index=X.columns)
        # print("Resultados rfe: \n", df_importance)

        # Convierto ranking en importancia (a mayor ranking, menor importancia)
        func = lambda x: len(X.columns) - x + 1
        df_importance['rfe'] = df_importance['rfe'].apply(func)
        # print("Resultados rfe dsp convertir: \n", df_importance)

        if self.graficar_cada_metodo:
            self.graficar_importancia_atrib(X=df_importance['rfe'], y=df_importance.index)

        return df_importance

    def lasso_selection(self, X, y):
        """
        Calculo de importancia de cada variable segun lasso.

        :param X: Dataframe con variables predictoras. (DataFrame)
        :param y: Dataframe solo con variable respuesta. (DataFrame)
        :return: Dataframe. Importancia por variable. (Dataframe)
        """
        # Separo en train y val
        X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        # Entreno modelo con los mejores hiperparametros
        model, params, best_metric, results  = select_best_hiperparameters(
            Lasso(), 
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val, 
            y_val=y_val, 
            k=self.k,  
            bayes=self.bayes,
            verbose=self.verbose)  # Tarda puesto que X no es del tamaño de X_val sino que de X_train

        # Obtengo importancias por variable
        df_importance = pd.DataFrame({'lasso': model.coef_}, index=X.columns)

        # Convierto coeficiente en importancia (a mayor coef en valor abs, mas importancia)
        df_importance['lasso'] = df_importance['lasso'].apply(lambda x: abs(x))  # x es coef

        if self.graficar_cada_metodo:
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
        for col in df_importance.columns:
            # remove inf
            df_importance = df_importance.replace([np.inf, -np.inf], 0)
            df_importance = normalize_column(df_importance, col, norm_extension="_norm")

        # Seleccionar solo las columnas normalizadas
        norm_columns = [col for col in df_importance.columns if col.endswith('_norm')]

        # Calcular el promedio de las columnas normalizadas y almacenarlo en una nueva columna
        df_importance['suma_de_imp_norm'] = df_importance[norm_columns].mean(axis=1)
        return df_importance

def visualize_tree(model_best_params, X_train):
    
    # Verificá que best_model sea un DecisionTreeClassifier
    if isinstance(model_best_params, DecisionTreeClassifier):
        plt.figure(figsize=(20, 10))
        plot_tree(
            model_best_params, 
            feature_names=X_train.columns, 
            class_names=[str(c) for c in model_best_params.classes_], 
            filled=True,
            rounded=True,
            # max_depth=3  # Opcional: limita la profundidad visualizada
        )
        plt.title("Árbol de Decisión - Mejor Modelo")
        plt.show()
    else:
        print("El mejor modelo no es un DecisionTreeClassifier")
    
    '''
    dot_data = export_graphviz(
        model_best_params, 
        out_file=None, 
        feature_names=X_train.columns,
        class_names=[str(c) for c in model_best_params.classes_],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("mejor_arbol", format='png', cleanup=True)
    graph.view()  # Muestra el árbol en una ventana emergente
    graph.save('images/mejor_arbol.png')  # Guarda el árbol como imagen PNG
    '''

def select_best_features(df: pd.DataFrame, var_resp: str, thr_fs: float, thr_type: str = None, graf: bool = False):
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
    fs = FeatureSelection(graficar_cada_metodo=False)

    # Separo en X e y
    X, y = df.drop(var_resp, axis=1), df[var_resp]
    # print(np.any(np.isinf(X))) # Tiene que dar False

    # Detemino importancia de cada variable para cada modelo
    df_importance = pd.DataFrame(index=X.columns)

    # Verificar si es continua o discreta
    if pd.api.types.is_integer_dtype(y):  
        print("La variable respuesta es DISCRETA (clase)")
        df_importance = df_importance.merge(fs.anova(X, y), left_index=True, right_index=True)

    # Código para clasificación
    elif pd.api.types.is_numeric_dtype(y):  
        print("La variable respuesta es CONTINUA (regresión)")
        df_importance = df_importance.merge(fs.anova(X, y), left_index=True, right_index=True)

    # Normalizo importancias para poder sumarlas
    df_normalized = fs.sum_and_normalize_importances(df_importance)

    # Determino columnas mas importantes
    if thr_type is None:
        # Numero fijo
        l_important_features = df_normalized.loc[df_normalized['suma_de_imp_norm'] >= df_normalized['suma_de_imp_norm'].max() * thr_fs].index.tolist()  
    elif thr_type == 'percentil':
        # Basadas en percentil
        percentile_value = df_normalized['suma_de_imp_norm'].quantile(thr_fs)
        logger.info(f"El valor del percentil {thr_fs * 100}% es {percentile_value}")  # Loggear el valor del percentil
        l_important_features = df_normalized.loc[df_normalized['suma_de_imp_norm'] >= percentile_value].index.tolist()
    else:
        logger.error("El tipo de threshold no existe.")
        raise ValueError

    # Imprimir importancias por pantalla
    df_top_10 = df_normalized.sort_values(by='suma_de_imp_norm', ascending=False).head(10)    # Ordenar y seleccionar las 10 variables más importantes
    for pos, (idx, suma_de_imp_norm) in enumerate(zip(df_top_10.index, df_top_10['suma_de_imp_norm']), start=1):
        logger.info(f"Nº{pos}: Variable {idx} con importancia: {suma_de_imp_norm}")

    # Grafico importancias teniendo en cuenta todos los modelos
    if graf:
        fs.graficar_importancia_atrib(X=df_normalized['suma_de_imp_norm'], y=df_normalized.index)
    
    return l_important_features, df_normalized

def determine_country_competitions(id_country):
    """
    Determina los grupos de competencias para el pais.
    """
    # Levanto competencias
    df_comp = pd.read_excel('./data/_shared/master_tables/df_competencies.xlsx')

    if id_country == -1:
        df_comp_country = df_comp[(df_comp['id_country'].isin([48, 55, 59, 77, 148]))]
    else:
        # Selecciono las del pais
        df_comp_country = df_comp[(df_comp['id_country'] == id_country)]

    # Filtro
    df_comp_country_sin_sec_div = df_comp_country[(df_comp_country['is_second_division'] == 0)]
    df_comp_country_sin_cups = df_comp_country[(df_comp_country['is_cup'] == 0)]

    # Guardo datos
    all_comp = list(df_comp_country['id_competition'].values)
    comp_sin_b = list(df_comp_country_sin_sec_div['id_competition'].values)
    comp_sin_cups = list(df_comp_country_sin_cups['id_competition'].values)
    comp_solo_liga = list(df_comp_country_sin_cups[(df_comp_country_sin_cups['is_second_division'] == 0)]['id_competition'].values)
    d = {'all_comp': all_comp, 'comp_sin_b': comp_sin_b, 'comp_sin_cups': comp_sin_cups, 'comp_solo_liga': comp_solo_liga}
    print(d)
    
    return d

def prueba():
    from predictor.data_preparation import format_data
    import os
    from dotenv import load_dotenv
    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    # Definicion de variables
    var_resp = 'result'
    country = 'england'
    export = False

    # Definicion de hiperparametros
    thr_corr = None  # Correlacion minima entre dos variables para indicar una alta correlacion [0-1] (siendo 1 correlacion maxima y 0 sin correlacion)
    thr_fs = None  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)
    export = False

    # Levanto dataset de prueba
    df = pd.read_excel(f'./data/{country}/data_preparation/df_constructed.xlsx', index_col=0)
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
        df_etiquetas.to_excel(f'{BASE_DIR_LOCAL}/df_etiquetas.xlsx', index=False)
        df.to_excel(f'{BASE_DIR_LOCAL}/df_etiquetado.xlsx', index=False)

    # Elimino variables altamente correlacionadas
    if thr_corr is not None:
        l_columnas_a_eliminar = delete_correlated_columns(df, var_resp, thr_corr)
        df = df.drop(l_columnas_a_eliminar, axis=1)
        print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")
        # df.to_excel(f'{BASE_DIR_LOCAL}/df_eliminado_corr.xlsx')

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
        df.to_excel(f'{BASE_DIR_LOCAL}/df_selected_prueba.xlsx', index=False)
    
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()