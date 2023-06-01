import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import f_regression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


from data_preparation import clean_data


from sklearn.decomposition import PCA



def pca(X, n_components):
    # X es el conjunto de datos de características (variables predictoras)

    # Inicializar el objeto PCA con el número de componentes deseados
    pca = PCA(n_components=n_components)

    # Ajustar y transformar los datos
    X_pca = pca.fit_transform(X)

    # Obtener la varianza explicada por cada componente principal
    explained_variance_ratio = pca.explained_variance_ratio_

    # Imprimir la varianza explicada por cada componente principal
    for i in range(n_components):
        print(f"Varianza explicada por el Componente Principal {i + 1}: {explained_variance_ratio[i]}")

    # Obtener los componentes principales
    components = pca.components_



def random_forest_classifier(df, var_resp):  # Es sin variables categoricas

    # Dividir los datos en características (X) y variable objetivo (y)
    X, y = df.drop(var_resp, axis=1), df[var_resp]

    # Crear el modelo de Random Forest
    model = RandomForestClassifier()

    # Ajustar el modelo a los datos
    model.fit(X, y)

    # Obtener la importancia de características
    feature_importance = model.feature_importances_

    # Crear un DataFrame con la importancia de características
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

    # Ordenar las características por importancia descendente
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    # Imprimir las características más importantes
    print(feature_importance_df.head(10))

    # Selecciona las características con importancia por encima de un umbral
    threshold = 0.05
    selector = SelectFromModel(model, threshold='mean', prefit=True)  # 'mean'
    X_new = selector.transform(X)

    # Muestra la importancia de las características seleccionadas
    support = selector.get_support()
    for feature, importance, supported in zip(X.columns, feature_importance, support):
        if supported:
            print(f"{feature}: {importance:.3f} (selected)")
        else:
            print(f"{feature}: {importance:.3f}")

    # Ajusta un nuevo modelo de bosque aleatorio solo con las características seleccionadas
    rf_new = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_new.fit(X_new, y)

def f_regressiones(df, var_resp):

    # Dividir los datos en características (X) y variable objetivo (y)
    X, y = df.drop(var_resp, axis=1), df[var_resp]

    # Realizar la prueba F y obtener los valores F y p-valores
    f_values, p_values = f_regression(X, y)

    # Ordenar las características por su p-valor
    sorted_indices = np.argsort(p_values)
    sorted_features = X.columns[sorted_indices]

    # Seleccionar las características con un p-valor significativo (por ejemplo, p < 0.05)
    significant_features = sorted_features[p_values[sorted_indices] < 0.1]

    # Imprimir las características seleccionadas
    print(significant_features)

def rfe(df, var_resp, n_features_to_select = 5):
    """
    :param df:
    :param var_resp:
    :param n_features_to_select: Número de características a seleccionar
    :return:
    """

    # Dividir los datos en características (X) y variable objetivo (y)
    X, y = df.drop(var_resp, axis=1), df[var_resp]

    # Inicializar el estimador del modelo
    estimator = LogisticRegression()

    # Inicializar el selector RFE
    selector = RFE(estimator)

    # Definir el número de características a seleccionar
    selector.n_features_to_select = n_features_to_select

    # Realizar la selección de características
    selector.fit(X, y)

    # Obtener las características seleccionadas
    selected_features = X.columns[selector.support_]
    print(selected_features)

def prueba():
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_constructed.xlsx')

    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)
    df = df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)


    # # Eliminacion de NaN values
    # for col in df.select_dtypes(include=['float64', 'int64']).columns:
    #     mean = df[col].mean()  # Calcula la media de una columna
    #     df[col] = df[col].fillna(mean)  # Rellena los NaN en esa columna con la media
    #     print(f"Columna: {col} \nMedia: {mean}")

    # df = df.drop(['dif_pases_comp_segun_ult_part', 'dif_edad_aus', 'dif_alt_aus', 'dif_rat_aus', 'dif_pases_segun_ult_part', 'dif_offsides_segun_ult_part', 'dif_ataques_segun_ult_part', 'dif_ataques_pelig_segun_ult_part'], axis=1) # Tienen mucho nan, solo me quedan 229 registros...
    df = df.dropna()  # inplace=True  # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)
    print(df.shape)

    # Convertir variables categoricas string a categoricas numericas
    df = clean_data.convert_columns_to_int(df)


    # df_correlation_matrix = df.drop('equipo_ganador', axis=1).corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"
    # df_correlation_matrix.to_excel('/Users/nachomondino/Desktop/correlation_matrix.xlsx')



    # random_forest_classifier(df, 'equipo_ganador')

    # f_regressiones(df, 'equipo_ganador')

    # rfe(df, 'equipo_ganador', 8)

    pca(df.drop('equipo_ganador', axis=1), 2)

    # df.drop(['dif_pases', 'dif_pases_comp', 'dif_remates_a_puerta', 'dif_tarjetas_amarillas', 'dif_ataques_pelig'], inplace=True, axis=1)

# prueba()
