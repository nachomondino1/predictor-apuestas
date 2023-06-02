# Importo librerias
import pandas as pd
import time
import warnings

# # Data understanding
# from data_understanding.collect_data import scraper_sofifa, scraper_flashscore
# from dspy.data_understanding.describe_data import getting_to_know_data
#
# # Data preparation
# from data_preparation import format_data, integrate_data, construct_data, select_data, clean_data
# from dspy.data_preparation import clean_data as cd  # Para categorizar variables numericas


# Modeling
# Generate test design
from dspy.modeling import test_design
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
# Build model
from modeling import build_model, asses_model
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
import lightgbm as lgb  # Gradient Boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
# Assess model
from modeling.asses_model import calculate_ROI, confusion_matrix
from sklearn.metrics import accuracy_score


class DataPreparation:

    def __init__(self, var_resp):
        self.var_resp = var_resp

    def format_data(self, df_part=None, df_jug=None, export=False):  # 0.2 min
        """
        Arreglo el data type de algunas variables
        :param df_part: Dataframe de los datos de los partidos. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        if df_part is None and df_jug is None:
            # df_part = pd.read_excel('data_understanding/collect_data/data/entidad_partido_argentina.xlsx')
            # df_jug = pd.read_excel('data_understanding/collect_data/data/entidad_jugadores.xlsx')
            df_part = pd.read_excel('data_understanding/collect_data/data_seg/entidad_partido_argentina.xlsx')
            df_jug = pd.read_excel('data_understanding/collect_data/data_seg/entidad_jugadores.xlsx')
            df_jug = df_jug[df_jug['pais'] == 'argentina']  # Selecciono solo jugadores de argentina para hacer mas rapido...

        start = time.time()
        print("\nFormateando los datos...")

        # Entidad partido: fecha de string a datetime, posesion de str a float
        df_part = format_data.convert_fecha_to_datetime(df_part, string_format='%d.%m.%Y %H:%M') # Fundamental para poder ordenar el df por 'fecha' # A pesar de transformalo en la extraccion, lo vuelve a entender como str y no como dt
        df_part = format_data.convert_posesion_to_int(df_part)
        df_part['es_copa'] = df_part['es_copa'].replace(True, 1).replace(False, 0)

        # Entidad jugador: fecha de string a datetime y convierto valor de mercado en entero
        df_jug = format_data.convert_fecha_to_datetime(df_jug, string_format='%b %d, %Y')
        df_jug = format_data.convert_valor_mercado_to_int(df_jug)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel('./data_preparation/data/df_part_formated.xlsx', index=False)
            df_jug.to_excel('./data_preparation/data/df_jug_formated.xlsx', index=False)

        return df_part, df_jug

    def integrate_data(self, df_part=None, df_jug=None, export=False):  # 42.6 min (sin copa arg y otras comp)
        """
        Integra los datos de partidos y jugadores en un solo dataframe.
        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        if df_part is None and df_jug is None:
            df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_part_formated.xlsx')
            df_jug = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_jug_formated.xlsx')

        start = time.time()
        print("\nIntegrando los datos...")

        # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
        df_part = clean_data.remove_strings_from_teams(df_part)  # remuevo strings adicionales en los nombres de los equipos
        df_part = clean_data.prepare_text_columns(df_part)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales
        df_jug = clean_data.prepare_text_columns(df_jug)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales
        df_part = clean_data.change_teams_names(df_part)  # e.g. atl. tucuman --> atletico tucuman

        # Integro entidad partido y jugador
        df_integrated = integrate_data.player_data_in_match(df_part, df_jug)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_integrated.to_excel('./data_preparation/data/df_integrated.xlsx', index=False)

        return df_integrated

    def construct_data(self, df=None, N_ULT_PART = 5, export=False):  # 4.2 min
        """
        Construye nuevos datos a partir de un dataframe existente.
        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        if df is None:
            df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_integrated.xlsx')

        start = time.time()
        print("\nConstruyendo nuevos datos...")

        # Ordeno por campo 'fecha'
        # df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Todos son ascending=True salvo historial_entre_si_segun_localia

        # Construct data
        df = construct_data.determinar_equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis

        df = construct_data.historial_entre_si_segun_localia(df, n_ult_part=int(N_ULT_PART/2))

        l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, l_var=l_variables_a_prom)

        df = construct_data.promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos

        df = construct_data.forma_reciente(df, n_part=N_ULT_PART)  # df = derive_forma_ponderada(df, n_part=N_ULT_PART)

        # Calculo diferencias para las variables promedio de los jugadores
        df = construct_data.calculate_dif_col_jugadores(df)

        df = construct_data.n_dias_ult_partido(df)
        # df = convert_odds_to_prob(df)
        # df = numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel('./data_preparation/data/df_constructed.xlsx', index=False)

        return df

    def clean_data(self, df=None, export=False):  # tengo que limpiar los datos antes de seleccionar porque no le pueden entrar NaN ni columnas no numericas. A su vez, tengo que eliminar las columnas que no sirven para el modelo puesto que puede que me hagan borrar mas registros al tener mas nan values.
        """
        Limpia los datos de un dataframe. Elimina variables que no se usan para analizar los datos, remueve NaNs y
        convierte variables str a int.
        :param df: Dataframe de los datos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe limpiado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe limpiado. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        if df is None:
            df = pd.read_excel('data_preparation/data/df_constructed.xlsx')

        start = time.time()
        print("\nLimpiando los datos...")

        # Categorizo columnas numericas
        # df = clean_data.categorize_numeric_columns(df)

        # Eliminacion de outliers?

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

        # Remover NaN values
        df = df.dropna()  # inplace=True  # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)

        # Convertir variables categoricas string a categoricas numericas
        df = clean_data.convert_columns_to_int(df)

        end = time.time()
        print(f"Limpieza de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel('./data_preparation/data/df_cleaned.xlsx', index=False)

        return df

    def select_data(self, df=None, export=False):  # Implementar feature selection...
        """
        Selecciona las variables relevantes del dataframe.
        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        if df is None:
            df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_cleaned.xlsx')

        start = time.time()
        print("\nSeleccionado datos...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        # df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

        # No usaré la matriz de correlacion puesto que haré feature selection

        # Elimino variables no son importantes
        # df = select_data.feature_selection(df)
        # df = df.drop(['dif_faltas_segun_ult_part', 'dif_offsides_segun_ult_part'], axis=1)  # Para red neuronal
        # df.drop(['dif_pases_segun_ult_part', 'dif_pases_comp_segun_ult_part', 'dif_remates_a_puerta_segun_ult_part', 'dif_tarjetas_amarillas_segun_ult_part', 'dif_ataques_segun_ult_part', 'dif_ataques_pelig_segun_ult_part'], inplace=True, axis=1)

        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel('./data_preparation/data/df_selected.xlsx', index=False)

        return df

class Modeling:

    def __init__(self, var_resp, var_pred):
        self.var_resp = var_resp
        self.var_pred = var_pred

    def generate_test_design(self, df=None, export=True):
        """
        Balancea el dataset y separa en conjuntos de entrenamiento y testeo
        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe de entrenamiento y de testeo balanceados (DataFrame)
        """
        if df is None:
            # Levanto dataset ya preparado
            df = pd.read_excel('data_preparation/data/df_selected.xlsx')  # Cambiar a cleaned...

        # Definicion de variables
        oversampler = RandomOverSampler()

        # Shuffle dataset
        df = pd.DataFrame(shuffle(df))  # df = df.sample(frac=1).reset_index(drop=True)
        print(f"Shape dataframe original: {df.shape}")

        # Balanceamos segun variable respuesta     # df = clean_data.balance_dataset(df, var_resp=self.var_resp)
        X_bal, y_bal = oversampler.fit_resample(df.drop(self.var_resp, axis=1), df[self.var_resp])
        df_balanced = pd.concat([X_bal, y_bal], axis=1)
        print(f"Shape dataframe luego de balanceo: {df_balanced.shape}")

        # Separo conjunto de datos en train y test
        df_train, df_test = test_design.separate_train_and_test(df_balanced, porc_corte=0.8)
        print(f"Shape de df_train y df_test : {df_train.shape} {df_test.shape}")

        # Elimino variables de cuotas puesto que no las usare para entrenar sino que solo para calcular el roi
        # df_train = df_train.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
        # Lo llevo momentaneamente a train_model() para ver si puedo calcular el roi

        if export:
            df_train.to_excel('./modeling/data/df_train.xlsx', index=False)
            df_test.to_excel('./modeling/data/df_test.xlsx', index=False)

        return df_train, df_test

    def select_best_model(self, df_train, l_modelos, best_params=False, k=10):
        """
        Selecciona el mejor modelo a partir de la precision
        :param df_train: Dataframe de entrenamiento. (DataFrame)
        :param l_modelos: Lista de nombres de modelos a probar. (Lista)
        :param best_params: Booleano para indicar si se deben buscar los mejores hiperparametros para cada modelo. True
        para buscar, False de lo contrario. (bool)
        :param k: Numero de folds. (int)
        :return: Mejor modelo. (sklearn.ensemble?)
        """
        # Definicion de variables
        warnings.filterwarnings("ignore")
        best_acurracy = 0
        best_model = None

        best_roi = 0
        best_model_2 = 0

        # Por modelo a probar
        for modelo in l_modelos:

            # Entreno modelo
            print(f" Modelo: {str(modelo)[:str(modelo).find('(')]} ".center(120, '#'))
            model, cv_accuracy, cv_roi = build_model.train_model(df_train, self.var_resp, modelo, best_params, k)

            # Si es el mejor modelo hasta aqui # Uso accuracy en todos los folds de cv... para elegir el mejor modelo
            if cv_accuracy > best_acurracy:  # GUARDAR MAS METRICAS? HAGO EL ASSESS MODEL ACA?
                # Guardo modelo
                best_acurracy = cv_accuracy
                # best_roi = roi?
                best_model = model

            if cv_roi > best_roi:
                best_roi = cv_roi
                best_model_2 = model

        print(f"\nEl mejor modelo es: {best_model}")

        print(f"\nEl mejor modelo es: {best_model_2}")

        return best_model  #df_result y_real e y_pred

    def assess_model(self, model, df_test):
        """
        # Hago prediccion aca? y calculo metricas?

        :param model: Modelo de Machine Learning. (sklearn.ensemble)
        :param df_test: Dataframe de testeo. (DataFrame)
        :return: Precision del modelo y roi en el conjunto de testeo. (int) y (float)
        """
        # Quito odds de df_test para evitar error
        df_test_without_odds = df_test.copy().drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)

        # Predecir las etiquetas para los datos de prueba
        y_pred = model.predict(df_test_without_odds.drop(self.var_resp, axis=1))  # es un numpy array

        # Calculo metricas
        test_accuracy = accuracy_score(df_test[self.var_resp], y_pred)
        print(f"Precisión del modelo en los datos de prueba: {test_accuracy:.3f}")

        # Agrego y_pred a df_test para poder calcular ROI
        df_test['y_pred'] = y_pred
        roi = calculate_ROI(df_test, self.var_resp, self.var_pred)
        print(f"ROI del modelo en los datos de prueba: {roi:.3f}")

        confusion_matrix(df_test, self.var_resp, self.var_pred)

        df_test.to_excel('/Users/nachomondino/Desktop/df_results.xlsx')
        return test_accuracy, roi


def main():  # La idea es poner toda el camino de los datos aqui...

    # Definicion de variables
    data_unders, data_prep, modeling = False, False, True
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    prepare, modeler = DataPreparation(var_resp), Modeling(var_resp, var_pred)

    if data_unders is True:
        # Collect initial data
        df_part = scraper_flashscore.extract_flashscore()  # Tengo que ver que no se corran igual por no comentar la funcion en su archivo...
        df_jug = scraper_sofifa.extract_sofifa
        print(f"Dataframe partido:\n{df_part} \nDataframe jugadores:\n{df_jug}")

        # Describe data (quiero describir los datos igual aunque no los extraiga...)
        getting_to_know_data(df_part)
        getting_to_know_data(df_jug)

    if data_prep is True:
        # Hiperparametros
        N_ULT_PART = 5  # Numero de partidos a tener en cuenta para variables historicas como posesion en ult partidos

        # Preparo el dataset para el analisis
        # df_part, df_jug = prepare.format_data(export=True)  # df_part, df_jug,
        # df = prepare.integrate_data(df_part, df_jug, export=True)
        # df = prepare.construct_data(df, N_ULT_PART=N_ULT_PART, export=True)
        df = prepare.clean_data(export=True)  # df
        df = prepare.select_data(df, export=True)

    if modeling is True:

        # Hiperparametros
        best_params = False  # True para hacer GridSearch para buscar los mejeres hiperparametros.
        k = 2  # Numero de folds
        l_modelos = [DecisionTreeClassifier(max_depth=30),
                     RandomForestClassifier(n_estimators=200, max_depth = None, random_state=42),
                     xgb.XGBClassifier(n_estimators=50, objective='multi:softmax', num_class=3, max_depth=20),  # num_class = len(y.unique()) Depende del numero de clases...
                     LogisticRegression(multi_class='multinomial', penalty='l2', C=0.1, solver='lbfgs', max_iter=500),
                     SVC(kernel='rbf', decision_function_shape='ovo'), # --> tarda mas de 1 hora
                     MLPClassifier(hidden_layer_sizes=128, activation='tanh', solver='adam', learning_rate='invscaling', max_iter=300),
                     GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=7)
                     # self.red_neuronal(n_folds_cv=10, n_epochs=1000, batches=256),
                     ]

        # Analizo los datos
        df_train, df_test = modeler.generate_test_design(export=True)
        best_model = modeler.select_best_model(df_train, l_modelos, best_params, k)
        precision, roi = modeler.assess_model(best_model, df_test)

main()