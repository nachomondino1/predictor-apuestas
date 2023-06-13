# Importo librerias
import pandas as pd
import time
import warnings
# Data understanding
from data_understanding import collect_initial_data
from dspy.data_understanding.describe_data import getting_to_know_data
# Data preparation
from data_preparation import format_data, integrate_data, construct_data, select_data, clean_data
# Modeling
# Generate test design
from dspy.modeling import test_design
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
# Build model
from modeling import build_model
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
import lightgbm as lgb  # Gradient Boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
# Assess model
from modeling.asses_model import calculate_ROI, confusion_matrix
from sklearn.metrics import accuracy_score
import pickle


class DataPreparation:  # 17.4 min

    def __init__(self, var_resp, pais):
        self.var_resp = var_resp
        self.pais = pais

    def format_data(self, df_part=None, df_jug=None, export=False):  # 0.0 min
        """
        Arreglo el data type de algunas variables
        :param df_part: Dataframe de los datos de los partidos. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{self.pais}/entidad_partido.xlsx') if df_part is None else df_part
        df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{self.pais}/entidad_jugadores.xlsx') if df_jug is None else df_jug

        start = time.time()
        print("\nFormateando los datos...")

        # Entidad partido: fecha, posesion y es_copa
        df_part = format_data.convert_fecha_to_datetime(df_part, string_format='%d.%m.%Y %H:%M')  # ya lo voy a extraer datetime... # Fundamental para poder ordenar el df por 'fecha'
        df_part = format_data.convert_posesion_to_int(df_part)  # Podria usar la limpieza de punct de tp y luego convertir a int64 pero as al pedo
        df_part['es_copa'] = df_part['es_copa'].replace(True, 1).replace(False, 0)  # ya no va a ser necesario...

        # Entidad jugador: fecha y valor de mercado
        df_jug = format_data.convert_fecha_to_datetime(df_jug, string_format='%b %d, %Y')
        df_jug = format_data.convert_valor_mercado_to_int(df_jug)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_part_formated.xlsx', index=False)
            df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_jug_formated.xlsx', index=False)

        return df_part, df_jug

    def clean_data(self, df_part=None, df_jug=None, export=False):  # 0.0 min
        """
        Limpia los datos de un dataframe.
        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe limpiado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe limpiado. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df_part = pd.read_excel(f'./data_preparation/data/{self.pais}/df_part_formated.xlsx') if df_part is None else df_part
        df_jug = pd.read_excel(f'./data_preparation/data/{self.pais}/df_jug_formated.xlsx') if df_jug is None else df_jug

        start = time.time()
        print("\nLimpiando los datos...")

        # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
        df_part = clean_data.prepare_text_columns(df_part, l_col_to_except=['id', 'temporada'])  # df_part = clean_data.prepare_text_columns(df_part)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales
        df_jug = clean_data.prepare_text_columns(df_jug, l_col_to_except=['id'])  # df_jug = clean_data.prepare_text_columns(df_jug)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

        # Remuevo strings adicionales en los nombres de los equipos
        df_part = clean_data.clean_teams_names(df_part)

        end = time.time()
        print(f"Limpieza de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_part_cleaned.xlsx', index=False)
            df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_jug_cleaned.xlsx', index=False)

        return df_part,df_jug

    def integrate_data(self, df_part=None, df_jug=None, export=False):  # 13.3 min (sin copa arg y otras comp)
        """
        Integra los datos de partidos y jugadores en un solo dataframe.
        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_part_cleaned.xlsx') if df_part is None else df_part
        df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_jug_cleaned.xlsx') if df_jug is None else df_jug

        start = time.time()
        print("\nIntegrando los datos...")

        # Integro entidad partido y jugador
        df_integrated = integrate_data.player_data_in_match(df_part, df_jug)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_integrated.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_integrated.xlsx', index=False)

        return df_integrated

    def construct_data(self, df=None, N_ULT_PART=5, export=False):  # 2.7 minutos
        """
        Construye nuevos datos a partir de un dataframe existente.
        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_integrated.xlsx') if df is None else df

        start = time.time()
        print("\nConstruyendo nuevos datos...")

        # Ordeno por campo 'fecha'
        # df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Todos son ascending=True salvo historial_entre_si_segun_localia

        # Construyo variable respuesta: "equipo_gandor"
        df = construct_data.determinar_equipo_ganador(df)

        # Construyo variables historicas
        l_estad_part = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
        df = construct_data.historial_entre_si_segun_localia(df, n_ult_part=int(N_ULT_PART/2))
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, l_var=l_estad_part)  # Estadisticas del partido
        df = construct_data.promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Diferencia de gol
        df = construct_data.forma_reciente(df, n_part=N_ULT_PART)  # Rendimieento del equipo
        df = construct_data.n_dias_ult_partido(df)  # Numero de dias desde ultimo partido

        # Construyo variables de diferencias para las variables promedio de los jugadores
        df = construct_data.calculate_dif_col_jugadores(df)

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_constructed.xlsx', index=False)

        return df

    def select_data(self, df=None, thr_corr=0.6, perc_fs=0.5, treat_nan='drop', export=False):  # 1.3 minutos
        """
        Selecciona las variables relevantes del dataframe.
        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_constructed.xlsx') if df is None else df

        warnings.filterwarnings('ignore')
        start = time.time()
        print("\nSeleccionado datos...")

        # Para no borrar registros utiles que no tienen alguna columna en especifico como ataques_pelig, borro columnas con mas nan
        print(df.shape)
        # si no borra columnas que tienen mucho nan por ser nan en partidos viejos
        df = df.dropna(subset=['arbitro', 'odds_loc']) # Si conviene, conviene hacerlo antes o despues?
        df = select_data.eliminar_columnas_nan(df, 0.5)
        print(df.shape)

        # Elimino registros sin estadisticas ni formaciones (mucho NaN)
        print(df.shape)
        df = df.dropna() # Si conviene, conviene hacerlo antes o despues?
        df.to_excel('/Users/nachomondino/Desktop/df_selected_dsp_drop_na.xlsx', index=False)
        print(df.shape)

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df = format_data.convert_columns_to_int(df)

        # Selecciono las variables con menor correlacion  # No usaré la matriz de correlacion puesto que haré feature selection??
        l_columnas_a_eliminar = select_data.eliminar_columnas_correlacionadas(df, self.var_resp, thr_corr)
        df = df.drop(l_columnas_a_eliminar, axis=1)

        # Selecciono las variables mas importantes (feature selection)
        # df.to_excel('/Users/nachomondino/Desktop/df_selected_antes_drop_na.xlsx', index=False)
        l_selected_features = select_data.feature_selection(df.dropna(), self.var_resp, percentil=perc_fs)   # Le paso el df sin NaN values para evitar ""ValueError: Input X contains NaN.".  Pero no hago fillna() puesto que introduce sesgo
        df = df.loc[:, l_selected_features + ['odds_loc', 'odds_emp', 'odds_vis', self.var_resp]]

        # Tratamiento de NaN values (drop, fillna con moda, fillna con random forest)
        # df = clean_data.treat_nan_values(df, type=treat_nan)

        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_selected.xlsx', index=False)

        return df

class Modeling:

    def __init__(self, var_resp, var_pred, pais):
        self.var_resp = var_resp
        self.var_pred = var_pred
        self.pais = pais

    def generate_test_design(self, df=None, porc_corte=0.8, export=True):
        """
        Balancea el dataset y separa en conjuntos de entrenamiento y testeo
        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe de entrenamiento y de testeo balanceados (DataFrame)
        """
        print("\nGenerando datasets de entrenamiento y testeo...")
        if df is None:
            # Levanto dataset ya preparado
            df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_selected.xlsx')

        # Definicion de variables
        oversampler = RandomOverSampler()

        # Shuffle dataset
        df = pd.DataFrame(shuffle(df)).reset_index(drop=True)  # df = df.sample(frac=1).reset_index(drop=True)
        print(f"Shape dataframe original: {df.shape}")

        # # Balanceamos segun variable respuesta   --> no es el problema. Las precisiones son peores sin el pero   # df = clean_data.balance_dataset(df, var_resp=self.var_resp)
        X_bal, y_bal = oversampler.fit_resample(df.drop(self.var_resp, axis=1), df[self.var_resp])
        df_balanced = pd.concat([X_bal, y_bal], axis=1)
        print(f"Shape dataframe luego de balanceo: {df_balanced.shape}")

        # Shuffle dataset (si bien separate_train_and_test() hará shuffle, me quiero asegurar siempre de evitar cualquier sesgo tras el agregado de filas por el balanceo)
        df_balanced = pd.DataFrame(shuffle(df_balanced)).reset_index(drop=True)  # df = df.sample(frac=1).reset_index(drop=True)

        # Separo conjunto de datos en train y test --> tampoco es el problema...
        df_train, df_test = test_design.separate_train_and_test(df_balanced, porc_corte=porc_corte)
        print(f"Shape de df_train y df_test : {df_train.shape} {df_test.shape}")

        if export:
            df_train.to_excel(f'./modeling/data/{self.pais}/df_train.xlsx', index=False)
            df_test.to_excel(f'./modeling/data/{self.pais}/df_test.xlsx', index=False)

        return df_train, df_test

    def select_best_model(self, df_train, l_modelos, best_params=False, k=10, export=True):
        """
        Selecciona el mejor modelo a partir de la precision
        :param df_train: Dataframe de entrenamiento. (DataFrame)
        :param l_modelos: Lista de nombres de modelos a probar. (Lista)
        :param best_params: Booleano para indicar si se deben buscar los mejores hiperparametros para cada modelo. True
        para buscar, False de lo contrario. (bool)
        :param k: Numero de folds. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe. True para exportar, False de lo contrario. (bool)
        :return: Mejor modelo. (sklearn.ensemble?)
        """
        # Definicion de variables
        warnings.filterwarnings("ignore")
        df_models = pd.DataFrame(columns=['model', 'cv_accuracy', 'cv_roi'])  # Datos del modelo y su precision y roi... --> en vez de imprimirlo por pantalla, genero un df...
        print("\nSeleccionando el mejor modelo...")

        # Entreno modelos
        for modelo in l_modelos:

            # print(f" Modelo: {str(modelo)[:str(modelo).find('(')]} ".center(120, '-'))
            model, cv_accuracy, cv_roi = build_model.train_model(df_train, self.var_resp, modelo, best_params, k)  # Le paso pais por df_etiquetas...?
            df_models.loc[len(df_models)] = [model, cv_accuracy, cv_roi]

        # Selecciono el mejor modelo
        idx = df_models[df_models['cv_roi'] == max(df_models['cv_roi'])].index[0]
        best_model, best_accuracy, best_roi = df_models.loc[idx, 'model'], df_models.loc[idx, 'cv_accuracy'], df_models.loc[idx, 'cv_roi']
        print(f"El mejor modelo es: {best_model} con ROI: {best_roi:.1f}% y precision: {best_accuracy:.1f}%")

        if export:
            df_models.to_excel(f'./modeling/data/{self.pais}/df_modelos.xlsx')
            pickle.dump(best_model, open(f"./modeling/data/{self.pais}/modelo.pkl", "wb"))

        return best_model

    def assess_model(self, model, df_test, export=True):
        """
        # Hago prediccion aca? y calculo metricas?

        :param model: Modelo de Machine Learning. (sklearn.ensemble)
        :param df_test: Dataframe de testeo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Precision del modelo y roi en el conjunto de testeo. (int) y (float)
        """
        print("\nEvaluando modelo con datos de prueba...")

        # Quito cuotas de casas de apuestas y variable respuesta de df_test
        df_test_pred = df_test.copy().drop([self.var_resp, 'odds_loc', 'odds_emp', 'odds_vis'], axis=1)  # Esto esta ok

        # Predecir las etiquetas para los datos de prueba
        y_pred = model.predict(df_test_pred)  # es un numpy array

        # Asignar las predicciones a una nueva columna en df_test (para poder calcular ROI)
        df_test[self.var_pred] = y_pred

        # Calculo metricas e imprimo resultados
        test_accuracy = accuracy_score(df_test[self.var_resp], df_test[self.var_pred]) * 100
        roi = calculate_ROI(df_test, self.var_resp, self.var_pred)
        print(f"Resultados promedios del modelo en los datos de prueba: \n  - Precision prom: {test_accuracy:.1f}% \n  - ROI prom: {roi:.1f}%")
        confusion_matrix(df_test, self.var_resp, self.var_pred)  # podria exportar el archivo? para evitar tener que cerrarla para que continue el programa

        if export:
            df_test.to_excel(f'./modeling/data/{self.pais}/df_results.xlsx')

        return test_accuracy, roi

def main():  # La idea es poner toda el camino de los datos aqui...

    # Definicion de variables
    data_unders, data_prep, modeling = False, False, True
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    pais = "argentina"  # Ver si puedo evitar el pais como argumento en train model por tener que meterlo en el calculo del roi en df_etiquetas...
    export = True
    prepare, modeler = DataPreparation(var_resp, pais), Modeling(var_resp, var_pred, pais)

    if data_unders is True:

        print(" Data understanding ".center(120, "#"))

        # Collect initial data
        print(" Recolectando datos... ")
        # df_part = collect_initial_data.extract_partidos()         df_part = scraper_flashscore.extract_flashscore()  # Tengo que ver que no se corran igual por no comentar la funcion en su archivo...
        # df_jug = collect_initial_data.extract_jugadores()
        # print(f"Dataframe partido:\n{df_part} \nDataframe jugadores:\n{df_jug}")

        # Describe data (quiero describir los datos igual aunque no los extraiga...)
        print(" Describiendo datos... ")
        # getting_to_know_data(df_part)
        # getting_to_know_data(df_jug)

    if data_prep is True:
        # Hiperparametros
        N_ULT_PART = 5  # Numero de partidos a tener en cuenta para variables historicas como posesion en ult partidos
        thr_corr = 0.7  # Correlacion umbral para la eliminacion de variables altamente correlacionadas  # Con 0.6 : {'dif_valor_sup', 'dif_pases_comp_segun_ult_part', 'dif_rat_sup', 'dif_valor_aus', 'dif_pases_segun_ult_part', 'dif_gol', 'dif_valor_tit', 'dif_remates_segun_ult_part', 'dif_ataques_segun_ult_part'}
        perc_fs = 0.5  # Percentil de importancias para la seleccion de variables mas importantes  # Con 0.6: ['dt_vis', 'historial_entre_si', 'dif_posesion_segun_ult_part', 'dif_remates_a_puerta_segun_ult_part', 'dif_offsides_segun_ult_part', 'dif_ataques_pelig_segun_ult_part', 'dif_edad_tit', 'dif_rat_tit', 'dif_edad_sup', 'dif_rat_aus']  Con 0.7: ['historial_entre_si', 'dif_posesion_segun_ult_part', 'dif_remates_a_puerta_segun_ult_part',  'dif_ataques_pelig_segun_ult_part', 'dif_rat_tit', 'dif_edad_sup', 'dif_rat_aus']
        treat_nan = 'fillna_with_ml' # Tratamiento de nan values: dropna, fillna_with_mode, fillna_with_ml
        print(" Data preparation ".center(120, "#"))

        # Preparo el dataset para el analisis
        # df_part, df_jug = prepare.format_data(export=export)  # df_part, df_jug,
        # df_part, df_jug = prepare.clean_data(df_part, df_jug, export=export)
        # df = prepare.integrate_data(df_part, df_jug, export=export)
        # df = prepare.construct_data(df, N_ULT_PART=N_ULT_PART, export=export)
        df = prepare.select_data(thr_corr=thr_corr, perc_fs=perc_fs, treat_nan=treat_nan, export=export)

    if modeling is True:

        # Hiperparametros
        porc_corte = 0.8 # Con 0.005 (usa solo 20 registros para entrenar) obtiene una precision del 65% y un roi del 100%...  --> ENCONTRÉ FALLA
        best_params = True  # True para hacer GridSearch para buscar los mejeres hiperparametros.
        k = 10  # Numero de folds
        l_modelos = [DecisionTreeClassifier(max_depth=30),
                     RandomForestClassifier(n_estimators=200, max_depth = None, random_state=42),  # Tarda cdo hago best_params y k=10
                     xgb.XGBClassifier(n_estimators=50, objective='multi:softmax', num_class=3, max_depth=20),  # num_class = len(y.unique()) Depende del numero de clases...
                     LogisticRegression(multi_class='multinomial', penalty='l2', C=0.1, solver='lbfgs', max_iter=500),
                     SVC(kernel='rbf', decision_function_shape='ovo'),  # --> tarda mas de 1 hora
                     MLPClassifier(hidden_layer_sizes=128, activation='tanh', solver='adam', learning_rate='invscaling', max_iter=300),
                     GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=7)
                     # self.red_neuronal(n_folds_cv=10, n_epochs=1000, batches=256),
                     ]
        print(" Modeling ".center(120, "#"))

        # Analizo los datos
        df_train, df_test = modeler.generate_test_design(porc_corte=porc_corte, export=export)
        best_model = modeler.select_best_model(df_train, l_modelos, best_params, k, export=export)
        modeler.assess_model(best_model, df_test)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()