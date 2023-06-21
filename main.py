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
from sklearn.model_selection import train_test_split
from modeling import generate_test_design
from sklearn.utils import shuffle
# Build model
from modeling import build_model
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
import lightgbm as lgb  # Gradient Boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
# Assess model
from modeling.asses_model import calculate_roi, confusion_matrix
from sklearn.metrics import accuracy_score
import pickle


class DataPreparation:  # 17.4 min

    def __init__(self, var_resp, pais):
        self.var_resp = var_resp
        self.pais = pais

    def format_data(self, df_part, df_jug, export=True):  # 0.0 min
        """
        Arreglo el data type de algunas variables.

        :param df_part: Dataframe de los datos de los partidos. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormateando los datos...")

        # Entidad partido: fecha, posesion y es_copa
        df_part['fecha'] = pd.to_datetime(df_part['fecha'], format='%d.%m.%Y %H:%M')  # ya lo voy a extraer datetime... # Fundamental para poder ordenar el df por 'fecha'
        df_part = format_data.convert_posesion_to_int(df_part)  # Podria usar la limpieza de punct de tp y luego convertir a int64 pero as al pedo
        df_part['es_copa'] = df_part['es_copa'].replace(True, 1).replace(False, 0)  # ya no va a ser necesario...

        # Entidad jugador: fecha y valor de mercado
        df_jug['fecha'] = pd.to_datetime(df_jug['fecha'], format='%b %d, %Y')
        df_jug = format_data.convert_valor_mercado_to_int(df_jug)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_part_formated.xlsx', index=False)
            df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_jug_formated.xlsx', index=False)

        return df_part, df_jug

    def clean_data(self, df_part, df_jug, export=True):  # 0.0 min
        """
        Limpia los datos de un dataframe.

        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe limpiado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe limpiado. (DataFrame)
        """
        start = time.time()
        print("\nLimpiando los datos...")

        # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
        df_part = clean_data.prepare_text_columns(df_part, l_col_to_except=['id', 'temporada'])  # df_part = clean_data.prepare_text_columns(df_part)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales
        df_jug = clean_data.prepare_text_columns(df_jug, l_col_to_except=['id'])  # df_jug = clean_data.prepare_text_columns(df_jug)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

        # Remuevo strings adicionales en los nombres de los equipos
        df_part = clean_data.clean_teams_names(df_part)

        # Elimino filas con alto porcentaje de NaN values --> partidos con pocos datos... que no sirve integrar ni para construir
        # df_part = clean_data.eliminar_filas_nan(df_part, umbral=0.5) # Lo dejo aqui? seria para evitar tener un df enorme en integrate y construct...

        end = time.time()
        print(f"Limpieza de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_part_cleaned.xlsx', index=False)
            df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_jug_cleaned.xlsx', index=False)

        return df_part,df_jug

    def integrate_data(self, df_part, df_jug, export=True):  # 13.3 min (sin copa arg y otras comp)
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        start = time.time()
        print("\nIntegrando los datos...")

        # Integro entidad partido y jugador
        df_integrated = integrate_data.player_data_in_match(df_part, df_jug)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_integrated.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_integrated.xlsx', index=False)

        return df_integrated

    def construct_data(self, df, N_ULT_PART=5, export=True):  # 2.7 minutos
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        start = time.time()
        print("\nConstruyendo nuevos datos...")

        # Construyo variable respuesta: "equipo_gandor"
        df = construct_data.determinar_equipo_ganador(df)

        # Construyo variables historicas
        l_estad_part = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
        df = construct_data.historial_entre_si_segun_localia(df, n_ult_part=int(N_ULT_PART/2))
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, l_var=l_estad_part)  # Estadisticas del partido
        df = construct_data.rendimiento_equipo(df, n_ult_part=N_ULT_PART, peso_puntos=0.6)  # Segun diferencia de gol y puntos
        # df = construct_data.n_dias_ult_partido(df)  # Numero de dias desde ultimo partido --> requiere 1) partidos de copa 2) eliminacion de valores atipicos

        # Construyo variables de diferencias para las variables promedio de los jugadores
        df = construct_data.calculate_dif_col_jugadores(df)

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_constructed.xlsx', index=False)

        return df

    def select_data(self, df, thr_corr=0.6, thr_nan_col=0.2 , thr_fs=0.5, export=True):  # 1.3 minutos # Chequear cambios
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        warnings.filterwarnings('ignore')
        start = time.time()
        print("\nSeleccionado datos...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

        # Elimino filas y columnas con alto porcentaje de NaN values
        df = clean_data.eliminar_filas_nan(df, umbral=0.5)  # 1º elimino registros con muchos nan --> puesto que quiero preservar variables antes que registros
        if thr_nan_col is not None:
            df = clean_data.eliminar_columnas_nan(df, umbral=thr_nan_col)  # 2º elimino columnas con mucho NaN # ojo que asi puede borrar odds

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df)
        df_etiquetas.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_etiquetas.xlsx')

        # Elimino variables altamente correlacionadas
        l_columnas_a_eliminar = select_data.eliminar_columnas_correlacionadas(df, self.var_resp, thr_corr)
        df = df.drop(l_columnas_a_eliminar, axis=1)

        # Selecciono las variables mas importantes (feature selection)
        l_selected_features = select_data.select_best_features(df, self.var_resp, thr_fs=thr_fs, graf=export)
        columns_to_select = l_selected_features + ['odds_loc', 'odds_emp', 'odds_vis', self.var_resp]
        df = df.filter(columns_to_select)

        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_selected.xlsx', index=False)

        return df

class Modeling:

    def __init__(self, var_resp, var_pred, pais):
        if not isinstance(var_resp, str) or not isinstance(var_pred, str):
            raise TypeError("Los parámetros var_resp y var_pred deben ser cadenas de texto.")
        if not isinstance(pais, str):
            raise TypeError("El parámetro pais debe ser una cadena de texto.")

        self.var_resp = var_resp
        self.var_pred = var_pred
        self.pais = pais

    def generate_test_design(self, df, bal_type, test_val_size=0.2, test_size=0.5, treat_nan='drop', export=True):
        """
        Separa conjuntos de datos en train, validacion y test, balancea las clases del dataset y elimina los NaN values.

        :param bal_type: Tipo de balanceo de clases a realizar.(string)
        :param test_val_size: Porcentaje del total de datos destinado a validacion y test. (float)
        :param test_size: # Porcentaje de test_val_size destinado a test. (float)
        :param treat_nan: Tipo de tratamiento de NaN values.(string)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe de entrenamiento y de testeo balanceados (DataFrame)
        """
        print("\nGenerando datasets de entrenamiento y testeo...")

        # Eliminacion de NaN values --> conviene al principio... para separar en las proporciones que digo...
        if treat_nan == 'drop':
            df = clean_data.eliminar_filas_nan(df, umbral=0)  # Eliminar filas con valores nulos en X e y   # Opcion 1: Elimino filas con al menos un NaN value teniendo en cuenta solo las columnas seleccionadas

        # Separo en X e y
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]  # Separar en X e y

        # Separo conjunto de datos en train, validation y test
        X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_val_size,random_state=42, shuffle=True)  # Divido todos los  datos en train y validacion + prueba
        X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=42, shuffle=True) # Divido validacion + prueba en validacion y prueba

        # Elimino variables odds del dataset de entrenamiento y validacion (de test no porque necesito calcular roi)
        X_train = X_train.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
        X_val = X_val.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)

        print(f'Train: {X_train.shape} {y_train.shape}')
        print(f'Val: {X_val.shape} {y_val.shape}')
        print(f'Test: {X_test.shape} {y_test.shape}')

        # Relleno nan --> solo en train... para no sesgar df_test ni df_val y asi evitar overfitting
        if treat_nan == "ml" or treat_nan == 'mode':
            X_train, y_train = clean_data.fill_nan_values(X_train, y_train, type=treat_nan)  #  Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las precisiones casi siempre seran mayores que dropna() en train y test, lo que cuenta es la precision en next_matches o en un dataset que no haya sido filleado...
            print(f"Se realizo el rellenado de NaN values. Shape X_train luego de rellenado: {X_train.shape}")

            # Elimino NaN de df_val y df_test para evitar "ValueError: Input X contains NaN."
            df_val = pd.concat([X_val, y_val], axis=1)
            df_val = df_val.dropna()
            X_val, y_val = df_val.drop(self.var_resp, axis=1), df_val[self.var_resp]  # Separar en X e y
            print(f"Se elimino NaN values en validacion. Shape X_val: {X_val.shape}")

            df_test = pd.concat([X_test, y_test], axis=1)
            df_test = df_test.dropna()
            X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]  # Separar en X e y
            print(f"Se elimino NaN values en test. Shape X_test: {X_test.shape}")

        # Balanceo el dataset de entrenamiento --> solo en train... para no sesgar df_test ni df_val y asi evitar overfitting
        if bal_type is not None:
            X_train, y_train = generate_test_design.balance_dataset(X_train, y_train, type=bal_type)
            print(f"Shape X_train luego de balanceo: {X_train.shape}")

        # Shuffle y borro index --> fundamental para evitar problemas en CV en la division de los folds (si devuelve el df ordenado por clase, fallara el cv)
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train = df_train.sample(frac=1).reset_index(drop=True)  # Para evitar que queden misma clase en un fold de CV?
        X_train, y_train = df_train.drop(self.var_resp, axis=1), df_train[self.var_resp]

        if export:
            X_train.to_excel(f'./modeling/data/{self.pais}/X_train.xlsx', index=False)
            X_val.to_excel(f'./modeling/data/{self.pais}/X_val.xlsx', index=False)
            X_test.to_excel(f'./modeling/data/{self.pais}/X_test.xlsx', index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, model, X_val, y_val, X_train, y_train, k=10):
        """
        Selecciona el mejor modelo a partir de la precision.

        :param model: Modelo de Machine Learning. (sklearn.ensemble)
        :param X_val: Dataframe de validacion con variables predictoras. (DataFrame)
        :param y_val: Dataframe de validacion solo con variable respuesta. (DataFrame)
        :param X_train: Dataframe de entrenamiento con variables predictoras.  (DataFrame)
        :param y_train: Dataframe de entrenamiento solo con variable respuesta. (DataFrame)
        :param k: Numero de folds. (int)
        :return: Mejor modelo. (sklearn.ensemble?)
        """
        # Definicion de variables
        warnings.filterwarnings("ignore")
        model_name = str(model)[:str(model).find('(')]  # Defino el nombre del modelo (e.g. "RandomForest")

        # Find best hiperparameters
        print(f" Modelo: {model_name} ".center(120, '-'))
        # print("Buscando mejores hiperparametros...")
        model_best_params = build_model.select_best_hiperparameters(model, X_val, y_val, k)

        # Entreno el modelo
        # print("Entrenando modelo con mejores hiperparametros...")
        model_best_params.fit(X_train, y_train)

        # Evaluo el modelo con Cross Validation
        # print("Evaluo rendimiento del modelo con Cross Validation...")
        cv_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
        print(f"\nPrecision promedio de validación cruzada: {cv_accuracy:.1f}%")
        return model_name, model_best_params, cv_accuracy

    def assess_model(self, model, X_test, y_test, export=True):
        """
        Evalúa un modelo de machine learning utilizando datos de prueba y calcula métricas de desempeño.

        :param model: Modelo de Machine Learning entrenado. (sklearn.ensemble)
        :param X_test: Dataframe de prueba con variables predictoras. (DataFrame)
        :param y_test: Dataframe de prueba solo con variable respuesta. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el DataFrame seleccionado. True para exportar, False
        de lo contrario. (bool)
        :return: Precisión del modelo y ROI en el conjunto de prueba. (int) y (float)
        """
        # print("Evaluando modelo con datos de prueba...")
        df_etiquetas = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{self.pais}/df_etiquetas.xlsx')

        # Quito cuotas de casas de apuestas y variable respuesta de df_test
        X_test_without_odds = X_test.copy().drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)

        # Predecir las etiquetas para los datos de prueba
        y_pred = model.predict(X_test_without_odds)  # es un numpy array

        # Calculo precision
        test_accuracy = accuracy_score(y_test, y_pred) * 100

        # Calculo roi y matriz de confusion
        # Asignar las predicciones a una nueva columna en df_test (para poder calcular ROI)
        df_results = X_test.copy().loc[:, ['odds_loc', 'odds_emp', 'odds_vis']]
        df_results[self.var_resp] = y_test
        df_results[self.var_pred] = y_pred

        # Convierto variable respuesta y variable predicha en etiqueta
        df_results_etiquetas = format_data.revert_columns_from_int(df_results, df_etiquetas, columns=[self.var_resp, self.var_pred])

        # Calculo roi
        roi = calculate_roi(df_results_etiquetas, self.var_resp, self.var_pred) * 100
        print(f"Precision promedio de prueba: {test_accuracy:.1f}%")
        print(f"ROI promedio de prueba: {roi:.1f}%")

        if export:
            confusion_matrix(df_results_etiquetas[self.var_resp], df_results_etiquetas[self.var_pred])  # podria exportar el archivo? para evitar tener que cerrarla para que continue el programa
            df_results.to_excel(f'./modeling/data/{self.pais}/df_results.xlsx')

        return test_accuracy, roi

def main():

    # Definicion de variables
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    pais = "inglaterra"  # tiene sentido solo si hago un modelo por pais? y si no? # Ver si puedo evitar el pais como argumento en train model por tener que meterlo en el calculo del roi en df_etiquetas...
    export = True

    data_unders = False
    if data_unders:

        print(" Data understanding ".center(120, "#"))

        # Collect initial data
        print(" Recolectando datos... ")
        df_part = collect_initial_data.extract_partidos_flashscore(l_paises=[pais])
        df_jug = collect_initial_data.extract_jugadores_sofifa(l_paises=[pais])
        print(f"Dataframe partido:\n{df_part} \nDataframe jugadores:\n{df_jug}")

        # Describe data (quiero describir los datos igual aunque no los extraiga...)
        print(" Describiendo datos... ")
        getting_to_know_data(df_part)
        getting_to_know_data(df_jug)

    data_prep = True
    if data_prep:

        # Definicion de variables
        print(" Data preparation ".center(120, "#"))
        dp = DataPreparation(var_resp, pais) # Creo objeto de clase DataPreparation

        # Hiperparametros
        N_ULT_PART = 5  # Numero de partidos a tener en cuenta para variables historicas como posesion en ult partidos
        thr_corr = 0.7  # Correlacion umbral para la eliminacion de variables altamente correlacionadas  # Con 0.6 : {'dif_valor_sup', 'dif_pases_comp_segun_ult_part', 'dif_rat_sup', 'dif_valor_aus', 'dif_pases_segun_ult_part', 'dif_gol', 'dif_valor_tit', 'dif_remates_segun_ult_part', 'dif_ataques_segun_ult_part'}
        thr_fs = 0.3  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)
        thr_nan_col = 0.2

        # Levanto datasets
        df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_partido.xlsx')
        df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_jugadores.xlsx')
        # df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_constructed.xlsx')

        # Preparo el dataset para el analisis
        df_part, df_jug = dp.format_data(df_part, df_jug)  # df_part, df_jug,
        df_part, df_jug = dp.clean_data(df_part, df_jug)
        df = dp.integrate_data(df_part, df_jug)
        # df = dp.construct_data(df, N_ULT_PART=N_ULT_PART)
        # df = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, thr_nan_col=thr_nan_col, export=False)

    modeling = False
    if modeling:
        # Definicion de variables
        print(" Modeling ".center(120, "#"))
        mo = Modeling(var_resp, var_pred, pais)  # Creo objeto de clase Modeling
        df_models = pd.DataFrame(columns=['model_name', 'model_trained', 'train_cv_accuracy', 'test_accuracy', 'test_roi'])  # Datos del modelo y su precision y roi
        l_modelos = [DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier(), LogisticRegression(),
                     SVC(), MLPClassifier(), GradientBoostingClassifier()]

        # Hiperparametros
        test_val_size = 0.25  # Porcentaje del total de datos destinado a validacion y test.
        test_size = 0.5  # Porcentaje de test_val_size destinado a test.
        bal_type = 'over'  # Tipo de balanceo a realizar [None, 'over', 'under']
        treat_nan = 'drop'  # Eliminacion de nan values [drop, mode, ml]
        k = 5  # Numero de folds para seleccionar best parameters y para entrenar modelo

        # Levanto dataset para prueba
        df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_selected.xlsx')

        # General el diseño de la prueba
        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df, bal_type, test_val_size, test_size, treat_nan=treat_nan)

        # Por modelo
        for modelo in l_modelos:

            # Entreno modelo y evaluo su rendimiento
            model_name, model_best_params, cv_accuracy = mo.build_model(modelo, X_val, y_val, X_train, y_train, k)
            test_accuracy, test_roi = mo.assess_model(model_best_params, X_test, y_test)

            # Guardo modelo
            df_models.loc[len(df_models)] = [model_name, model_best_params, cv_accuracy, test_accuracy, test_roi]

        # Selecciono el mejor modelo
        idx = df_models[df_models['test_accuracy'] == max(df_models['test_accuracy'])].index[0]
        best_model = df_models.loc[idx, 'model_trained']
        best_model_train_prec = df_models.loc[idx, 'train_cv_accuracy']
        best_model_test_prec = df_models.loc[idx, 'test_accuracy']
        best_model_test_roi = df_models.loc[idx, 'test_roi']
        print(f"\nEl mejor modelo es: {best_model} con: \n\t- Train Precision: {best_model_train_prec:.1f}% "
              f"\n\t- Test Precision: {best_model_test_prec:.0f}% \n\t- Test ROI: {best_model_test_roi:.1f}%")

        if export:
            df_models.to_excel(f'./modeling/data/{pais}/df_modelos.xlsx')
            pickle.dump(best_model, open(f"./modeling/data/{pais}/modelo.pkl", "wb"))

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()