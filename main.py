# Importo librerias
import pandas as pd
import time
import warnings
import datetime
# Data understanding
from p2_data_understanding import scraper_whoscored, scraper_flashscore, describe_data
from dspy.data_understanding.describe_data import getting_to_know_data
# Data preparation
from sklearn.preprocessing import StandardScaler
from p3_data_preparation import format_data, integrate_data, construct_data, select_data, clean_data
# Modeling
# Generate test design
from sklearn.model_selection import train_test_split
from p4_modeling import generate_test_design
from sklearn.utils import shuffle
# Build model
from p4_modeling import build_model
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
import lightgbm as lgb  # Gradient Boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
# Assess model
from p4_modeling.asses_model import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pickle


class DataUnderstanding:

    def __init__(self, pais):
        self.pais = pais

    def collect_initial_data(self):

        print(" Recolectando datos... ")

        # Extraigo partidos
        df_part_ws, df_jug_part = scraper_whoscored.extract_partidos_whoscored(self.pais)
        df_part_ws.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{self.pais}/df_part.xlsx', index=False)
        df_jug_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{self.pais}/df_jug_part.xlsx', index=False)

        # Extraigo datos de jugadores
        df_jug = scraper_whoscored.extract_player_data(df_jug_part)
        df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{self.pais}/df_jug.xlsx', index=False)

        # Extraigo partidos de Flashscore
        df_part_fs = scraper_flashscore.extract_partidos_flashscore(self.pais)
        df_part_fs.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{self.pais}/df_part_fs.xlsx', index=False)
        return df_part_ws, df_part_fs, df_jug_part, df_jug

    def describe_data(self, df_part, df_jug_part, df_jug):  # Agregar df_jug a descripcion y df_part de flashscore?

        print(" Describiendo datos... ")
        getting_to_know_data(df_part)
        getting_to_know_data(df_jug_part)
        getting_to_know_data(df_jug)

        # Verifico unicidad de registros segun campos id
        describe_data.verificar_unicidad_registros(df_part, columns_id='id_part')
        describe_data.verificar_unicidad_registros(df_jug_part, columns_id=['id_jug', 'id_part'])
        describe_data.verificar_unicidad_registros(df_jug_part, columns_id=['id_jug'])

        # Verifico consistencia en campos que relacionan entidades
        describe_data.verificar_relacion_entidades(df_part, df_jug_part)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de jugadores: verificar_relacion_entidades(df_jug_part, df_part)


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

        # Entidad partido WhoScored: fecha, resultados de medio tiempo y final
        df_part['fecha'] = pd.to_datetime(df_part['fecha'] + ' ' + df_part['hora'], format='%a, %d-%b-%y %H:%M')
        df_part['fecha'] = df_part['fecha'] - datetime.timedelta(hours=4)  # Resto 4 horas a la columna 'fecha' para que este en horario argentino
        df_part[['ht_goles_loc', 'ht_goles_vis']] = df_part['ht_result'].str.split(' : ', expand=True)  # Separar ht_result en ht_goles_loc y ht_goles_vis
        df_part[['goles_loc', 'goles_vis']] = df_part['ft_result'].str.split(' : ', expand=True)  # Separar ft_result en goles_loc y goles_vis
        df_part = df_part.drop(['hora', 'ht_result', 'ft_result'], axis=1)

        # Entidad jugador: fecha
        df_jug['fecha_nac'] = pd.to_datetime(df_jug['fecha_nac'], format='%d-%m-%Y')

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_part_formated.xlsx', index=False)
            df_jug.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_jug_formated.xlsx', index=False)

        return df_part, df_jug

    def integrate_data(self, df_part, df_jug_part, df_jug, fill_data_with_flashcore=False, n_dias_player_data=30, export=True):  # 13.3 min (sin copa arg y otras comp)
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        start = time.time()
        print("\nIntegrando los datos...")

        # Integro Flashscore a Whoscored para rellenar estadisticas en partidos de Whoscored
        if fill_data_with_flashcore:
            df_part_flash = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{self.pais}/df_part_fs.xlsx')
            df_part = integrate_data.fill_whoscored_with_flashscore(df_part, df_part_flash)

        # Integro df_jug a df_jug_part
        df_jug_part = integrate_data.map_player_entities(df_jug, df_jug_part)

        # Calculo edad y minutos jugados por jugador en cada partido
        df_jug_part = construct_data.add_fecha(df_part, df_jug_part)
        df_jug_part = construct_data.determine_edad(df_jug_part)
        df_jug_part = construct_data.determine_min_played(df_jug_part)
        df_jug_part = construct_data.determine_player_var_en_ult_partidos(df_jug_part, 'min_played', n_dias=n_dias_player_data, tipo='sum')
        df_jug_part = construct_data.determine_player_var_en_ult_partidos(df_jug_part, 'rating', n_dias=n_dias_player_data, tipo='mean_pond', var_pond='min_played')

        # Integro df_jug_part_integ (df_jug_part + df_jug) a df_part
        df = integrate_data.map_player_to_part(df_jug_part, df_part)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_integrated.xlsx', index=False)

        return df

    def construct_data(self, df, n_dias, n_anios_historial, export=True):  # 2.7 minutos
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        start = time.time()
        print("\nConstruyendo nuevos datos...")

        # Definicion de variables
        n_dias_loc = n_dias * 2  # 30 es como N_ULT_PART igual a 2
        n_anios_historial_loc = n_anios_historial * 2
        l_estadisticas = ['goles', 'puntos', 'posesion', 'remates', 'remates_a_puerta', 'remates_palos',
                          'porc_pases_comp', 'pases', 'pases_clave', 'amagues', 'duelos_aereos', 'tackles',
                          'intercepciones', 'corners', 'offsides', 'faltas']  # 'remates_fuera', 'remates_block', 'pases_comp'

        # Construyo variables: "equipo_gandor" y puntos obtenidos
        df = construct_data.determinar_equipo_ganador(df)
        df = construct_data.determinar_puntos(df)

        # Variables historicas
        df = construct_data.historial_entre_si_segun_fecha(df, n_anios=n_anios_historial)
        df = construct_data.historial_entre_si_localia_segun_fecha(df, n_anios=n_anios_historial_loc)

        # Por estadistica del partido
        for var in l_estadisticas:
            # Determinar la diferencia de la estadistica entre equipo local y visitante de cada partido
            df[f'dif_{var}'] = df[f'{var}_loc'] - df[f'{var}_vis']  # (e.g. dif_goles = goles_loc - goles_vis)
            df.drop([f'{var}_loc', f'{var}_vis'], axis=1)  # (e.g. borro goles_loc y goles_vis)

            # Determinar para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
            df = construct_data.determine_prom_en_ult_partidos(df, n_dias=n_dias, variable=var, tipo='mean')
            df = construct_data.determine_prom_en_ult_partidos_localia(df, n_dias=n_dias_loc, variable=var, tipo='mean')
            df.drop([f'dif_{var}'], axis=1)

            # Determinar la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_loc y prom_dif_goles_vis)
            df[f'dif_prom_ult_part_dif_{var}'] = df[f'prom_ult_part_dif_{var}_loc'] - df[f'prom_ult_part_dif_{var}_vis']
            df[f'dif_prom_ult_part_segun_localia_dif_{var}'] = df[f'prom_ult_part_segun_localia_dif_{var}_loc'] - df[f'prom_ult_part_segun_localia_dif_{var}_vis']
            df = df.drop(columns=[f'prom_ult_part_dif_{var}_loc', f'prom_ult_part_dif_{var}_vis', f'prom_ult_part_segun_localia_dif_{var}_loc', f'prom_ult_part_segun_localia_dif_{var}_vis'], axis=1)

        # Construyo variables de diferencias para las variables promedio de los jugadores
        df = construct_data.calculate_dif_col_jugadores(df)

        # Elimino variables que no construire
        df = df.drop(columns=['remates_fuera_loc', 'remates_fuera_vis', f'remates_block_loc', f'remates_block_vis', 'pases_comp_loc', 'pases_comp_vis'], axis=1)

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_constructed.xlsx', index=False)

        return df

    def clean_data(self, df, thr_nan_col=None, export=True):  # 0.0 min
        """
        Limpia los datos de un dataframe.

        :param df: Dataframe de los datos de los partidos. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe limpiado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe limpiado. (DataFrame)
        """
        start = time.time()
        print("\nLimpiando los datos...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df = df.drop(['id_part', 'pais', 'competicion', 'temporada', 'fecha', 'cancha'], axis=1)  # elimino aca por si thr_nan_col elimina una de ellas antes y por ende falla el programa

        # Eliminacion de NaN values
        # Elimino filas y columnas con alto porcentaje de NaN values
        largo_inicial = len(df)
        df = df.dropna(subset=['dif_remates_segun_ult_part'], how='any')
        print(f"Se eliminó el {(largo_inicial - len(df)) / largo_inicial * 100:.0f}% de filas, quedan {len(df)} filas.")

        if thr_nan_col is not None:
            df = clean_data.eliminar_columnas_nan(df, umbral=thr_nan_col)  # 2º elimino columnas con mucho NaN # ojo que asi puede borrar odds

        # Verificar que no haya outliers
        # ...

        end = time.time()
        print(f"Limpieza de datos en {(end - start) / 60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_part_cleaned.xlsx',index=False)

        return df

    def select_data(self, df, thr_corr=None, thr_fs=None, export=True):  # 1.3 minutos # Chequear cambios
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        warnings.filterwarnings('ignore')
        start = time.time()
        print("\nSeleccionado datos...")

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df)
        df_etiquetas.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_etiquetas.xlsx')

        # Elimino variables altamente correlacionadas
        if thr_corr is not None:
            l_columnas_a_eliminar = select_data.eliminar_columnas_correlacionadas(df, self.var_resp, thr_corr)
            df = df.drop(l_columnas_a_eliminar, axis=1)

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            l_not_important_features = select_data.select_best_features(df, self.var_resp, thr_fs, graf=export)
            df = df.drop(l_not_important_features, axis=1)

        end = time.time()
        print(f"Las siguientes {len(df.drop(self.var_resp, axis=1).columns)} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_selected.xlsx', index=False)

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

    def generate_test_design(self, df, bal_type, test_val_size=0.2, test_size=0.5, fill_na=None, export=True):
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

        # Elimino filas con al menos un NaN puesto que al modelo no le pueden ingresar NaN values
        if fill_na is None:
            df = clean_data.eliminar_filas_nan(df, umbral=0)  # df = df.dropna()

        # Separo en X e y
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

        # Separo conjunto de datos en train, validation y test
        X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_val_size,random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=42, shuffle=True)
        print(f'Train: {X_train.shape} {y_train.shape}')
        print(f'Val: {X_val.shape} {y_val.shape}')
        print(f'Test: {X_test.shape} {y_test.shape}')

        # Relleno nan en el dataset de entrenamiento
        if fill_na is not None:
            X_train, y_train = clean_data.fill_nan_values(X_train, y_train, type=fill_na)  #  Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las precisiones casi siempre seran mayores que dropna() en train y test, lo que cuenta es la precision en next_matches o en un dataset que no haya sido filleado...
            print(f"Se realizó el rellenado de NaN values. Shape X_train luego de rellenado: {X_train.shape}")

            # Elimino NaN de df_val y df_test para evitar "ValueError: Input X contains NaN."
            df_val = pd.concat([X_val, y_val], axis=1).dropna()
            X_val, y_val = df_val.drop(self.var_resp, axis=1), df_val[self.var_resp]  # Separar en X e y
            print(f"Se elimino NaN values en validacion. Shape X_val: {X_val.shape}")

            df_test = pd.concat([X_test, y_test], axis=1).dropna()
            X_test, y_test = df_test.drop(self.var_resp, axis=1), df_test[self.var_resp]  # Separar en X e y
            print(f"Se elimino NaN values en test. Shape X_test: {X_test.shape}")

        # Balanceo el dataset de entrenamiento
        if bal_type is not None:
            X_train, y_train = generate_test_design.balance_dataset(X_train, y_train, type=bal_type)
            print(f"Shape X_train luego de balanceo: {X_train.shape}")

        # Shuffle el dataset de entrenamiento
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        X_train, y_train = df_train.drop(self.var_resp, axis=1), df_train[self.var_resp]

        if export:
            X_train.to_excel(f'./p4_modeling/data/{self.pais}/X_train.xlsx', index=False)
            X_val.to_excel(f'./p4_modeling/data/{self.pais}/X_val.xlsx', index=False)
            X_test.to_excel(f'./p4_modeling/data/{self.pais}/X_test.xlsx', index=False)

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
        model_best_params, best_params = build_model.select_best_hiperparameters(model, X_val, y_val, k)

        # Entreno el modelo
        # print("Entrenando modelo con mejores hiperparametros...")
        model_best_params.fit(X_train, y_train)

        # Evaluo el modelo con Cross Validation
        # print("Evaluo rendimiento del modelo con Cross Validation...")
        cv_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
        print(f"\nPrecision promedio de validación cruzada: {cv_accuracy:.1f}%")

        return model_name, model_best_params, best_params, cv_accuracy

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
        df_etiquetas = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{self.pais}/df_etiquetas.xlsx')

        # Predecir las etiquetas para los datos de prueba
        y_pred = model.predict(X_test)  # es un numpy array

        # Calculo metricas
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100  # recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro') * 100  # f1 = f1_score(y_test, y_pred)
        print(f"Precision de test: {test_accuracy:.1f}%")
        print(f"Recall de prueba: {recall:.1f}%")
        print(f"F1-score de prueba: {f1:.1f}%")

        # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
        # Asignar las predicciones a una nueva columna en df_test (para poder calcular ROI)
        df_results = pd.DataFrame()
        df_results[self.var_resp] = y_test  # Agrego y_real
        df_results[self.var_pred] = y_pred  # Agrego y_pred

        # Convierto variable respuesta y variable predicha en etiqueta
        df_results_etiquetas = format_data.revert_columns_from_int(df_results, df_etiquetas, columns=[self.var_resp, self.var_pred])

        if export:
            # confusion_matrix(df_results_etiquetas[self.var_resp], df_results_etiquetas[self.var_pred])  # podria exportar el archivo? para evitar tener que cerrarla para que continue el programa
            df_results.to_excel(f'./p4_modeling/data/{self.pais}/df_results.xlsx')

        # return test_accuracy, recall, f1, roi
        return test_accuracy, recall, f1

def main():

    # Definicion de variables
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    pais = "argentina"  # tiene sentido solo si hago un modelo por pais? y si no? # Ver si puedo evitar el pais como argumento en train model por tener que meterlo en el calculo del roi en df_etiquetas...
    pais = "argentina_south_america"
    export = True

    # Procesamiento
    data_unders = False
    data_prep = True
    modeling = False

    if data_unders:

        print(" Data understanding ".center(120, "#"))
        du = DataUnderstanding(pais) # Creo objeto de clase DataPreparation

        df_part_ws, df_part_fs, df_jug_part, df_jug = du.collect_initial_data()
        du.describe_data(df_part_ws, df_jug_part, df_jug)
        print(f"Dataframe partido:\n{df_part_ws} \nDataframe jugadores:\n{df_jug}")

    if data_prep:

        # Definicion de variables
        print(" Data preparation ".center(120, "#"))
        dp = DataPreparation(var_resp, pais) # Creo objeto de clase DataPreparation

        # Hiperparametros
        fill_with_fs = True  # Relleno datos con flashscore {True, False}
        n_dias_player_data = 365  # Numero de dias para tener en cuenta en construccion de variables historicas para jugadores
        n_dias = 30  # 30 es como N_ULT_PART igual a 5...
        n_anios_historial = 2
        thr_nan_col = 0.5
        thr_corr = 0.7  # Correlacion umbral para la eliminacion de variables altamente correlacionadas
        thr_fs = 0.3  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)

        # Levanto datasets
        df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_part.xlsx')
        df_jug_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_jug_part.xlsx')
        df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_jug.xlsx')
        # df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_constructed.xlsx', index_col=0)

        # Preparo el dataset para el analisis
        df_part, df_jug = dp.format_data(df_part, df_jug)
        df = dp.integrate_data(df_part, df_jug_part, df_jug, fill_data_with_flashcore=fill_with_fs, n_dias_player_data=n_dias_player_data)
        df = dp.construct_data(df, n_dias=n_dias, n_anios_historial=n_anios_historial)
        df = dp.clean_data(df, thr_nan_col=thr_nan_col)
        df = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, export=True)

    if modeling:
        # Definicion de variables
        print(" Modeling ".center(120, "#"))
        mo = Modeling(var_resp, var_pred, pais)  # Creo objeto de clase Modeling
        df_models = pd.DataFrame(columns=['model_name', 'model_trained', 'train_cv_accuracy', 'test_accuracy', 'test_recall', 'test_f1_score'])  # Datos del modelo y su precision y roi
        l_modelos = [DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier(), LogisticRegression(),
                     SVC(), MLPClassifier(), GradientBoostingClassifier()]

        # Hiperparametros
        test_val_size = 0.25  # Porcentaje del total de datos destinado a validacion y test.
        test_size = 0.5  # Porcentaje de test_val_size destinado a test.
        bal_type = None # Tipo de balanceo a realizar [None, 'over', 'under']
        fill_na = 'ml'  # Relleno de nan values [None, mode, ml]
        k = 5  # Numero de folds para seleccionar best parameters y para entrenar modelo

        # Levanto dataset para prueba
        df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_selected.xlsx')

        # General el diseño de la prueba
        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df, bal_type, test_val_size, test_size, fill_na=fill_na)

        # Por modelo
        for modelo in l_modelos:

            # Entreno modelo y evaluo su rendimiento
            model_name, model_best_params, cv_accuracy = mo.build_model(modelo, X_val, y_val, X_train, y_train, k)
            # accuracy, recall, f1, roi = mo.assess_model(model_best_params, X_test, y_test)
            accuracy, recall, f1 = mo.assess_model(model_best_params, X_test, y_test)

            # Guardo modelo
            # df_models.loc[len(df_models)] = [model_name, model_best_params, cv_accuracy, accuracy, recall, f1, roi]
            df_models.loc[len(df_models)] = [model_name, model_best_params, cv_accuracy, accuracy, recall, f1]

        # Selecciono el mejor modelo
        idx = df_models['test_accuracy'].idxmax()  # idx = df_models[df_models['test_accuracy'] == max(df_models['test_accuracy'])].index[0]
        bm_name = df_models.loc[idx, 'model_name']
        bm_params = df_models.loc[idx, 'model_trained']
        bm_train_acc = df_models.loc[idx, 'train_cv_accuracy']
        bm_test_acc = df_models.loc[idx, 'test_accuracy']
        bm_test_rec = df_models.loc[idx, 'test_recall']
        bm_test_f1 = df_models.loc[idx, 'test_f1_score']
        # bm_test_roi = df_models.loc[idx, 'test_roi']
        print(f"\nEl mejor modelo es: {bm_name} con: "
              f"\n\t- Train Precision: {bm_train_acc:.1f}% "
              f"\n\t- Test Precision: {bm_test_acc:.1f}% "
              f"\n\t- Test recall: {bm_test_rec:.1f}%\n"
              f"\n\t- Test f1-score: {bm_test_f1:.1f}%\n"
              # f"\n\t- Test ROI: {bm_test_roi:.1f}%\n"
              )

        if export:
            df_models.to_excel(f'./p4_modeling/data/{pais}/df_modelos.xlsx')
            pickle.dump(bm_params, open(f"./p4_modeling/data/{pais}/modelo.pkl", "wb"))

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()