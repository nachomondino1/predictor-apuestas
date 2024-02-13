# Importo librerias
import pandas as pd
import os
## Data understanding
from p2_data_understanding.collect_initial_data import scraper_flashscore, scraper_sofifa
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
from sklearn.preprocessing import StandardScaler
## Modeling
from p4_modeling import generate_test_design, build_model, asses_model
### Generate test design
from random import randint
from sklearn.model_selection import train_test_split
### Build model
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
### Assess model
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pickle


class DataUnderstanding:

    def __init__(self, pais: str):
        self.pais = pais
        self.make_directories()

    def make_directories(self):
        ruta_base = f'./p2_data_understanding/data/{self.pais.lower()}/data_seg'
        l_directorios = [f'{ruta_base}/por_temporada/df_part/',
                         f'{ruta_base}/por_temporada/df_part_jug/',
                         f'{ruta_base}/por_temporada/df_jug/',
                         f'{ruta_base}/por_competicion/df_part/',
                         f'{ruta_base}/por_competicion/df_part_jug/',
                         f'{ruta_base}/por_competicion/df_jug/'
                         ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def collect_initial_data(self, export: bool =True):

        print(" Recolectando datos... ")
        # Definicion de variables
        df_part_concat, df_part_jug_concat, df_jug_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Selecciono competencias del pais
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencias.xlsx')
        df_comp_pais = df_comp[df_comp['pais_flashscore'] == self.pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
        print(f' PAIS: {self.pais} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_pais['competicion_flashscore']}")

        # POR COMPETICION
        for i, row in df_comp_pais.iterrows():

            # Extraigo partidos de Flashscore (df_part y df_part_jug)
            df_part, df_part_jug = scraper_flashscore.extract_data_flashscore(row['pais_flashscore'], row['competicion_flashscore'], row['is_cup'], n_temps_max=16, export=export)
            df_part_concat = pd.concat([df_part_concat, df_part], axis=0)
            df_part_jug_concat = pd.concat([df_part_jug_concat, df_part_jug], axis=0)

            # Si la competicion es una liga
            if row['is_cup'] == 0:

                # Extraigo datos de jugadores de Sofifa (df_jug)
                df_jug = scraper_sofifa.extract_jugadores_sofifa(row['pais_flashscore'], row['competicion_flashscore'], export=export)
                df_jug_concat = pd.concat([df_jug_concat, df_jug], axis=0)

        # Exporto datasets con competiciones del pais
        if export:
            df_part_concat.to_excel(f'./p2_data_understanding/data/{self.pais}/df_part.xlsx', index=False)
            df_part_jug_concat.to_excel(f'./p2_data_understanding/data/{self.pais}/df_part_jug.xlsx', index=False)
            df_jug_concat.to_excel(f'./p2_data_understanding/data/{self.pais}/df_jug.xlsx', index=False)

        return df_part_concat, df_part_jug_concat, df_jug_concat

    def describe_data(self, df_part: pd.DataFrame, df_part_jug: pd.DataFrame, df_jug: pd.DataFrame):

        print(" Describiendo datos... ")
        describe_data.getting_to_know_data(df_part)
        describe_data.getting_to_know_data(df_part_jug)
        describe_data.getting_to_know_data(df_jug)

        # Verifico unicidad de registros segun campos id
        describe_data.verificar_unicidad_registros(df_part, columns_id='id_part')

        # Verifico consistencia en campos que relacionan entidades
        describe_data.verificar_relacion_entidades(df_part, df_part_jug)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de jugadores: verificar_relacion_entidades(df_jug_part, df_part)


class DataPreparation:

    def __init__(self, var_resp: str, pais: str):
        self.var_resp = var_resp
        self.pais = pais
        self.make_directories()

    def make_directories(self):
        directorio = f'./p3_data_preparation/data/{self.pais.lower()}'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

    def format_data(self, df_part: pd.DataFrame, df_part_jug: pd.DataFrame, df_jug: pd.DataFrame, export: bool = True):
        """
        Arreglo el data type de algunas variables.

        :param df_part: Dataframe de los datos de los partidos. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormateando los datos...")

        # Dataframe partido
        ## Fecha
        df_part['fecha'] = pd.to_datetime(df_part['fecha'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        # df_part['fecha'] = df_part['fecha'] - datetime.timedelta(hours=4)  # Resto 4 horas a la columna 'fecha' para que este en horario argentino
        ## Posesion
        df_part = format_data.convert_posesion_to_int(df_part)
        ## Goles_loc y goles_vis  # Eliminar las filas cuyos goles no son float
        df_part = format_data.keep_goles_int(df_part)
        df_part_jug = df_part_jug[df_part_jug['id_part'].isin(df_part['id_part'])]
        df_part_jug = df_part_jug.reset_index(drop=True)
        df_part = df_part.reset_index(drop=True)

        # Dataframe jugador
        ## Fecha
        df_jug['fecha'] = pd.to_datetime(df_jug['fecha'], format='%b %d, %Y')
        ## Valor de mercado
        df_jug = format_data.convert_valor_mercado_to_int(df_jug)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel(f'./p3_data_preparation/data/{self.pais}/df_part_formated.xlsx', index=False)
            df_part_jug.to_excel(f'./p3_data_preparation/data/{self.pais}/df_part_jug_formated.xlsx', index=False)
            df_jug.to_excel(f'./p3_data_preparation/data/{self.pais}/df_jug_formated.xlsx', index=False)

        return df_part, df_part_jug, df_jug

    def clean_data(self, df_part: pd.DataFrame, df_part_jug: pd.DataFrame, df_jug: pd.DataFrame, thr_nan_col: float, export: bool = True):
        """
        Limpieza inicial de los dataframes
        :param df_part:
        :param df_part_jug:
        :param df_jug:
        :param export:
        :return:
        """
        start = time.time()
        print("\nFormateando los datos...")
        warnings.filterwarnings('ignore')

        # ELIMINACION DE FILAS NAN SEGUN % NAN, O BIEN, SELECCION DE DATOS SEGUN TEMPORADA....
        ## Elimino filas con alto porcentaje de NaN values
        n_filas = len(df_part)
        df_part_jug = df_part_jug.dropna(subset=['jug_tit_loc_11', 'jug_tit_vis_11'], how='any').reset_index(drop=True)
        df_part = df_part[df_part['id_part'].isin(df_part_jug['id_part'])].reset_index(drop=True)
        print(f"De las {n_filas} filas, se eliminan {(n_filas - len(df_part))} por no tener formaciones del "
              f"partido, quedan {len(df_part)} filas.")

        ## Elimino columnas con alto porcentaje de NaN values
        if thr_nan_col is not None:
            # df_prop_nan = df_part.isna().mean()
            # print("Porcentaje de Nan values por columna: \n", df_prop_nan)
            # df_prop_nan.to_excel(f'/Users/nachomondino/Desktop/df_prop_nan.xlsx')

            df_part = clean_data.eliminar_columnas_nan(df_part, porc_nan_max=thr_nan_col)  # 2º elimino columnas con mucho NaN # ojo que asi puede borrar odds
            df_part_jug = clean_data.eliminar_columnas_nan(df_part_jug, porc_nan_max=0.9)  # TEMPORAL? elimino columnas nan que quedan por el concat y luego la eliminacion de temporadas viejas

        # Dataframe partido:
        ## Equipo_loc y equipo_vis
        df_part = clean_data.prepare_text_columns(df_part, l_cols_to_process=['equipo_loc', 'equipo_vis'])  # Preparacion texto para facilitar construccion de datos bassado en equipos
        df_part = clean_data.clean_teams_names(df_part)  # Eliminar strings adicionales en nombres de equipos

        # Dataframe partido jugador:
        ## jug_tit_loc_1, jug_tit_loc2, ..., jug_aus_sup_18
        df_part_jug = clean_data.prepare_text_columns(df_part_jug, l_col_to_except=['id_part'])

        # Dataframe jugador:
        ## Nombre de jugador
        df_jug = clean_data.prepare_text_columns(df_jug, l_cols_to_process=['nombre'])  # Preaparo texto para integrar
        ## Valor de mercado
        scaler = StandardScaler()  # Crea un objeto StandardScaler
        df_jug['valor_mercado'] = scaler.fit_transform(df_jug['valor_mercado'].values.reshape(-1, 1))

        # Verificar que no haya outliers
        # algo (sacar de mi tesis)

        # Describo datos post limpieza
        du = DataUnderstanding(self.pais)
        du.describe_data(df_part, df_part_jug, df_jug)

        end = time.time()
        print(f"Limpieza inicial de datos en {(end - start) / 60:.1f} minutos")

        if export:
            df_part.to_excel(f'./p3_data_preparation/data/{self.pais}/df_part_form_clean.xlsx', index=False)
            df_part_jug.to_excel(f'./p3_data_preparation/data/{self.pais}/df_part_jug_form_clean.xlsx', index=False)
            df_jug.to_excel(f'./p3_data_preparation/data/{self.pais}/df_jug_form_clean.xlsx', index=False)

        return df_part, df_part_jug, df_jug

    def integrate_data(self, df_part: pd.DataFrame, df_part_jug: pd.DataFrame, df_jug: pd.DataFrame, export: bool = True):
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        :param df_part: Dataframe de los datos de los partidos.
        :param df_part_jug: Dataframe de los datos de los jugadores en cada partido.
        :param df_jug: Dataframe de los datos de los jugadores.
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de
        lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        start = time.time()
        print("\nIntegrando los datos...")

        # Obtengo listado unicos de jugadores en df_jug (Sofifa) y df_part_jug (Flashscore) para agilizar vinculacion
        df_part_jug_unique_players = unique_players_df_part_jug(df_part_jug)
        df_jug_unique_players = unique_players_df_jug(df_jug)

        # Vinculo con "id_jugador" a df_jug (Sofifa) y df_part_jug (Flashscore) utilizando los nombres de los jugadores
        df_part_jug_vinc_df_jug = integrate_players_by_name(df_part_jug_unique_players, df_jug_unique_players)
        
        # Reemplazo los nombres de los jugadores por su id en df_part_jug (Flashscore)
        df_part_jug = reemplazar_name_por_id(df_part_jug, df_part_jug_vinc_df_jug)

        # Sintetizar la data de df_jug (Sofifa) en df_part (Flashscore) gracias al vinculo con df_part_jug (Flashscore) -->   Aca dentro hago esto:  # Traer fecha, equipo y no se que mas de df_part (Flashscore) y agregar a df_part_jug (Flashscore) para poder saber en que momento traer la info del jugador (Sofifa tiene varias veces un mismo jugador porque es el jugador en ≠ fifas)
        df = player_data_in_match(df_part, df_part_jug, df_jug)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part_jug_vinc_df_jug.to_excel(f'./p3_data_preparation/data/{self.pais}/df_part_jug_vinc_df_jug.xlsx', index=False)
            df.to_excel(f'./p3_data_preparation/data/{self.pais}/df_integrated.xlsx', index=False)

        return df

    def construct_data(self, df: pd.DataFrame, n_dias: int, n_anios_historial:int , export: bool = True):
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        start = time.time()
        print("\nConstruyendo nuevos datos...")

        # Construyo variables: "equipo_ganador" y puntos obtenidos
        df = construct_data.determinar_equipo_ganador(df)
        df = construct_data.determinar_puntos(df)
        # df = construct_data.determinar_equipo_ganador_segun_casa_apuesta(df)

        # Variables historicas
        df = construct_data.historial_entre_si_segun_fecha(df, n_anios=n_anios_historial)

        # Por estadistica del partido
        l_estadisticas = construct_data.determine_l_estadisticas(df)
        print(f"Estadisticas a promediar en ultimos partidos: {l_estadisticas}")

        for var in l_estadisticas:
            print(f"\tEstadistica a promediar: {var}", df[f"{var}_loc"].dtype, df[f"{var}_vis"].dtype)

            # Determinar la diferencia de la estadistica entre equipo local y visitante de cada partido
            df[f'dif_{var}'] = df[f'{var}_loc'] - df[f'{var}_vis']  # (e.g. dif_goles = goles_loc - goles_vis)
            df = df.drop([f'{var}_loc', f'{var}_vis'], axis=1)  # (e.g. borro goles_loc y goles_vis)

            # Determinar para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
            df = construct_data.determine_prom_en_ult_partidos(df, n_dias=n_dias, variable=var, tipo='mean')
            df = df.drop([f'dif_{var}'], axis=1)

            # Determinar la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_loc y prom_dif_goles_vis)
            df[f'dif_prom_ult_part_dif_{var}'] = df[f'prom_ult_part_dif_{var}_loc'] - df[f'prom_ult_part_dif_{var}_vis']
            df = df.drop(columns=[f'prom_ult_part_dif_{var}_loc', f'prom_ult_part_dif_{var}_vis'], axis=1)

        # Construyo variables de diferencias para las variables promedio de los jugadores
        # df = construct_data.suma_rat_jug_aus(df)
        df = construct_data.calculate_dif_col_jugadores(df)

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p3_data_preparation/data/{self.pais}/df_constructed.xlsx', index=False)
        return df

    def select_data(self, df: pd.DataFrame, thr_corr=None, thr_fs= None, export: bool = True):  # 1.3 minutos # Chequear cambios
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
        n_col = len(df.columns)
        df = df.drop(['id_part', 'pais', 'fecha'], axis=1)  # df = df.drop(['id_part', 'pais', 'competicion', 'temporada', 'fecha', 'cancha', 'es_copa'], axis=1)  # elimino aca por si thr_nan_col elimina una de ellas antes y por ende falla el programa
        print(f"Se eliminó {n_col - len(df.columns)} de {n_col} columnas puesto que no sirven para el analisis (e.g. id_part, fecha, etc).")

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df)

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
            df_etiquetas.to_excel(f'./p3_data_preparation/data/{self.pais}/df_etiquetas.xlsx', index=False)
            df.to_excel(f'./p3_data_preparation/data/{self.pais}/df_selected.xlsx', index=False)
        return df

class Modeling:

    def __init__(self, var_resp: str, var_pred: str, pais: str):
        if not isinstance(var_resp, str) or not isinstance(var_pred, str):
            raise TypeError("Los parámetros var_resp y var_pred deben ser cadenas de texto.")
        if not isinstance(pais, str):
            raise TypeError("El parámetro pais debe ser una cadena de texto.")

        self.var_resp = var_resp
        self.var_pred = var_pred
        self.pais = pais

    def generate_test_design(self, df: pd.DataFrame, bal_type, test_val_size: float = 0.2, test_size: float = 0.5, fill_na=None, with_pca: bool = False, export: bool = True):
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

        # Si no relleno NaN values
        if fill_na is None:

            # Elimino filas con al menos un NaN puesto que al modelo no le pueden ingresar NaN values (solo en variables selected)
            df = clean_data.eliminar_filas_nan(df, porc_nan_max=0)  # df = df.dropna()

            # Separo en X e y
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

            # Separo conjunto de datos en train, validation y test --> Creo que no hace shuffle......
            X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_val_size, random_state=randint(1, 1000), shuffle=True)
            X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=randint(1, 1000), shuffle=True)

            # Elimino variables odds del dataset de entrenamiento y validacion (de test no porque necesito calcular roi)
            # X_train = X_train.drop(['odds_loc', 'odds_emp', 'odds_vis', 'y_pred_ca'], axis=1)
            # X_val = X_val.drop(['odds_loc', 'odds_emp', 'odds_vis', 'y_pred_ca'], axis=1)

        # Si relleno NaN values
        else:
            # Separo en X e y
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

            # Separo conjunto de datos en train, validation y test dejando los NaN values en df_train
            X_train, X_val, X_test, y_train, y_val, y_test = generate_test_design.separate_train_val_and_test(X, y, test_val_size=test_val_size, test_size=test_size)

            # Elimino variables odds del dataset de entrenamiento y validacion (de test no porque necesito calcular roi)
            # X_train = X_train.drop(['odds_loc', 'odds_emp', 'odds_vis', 'y_pred_ca'], axis=1)
            # X_val = X_val.drop(['odds_loc', 'odds_emp', 'odds_vis', 'y_pred_ca'], axis=1)

            # Relleno nan en el dataset de entrenamiento
            X_train, y_train = clean_data.fill_nan_values(X_train, y_train, type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las precisiones casi siempre seran mayores que dropna() en train y test, lo que cuenta es la precision en next_matches o en un dataset que no haya sido filleado...
            print(f"Se realizó el rellenado de NaN values. Shape X_train luego de rellenado: {X_train.shape}")

        # Implemento PCA?
        if with_pca:
            # Selecciono los mejores hiperparametros
            pca = build_model.select_best_hiperparameters(PCA(), X_val, y_val, k=10)

            # Si conviene implementar PCA
            n_comp_opt = pca.get_params()['n_components']
            if n_comp_opt != None:

                # Entreno el modelo
                pca.fit(X_train)
                print("Implementando PCA()...")

                # Transformo X
                X_train = pd.DataFrame(pca.transform(X_train))
                X_val = pd.DataFrame(pca.transform(X_val))
                X_test = pd.DataFrame(pca.transform(X_test))
            else:
                print("No hago PCA() porque gano n_components=None")

        # Balanceo el dataset de entrenamiento
        if bal_type is not None:
            X_train, y_train = generate_test_design.balance_dataset(X_train, y_train, tipo=bal_type)
            print(f"Shape X_train luego de balanceo: {X_train.shape}")

        # Shuffle el dataset de entrenamiento (Funciona mal el shuffle)
        df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)  # concatena mal sin el reset_index()
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        X_train, y_train = df_train.drop(self.var_resp, axis=1), df_train[self.var_resp]
        print(f'Train: {X_train.shape} {y_train.shape}', f'\nVal: {X_val.shape} {y_val.shape}', f'\nTest: {X_test.shape} {y_test.shape}')

        if export:
            X_train.to_excel(f'./p4_modeling/data/{self.pais}/X_train.xlsx', index=False)
            X_val.to_excel(f'./p4_modeling/data/{self.pais}/X_val.xlsx', index=False)
            X_test.to_excel(f'./p4_modeling/data/{self.pais}/X_test.xlsx', index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, model, X_val: pd.DataFrame, y_val: pd.DataFrame, X_train: pd.DataFrame, y_train, k: int):
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
        warnings.filterwarnings("ignore")

        # Find best hiperparameters
        model_best_params = build_model.select_best_hiperparameters(model, X_val, y_val, k=10)

        # Entreno el modelo
        model_best_params.fit(X_train, y_train)

        # Evaluo el modelo con Cross Validation
        cv_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
        print(f"\nPrecision promedio de validación cruzada: {cv_accuracy:.1f}%")

        return model_best_params, cv_accuracy

    def assess_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, export: bool = True):
        """
        Evalúa un modelo de machine learning utilizando datos de prueba y calcula métricas de desempeño.

        :param model: Modelo de Machine Learning entrenado. (sklearn.ensemble)
        :param X_test: Dataframe de prueba con variables predictoras. (DataFrame)
        :param y_test: Dataframe de prueba solo con variable respuesta. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el DataFrame seleccionado. True para exportar, False
        de lo contrario. (bool)
        :return: Precisión del modelo y ROI en el conjunto de prueba. (int) y (float)
        """
        print("\nEvaluando modelo con datos de prueba...")

        # Levanto df_etiquetas
        # df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.pais}/df_etiquetas.xlsx')

        # Predecir las etiquetas para los datos de prueba
        # X_test_without_odds = X_test.copy().drop(['odds_loc', 'odds_emp', 'odds_vis', 'y_pred_ca'], axis=1)  # Quito cuotas de casas de apuestas de X_test
        # y_pred = model.predict(X_test_without_odds)  # es un numpy array
        y_pred = model.predict(X_test)  # es un numpy array

        # Calculo metricas
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100
        print(f"\t- Precision de test: {test_accuracy:.1f}% \n\t- Recall de prueba: {recall:.1f}% \n\t- F1-score de prueba: {f1:.1f}% ")

        """
        # Colculo precision y cuota promedio de casa de apuesta
        df_results = pd.DataFrame({self.var_resp: y_test, self.var_pred: y_pred, 'y_pred_ca': X_test['y_pred_ca'],
                                   'odds_loc': X_test['odds_loc'], 'odds_emp': X_test['odds_emp'],
                                   'odds_vis': X_test['odds_vis']})
        df_results = format_data.revert_columns_from_int(df_results, df_etiquetas)
        # df_results.to_excel(f'./p4_modeling/data/{self.pais}/df_results_prueba.xlsx')
        test_accuracy_ca = accuracy_score(df_results[self.var_resp], df_results['y_pred_ca']) * 100
        print(f"\t- Precision de casa de apuesta: {test_accuracy_ca:.1f}%")
        # Calcular la cuota promedio acertada por la casa de apuesta vs la cuota promedio acertada por mi algoritmo.
        # print(f"\t- Cuota promedio de casa de apuesta: {test_accuracy_ca:.1f}%")
        # print(f"\t- Cuota promedio de mi algoritmo: {test_accuracy_ca:.1f}%")
        """

        # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
        # df_etiquetas_var_resp = df_etiquetas[df_etiquetas['variable'] == self.var_resp]  # solo etiquetas de la var resp
        # df_conf_mat = asses_model.confusion_matrix(y_test, y_pred, df_etiquetas_var_resp)
        # df_conf_mat.to_excel(f'./find_best_hyper/data/{self.pais}/df_conf_matrix.xlsx')

        if export:
            # df_results.to_excel(f'./p4_modeling/data/{self.pais}/df_results.xlsx')
            # df_conf_mat.to_excel(f'./p4_modeling/data/{self.pais}/df_conf_matrix.xlsx')
            pass

        return test_accuracy, recall, f1
    
    def select_best_model(self, l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, export=True):
        """
        Pruebo varios modelos 
        """
        # Definicion de variables
        df_models = pd.DataFrame(columns=['model_name', 'model_trained', 'train_cv_accuracy', 'test_accuracy', 'test_recall', 'test_f1_score'])  # Datos del modelo y su precision y roi
        l_modelos = [RandomForestClassifier(), xgb.XGBClassifier(), LogisticRegression(), SVC(), MLPClassifier(), GradientBoostingClassifier()]  # [DecisionTreeClassifier()]

        # Por modelo --> podria ponerlo como metodo en Modeling()
        for modelo in l_modelos:

            model_name = str(modelo)[:str(modelo).find('(')]  # Defino el nombre del modelo (e.g. "RandomForest")
            print(f" Modelo: {model_name} ".center(120, '-'))

            # Entreno modelo y evaluo su rendimiento
            model_best_params, cv_accuracy = self.build_model(modelo, X_val, y_val, X_train, y_train, k)
            accuracy, recall, f1 = self.assess_model(model_best_params, X_test, y_test)  # accuracy, recall, f1, roi

            # Guardo modelo
            df_models.loc[len(df_models)] = [model_name, model_best_params, cv_accuracy, accuracy, recall, f1]  # roi

        # Selecciono el mejor modelo
        idx = df_models['test_accuracy'].idxmax()  # idx = df_models[df_models['test_accuracy'] == max(df_models['test_accuracy'])].index[0]
        bm_name = df_models.loc[idx, 'model_name']
        bm_params = df_models.loc[idx, 'model_trained']
        bm_train_acc = df_models.loc[idx, 'train_cv_accuracy']
        bm_test_acc = df_models.loc[idx, 'test_accuracy']
        bm_test_rec = df_models.loc[idx, 'test_recall']
        bm_test_f1 = df_models.loc[idx, 'test_f1_score']
        # bm_test_roi = df_models.loc[idx, 'test_roi']
        print(f"\nEl mejor modelo es: {bm_name} con: \n\t- Train Precision: {bm_train_acc:.1f}% \n\t- Test Precision: "
              f"{bm_test_acc:.1f}% \n\t- Test recall: {bm_test_rec:.1f}% \n\t- Test f1-score: {bm_test_f1:.1f}%") # f"\n\t- Test ROI: {bm_test_roi:.1f}%\n"

        if export:
            df_models.to_excel(f'./p4_modeling/data/{self.pais}/df_modelos.xlsx')
            pickle.dump(bm_params, open(f"./p4_modeling/data/{self.pais}/modelo.pkl", "wb"))

        return df_models

def main():

    # Definicion de variables
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    pais = "inglaterra"  # Ponelo en miniscula
    export = True

    # Procesamiento
    data_unders, data_prep, modeling = False, True, True

    if data_unders:
        print(" Data understanding ".center(120, "#"))
        du = DataUnderstanding(pais) # Creo objeto de clase DataPreparation

        # Extriago datos o los levanto
        df_part, df_part_jug, df_jug = du.collect_initial_data(export=export)

        # Describo datos
        du.describe_data(df_part, df_part_jug, df_jug)

    # Si no extraigo datos
    else:
        # Levanto datos ya extraidos
        df_part = pd.read_excel(f'./p2_data_understanding/data/{pais}/df_part.xlsx')
        df_part_jug = pd.read_excel(f'./p2_data_understanding/data/{pais}/df_part_jug.xlsx')
        df_jug = pd.read_excel(f'./p2_data_understanding/data/{pais}/df_jug.xlsx')

        # Si hay datos en df_missing --> Al parecer funciona bien. Exporto los df concatenados?
        try:
            # Levanto datos de partidos missing
            df_part_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_missing.xlsx')
            df_part_jug_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_jug_missing.xlsx')

            # Concateno los partidos extraidos y los missing
            df_part_concat = pd.concat([df_part, df_part_missing], axis=0)
            df_part_jug_concat = pd.concat([df_part_jug, df_part_jug_missing], axis=0)

            # Elimino duplicados para verificar que efectivamente los missing no estaban ya en los extraidos
            df_part = df_part_concat.drop_duplicates().reset_index(drop=True)
            df_part_jug = df_part_jug_concat.drop_duplicates().reset_index(drop=True)
            print(f"Nº de filas repetidas: {len(df_part_concat) - len(df_part)}")
            print(f"Nº de filas repetidas: {len(df_part_jug_concat) - len(df_part_jug)}")
        except:
            print("No hay datos de missing matches, o bien, fallo la concatenacion de los dfs.")

        #du = DataUnderstanding(pais) # Creo objeto de clase DataPreparation
        # du.describe_data(df_part, df_part_jug, df_jug)

    if data_prep:
        # Definicion de variables
        print(" Data preparation ".center(120, "#"))
        dp = DataPreparation(var_resp, pais) # Creo objeto de clase DataPreparation

        # Hiperparametros
        thr_nan_col = 0.7  # Porcentaje maximo de nan values en una columna
        n_dias = 30  # 30 es como N_ULT_PART igual a 5...
        n_anios_historial = 3
        thr_corr = None  # Correlacion umbral para la eliminacion de variables altamente correlacionadas
        thr_fs = 0.2  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)

        # Preparo el dataset para el analisis
        df_part, df_part_jug, df_jug = dp.format_data(df_part, df_part_jug, df_jug, export=False)
        df_part, df_part_jug, df_jug = dp.clean_data(df_part, df_part_jug, df_jug, thr_nan_col=thr_nan_col, export=export)
        df = dp.integrate_data(df_part, df_part_jug, df_jug, export=export) 
        df = dp.construct_data(df, n_dias=n_dias, n_anios_historial=n_anios_historial, export=export)
        df = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, export=export)

    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p3_data_preparation/data/{pais}/df_selected.xlsx')
        print(df.head(1), df.shape)

    if modeling:

        df = df.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1) # 'y_pred_ca' # Temporalmente, las elimino para que no entrene con ellas... dsp las usare para el ROI tal vez

        # Definicion de variables
        print(" Modeling ".center(120, "#"))
        mo = Modeling(var_resp, var_pred, pais)  # Creo objeto de clase Modeling
        modelo = LogisticRegression()

        # Hiperparametros
        test_val_size = 0.25  # Porcentaje del total de datos destinado a validacion y test.
        test_size = 0.5  # Porcentaje de test_val_size destinado a test.
        bal_type = None # Tipo de balanceo a realizar [None, 'over', 'under']
        fill_na = None  # Relleno de nan values [None, mode, ml]
        k = 5  # Numero de folds para seleccionar best parameters y para entrenar modelo

        # Un solo modelo
        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df, bal_type, test_val_size, test_size, fill_na=fill_na)
        model_best_params, cv_accuracy = mo.build_model(modelo, X_val, y_val, X_train, y_train, k)
        test_accuracy, recall, f1 = mo.assess_model(model_best_params, X_test, y_test)  # accuracy, recall, f1, roi
        # print(f"Metricas:\n - Precision train: {cv_accuracy}\n - Precision test: {test_accuracy}\n - Recall: {recall}\n - F1 Score: {f1}")

        if export:
            pickle.dump(model_best_params, open(f"./p4_modeling/data/{pais}/modelo.pkl", "wb"))

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()