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

    def __init__(self, id_country: int, country: str):
        self.id_country = id_country
        self.country = country
        self.make_directories()

    def make_directories(self):
        ruta_base = f'./p2_data_understanding/data/{self.country.lower()}/data_seg'
        l_directorios = [f'{ruta_base}/per_season/df_match/',
                         f'{ruta_base}/per_season/df_match_player/',
                         f'{ruta_base}/per_season/df_player/',
                         f'{ruta_base}/per_competition/df_match/',
                         f'{ruta_base}/per_competition/df_match_player/',
                         f'{ruta_base}/per_competition/df_player/'
                         ]

        for directorio in l_directorios:
            if not os.path.exists(directorio):
                # Si no existe, crear el directorio
                os.makedirs(directorio)

    def collect_initial_data(self, export: bool =True):
        """
        Collecting data from Flashscore and Sofifa
        """
        print(" Collecting data... ")
        # Definicion de variables
        df_match_concat, df_match_player_concat, df_player_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        l_to_avoid = ['Premier League']  # Pensar alguna manera para automatizar? 

        # Selecciono competencias del country
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
        df_comp_country = df_comp[df_comp['id_country'] == self.id_country]
        print(f' COUNTRY: {self.country} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_country['competition_flashscore']}")

        # POR COMPETITION
        for i, row in df_comp_country.iterrows():

            print(f' Competition: {row["competition_flashscore"]} '.center(120, '+'))

            if row["competition_flashscore"] not in l_to_avoid:

                # Extraigo partidos de Flashscore (df_match y df_match_player)
                df_match, df_match_player = scraper_flashscore.extract_data_flashscore(self.country, row['competition_flashscore'], n_seasons_max=16, export=export)

                # Add columns: id_country, is_cup and id_competition
                df_match['id_country'] = self.id_country
                df_match['id_competition'] = row['id_competition']
                df_match['is_cup'] = row['is_cup']

                # Guardo datos de competition
                df_match_concat = pd.concat([df_match_concat, df_match], axis=0)
                df_match_player_concat = pd.concat([df_match_player_concat, df_match_player], axis=0)

            # Si la competition es una liga
            if row['is_cup'] == 0:

                # Extraigo datos de players de Sofifa (df_player)
                df_player = scraper_sofifa.extract_players_sofifa(self.country, row['competition_sofifa'], export=export)

                # Add columns: id_country and id_competition
                df_player['id_country'] = self.id_country
                df_player['id_competition'] = row['id_competition']

                # Save data
                df_player_concat = pd.concat([df_player_concat, df_player], axis=0)

        # Supongamos que df es tu DataFrame original 
        df_match_odds = df_match_concat.loc[:, ['odds_home', 'odds_draw', 'odds_away']]
        df_match_concat = df_match_concat.drop(['odds_home', 'odds_draw', 'odds_away'], axis=1)
        
        # Exporto datasets con competiciones del country
        if export:
            df_match_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_match_player.xlsx', index=True)
            df_player_concat.to_excel(f'./p2_data_understanding/data/{self.country}/df_player.xlsx', index=True)
            df_match_odds.to_excel(f'./p2_data_understanding/data/{self.country}/df_match_odds.xlsx', index=True)

        return df_match_concat, df_match_player_concat, df_player_concat

    def describe_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame):

        print(" Describiendo datos... ")
        describe_data.getting_to_know_data(df_match)
        describe_data.getting_to_know_data(df_match_player)
        describe_data.getting_to_know_data(df_player)

        # Verifico unicidad de registros segun campos id
        describe_data.verificar_unicidad_registros(df_match)

        # Verifico consistencia en campos que relacionan entidades
        describe_data.check_ids_in_both_dataframes(df_match, df_match_player)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de players: verificar_relacion_entidades(df_player_part, df_match)


class DataPreparation:

    def __init__(self, var_resp: str, country: str):
        self.var_resp = var_resp
        self.country = country
        self.make_directories()

    def make_directories(self):
        directorio = f'./p3_data_preparation/data/{self.country.lower()}'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

    def format_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame, export: bool = True):
        """
        Arreglo el data type de algunas variables.

        :param df_match: Dataframe de los datos de los partidos. (DataFrame)
        :param df_player: Dataframe de los datos de los players. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormateando los datos...")

        # Dataframe match
        ## Date
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        ## Ball posession
        df_match = format_data.convert_posesion_to_int(df_match)
        ## Capacity & Attendance
        df_match = format_data.convert_capacity_to_int(df_match)  # Ver si funciona
        ## Goals (A pesar de que borra filas con goles = "-", son partidos especificos que se suspendieron. Ademas tiene que ver con el dtype)
        df_match = format_data.convert_goles_to_int(df_match)
        df_match_player = df_match_player[df_match_player.index.isin(df_match.index)]  # df_match_player = df_match_player[df_match_player['id_match'].isin(df_match['id_match'])]

        # Dataframe player
        ## Fecha
        df_player['date'] = pd.to_datetime(df_player['date'], format='%b %d, %Y')
        ## Market value
        df_player = format_data.convert_value_to_int(df_player)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_match.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_formated.xlsx', index=True)
            df_match_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_player_formated.xlsx', index=True)
            df_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_player_formated.xlsx', index=True)

        return df_match, df_match_player, df_player

    def clean_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame, thr_nan_col: float, export: bool = True):
        """
        Limpieza inicial de los dataframes
        :param df_match:
        :param df_match_player:
        :param df_player:
        :param export:
        :return:
        """
        start = time.time()
        print("\nCleanning data...")
        warnings.filterwarnings('ignore')

        # ELIMINACION DE FILAS NAN SEGUN % NAN, O BIEN, SELECCION DE DATOS SEGUN TEMPORADA....
        ## Elimino filas con alto porcentaje de NaN values
        n_filas = len(df_match)
        df_match_player = df_match_player.dropna(subset=['player_start_home_11', 'player_start_away_11'], how='any')  # .reset_index(drop=True)
        df_match = df_match[df_match.index.isin(df_match_player.index)] # df_match = df_match[df_match['id_match'].isin(df_match_player['id_match'])]  # .reset_index(drop=True)
        print(f"De las {n_filas} filas, se eliminan {(n_filas - len(df_match))} por no tener formaciones del "
              f"partido, quedan {len(df_match)} filas.")

        ## Elimino columnas con alto porcentaje de NaN values
        if thr_nan_col is not None:

            df_match = clean_data.delete_columns_nan(df_match, porc_nan_max=thr_nan_col)  # 2º elimino columnas con mucho NaN # ojo que asi puede borrar odds
            df_match_player = clean_data.delete_columns_nan(df_match_player, porc_nan_max=0.95)  # TEMPORAL? elimino columnas nan que quedan por el concat y luego la eliminacion de temporadas viejas

        # Dataframe partido:
        ## Team_home y team_away
        df_match = clean_data.prepare_text_columns(df_match, l_cols_to_process=['team_home', 'team_away'])  # Preparacion texto para facilitar construccion de datos bassado en equipos
        df_match = clean_data.clean_teams_names(df_match)  # Eliminar strings adicionales en names de equipos

        # Dataframe partido player:
        ## player_start_home_1, player_start_home2, ..., player_miss_away_18
        df_match_player = clean_data.prepare_text_columns(df_match_player)

        # Dataframe player:
        ## Player Name
        df_player = clean_data.prepare_text_columns(df_player, l_cols_to_process=['name'])  # Preaparo texto para integrar
        ## Market Value
        scaler = StandardScaler()  # Crea un objeto StandardScaler
        df_player['value'] = scaler.fit_transform(df_player['value'].values.reshape(-1, 1))

        # Verificar que no haya outliers
        # algo (sacar de mi tesis)

        end = time.time()
        print(f"Limpieza de datos en {(end - start) / 60:.1f} minutos")

        if export:
            df_match.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_form_clean.xlsx', index=True)
            df_match_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_player_form_clean.xlsx', index=True)
            df_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_player_form_clean.xlsx', index=True)

        return df_match, df_match_player, df_player

    def integrate_data(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, df_player: pd.DataFrame, export: bool = True):
        """
        Integra los datos de partidos y players en un solo dataframe.

        :param df_match: Dataframe de los datos de los partidos.
        :param df_match_player: Dataframe de los datos de los players en cada partido.
        :param df_player: Dataframe de los datos de los players.
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de
        lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        start = time.time()
        print("\nIntegrando los datos...")

        # Mapeo jugadores por nombre
        df_map_players_name_id = match_players_by_name(df_match_player, df_player)

        # Reemplazo nombre de jugadores por id en df_match_player
        df_match_player = replace_players_name_with_id(df_match_player, df_map_players_name_id)
    
        # Sintetizar la data de df_player (Sofifa) en df_match (Flashscore) gracias al vinculo con df_match_player (Flashscore) -->   Aca dentro hago esto:  # Traer fecha, equipo y no se que mas de df_match (Flashscore) y agregar a df_match_player (Flashscore) para poder saber en que momento traer la info del player (Sofifa tiene varias veces un mismo player porque es el player en ≠ fifas)
        df = integrate_player_data_in_match(df_match, df_match_player, df_player)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_map_players_name_id.to_excel(f"./p3_data_preparation/data/{self.country}/df_map_players_name_id.xlsx")
            df_match_player.to_excel(f'./p3_data_preparation/data/{self.country}/df_match_player_with_id.xlsx', index=True)
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_integrated.xlsx', index=True)
        return df

    def construct_data(self, df: pd.DataFrame, n_days: int, n_years_h2h: int , export: bool = True):
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de players. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        start = time.time()
        print("\nConstructing new data...")

        # Construyo variables: "result" y points obtenidos
        df = construct_data.determine_result(df)
        df = construct_data.determine_points(df)

        # Variables historicas
        df = construct_data.h2h_by_date(df, n_years=n_years_h2h)  # Resetea el indice....

        # Determino cuales son las variables stats automaticamente
        l_stats = construct_data.determine_l_stats(df)
        print(f"Stats a promediar en ultimos partidos: {l_stats}")

        # Por estadistica del partido
        for var in l_stats:
            print(f"\tEstadistica a promediar: {var}", df[f"{var}_home"].dtype, df[f"{var}_away"].dtype)

            # Determine la diferencia de la estadistica entre equipo local y visitante de cada partido
            df[f'dif_{var}'] = df[f'{var}_home'] - df[f'{var}_away']  # (e.g. dif_goles = goles_home - goles_away)
            df = df.drop([f'{var}_home', f'{var}_away'], axis=1)  # (e.g. borro goles_home y goles_away)

            # Determine para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
            df = construct_data.determine_mean_in_last_match(df, n_days=n_days, variable=f'dif_{var}', tipo='mean')  # mean_last_match_dif_points_home
            df = df.drop([f'dif_{var}'], axis=1)

            # Determine la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            df[f'dif_mean_last_match_dif_{var}'] = df[f'mean_last_match_dif_{var}_home'] - df[f'mean_last_match_dif_{var}_away']  # KeyError: 'mean_last_match_dif_points_home'
            df = df.drop(columns=[f'mean_last_match_dif_{var}_home', f'mean_last_match_dif_{var}_away'], axis=1)
        
        # Construyo variables de diferencias para las variables promedio de los players
        # df = construct_data.suma_rat_jug_aus(df)
        df = df.drop(columns=['n_player_miss_home', 'n_player_miss_away'], axis=1)  # Temporalmente hasta que vea que hago con esta variable

        df = construct_data.calculate_dif_col_players(df)

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_constructed.xlsx', index=True)
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
        df = df.drop(['date'], axis=1)  
        print(f"Se eliminó {n_col - len(df.columns)} de {n_col} columnas puesto que no sirven para el analisis (e.g. id_match, fecha, etc).")

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df, df_etiquetas = format_data.convert_columns_to_int(df)

        # Elimino variables altamente correlacionadas
        if thr_corr is not None:
            l_columnas_a_eliminar = select_data.eliminar_columnas_correlacionadas(df, self.var_resp, thr_corr)
            df = df.drop(l_columnas_a_eliminar, axis=1)
            print(f"\tSe eliminaron {len(l_columnas_a_eliminar)} de {len(df.columns)-1+len(l_columnas_a_eliminar)} columnas por tener una correlacion mayor a thr_corr={thr_corr*100:.0f}%: {l_columnas_a_eliminar}")

        # Elimino variables menos importantes (feature selection)
        if thr_fs is not None:
            l_not_important_features = select_data.select_best_features(df, self.var_resp, thr_fs, graf=export)
            df = df.drop(l_not_important_features, axis=1)
            print(f"\tSe eliminaron {len(l_not_important_features)} de {len(df.columns)-1+len(l_not_important_features)} columnas por tener un peso menor a thr_fs={thr_fs * 100:.0f}%: {l_not_important_features}")

        print(f"\nLas siguientes {len(df.columns)-1} columnas son las seleccionadas: {list(df.drop(self.var_resp, axis=1).columns)}")

        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_etiquetas.to_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx', index=True)
            df.to_excel(f'./p3_data_preparation/data/{self.country}/df_selected.xlsx', index=True)
        return df

class Modeling:

    def __init__(self, var_resp: str, var_pred: str, country: str):
        if not isinstance(var_resp, str) or not isinstance(var_pred, str):
            raise TypeError("Los parámetros var_resp y var_pred deben ser cadenas de texto.")
        if not isinstance(country, str):
            raise TypeError("El parámetro country debe ser una cadena de texto.")

        self.var_resp = var_resp
        self.var_pred = var_pred
        self.country = country
        self.make_directories()

    def make_directories(self):
        directorio = f'./p4_modeling/data/{self.country.lower()}'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)
        
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
            df = clean_data.delete_rows_nan(df, porc_nan_max=0)  # df = df.dropna()

            # Separo en X e y
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

            # Separo conjunto de datos en train, validation y test
            X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_val_size, random_state=randint(1, 1000), shuffle=True)
            X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=randint(1, 1000), shuffle=True)

        # Si relleno NaN values
        else:
            # Separo en X e y
            X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]

            # Separo conjunto de datos en train, validation y test dejando los NaN values en df_train
            X_train, X_val, X_test, y_train, y_val, y_test = generate_test_design.separate_train_val_and_test(X, y, test_val_size=test_val_size, test_size=test_size, shuffle=True)

            # Relleno nan en el dataset de entrenamiento
            X_train, y_train = clean_data.fill_nan_values(X_train, y_train, type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las accuracyes casi siempre seran mayores que dropna() en train y test, lo que cuenta es la accuracy en next_matches o en un dataset que no haya sido filleado...
            print(f"Se realizó el rellenado de NaN values. Shape X_train luego de rellenado: {X_train.shape}")

        # Implemento PCA
        if with_pca:
            print("Implementando PCA()...")

            # Selecciono los mejores hiperparametros
            pca = build_model.select_best_hiperparameters(PCA(), X_val, y_val, k=10)

            # Si conviene implementar PCA
            n_comp_opt = pca.get_params()['n_components']
            if n_comp_opt != None:

                # Entreno el modelo
                pca.fit(X_train)

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

        print(f'Train: {X_train.shape} {y_train.shape}', f'\nVal: {X_val.shape} {y_val.shape}', f'\nTest: {X_test.shape} {y_test.shape}')

        if export:
            X_train.to_excel(f'./p4_modeling/data/{self.country}/X_train.xlsx', index=True)
            X_val.to_excel(f'./p4_modeling/data/{self.country}/X_val.xlsx', index=True)
            X_test.to_excel(f'./p4_modeling/data/{self.country}/X_test.xlsx', index=True)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, model, X_val: pd.DataFrame, y_val: pd.DataFrame, X_train: pd.DataFrame, y_train, k: int):
        """
        Selecciona el mejor modelo a partir de la accuracy.
        :param model: Modelo de Machine Learning. (sklearn.ensemble)
        :param X_val: Dataframe de validacion con variables predictoras. (DataFrame)
        :param y_val: Dataframe de validacion solo con variable respuesta. (DataFrame)
        :param X_train: Dataframe de entrenamiento con variables predictoras.  (DataFrame)
        :param y_train: Dataframe de entrenamiento solo con variable respuesta. (DataFrame)
        :param k: Numero de folds. (int)
        :return: Mejor modelo. (sklearn.ensemble?)
        """
        warnings.filterwarnings("ignore")
        print("Training model...")

        # Find best hiperparameters
        model_best_params = build_model.select_best_hiperparameters(model, X_val, y_val, k=k)

        # Entreno el modelo
        model_best_params.fit(X_train, y_train)

        # Evaluo el modelo con Cross Validation
        cv_accuracy = build_model.manual_cross_validation(model_best_params, X_train, y_train, k)
        print(f"\nAccuracy promedio de validación cruzada: {cv_accuracy:.1f}%")

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
        print("\nEvaluating trained model with test sets...")

        # Levanto df_etiquetas
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')

        # Predecir las etiquetas para los datos de prueba
        y_pred = model.predict(X_test)  # es un numpy array
        y_pred_prob = model.predict_proba(X_test) # --> Supuestamente te da las probabilidad de cada clase... la tengo que probar. Funciona para todos los modelos?

        # Guardo predicciones en DataFrame
        df = pd.DataFrame({
            'result': y_test,
            'predicted_result': y_pred,
            'probability_class_0': y_pred_prob[:, 0],
            'probability_class_1': y_pred_prob[:, 1],  # Si hay más de dos clases, continúa añadiendo columnas
            'probability_class_2': y_pred_prob[:, 2],  # Si hay más de dos clases, continúa añadiendo columnas
            # Agregar más columnas si hay más clases
        })
        df.index = X_test.index
        # df.to_excel('/Users/nachomondino/Desktop/df_prueba.xlsx')

        # Agrego predicciones de bookmaker
        df = asses_model.calculate_bookmaker_precision(df)
        # df.to_excel('/Users/nachomondino/Desktop/df_results_con_odds.xlsx', index=True)

        # Convierto 'result' y 'predicted_result' de numeros a clases (e.g. 'away', 'home', 'draw')
        df = format_data.revert_columns_from_int(df, df_etiquetas)
        # df.to_excel('/Users/nachomondino/Desktop/df_results_reverted.xlsx', index=True)

        # Calculo metricas
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100
        print(f"\t- Accuracy de test: {test_accuracy:.1f}% \n\t- Recall de prueba: {recall:.1f}% \n\t- F1-score de prueba: {f1:.1f}% ")
        test_precision_bookmaker = accuracy_score(df['result'], df['bookmaker_result']) * 100
        print(f"\t- Precision de casa de apuesta: {test_precision_bookmaker:.1f}%")

        # Por estrategia de inversion
            # Calcular el roi
            # Guardar el mejor ROI
        # Calculo de ROI --> ya tengo las cuotas!!!!!!!
        roi = asses_model.calculate_roi(df)

        # Calculo matriz de confusion  --> Hacerlo solo del mejor modelo?
        df_etiquetas_var_resp = df_etiquetas[df_etiquetas['variable'] == self.var_resp]  # solo etiquetas de la var resp
        df_conf_mat = asses_model.confusion_matrix(y_test, y_pred, df_etiquetas_var_resp)

        if export:
            df.to_excel(f'./p4_modeling/data/{self.country}/df_results.xlsx')
            df_conf_mat.to_excel(f'./p4_modeling/data/{self.country}/df_conf_matrix.xlsx')
    
        return test_accuracy, recall, f1
    
    def select_best_model(self, l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, export=True):
        """
        Pruebo varios modelos 
        """
        # Definicion de variables
        df_models = pd.DataFrame(columns=['model_name', 'model_trained', 'train_cv_accuracy', 'test_accuracy', 'test_recall', 'test_f1_score'])  # Datos del modelo y su accuracy y roi
        l_modelos = [RandomForestClassifier(), xgb.XGBClassifier(), LogisticRegression(), SVC(), MLPClassifier(), GradientBoostingClassifier()]  # [DecisionTreeClassifier()]

        # Por modelo --> podria ponerlo como metodo en Modeling()
        for modelo in l_modelos:

            model_name = str(modelo)[:str(modelo).find('(')]  # Defino el name del modelo (e.g. "RandomForest")
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
        print(f"\nEl mejor modelo es: {bm_name} con: \n\t- Train Accuracy: {bm_train_acc:.1f}% \n\t- Test Accuracy: "
              f"{bm_test_acc:.1f}% \n\t- Test recall: {bm_test_rec:.1f}% \n\t- Test f1-score: {bm_test_f1:.1f}%") # f"\n\t- Test ROI: {bm_test_roi:.1f}%\n"

        if export:
            df_models.to_excel(f'./p4_modeling/data/{self.country}/df_modelos.xlsx')
            pickle.dump(bm_params, open(f"./p4_modeling/data/{self.country}/modelo.pkl", "wb"))

        return df_models

def main():
    """
    Extraction, processing and analysis of matches to predict match results.
    """
    # Definicion de variables
    var_resp, var_pred = 'result', 'predicted_result'
    data_unders, data_prep, modeling = True, False, False
    export = True

    # Selecciono country a extraer por terminal
    # df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    country = 'Germany'  # country = str(input("Choose country to extract (e.g. England, Germany, etc): "))
    id_country = 59  # id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]

    # DATA UNDERSTANDING
    if data_unders:
        print(" Data understanding ".center(120, "#"))
        du = DataUnderstanding(id_country, country) # Creo objeto de clase DataPreparation

        # Extriago datos o los levanto
        df_match, df_match_player, df_player = du.collect_initial_data(export=export)

        # Describo datos
        du.describe_data(df_match, df_match_player, df_player)
    elif data_prep:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index_col=0)
        df_player = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player.xlsx')

        '''
        # Que se fije si hay datos actualizados y, si hay, que los reemplace
        # Si hay datos en df_missing --> Al parecer funciona bien. Exporto los df concatenados?
        try:
            # Levanto datos de partidos missing
            df_match_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_missing.xlsx')
            df_match_player_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_player_missing.xlsx')

            # Concateno los partidos extraidos y los missing
            df_match_concat = pd.concat([df_match, df_match_missing], axis=0)
            df_match_player_concat = pd.concat([df_match_player, df_match_player_missing], axis=0)

            # Elimino duplicados para verificar que efectivamente los missing no estaban ya en los extraidos
            df_match = df_match_concat.drop_duplicates()  # .reset_index(drop=True)
            df_match_player = df_match_player_concat.drop_duplicates()  # .reset_index(drop=True)
            print(f"Nº de filas repetidas: {len(df_match_concat) - len(df_match)}")
            print(f"Nº de filas repetidas: {len(df_match_player_concat) - len(df_match_player)}")
        except:
            print("No hay datos de missing matches, o bien, fallo la concatenacion de los dfs.")
        '''

        # du = DataUnderstanding(id_country, country) # Creo objeto de clase DataPreparation
        # du.describe_data(df_match, df_match_player, df_player)

    # DATA PREPARATION
    if data_prep:
        # Definicion de variables
        print(" Data preparation ".center(120, "#"))
        dp = DataPreparation(var_resp, country) # Creo objeto de clase DataPreparation

        # Hiperparametros # PODRIA PONERLOS EN UN DICT Y HACER EL DATAFRAME MAS AUTOMATICO
        thr_nan_col = 0.7  # Porcentaje maximo de nan values en una columna
        n_days = 30  # 30 es como N_ULT_PART igual a 5...
        n_years_h2h = 3
        thr_corr = 0.8  # Correlacion umbral para la eliminacion de variables altamente correlacionadas
        thr_fs = 0.2  # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)
        df_hiper_prep = pd.DataFrame(data={'thr_nan_col': [thr_nan_col], 'n_days': [n_days], 'n_years_h2h': [n_years_h2h], 'thr_corr': [thr_corr], 'thr_fs': [thr_fs]}, index=[0])
        print(df_hiper_prep)
        
        df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_constructed.xlsx', index_col=0)
        print(df.head(2))

        # Preparo el dataset para el analisis
        # df_match, df_match_player, df_player = dp.format_data(df_match, df_match_player, df_player, export=export)
        # df_match, df_match_player, df_player = dp.clean_data(df_match, df_match_player, df_player, thr_nan_col=thr_nan_col, export=export)
        # df = dp.integrate_data(df_match, df_match_playser, df_player, export=export) 
        # df = dp.construct_data(df, n_days=n_days, n_years_h2h=n_years_h2h, export=export)
        df = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, export=export)
        
        if export:
            df_hiper_prep.to_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx', index=False)
    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p3_data_preparation/data/{country}/df_selected.xlsx', index_col=0)
        print(df.head(3), df.shape)

    # MODELING
    if modeling:
        print(" Modeling ".center(120, "#"))
        # Definicion de variables
        mo = Modeling(var_resp, var_pred, country)  # Creo objeto de clase Modeling
        modelo = LogisticRegression()

        # Hiperparametros
        test_val_size = 0.25  # Porcentaje del total de datos destinado a validacion y test.
        test_size = 0.5  # Porcentaje de test_val_size destinado a test.
        fill_na = None  # Relleno de nan values [None, mode, ml]
        bal_type = None # Tipo de balanceo a realizar [None, 'over', 'under']
        k = 5  # Numero de folds para seleccionar best parameters y para entrenar modelo
        df_hiper_mod = pd.DataFrame(data={'test_val_size': [test_val_size], 'test_size': [test_size], 'bal_type': [bal_type], 'fill_na': [fill_na], 'k': [k]}, index=[0])        
        print(df_hiper_mod)

        # Un solo modelo
        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df, bal_type, test_val_size, test_size, fill_na=fill_na)
        model_best_params, cv_accuracy = mo.build_model(modelo, X_val=X_val, y_val=y_val, X_train=X_train, y_train=y_train, k=k)
        test_accuracy, recall, f1 = mo.assess_model(model_best_params, X_test, y_test)  # accuracy, recall, f1, roi
        # print(f"Metricas:\n - Accuracy train: {cv_accuracy}\n - Accuracy test: {test_accuracy}\n - Recall: {recall}\n - F1 Score: {f1}")

        if export:
            df_hiper_mod.to_excel(f'./p4_modeling/data/{country}/df_hiper_mod.xlsx', index=True)
            pickle.dump(model_best_params, open(f"./p4_modeling/data/{country}/modelo.pkl", "wb"))
    
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()