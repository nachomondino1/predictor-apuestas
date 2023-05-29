# Importo librerias
import pandas as pd
import time
import numpy as np
import warnings

# Data understanding
from data_understanding.collect_data import scraper_sofifa, scraper_flashscore

# Data preparation
from data_preparation import format_data, integrate_data, construct_data, select_data, clean_data
from dspy.data_preparation import clean_data as cd

# # Modeling
# # Generate test design
# from dspy.modeling import test_design
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.utils import shuffle
#
#
# # Build model
# from modeling import copia_build_model, asses_model
# from dspy.modeling.supervised_learning import naive_bayes
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# import xgboost as xgb  # XGBoost
# from sklearn.linear_model import LogisticRegression  # Regresion Logistica
# import lightgbm as lgb  # Gradient Boosting
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
# from sklearn.svm import SVC  # SVM
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, cross_val_predict, KFold, StratifiedKFold
#
# # Assess model
# from sklearn.metrics import accuracy_score


class DataPreparation:

    def __init__(self, var_resp):
        self.var_resp = var_resp

    def format_data(self, df_part, df_jug, export=False): # 1 min

        start = time.time()
        print("\nFormateando los datos...")

        # Convierto posesion de string a integer
        df_part = format_data.convert_posesion_to_int(df_part)

        # Convierto fecha de string a datetime
        df_part = format_data.convert_fecha_to_datetime(df_part, string_format='%d.%m.%Y %H:%M')  # Fundamental para poder ordenar el df por 'fecha'
        df_jug = format_data.convert_fecha_to_datetime(df_jug, string_format='%b %d, %Y')

        # Remuevo strings adicionales en los nombres de los equipos  (lo hago aca porque requiero los equipos limpios para integrar datos)
        df_part = clean_data.remove_strings_from_teams(df_part)

        # Separo columnas listas en multiples columnas (NO VA A SER NECESARIO CUANDO DESDE LA MISMA EXTRACCION EXTRAIGA VARIAS COLUMNAS...)
        l_var = ['l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc', 'l_jug_sup_vis', 'l_jug_ausentes_loc', 'l_jug_ausentes_vis']
        for var in l_var:
            df_part = format_data.separate_lists_in_columns(df_part, var)

        # Convierto valor de mercado en entero
        df_jug = format_data.convert_valor_mercado_to_int(df_jug)

        if export:
            df_part.to_excel('./data_preparation/data/df_part_formated.xlsx', index=False)
            df_jug.to_excel('./data_preparation/data/df_jug_formated.xlsx', index=False)

        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")
        return df_part, df_jug

    def integrate_data(self, df_part, df_jug, export=False):  # 55 min

        start = time.time()
        print("\nIntegrando los datos...")

        # Preparo las columnas con texto como los nombres de equipos y los nombre de jugadores (lo hago aqui y no en clean_data porque uso variables strings para integrar datos)
        df_part = clean_data.prepare_text_columns(df_part)
        df_jug = clean_data.prepare_text_columns(df_jug)

        # Integro datasets
        df_integrated = integrate_data.search_player_data(df_part, df_jug)

        if export:
            df_integrated.to_excel('./data_preparation/data/df_integrated.xlsx', index=False)

        end = time.time()
        print(f"Integracion de datos en {(end - start)/60:.1f} minutos")
        return df_integrated

    def construct_data(self, df, N_ULT_PART = 5, export=False):

        start = time.time()
        print("\nConstruyendo nuevos datos...")

        # Ordeno por campo 'fecha'
        # df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Todos son ascending=True salvo historial_entre_si_segun_localia

        # Construct data
        df = construct_data.determinar_equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis

        df = construct_data.historial_entre_si_segun_localia(df, n_ult_part=int(N_ULT_PART/2))

        l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
        for var in l_variables_a_prom:
            df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)

        df = construct_data.promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos

        df = construct_data.forma_reciente(df, n_part=N_ULT_PART)  # df = derive_forma_ponderada(df, n_part=N_ULT_PART)

        # Calculo diferencias para las variables promedio de los jugadores
        # Podria hacer una resta de todas las variables que tengan "loc" en su nombre con "vis"...?
        df['dif_rat_tit'] = df['l_jug_tit_loc_prom_rat'] - df['l_jug_tit_vis_prom_rat']
        df['dif_edad_tit'] = df['l_jug_tit_loc_prom_edad'] - df['l_jug_tit_vis_prom_edad']
        df['dif_alt_tit'] = df['l_jug_tit_loc_prom_alt'] - df['l_jug_tit_vis_prom_alt']
        df['dif_rat_sup'] = df['l_jug_sup_loc_prom_rat'] - df['l_jug_sup_vis_prom_rat']
        df['dif_edad_sup'] = df['l_jug_sup_loc_prom_edad'] - df['l_jug_sup_vis_prom_edad']
        df['dif_alt_sup'] = df['l_jug_sup_loc_prom_alt'] - df['l_jug_sup_vis_prom_alt']

        df = construct_data.n_dias_ult_partido(df)
        # df = convert_odds_to_prob(df)
        # df = numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados

        if export:
            df.to_excel('./data_preparation/data/df_constructed.xlsx', index=False)

        end = time.time()
        print(f"Construccion de datos en {(end - start)/60:.1f} minutos")
        return df

    def select_data(self, df, export=False):

        start = time.time()
        print("\nSeleccionado datos...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df = df.drop(['id', 'fecha', 'cancha', 'es_copa', 'historial_entre_si'], axis=1)

        # Remocion de variables redundantes (las de mayor correlacion)
        df_correlation_matrix = df.drop(self.var_resp, axis=1).corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"
        df_correlation_matrix.to_excel('/Users/nachomondino/Desktop/correlation_matrix.xlsx')
        df.drop(['dif_pases_segun_ult_part', 'dif_pases_comp_segun_ult_part', 'dif_remates_a_puerta_segun_ult_part', 'dif_tarjetas_amarillas_segun_ult_part', 'dif_ataques_segun_ult_part', 'dif_ataques_pelig_segun_ult_part'], inplace=True, axis=1)

        # Elimino otras variables no son importantes... (lo hice manual sin algoritmo pero falta algoritmo)
        # select_data.feature_selection(df)
        df = df.drop(['dif_faltas_segun_ult_part', 'dif_offsides_segun_ult_part'], axis=1)  # Para red neuronal

        if export:
            df.to_excel('./data_preparation/data/df_selected.xlsx', index=False)

        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")
        return df

    def clean_data(self, df, export=False):

        start = time.time()
        print("\nLimpiando los datos...")

        # Categorizo columnas numericas
        # df = clean_data.categorize_numeric_columns(df)

        # Remover NaN values
        df = df.dropna()  # inplace=True  # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)

        # Eliminacion de outliers?

        # Convertir variables categoricas string a categoricas numericas
        df = clean_data.convert_columns_to_int(df)

        if export:
            df.to_excel('./data_preparation/data/df_cleaned.xlsx', index=False)

        end = time.time()
        print(f"Limpieza de datos en {(end - start)/60:.1f} minutos")
        return df

class Modeling:

    def __init__(self, var_resp):
        self.var_resp = var_resp

    def generate_test_design(self, df, export=True):

        # Definicion de variables
        oversampler = RandomOverSampler()

        # Shuffle dataset
        df = pd.DataFrame(shuffle(df))  # df = df.sample(frac=1).reset_index(drop=True)
        print(f"Shape dataframe original: {df.shape}")

        # Balanceamos segun variable respuesta     # df = clean_data.balance_dataset(df, var_resp=self.var_resp)
        X, y = df.drop(self.var_resp, axis=1), df[self.var_resp]
        X_bal, y_bal = oversampler.fit_resample(X, y)
        df_balanced = pd.concat([X_bal, y_bal], axis=1)  # Funciona?
        print(f"Shape dataframe luego de balanceo: {df_balanced.shape}")

        # Separo conjunto de datos en train y test
        df_train, df_test = test_design.separate_train_and_test(df_balanced, porc_corte=0.8)
        df_train = df_train.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
        print(f"Shape de df_train y df_test : {df_train.shape} {df_test.shape}")

        if export:
            df_train.to_excel('./modeling/data/df_train.xlsx', index=False)
            df_test.to_excel('./modeling/data/df_test.xlsx', index=False)

        return df_train, df_test

    def select_best_model(self, df_train, df_test, l_modelos, best_params=False, k=10):  # que argumentos? l_modelos, best_params?, k?

        # Definicion de variables
        warnings.filterwarnings("ignore")
        best_acurracy = 0

        # Por modelo a probar
        for modelo in l_modelos:

            # Entreno modelo
            print(f" Modelo: {str(modelo)[:str(modelo).find('(')]} ".center(120, '#'))
            model, cv_accuracy, test_accuracy = copia_build_model.build_model(df_train, df_test, self.var_resp, modelo, best_params=best_params, k=k) # model = DecisionTreeClassifier()  # model2 = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], random_state=42)

            # Si es el mejor modelo hasta aqui
            if test_accuracy > best_acurracy:  # GUARDAR MAS METRICAS? HAGO EL ASSESS MODEL ACA?
                # Guardo modelo
                best_acurracy = test_accuracy
                best_model = model

        print(f"\nEl mejor modelo es: {best_model}")
        return best_model, df_result #df_result y_real e y_pred

    def assess_model(self, model, df_test):

        # Hago prediccion aca? y calculo metricas?

        '''
        df_result = naive_bayes.predict_naive_bayes(modelo_nb, df_test, var_resp=var_resp, con_prob=True)
        df_result.to_excel('/Users/nachomondino/Desktop/df_result.xlsx')
        precision = calculate_precision(df_result, var_resp=var_resp, var_pred=var_pred)
        roi = calculate_ROI(df_result, var_resp=var_resp, var_pred=var_pred)
        # confusion_matrix(df_result, var_resp=var_resp, var_pred=var_pred)
        '''

        return precision, roi


def main():  # La idea es poner toda el camino de los datos aqui...

    # Definicion de variables
    data_unders = False
    data_prep = True
    modeling = False
    var_resp = 'equipo_ganador'

    # Hiperparametros
    N_ULT_PART = 5  # Numero de partidos a tener en cuenta para variables historicas como posesion en ult partidos
    '''l_modelos = [DecisionTreeClassifier(max_depth=30),
                 # RandomForestClassifier(n_estimators=200, max_depth = None, random_state=42),
                 # xgb.XGBClassifier(n_estimators=50, objective='multi:softmax', num_class=len(y.unique()), max_depth=20),
                 # LogisticRegression(multi_class='multinomial', penalty='l2', C=0.1, solver='lbfgs', max_iter=500),
                 SVC(kernel='rbf', decision_function_shape='ovo'),
                 MLPClassifier(hidden_layer_sizes=128, activation='tanh', solver='adam', learning_rate='invscaling',
                               max_iter=300),
                 GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=7)
                 # self.red_neuronal(n_folds_cv=10, n_epochs=1000, batches=256),
                 ]'''

    if data_unders is True:
        df_part = scraper_flashscore.extract_flashscore()
        df_jug = scraper_sofifa.extract_sofifa
    else:
        # Levanto datasets
        df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_partido_argentina.xlsx')
        df_jug = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_jugadores.xlsx', index_col=0)
        print(f"Dataframe partido:\n{df_part} \nDataframe jugadores:\n{df_jug}")

    if data_prep is True:
        prepare = DataPreparation(var_resp)

        # Preparo el dataset para el analisis
        df_part, df_jug = prepare.format_data(df_part, df_jug, export=True)
        df = prepare.integrate_data(df_part, df_jug, export=True)
        # df = prepare.construct_data(df, N_ULT_PART=N_ULT_PART, export=True)
        # df = prepare.select_data(df, export=True)
        # df = prepare.clean_data(df, export=True)
    else:
         # Levanto dataset ya preparado
         df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_selected_manual.xlsx')


    if modeling is True:

        # Definicion de variables
        modeler = Modeling(var_resp)

        # Analizo los datos
        df_train, df_test = modeler.generate_test_design(df, export=True)
        # best_model, df_result = modeler.select_best_model(df_train, df_test, l_modelos, best_params=True, k=5)
        # precision, roi = modeler.assess_model(best_model, df_result)

main()