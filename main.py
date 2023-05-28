# Importo librerias
import pandas as pd
import time

# Data preparation
from data_preparation import format_data, integrate_data, construct_data, select_data, clean_data
from dspy.data_preparation import clean_data
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Modeling
from modeling import copia_build_model, asses_model
from sklearn.utils import shuffle
from dspy.modeling import test_design
from dspy.modeling.supervised_learning import naive_bayes


class DataPreparation:

    def __init__(self, df_part, df_jug):
        self.df_part = df_part
        self.df_jug = df_jug

    def format_data(self):

        # Convierto posesion de string a integer
        self.df_part = format_data.remove_percent_sign(self.df_part)

        # Convierto fecha de string a datetime
        self.df_part = format_data.transform_date_column(self.df_part, string_format='%d.%m.%Y %H:%M')  # Fundamental para poder ordenar el df por 'fecha'
        self.df_jug = format_data.transform_date_column(self.df_jug, string_format='%b %d, %Y')

        # Remuevo strings adicionales en los nombres de los equipos
        self.df_part = format_data.remove_strings_from_teams(self.df_part)

    def integrate_data(self):

        # Definicion de variables
        l_var = ['l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc', 'l_jug_sup_vis', 'l_jug_ausentes_loc',
                 'l_jug_ausentes_vis']

        # Preparo las columnas con texto como los nombres de equipos y los nombre de jugadores
        self.df_part = integrate_data.prepare_text_columns(self.df_part)
        self.df_jug = integrate_data.prepare_text_columns(self.df_jug)

        # Separo columnas listas en multiples columnas (NO VA A SER NECESARIO CUANDO DESDE LA MISMA EXTRACCION EXTRAIGA VARIAS COLUMNAS...)
        for var in l_var:
            self.df_part = integrate_data.separate_lists_in_columns(self.df_part, var)

        # Integro datasets
        for var in l_var:
            self.df = integrate_data.search_player_data(self.df_part, self.df_jug, var)

        self.df.to_excel('./df_integrated.xlsx', index=False)

    def construct_data(self, N_ULT_PART = 5):

         # Ordeno por campo 'fecha'
         # df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

         l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases',
                               'pases_comp', 'offsides', 'ataques', 'ataques_pelig']

         # Construct data
         self.df = construct_data.equipo_ganador(self.df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis

         self.df = construct_data.historial_entre_si_segun_localia(self.df, n_ult_part=3)

         self.df = construct_data.promedio_dif_gol_ult_part(self.df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos

         self.df = construct_data.forma_reciente(self.df, n_part=N_ULT_PART)  # df = derive_forma_ponderada(df, n_part=N_ULT_PART)

         # Calculo diferencias para las variables promedio de los jugadores
         self.df['dif_rat_tit'] = self.df['l_jug_tit_loc_prom_rat'] - self.df['l_jug_tit_vis_prom_rat']
         self.df['dif_edad_tit'] = self.df['l_jug_tit_loc_prom_edad'] - self.df['l_jug_tit_vis_prom_edad']
         self.df['dif_alt_tit'] = self.df['l_jug_tit_loc_prom_alt'] - self.df['l_jug_tit_vis_prom_alt']
         self.df['dif_rat_sup'] = self.df['l_jug_sup_loc_prom_rat'] - self.df['l_jug_sup_vis_prom_rat']
         self.df['dif_edad_sup'] = self.df['l_jug_sup_loc_prom_edad'] - self.df['l_jug_sup_vis_prom_edad']
         self.df['dif_alt_sup'] = self.df['l_jug_sup_loc_prom_alt'] - self.df['l_jug_sup_vis_prom_alt']

         for var in l_variables_a_prom:
             self.df = construct_data.promedio_ult_partidos(self.df, n_ult_part=N_ULT_PART, variable=var)

         self.df = construct_data.n_dias_ult_partido(self.df)
         # df = convert_odds_to_prob(df)
         # df = numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados

    def select_data(self):

         # Caro
         # df = df.drop(['historial_entre_si', 'fecha', 'odds_loc', 'odds_emp', 'odds_vis', 'es_copa'], axis = 1)

         # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
         self.df = self.df.drop(['id', 'fecha', 'cancha'], axis=1)

         # Remocion de variables redundantes (las de mayor correlacion)
         df_correlation_matrix = self.df.drop('equipo_ganador', axis=1).corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"
         df_correlation_matrix.to_excel('/Users/nachomondino/Desktop/correlation_matrix.xlsx')
         self.df.drop(['dif_pases_segun_ult_part', 'dif_pases_comp_segun_ult_part', 'dif_remates_a_puerta_segun_ult_part',
                  'dif_tarjetas_amarillas_segun_ult_part', 'dif_ataques_segun_ult_part', 'dif_ataques_pelig_segun_ult_part'],
                 inplace=True, axis=1)

         # Elimino otras variables no son importantes... (lo hice manual sin algoritmo pero falta algoritmo)
         # select_data.feature_selection(df)
         self.df = self.df.drop(['dif_faltas_segun_ult_part', 'dif_offsides_segun_ult_part'], axis=1)  # Para red neuronal

    def clean_data(self):
         # Categorizo columnas numericas
         # df = clean_data.categorize_numeric_columns(df)

         # Remover NaN values
         self.df = self.df.dropna()  # inplace=True  # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)

         # Eliminacion de outliers


class Modeling:

    def __init__(self, df, var_resp):
        self.df = df
        self.var_resp = var_resp

    def generate_test_design(self):

         # Definicion de variables
         le = LabelEncoder()
         oversampler = RandomOverSampler()

         # Shuffle dataset
         self.df = pd.DataFrame(shuffle(self.df))  # df = df.sample(frac=1).reset_index(drop=True)

         # Convertir variables categoricas string a categoricas numericas
         for col in self.df.select_dtypes(include=['object']).columns:
             self.df[col] = le.fit_transform(self.df[col])

         # Balanceamos segun variable respuesta
         self.df = clean_data.balance_dataset(self.df, var_resp=self.var_resp)
         # X, y = df_mezclado.drop(var_resp, axis=1), df_mezclado[var_resp]
         # X_bal, y_bal = oversampler.fit_resample(X, y)

         # Separo conjunto de datos en train y test
         self.df_train, self.df_test = test_design.separate_train_and_test(self.df, porc_corte=0.8)
         self.df_train = self.df_train.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
         print(self.df_train.shape, self.df_test.shape)

    def build_model(self):

        # Definicion de variables
        warnings.filterwarnings("ignore")
        best_acurracy = 0
        l_modelos = [DecisionTreeClassifier(max_depth=30),
                  # RandomForestClassifier(n_estimators=200, max_depth = None, random_state=42),
                  # xgb.XGBClassifier(n_estimators=50, objective='multi:softmax', num_class=len(y.unique()), max_depth=20),
                  # LogisticRegression(multi_class='multinomial', penalty='l2', C=0.1, solver='lbfgs', max_iter=500),
                  SVC(kernel='rbf', decision_function_shape='ovo'),
                  MLPClassifier(hidden_layer_sizes=128, activation='tanh', solver='adam', learning_rate='invscaling',
                                max_iter=300),
                  GradientBoostingClassifier(learning_rate= 0.1, n_estimators=200, max_depth=7)
                  # self.red_neuronal(n_folds_cv=10, n_epochs=1000, batches=256),
                 ]

        # Por modelo a probar
        for modelo in l_modelos:

            # Entreno modelo
            print(f" Modelo: {str(modelo)[:str(modelo).find('(')]} ".center(120, '#'))
            model, cv_accuracy, test_accuracy = copia_build_model.build_model(self.df_train, self.df_test, self.var_resp, modelo, best_params=True, k=5) # model = DecisionTreeClassifier()  # model2 = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], random_state=42)

            # Si es el mejor modelo hasta aqui
            if test_accuracy > best_acurracy:  # GUARDAR MAS METRICAS? HAGO EL ASSESS MODEL ACA?
                # Guardo modelo
                best_acurracy = test_accuracy
                best_model = model

        print(f"\nEl mejor modelo es: {self.best_model}")
        self.best_model = best_model
        self.df_result #df_result y_real e y_pred

    def assess_model(self):  # model, df_test, var_resp):

        # Hago prediccion aca? y calculo metricas?

        df_result = naive_bayes.predict_naive_bayes(modelo_nb, df_test, var_resp=var_resp, con_prob=True)
        df_result.to_excel('/Users/nachomondino/Desktop/df_result.xlsx')
        precision = calculate_precision(df_result, var_resp=var_resp, var_pred=var_pred)
        roi = calculate_ROI(df_result, var_resp=var_resp, var_pred=var_pred)

        # confusion_matrix(df_result, var_resp=var_resp, var_pred=var_pred)
        return precision, roi


def main():  # La idea es poner toda el camino de los datos aqui...

    # Definicion de variables
    data_prep = False
    modeling = True
    var_resp = 'equipo_ganador'

    # Hiperparametros
    N_ULT_PART = 5

    if data_prep is True:

        # Levanto datasets
        df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/liga_argentina_historico.xlsx')
        df_jug = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_jugadores.xlsx')

        prepare = DataPreparation(df_part, df_jug)

        # Preparo el dataset para el analisis
        prepare.format_data()
        prepare.integrate_data()
        prepare.construct_data(N_ULT_PART=N_ULT_PART)
        prepare.select_data()
        prepare.clean_data()

        # Obtengo el dataset ya preparado
        df = prepare.df

    else:
         # Levanto dataset ya preparado
         df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/df_selected_manual.xlsx')


    if modeling is True:

        # Definicion de variables
        modeler = Modeling(df, 'equipo_ganador')

        # Analizo los datos
        modeler.generate_test_design()
        modeler.build_model()
        modeler.assess_model()

main()