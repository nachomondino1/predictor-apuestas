# Importo librerias
import pandas as pd
from data_preparation import construct_data, format_data, select_data
from modeling.asses_model import calculate_precision, confusion_matrix, calculate_ROI
from dspy.data_preparation import clean_data
from dspy.modeling.supervised_learning import naive_bayes
from dspy.modeling import test_design
import time

################################################## 3) DATA PREPARATION #################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++ 3.1 Format data ++++++++++++++++++++++++++++++++++++++++++++++++++#
def format(df_part, df_jug):

    # Convierto posesion de string a integer
    df_part = format_data.remove_percent_sign(df_part)

    # Convierto fecha de string a datetime
    df_part = format_data.transform_date_column(df_part, string_format='%d.%m.%Y %H:%M')  # Fundamental para poder ordenar el df por 'fecha'
    df_jug = format_data.transform_date_column(df_jug, string_format='%b %d, %Y')

    # Remuevo strings adicionales en los nombres de los equipos
    df_part = format_data.remove_strings_from_teams(df_part)
    return df_part, df_jug

#+++++++++++++++++++++++++++++++++++++++++++++++++ 3.2 Integrate data +++++++++++++++++++++++++++++++++++++++++++++++++#
def integrate(df_part, df_jug):

    # Prepato texto de df_part
    l_var_to_prepare_part = ['equipo_loc', 'equipo_vis', 'l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc',
                             'l_jug_sup_vis', 'l_jug_ausentes_loc', 'l_jug_ausentes_vis']
    for var in l_var_to_prepare_part:
        prepare_text = TextPreparation(textos=df_part[var])
        prepare_text.to_lower()
        prepare_text.delete_accent()
        prepare_text.delete_special_characters()
        df_part[var] = prepare_text.textos

    # Prepato texto de df_jug
    l_var_to_prepare_jug = ['nombre', 'equipo_actual']
    for var in l_var_to_prepare_jug:
        prepare_text = TextPreparation(textos=df_jug[var])
        prepare_text.to_lower()
        prepare_text.delete_accent()
        prepare_text.delete_special_characters()
        df_jug[var] = prepare_text.textos

    # Separo columnas listas en multiples columnas
    l_var = ['l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc', 'l_jug_sup_vis', 'l_jug_ausentes_loc', 'l_jug_ausentes_vis']
    for var in l_var:
        df_part = separate_lists_in_columns(df_part, var)

    # Relaciono jugador de entidad partido con jugador de entidad jugador
    l_var_2 = ['l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc', 'l_jug_sup_vis', 'l_jug_ausentes_loc', 'l_jug_ausentes_vis']
    for var in l_var_2:
        df = search_player_data(df_part, df_jug, var)

    return df

#+++++++++++++++++++++++++++++++++++++++++++++++++ 3.3 Construct data +++++++++++++++++++++++++++++++++++++++++++++++++#
def construct(df, N_ULT_PART = 5):

     # Ordeno por campo 'fecha'
     # df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

     # Construct data
     df = construct_data.equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis
     df = construct_data.historial_entre_si_segun_localia(df, n_ult_part=N_ULT_PART)

     # No genero dif_gol porque tiene alta correlacion (0.9) con dif_forma
     df = construct_data.promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos

     df = construct_data.forma_reciente(df, n_part=N_ULT_PART)

     l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases',
                           'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
     for var in l_variables_a_prom:
         df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)

     return df

#++++++++++++++++++++++++++++++++++++++++++++++++++ 3.4 Select data ++++++++++++++++++++++++++++++++++++++++++++++++++#
def select(df):

     # Caro
     # df = df.drop(['historial_entre_si', 'fecha', 'odds_loc', 'odds_emp', 'odds_vis', 'es_copa'], axis = 1)

     # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
     df = df.drop(['id', 'fecha', 'cancha'], axis=1)

     # Remocion de variables redundantes (las de mayor correlacion)
     df_correlation_matrix = df.drop('equipo_ganador', axis=1).corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"
     df_correlation_matrix.to_excel('/Users/nachomondino/Desktop/correlation_matrix.xlsx')
     df.drop(['dif_pases_segun_ult_part', 'dif_pases_comp_segun_ult_part', 'dif_remates_a_puerta_segun_ult_part',
              'dif_tarjetas_amarillas_segun_ult_part', 'dif_ataques_segun_ult_part', 'dif_ataques_pelig_segun_ult_part'],
             inplace=True, axis=1)

     # Elimino otras variables no son importantes... (lo hice manual sin algoritmo pero falta algoritmo)
     # select_data.feature_selection(df)
     df = df.drop(['dif_faltas_segun_ult_part', 'dif_offsides_segun_ult_part'], axis=1)  # Para red neuronal
     return df

#++++++++++++++++++++++++++++++++++++++++++++++++++ 3.5 Clean data ++++++++++++++++++++++++++++++++++++++++++++++++++#
def clean(df):
     # Categorizo columnas numericas
     # df = clean_data.categorize_numeric_columns(df)

     # Remover NaN values
     df = df.dropna()  # inplace=True  # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)

     # Eliminacion de outliers

     return df

##################################################### 4) MODELING #####################################################
#++++++++++++++++++++++++++++++++++++++++++++++ 4.2 Generate test design ++++++++++++++++++++++++++++++++++++++++++++++#
def generate_test(df, var_resp):

     # Definicion de variables
     le = LabelEncoder()
     oversampler = RandomOverSampler()

     # Shuffle dataset
     df_mezclado = pd.DataFrame(shuffle(df))  # df = df.sample(frac=1).reset_index(drop=True)

     # Convertir variables categoricas string a categoricas numericas
     for col in df_mezclado.select_dtypes(include=['object']).columns:
         df_mezclado[col] = le.fit_transform(df_mezclado[col])

     # Balanceamos segun variable respuesta
     df = clean_data.balance_dataset(df, var_resp=var_resp)
     # X, y = df_mezclado.drop(var_resp, axis=1), df_mezclado[var_resp]
     # X_bal, y_bal = oversampler.fit_resample(X, y)

     # Separo conjunto de datos en train y test
     df_train, df_test = test_design.separate_train_and_test(df, porc_corte=0.8)
     df_train = df_train.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
     print(df_train.shape, df_test.shape)
     return df_train, df_test  # retorno df_train y df_test?

#++++++++++++++++++++++++++++++++++++++++++++++++++ 4.3 Build model ++++++++++++++++++++++++++++++++++++++++++++++++++#
def build_model(df_train, df_test, var_resp):

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
        model, cv_accuracy, test_accuracy = build_model(df_train, df_test, var_resp, modelo, best_params=True, k=5) # model = DecisionTreeClassifier()  # model2 = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], random_state=42)

        # Si es el mejor modelo hasta aqui
        if test_accuracy > best_acurracy:  # GUARDAR MAS METRICAS? HAGO EL ASSESS MODEL ACA?
            # Guardo modelo
            best_acurracy = test_accuracy
            best_model = model

    print(f"\nEl mejor modelo es: {best_model}")
    return best_model, df_result   #df_result y_real e y_pred

#++++++++++++++++++++++++++++++++++++++++++++++++++ 4.4 Assess model ++++++++++++++++++++++++++++++++++++++++++++++++++#
def assess_model(model, df_test, var_resp):

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

    if data_prep is True:

        # Levanto datasets
        df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/liga_argentina_historico.xlsx')
        df_jug = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_jugadores.xlsx')

        # 3.1 Format data
        df_part, df_jug = format(df_part, df_jug)

        # 3.2 Integrate data
        df = integrate(df_part, df_jug)

        # 3.3 Construct data
        df = construct(df, N_ULT_PART=5)

        # 3.4 Select data
        df = select(df)

        # 3.5 Clean data
        df = clean(df)

    else:
         # Levanto dataset ya preparado
         df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/df_selected_manual.xlsx')


    if modeling is True:

        # 4.2 Generate test design
        df_train, df_test = generate_test(df, var_resp=var_resp)

        # 4.3 Build model
        model = build_model(df_train, df_test, var_resp=var_resp)

        # 4.4 Assess model
        assess_model(model, df_test, var_resp=var_resp)

main()