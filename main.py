# Importo librerias
import pandas as pd
from data_preparation import construct_data, format_data, select_data
from modeling.asses_model import calculate_precision, confusion_matrix, calculate_ROI
from dspy.data_preparation import clean_data
from dspy.modeling.supervised_learning import naive_bayes
from dspy.modeling import test_design
from modeling.build_model import Modelado
import time

def format(df, df_jug):

    # Convierto posesion de string a integer
    df = format_data.remove_percent_sign(df)

    # Convierto fecha de string a datetime
    df = format_data.transform_date_column(df, string_format='%d.%m.%Y %H:%M')  # Fundamental para poder ordenar el df por 'fecha'
    df_jug = format_data.transform_date_column(df_jug, string_format='%b %d, %Y')

    # Remuevo strings adicionales en los nombres de los equipos
    df = format_data.remove_strings_from_teams(df)

    # Ordeno por campo 'fecha'
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)
    return df, df_jug

def integrate(df):
    pass

def construct(df, N_ULT_PART = 5):

    # Construct data
    df = construct_data.equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis
    df = construct_data.numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados
    df = construct_data.historial_entre_si(df, n_ult_part=N_ULT_PART)

    # No genero dif_gol porque tiene alta correlacion (0.9) con dif_forma
    df = construct_data.promedio_dif_gol_ult_part(df,
                                                  n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos

    df = construct_data.forma_reciente(df, n_part=N_ULT_PART)

    l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases',
                          'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
    for var in l_variables_a_prom:
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)

    df.to_excel('/Users/nachomondino/Desktop/df_constructed.xlsx')

    return df

def select(df):

    # Caro
    df = df.drop(['historial_entre_si', 'fecha', 'odds_loc', 'odds_emp', 'odds_vis', 'es_copa'], axis = 1)


    '''
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
    '''
    return df

def clean(df):
    # Categorizo columnas numericas
    # df = clean_data.categorize_numeric_columns(df)

    # Remover NaN values
    df = df.dropna()  # inplace=True
    # df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)
    # df.to_excel('/Users/nachomondino/Desktop/df_prepared.xlsx', index=False)

    return df

def generate_test(df, var_resp):

    # Shuffle dataset
    df_mezclado = pd.DataFrame(shuffle(df))
    # df = df.sample(frac=1).reset_index(drop=True)

    # Convertir variables categoricas string a categoricas numericas
    le = LabelEncoder()
    oversampler = RandomOverSampler()
    for col in df_mezclado.select_dtypes(include=['object']).columns:
        df_mezclado[col] = le.fit_transform(df_mezclado[col])

    # Balanceamos segun variable respuesta
    X, y = df_mezclado.drop(var_resp, axis=1), df_mezclado[var_resp]
    X_bal, y_bal = oversampler.fit_resample(X, y)
    # df = clean_data.balance_dataset(df, var_resp='equipo_ganador')

    # Separo conjunto de datos en train y test
    # df_train, df_test = test_design.separate_train_and_test(df, porc_corte=0.8)
    # df_train = df_train.drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
    # print(df_train.shape, df_test.shape)
    return X_bal, y_bal

def build_model(X_bal, y_bal):

    modeler = Modelado(X_bal, y_bal, 'equipo_ganador')

    # Probamos varios modelos
    modeler.arbol_decision(plot_feature_importance=True) # max_depth_tree=25, n_folds_cv=10
    modeler.random_forest(n_folds_cv=10, n_tress_in_forest=100, max_depth_tree=25, plot_feature_importance=True) 
    modeler.xgboost(n_folds_cv=10, n_tress_in_forest=50, max_depth_tree=15, plot_feature_importance=True)
    modeler.regresion_logistica(n_folds_cv= 10, penal = 'l2', c_value = 1, solv = 'lbfgs', max_iter= 500) # n_folds_cv= 10, penal = 'l2', c_value = 1, solv = 'lbfgs', max_iter= 500 
    modeler.svm(n_folds_cv = 10, kernel_type = 'poly', ovo_o_ovr= 'ovo')
    modeler.perceptron_multiple(n_folds_cv = 10, activ = 'tanh', solv = 'lbfgs', lear_rate = 'invscaling', max_itera = 300)
    modeler.red_neuronal(n_folds_cv = 10, n_epochs = 1000, batches = 256)
    modeler.gradient_boosting(n_folds_cv = 10, lear_rate = 0.1, n_trees = 200, max_depth = 7)
    # warnings.filterwarnings("ignore")

    # Seleccionamos mejor modelo
    t0 = time.time() # Registramos el tiempo de inicio
    best_model = modeler.seleccionar_mejor_modelo()
    t1 = time.time() # Registramos el tiempo de fin
    print(f"La función tardó {(t1-t0)/60:.2f} minutos en ejecutarse") # Imprimimos el tiempo transcurrido

    return best_model, df_result   #df_result y_real e y_pred

def assess_model(model, df_test, var_resp):

    df_result = naive_bayes.predict_naive_bayes(modelo_nb, df_test, var_resp=var_resp, con_prob=True)
    df_result.to_excel('/Users/nachomondino/Desktop/df_result.xlsx')
    precision = calculate_precision(df_result, var_resp=var_resp, var_pred=var_pred)
    roi = calculate_ROI(df_result, var_resp=var_resp, var_pred=var_pred)

    # confusion_matrix(df_result, var_resp=var_resp, var_pred=var_pred)
    return precision, roi

def main():  # La idea es poner toda el camino de los datos aqui...

    # Definicion de variables
    N_MODELOS = 15
    l_aciertos, l_roi = [], []

    '''
    # Levanto el dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/liga_argentina_historico.xlsx')

    # 3) Data preparation
    df = format(df, df_jug)

    df = integrate(df, df_jug)

    df = construct(df, N_ULT_PART=5)

    df = clean(df)
    '''

    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/df_selected_manual.xlsx')
    df = clean(df)

    x_bal, y_bal = generate_test(df, var_resp="equipo_ganador")


    best_model = build_model(x_bal, y_bal)
    # Guardar best_model

    # Modeler() --> best_model y df_result

    assess_model()


    '''
    # 4) Modeling
    # Por modelo
    for i in range(N_MODELOS):
        precision, roi = modeling(df)
        l_aciertos.append(precision)
        l_roi.append(roi)

    print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos) / len(l_aciertos)}")
    print(f"Max: {max(l_roi)} Min: {min(l_roi)} Prom: {sum(l_roi) / len(l_roi)}")
    '''
    # que hacer CV?
    # Dejamos metricas en Modeler? o ponemos en assess_model?
main()