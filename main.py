# Importo librerias
import pandas as pd
from data_preparation import construct_data, format_data, select_data
from modeling.asses_model import calculate_precision, confusion_matrix
from dspy.data_preparation import clean_data
from dspy.modeling.supervised_learning import naive_bayes
from dspy.modeling import test_design


def data_preparation(df):
    # Hiperparametros
    N_ULT_PART = 5

    # 3.1) Format data
    df = format_data.format_column_date(df)
    df = format_data.posesion_balon(df)
    df = format_data.clean_teams(df)

    # 3.2) Construct data
    # Construct data
    df = construct_data.equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis
    df = construct_data.numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados
    df = construct_data.historial_entre_si(df, n_ult_part=N_ULT_PART)

    # No genero dif_gol porque tiene alta correlacion (0.9) con dif_forma
    # df = construct_data.promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos
    # df = construct_data.diferencia_col(df, col1='dif_gol_ult_part_loc', col2='dif_gol_ult_part_vis', nombre_nueva_col='dif_gol')  # ???? Calculo la dif de gol de loc - dif de gol de vis
    # df = df.drop(['dif_gol_ult_part_loc', 'dif_gol_ult_part_vis'],  axis=1)

    df = construct_data.forma_reciente(df, n_part=N_ULT_PART)
    df = construct_data.diferencia_col(df, col1='forma_loc', col2='forma_vis', nombre_nueva_col='dif_forma')  # ???? Calculo la dif de gol de loc - dif de gol de vis
    df = df.drop(['forma_loc', 'forma_vis'], axis=1)

    l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases',
                          'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
    for var in l_variables_a_prom:
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)

    df = construct_data.diferencia_col(df, col1='l_jug_lesionados_loc', col2='l_jug_lesionados_vis', nombre_nueva_col='dif_lesionados')
    df.to_excel('/Users/nachomondino/Desktop/df_constructed.xlsx')

    # 3.3) Select data
    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha'], axis=1)

    # Remocion de variables redundantes (las de mayor correlacion)
    df_correlation_matrix = df.drop('equipo_ganador',
                                    axis=1).corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"
    df_correlation_matrix.to_excel('/Users/nachomondino/Desktop/correlation_matrix.xlsx')
    df.drop(['dif_pases_segun_ult_part', 'dif_pases_comp_segun_ult_part', 'dif_remates_a_puerta_segun_ult_part',
             'dif_tarjetas_amarillas_segun_ult_part', 'dif_ataques_segun_ult_part', 'dif_ataques_pelig_segun_ult_part'],
            inplace=True, axis=1)

    # Elimino otras variables no son importantes... (lo hice manual sin algoritmo pero falta algoritmo)
    # select_data.feature_selection(df)
    df = df.drop(['dif_faltas_segun_ult_part', 'dif_offsides_segun_ult_part'], axis=1)  # Para red neuronal

    # 3.4) Clean data
    df = clean_data.categorize_numeric_columns(df)  # Categorizo columnas numericas

    df = df.dropna(subset=['dif_forma']).reset_index()  # Elimina filas con al menos un valor nulo en dif_gol (primeros partidos)
    df.to_excel('/Users/nachomondino/Desktop/df_prepared.xlsx', index=False)


def modeling(df):  # el for i in range(n_modelos) podrian ir llamando a modeling...
    """
    Implementa la etapa de modelado del proyecto
    :param df:
    :return:
    """
    # Defino variables
    var_resp, var_pred = 'equipo_ganador', 'y_pred'

    # 4.2) Generate test design
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # df = clean_data.balance_dataset(df, var_resp='equipo_ganador')

    # Separo conjunto de datos en train y test
    df_train, df_test = test_design.separate_train_and_test(df, porc_corte=0.8)
    print(df_train.shape, df_test.shape)

    # 4.3) Build model
    # Implemento Naive Bayes
    modelo_nb = naive_bayes.train_naive_bayes(df_train, var_resp=var_resp)
    modelo_nb.to_excel('/Users/nachomondino/Desktop/df_prob.xlsx')

    # 4.4) Asses model
    df_result = naive_bayes.predict_naive_bayes(modelo_nb, df_test, var_resp=var_resp, con_prob=True)
    df_result.to_excel('/Users/nachomondino/Desktop/df_result.xlsx')
    precision = calculate_precision(df_result, var_resp=var_resp, var_pred=var_pred)

    confusion_matrix(df_result, var_resp=var_resp, var_pred=var_pred)
    return precision

def main():  # La idea es poner toda el camino de los datos aqui...

    # Definicion de variables
    N_MODELOS = 10
    l_aciertos = []

    # Levanto el dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/liga_argentina_historico.xlsx')
    # Describe y explore data en Google Colab con la libreria ydata-profiling

    df = data_preparation(df)

    # Ojo con las probabilidad de estas columnas
    # df = df.drop(['goles_loc', 'goles_vis', 'l_jug_lesionados_loc', 'l_jug_lesionados_vis'], axis=1)

    # Por modelo
    for i in range(N_MODELOS):
        precision = modeling(df)
        l_aciertos.append(precision)

    print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos) / len(l_aciertos)}")

main()