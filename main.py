# Importo librerias
import pandas as pd
from dspy.data_preparation import clean_data
from data_preparation import construct_data, format_data, select_data
from dspy.modeling import naive_bayes, test_design
from evaluation.evaluation import calculate_precision

def main():  # La idea es poner toda el camino de los datos aqui...
    # Hiperparametros
    N_ULT_PART = 5

    # 2) DATA UNDERSTANDING
    # Levanto el dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/liga_argentina_historico.xlsx')
    # Describe y explore data en Google Colab con la libreria ydata-profiling

    # 3) DATA PREPARATION
    # 3.1) Format data
    df = format_data.format_column_date(df)
    df = format_data.posesion_balon(df)
    df = format_data.clean_teams(df)  # no funca o si?

    # 3.2) Construct data
    df = construct_data.equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis
    df = construct_data.numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados
    df = construct_data.dif_gol(df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos
    df = construct_data.diferencia_col(df, col1='dig_gol_ult_part_loc', col2='dig_gol_ult_part_vis', nombre_nueva_col='dif_gol')
    df = construct_data.historial_entre_si(df, n_part=N_ULT_PART)  # Determino columna "historial_entre_si"
    l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
    for var in l_variables_a_prom:
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)
        nombre_loc , nombre_vis = f'{var}_ult_part_loc',  f'{var}_ult_part_vis'
        df = construct_data.diferencia_col(df, col1=nombre_loc, col2=nombre_vis , nombre_nueva_col=f'dif_{var}')
    print(df.head())
    df.to_excel('/Users/nachomondino/Desktop/df_constructed.xlsx')

    # 3.3) Select data
    # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
    df = df.drop(['id', 'fecha', 'cancha'], axis=1)

    # Remocion de variables redundantes (las de mayor correlacion)
    df_correlation_matrix = df.drop('equipo_ganador', axis=1).corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"
    df_correlation_matrix.to_excel('/Users/nachomondino/Desktop/correlation_matrix.xlsx')
    df.drop(['dif_pases', 'dif_pases_comp', 'dif_remates_a_puerta', 'dif_tarjetas_amarillas', 'dif_ataques_pelig'], inplace=True, axis=1)

    # Elimino otras variables no son importantes... (lo hice manual sin algoritmo pero falta algoritmo)
    # select_data.feature_selection(df)
    df = df.drop(['dif_faltas', 'dif_offsides', 'dif_ataques_ataques'], axis=1)  # Para red neuronal
    df.to_excel('/Users/nachomondino/Desktop/df_selected.xlsx')

    # Elimino filas con al menos un NaN?
    # df = df.dropna(subset=['historial_entre_si']).reset_index()  # Elimina filas con al menos un valor nulo

    # 3.4) Clean data
    df = clean_data.categorize_numeric_columns(df)  # Categorizo columnas numericas

    df = df.dropna(subset=['historial_entre_si']).reset_index()  # Elimina filas con al menos un valor nulo

    # df = clean_data.balance_dataset(df, var_resp='equipo_ganador')

    df.to_excel('/Users/nachomondino/Desktop/df_prepared.xlsx')

    # 4) MODELING
    # Definicion de variables
    N_MODELOS = 10
    l_aciertos = []

    # Por modelo
    for i in range(N_MODELOS):

        # Shuffle dataset
        df = df.sample(frac=1).reset_index(drop=True)

        # Separo conjunto de datos en train y test
        df_train, df_test = test_design.separate_train_and_test(df)

        # Implemento Naive Bayes
        modelo_nb = naive_bayes.train_naive_bayes(df_train, var_resp='equipo_ganador')

        # 5) EVALUACION DEL MODELO
        df_result = naive_bayes.predict_naive_bayes(modelo_nb, df_test, var_resp='equipo_ganador', col_prob_clase=True)
        precision = calculate_precision(df_result, var_resp='equipo_ganador')
        l_aciertos.append(precision)

    print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos) / len(l_aciertos)}")

main()