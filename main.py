# Importo librerias
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from data_preparation.clean_data import clean_teams
from data_preparation import integrate_data
from dspy.data_understanding import describe_data
from dspy.data_preparation import format_data, clean_data
from dspy.modeling import naive_bayes, test_design


def main():
    # La idea es poner toda el camino de los datos aqui...
    # Levanto el dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/flashscore/liga_argentina_historico.xlsx')

    # 2) DATA UNDERSTANDING
    # 2.1) Describe data
    describe_data.getting_to_know_data(df)

    # 2.2) Format data
    df = format_data.format_column_date(df)

    # 3) DATA PREPARATION
    # 3.1) Clean data
    df = clean_teams(df)
    df = clean_data.categorize_numeric_columns(df)  # Categorizo columnas numericas

    # 3.2) Integrate data
    df = integrate_data.derive_winning_team(df)  # Determino columna "equipo_ganador"
    df = integrate_data.derive_historial_entre_si(df, n_part=5)  # Determino columna "historial_entre_si"
    df = integrate_data.derive_dif_gol_last_matches(df, n_part=5)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos
    df = integrate_data.derive_forma_ponderada(df=df, n_part=5)
    print(df.head())

    # Remuevo columnas fecha
    df.drop('id', inplace=True, axis=1)
    df.drop('fecha', inplace=True, axis=1)
    df.drop('goles_loc', inplace=True, axis=1)
    df.drop('goles_vis', inplace=True, axis=1)
    df['dif_gol'] = df['dif_gol_loc'] - df['dif_gol_vis']
    df.drop('dif_gol_loc', inplace=True, axis=1)
    df.drop('dif_gol_vis', inplace=True, axis=1)
    print(df.shape)

    # Remuevo ultimos  partidos de cada equipo (pues no puedo calcular bien la forma)
    df = df.dropna(subset='dif_forma_pond').reset_index(drop=True)  # Son 15 partidos por jornada y saco las ultimas 10 jornadas pues la forma la calculo 10 partidos para atras...
    print(df.shape)
    print(df.head)

    # 4) MODELING
    # Corro 100 modelos y promedio resultados
    N_CORRIDAS = 1
    l_aciertos, l_prob, l_prob_2 = [], [], []
    for i in range(N_CORRIDAS):
        print(" MODELO Nº{} ".format(i).center(120, '#'))

        # Balanceo dataset  (Para mi en este proeycto no conviene balancear el dataset. Puesto que es importante que es mas probable ganar de local que de visitante). Sin embargo entiendo que el modelo puede parecer que da bien pero en realidad siempre decir que gana el local...
        # df = balance_dataset(df_shuf)

        # Separo conjunto de datos en train y test
        df_train, df_test = test_design.separate_train_and_test(df)

        # Implemento Naive Bayes
        df_prob = naive_bayes.train_naive_bayes(df_train)
        print(df_prob)

        # 5) EVALUACION DEL MODELO
        df_result = naive_bayes.predict_naive_bayes(df_prob, df_test)

        n_aciertos = 0
        prob_certeza = 0
        for i in range(len(df_result)):
            y_real = df_result.loc[i, 'y_real']
            y_pred = df_result.loc[i, 'y_pred']

            if y_real == y_pred:
                n_aciertos += 1
                prob_certeza += df_result.loc[i, 'y_pred_prob']

        # Guardo resultados del modelo
        l_aciertos.append(n_aciertos / len(df_result) * 100)
        l_prob.append(prob_certeza / n_aciertos)
        l_prob_2.append(sum(df_result['y_pred_prob']) / len(df_result))
        print(l_aciertos)
        print(l_prob)
        print(l_prob_2)

    print("El % de aciertos es {:.3f}%".format(sum(l_aciertos) / N_CORRIDAS))
    print("La probabilidad de certeza promedio en los aciertos es {:.3f}%".format(sum(l_prob) / N_CORRIDAS))
    print("La probabilidad de certeza promedio es {:.3f}%".format(sum(l_prob_2) / N_CORRIDAS))

    # MATRIZ DE CONFUSION
    confusion_matrix = metrics.confusion_matrix(df_result['y_real'], df_result['y_pred'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=["Local", "Empate", "Visitante"])
    cm_display.plot()
    plt.show()

    df.to_excel("./df_results_naive.xlsx")


main()