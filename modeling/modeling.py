import pandas as pd
from dspy.data_preparation import clean_data
from dspy.modeling import naive_bayes, test_design
from evaluation.evaluation import calculate_precision

def main():
    # Definicion de variables
    N_MODELOS = 10
    l_aciertos = []

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Desktop/df_prepared.xlsx', index_col=0)
    print(df.head())

    # Balanceo dataset y elimino filas con historial=NaN
    df = df.dropna(subset=['historial_entre_si']).reset_index()  # Elimina filas con al menos un valor nulo
    # df = clean_data.balance_dataset(df, var_resp='equipo_ganador')

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

    print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos)/len(l_aciertos)}")

main()