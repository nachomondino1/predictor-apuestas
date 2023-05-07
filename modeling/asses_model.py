import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


def calculate_precision(df_result, var_resp, var_pred):
    """
    Calcula precision del modelo comparando sus predicciones y lo real
    :param df_result: Dataframe test. Con variable respuesta y con la prediccion del modelo
    :param var_resp: String. Nombre de la variable respuesta
    :param var_pred: String. Nombre de la variable con la prediccion del modelo
    :return: Float. Precision del modelo en sus predicciones
    """
    n_aciertos = 0

    # Por registro
    for i in range(len(df_result)):

        # Si el modelo predijo bien, sumo 1
        if df_result.loc[i, var_resp] == df_result.loc[i, var_pred]:
            n_aciertos += 1
    return n_aciertos / len(df_result) * 100

def confusion_matrix(df_result, var_resp, var_pred):
    """
    Muestra matriz de confusion del modelo
    :param df_result: Dataframe test. Con variable respuesta y con la prediccion del modelo
    :param var_resp: String. Nombre de la variable respuesta
    :param var_pred: String. Nombre de la variable con la prediccion del modelo
    """
    # Calculo los numeros para la matriz de confusion
    confusion_matrix = metrics.confusion_matrix(df_result[var_resp], df_result[var_pred])

    # Imprimo matriz de confusion
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=df_result[var_resp].unique())
    cm_display.plot(cmap='Blues')
    plt.show()

def main():
    df = pd.read_excel('/Users/nachomondino/Desktop/df_result.xlsx')

    calculate_precision(df, var_resp='equipo_ganador', var_pred='y_pred')

    confusion_matrix(df, var_resp='equipo_ganador', var_pred='y_pred')
    # df_cm.to_excel('/Users/nachomondino/Desktop/cm.xlsx')

# main()