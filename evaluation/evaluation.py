import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

def calculate_precision(df_result, var_resp):

    n_aciertos, prob_certeza = 0, 0
    for i in range(len(df_result)):

        # Si el modelo predijo bien
        if df_result.loc[i, var_resp] == df_result.loc[i, 'y_pred']:
            n_aciertos += 1
            prob_certeza += df_result.loc[i, 'y_pred_prob']

    # Guardo resultados del modelo
    return n_aciertos / len(df_result) * 100

# MATRIZ DE CONFUSION
def confusion_matrix(df):
    confusion_matrix = metrics.confusion_matrix(df['equipo_ganador'], df['y_pred'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=["Local", "Empate", "Visitante"])
    cm_display.plot()
    plt.show()

def main():
    df = pd.read_excel('/Users/nachomondino/Desktop/df_result.xlsx')

    confusion_matrix(df)
    # df_cm.to_excel('/Users/nachomondino/Desktop/cm.xlsx')

main()