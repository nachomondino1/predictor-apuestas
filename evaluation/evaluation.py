import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

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