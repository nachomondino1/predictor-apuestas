import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report

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

def confusion_matrix(y_real, y_pred, df_etiquetas):
    """
    Muestra matriz de confusion del modelo
    :param df_result: Dataframe test. Con variable respuesta y con la prediccion del modelo
    :param var_resp: String. Nombre de la variable respuesta
    :param var_pred: String. Nombre de la variable con la prediccion del modelo
    """
    # Calculo los numeros para la matriz de confusion
    confusion_matrix = metrics.confusion_matrix(y_real, y_pred)  # numpy.ndarray

    # Imprimo matriz de confusion (evito este codigo porque tenes que cerrar el plot para que el programa continue)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=y_real.unique())
    # cm_display.plot(cmap='Blues')
    # plt.show()

    # Convertir el array a un DataFrame de pandas
    df_cm = pd.DataFrame(confusion_matrix)
    df_cm.index.name = "Resultado real"

    # Cambio numeros de conf matrix por etiquetas
    for i, row in df_etiquetas.iterrows():

        # Renombro columna
        df_cm.rename(columns={row['valor_int']: row['valor_orig']}, inplace=True)

        # Renombro filas
        df_cm = df_cm.rename(index={row['valor_int']: row['valor_orig']})

    print(f"\n\nMatriz de confusion:\n {df_cm}")
    return df_cm

def prueba():
    var_resp = 'equipo_ganador'
    var_pred = 'y_pred'

    df = pd.read_excel('/Users/nachomondino/Desktop/df_results.xlsx')

    calculate_precision(df, var_resp=var_resp, var_pred=var_pred)

    # confusion_matrix(df, var_resp, var_pred)
    # df_cm.to_excel('/Users/nachomondino/Desktop/cm.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()


''' Funciones de caro
def graficar_curva_roc(self) -> None:
    y_true = self.y_bal.values
    classes = np.unique(y_true)
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true == c, self.y_pred)
        roc_auc[c] = auc(fpr[c], tpr[c])

    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def graficar_matriz_conf(self, n_cv: int) -> None:
    # Calcular matriz de confusion, obteniendo el mejor modelo
    best_model_idx = self.cv_results['test_precision_macro'].argmax()
    best_model = self.cv_results['estimator'][best_model_idx]
    self.y_pred = cross_val_predict(best_model, self.X_bal, self.y_bal, cv=n_cv)
    
    # Calculamos la matriz de confusión utilizando los datos de prueba
    conf_mat = confusion_matrix(self.y_bal, self.y_pred)
    print("Precision del mejor modelo: {:.3f}".format(self.cv_results['test_precision_macro'][best_model_idx]))
    print(conf_mat)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    cm_display.plot(cmap='Blues')
    """
    # Elegir el mejor modelo y calcular la matriz de confusión
    y_pred = cross_val_predict(model, self.X_bal, self.y_bal, cv=n_folds_cv)
    conf_mat = confusion_matrix(self.y_bal, y_pred)
    print("y_pred.shape ", y_pred.shape)
    print(conf_mat)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    cm_display.plot(cmap='Blues')
    plt.show()
    """

'''