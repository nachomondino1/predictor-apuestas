import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from data_preparation import format_data


def calculate_ROI(df_result, var_resp, var_pred, df_etiquetas): # NO SE QUE NRO LE CORRESPONDE A QUE CLASE (E.G. SI LOCAL ES 2, 1, 0) --> al parecer lo solucione...
    """
    Calcula ROI comparando las predicciones del modelo y los resultados reales.
    :param df_result: Dataframe de prueba con la variable respuesta y la predicción del modelo. (DataFrame)
    :param var_resp: Nombre de la variable respuesta. (str)
    :param var_pred: Nombre de la variable con la predicción del modelo. (str)
    :return: ROI del modelo. (float)
    """
    # Definicion de variables
    ingresos = 0
    inversion = len(df_result)  # Suponiendo 1 euro por cada partido del df_test

    # Elimino los registros del dataset de testeo que no tienen cuotas --> Ya no es necesario puesto uso fill_na_with_ml
    # df_result = df_result.dropna(subset=['odds_loc', 'odds_emp', 'odds_vis']).reset_index(drop=True)  # no puede haber nan en las odds puesto que es lo que determina el rendimiento del modelo
    df_result = df_result.reset_index(drop=True)
    # print(df_result.shape)

    # Filtrar el dataframe solo a las filas donde el modelo predijo correctamente
    df_correct = df_result[df_result[var_resp] == df_result[var_pred]]
    # print(df_correct.shape)

    # Convierto variable respuesta a etiqueta
    df_correct = format_data.revert_columns_from_int(df_correct, df_etiquetas, columns=[var_resp])
    # print(df_correct[var_resp])

    # Por registro
    for idx in df_correct.index:

        etiqueta = df_correct.loc[idx, var_resp]

        # Obtengo el ingreso obtenido segun la etiqueta
        ingreso = df_result.loc[idx, 'odds_loc'] if etiqueta == "Local" else df_result.loc[idx, 'odds_emp'] if etiqueta == "Empate" else df_result.loc[idx, 'odds_vis']  # Vefificada
        ingresos += ingreso
        # print(f"Ganamos ${ingreso}")

    '''        
    # Por registro
    for i in range(len(df_result)):

        y_real = df_result.loc[i, var_resp]
        y_pred = df_result.loc[i, var_pred]

        # Si el modelo predijo bien
        if y_real == y_pred:

            # Obtengo etiqueda a partir del codigo
            # idx = df_etiquetas[df_etiquetas['Código'] == y_real].index[0]
            # etiqueta = df_etiquetas.loc[idx, 'Etiqueta']
            

            # Obtengo el ingreso obtenido segun la etiqueta
            ingreso = df_result.loc[i, 'odds_loc'] if etiqueta == "Local" else df_result.loc[i, 'odds_emp'] if etiqueta == "Empate" else df_result.loc[i, 'odds_vis']  # Vefificada
            ingresos += ingreso
            # print(f"Ganamos ${ingreso}")
    '''

    # Calculo el ROI e imprimo resultados
    roi = (ingresos - inversion) / inversion
    # print(f" RESULTADOS ".center(120, "#"))
    # print(f"Dinero invertido: ${inversion}")
    # print(f"Dinero luego de apuestas: ${ingresos}")
    # print(f"ROI: {roi:.2f}%")
    return roi

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


def prueba():
    var_resp = 'equipo_ganador'
    var_pred = 'y_pred'

    df = pd.read_excel('/Users/nachomondino/Desktop/df_results.xlsx')

    calculate_precision(df, var_resp=var_resp, var_pred=var_pred)

    calculate_ROI(df, var_resp, var_pred)

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