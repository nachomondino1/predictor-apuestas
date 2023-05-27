import pandas as pd
# from sklearn import metrics
import matplotlib.pyplot as plt


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

def calculate_ROI(df_result, var_resp, var_pred):
    """
    Calcula roi comparando sus predicciones y lo real
    :param df_result: Dataframe test. Con variable respuesta y con la prediccion del modelo
    :param var_resp: String. Nombre de la variable respuesta
    :param var_pred: String. Nombre de la variable con la prediccion del modelo
    :return: Float. ROI del modelo
    """
    ingresos = 0
    df_result = df_result.dropna(subset=['odds_loc', 'odds_emp', 'odds_vis']).reset_index(drop=True)  # no puede haber nan en las odds puesto que es lo que determina el rendimiento del modelo
    inversion = len(df_result)  # Suponiendo 1 euro por cada partido del df_test

    # Por registro
    for i in range(len(df_result)):

        # Si el modelo predijo bien
        if df_result.loc[i, var_resp] == df_result.loc[i, var_pred]:

            # Si es local --> saco cuota de local, si es empate saco cuota de empate y asi
            if df_result.loc[i, var_resp] == "Local":

                print(f"Ganamos ${df_result.loc[i, 'odds_loc']}")
                ingresos += df_result.loc[i, 'odds_loc']

            elif df_result.loc[i, var_resp] == "Empate":
                print(f"Ganamos ${df_result.loc[i, 'odds_emp']}")
                ingresos += df_result.loc[i, 'odds_emp']

            else:
                print(f"Ganamos ${df_result.loc[i, 'odds_vis']}")
                ingresos += df_result.loc[i, 'odds_vis']

    roi = (ingresos - inversion) / inversion * 100
    print(f" RESULTADOS ".center(120, "#"))
    print(f"Dinero invertido: ${inversion}")
    print(f"Dinero luego de apuestas: ${ingresos}")
    print(f"ROI: {roi:.2f}%")
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

def entrenamiento_y_calcular_metricas(self, model, n_folds_cv: int = 10, select_best_by: str = 'accuracy') -> None:
    '''
    Calcula métricas de evaluación
    '''
    # Especificar las métricas que se desean calcular
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    # Realizar validación cruzada y obtener los resultados
    if isinstance(model, keras.models.Sequential):  #  Si es una red neuronal, el y que usamos tiene que ser de tipo one hot encoder
        print("ES UNA NN")
        self.cv_results = cross_validate(model, self.X_bal, self.y_bal_encoded, cv=n_folds_cv, scoring=scoring,
                                         return_train_score=True, return_estimator=True)
    else:
        self.cv_results = cross_validate(model, self.X_bal, self.y_bal, cv=n_folds_cv, scoring=scoring,
                                         return_train_score=True, return_estimator=True)
    cv_score = self.cross_validation(model)

    # Imprimir los resultados promedio de cada métrica
    print("Cross Validation Score:", cv_score)
    print("Accuracy: {:.3f}".format(self.cv_results['test_accuracy'].mean()))
    print("Precision: {:.3f}".format(self.cv_results['test_precision_macro'].mean()))
    print("Recall: {:.3f}".format(self.cv_results['test_recall_macro'].mean()))
    print("F1 score: {:.3f}".format(self.cv_results['test_f1_macro'].mean()))

    # Graficar matriz de confusion del mejor resultado
    self.graficar_matriz_conf(n_folds_cv)

    # Graficar curva ROC AUC
    self.graficar_curva_roc()
    plt.show()

def main():
    df = pd.read_excel('/Users/nachomondino/Desktop/df_result.xlsx')

    calculate_precision(df, var_resp='equipo_ganador', var_pred='y_pred')

    confusion_matrix(df, var_resp='equipo_ganador', var_pred='y_pred')
    # df_cm.to_excel('/Users/nachomondino/Desktop/cm.xlsx')

# main()