import pandas as pd
import numpy as np
from sklearn import metrics
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

    # Convertir el array a un DataFrame de pandas
    df_cm = pd.DataFrame(confusion_matrix)
    df_cm.index.name = "Resultado real"

    # Cambio numeros de conf matrix por etiquetas
    for i, row in df_etiquetas.iterrows():

        # Renombro columna
        df_cm.rename(columns={row['int_value']: row['str_value']}, inplace=True)

        # Renombro filas
        df_cm = df_cm.rename(index={row['int_value']: row['str_value']})
    print(f"\n\nMatriz de confusion:\n {df_cm}")
    return df_cm


def convert_pred_str_to_int(df, name_var_str, name_var_int, df_etiquetas_y): # Esto funciona bien
    """
    Obtengo las predicciones segun la casa de apuestas
    """
    # Convertir de str a num
    for idx, row in df_etiquetas_y.iterrows():
        # print(f"{row['str_value']} --> {row['int_value']}")
        
        # Obtengo indices de partidos con determinado resultado
        idx_to_change = df[df[name_var_str] == row['str_value']].index

        # Reemplazar valores en la columna específica
        df.loc[idx_to_change, name_var_int] = row['int_value']

    return df

def convert_pred_int_to_str(df, name_var_int, name_var_str, df_etiquetas_y): # Esto funciona bien
    """
    Obtengo las predicciones segun la casa de apuestas
    """
    # Convertir de str a num
    for idx, row in df_etiquetas_y.iterrows():
        # print(f"{row['int_value']} --> {row['str_value']}")
        
        # Obtengo indices de partidos con determinado resultado
        idx_to_change = df[df[name_var_int] == row['int_value']].index

        # Reemplazar valores en la columna específica
        df.loc[idx_to_change, name_var_str] = row['str_value']

    return df

def determine_bookmaker_result(df, var_pred):
    
    # Por fila
    for i, row in df.iterrows():

        # Determino la cuota minima de las 3 posibles
        odds_min = min(row['odds_home'], row['odds_draw'], row['odds_away'])

        # Si la cuota minima es la del team home
        if row['odds_home'] == odds_min:
            df.loc[i, var_pred] = "Home"

        # Si la cuota minima es la del team away
        elif row['odds_away'] == odds_min:
            df.loc[i, var_pred] = "Away"

        # Si la cuota minima es la del draw
        else:
            df.loc[i, var_pred] = "Draw"

    return df

def calculate_probas_bookmarker(df_match_odds): # Funciona bien. Comprobado.
    """
    Calcula las probabilidades de cada resultado (Home, Draw y Away) segun la casa de apuesta
    """
    # Por partido
    for idx, row in df_match_odds.iterrows():

        # Calcular probabilidades a partir de invertir las cuotas
        prob_home = 1 / row['odds_home']
        prob_draw = 1 / row['odds_draw']
        prob_away = 1 / row['odds_away']

        # Sumo las probabilidades (deberia ser >1 por el margen de ganancia de la casa de apuesta)
        sum_prob = prob_home + prob_draw + prob_away

        # Calculo probabilidades sin el margen
        prob_home = prob_home / sum_prob
        prob_draw = prob_draw / sum_prob
        prob_away = prob_away / sum_prob

        df_match_odds.loc[idx, ['prob_home_bm', 'prob_draw_bm', 'prob_away_bm']] = [prob_home, prob_draw, prob_away]

    return df_match_odds

def calculate_roi_same_stake(df, _print: bool = False):  # Mismo stake y apuesto a todos los partidos # Deberia adaptarlo para pred en int (en vez de str)?? --> dificulta el calculo de cuotas...
    """
    Calcula ROI comparando las predicciones del modelo y los resultados reales.
    :param df_result: Dataframe de prueba con la variable respuesta y la predicción del modelo. (DataFrame)
    :param var_resp: Nombre de la variable respuesta. (str)
    :param var_pred: Nombre de la variable con la predicción del modelo. (str)
    :return: ROI del modelo. (float)
    """
    # Definicion de variables
    ingresos = 0
    stake = 100  # Suponiendo 1000 pesos por cada partido del df_test
    n_partidos_a_invertir = len(df) # Suponiendo que invierto en todos los partidos
    dinero_a_invertir = stake * n_partidos_a_invertir  

    # Filtrar el dataframe solo a las filas donde el modelo predijo correctamente
    df_correct = df[df['result_str'] == df['predicted_result_str']]
    if _print:
        print(df_correct.shape)

    # Por registro
    for idx in df_correct.index:

        cuota_home =  df_correct.loc[idx, 'odds_home'] 
        cuota_draw = df_correct.loc[idx, 'odds_draw'] 
        cuota_away = df_correct.loc[idx, 'odds_away'] 
        result_etiqueta = df_correct.loc[idx, 'result_str']

        # Obtengo el ingreso obtenido segun la etiqueta
        cuota = cuota_home if result_etiqueta == "Home" else cuota_draw if result_etiqueta == "Draw" else cuota_away  # Vefificada
        ingresos += stake * cuota
        if _print:
            print(f"Ganamos ${stake * cuota}")

    # Calculo el ROI e imprimo resultados
    roi = (ingresos - dinero_a_invertir) / dinero_a_invertir * 100
    # print(f" RESULTADOS ".center(120, "#"))
    print(f"Dinero invertido: ${dinero_a_invertir}")
    print(f"Dinero luego de apuestas: ${ingresos}")
    print(f"ROI: {roi:.2f}%")
    return roi


def calculate_roi_dif_stake(df):  # Mismo stake y apuesto a todos los partidos # Deberia adaptarlo para pred en int (en vez de str)?? --> dificulta el calculo de cuotas...
    """
    Calcula ROI comparando las predicciones del modelo y los resultados reales.
    :param df_result: Dataframe de prueba con la variable respuesta y la predicción del modelo. (DataFrame)
    :param var_resp: Nombre de la variable respuesta. (str)
    :param var_pred: Nombre de la variable con la predicción del modelo. (str)
    :return: ROI del modelo. (float)
    """
    # Definicion de variables
    ingresos = 0
    stake_base = 100  # Suponiendo 1000 pesos por cada partido del df_test
    n_partidos_a_invertir = len(df) # Suponiendo que invierto en todos los partidos
    # dinero_a_invertir = stake_base * n_partidos_a_invertir  

    # calcular cuotas de mi modelo
    df = cuotas_de_mi_modelo(df)
    df = calculate_multiplicador(df)

    # mean_dif_cuota_bm_mod = df['dif_cuota_bm_mod'].mean()
    for i, row in df.iterrows():
        mod = stake_base * row['multiplicador']
        df.loc[i, 'mod'] = mod

        stake_mod = stake_base + mod
        if stake_mod < 0:
            stake_mod = 0

        df.loc[i, 'stake_mod'] = stake_mod
    df.to_excel("/Users/nachomondino/Desktop/df_cuotas.xlsx")

    dinero_a_invertir = sum(df['stake_mod'])

    # Filtrar el dataframe solo a las filas donde el modelo predijo correctamente
    df_correct = df[df['result_str'] == df['predicted_result_str']]
    print(df_correct.shape)
     
    # Por registro
    for idx, row in df_correct.iterrows():

        cuota_home =  row['odds_home']  # df_correct.loc[idx, 'odds_home'] 
        cuota_draw = row['odds_draw']  # df_correct.loc[idx, 'odds_draw'] 
        cuota_away = row['odds_away'] # df_correct.loc[idx, 'odds_away'] 
        result_etiqueta = row['result_str']  # df_correct.loc[idx, 'result_str']

        # Obtengo el ingreso obtenido segun la etiqueta
        cuota = cuota_home if result_etiqueta == "Home" else cuota_draw if result_etiqueta == "Draw" else cuota_away  # Vefificada

        stake = row['stake_mod']
        
        ingreso = stake * cuota        

        ingresos += ingreso
        # print(f"Ganamos ${ingreso}")

    # Calculo el ROI e imprimo resultados
    roi = (ingresos - dinero_a_invertir) / dinero_a_invertir * 100
    # print(f" RESULTADOS ".center(120, "#"))
    print(f"Dinero invertido: ${dinero_a_invertir}")
    print(f"Dinero luego de apuestas: ${ingresos}")
    print(f"ROI: {roi:.2f}%")
    return roi

def cuotas_de_mi_modelo(df):
    """
    Calcula las probabilidades de cada resultado (Home, Draw y Away) segun la casa de apuesta
    """
    # Por partido
    for idx, row in df.iterrows():

        # Calcular probabilidades a partir de invertir las cuotas
        cuota_home = 1 / row['Home']
        cuota_draw = 1 / row['Draw']
        cuota_away = 1 / row['Away']
        cuota_min = min(cuota_home, cuota_draw, cuota_away)


        pred_mod = df.loc[idx, 'predicted_result_str']
        cuota_bm = row['odds_home'] if pred_mod == "Home" else row['odds_draw']  if pred_mod == "Draw" else row['odds_away']   # Vefificada
        dif_cuota_bm_mod = cuota_bm - cuota_min

        df.loc[idx, ['odds_home_mod', 'odds_draw_mod', 'odds_away_mod', 'cuota_min', 'cuota_bm', 'dif_cuota_bm_mod']] = [cuota_home, cuota_draw, cuota_away, cuota_min, cuota_bm, dif_cuota_bm_mod]

    return df

def calculate_multiplicador(df):

    y2, y1 = 1, -2
   
    max_val = max(df['dif_cuota_bm_mod'])
    min_val = min(df['dif_cuota_bm_mod'])
    x2 = np.percentile(df['dif_cuota_bm_mod'], 98)
    x1 = np.percentile(df['dif_cuota_bm_mod'], 2)

    print(max_val, min_val)
    print(y2, x2, y1, x1)
    
    m = (y2-y1) / (x2-x1)
    b = y1-m*x1

    df['multiplicador'] = df['dif_cuota_bm_mod'] * m + b
    return df


def calculate_estrategia_inversion():
    """
    Que calcule la precision y/o ROI obtenido con ≠ estrategias de inversion. Por ejemplo, 
    - Cuando el predictor supera el 60% en clase predicha.
    - Cuando el predictor se opone al resultado de la casa de apuesta
    - etc...
    """
    # stake = # Podria variar y apostar mas en algunos partidos y menos en otros
    # umbral_confianza = # Probabilidad minima de la clase predicha mas probable para apostar en ese partido
    
    # roi_est_1 = calculate_roi(stake="same", umbral_confianza=0.3)
    # roi_est_2 = calculate_roi(stake="same", umbral_confianza=0.6)
    # roi_est_3 = calculate_roi(stake="lineal", umbral_confianza=0.3) # Cuanta mas confianza tiene el predictor, mas dinero
    # roi_est_4 = calculate_roi(stake="exponencial", umbral_confianza=0.3)
    # etc
    pass

def prueba():
    var_resp = 'result'
    var_pred = 'predict_result'

    df = pd.read_excel('/Users/nachomondino/Desktop/df_results.xlsx')

    calculate_precision(df, var_resp=var_resp, var_pred=var_pred)

    # confusion_matrix(df, var_resp, var_pred)
    # df_cm.to_excel('/Users/nachomondino/Desktop/cm.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()


''' Funciones de caro
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

'''