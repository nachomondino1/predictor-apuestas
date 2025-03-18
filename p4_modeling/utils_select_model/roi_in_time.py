import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd
import numpy as np
from utils.set_up_logging import logger
import matplotlib.pyplot as plt

def main(df_ite, country, iteration_date, graf_all_models: bool = False):
    """
    Hacer grafico de ROI sin ea por numero de partido. Asi saber cuando comienza la declive del ROI x falta de representatividad y es momento de reentrenar.
    """
    plt.figure(figsize=(10, 5))  # Crear una única figura antes del loop

    # Listas para combinar los datos de todos los modelos
    all_bet_numbers = []
    all_rois = []
    cont = 0

    # Iterar sobre los modelos
    for idx, row in df_ite.iterrows():
        n_model, model_name = row['n_iteration'], row['model_name'] # n_model
        logger.info(f'{n_model} {model_name}')
        
        # Levanto predicciones del modelo --> no assess.
        try:
            df_pred = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx")
            print(df_pred.shape)
        except FileNotFoundError:
            continue

        # Calculo el ROI por fila
        df_pred['roi_'] = (df_pred['bank_final'] - 100) / 100  # 100 es el bank inicial

        # Crear columna de número de apuesta (1, 2, 3, ...)
        df_pred['bet_number'] = range(1, len(df_pred) + 1)

        # Graficar evolución del ROI en el mismo gráfico
        if cont <= 10:
            plt.plot(df_pred['bet_number'], df_pred['roi_'], marker='o', linestyle='-', label=f'Modelo {n_model} - {model_name}')
        # plt.plot(df_pred['bet_number'], df_pred['recall'], marker='o', linestyle='-', label=f'Modelo {n_model} - {model_name}')

        # Guardar datos combinados
        all_bet_numbers.extend(df_pred['bet_number'])
        all_rois.extend(df_pred['roi_'])
        cont += 1

    # Calcular una única línea de tendencia para todos los modelos
    z = np.polyfit(all_bet_numbers, all_rois, 2)  # Ajuste lineal (grado 1)
    p = np.poly1d(z)  # Crear la función polinómica
    plt.plot(sorted(all_bet_numbers), p(sorted(all_bet_numbers)), color='black', linestyle='--', label='Tendencia General')

    # Personalización del gráfico
    plt.xlabel('Número de Apuesta')
    plt.ylabel('ROI')
    plt.title('Tendencia del ROI por Número de Apuesta')
    plt.axhline(y=0, color='red', linestyle='--', label='ROI = 0')
    plt.legend()
    plt.grid(True)

    # Mostrar gráfico
    plt.show()

def roi_in_time_one_model(n_model, model_name):

    logger.info(f'{n_model} {model_name}')
    
    # Levanto predicciones del modelo (test o test + assess)
    df_pred = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/models/{n_model}__{model_name}_predicciones.xlsx")

    # Calculo el ROI por fila
    df_pred['roi'] = (df_pred['bank_final'] - 100) / 100 # 100 es el bank inicial

    # Crear columna de número de apuesta (1, 2, 3, ...)
    df_pred['bet_number'] = range(1, len(df_pred) + 1)

    # Graficar evolución del ROI con la nueva columna en el eje X
    plt.figure(figsize=(10, 5))
    plt.plot(df_pred['bet_number'], df_pred['roi'], marker='o', linestyle='-', color='b', label='ROI')

    # Personalización del gráfico
    plt.xlabel('Número de Apuesta')
    plt.ylabel('ROI')
    plt.title(f'Evolución del ROI - {n_model} {model_name}')
    plt.axhline(y=0, color='red', linestyle='--', label='ROI = 0')
    plt.legend()
    plt.grid(True)

    # Mostrar gráfico
    plt.show()
    
if __name__ == "__main__":
    # Defino parametroçs
    l_countries = [48, 55, 59, 77, 148]

    d_countries = {
        # 48: ["england", '2025-02-05'],
        # 55: ["france", '2025-02-05'], 
        # 59: ["germany", '2025-02-05'],
        # 77: ["italy", '2025-02-05'],
        # 148: ["spain", '2025-02-05'], 
        # Train nuevos
        # 6: ["argentina", '2025-02-06'], 
        48: ["england", '2025-03-03'],
        55: ["france", '2025-03-03'], 
        59: ["germany", '2025-03-04'],
        77: ["italy", '2025-03-04'],
        148: ["spain", '2025-03-04'], 
        }

    for id_country in l_countries:
        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]

        df_ite = pd.read_excel(f"data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx")
        df_ite = df_ite.sort_values(by='roi', ascending=False)

        # df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/best_model/1_filter_models/df_filt_by_metric_cand.xlsx')
        # df_ite = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/best_model/3_bet_strategy/df_ite_bs.xlsx')
        # df_ite = df_ite.head(10)
        print(df_ite)

        main(
            df_ite=df_ite,
            country=country, iteration_date=iteration_date, 
            )
        