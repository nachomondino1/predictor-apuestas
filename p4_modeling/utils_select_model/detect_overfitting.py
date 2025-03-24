# Archivo para medir overfitting y underfitting.
# comparo metricas de trian, val (cv) y test 

# Elige modelos con bajo gap entre train y test, métricas consistentes en validación cruzada y baja desviación estándar (std).

# Deberia comparar error de train y test por valor (e.g. 'ml', '0') de cada hiperparametro de entrenamiento (e.g. fill_na) para ver si cierto valor genera overfitting.
    # Por ejemplo nos va muy mal con el rellenado de datos con machine learning y el f1 score es de 20. Pero solo sucede en esos casos.

import sys
sys.path.append('.')
import pandas as pd
import numpy as np

# Analisis de metricas promedio por hiperparametro de entrenamiento
def detect_overfitting_in_parameter(df):
    """
    Evalúa la variación del error entre train y test para cada valor de los parámetros seleccionados.

    Parámetros:
    - df: DataFrame con las métricas de error y los valores de los parámetros.

    Retorna:
    - Diccionario con la variación del error por parámetro.
    """

    # Lista de parámetros a analizar (deben ser nombres de columnas en df)
    l_params = [
        "comp_to_select", "n_last_matches", "n_anios_hist",
        "segun_localia", "calculate_dif", 'decay_rate',
        "thr_corr", "thr_fs",
        "n_years_to_select", "fill_na", "bal_type"
    ]

    d = {}

    # Iterar por cada parámetro
    for param in l_params:
        if param not in df.columns:
            print(f"⚠️ Advertencia: La columna '{param}' no existe en el DataFrame.")
            continue

        d[param] = {}
        print(f"\n📊 Parámetro: {param}")

        # Iterar por cada valor único del parámetro
        for value in df[param].unique():
            # Si el valor es NaN, reemplazarlo por None
            if pd.isna(value):
                idxs = df[param].isna()
            else:  
                idxs = df[param] == value

            f1_test = df.loc[idxs, 'f1_score'].mean()
            f1_train = df.loc[idxs, 'f1_score_train'].mean()

            test_error = df.loc[idxs, 'error'].mean()
            train_error = df.loc[idxs, 'error_train'].mean()

            if train_error == 0:  # Para evitar división por cero
                var = float('inf') if test_error > 0 else 0
            else:
                var = (test_error - train_error) / train_error * 100
                var_f1 = (f1_test - f1_train) / f1_test * 100

            d[param][value] = var
            # print(f"📊 Parámetro: {param} | Valor: {value} | Variación del error: {var:.0f}%")
            # print(f"📊 Parámetro: {param} | Valor: {value} | \n\tVariación del f1_score: {var_f1:.0f}% \n\tVariación del error: {var:.0f}%")
            print(f"Valor: {value} --> f1_train={f1_train:.1f}% f1_test={f1_test:.1f}% => Var: {var_f1:.0f}%")

    return d

# Analisis de metricas promedio
def detect_overfitting(df_ite):
    """
    Evalúa si hay overfitting comparando F1-score y error entre train, validation y test.

    Parámetros:
    - df_ite: DataFrame con métricas de entrenamiento, validación y test.

    Retorna:
    - Diccionario con medias y desviaciones estándar de las métricas relevantes.
    - Imprime una evaluación sobre posibles casos de overfitting.
    """
    # Definición de métricas a analizar
    metrics = {
        'f1_score': ["cv_f1_score", "f1_score_train", "f1_score"],
        'error': ["cv_cross_entropy_loss", "error_train", 'error']
    }

    # Almacenar resultados para cada métrica
    results = {}

    print("\n📊 **Métricas de F1-score y Error** 📊")
    for metric, l_cols in metrics.items():
        
        print(f"\nMetric: {metric}")

        for col in l_cols:

            if col in df_ite.columns:  # Verificar que la columna existe
                mean = df_ite[col].mean()
                std = df_ite[col].std()
                results[col] = {"mean": mean, "std": std}
                print(f"🔹 {col}: {mean:.3f} ± {std:.2f}")
            else:
                print(f"⚠️ La métrica '{col}' no está presente en el DataFrame.")
                results[col] = {"mean": None, "std": None}

    print("\n🚨 **Evaluación de Overfitting** 🚨")

    # Validación: Comparación entre conjuntos (Train, Validation y Test)
    analyze_overfitting(
        f1_train=results.get("f1_score_train", {}).get("mean"),
        error_train=results.get("error_train", {}).get("mean"),
        f1_cv=results.get("cv_f1_score", {}).get("mean"),
        error_cv=results.get("cv_cross_entropy_loss", {}).get("mean"),
        f1_test=results.get("f1_score", {}).get("mean"),
        error_test=results.get("error", {}).get("mean"),
    )

    f1_test = results.get("f1_score", {}).get("mean")
    f1_bet = df_ite['f1_score_bm'].values[0]
    diff = (f1_test - f1_bet) / f1_bet *100

    corr_f1_score = df_ite['f1_score_train'].corr(df_ite['f1_score']) * 100
    corr_error = df_ite['error_train'].corr(df_ite['error']) * 100
    print(f"Correlacion f1_score train y test: {corr_f1_score:.1f}%")
    print(f"Correlacion error train y test: {corr_error:.1f}%")
    print(f"\nMetric BET: {f1_bet:.1f} => Diff: {diff:.0f}%")

    print("\n✅ Análisis completado.")
    return results

def analyze_overfitting(f1_train, error_train, f1_cv, error_cv, f1_test, error_test, threshold=0.1):
    """
    Analiza si hay indicios de overfitting comparando métricas clave.

    Parámetros:
    - f1_train: F1-score en train.
    - error_train: Error en train.
    - f1_cv: F1-score en validación (CV).
    - error_cv: Error en validación.
    - f1_test: F1-score en test.
    - error_test: Error en test.
    - threshold: Umbral para detectar discrepancias significativas.
    """
    if f1_train and f1_test and f1_train > f1_test * (1 + threshold):
        print(f"⚠️ Posible overfitting: F1-score en Train ({f1_train:.4f}) es significativamente mayor que en Test ({f1_test:.4f}).")

    if error_train and error_test and error_train < error_test * (1 - threshold):
        print(f"⚠️ Posible overfitting: Error en Train ({error_train:.4f}) es mucho menor que en Test ({error_test:.4f}).")

    if f1_cv and f1_test and f1_cv > f1_test * (1 + threshold):
        print(f"⚠️ Posible inconsistencia: F1-score en Validación (CV) ({f1_cv:.4f}) es significativamente mayor que en Test ({f1_test:.4f}).")

    if error_cv and error_test and error_cv < error_test * (1 - threshold):
        print(f"⚠️ Posible inconsistencia: Error en Validación (CV) ({error_cv:.4f}) es mucho menor que en Test ({error_test:.4f}).")


if __name__ == "__main__":
    # Defino parametros
    l_countries = [48, 55, 59, 77, 148]
    l_countries = [148]

    d_countries = { 
        # train nuevos
        48: ["england", '2025-03-23'],
        55: ["france", '2025-03-23'], 
        59: ["germany", '2025-03-23'],
        77: ["italy", '2025-03-23'],
        148: ["spain", '2025-03-24']
        }
    
    for id_country in l_countries:

        country = d_countries[id_country][0]
        iteration_date = d_countries[id_country][1]
        print(f" {country.upper()} ".center(120, "$"))
    
        df = pd.read_excel(f'data/{country}/p4_modeling/{iteration_date}/df_iteration.xlsx')
        
        # Elimino n_years=3 y bal_type=None
        df_ite_filt = df.loc[(df['n_years_to_select']==3) | (df['bal_type'].isna())]
        df.drop(index=df_ite_filt.index, inplace=True)

        detect_overfitting(df)
        detect_overfitting_in_parameter(df)
    
