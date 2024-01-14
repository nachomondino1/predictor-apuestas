# Importo librerias
import pandas as pd
from itertools import product
from main import DataPreparation, Modeling
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier


def find_best_hiperparameters(var_resp, var_pred, pais):

    # Definicion de variables
    dp = DataPreparation(var_resp, pais)
    mo = Modeling(var_resp, var_pred, pais)  # Creo objeto de clase Modeling
    df_res = pd.DataFrame()
    df_hiper = pd.DataFrame(columns=['modelo', 'hiper', 'value', 'cuenta'])
    best_accuracy = 0.0
    cont_iter = 1
    l_modelos = [RandomForestClassifier(), LogisticRegression(), SVC(), MLPClassifier()]  # Mejor: RandomForestClassifier() seguido por Logistic # Dejo solo los que tardan poco  # MLPClassifier()  # DecisionTreeClassifier()  xgb.XGBClassifier(), GradientBoostingClassifier()

    # Definicion de hiperparametros
    d_params = {
        'construct': {
            'n_dias_ult_part': [30, 45, 60],  # Nº dias para determinar promedio de estadisticas como posesion
            'n_anios_historial': [3]  # Probar 3 # Nº años para determinar historial entre equipos
        },
        'clean': {
            'thr_nan_col': [None]  # , # Gano 0.2 # Porcentaje de NaN values maximo para las columnas
        },
        'select': {
            'thr_corr': [0.7, 0.5, None], # Correlacion minima para considerar correlacion entre variables
            'thr_fs': [0.4, 0.2, None]  # Muy parecido, el mejorcito fue 0.2 pero por nada (x2) # Peso minimo de una variable para ser considerada como importante [0-1] (siendo 1 el peso de la variable mas importante y 0 la menos)
        },
        'modeling': {
            'test_val_size': [0.3],  # Proporcion de datos destinado a test y validation, el resto es train
            'test_size': [0.5],  # Proporcion de datos destinado test, el resto es validation
            'fill_na': [None, 'ml'], # Opcion de rellenar NaN values en dataset de entrenamiento
            'bal_type': [None],  # Opcion de balancear dataset de entrenamiento
            'with_pca': [True],
            'k': [5]  # Numero de folds tanto para seleccionar hiperparametros como para entrenar el modelo
        }
    }

    # Imprimo largo de iteraciones
    n_iter = define_n_iterations(d_params)
    print(f"Numero de iteraciones totales: {n_iter}")

    # Levanto dataset formateado e integrado (estos no cambian entre iteraciones)
    df_integrated = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_integrated.xlsx')

    # Por combinacion de parametros de construct_data
    for i, param_values_2 in enumerate(product(*d_params['construct'].values()), start=1):

        # Asigno valor a cada hiperpametro
        n_dias_ult_part, n_anios_historial = param_values_2[0], param_values_2[1]
        print(f" Iter Nº {i} ".center(120, "#"))
        print(f'Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_anios_historial: {n_anios_historial}')

        # Construyo datos
        try:
            df_constructed = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_constructed_{n_dias_ult_part}_{n_anios_historial}.xlsx')
        except FileNotFoundError:
            df_constructed = dp.construct_data(df_integrated, n_dias=n_dias_ult_part, n_anios_historial=n_anios_historial)
            df_constructed.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_constructed_{n_dias_ult_part}_{n_anios_historial}.xlsx', index=False)

        # Por combinacion de parametros de clean_data
        for m, param_values_3 in enumerate(product(*d_params['clean'].values()), start=1):

            # Asigno valor a cada hiperpametro
            thr_nan_col = param_values_3[0]
            print(f" Iter Nº {i}.{m} ".center(120, "#"))
            print(f"Hiper clean --> thr_nan_col: {thr_nan_col}")

            # Limpio datos
            df_clean = dp.clean_data(df_constructed, thr_nan_col=thr_nan_col, export=False)
            # df_clean.to_excel('/Users/nachomondino/Desktop/df_clean.xlsx', index=False)

            # Por combinacion de parametros de select_data
            for j, param_values_4 in enumerate(product(*d_params['select'].values()), start=1):

                # Asigno valor a cada hiperpametro
                thr_corr, thr_fs = param_values_4[0], param_values_4[1]
                print(f" Iter Nº {i}.{m}.{j} ".center(120, "#"))
                print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")

                # Selecciono datos
                df_sel = dp.select_data(df_clean, thr_corr=thr_corr, thr_fs=thr_fs, export=False)
                # df_sel.to_excel('/Users/nachomondino/Desktop/df_sel.xlsx', index=False)

                # Por combinacion de parametros de modeling
                for h, param_values_5 in enumerate(product(*d_params['modeling'].values()), start=1):

                    # Asigno valor a cada hiperparametro
                    test_val_size, test_size, fill_na, bal_type, with_pca, k = param_values_5[0], param_values_5[1], param_values_5[2], param_values_5[3], param_values_5[4], param_values_5[5]
                    print(f" Iter Nº {i}.{m}.{j}.{h} ".center(120, "#"))
                    print(f" Iteracion Nº {cont_iter} ")
                    print(f'Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_anios_historial: {n_anios_historial}')
                    print(f"Hiper clean --> thr_nan_col: {thr_nan_col}")
                    print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")
                    print(f"Hiper modeling --> test_val_size: {test_val_size} ; test_size: {test_size}; fill_na: {fill_na} ; bal_type: {bal_type} ; with_pca: {with_pca} ; k: {k}")

                    df_models = pd.DataFrame(columns=['model_name', 'model_trained', 'train_cv_accuracy', 'test_accuracy', 'test_recall', 'test_f1_score'])  # Datos del modelo y su precision y roi

                    # Generar el diseño de la prueba
                    X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df_sel, bal_type=bal_type, test_val_size=test_val_size, test_size=test_size, fill_na=fill_na, with_pca=with_pca)

                    # Por modelo
                    for modelo in l_modelos:

                        # Entreno modelo y evaluo su rendimiento
                        model_name, model_best_params, cv_accuracy = mo.build_model(modelo, X_val, y_val, X_train, y_train, k)
                        accuracy, recall, f1 = mo.assess_model(model_best_params, X_test, y_test, export=False)

                        # Obtengo hiperparametros optimos del modelo
                        d_best_params = model_best_params.get_params()

                        # Guardo modelo
                        df_models.loc[len(df_models)] = [model_name, d_best_params, cv_accuracy, accuracy, recall, f1]

                        # Guardo mejores de hiperparametros
                        for hiper, value in d_best_params.items():
                            fila_deseada = df_hiper.loc[(df_hiper['modelo'] == model_name) & (df_hiper['hiper'] == hiper) & (df_hiper['value'] == value)]

                            # Si ya tiene cuenta
                            if len(fila_deseada) > 0:
                                df_hiper.loc[fila_deseada.index, 'cuenta'] += 1
                            else:
                                hiper_data = [{'modelo': model_name, 'hiper': hiper, 'value': value, 'cuenta': 1}]
                                df_hiper = pd.concat([df_hiper, pd.DataFrame(hiper_data)], axis=0).reset_index(drop=True)

                    # Selecciono el mejor modelo
                    idx = df_models['test_accuracy'].idxmax()
                    bm_name = df_models.loc[idx, 'model_name']
                    bm_params = df_models.loc[idx, 'model_trained']
                    bm_train_acc = df_models.loc[idx, 'train_cv_accuracy']
                    bm_test_acc = df_models.loc[idx, 'test_accuracy']
                    bm_test_rec = df_models.loc[idx, 'test_recall']
                    bm_test_f1 = df_models.loc[idx, 'test_f1_score']
                    print(f"\nEl mejor modelo es: {bm_name} con: "
                          f"\n\t- Train Precision: {bm_train_acc:.1f}% "
                          f"\n\t- Test Precision: {bm_test_acc:.1f}% "
                          f"\n\t- Test recall: {bm_test_rec:.1f}%"
                          f"\n\t- Test f1-score: {bm_test_f1:.1f}%"
                          )

                    # Guardo datos en dataframe
                    row_data = {'n_dias_ult_part': n_dias_ult_part, 'n_anios_hist': n_anios_historial,
                                'thr_nan_col': thr_nan_col, 'thr_corr': thr_corr, 'thr_fs': thr_fs,
                                'fill_na': fill_na, 'bal_type': bal_type, 'with_pca': with_pca ,
                                'test_val_size': test_val_size, 'test_size': test_size, 'X_train': X_train.shape,
                                'X_val': X_val.shape, 'X_test': X_test.shape, "X_columns": list(X_train.columns),
                                'k': k, 'best_model': bm_name, 'best_model_params': bm_params,
                                'test_recall': bm_test_rec, 'test_f1': bm_test_f1,
                                'train_accuracy': bm_train_acc, 'test_accuracy': bm_test_acc}
                    df_res = df_res.append(row_data, ignore_index=True)
                    df_res.to_excel('/Users/nachomondino/Desktop/df_best_hyper_df_mod.xlsx', index=False)
                    df_hiper.to_excel('/Users/nachomondino/Desktop/df_hiper.xlsx', index=False)

                    cont_iter += 1
                    # Verificar si la precisión actual es la mejor hasta ahora
                    if bm_test_acc > best_accuracy:
                        best_accuracy = bm_test_acc
                        best_hyperparameters = row_data
                        print("EL MEJOR MODELO HASTA AHORA!")
                    else:
                        print(f"Tiene un precision menor a {best_accuracy}")

    # Imprimir los hiperparámetros óptimos y la precisión correspondiente
    print("Mejores hiperparámetros:", best_hyperparameters)
    print("Precisión obtenida:", best_accuracy)
    df_res.to_excel('/Users/nachomondino/Desktop/df_best_hyper_df_mod.xlsx', index=False)
    return best_hyperparameters

def define_n_iterations(d_params):
    """
    Calcula el numero de iteraciones y el tiempo estimado para terminar
    :param l_dicts: Lista de diccionarios de hiperparametros. (list)
    :return: Numero de iteraciones y tiempo estimado (int y float)
    """
    # Cuidado con el nro de iteraciones sobretodo en select. nº comb = producto de posibles comb de cada hiper  EJ: {'thr_corr': [0.5, 0.6, 0.7], 'thr_fs': [0.25, 0.2, 0.15, 0.1, 0.05], 'thr_nan_col': [0.2, 0.5, None]} --> nºcomb = 3x5x3=45
    # n comb totales = 3 x 45 x 6 = 810 --> 90 iteraciones en 17 horas --> 5,35 iter/hora => 810 iteraciones = 151 horas  # n comb totales = 1 x 18 x 6 = 108 --> 90 iteraciones en 17 horas --> 5,35 iter/hora => 216 iteraciones = 151 horas
    n_iter = 1
    for task, d_params_task in d_params.items():
        for key in d_params_task.keys():
            n_iter *= len(d_params_task[key])
    return n_iter

def main():
    pais, var_resp, var_pred = "England", 'equipo_ganador', 'y_pred'
    find_best_hiperparameters(var_resp, var_pred, pais)

if __name__ == '__main__':
    main()