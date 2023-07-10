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
    df_res = pd.DataFrame()
    best_accuracy = 0.0
    l_modelos = [DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier(), LogisticRegression(),
                 SVC(), MLPClassifier(), GradientBoostingClassifier()]
    dp = DataPreparation(var_resp, pais)
    mo = Modeling(var_resp, var_pred, pais)  # Creo objeto de clase Modeling

    # Definicion de hiperparametros
    param_integrate = {'fill_with_fs': [True, False], 'n_dias_player_data': [365]}
    param_construct = {'n_dias': [30], 'n_anios_historial': [2]}
    param_clean = {'thr_nan_col': [0.2, None]}
    param_select = {'thr_corr': [0.5, None], 'thr_fs': [0.2, 0.1, 0.3]}  # 'thr_nan_col': [0.2, 0.5, None]
    param_mod = {'test_val_size': [0.2], 'test_size': [0.5], 'fill_na': [None, 'ml'],
                 'bal_type': [None, 'under', 'over'], 'k': [5]}

    # Imprimo largo de iteraciones
    n_iter = define_n_iterations(l_dicts=[param_construct, param_select, param_mod])
    print(f"Numero de iteraciones totales: {n_iter}")

    for x, param_values_1 in enumerate(product(*param_integrate.values()), start=1):

        # Asigno valor a cada hiperpametro
        fill_with_fs, n_dias_player_data = param_values_1[0], param_values_1[1]
        print(f'Hiper integrate --> fill_with_fs: {fill_with_fs} ; n_dias_player_data: {n_dias_player_data}')

        # Integro datos
        df = dp.integrate_data(df_part, df_jug_part, df_jug, fill_data_with_flashcore=fill_with_fs, n_dias_player_data=n_dias_player_data)

        # Por combinacion de parametros de construct_data
        for i, param_values_2 in enumerate(product(*param_construct.values()), start=1):

            # Asigno valor a cada hiperpametro
            n_dias, n_anios_historial = param_values_2[0], param_values_2[1]
            print(f" Iteracion Nº {i} ".center(120, "#"))
            print(f'Hiper construct --> n_dias: {n_dias} ; n_anios_historial: {n_anios_historial}')

            # Pruebo a levantar dataset ya construido
            try:
                df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_constructed_{n_dias}_{n_anios_historial}.xlsx')

            except:
                # Levanto dataset integrado (es el mismo siempre)
                df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_integrated.xlsx')

                # Construyo datos
                df = dp.construct_data(df, n_dias=n_dias, n_anios_historial=n_anios_historial)
                df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_constructed_{n_dias}_{n_anios_historial}.xlsx', index=False)

            for m, param_values_3 in enumerate(product(*param_clean.values()), start=1):

                # Asigno valor a cada hiperpametro
                thr_nan_col = param_values_3[0]

                # Limpio datos
                df = dp.clean_data(df, thr_nan_col=thr_nan_col, export=False)
                print(f"Hiper clean --> thr_nan_col: {thr_nan_col}")

                # Por combinacion de parametros de select_data
                for j, param_values_4 in enumerate(product(*param_select.values()), start=1):

                    # Asigno valor a cada hiperpametro
                    thr_nan_col, thr_corr, thr_fs = param_values_4[0], param_values_4[1], param_values_4[2]
                    print(f" Iteracion Nº {i}.{j} ".center(120, "+"))
                    print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")

                    # Selecciono datos
                    df_sel = dp.select_data(df, thr_nan_col=thr_nan_col, thr_corr=thr_corr, thr_fs=thr_fs, export=False)
                    # df_sel.to_excel(f'/Users/nachomondino/Desktop/df_selected_{pais}.xlsx')

                    # Por combinacion de parametros de modeling
                    for h, param_values_5 in enumerate(product(*param_mod.values()), start=1):

                        # Asigno valor a cada hiperparametro
                        test_val_size, test_size, fill_na, bal_type, k = param_values_5[0], param_values_5[1], param_values_5[2], param_values_5[3], param_values_5[4]
                        print(f" Iteracion Nº {i}.{j}.{h} ".center(120, "-"))
                        print(f"Hiper modeling --> test_val_size: {test_val_size} ; test_size: {test_size}; fill_na: {fill_na} ; bal_type: {bal_type} ; k: {k}")

                        df_models = pd.DataFrame(columns=['model_name', 'model_trained', 'train_cv_accuracy', 'test_accuracy', 'test_recall', 'test_f1_score'])  # Datos del modelo y su precision y roi

                        # Generar el diseño de la prueba
                        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df_sel, bal_type=bal_type, test_val_size=test_val_size, test_size=test_size, fill_na=fill_na)

                        # Por modelo
                        for modelo in l_modelos:

                            # Entreno modelo y evaluo su rendimiento
                            model_name, model_best_params, cv_accuracy = mo.build_model(modelo, X_val, y_val, X_train, y_train, k)
                            accuracy, recall, f1 = mo.assess_model(model_best_params, X_test, y_test, export=False)

                            # Guardo modelo
                            df_models.loc[len(df_models)] = [model_name, model_best_params, cv_accuracy, accuracy, recall, f1]

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
                        row_data = {'n_dias': n_dias, 'n_anios_hist': n_anios_historial,
                                    'thr_nan_col': thr_nan_col, 'thr_corr': thr_corr, 'thr_fs': thr_fs,
                                    'fill_na': fill_na, 'bal_type': bal_type, 'test_val_size': test_val_size, 'test_size': test_size, 'X_train': X_train.shape,
                                    'X_val': X_val.shape, 'X_test': X_test.shape, "X_columns": list(X_train.columns),
                                    'k': k, 'best_model': bm_name, 'best_model_params': bm_params,
                                    'test_recall': bm_test_rec, 'test_f1': bm_test_f1,
                                    'train_accuracy': bm_train_acc, 'test_accuracy': bm_test_acc}
                        df_res = df_res.append(row_data, ignore_index=True)
                        print(row_data)
                        print(df_res)
                        df_res.to_excel('/Users/nachomondino/Desktop/df_best_hyper_df_mod.xlsx', index=False)

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

def define_n_iterations(l_dicts):
    """
    Calcula el numero de iteraciones y el tiempo estimado para terminar
    :param l_dicts: Lista de diccionarios de hiperparametros. (list)
    :return: Numero de iteraciones y tiempo estimado (int y float)
    """
    # Cuidado con el nro de iteraciones sobretodo en select. nº comb = producto de posibles comb de cada hiper  EJ: {'thr_corr': [0.5, 0.6, 0.7], 'thr_fs': [0.25, 0.2, 0.15, 0.1, 0.05], 'thr_nan_col': [0.2, 0.5, None]} --> nºcomb = 3x5x3=45
    # n comb totales = 3 x 45 x 6 = 810 --> 90 iteraciones en 17 horas --> 5,35 iter/hora => 810 iteraciones = 151 horas  # n comb totales = 1 x 18 x 6 = 108 --> 90 iteraciones en 17 horas --> 5,35 iter/hora => 216 iteraciones = 151 horas
    n_iter = 1
    for dict in l_dicts:
        for key in dict.keys():
            n_iter *= len(dict[key])
    return n_iter


def main():
    pais, var_resp, var_pred = "argentina_south_america", 'equipo_ganador', 'y_pred'
    find_best_hiperparameters(var_resp, var_pred, pais)

if __name__ == '__main__':
    main()