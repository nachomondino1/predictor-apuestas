# Importo librerias
import pandas as pd
from itertools import product
from main import DataPreparation, Modeling
from modeling import build_model
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
import lightgbm as lgb  # Gradient Boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier


def find_best_hiperparameters(df_part, df_jug, var_resp, var_pred, pais):

    # Definicion de variables
    df_res = pd.DataFrame()
    best_accuracy = 0.0
    l_modelos = [DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier(), MLPClassifier()]
    dp = DataPreparation(df_part, df_jug, var_resp, pais)

    # Hiperparametros de construct_data
    param_construct = {'n_ult_part': [3, 5, 10, 15]}  # n_ult_part --> mas tirando a 10

    # Por combinacion de parametros
    for i, param_values in enumerate(product(*param_construct.values()), start=1):

        # Levanto dataset integrado
        df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_integrated.xlsx')

        n_ult_part = param_values[0]

        # Construyo datos
        df = dp.construct_data(df, N_ULT_PART=n_ult_part, export=False)
        df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_integrated_{n_ult_part}.xlsx')
        print(f" Iteracion Nº {i} ".center(120, "#"))

        # Hiperparametros de select_data
        param_select = {'thr_corr': [0.7, 0.6, 0.5], 'thr_fs': [0.1, 0.2, 0.3], 'thr_nan_col': [0.2, 0.5, None]}  # thr_fs --> mas tirando a 0.2 # thr_corr --> medio indep # thr_nan_col --> 0.2 es casi lo mismo que 0.35 pues solo no borra historial_entre_si que igual no es tenida en cuenta nunca por feature_selection

        # Por combinacion de parametros
        for j, param_values in enumerate(product(*param_select.values()), start=1):

            thr_corr, thr_fs, thr_nan_col = param_values[0], param_values[1], param_values[2]
            print(f" Iteracion Nº {i}.{j} ".center(120, "+"))
            print(f"thr_corr: {thr_corr} ; thr_fs: {thr_fs} ; thr_nan_col: {thr_nan_col}")

            # Preparo el dataset para el analisis
            df_sel = dp.select_data(df, thr_corr=thr_corr, thr_fs=thr_fs, thr_nan_col=thr_nan_col, export=False)

            # Creo objeto de clase Modeling con df_selected
            mo = Modeling(df_sel, var_resp, var_pred, pais)

            # Hiperparametros de modeling
            param_dist_mod = {'bal_type': [None, 'under', 'over'], 'test_val_size': [0.2], 'test_size': [0.5], 'k': [5], 'treat_nan': ['drop', 'ml']}  # bal_type =  'over', 'under'

            # Por combinacion de parametros
            for h, param_val in enumerate(product(*param_dist_mod.values()), start=1):

                # Hiperparametros
                bal_type, test_val_size, test_size, k, treat_nan = param_val[0], param_val[1], param_val[2], param_val[3], param_val[4]
                print(f" Iteracion Nº {i}.{j}.{h} ".center(120, "-"))
                print(f"bal_type: {bal_type} ; treat_nan: {treat_nan}")
                print(f"test_val_size: {test_val_size} ; test_size: {test_size} ; k: {k}")

                df_models = pd.DataFrame(columns=['model_name', 'model_trained', 'train_cv_accuracy', 'test_accuracy', 'test_roi'])  # Datos del modelo y su precision y roi

                # Generar el diseño de la prueba
                X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(bal_type, test_val_size, test_size, treat_nan=treat_nan)

                # Por modelo
                for modelo in l_modelos:

                    # Entreno modelo y evaluo su rendimiento
                    model_name, model_best_params, cv_accuracy = mo.build_model(modelo, X_val, y_val, X_train, y_train, k)
                    test_accuracy, test_roi = mo.assess_model(model_best_params, X_test, y_test, export=False)

                    # Guardo modelo
                    df_models.loc[len(df_models)] = [model_name, model_best_params, cv_accuracy, test_accuracy, test_roi]

                # Selecciono el mejor modelo
                idx = df_models[df_models['test_accuracy'] == max(df_models['test_accuracy'])].index[0]
                best_model = df_models.loc[idx, 'model_trained']
                best_model_train_prec = df_models.loc[idx, 'train_cv_accuracy']
                best_model_test_prec = df_models.loc[idx, 'test_accuracy']
                best_model_test_roi = df_models.loc[idx, 'test_roi']
                print(f"\nEl mejor modelo es: {best_model} con: \n\t- Train Precision: {best_model_train_prec:.1f}% "
                      f"\n\t- Test Precision: {best_model_test_prec:.1f}% \n\t- Test ROI: {best_model_test_roi:.1f}%\n")

                # Guardo datos en dataframe
                row_data = {'N_ULT_PART': n_ult_part, 'thr_nan_col': thr_nan_col, 'thr_corr': thr_corr, 'thr_fs': thr_fs,
                            'treat_nan': treat_nan, 'bal_type': bal_type, 'test_val_size': test_val_size,
                            'test_size': test_size, 'X_train': X_train.shape, 'X_val': X_val.shape,
                            'X_test': X_test.shape, "X_columns": list(X_train.columns), 'k': k, 'best_model': best_model,
                            'train_accuracy': best_model_train_prec, 'test_accuracy': best_model_test_prec,
                            'test_roi': best_model_test_roi}
                df_res = df_res.append(row_data, ignore_index=True)
                print(row_data)
                print(df_res)
                df_res.to_excel('/Users/nachomondino/Desktop/df_best_hyper_df_mod.xlsx', index=False)

                # Verificar si la precisión actual es la mejor hasta ahora
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_hyperparameters = row_data

    # Imprimir los hiperparámetros óptimos y la precisión correspondiente
    print("Mejores hiperparámetros:", best_hyperparameters)
    print("Precisión obtenida:", best_accuracy)
    df_res.to_excel('/Users/nachomondino/Desktop/df_best_hyper_dp.xlsx', index=False)
    return best_hyperparameters


def main():
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    pais = "argentina"
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_partido.xlsx')
    df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_jugadores.xlsx')
    find_best_hiperparameters(df_part, df_jug, var_resp, var_pred, pais)

if __name__ == '__main__':
    main()