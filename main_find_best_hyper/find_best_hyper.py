# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from itertools import product
from main import DataPreparation, Modeling
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
import pickle
import warnings
import os

def find_best_hiperparameters(var_resp, var_pred, country):

    # Definicion de variables
    dp = DataPreparation(country)
    mo = Modeling(var_resp, var_pred, country)  # Creo objeto de clase Modeling
    df_iteration = pd.DataFrame()
    best_roi_max = -100
    cont_iter = 0
    l_modelos = [RandomForestClassifier(), LogisticRegression()] # SVC(), MLPClassifier(), GradientBoostingClassifier()]  # , XGBClassifier(), 

    # Creo directorio automaticamente
    make_directories(country)

    # Definicion de hiperparametros
    d_params = {
        'construct': {
            'n_dias_ult_part': [30, 45],
            'n_years_h2h': [3],
            'segun_localia': [False],
        },
        'select': {
            'thr_corr': [0.7, 0.8, 0.9, None],
            'thr_fs': [0.2, 0.1, None], 
        },
        'treat_nan': {
            'fill_na': [None], # 'ml', 'mode', 
        },
        'modeling': {
            'val_size': [0.125],
            'test_size': [0.125], 
            'bal_type': ['under', None], # 'over'
            'k': [10]
        }
    }

    # Imprimo largo de iteraciones
    n_iter = define_n_iterations(d_params)
    print(f"Numero de iteraciones totales: {n_iter}")

    # Por combinacion de parametros de construct_data
    for i, param_values_2 in enumerate(product(*d_params['construct'].values()), start=1):

        # Asigno valor a cada hiperpametro
        n_dias_ult_part, n_years_h2h, segun_localia= param_values_2[0], param_values_2[1], param_values_2[2]
        print(f" Iteracion Construct Nº {i} ".center(120, "#"))
        print(f'Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h}; segun_localia: {segun_localia}')

        # Construyo datos
        path = f'./main_find_best_hyper/data/{country}/data_preparation/df_constructed_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}.xlsx'
        try:
            df_constructed = pd.read_excel(path, index_col=0)
            print("\n DF_CONSTRUCTED \n", df_constructed.head(2))
        except FileNotFoundError:
            # Levanto dataset formateado e integrado (estos no cambian entre iteraciones)
            df_integrated = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
            print("\n DF_INTEGRATED \n", df_integrated.shape, df_integrated.head(2))

            df_constructed = dp.construct_data(df_integrated, n_days=n_dias_ult_part, n_years_h2h=n_years_h2h, segun_localia=segun_localia, export=False)
            df_constructed.to_excel(path, index=True)

        # Etiqueto df_constructed
        df_cons_etiquetado = dp.etiquetado(df_constructed, export=False)
        
        # Por combinacion de parametros de select_data
        for j, param_values_4 in enumerate(product(*d_params['select'].values()), start=1):

            # Asigno valor a cada hiperpametro
            thr_corr, thr_fs = param_values_4[0], param_values_4[1]
            print(f" Iteracion Select Nº {i}.{j} ".center(120, "#"))
            print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")            

            # Selecciono datos
            path = f'./main_find_best_hyper/data/{country}/data_preparation/df_selected_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}_{thr_corr}_{thr_fs}.xlsx'
            try:
                df_sel = pd.read_excel(path, index_col=0)
            except FileNotFoundError:
                df_sel = dp.select_data(df_cons_etiquetado, thr_corr=thr_corr, thr_fs=thr_fs, export=False)
                df_sel.to_excel(path, index=True)
          
            for z, param_values_3 in enumerate(product(*d_params['treat_nan'].values()), start=1):

                fill_na = param_values_3[0]
                print(f" Iteracion Treat NaN Nº {i}.{j}.{z} ".center(120, "#"))
                print(f"Hiper treat --> fill_na: {fill_na}")

                df_sel_treated = dp.treat_nan_values(df_sel, fill_na=fill_na, export=False)

                # Por combinacion de parametros de modeling
                for h, param_values_5 in enumerate(product(*d_params['modeling'].values()), start=1):
                    cont_iter += 1

                    # Asigno valor a cada hiperparametro
                    val_size, test_size, bal_type, k = param_values_5[0], param_values_5[1], param_values_5[2], param_values_5[3]
                    print(f" Iteracion Modeling Nº {i}.{j}.{z}.{h} ".center(120, "#"))
                    print(f" Iteracion Nº {cont_iter} de {n_iter} {cont_iter/n_iter:.0f}%")
                    print(f'\n - Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h} ; segun_localia: {segun_localia} \n - Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs} \n - Hiper treat_nan --> {fill_na} \n - Hiper modeling --> val_size: {val_size} ; test_size: {test_size}; bal_type: {bal_type} ; k: {k}')

                    # Generar el diseño de la prueba
                    X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df_sel_treated, bal_type=bal_type, val_size=val_size, test_size=test_size, export=False)

                    # Si hay suficientes datos
                    if len(X_test) >= 100:
                        
                        # Select best model
                        d_best_hiper, train_model, d_best_model = mo.select_best_model(l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, export=False)

                        # Guardo datos en dataframe
                        row_data = {'n_iteration': cont_iter, 
                                    'n_dias_ult_part': n_dias_ult_part, 'n_anios_hist': n_years_h2h, 'segun_localia': segun_localia,
                                    'thr_corr': thr_corr, 'thr_fs': thr_fs,
                                    'fill_na': fill_na, 'bal_type': bal_type,
                                    'val_size': val_size, 'test_size': test_size, 'X_train': X_train.shape,
                                    'X_val': X_val.shape, 'X_test': X_test.shape, "X_columns": list(X_train.columns),
                                    'k': k}
                        row_data.update(d_best_model)
                        df_iteration = pd.concat([df_iteration, pd.DataFrame([row_data])], axis=0)
                        df_iteration.to_excel(f'./main_find_best_hyper/data/{country}/df_iteration.xlsx', index=False)
                    
                        # Guardo datos del modelo
                        pickle.dump(train_model, open(f"./main_find_best_hyper/data/{country}/modeling/{cont_iter}_model.pkl", "wb"))
                        try:
                            df_best_model_hiper = pd.DataFrame.from_dict(d_best_hiper, orient='index', columns=['Valor'])
                            df_best_model_hiper.to_csv(f"./main_find_best_hyper/data/{country}/modeling/{cont_iter}_model_hiper.csv")
                        except:
                            pass

                        # Verificar si la precisión actual es la mejor hasta ahora
                        if d_best_model['best_roi'] > best_roi_max:
                            best_roi_max = d_best_model['best_roi']
                            best_hyperparameters = row_data
                            best_model, hiper_best_model = train_model, train_model.get_params()
                            print(f"EL MEJOR MODELO HASTA AHORA! ROI: {d_best_model['best_roi']:.2f}%. Precision de test de {d_best_model['test_accuracy']:.2f}%")
                        else:
                            print(f"El ROI de {d_best_model['best_roi']:.2f}% es menor a {best_roi_max:.2f}%")
                    else:
                        texto = f"Se evitó el entrenamiento con tan pocos datos disponibles (X_test = {X_test.shape[0]} filas)."
                        warnings.warn(texto)

    # Guardo datos de todas las iteraciones
    df_iteration.to_excel(f'./main_find_best_hyper/data/{country}/df_iteration.xlsx', index=False)
    pickle.dump(best_model, open(f"./main_find_best_hyper/data/{country}/best_model.pkl", "wb"))
    df_best_model_hiper = pd.DataFrame.from_dict(hiper_best_model, orient='index', columns=['Valor'])
    df_best_model_hiper.to_csv(f"./main_find_best_hyper/data/{country}/df_best_model_hiper.csv")
    
    # Imprimir los hiperparámetros óptimos y la precisión correspondiente
    print("Mejores hiperparámetros:", best_hyperparameters)
    print("ROI obtenido:", best_roi_max)
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

def make_directories(country):
    country = country.lower()
    l_directorios = [
        f'./main_find_best_hyper/data/{country}/data_preparation',
        f'./main_find_best_hyper/data/{country}/modeling',
    ]
    
    for directorio in l_directorios:
        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

def main():
    country, var_resp, var_pred = "england", 'result', 'predicted_result'
    find_best_hiperparameters(var_resp, var_pred, country)

if __name__ == '__main__':
    main()