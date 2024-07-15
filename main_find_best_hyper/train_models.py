# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from itertools import product
from main import DataPreparation, Modeling
from p3_data_preparation.select_data import determine_country_competitions
import pickle
import joblib

def main(country, ruta_base, d_params, l_modelos, export:bool = True):
    """
    Busco los hiperparametros optimos en DataPreparation y Modeling de main.py
    """
    # Definicion de variables
    df_iteration = pd.DataFrame()
    cont_iter = 0
    var_resp, var_pred = 'result', 'predicted_result'
    dp = DataPreparation(country)
    mo = Modeling(var_resp, var_pred, country)  # Creo objeto de clase Modeling

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
        path = f'{ruta_base}/data_preparation/df_constructed_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}.xlsx'
        try:
            df_constructed = pd.read_excel(path, index_col=0)
            print("\n DF_CONSTRUCTED \n", df_constructed.head(2))
        except FileNotFoundError:
            # Levanto dataset formateado e integrado (estos no cambian entre iteraciones)
            df_integrated = pd.read_excel(f'./p3_data_preparation/data/{country}/df_integrated.xlsx', index_col=0)
            print("\n DF_INTEGRATED \n", df_integrated.shape, df_integrated.head(2))

            df_constructed = dp.construct_data(df_integrated, n_days=n_dias_ult_part, n_years_h2h=n_years_h2h, segun_localia=segun_localia, export=False)
            if export:
                df_constructed.to_excel(path, index=True)

        # Etiqueto df_constructed
        df_cons_etiquetado, df_etiquetas = dp.tag_string_data_to_integer(df_constructed, export=False)
        path = f'{ruta_base}/data_preparation/df_etiquetas_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}.xlsx'
        if export:
            df_etiquetas.to_excel(path, index=True)

        # Clean data 2
        for zz, param_values_00 in enumerate(product(*d_params['clean_data_2'].values()), start=1):

            n_years_to_select, comp_to_select = param_values_00[1], param_values_00[0]
            print(f" Iteracion clean_data 2 Nº {i}.{zz} ".center(120, "#"))
            print(f"Hiper clean_data_2 --> n_years_to_select: {n_years_to_select} ; comp_to_select: {comp_to_select}")            

            df_cons_clean, scaler, columns_used = dp.clean_data_2(df_cons_etiquetado, n_years_to_select, comp_to_select, export=False)
            joblib.dump((scaler, columns_used), f'{ruta_base}/data_preparation/scaler_model_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}_{n_years_to_select}_{comp_to_select}.pkl')
        
            # Por combinacion de parametros de select_data
            for j, param_values_4 in enumerate(product(*d_params['select'].values()), start=1):

                # Asigno valor a cada hiperpametro
                thr_corr, thr_fs = param_values_4[0], param_values_4[1]
                print(f" Iteracion Select Nº {i}.{j} ".center(120, "#"))
                print(f"Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs}")            

                # Selecciono datos
                path = f'{ruta_base}/data_preparation/df_selected_{n_dias_ult_part}_{n_years_h2h}_{segun_localia}_{n_years_to_select}_{comp_to_select}_{thr_corr}_{thr_fs}.xlsx'
                try:
                    df_sel = pd.read_excel(path, index_col=0)
                except FileNotFoundError:
                    df_sel = dp.select_data(df_cons_clean, thr_corr=thr_corr, thr_fs=thr_fs, export=False)
                    # df_sel.to_excel(path, index=True)
            
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
                        print(f'\n - Hiper construct --> n_dias_ult_part: {n_dias_ult_part} ; n_years_h2h: {n_years_h2h} ; segun_localia: {segun_localia} \n - Hiper clean_data_2 n_years_to_sel: {n_years_to_select} comp_to_select: {comp_to_select} \n- Hiper select --> thr_corr: {thr_corr} ; thr_fs: {thr_fs} \n - Hiper treat_nan --> {fill_na} \n - Hiper modeling --> val_size: {val_size} ; test_size: {test_size}; bal_type: {bal_type} ; k: {k}')

                        # Generar el diseño de la prueba
                        X_train, X_val, X_test, y_train, y_val, y_test = mo.generate_test_design(df_sel_treated, bal_type=bal_type, val_size=val_size, test_size=test_size, export=False)

                        # Si hay suficientes datos
                        if len(X_test) >= 50:
                            
                            # Select best model
                            best_model, d_hiper_best_model, d_metrics_best_model, df_pred = mo.select_best_model(l_modelos, X_val, y_val, X_train, y_train, X_test, y_test, k, export=False)

                            # Guardo datos en dataframe
                            row_data = {'n_iteration': cont_iter, 
                                        'n_dias_ult_part': n_dias_ult_part, 'n_anios_hist': n_years_h2h, 'segun_localia': segun_localia,
                                        'thr_corr': thr_corr, 'thr_fs': thr_fs,
                                        'n_years_to_select': n_years_to_select, 'comp_to_select': comp_to_select,
                                        'fill_na': fill_na, 'bal_type': bal_type,
                                        'val_size': val_size, 'test_size': test_size, 'X_train': X_train.shape,
                                        'X_val': X_val.shape, 'X_test': X_test.shape, "X_columns": list(X_train.columns),
                                        'k': k}
                            row_data.update(d_metrics_best_model)
                            df_iteration = pd.concat([df_iteration, pd.DataFrame([row_data])], axis=0)
                            if export:
                                df_iteration.to_excel(f'{ruta_base}/df_iteration.xlsx', index=False)

                                # Guardo datos del modelo
                                pickle.dump(best_model, open(f"{ruta_base}/modeling/{cont_iter}_model.pkl", "wb"))
                                df_pred.to_excel(f'{ruta_base}/modeling/{cont_iter}_df_predicciones.xlsx', index=True)
                                try:
                                    df_hiper_best_model = pd.DataFrame.from_dict(d_hiper_best_model, orient='index', columns=['Valor'])
                                    df_hiper_best_model.to_csv(f"{ruta_base}/modeling/{cont_iter}_model_hiper.csv")
                                except:
                                    pass

    # Guardo datos de todas las iteraciones
    if export:
        df_iteration.to_excel(f'{ruta_base}/df_iteration.xlsx', index=False)

    return df_iteration

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


if __name__ == '__main__':

    # Importo librerias
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier  # XGBoost
    from sklearn.linear_model import LogisticRegression  # Regresion Logistica
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC  # SVM
    from sklearn.neural_network import MLPClassifier

    # Parametros de corrida
    id_country = 167

    # Defino hiperparametros a probar
    d_comps = determine_country_competitions(id_country)
    l_modelos = [LogisticRegression(), SVC()]  #RandomForestClassifier(), XGBClassifier(), GradientBoostingClassifier(),  MLPClassifier()
    d_params = {
        'construct': {
            'n_dias_ult_part': [30, 60],
            'n_years_h2h': [3],
            'segun_localia': [True, False]
        },
        'clean_data_2': {
            'competencies_to_select': [d_comps['comp_sin_b']], # d_comps['comp_sin_b'] # d_comps['comp_solo_liga'],  # Italy y Spain no tienen la b en df_match
            'n_years_to_select': [3, 5, 10, None],
        },
        'select': {
            'thr_corr': [0.7, 0.8, 0.9, None],
            'thr_fs': [0.2, 0.1, None], 
        },
        'treat_nan': {
            'fill_na': [None, 'ml'],
        },
        'modeling': {
            'val_size': [0.125],
            'test_size': [0.125], 
            'bal_type': [None, 'under'], #  'over'
            'k': [10] 
        }
    }

    # df_iteration = find_best_hiperparameters(country, ruta_base, d_params, l_modelos, export=False):
