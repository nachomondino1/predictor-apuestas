import sys
sys.path.append('.')  # Fallaba el import de main
from utils.set_up_logging import logger
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from random import randint

def balance_dataset(X, y, bal_type: str, verbose: int = 0):
    """
    Balanceo de datos
    # Parameters
        X: Dataframe solo con variables predictoras.
        y: Dataframe solo con variable respuesta.
        bal_type: Tipo de balanceo a aplicar. Puede ser 'over' o 'under'. (str)

    # Returns
        X e y pasados como parametros balanceados segun bal_type indicado.
    """
    # Balanceamos segun variable respuesta
    if bal_type == 'over':
        oversampler = RandomOverSampler()  #
        X_bal, y_bal = oversampler.fit_resample(X, y)

    elif bal_type == 'under':
        undersampler = RandomUnderSampler()
        X_bal, y_bal = undersampler.fit_resample(X, y)

    else:
        if verbose >= 0:
            print("El bal_type ingresado para balancear los datos no es una opcion")
        raise ValueError(f"Error: El bal_type de balanceo '{bal_type}', no es una opcion")

    if verbose >= 1:
        print(f"Shape X e y antes de balanceo y despues: \n X: {X.shape} --> {X_bal.shape} \n y: {y.shape} --> {y_bal.shape} ")
    return X_bal, y_bal

def separate_train_val_and_test(X, y, test_val_size=0.8, test_size=0.5, shuffle=True):
    """
    Separa un dataframe en train, validación y test según el porcentaje de corte indicado.
    Los conjuntos de validación y prueba no deben contener registros con NaN.

    Me aseguro de que los registros con NaN values esten en df_train y no en df_val ni df_test...

    :param X: DataFrame con las variables predictoras.
    :param y: DataFrame con la variable respuesta.
    :param porc_corte: Float entre 0 y 1. Porcentaje de registros en el conjunto de entrenamiento.
    :return: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # Concatenar X e y y luego shuffle del DataFrame
    df = pd.concat([X, y], axis=1).sample(frac=1)

    # Todos los registros con al menos un NaN value los guardo en el conjunto de entrenamiento
    df_train = df.loc[df.isna().any(axis=1)]

    # Completo el conjunto de entrenamiento con n registros al azar
    n_corte = int((1 - test_val_size) * len(df))
    n = n_corte - len(df_train)  # número de registros que deseas seleccionar al azar

    if n > 0: # Si hay mas filas con nan que el corte, dejo todas las filas con nan en df_train
        df_train = pd.concat([df_train, df.dropna().sample(n)], axis=0)

    X_train, y_train = df_train.drop(y.name, axis=1), df_train[y.name]
    # print(f"Corte: {corte} ; n: {n}")
    # print(df_train.shape)

    # Defino df_val_test segun los registros que quedan
    df_val_test = df.loc[~df.index.isin(df_train.index)]  # .reset_index(drop=True)
    X_val_and_test, y_val_and_test = df_val_test.drop(y.name, axis=1), df_val_test[y.name]
    # print(df_val_test.shape)

    # Separo en test y val
    X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=randint(1, 1000), shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test

def select_test_set(df, n_reg_test: int = 100, retrain: bool = True, country : str = '', verbose : int = 0):
    """
    Determina qué registros pueden ser utilizados en el test
    Requisitos para el test
        -1: Que id_competition sea publica (lo mismo que hago en assess).
        -2: Que sean partidos jugados en los ultimos meses.

    # Parameters
        df: Dataframe.
        n_years_to_select: Numero de años para seleccionar los ultimos partidos los cuales iran al df_test.
        n_max_reg: Numero maximo de registros para X_test. (int)
        verbose: 

    # Return
        X_test: Dataframe de testeo sin variable respuesta.
        y_test: Dataframe de testeo solo la variable respuesta.
    """
    # Levanto df_match del pais
    if retrain:
        df_match = pd.read_excel(f'data/{country}/p6_deployment/missing/old_updated/df_match.xlsx', index_col=0)
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        logger.warning(f"Se levanto el df_match con los missing pues retrain=True. Shape: {df_match.shape}")
    else:
        df_match = pd.read_excel(f"data/{country}/p3_data_preparation/clean_data/df_match_cleaned.xlsx", index_col=0)  
    
    # Ordeno por fecha descendiente
    df_match = df_match.sort_values(by='date', ascending=False)
    
    if verbose >= 2:
        logger.info(df_match.columns) # No quiero "unnamed"
        logger.info(df_match['date'].head(10))

    # Requisito 1: id competition.   # En df_match obtengo id_competition por match y determino posibles id_matches
    from p3_data_preparation.select_data import select_league_matches
    df1 = select_league_matches(df_match)
    index_comp = df1.index

    if verbose >= 2:
        df_filt_1 = df[df.index.isin(index_comp)]
        logger.info(f"Registros que pasan el requisito 1 (solo competencia publica): {len(df_filt_1)}")
    
    # Requisito 2: Last matches 
    df_match_comp = df_match[df_match.index.isin(df1.index)]  # Dejo solo las comp publicas
    df2 = df_match_comp.head(n_reg_test)
    index_last_matches = df2.index

    if verbose >= 2:
        df_filt_2 = df[df.index.isin(index_last_matches)]
        logger.info(f"Registros que pasan el requisito 2 (solo last matches): {len(df_filt_2)}")
    
    # Selecciono registros que cumplen los requisitos
    filters = df.index.isin(index_comp) & df.index.isin(index_last_matches)
    df_test = df[filters]

    '''
    df_aux = df_match[df_match.index.isin(index_last_matches)]
    if len(df_filt) != len(df_aux):
        logger.warning(f"(1 de 3) PROBLEMA DE NAN EN ULTIMOS PARTIDOS. De los {len(index_last_matches)}, solo hay {len(df_aux)} en el df que recibe select_test_data(). Deberian ser iguales --> {len(index_last_matches)} = {len(df_aux)}")
        logger.warning(f"(2 de 3) Por que no son iguales? Se estan eliminando 'ultimos partidos' en 1) clean_data_2 (eliminacion de filas por tener mucho NaN) o 2) treat nan values (dropna de columnas con 'poco' nan).")
        logger.warning(f"(3 de 3) Que podes hacer? No podemos hacer mucho sino investigar por qué los ultimos partidos tienen tanto NaN y evitar que tenga NaN. Seguramente el problema es en la extraccion de missing debido a algun cambio de Flashscore")
    '''

    # Construyo el dataset de testeo a partir de registros que no han sido rellenados
    # df_test = df_filt.sample(random_state=42) --> no hace falta y ademas es peor.
    return df_test

def n_rows_to_test(df, df_test, verbose: int = 1):
    """
    Imprime por pantalla cuantos registros quedarian en df_test segun los requisitos exigidos.
    """
    df_for_test = df[df.index.isin(df_test.index)]

    if verbose >= 1:
        logger.info(f"Nº partidos para test esperados: {len(df_test)}. Nº partidos para test que pasaron: {len(df_for_test)}")

    return len(df_for_test)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    from p3_data_preparation import clean_data

    country = 'argentina_south_america'
    var_resp = 'result'
    bal_type = None
    test_val_size = 0.2
    test_size = 0.5
    fill_na = 'ml'

    # Levanto dataset de prueba
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data/{country}/p3_data_preparation/df_selected.xlsx')

    # Elimino filas con al menos un NaN puesto que al modelo no le pueden ingresar NaN values
    if fill_na is None:
        df = clean_data.eliminar_filas_nan(df, umbral=0)  # df = df.dropna()

        # Separo en X e y
        X, y = df.drop(var_resp, axis=1), df[var_resp]

        # Separo conjunto de datos en train, validation y test
        X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_val_size, random_state=42,shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=42, shuffle=True)

    else:
        # Separo en X e y
        X, y = df.drop(var_resp, axis=1), df[var_resp]

        # Separo conjunto de datos en train, validation y test dejando los NaN values en df_train
        X_train, X_val, X_test, y_train, y_val, y_test = separate_train_val_and_test(X, y, test_val_size=test_val_size, test_size=test_size)

        # Relleno nan en el dataset de entrenamiento
        X_train, y_train = clean_data.drop_and_fill_nan_values(X_train, y_train, fill_type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las precisiones casi siempre seran mayores que dropna() en train y test, lo que cuenta es la precision en next_matches o en un dataset que no haya sido filleado...
        print(f"Se realizó el rellenado de NaN values. Shape X_train luego de rellenado: {X_train.shape}")

    # Balanceo el dataset de entrenamiento
    if bal_type is not None:
        X_train, y_train = balance_dataset(X_train, y_train, fill_type=bal_type)
        print(f"Shape X_train luego de balanceo: {X_train.shape}")

    # Shuffle el dataset de entrenamiento
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train = df_train.sample(frac=1)  # .reset_index(drop=True)
    X_train, y_train = df_train.drop(var_resp, axis=1), df_train[var_resp]

    print(f'Train: {X_train.shape} {y_train.shape}')
    print(f'Val: {X_val.shape} {y_val.shape}')
    print(f'Test: {X_test.shape} {y_test.shape}')

