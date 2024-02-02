import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from random import randint

def balance_dataset(X, y, tipo):

    # Balanceamos segun variable respuesta
    if tipo == 'over':  # Genera overfitting
        oversampler = RandomOverSampler()  #
        X_bal, y_bal = oversampler.fit_resample(X, y)

    elif tipo == 'under':  # Genera underfitting si no hago shuffle despues
        undersampler = RandomUnderSampler()  # Funciona igual que mi funcion pero no cambia dtypes, por lo que, no arroja errores
        X_bal, y_bal = undersampler.fit_resample(X, y)

    else:
        print("El tipo ingresado para balancear los datos no es una opcion")
        raise ValueError(f"Error: El tipo de balanceo '{tipo}', no es una opcion")

    return X_bal, y_bal

def separate_train_val_and_test(X, y, test_val_size=0.8, test_size=0.5):
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
    df = pd.concat([X, y], axis=1).sample(frac=1).reset_index(drop=True)

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
    df_val_test = df.loc[~df.index.isin(df_train.index)].reset_index(drop=True)
    X_val_and_test, y_val_and_test = df_val_test.drop(y.name, axis=1), df_val_test[y.name]
    # print(df_val_test.shape)

    # Separo en test y val
    X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size, random_state=randint(1, 1000), shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def prueba():

    from p3_data_preparation import clean_data

    pais = 'argentina_south_america'
    var_resp = 'equipo_ganador'
    bal_type = None
    test_val_size = 0.2
    test_size = 0.5
    fill_na = 'ml'

    # Levanto dataset de prueba
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_selected.xlsx')

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
        X_train, y_train = clean_data.fill_nan_values(X_train, y_train, type=fill_na)  # Relleno NaN values en las columnas seleccionadas. Tener cuidado de no introducir sesgo en el modelo, las precisiones casi siempre seran mayores que dropna() en train y test, lo que cuenta es la precision en next_matches o en un dataset que no haya sido filleado...
        print(f"Se realizó el rellenado de NaN values. Shape X_train luego de rellenado: {X_train.shape}")

    # Balanceo el dataset de entrenamiento
    if bal_type is not None:
        X_train, y_train = balance_dataset(X_train, y_train, tipo=bal_type)
        print(f"Shape X_train luego de balanceo: {X_train.shape}")

    # Shuffle el dataset de entrenamiento
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    X_train, y_train = df_train.drop(var_resp, axis=1), df_train[var_resp]

    print(f'Train: {X_train.shape} {y_train.shape}')
    print(f'Val: {X_val.shape} {y_val.shape}')
    print(f'Test: {X_test.shape} {y_test.shape}')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()

