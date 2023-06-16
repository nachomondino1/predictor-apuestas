from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def balance_dataset(X, y, type):

    # Balanceamos segun variable respuesta
    if type == 'over':  # CUIDADO: 'over'usa odds para balancear... No hay forma de evitar esas columnas para balancear...
        # Genera odds que no son reales al agregar registros... eso afecta el calculo del roi...
        oversampler = RandomOverSampler()  #
        X_bal, y_bal = oversampler.fit_resample(X, y)

    elif type == 'under':
        undersampler = RandomUnderSampler()  # precision baja
        X_bal, y_bal = undersampler.fit_resample(X, y)

    elif type == "over_and_under":  # funciona mal, es lo que mismo que 'over'
        pipeline = make_pipeline(SMOTE(), RandomUnderSampler())
        X_bal, y_bal = pipeline.fit_resample(X, y)

    else:
        print("El tipo ingresado para balancear los datos no es una opcion")
        raise ValueError("Error: El tipo de balanceo ingresado no es una opcion")

    return X_bal, y_bal


def separate_train_val_and_test(X, y, test_val_size, test_size):

    # Divido todos los  datos en train y validacion + prueba
    X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=test_val_size, random_state=42)

    # Divido validacion + prueba en validacion y prueba
    X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=test_size,random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test