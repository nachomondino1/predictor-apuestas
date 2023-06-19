import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from dspy.data_preparation import clean_data


def balance_dataset(X, y, type):

    # Balanceamos segun variable respuesta
    if type == 'over':  # Tiene overfitting. CUIDADO: 'over'usa odds para balancear... No hay forma de evitar esas columnas para balancear...
        # Genera odds que no son reales al agregar registros... eso afecta el calculo del roi...
        oversampler = RandomOverSampler()  #
        X_bal, y_bal = oversampler.fit_resample(X, y)

    elif type == 'under':  # Tiene underfitting...
        undersampler = RandomUnderSampler()  # Funciona igual que mi funcion pero no cambia dtypes, por lo que, no arroja errores
        X_bal, y_bal = undersampler.fit_resample(X, y)
        print(X_bal.shape)

        # Uso mi propia funcion porque me llama la atencion que de tan mal con la libreria...
        # X_bal, y_bal = clean_data.balance_dataset(X, y)
        # print(X_bal.shape)

    elif type == "over_and_under":  # funciona mal, es lo que mismo que 'over'  --> como sabe que tiene que hacer over de la clase minoritaria y under de la clase mayoritaria?
        pipeline = make_pipeline(SMOTE(), RandomUnderSampler())
        X_bal, y_bal = pipeline.fit_resample(X, y)

    else:
        print("El tipo ingresado para balancear los datos no es una opcion")
        raise ValueError(f"Error: El tipo de balanceo '{type}', no es una opcion")

    # Shuffle y borro index --> fundamental para evitar problemas en CV por los folds?
    df_bal = pd.concat([X_bal, y_bal], axis=1)
    df_bal = df_bal.sample(frac=1).reset_index(drop=True)  # Para evitar que queden misma clase en un fold de CV?
    X_bal, y_bal = df_bal.drop(columns=[y.name]), df_bal[y.name]
    return X_bal, y_bal