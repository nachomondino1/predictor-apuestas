import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from dspy.data_preparation import clean_data


def balance_dataset(X, y, type):

    # Balanceamos segun variable respuesta
    if type == 'over':  # Genera overfitting
        oversampler = RandomOverSampler()  #
        X_bal, y_bal = oversampler.fit_resample(X, y)

    elif type == 'under':  # Genera underfitting si no hago shuffle despues
        undersampler = RandomUnderSampler()  # Funciona igual que mi funcion pero no cambia dtypes, por lo que, no arroja errores
        X_bal, y_bal = undersampler.fit_resample(X, y)

    elif type == "over_and_under":  # funciona mal, es lo que mismo que 'over'  --> como sabe que tiene que hacer over de la clase minoritaria y under de la clase mayoritaria?
        pipeline = make_pipeline(SMOTE(), RandomUnderSampler())
        X_bal, y_bal = pipeline.fit_resample(X, y)

    else:
        print("El tipo ingresado para balancear los datos no es una opcion")
        raise ValueError(f"Error: El tipo de balanceo '{type}', no es una opcion")

    return X_bal, y_bal