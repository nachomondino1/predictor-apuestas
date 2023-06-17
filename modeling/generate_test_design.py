from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline


def balance_dataset(X, y, type):

    # Balanceamos segun variable respuesta
    if type == 'over':  # CUIDADO: 'over'usa odds para balancear... No hay forma de evitar esas columnas para balancear...
        # Genera odds que no son reales al agregar registros... eso afecta el calculo del roi...
        oversampler = RandomOverSampler()  #
        X_bal, y_bal = oversampler.fit_resample(X, y)

    elif type == 'under':
        undersampler = RandomUnderSampler()  # precision baja
        X_bal, y_bal = undersampler.fit_resample(X, y)

    elif type == "over_and_under":  # funciona mal, es lo que mismo que 'over'  --> como sabe que tiene que hacer over de la clase minoritaria y under de la clase mayoritaria?
        pipeline = make_pipeline(SMOTE(), RandomUnderSampler())
        X_bal, y_bal = pipeline.fit_resample(X, y)

    else:
        print("El tipo ingresado para balancear los datos no es una opcion")
        raise ValueError(f"Error: El tipo de balanceo '{type}', no es una opcion")

    return X_bal, y_bal