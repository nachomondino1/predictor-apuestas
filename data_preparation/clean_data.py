import pandas as pd



def procesar_datos() -> None:
    '''
    Procesamiento del dataframe: tratamos los nans, balanceamos los datos y aplicamos encoder
    '''
    # Remover NaN values
    self.df = self.df.dropna()  # inplace=True

    # Shuffle
    df_mezclado = pd.DataFrame(shuffle(self.df))

    # Convertir variables categoricas string a categoricas numericas
    for col in df_mezclado.select_dtypes(include=['object']).columns:
        df_mezclado[col] = self.le.fit_transform(df_mezclado[col])

    # Balanceamos segun variable respuesta
    X, y = df_mezclado.drop(self.target_col, axis=1), df_mezclado[self.target_col]
    self.X_bal, self.y_bal = self.oversampler.fit_resample(X, y)
