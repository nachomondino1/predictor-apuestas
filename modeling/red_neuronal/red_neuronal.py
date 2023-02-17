# https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/
# Importo librerias
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
from sklearn.model_selection import train_test_split

def format_resp(df):
    # Por partido
    for i in range(len(df)):

        # Obtengo goles de cada equipo
        ng1, ng2 = df.loc[i, 'goles_loc'], df.loc[i, 'goles_vis']

        # Si el equipo local hizo mas goles que el equipo visitante
        if ng1 > ng2:
            # Ganó equipo local
            df.loc[i, 'equipo_ganador'] = float(1)
        # Si el equipo local hizo la misma cantidad de goles que el equipo visitante
        elif ng1 == ng2:
            # Empataron
            df.loc[i, 'equipo_ganador'] = float(2)
        # Si el equipo local hizo menos goles que el equipo visitante
        else:
            # Ganó equipo vis
            df.loc[i, 'equipo_ganador'] = float(3)
    return df

# Levanto el dataset
df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/integrate_data/df_derived_data.xlsx',index_col=0)
print(df)
print(df.shape)

# Requiere de atributos NUMERICOS... Por eso convierto la var resp en numerica
df = format_resp(df)

# Remuevo columnas fecha
df.drop('id', inplace=True, axis=1)
df.drop('fecha', inplace=True, axis=1)
df.drop('equipo_loc', inplace=True, axis=1)
df.drop('equipo_vis', inplace=True, axis=1)
df.drop('arbitro', inplace=True, axis=1)
df.drop('dt_loc', inplace=True, axis=1)
df.drop('dt_vis', inplace=True, axis=1)
df.drop('goles_loc', inplace=True, axis=1)
df.drop('goles_vis', inplace=True, axis=1)
df['dif_gol'] = df['dif_gol_loc'] - df['dif_gol_vis']
df.drop('dif_gol_loc', inplace=True, axis=1)
df.drop('dif_gol_vis', inplace=True, axis=1)
print(df.shape)

# Remuevo ultimos  partidos de cada equipo (pues no puedo calcular bien la forma)
df = df.dropna().reset_index(drop=True)
print(df.shape)
print(df.head)



x_train = df.drop(['equipo_ganador'], axis=1).to_numpy()
y_train = df['equipo_ganador'].to_numpy()

x_train = np.asarray(x_train).astype(np.int)
y_train = np.asarray(y_train).astype(np.int)

# split train & test
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=23, shuffle=True)

# Entreno red
model = Sequential()
model.add(Dense(16, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# model.fit(training_data, target_data, epochs=1000)
model.fit(x_train, y_train, epochs=1000, verbose=0)


# evaluamos el modelo
# scores = model.evaluate(training_data, target_data)
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# print(model.predict(training_data).round())
print(model.predict(x_train).round())