import pandas as pd

df = pd.DataFrame(columns=['id', 'nombre', 'edad', 'a', 'b'])

d_nueva_fila = {'id': [1, ], 'nombre': "Ignacio"}

d_nueva_fila['a'], d_nueva_fila['b'] = 1, 2
print(d_nueva_fila)

df_nueva_fila = pd.DataFrame(d_nueva_fila, index=[0])
print(df_nueva_fila)


df = pd.concat([df, df_nueva_fila])
print(df)