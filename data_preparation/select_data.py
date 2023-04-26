import pandas as pd
from sklearn.linear_model import LinearRegression

# IMPLEMENTAR CORRELACION DE VARIABLES PARA DESCARTAR LAS REDUNDANTES
# cargar los datos en un dataframe
path_read = '/Users/nachomondino/Desktop/df_constructed.xlsx'  # '/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/flashscore/liga_argentina_historico_2_prueba.xlsx'
path_export = '/Users/nachomondino/Desktop/correlation_matrix_2.xlsx'  # '/Users/nachomondino/Desktop/correlation_matrix.xlsx'

df = pd.read_excel(path_read)

'''
# calcular la matriz de correlación
df_correlation_matrix = df.corr()  # OJO que no tiene en cuenta las variables categoricas... y si quiero tenerlas en cuenta como "equipo ganador"

# mostrar la matriz de correlación
df_correlation_matrix.to_excel(path_export)  # Cambiar la ruta del archivo
'''


# IMPLEMENTAR REGRESION -->  seleccionar las variables más importantes.
# separar las variables predictoras (X) y la variable objetivo (y)
df = df.drop(['id', 'fecha', 'equipo_loc', 'equipo_vis', 'arbitro', 'cancha', 'dt_loc', 'dt_vis', 'l_jug_lesionados_loc', 'l_jug_lesionados_vis', 'historial_entre_si'], axis=1)
df = df.dropna(axis=0, how='any')  # Borro partidos (filas) con al menos un nan

X = df.drop('equipo_ganador', axis=1)
y = df['equipo_ganador']
l_col = X.columns

# ajustar el modelo de regresión lineal
model = LinearRegression().fit(X, y)

importance = model.coef_
for i,v in enumerate(importance):
    print('Variable %0s: %.5f' % (l_col[i],v))