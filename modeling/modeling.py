import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from dspy.data_preparation import clean_data
# from dspy.modeling import naive_bayes, test_design
# from evaluation.evaluation import calculate_precision

# Arbol de decision
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Levanto dataset
df = pd.read_excel('data_preparation/df_prepared.xlsx')
df.drop(['historial_entre_si'], axis = 1, inplace=True) # Elimino esta variable porque tiene muchos nans
df = df.dropna()  # Elimina filas con al menos un valor nulo
print(df.head())
print(df.shape)

# Codificamos las variables categoricas string en numericas
labelencoder = LabelEncoder()
cat_columns = ['equipo_loc', 'equipo_vis', 'arbitro', 'dt_loc', 'dt_vis']
for column in cat_columns:
    df[column] = labelencoder.fit_transform(df[column])
    
N_MODELOS = 15
l_aciertos = []

# Definir el número de folds para la validación cruzada
num_folds = 10
# Dividir los datos en k folds
folds = np.array_split(df, num_folds)

# Iterar sobre cada fold y entrenar el modelo
for i in range(num_folds):

    # Separar los datos de entrenamiento y prueba para el fold actual
    test_data = folds[i]
    train_data = pd.concat([f for j, f in enumerate(folds) if j != i])
    X_train = train_data.drop("equipo_ganador", axis=1)
    y_train = train_data["equipo_ganador"]
    X_test = test_data.drop("equipo_ganador", axis=1)
    y_test = test_data["equipo_ganador"]
    
    ## Entrenar el modelo en los datos de entrenamiento del fold actual
    # Arbol de decision
    # modelo = DecisionTreeClassifier(max_depth=6)
    # modelo.fit(X_train, y_train)

    # Random Forest
    modelo = RandomForestClassifier(n_estimators=100, random_state=42) # max_depth=30
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    l_aciertos.append(precision)
    matriz_confusion = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matriz_confusion,
                                                display_labels=["Local", "Empate", "Visitante"])
    cm_display.plot(cmap='Blues')

    print("precision: ", precision)
    print(matriz_confusion)
    print("Fold %d - Score: %.3f" % (i+1, modelo.score(X_test, y_test)))

    # Graficar el árbol
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plot_tree(modelo, feature_names=X.columns, class_names=y.unique(), filled=True, ax=ax)
    plt.show()    
print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos)/len(l_aciertos)}")


# Vamos a usar modelos de librerias independientes de la libreria que hizo Nacho
def main():
    # Definicion de variables
    N_MODELOS = 10
    l_aciertos = []

    # Levanto dataset
    df = pd.read_excel('data_preparation/df_prepared.xlsx', index_col=0)
    print(df.head())

    # Balanceo dataset y elimino filas con historial=NaN
    df = df.dropna(subset=['historial_entre_si']).reset_index()  # Elimina filas con al menos un valor nulo
    # df = clean_data.balance_dataset(df, var_resp='equipo_ganador')

    # Por modelo
    for i in range(N_MODELOS):
        # Shuffle dataset
        df = df.sample(frac=1).reset_index(drop=True)

        # Separo conjunto de datos en train y test
        # df_train, df_test = test_design.separate_train_and_test(df)

        # Implemento Naive Bayes
        # modelo_nb = naive_bayes.train_naive_bayes(df_train, var_resp='equipo_ganador')

        # 5) EVALUACION DEL MODELO
        # df_result = naive_bayes.predict_naive_bayes(modelo_nb, df_test, var_resp='equipo_ganador', col_prob_clase=True)
        # precision = calculate_precision(df_result, var_resp='equipo_ganador')
        # l_aciertos.append(precision)

    # print(f"Max: {max(l_aciertos)} Min: {min(l_aciertos)} Prom: {sum(l_aciertos)/len(l_aciertos)}")

# main()