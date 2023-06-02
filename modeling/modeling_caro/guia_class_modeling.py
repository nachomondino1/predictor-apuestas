import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


class Modelado:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.le = LabelEncoder()

    def procesar_datos(self):

        # Elimino todos los registros con al menos un NaN
        self.df = self.df.dropna().reset_index()  # subset=['dif_forma']
        print(self.df.shape)

        # Convertir variables categoricas string a categoricas numericas
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.le.fit_transform(self.df[col])

        # Separar los datos en variables de entrada y salida
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        # Realizar balanceo de datos
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Dividir los datos en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

    def cross_validation(self, model):
        # Realizar Cross Validation
        scores = cross_val_score(model, self.X_train, self.y_train, cv=10)
        return scores.mean()

    def regresion_logistica(self):
        # Crear modelo de Regresión Logística y ajustar a los datos de entrenamiento
        lr = LogisticRegression()
        lr.fit(self.X_train, self.y_train)

        # Realizar Cross Validation
        cv_score = self.cross_validation(lr)

        # Calcular precisión en datos de prueba
        test_score = lr.score(self.X_test, self.y_test)

        # Imprimir resultados
        print("\nModelo de Regresión Logística")
        print("Cross Validation Score:", cv_score)
        print(f"Test Score: {test_score}")

        return lr

    def arbol_decision(self):
        # Crear modelo de Árbol de Decisión y ajustar a los datos de entrenamiento
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.y_train)

        # Realizar Cross Validation
        cv_score = self.cross_validation(dt)

        # Calcular precisión en datos de prueba
        test_score = dt.score(self.X_test, self.y_test)

        # Imprimir resultados
        print("\nModelo de Árbol de Decisión")
        print("Cross Validation Score:", cv_score)
        print("Test Score:", test_score)

        return dt

    def svm(self):
        # Definir los hiperparámetros para ajustar
        params = {'C': [0.1, 1, 10],
                  'gamma': [0.1, 1, 10]}

        # Crear modelo de SVM
        svc = SVC()

        # Ajustar los hiperparámetros utilizando GridSearchCV
        svc_cv = GridSearchCV(svc, params, cv=10)
        svc_cv.fit(self.X_train, self.y_train)

        # Imprimir el mejor valor de los hiperparámetros
        print("\nModelo de SVM")
        print("Mejores hiperparámetros para SVM:", svc_cv.best_params_)

        # Entrenar el modelo con los mejores hiperparámetros
        svc = SVC(**svc_cv.best_params_)
        svc.fit(self.X_train, self.y_train)

        # Realizar Cross Validation
        cv_score = self.cross_validation(svc)

        # Calcular precisión en datos de prueba
        test_score = svc.score(self.X_test, self.y_test)

        # Imprimir resultados
        print("Cross Validation Score:", cv_score)
        print("Test Score:", test_score)

        return svc

    def red_neuronal(self):
        # Crear modelo de Red Neuronal y ajustar a los datos de entrenamiento
        mlp = MLPClassifier()
        mlp.fit(self.X_train, self.y_train)

        # Realizar Cross Validation
        cv_score = self.cross_validation(mlp)

        # Calcular precisión en datos de prueba
        test_score = mlp.score(self.X_test, self.y_test)

        # Imprimir resultados
        print("\nModelo de Red Neuronal")
        print("Cross Validation Score:", cv_score)
        print("Test Score:", test_score)

        return mlp

    def xgboost(self):
        # Definir los hiperparámetros para ajustar
        params = {'max_depth': [3, 5, 7],
                  'n_estimators': [50, 100, 150]}

        # Crear modelo de XGBoost
        xgb = XGBClassifier()

        # Ajustar los hiperparámetros utilizando GridSearchCV
        xgb_cv = GridSearchCV(xgb, params, cv=10)
        xgb_cv.fit(self.X_train, self.y_train)

        # Imprimir el mejor valor de los hiperparámetros
        print("\nModelo de XGBoost")
        print("Mejores hiperparámetros para XGBoost:", xgb_cv.best_params_)

        # Entrenar el modelo con los mejores hiperparámetros
        xgb = XGBClassifier(**xgb_cv.best_params_)
        xgb.fit(self.X_train, self.y_train)

        # Realizar Cross Validation
        cv_score = self.cross_validation(xgb)

        # Calcular precisión en datos de prueba
        test_score = xgb.score(self.X_test, self.y_test)

        # Imprimir resultados
        print("Cross Validation Score:", cv_score)
        print("Test Score:", test_score)

        return xgb

    def seleccionar_mejor_modelo(self):

        # Crear una lista de modelos
        modelos = [self.regresion_logistica(), self.arbol_decision(), self.svm(), self.red_neuronal(), self.xgboost()]

        # Calcular precisión en datos de prueba para cada modelo
        test_scores = [model.score(self.X_test, self.y_test) for model in modelos]

        # Seleccionar el mejor modelo según la precisión en datos de prueba
        best_model_idx = test_scores.index(max(test_scores))
        best_model = modelos[best_model_idx]

        print("\nEl mejor modelo es:")
        print(best_model)

        return best_model

    def graficar_matriz_confusion(self, model):
        # Realizar predicciones en datos de prueba
        y_pred = model.predict(self.X_test)

        # Crear matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)

        # Graficar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues")
        plt.title("Matriz de confusión")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


def main():

    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/df_prepared.xlsx')
    print(df.head())

    modeler = Modelado(df, 'equipo_ganador')
    modeler.procesar_datos()
    best_model = modeler.seleccionar_mejor_modelo()
    modeler.graficar_matriz_confusion(best_model)

main()