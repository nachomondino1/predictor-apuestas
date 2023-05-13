from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

class Modelado:
    
    def __init__(self, df, target_col):
        
        self.df = df
        self.le = LabelEncoder()
        self.oversampler = RandomOverSampler(random_state=42)
        self.target_col = target_col

    def procesar_datos(self):
        self.df = self.df.drop(['historial_entre_si'], axis = 1) # inplace=True
        self.df = self.df.dropna() # inplace=True 
        df_mezclado = pd.DataFrame(shuffle(self.df)) # Hacemos Shuffle

        # Convertir variables categoricas string a categoricas numericas
        for col in df_mezclado.select_dtypes(include=['object']).columns:
            df_mezclado[col] = self.le.fit_transform(df_mezclado[col])

        # Balanceamos segun variable respuesta
        X, y  = df_mezclado.drop(self.target_col, axis=1), df_mezclado[self.target_col]
        self.X_train_bal, self.y_train_bal = self.oversampler.fit_resample(X, y)


    def cross_validation(self, model):
        # Realizar Cross Validation
        scores = cross_val_score(model, self.X_train_bal, self.y_train_bal, cv=10)
        return scores.mean()

    def arbol_decision(self, max_depth_tree, num_folds_cv): # Si args son None --> Poner grid
        # Crear modelo de Árbol de Decisión y ajustar a los datos de entrenamiento
        dt = DecisionTreeClassifier(max_depth=max_depth_tree)
        dt.fit(self.X_train_bal, self.y_train_bal)

        # Realizar Cross Validation
        cv_score = self.cross_validation(dt)

        # Calcular precisión en datos de prueba
        # test_score = cross_val_score(dt, self.X_train_bal, self.y_train_bal, cv=num_folds_cv, scoring=['accuracy', 'precision'])
        # Especificar las métricas que se desean calcular
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        # Realizar validación cruzada y obtener los resultados
        cv_results = cross_validate(dt, self.X_train_bal, self.y_train_bal, cv=num_folds_cv, scoring=scoring)

        print(cv_results)

        # Imprimir los resultados promedio de cada métrica
        print("Accuracy: {:.3f}".format(cv_results['test_accuracy'].mean()))
        print("Precision: {:.3f}".format(cv_results['test_precision_macro'].mean()))
        print("Recall: {:.3f}".format(cv_results['test_recall_macro'].mean()))
        print("F1 score: {:.3f}".format(cv_results['test_f1_macro'].mean()))

        # Imprimir resultados
        print("\nModelo de Árbol de Decisión")
        print("Cross Validation Score:", cv_score)
        # print("Test Score:", test_score.mean())

        return dt

    def random_forest(self, num_folds_cv, number_tress_in_forest, max_depth_tree):
        rf = RandomForestClassifier(n_estimators=number_tress_in_forest, random_state=42, max_depth = max_depth_tree) # max_depth=30
        rf.fit(self.X_train_bal, self.y_train_bal)

        # Realizar Cross Validation
        cv_score = self.cross_validation(rf)

        # Calcular precisión en datos de prueba
        # test_score = cross_val_score(dt, self.X_train_bal, self.y_train_bal, cv=num_folds_cv, scoring=['accuracy', 'precision'])
        # Especificar las métricas que se desean calcular
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        # Realizar validación cruzada y obtener los resultados
        cv_results = cross_validate(rf, self.X_train_bal, self.y_train_bal, cv=num_folds_cv, scoring=scoring)

        print(cv_results)

        # Imprimir los resultados promedio de cada métrica
        print("Accuracy: {:.3f}".format(cv_results['test_accuracy'].mean()))
        print("Precision: {:.3f}".format(cv_results['test_precision_macro'].mean()))
        print("Recall: {:.3f}".format(cv_results['test_recall_macro'].mean()))
        print("F1 score: {:.3f}".format(cv_results['test_f1_macro'].mean()))

        # Imprimir resultados
        print("\nModelo de Random Forest")
        print("Cross Validation Score:", cv_score)
        # print("Test Score:", test_score.mean())
    
    def xgboost(self):
        pass

    def regresion_logistica(self):
        pass

    def svm(self):
        pass

    def red_neuronal(self):
        pass

    def seleccionar_mejor_modelo(self):
        pass

    def graficar_matriz_confusion(self, model):
        pass

    def graficar_curva_roc(self):
        pass


def main():
    
    df = pd.read_excel('data_preparation/df_prepared.xlsx')
    print(df.head())

    modeler = Modelado(df, 'equipo_ganador')
    modeler.procesar_datos()

    modeler.arbol_decision(max_depth_tree=25, num_folds_cv=10)
    modeler.random_forest(num_folds_cv=10, number_tress_in_forest=100, max_depth_tree=25)

main()