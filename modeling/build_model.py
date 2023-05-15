from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from typing import Optional
import time
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

class Modelado:
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        
        self.df = df
        self.le = LabelEncoder()
        self.oversampler = RandomOverSampler(random_state=42)
        self.target_col = target_col

    def procesar_datos(self) -> None:
        '''
        Procesamiento del dataframe: tratamos los nans, balanceamos los datos y aplicamos encoder
        ''' 
        self.df = self.df.dropna() # inplace=True 
        df_mezclado = pd.DataFrame(shuffle(self.df)) # Hacemos Shuffle

        # Convertir variables categoricas string a categoricas numericas
        for col in df_mezclado.select_dtypes(include=['object']).columns:
            df_mezclado[col] = self.le.fit_transform(df_mezclado[col])

        # Balanceamos segun variable respuesta
        X, y  = df_mezclado.drop(self.target_col, axis=1), df_mezclado[self.target_col]
        self.X_bal, self.y_bal = self.oversampler.fit_resample(X, y)
        print("self.X_bal.shape ", self.X_bal.shape)


    def cross_validation(self, model) -> np.ndarray:
        '''
        Aplica cross validation
        ''' 
        # Realizar Cross Validation
        scores = cross_val_score(model, self.X_bal, self.y_bal, cv=10)
        return scores.mean()

    def calcular_metricas(self, model, n_folds_cv: int = 10, select_best_by: str = 'accuracy') -> None:
        '''
        Calcula métricas de evaluación
        ''' 
        # Especificar las métricas que se desean calcular
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        # Realizar validación cruzada y obtener los resultados
        cv_results = cross_validate(model, self.X_bal, self.y_bal, cv=n_folds_cv, scoring=scoring, return_train_score=True, return_estimator=True)
        cv_score = self.cross_validation(model)

        # Imprimir los resultados promedio de cada métrica
        print("Cross Validation Score:", cv_score)
        print("Accuracy: {:.3f}".format(cv_results['test_accuracy'].mean()))
        print("Precision: {:.3f}".format(cv_results['test_precision_macro'].mean()))
        print("Recall: {:.3f}".format(cv_results['test_recall_macro'].mean()))
        print("F1 score: {:.3f}".format(cv_results['test_f1_macro'].mean()))

        # Elegir el mejor modelo y calcular la matriz de confusión
        y_pred = cross_val_predict(model, self.X_bal, self.y_bal, cv=n_folds_cv)
        conf_mat = confusion_matrix(self.y_bal, y_pred)
        print("y_pred.shape ", y_pred.shape)
        print(conf_mat)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        cm_display.plot(cmap='Blues')
        plt.show()

        # Otra manera:
        # Obtenemos el mejor modelo según el puntaje en validación cruzada
        best_model_idx = cv_results['test_precision_macro'].argmax()
        best_model = cv_results['estimator'][best_model_idx]
        y_pred = cross_val_predict(best_model,self.X_bal, self.y_bal, cv=5)
        # Calculamos la matriz de confusión utilizando los datos de prueba
        conf_mat = confusion_matrix(self.y_bal, y_pred)
        print(conf_mat)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        cm_display.plot(cmap='Blues')
        plt.show()

    def arbol_decision(self, max_depth_tree: Optional[int] = None, n_folds_cv: Optional[int] = None) -> DecisionTreeClassifier: # Si args son None --> Poner grid
        '''
        Entrena un arbol de decisión
        ''' 
        print('Arbol de decision')
        if max_depth_tree is None and n_folds_cv is None: # El usuario no pasa parametros, por lo que se buscan los optimos
            params = {'max_depth': [None, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]}
            grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=10)
            grid.fit(self.X_bal, self.y_bal)
            print("Los mejores hiperparámetros son: ", grid.best_params_)
            dt = DecisionTreeClassifier(max_depth=grid.best_params_['max_depth'])

        else: # Uso los hiperparametros que me dio el usuario
            dt = DecisionTreeClassifier(max_depth=max_depth_tree)

        dt.fit(self.X_bal, self.y_bal)
        self.calcular_metricas(dt, n_folds_cv)
        return dt

    def random_forest(self, n_folds_cv: Optional[int] = None, n_tress_in_forest: Optional[int] = None, max_depth_tree: Optional[int] = None) -> RandomForestClassifier:
        '''
        Entrena Random Forest
        ''' 
        print('\nRandom Forest')

        if n_folds_cv is None and n_tress_in_forest is None and max_depth_tree is None: # El usuario no pasa parametros, por lo que se buscan los optimos

            # Definir los posibles valores para los hiperparámetros
            params = {'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
                        'n_estimators': [50, 100, 150, 200]}
            
            grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=params, cv=10, scoring='accuracy')
            grid_search.fit(self.X_bal, self.y_bal)
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score: {grid_search.best_score_}")
            rf = RandomForestClassifier(n_estimators = grid_search.best_params_['n_estimators'], 
                                        max_depth = grid_search.best_params_['max_depth'], 
                                        random_state = 42)
        else: 
            rf = RandomForestClassifier(n_estimators=n_tress_in_forest, max_depth = max_depth_tree, random_state=42) 
        
        rf.fit(self.X_bal, self.y_bal)        
        self.calcular_metricas(rf, n_folds_cv)
        return rf

    def xgboost(self, n_folds_cv: Optional[int] = None, n_tress_in_forest: Optional[int] = None, max_depth_tree: Optional[int] = None) -> xgb.sklearn.XGBClassifier:
        '''
        Aplica XGBoost
        ''' 
        print('\nXGBoost')

        if n_folds_cv is None and n_tress_in_forest is None and max_depth_tree is None:
            params = {
                'max_depth': [None, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35],
                'n_estimators': [50, 100, 150, 200]
                }
            grid_search = GridSearchCV(xgb.XGBClassifier(objective='multi:softmax', num_class=len(self.y_bal.unique())), param_grid=params, cv=10)
            grid_search.fit(self.X_bal, self.y_bal)
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score: {grid_search.best_score_}")            
            
            xg = xgb.XGBClassifier(n_estimators = grid_search.best_params_['n_estimators'], 
                                        max_depth = grid_search.best_params_['max_depth'])
        else: 
            xg = xgb.XGBClassifier(n_estimators=n_tress_in_forest, objective='multi:softmax', num_class=len(self.y_bal.unique()), max_depth=max_depth_tree)
        xg.fit(self.X_bal, self.y_bal)
        self.calcular_metricas(xg, n_folds_cv)

        return xg

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
    df = df.drop(['historial_entre_si'], axis = 1) 
    print(df.head())

    modeler = Modelado(df, 'equipo_ganador')
    modeler.procesar_datos()

    # modeler.arbol_decision() # max_depth_tree=25, n_folds_cv=10

    t0 = time.time() # Registramos el tiempo de inicio
    modeler.random_forest(n_folds_cv=10, n_tress_in_forest=100, max_depth_tree=25) 
    t1 = time.time() # Registramos el tiempo de fin
    print(f"La función tardó {(t1-t0)/60:.2f} minutos en ejecutarse") # Imprimimos el tiempo transcurrido

    t0 = time.time() # Registramos el tiempo de inicio
    # modeler.xgboost(n_folds_cv=10, n_tress_in_forest=50, max_depth_tree=15)
    t1 = time.time() # Registramos el tiempo de fin
    print(f"La función tardó {(t1-t0)/60:.2f} minutos en ejecutarse") # Imprimimos el tiempo transcurrido

main()