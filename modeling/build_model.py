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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from itertools import cycle
# Regresion Logistica
from sklearn.linear_model import LogisticRegression
import warnings
import plotly.graph_objects as go
# SVM
from sklearn.svm import SVC
# Redes Nueronales
from sklearn.neural_network import MLPClassifier


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

    def cross_validation(self, model) -> np.ndarray:
        '''
        Aplica cross validation
        ''' 
        # Realizar Cross Validation
        scores = cross_val_score(model, self.X_bal, self.y_bal, cv=10)
        return scores.mean()
    
    def graficar_curva_roc(self) -> None:

        y_true = self.y_bal.values
        classes = np.unique(y_true)
        n_classes = len(classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 3
        for c in range(n_classes):
            fpr[c], tpr[c], _ = roc_curve(y_true == c, self.y_pred)
            roc_auc[c] = auc(fpr[c], tpr[c])
        
        plt.figure()
        lw = 2
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def graficar_matriz_conf(self, n_cv:int) -> None:

        # Calcular matriz de confusion, obteniendo el mejor modelo
        best_model_idx = self.cv_results['test_precision_macro'].argmax()
        best_model = self.cv_results['estimator'][best_model_idx]
        self.y_pred = cross_val_predict(best_model, self.X_bal, self.y_bal, cv=n_cv)
        # Calculamos la matriz de confusión utilizando los datos de prueba
        conf_mat = confusion_matrix(self.y_bal, self.y_pred)
        print("Precision del mejor modelo: {:.3f}".format(self.cv_results['test_precision_macro'][best_model_idx]))
        print(conf_mat)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        cm_display.plot(cmap='Blues')
        """
        # Elegir el mejor modelo y calcular la matriz de confusión
        y_pred = cross_val_predict(model, self.X_bal, self.y_bal, cv=n_folds_cv)
        conf_mat = confusion_matrix(self.y_bal, y_pred)
        print("y_pred.shape ", y_pred.shape)
        print(conf_mat)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        cm_display.plot(cmap='Blues')
        plt.show()
        """

    def calcular_metricas(self, model, n_folds_cv: int = 10, select_best_by: str = 'accuracy') -> None:
        '''
        Calcula métricas de evaluación
        ''' 
        # Especificar las métricas que se desean calcular
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        # Realizar validación cruzada y obtener los resultados
        self.cv_results = cross_validate(model, self.X_bal, self.y_bal, cv=n_folds_cv, scoring=scoring, return_train_score=True, return_estimator=True)
        cv_score = self.cross_validation(model)

        # Imprimir los resultados promedio de cada métrica
        print("Cross Validation Score:", cv_score)
        print("Accuracy: {:.3f}".format(self.cv_results['test_accuracy'].mean()))
        print("Precision: {:.3f}".format(self.cv_results['test_precision_macro'].mean()))
        print("Recall: {:.3f}".format(self.cv_results['test_recall_macro'].mean()))
        print("F1 score: {:.3f}".format(self.cv_results['test_f1_macro'].mean()))

        # Graficar matriz de confusion del mejor resultado
        self.graficar_matriz_conf(n_folds_cv)

        # Graficar curva ROC AUC
        self.graficar_curva_roc()
        plt.show()

    def graficar_importancia_atrib(self, model):
        features = self.X_bal.columns
        feature_importances = model.feature_importances_

        # Crear figura
        fig = go.Figure()

        # Agregar barras al gráfico
        fig.add_trace(go.Bar(
            x=feature_importances,
            y=features,
            orientation='h'
        ))

        # Configurar el diseño del gráfico
        fig.update_layout(
            title='Importancia de las características',
            xaxis_title='Importancia',
            yaxis_title='Características',
            yaxis=dict(autorange="reversed")  # Invertir el orden de las características
        )

        # Mostrar el gráfico
        fig.show()

    def arbol_decision(self, max_depth_tree: Optional[int] = None, n_folds_cv: Optional[int] = None, plot_feature_importance: Optional[bool] = False) -> DecisionTreeClassifier: # Si args son None --> Poner grid
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
        
        if plot_feature_importance == True:
            self.graficar_importancia_atrib(dt)

        return dt

    def random_forest(self, n_folds_cv: Optional[int] = None, n_tress_in_forest: Optional[int] = None, max_depth_tree: Optional[int] = None, plot_feature_importance: Optional[bool] = False) -> RandomForestClassifier:
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
        
        if plot_feature_importance == True:
            self.graficar_importancia_atrib(rf)

        return rf

    def xgboost(self, n_folds_cv: Optional[int] = None, n_tress_in_forest: Optional[int] = None, max_depth_tree: Optional[int] = None, plot_feature_importance: Optional[str] = False) -> xgb.sklearn.XGBClassifier:
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

        if plot_feature_importance == True:
            self.graficar_importancia_atrib(xg)

        return xg

    def regresion_logistica(self, n_folds_cv: Optional[int] = None, penal: Optional[str] = None, c_value: Optional[int] = None, solv: Optional[str] = None, max_iteraciones: Optional[int] = None,) -> LogisticRegression:
        '''
        Aplica Regresion Logistica
        ''' 
        print('\nRegresion Logistica')

        if n_folds_cv is None and penal is None and c_value is None and solv is None and max_iter is None:
            # Definir los hiperparámetros a ajustar
            params = {
                'penalty': [None, 'l2'],
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'newton-cg', 'sag', 'saga', 'lbfgs'],
                'max_iter': [100, 500, 1000], 
                'multi_class':['multinomial']
            }

            grid_search = GridSearchCV(LogisticRegression(multi_class='multinomial'), param_grid=params, cv=10)
            grid_search.fit(self.X_bal, self.y_bal)
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score: {grid_search.best_score_}")  

            best_params = grid_search.best_params_          
            lr = LogisticRegression(**best_params)
            """lr = LogisticRegression(multi_class='multinomial', 
                                    penalty = grid_search.best_params_['penalty'],
                                    C = grid_search.best_params_['C'],
                                    solver = grid_search.best_params_['solver'],
                                    max_iter = grid_search.best_params_['max_iter'])
                                    """
        else: 
            lr = LogisticRegression(multi_class='multinomial', 
                                    penalty = penal,
                                    C = c_value,
                                    solver = solv,
                                    max_iter = max_iteraciones)

        lr.fit(self.X_bal, self.y_bal)
        self.calcular_metricas(lr, n_folds_cv)

        return lr

    def svm(self, n_folds_cv: Optional[int] = None, kernel_type: Optional[str] = None, ovo_o_ovr: Optional[str] = None) -> SVC:
        '''
        Aplica SVM
        ''' 
        print('\nSVM')

        if n_folds_cv is None and kernel_type is None and ovo_o_ovr is None:
            # Definir los hiperparámetros a ajustar
            params = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'decision_function_shape': ['ovo', 'ovr'],
            }

            grid_search = GridSearchCV(SVC(), param_grid=params, cv=10)
            grid_search.fit(self.X_bal, self.y_bal)
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score: {grid_search.best_score_}")  

            best_params = grid_search.best_params_          
            lr = SVC(**best_params)

        else: 
            lr = SVC(kernel=kernel_type, decision_function_shape=ovo_o_ovr)

        lr.fit(self.X_bal, self.y_bal)
        self.calcular_metricas(lr, n_folds_cv)
        

    def red_neuronal(self, n_folds_cv: Optional[int] = None) -> MLPClassifier:
        '''
        Aplica Redes Neuronales
        ''' 
        print('\nRedes Neuronales')

        if n_folds_cv is None:
            # Definir los hiperparámetros a ajustar
            params = {
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['learning_rate', 'invscaling', 'adaptive'],
                'max_iter': [200, 300, 400],
            }

            grid_search = GridSearchCV(MLPClassifier(), param_grid=params, cv=10)
            grid_search.fit(self.X_bal, self.y_bal)
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score: {grid_search.best_score_}")  

            best_params = grid_search.best_params_          
            mlp = MLPClassifier(**best_params)

        else: 
            mlp = MLPClassifier()

        mlp.fit(self.X_bal, self.y_bal)
        self.calcular_metricas(mlp, n_folds_cv)

        return mlp

    def seleccionar_mejor_modelo(self):
        
        # Crear una lista de modelos
        modelos = [self.arbol_decision(), self.random_forest(), self.xgboost(), self.regresion_logistica(), self.svm(), self.red_neuronal()]

        test_scores = [cross_validate(model, self.X_bal, self.y_bal, scoring = 'accuracy') for model in modelos]

        # Calcular precisión en datos de prueba para cada modelo
        # test_scores = [model.score(self.X_test, self.y_test) for model in modelos]

        # Seleccionar el mejor modelo según la precisión en datos de prueba
        best_model_idx = test_scores.index(max(test_scores))
        best_model = modelos[best_model_idx]

        print("\nEl mejor modelo es:")
        print(best_model)

        return best_model

    def graficar_matriz_confusion(self, model):
        pass


def main():

    df = pd.read_excel('data_preparation/df_prepared.xlsx')
    df = df.drop(['historial_entre_si'], axis = 1) 
    print(df.head())

    modeler = Modelado(df, 'equipo_ganador')
    modeler.procesar_datos()

    """
    modeler.arbol_decision(plot_feature_importance=True) # max_depth_tree=25, n_folds_cv=10

    modeler.random_forest(n_folds_cv=10, n_tress_in_forest=100, max_depth_tree=25, plot_feature_importance=True) 
   
    modeler.xgboost(n_folds_cv=10, n_tress_in_forest=50, max_depth_tree=15, plot_feature_importance=True)

    warnings.filterwarnings("ignore")
    t0 = time.time() # Registramos el tiempo de inicio
    modeler.regresion_logistica(n_folds_cv= 10, penal = 'l2', c_value = 1, solv = 'lbfgs', max_iter= 500) # n_folds_cv= 10, penal = 'l2', c_value = 1, solv = 'lbfgs', max_iter= 500 
    t1 = time.time() # Registramos el tiempo de fin
    print(f"La función tardó {(t1-t0)/60:.2f} minutos en ejecutarse") # Imprimimos el tiempo transcurrido

    modeler.svm(n_folds_cv = 10, kernel_type = 'poly', ovo_o_ovr= 'ovo')
    """
    warnings.filterwarnings("ignore")
    t0 = time.time() # Registramos el tiempo de inicio
    modeler.red_neuronal()
    t1 = time.time() # Registramos el tiempo de fin
    print(f"La función tardó {(t1-t0)/60:.2f} minutos en ejecutarse") # Imprimimos el tiempo transcurrido

    
main()