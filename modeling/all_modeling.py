from utils_modeling import *

###### DATASET #######
# Levanto dataset
df, classes_names = load_dataset_and_clean(path='data_preparation/df_prepared.xlsx')

###### HIPERPARAMETROS #######
num_folds = 10 # Cantidad de divisiones de validacion cruzada
max_depth_tree = 25 # Profundidad del arbol
number_tress_in_forest = 50 # Cantidad de arboles en el bosque de Random Forest

###### MODELOS #######
# metricas = train_and_test(num_folds, 'arbol', df, max_depth_tree, number_tress_in_forest, plot_tree_bool = False, plot_conf_matrix = True)
# print(metricas) #  arbol xgboost random_forest

###### ANALIZAR HIPERPARAMETROS OPTIMOS #######
hiper_optimos(df, modelo_a_entrenar="xgboost")