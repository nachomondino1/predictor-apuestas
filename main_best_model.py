# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost
from sklearn.linear_model import LogisticRegression  # Regresion Logistica
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier
from p3_data_preparation.select_data import determine_country_competitions
from p4_modeling import assess_models_in_prod, train_models
import directories

# Guardado de assess actual y creacion de directorio para nuevo assess
def save_old_assess():
    """
    Muevo assess a old_assess_iterations para no sobreescribirlo con el nuevo assess.
    """
    directorio_origen = f"./data/{country}/p4_modeling/"
    directorio_destino = f"./data/{country}/old/"
    directories.make_directories([directorio_destino])
    directories.mover_archivo(directorio_origen, directorio_destino)

def select_best_model(df):
    """
    Selecciona el mejor modelo
    """

    # Encontrar el índice del máximo valor en la columna "ROI"
    indice_maximo = df['roi_por_partido'].idxmax()

    # Seleccionar la fila correspondiente al índice máximo
    fila_maximo = df.loc[indice_maximo]

    # best_hyperparameters =
    # pickle.dump(best_model, open(f"{ruta_base}/best_model.pkl", "wb"))
    # df_best_model_hiper = pd.DataFrame.from_dict(hiper_best_model, orient='index', columns=['Valor'])
    # df_best_model_hiper.to_csv(f"{ruta_base}/df_best_model_hiper.csv")

    # Imprimir los hiperparámetros óptimos y la precisión correspondiente
    # print("Mejores hiperparámetros:", best_hyperparameters)
    # print("ROI obtenido:", best_roi_max)

def main(l_modelos, d_params, export: bool = True):
    """
    Entrena modelos segun las combinaciones de hiperparametros deseadas. Luego los evalua en produccion y selecciona el mejor.
    """
    # Evito sobreescribir assess actual y lo muevo. Ademas, creo directorio para el nuevo assess.
    save_old_assess()      # Cuidado al correr este progrma, sobreescribis el assess que esta hoy actualmente. Si lo queres evitar, guarda el assess en carpeta "old_assess_iterations" 

    ruta_base_mod = f"./data/{country}/p4_modeling/{date}" 
    ruta_base_dp = f"./data/{country}/p4_modeling/{date}/p3_data_preparation"
    ruta_base_modelos = f"./data/{country}/p4_modeling/{date}/models" 
    ruta_assess = f'{ruta_base_mod}/assess_models_in_prod/data_preparation'
    ruta_assess_2 = f'{ruta_base_mod}/assess_models_in_prod/modeling'
    directories.make_directories(l_directorios=[ruta_base_dp, ruta_base_modelos, ruta_assess, ruta_assess_2])
    
    # Preparao datos, entreno modelos y evaluo en df_test
    df_iteration = train_models.main(country, ruta_base_dp, ruta_base_mod, ruta_base_modelos, d_params, l_modelos, export=export)

    # Preparo datos missing y evaluo modelos en produccion
    df_iteration_prod = assess_models_in_prod.main(df_iteration, country, date, ruta_base_dp, ruta_base_mod, export=export)

    # Selecciono el mejor modelo (mayor roi por partido en produccion)
    # best_model = select_best_model(df_iteration_prod)

    # Concateno df_iteration y df_iteration_prod para tener df_iteration_completo
    df_iteration.set_index('n_iteration', inplace=True) # Establecer 'n_iteration' como índice del DataFrame
    df_concat = pd.concat([df_iteration, df_iteration_prod], axis=1)    
    return df_concat

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
        
    # Parametros de ejecucion
    id_country = 48

    # Defino hiperparametros a probar
    d_comps = determine_country_competitions(id_country)
    l_modelos = [LogisticRegression()]  # SVC()  #RandomForestClassifier(), XGBClassifier(), GradientBoostingClassifier(),  MLPClassifier()
    d_params = {
        'construct': {
            'n_dias_ult_part': [[30, 180]],
            'n_years_h2h': [3],
            'segun_localia': [True, False]
        },
        'clean_data_2': {
            'competencies_to_select': [d_comps['comp_sin_b'], d_comps['all_comp']],
            'n_years_to_select': [3, 5, 10, None],
        },
        'select': {
            'thr_corr': [0.7, 0.8, 0.9, None], 
            'thr_fs': [0.3, 0.2, 0.1, None], 
        },
        'treat_nan': {
            'fill_na': [None, 'ml'], 
        },
        'modeling': {
            'val_size': [0.125],
            'test_size': [0.10], 
            'bal_type': [None, 'under'], # 'over'
            'k': [10] 
        }
    }

    # Creo directorios segun pais y fecha de corrida
    ## Determino fecha de hoy
    date_con_hora = datetime.datetime.now()
    date = date_con_hora.date()

    ## Obtengo el nombre del pais segun su id
    df_countries = pd.read_excel('./data/df_countries.xlsx')
    country = df_countries[df_countries['id_country'] == id_country]['country_name'].values[0]

    # Entreno modelos y los evaluo en produccion.
    df = main(l_modelos, d_params)
    df.to_excel(f"./data/{country}/p4_modeling/{date}/df_iteration.xlsx", index=True)