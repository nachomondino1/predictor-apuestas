import sys
sys.path.append('.')  # Fallaba el import de main
import pandas as pd
from p6_deployment import main_next_matches
from p3_data_preparation.construct_data import determine_result
from p4_modeling.asses_model import determine_winning_bets

### Que corra collect_predictions.py con missing True (y el resto False) y actualizo predicciones.xlsx con el resultado. Tengo que mantener menos codigo pues reciclo codigo. Agilizo get_predictions.yml al correrlo con missing=False.

def collect_missing():
    # Defino condiciones del analisis
    df = pd.DataFrame()
    d_run = {'run_missing': True, 'data_unders': False, 'data_prep': False, 'modeling': False, 'export': True}
    l_countries = [48, 55, 59, 77, 148]
    n_days = 1.5 # Mas de uno por seguridad.

    # Por country
    for id_country in l_countries:

        # Extraigo y preparo missing 
        df_country = main_next_matches.main(d_run, id_country, n_days, export=d_run['export'])

        df = pd.concat([df, df_country], axis=0)
    
    return df

def collect_results(df: pd.DataFrame):
    """
    Extraigo de Flashscore el resultado de los partidos pasados como parametro (df) y lo agrego como columna.
    """
    # Determino ganador y si acerté
    df_pred_with_result = determine_result(df, var_resp='result')

    # Determino acierto o fallo
    df_pred_with_result = determine_winning_bets(df_pred_with_result)    
    df_pred_with_result.index.name = 'id_match'  # Es importante para la base de datos MySQL
    print(df_pred_with_result)

    return df_pred_with_result

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    # Recolectar missing de todos los paises
    df = collect_missing()

    # Determino resultado y si acerte o falle
    df = collect_results(df)

    # Exportar dataset
    df.to_excel('data/predicciones_prueba.xlsx')