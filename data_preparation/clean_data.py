import pandas as pd
from dspy.data_preparation.text_preparation import TextPreparation
from sklearn.ensemble import RandomForestRegressor

def prepare_text_columns(df, l_col_to_except):  # Podria agregar un l_except_columns para eevitar analizar alguna columna de strings que no quiera preparar...
    '''
    Prepara el texto de las columnas que contengan strings.
    :param df: Dataframe.
    :return: Dataframe con columnas que contienen strings ya preparados para ser analizados
    '''
    # Creo objeto de la clase
    tp = TextPreparation(df, l_col_to_except)

    # Uso metodos de la clase
    tp.to_lower()
    tp.delete_accent()
    tp.delete_special_characters()
    tp.delete_punctuation()
    return tp.df

def clean_teams_names(df):
    """
    Limpia y cambia el nombre de algunos equipos en el dataframe.
    :param df: Dataframe con columnas "equipo_loc" y "equipo_vis"
    :return: Dataframe con nombres de equipos modificados y limpios
    """
    # Limpio string 'Vencedor' en el nombre de algunos equipos.
    df['equipo_loc'] = df['equipo_loc'].replace({'vencedor': '', 'equipo que avanza': ''}).str.strip()
    df['equipo_vis'] = df['equipo_vis'].replace({'vencedor': '', 'equipo que avanza': ''}).str.strip()

    # Quitar abreviaturas en nombres de equipos (NO FUNCIONA...)
    d_abrev_team_names = {' l p': ' la plata', 'atl ': 'atletico ', ' jrs': ' juniors', ' utd': ' united'}  # Tengo que tener cuidado, reemplazo strings... pueden ser substring y cambiarlo sin querer hacerlo.
    df['equipo_loc'] = df['equipo_loc'].replace(d_abrev_team_names, regex=True).str.strip()
    df['equipo_vis'] = df['equipo_vis'].replace(d_abrev_team_names, regex=True).str.strip()

    # Reemplazo nombres enteros de equipos para que sea igual a los de la entidad jugador
    d_team_names = {'estudiantes la plata': 'estudiantes',
        'qpr': 'queens park rangers',  'wolves': 'wolverhampton', 'west brom': 'west bromwich albion'}
    for equipo_part, equipo_jug in d_team_names.items():
        df['equipo_loc'] = df['equipo_loc'].replace(equipo_part, equipo_jug)
        df['equipo_vis'] = df['equipo_vis'].replace(equipo_part, equipo_jug)

    return df

def prueba():
    # Levanto dataset
    df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/inglaterra/df_part_formated.xlsx')
    df_jug =pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/inglaterra/df_jug_formated.xlsx')

    # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
    l_col_to_except = ['id', 'temporada']
    df_part = prepare_text_columns(df_part, l_col_to_except)
    df_jug = prepare_text_columns(df_jug, l_col_to_except)

    # Remuevo strings adicionales en los nombres de los equipos
    df_part = clean_teams_names(df_part)

    df_part.to_excel('/Users/nachomondino/Desktop/df_part_cleaned.xlsx', index=False)
    df_jug.to_excel('/Users/nachomondino/Desktop/df_jug_cleaned.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()