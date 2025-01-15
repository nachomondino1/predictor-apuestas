import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.set_up_logging import logger

# main.py
def convert_ball_possession_to_int(df):
    """
    Transformo posesion de string a float
    
    :param df: Dataframe. Con columnas 'posesion_loc' y 'posesion_vis' donde la posesion se interpreta como string. Por
    ejemplo '65%'.
    :return: Dataframe. Con columna 'fecha' interpretada como float. Por ejemplo, '0.65'
    """
    func = lambda x: float(x.replace('%', '')) if isinstance(x, str) and x.replace('%', '').isnumeric() else np.nan

    if ("ball_possession_home" in df.columns) and ('ball_possession_away' in df.columns):
        df["ball_possession_home"] = df["ball_possession_home"].apply(func)
        df["ball_possession_away"] = df["ball_possession_away"].apply(func)
    
        # Verifica si todos los elementos de la columna son de tipo float
        if not all(isinstance(value, float) for value in df["ball_possession_home"]):
            logger.error(f"Not all elements in the column '' are float.")  # Si no todos los elementos son de tipo float, raise una advertencia
            
    return df

def convert_value_to_int(df):
    """
    Transforma los valores numéricos de string a float para las columnas 'value' y 'wage'.

    :param df: Dataframe con columnas 'value' y 'wage', cuyos valores son strings, por ejemplo, '€1.2M'.
    :return: Dataframe con las columnas 'value' y 'wage' interpretadas como float, por ejemplo, 1.200.000.
    """
    def convertir_value(value_str):
        d = {'M': 1000000, 'K': 1000}

        # Si no se tiene el dato del valor de mercado
        if value_str == "€0":
            return None

        # Si se tiene el dato del valor de mercado
        else:
            for elem in d.keys():
                if elem in value_str:
                    value_int = float(value_str.replace("€", "").replace(elem, "")) * d[elem]
                    return value_int
            return None
    
    # Reemplazo strings por numbers
    df['value'] = df['value'].apply(convertir_value).astype(float)
    df['wage'] = df['wage'].apply(convertir_value).astype(float)
    return df

def convert_goals_to_int(df):
    """
    Elimina las filas que hacen que goals_home y goals_away no sean integers como deben ser. 
    Puede ser por NaN o por string "-".
    
    :param df: DataFrame con las columnas 'goals_home' y 'goals_away'
    :return: DataFrame con las filas inválidas eliminadas y las columnas convertidas a integers
    """
    # Convertir las columnas a numéricas, forzando errores a NaN
    df['goals_home'] = pd.to_numeric(df['goals_home'], errors='coerce')
    df['goals_away'] = pd.to_numeric(df['goals_away'], errors='coerce')
    
    # Contar y eliminar filas con NaN
    initial_count = len(df)
    df = df.dropna(subset=['goals_home', 'goals_away'])
    final_count = len(df)
    
    # Convertir las columnas a integers
    df['goals_home'] = df['goals_home'].astype(int)
    df['goals_away'] = df['goals_away'].astype(int)
    
    # Imprimir estadísticas
    removed_count = initial_count - final_count
    if removed_count > 0:
        logger.warning(f"Cantidad de partidos eliminados por no tener goles integer: {removed_count / initial_count * 100:.1f}%")
        
    return df

def convert_capacity_to_int(df):
    """
    Convierte columnas "capacity" y "attendance" de object a integer.
    """
    l_columns = ['capacity', 'attendance']

    # Por columna
    for col in l_columns:

        # Si la columna esta en el dataframe
        if col in df.columns:

            # Comprobar si la columna contiene valores de tipo cadena (string)
            if df[col].dtype == 'object':
                # Reemplazar los espacios en blanco en los valores de la columna
                df[col] = df[col].str.replace(' ', '')

                # Convertir la columna al tipo de datos correcto (entero)
                try: 
                    df[col] = df[col].astype(float) # Pues si tiene nan, es float.
                except ValueError: #ValueError: could not convert string to float: ''
                    pass
            else:
                print(f"Fallo la conversion de la columna {col} a float")
    return df

def format_percentage_columns(df, base_columns):
    """
    Formatea columnas que contienen porcentajes en sus versiones _home y _away.
    Extrae precisión, acciones exitosas y totales.
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas a procesar.
        base_columns (list): Lista de nombres base de las columnas (sin _home o _away).
    
    Returns:
        pd.DataFrame: DataFrame con las columnas formateadas.
    """
    columns = [col for col in base_columns if col in df.columns]  # Evita KeyError dentro del for en caso que no exista la stat en los nuevos partidos missing

    for base_col in columns:

        home_col = f"{base_col}_home"
        away_col = f"{base_col}_away"

        # Defino nombres de columnas tal que macheen los del df_match ya extraido
        accuracy_col_home = f'accuracy_{home_col}'
        accuracy_col_away = f'accuracy_{away_col}'
        completed_col_home = f'n_correct_{base_col}_home' # No uso 'completed' por si ya existe "completed_passes" en missing.
        completed_col_away = f'n_correct_{base_col}_away'
        total_col_home = f'n_{home_col}'  # No uso total por si ya existe "total_passes" en missing.
        total_col_away = f'n_{away_col}'

        # Procesar la columna `_home` (con warning) --> No usar .loc[] porque reformatea mal (no termina uniendo columnas nuevas a las viejas..)
        df[[accuracy_col_home, completed_col_home, total_col_home]] = df[home_col].str.extract(
            r'(\d+)% \((\d+)/(\d+)\)').astype(float)

        # Procesar la columna `_away`
        df[[accuracy_col_away, completed_col_away, total_col_away]] = df[away_col].str.extract(
            r'(\d+)% \((\d+)/(\d+)\)').astype(float)
    
        # Eliminar las columnas originales
        df.drop(columns=[home_col, away_col], inplace=True)
    
    return df

def rename_and_merge_columns(df, rename_dict): # Funciona perfecto! Verificado.
    """
    Función para renombrar, combinar y eliminar columnas viejas
    """
    for old_col, new_col in rename_dict.items():
        if old_col in df.columns:
            if new_col in df.columns:  # Si ya existe el nuevo nombre
                # Combina ambas columnas y elimina la vieja
                df[new_col] = df[new_col].combine_first(df[old_col])  # Warning pero esto lo hace mal: # df.loc[:, new_col] = df[new_col].combine_first(df[old_col]) 
                
                # Eliminar la columna vieja después de transferir los datos
                df.drop(columns=[old_col], inplace=True)

            else:
                # Renombrar directamente si no hay conflicto
                df.rename(columns={old_col: new_col}, inplace=True)

    return df

def convert_columns_to_float(df: pd.DataFrame, verbose: int = 0):
    """
    Intenta convertir las columnas object a float
    """
    # Selecciono las columnas object
    l_columnas_a_codificar = df.select_dtypes(include=['object']).columns

    # Por columna object
    for col in l_columnas_a_codificar:

        # Intento convertirla a float
        try:
            df[col] = df[col].astype(float)
            if verbose >= 1:
                print(f"Se convirtio la columna {col} a float!")
        except:
            pass
    return df

def convert_columns_to_int(df, verbose:int = 0):
    """
    Convierte las variables string a numéricas.

    # Parameters:
    df: DataFrame que contiene las variables a convertir. (DataFrame)
    df_etiquetas: DataFrame adicional con las etiquetas originales y enteros correspondientes. Si se proporciona, se utilizará 
    para la conversión en lugar de ajustar un nuevo LabelEncoder.(DataFrame, opcional)

    # Returns:
    DataFrame con las variables convertidas y un DataFrame adicional con las etiquetas originales y enteros correspondientes.
    """
    # Definicion de variables
    df_etiquetas = pd.DataFrame(columns=['variable', 'str_value', 'int_value'])
    le = LabelEncoder()
    l_columnas_a_codificar = list(df.select_dtypes(include=['object']).columns)  # Obtener columnas de tipo objeto
    if verbose >= 1:
        print(f"Columnas str a convertir a int: {l_columnas_a_codificar}")

    # Por variable string
    for col in l_columnas_a_codificar:

        # Obtengo valores a codificar evitando "NaN"
        valores_a_codificar = df[col].dropna().unique()
       
        # Mapeo valor str con valor int
        le.fit(valores_a_codificar)
        d_mapeo = dict(zip(le.classes_, le.transform(le.classes_)))

        # Reemplazo valor str por valor integer en DataFrame
        df[col] = df[col].map(d_mapeo)

        # Guardo string y su equivalente numerico
        df_etiquetas_col = pd.DataFrame({'variable': col, 'str_value': list(d_mapeo.keys()), 'int_value': list(d_mapeo.values())})
        df_etiquetas = pd.concat([df_etiquetas, df_etiquetas_col], axis=0)

    return df, df_etiquetas

# main_next_matches.py
def convert_columns_to_int_already_tagged(df, df_etiquetas, l_col_a_codificar: list = None, verbose: int = 0):
    """
    Etiquetado usando un df_etiquetas ya creado. Es para main_next_matches. Tengo en cuenta posibles nuevas etiquetas y las agrego a df_etiquetas
    """
    # Determino columnas a codificar de string a integer
    if l_col_a_codificar is None:
        l_col_a_codificar = df_etiquetas['variable'].unique()

    if verbose >= 0:
        print("Columnas a codificar: ", l_col_a_codificar)
    
    # Por columna a codificar
    for columna in l_col_a_codificar:
        
        # Obtengo etiquetas de la columna
        df_etiquetas_columna = df_etiquetas[df_etiquetas['variable']==columna]        
        d_mapeo = dict(zip(df_etiquetas_columna['str_value'], df_etiquetas_columna['int_value']))

        if verbose >= 1:
            print(f"Columna: {columna}, DF etiqeutas shape columna: {df_etiquetas_columna.shape}") 

        # Por partido
        for i, row in df.iterrows():
                    
            # Si la etiqueta no existe en el mapeo
            if row[columna] not in d_mapeo.keys():
                logger.error(f"No se encontró etiqueta para {row[columna]} en la fila {i}. Se rellena con 0.")
                valor = 0

            # Si la etiqueta existe
            else:
                # Reemplazo el valor
                valor = d_mapeo[row[columna]]
            
            # Actualizo el DataFrame
            df.loc[i, columna] = valor

    return df, df_etiquetas

def map_teams(df, df_teams):
    """
    Convierto id_team_home e id_team_away de ids a nombre de equipos.
    """
    # Revierto etiquetas para tener nombres de equipos en vez de ids
    d_mapeo = dict(zip(df_teams.index, df_teams['team_name']))        
    df['id_team_home'] = df['id_team_home'].replace(d_mapeo)
    df['id_team_away'] = df['id_team_away'].replace(d_mapeo)
    return df

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    # Levanto datasets
    country = 'argentina'
    df_match = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data/{country}/p2_data_understanding/df_match.xlsx')
    df_player = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/data/{country}/p2_data_understanding/df_player.xlsx", index_col=0)

    # Entidad partido WhoScored: fecha, resultados de medio tiempo y final
    df_match['fecha'] = pd.to_datetime(df_match['fecha'] + ' ' + df_match['hora'], format='%a, %d-%b-%y %H:%M')
    # df_match['fecha'] = df_match['fecha'] - datetime.timedelta(hours=4)  # Resto 4 horas a la columna 'fecha' para que este en horario argentino
    df_match[['ht_goals_home', 'ht_goals_away']] = df_match['ht_result'].str.split(' : ', expand=True)  # Separar ht_result en ht_goals_home y ht_goals_away
    df_match[['goals_home', 'goals_away']] = df_match['ft_result'].str.split(' : ', expand=True)  # Separar ft_result en goals_home y goals_away
    df_match = df_match.drop(['hora', 'ht_result', 'ft_result'], axis=1)

    # Entidad jugador: fecha
    df_player['fecha_nac'] = pd.to_datetime(df_player['fecha_nac'], format='%d-%m-%Y')

    # Exporto pruebas
    df_match.to_excel(f'{BASE_DIR_LOCAL}/df_match_formated.xlsx', index=False)
    df_player.to_excel(f'{BASE_DIR_LOCAL}/df_player_formated.xlsx', index=False)