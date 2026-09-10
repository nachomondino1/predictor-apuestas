import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.set_up_logging import logger

# main.py
def convert_ball_possession_to_int(df):
    """
    Transforma la posesión de balón a un entero de 0 a 100.
    
    :param df: DataFrame con columnas 'ball_possession_home' y 'ball_possession_away'.
    :return: DataFrame con las columnas de posesión convertidas a entero.
    """
    
    def convert_value(x):
        try:
            # Si es un string, elimina el '%' y convierte a entero
            if isinstance(x, str):
                # Verifica si el string sin '%' es un número
                clean_x = x.replace('%', '')
                if clean_x.isnumeric():
                    return int(clean_x)
                else:
                    return np.nan
            # Si ya es un número (int o float), conviértelo a entero directamente
            elif isinstance(x, (int, float)):
                return int(x)
            else:
                return np.nan
        except (ValueError, AttributeError):
            return np.nan

    # Lista de columnas que podrían representar posesión
    possession_cols = ["ball_possession_home", "ball_possession_away"]

    for col in possession_cols:

        # Si la columna está en el DataFrame
        if col in df.columns:
            df[col] = df[col].apply(convert_value)

            # Verifica si la conversión fue exitosa (debe ser int)
            if not pd.api.types.is_integer_dtype(df[col]):
                logger.warning(f"La columna '{col}' no se convirtió completamente a enteros.")
        
        # Si la columna no está en el df
        else:
            logger.warning(f"La columna '{col}' no está presente en el DataFrame. Se omite.")

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

        # Si value_str ya es un número, simplemente devuelve el número
        if isinstance(value_str, (int, float)):
            return value_str

        # Si se tiene el dato del valor de mercado
        else:
            for elem in d.keys():
                if elem in value_str:
                    value_int = float(value_str.replace("€", "").replace(elem, "")) * d[elem]
                    return value_int
            return None
    
    # Reemplazo strings por números
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

def format_percentage_columns(df, base_columns, verbose: int = 0):
    """
    Formatea columnas que contienen porcentajes en sus versiones _home y _away.
    Extrae precisión, acciones exitosas y totales.
    
    Args:
        df (pd.DataFrame): DataFrame con las columnas a procesar.
        base_columns (list): Lista de nombres base de las columnas (sin _home o _away).
    
    Returns:
        pd.DataFrame: DataFrame con las columnas formateadas.
    """
    column_specs = {
        'accuracy': [0, 100],
        'completed': [0, 3500],
        'total': [0, 3500]
    }

    for base_col in base_columns:
        for location in ['home', 'away']:
            col_name = f"{base_col}_{location}"
            if col_name not in df.columns:
                continue  # Si no existe la columna, la salteamos

            # Verificar si la columna contiene valores tipo string antes de aplicar .str
            if df[col_name].dtype == "object" or df[col_name].dtype.name == "string":
   
                # Paso 1: normalizar solo valores no nulos
                df[col_name] = df[col_name].where(df[col_name].isna(), df[col_name].str.replace(r'\s+', ' ', regex=True).str.strip()) 

                # Paso 2: crear columnas vacías para luego completar según cada caso
                df[f'accuracy_{col_name}'] = np.nan
                df[f'n_correct_{base_col}_{location}'] = np.nan
                df[f'n_{col_name}'] = np.nan

                # Paso 3: máscaras para los distintos casos
                mask_extracted = df[col_name].notna() & df[col_name].str.contains(r'\d+%\s*\(\d+/\d+\)', na=False)
                mask_direct = df[col_name].notna() & df[col_name].str.match(r'^\d+$')  # solo números

                # Caso 1: extracción con regex
                extracted = df.loc[mask_extracted, col_name].str.extract(r'(\d+)%\s*\((\d+)/(\d+)\)')
                if verbose >= 1:
                    print(f"Extracted: \n {extracted}")

                if extracted.isnull().any().any():
                    raise ValueError(f"Error al extraer datos en la columna {col_name}")

                extracted = extracted.apply(pd.to_numeric, errors='coerce')
                if extracted.isnull().any().any():
                    raise ValueError(f"Valores no convertibles a números en {col_name}")

                df.loc[mask_extracted, f'accuracy_{col_name}'] = extracted[0].values # # warning de Try using .loc[row_indexer,col_indexer] = value instead
                df.loc[mask_extracted, f'n_correct_{base_col}_{location}'] = extracted[1].values
                df.loc[mask_extracted, f'n_{col_name}'] = extracted[2].values

                # Caso 2: ya está en formato numérico, solo asignás `n_{col_name}`
                df.loc[mask_direct, f'n_{col_name}'] = df.loc[mask_direct, col_name].astype(float) # Por ejemplo, n_tackles es "15" o "61% (22/36)" (no funciona no se por qué)

                # Verificar rangos # Es molesto pero necesario. Si hay un error en el reformateo tiene que saltar ahora
                for new_col, (min_val, max_val) in zip([f'accuracy_{col_name}', f'n_correct_{base_col}_{location}', f'n_{col_name}'], column_specs.values()):
                    
                    # Verificar si la columna tiene el 100% de NaN
                    if df[new_col].isna().all():
                        logger.error(f"La columna {new_col} tiene el 100% de valores NaN.")
                        raise ValueError(f"La columna {new_col} está completamente vacía.")

                    # Filtrar NaN antes de verificar el rango (sino falla el rango a pesar de estar en el rango)
                    df_no_nan = df[new_col].dropna()

                    if not df_no_nan.between(min_val, max_val).all():
                        out_of_range = df_no_nan[~df_no_nan.between(min_val, max_val)]
                        logger.warning(f"Valores fuera del rango en {new_col}: {out_of_range.tolist()}") # no hago raise error, porque puede suceder que fs mida mal y prefiero borrarlo en clean_data()
                        logger.warning(f"Valores fuera de rango en {new_col}: {df_no_nan.min()} - {df_no_nan.max()} (esperado: {min_val} - {max_val})")

                # Eliminar columna original
                df.drop(columns=[col_name], inplace=True)
            
            else:
                logger.error(f"Falló el reformateo de {col_name}. Verifica que sea de tipo string o object.")
                print(df[col_name].dtype)
                print(df[col_name].head())
    return df

def format_penalties(df):

    df.loc[df['penalties'] == 'FINISHED', 'penalties'] = 0
    df.loc[df['penalties'] == 'AFTER EXTRA TIME', 'penalties'] = 0.5
    df.loc[df['penalties'] == 'AFTER PENALTIES', 'penalties'] = 1
    return df

def format_result_per_half(df):
    """
    Convierto resultado de un tiempo "2 - 1" en columnas goals "2" y "1".
    """
    
    df[['goals_1st_half_home', 'goals_1st_half_away']] = (
        df['result_1st_half']
        .str.strip()
        .str.split('-', expand=True)
        .apply(lambda x: x.str.strip().astype(int))
    )

    df[['goals_2nd_half_home', 'goals_2nd_half_away']] = (
        df['result_2nd_half']
        .str.strip()
        .str.split('-', expand=True)
        .apply(lambda x: x.str.strip().astype(int))
    )

    # Determino variable resultado por half
    from p3_data_preparation.construct_data import determine_result
    df = determine_result(df, var_resp="result_1h", var_goals="goals_1st_half") 
    df = determine_result(df, var_resp="result_2h", var_goals="goals_2nd_half")
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

def convert_columns_to_int(df, df_etiquetas: pd.DataFrame = None, prod: bool = False, verbose:int = 0):
    """
    Convierte las variables string a numéricas.

    # Parameters:
    df: DataFrame que contiene las variables a convertir. (DataFrame)
    df_etiquetas: DataFrame adicional con las etiquetas originales y enteros correspondientes. Si se proporciona, se utilizará 
    para la conversión en lugar de ajustar un nuevo LabelEncoder.(DataFrame, opcional)

    # Returns:
    DataFrame con las variables convertidas y un DataFrame adicional con las etiquetas originales y enteros correspondientes.
    """

    if not prod:

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

    else:
        if df_etiquetas is None:
            raise ValueError("Si prod=True, debés pasar df_etiquetas.")

        l_columnas_a_codificar = df_etiquetas['variable'].unique()
        l_columnas_a_codificar = [col for col in l_columnas_a_codificar if col in df.columns] 

        if verbose >= 0:
            print("[PROD] Columnas a codificar: ", l_columnas_a_codificar)

        # Por columna a codificar
        for col in l_columnas_a_codificar:

            # Obtengo etiquetas de la columna
            etiquetas_col = df_etiquetas[df_etiquetas['variable'] == col]
            d_mapeo = dict(zip(etiquetas_col['str_value'], etiquetas_col['int_value']))

            # Por partido
            for i, val in df[col].items():

                 # Si la etiqueta no existe en el mapeo
                if val not in d_mapeo:
                    if verbose >= 1:
                        print(f"[PROD] Valor desconocido '{val}' en fila {i}. Se reemplaza con 0.")
                    df.loc[i, col] = 0
                
                # Si la etiqueta existe
                else:
                    df.loc[i, col] = d_mapeo[val]

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
 
# VERIFICACION DE FORMATO
def verify_format(df, column_specs, verbose: int = 0):
    """
    Formatea un DataFrame según especificaciones de columnas.

    Parameters:
        df (pd.DataFrame): DataFrame a formatear.
        column_specs (dict): Diccionario con las especificaciones para cada columna.
            Formato: 
                {
                    'column_name': {
                        'dtype': tipo_dato,
                        'rango': [min, max] (opcional)
                    },
                    ...
                }
            
    Returns:
        pd.DataFrame: DataFrame formateado.

    Mejoras:
        - Verificar numero de valores unicos. 
        - Posibilidad de parsarle valores ejemplo por columna? Tal vez para las varibles que son string... porque con el rango ya esta.
    """

    # Warning si no se esta verificando el formato de una columna
    l_cols_missing = [col for col in df.columns if col not in column_specs.keys()]
    if verbose >= 0 and len(l_cols_missing) > 0: # No tiro error porque en prod no tengo todas las col (las reformat) y cuando reformateo tampoco.
        logger.warning(f"No se está verificando el formato de las siguientes columnas: {l_cols_missing}.")
        # logger.warning("Esto podria ser porque Flashscore tiene una nueva variable y habria que ver si es una vieja pero reformateada (como 'total_passes' que paso a ser 'passes')")

    for col, specs in column_specs.items():

        # Verificar si la columna existe en el DataFrame
        if col not in df.columns: # en prod no existen todas las col de df_match como las stats o goals, etc.
            if verbose >= 1:
                logger.warning(f"La columna '{col}' no existe en el DataFrame.")
            continue
        
        if 'dtype' in specs.keys():
            # Cambiar el tipo de dato
            df = convert_column_dtype(df, col, dtype=specs['dtype'], verbose=verbose)

            # Verificar dtype
            verify_column_dtype(df, col, dtype=specs['dtype'], verbose=verbose)
        
        if 'rango' in specs.keys():
            # Verificar rango, si está definido
            verify_column_range(df, col, specs['rango'], verbose=verbose)

    return df

def verify_column_dtype(df, col, dtype, verbose: int = 0):
    """
    Verifica si los valores de una columna son del tipo esperado.
    
    Args:
        df (pd.DataFrame): El DataFrame.
        col (str): Nombre de la columna a verificar.
        dtype (str): Tipo esperado.
    
    Returns:
        bool: True si todos los valores son del tipo esperado, de lo contrario lanza un error.
    """
    # Diccionario para mapear tipos de Pandas a tipos de Python
    pandas_to_python_types = {
        'Float64': (float, type(pd.NA)),  # Acepta float y valores NA de Pandas
        'Int64': (int, type(pd.NA)),      # Acepta int y valores NA de Pandas
        'string': (str, type(pd.NA)),     # Acepta strings y valores NA de Pandas
        'datetime64[ns]': (pd.Timestamp, type(pd.NaT)),  # Acepta fechas y valores NA de Pandas
    }

    # Seleccionar tipos válidos
    if dtype in pandas_to_python_types:
        python_types = pandas_to_python_types[dtype]
    else:
        python_types = dtype  # Usar el tipo directamente si no está en el diccionario

    # Filtrar valores no nulos
    non_nan_values = df[col].dropna()

    # Verificar tipos
    if not non_nan_values.apply(lambda x: isinstance(x, python_types)).all():
        raise ValueError(f"Columna '{col}' contiene valores que no son del tipo {dtype}.")
    
    if verbose >= 1:
        logger.critical(f"La columna {col} tiene el dtype esperado {dtype}")
    return True

def convert_column_dtype(df, col, dtype, verbose: int = 0):
    """
    Convierte el dtype de la columna al especificado solo si es necesario.

    Parameters:
        df (DataFrame): El DataFrame con la columna a convertir.
        col (str): El nombre de la columna a convertir.
        dtype (str): El tipo de dato al que se desea convertir.

    Returns:
        DataFrame: El DataFrame con la columna convertida (si fue necesario).
    """
    # Verificar si la columna ya tiene el dtype deseado
    if df[col].dtype == dtype:
        return df

    # Intentar convertir el tipo de dato
    try:
        df[col] = df[col].astype(dtype)
    except ValueError as e:
        raise ValueError(f"Error al convertir '{col}' a {dtype}: {e}")
    return df

def verify_column_range(df, col, rango, verbose: int = 0):
    """
    Verifica que las columnas cumplen los criterios de tipo y rango, ignorando valores NaN.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna a verificar.
        col (str): Nombre de la columna a verificar.
        rango (list): Lista con el mínimo y máximo permitido [min, max].
        dtype (tuple): Tipos de datos permitidos en la columna.
    """
    errors = []

    # Filtrar valores no nulos
    non_nan_values = df[col].dropna()
    col_dtype = df[col].dtype
    
    # Verificar si los valores están dentro del rango especificado
    if col_dtype in [int, float, 'Float64', 'Int64'] and rango is not None:

        val_min, val_max = rango[0], rango[1]

        if not non_nan_values.apply(lambda x: val_min <= x <= val_max).all():

            min_val = non_nan_values.min()
            max_val = non_nan_values.max()
            errors.append(
                f"Columna '{col}' contiene valores fuera del rango [{val_min}, {val_max}]. "
                f"Valores min y max: {min_val} --> {max_val}"
            )

        # Resultado de la verificación
        if errors:
            logger.error(f"Errores encontrados en la columna '{col}':")
            for error in errors:
                logger.error(error)
                raise ValueError
        else:
            if verbose >= 1:
                logger.critical(f"La columna '{col}' está dentro del rango esperado.")

def format_df_match(df, prod: bool = False):
    """
    Mejora:
        - Podria hacer una sola funcion que reciba un dictionary con el nombre al columna, el rango y el dtype deseado.
    """
    print("Formatting df_match de Flashscore...")

    column_specs = {
        'date': {'dtype': 'datetime64[ns]'},
        'referee': {'dtype': str},
        'venue': {'dtype': str},
        'capacity': {}, # {'dtype': int, 'rango': [0, 200000]}, # ValueError: Error al convertir 'capacity' a <class 'int'>: Cannot convert non-finite values (NA or inf) to integer
        'id_team_home': {'dtype': str},
        'id_team_away': {'dtype': str},
        'id_coach_home': {'dtype': str},
        'id_coach_away': {'dtype': str},
        'id_country': {'dtype': int, 'rango': [0, 300]},
        'id_competition': {'dtype': int, 'rango': [0, 10000]},
        'country': {'dtype': str},
        'competition': {'dtype': str},
        'is_cup': {'dtype': int, 'rango': [0, 1]},
        'season': {'dtype': str},
        'team_home': {'dtype': str},
        'team_away': {'dtype': str},
        'coach_home': {'dtype': str},	
        'coach_away': {'dtype': str},
        }
    
    # Evito columnas que se conocen post partido
    if not prod:
        column_specs.update({
            'attendance': {}, # {'dtype': int, 'rango': [0, 200000]}, 
            'goals_home': {'dtype': 'Int64', 'rango': [0, 12]},
            'goals_away': {'dtype': 'Int64', 'rango': [0, 12]},
            'expected_goals_(xg)_home': {'dtype': 'Float64', 'rango': [0, 12]},
            'expected_goals_(xg)_away': {'dtype': 'Float64', 'rango': [0, 12]},
            'ball_possession_home': {'dtype': 'Int64', 'rango': [0, 100]},  # Error al querer convertirlos al dtype por tener nan...
            'ball_possession_away': {'dtype': 'Int64', 'rango': [0, 100]},
            'goal_attempts_home': {'dtype': 'Int64', 'rango': [0, 80]},
            'goal_attempts_away': {'dtype': 'Int64', 'rango': [0, 80]},
            'shots_on_goal_home': {'dtype': 'Int64', 'rango': [0, 30]},
            'shots_on_goal_away': {'dtype': 'Int64', 'rango': [0, 30]},
            'shots_off_goal_home': {'dtype': 'Int64', 'rango': [0, 50]},
            'shots_off_goal_away': {'dtype': 'Int64', 'rango': [0, 50]},
            'free_kicks_home': {'dtype': 'Int64', 'rango': [0, 50]},
            'free_kicks_away': {'dtype': 'Int64', 'rango': [0, 50]},
            'corner_kicks_home': {'dtype': 'Int64', 'rango': [0, 30]},
            'corner_kicks_away': {'dtype': 'Int64', 'rango': [0, 30]},
            'offsides_home': {'dtype': 'Int64', 'rango': [0, 25]},
            'offsides_away': {'dtype': 'Int64', 'rango': [0, 25]},
            'throw-ins_home': {'dtype': 'Int64', 'rango': [0, 80]},
            'throw-ins_away': {'dtype': 'Int64', 'rango': [0, 80]},
            'goalkeeper_saves_home': {'dtype': 'Int64', 'rango': [0, 25]},
            'goalkeeper_saves_away': {'dtype': 'Int64', 'rango': [0, 25]},
            'fouls_home': {'dtype': 'Int64', 'rango': [0, 40]},
            'fouls_away': {'dtype': 'Int64', 'rango': [0, 40]},
            'yellow_cards_home': {'dtype': 'Int64', 'rango': [0, 14]},
            'yellow_cards_away': {'dtype': 'Int64', 'rango': [0, 14]},
            'total_passes_home': {'dtype': 'Int64', 'rango': [0, 3500]},
            'total_passes_away': {'dtype': 'Int64', 'rango': [0, 3500]},
            'tackles_home': {'dtype': 'Int64', 'rango': [0, 70]},
            'tackles_away': {'dtype': 'Int64', 'rango': [0, 70]},
            'attacks_home': {'dtype': 'Int64', 'rango': [0, 310]},
            'attacks_away': {'dtype': 'Int64', 'rango': [0, 310]},
            'dangerous_attacks_home': {'dtype': 'Int64', 'rango': [0, 260]},
            'dangerous_attacks_away': {'dtype': 'Int64', 'rango': [0, 260]},
            'clearances_completed_home': {'dtype': 'Int64', 'rango': [0, 80]},
            'clearances_completed_away': {'dtype': 'Int64', 'rango': [0, 80]},
            'red_cards_home': {'dtype': 'Int64', 'rango': [0, 5]},
            'red_cards_away': {'dtype': 'Int64', 'rango': [0, 5]},
            'blocked_shots_home': {'dtype': 'Int64', 'rango': [0, 30]},
            'blocked_shots_away': {'dtype': 'Int64', 'rango': [0, 30]},
            'completed_passes_home': {'dtype': 'Int64', 'rango': [0, 1500]},
            'completed_passes_away': {'dtype': 'Int64', 'rango': [0, 1500]},
            # 'pass_success_%_home': {'dtype': 'Int64', 'rango': [0, 100]},
            # 'pass_success_%_away': {'dtype': 'Int64', 'rango': [0, 100]},
            'goal_kicks_home': {'dtype': 'Int64', 'rango': [0, 10]},
            'goal_kicks_away':  {'dtype': 'Int64', 'rango': [0, 10]},
            'crosses_completed_home': {'dtype': 'Int64', 'rango': [0, 30]},
            'crosses_completed_away': {'dtype': 'Int64', 'rango': [0, 30]},
            'interceptions_home': {'dtype': 'Int64', 'rango': [0, 40]},
            'interceptions_away': {'dtype': 'Int64', 'rango': [0, 40]},
            'big_chances_home':	{'dtype': 'Int64', 'rango': [0, 20]},
            'big_chances_away':	{'dtype': 'Int64', 'rango': [0, 20]},
            'shots_inside_the_box_home': {'dtype': 'Int64', 'rango': [0, 50]},
            'shots_inside_the_box_away': {'dtype': 'Int64', 'rango': [0, 50]},
            'shots_outside_the_box_home': {'dtype': 'Int64', 'rango': [0, 30]},
            'shots_outside_the_box_away': {'dtype': 'Int64', 'rango': [0, 30]},	
            'hit_the_woodwork_home': {'dtype': 'Int64', 'rango': [0, 6]},
            'hit_the_woodwork_away': {'dtype': 'Int64', 'rango': [0, 6]},
            'headed_goals_home': {'dtype': 'Int64', 'rango': [0, 5]},
            'headed_goals_away': {'dtype': 'Int64', 'rango': [0, 5]},
            'touches_in_the_opposition_box_home': {'dtype': 'Int64', 'rango': [0, 100]},
            'touches_in_the_opposition_box_away': {'dtype': 'Int64', 'rango': [0, 100]},
            'passes_in_the_final_third_home':  {'dtype': 'Int64', 'rango': [0, 500]},
            'passes_in_the_final_third_away': {'dtype': 'Int64', 'rango': [0, 500]},
            'crosses_home': {'dtype': 'Int64', 'rango': [0, 70]},
            'crosses_away': {'dtype': 'Int64', 'rango': [0, 70]},
            'clearances_total_home': {'dtype': 'Int64', 'rango': [0, 110]},
            'clearances_total_away': {'dtype': 'Int64', 'rango': [0, 110]},
            })
            
    return verify_format(df, column_specs)
    
def format_df_match_player(df, prod: bool = False):
    print("Formatting df_match_player de Flashscore...")
    pass

def format_df_match_odds(df):
    print("Formatting df_match_odds de Flashscore...")

    column_specs = {
        'odds_home': {'dtype': 'Float64', 'rango': [1, 100]},
        'odds_draw': {'dtype': 'Float64', 'rango': [1, 100]},
        'odds_away': {'dtype': 'Float64', 'rango': [1, 100]},
        }
    
    return verify_format(df, column_specs)

def format_df_player_sofifa(df):
    print("Formatting df_player_sofifa de Sofifa...")

    column_specs = {
        'player_name': {'dtype': str},
        'player_name_short': {'dtype': str},
        'nationality': {'dtype': str},
        'height': {'dtype': int, 'rango': [100, 250]},
        'preferred_foot': {'dtype': str},
        'url_player': {'dtype': str},
        }
    
    df.index = df.index.astype(str)
    return verify_format(df, column_specs)

def format_df_player_fifa_sofifa(df):
    """
    Debo reformatear campos de sofifa extraidos nuevos. Esta fallando 'age' porque ahora es string en vez de int?
    """
    print("Formatting df_player_fifa_sofifa de Sofifa...")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    column_specs = {
        # Verifico formato
        'id_player': {'dtype': str},
        'date': {'dtype': 'datetime64[ns]'},
        'id_country': {'dtype': int, 'rango': [0, 300]},
        'id_competition': {'dtype': int, 'rango': [0, 10000]},
        'id_team': {'dtype': int},
        'age': {'dtype': int, 'rango': [14, 50]},
        'overall_rating': {'dtype': int, 'rango': [20, 100]},
        'potential': {'dtype': int, 'rango': [20, 100]},
        'value': {'dtype': float, 'rango': [100, 250000000]},
        'wage': {'dtype': float, 'rango': [100, 999999]},
        'int_reputation': {'dtype': int, 'rango': [0, 5]},
        'fifa': {'dtype': str},
        'fifa_year': {'dtype': int, 'rango': [6, 30]},
        }

    return verify_format(df, column_specs)

def value_nan_to_none(value):
    """
    Evitar nan y forzar None
    """
    if pd.isna(value):  # Verifica si es NaN o None
        return None
    else:
        return value
        
# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv() # Cargar las variables de entorno desde el archivo .env
    BASE_DIR_LOCAL = os.getenv('BASE_DIR_LOCAL')

    # Levanto datasets
    country = 'argentina'
    df_match = pd.read_excel(f'./data/{country}/p2_data_understanding/df_match.xlsx')
    df_player = pd.read_excel(f"./data/{country}/p2_data_understanding/df_player.xlsx", index_col=0)

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