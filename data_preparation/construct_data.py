import pandas as pd
import time
import math


def determinar_equipo_ganador(df):
    """
    Se determina el 'equipo_ganador' a partir de los goles que hizo cada equipo
    :param df: Dataframe. Unidad de analisis: partido. Columnas: entre ellas goles_loc y goles_vis
    :return: Dataframe pasado por parametro con nueva columna, 'equipo_ganador', que detalla el resultado del partido.
    """
    # Por fila (partido)
    for i in range(len(df)):

        # Obtengo goles de cada equipo
        ng1, ng2 = df.loc[i, 'goles_loc'], df.loc[i, 'goles_vis']

        # Guardo equipo ganador
        df.loc[i, 'equipo_ganador'] = 'Local' if ng1 > ng2 else 'Empate' if ng1==ng2 else "Visitante"
    return df

def promediar_var_en_ult_partidos(df, n_ult_part, variable, por_localia=False):
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos

    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :param variable: String. Nombre de variable a promediar
    :return: Dataframe con estadisticas promediadas
    """
    # Defincion de variables
    n_ult_part_min = int(0.6 * n_ult_part)  # Definir suficientes partidos mínimos
    str_adic = "_segun_loc" if por_localia else ""  # Variable adicional según localía

    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugó el equipo
        l_dfs = [df[df['equipo_loc'] == equipo], df[df['equipo_vis'] == equipo]] if por_localia else [
            df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]]

        # Inicializar lista de valores de la variable
        l = []

        for df_team in l_dfs:

            # Recorrer los partidos del equipo
            for idx, row in df_team.iterrows():

                # Definición de variables
                is_equipo_loc = row['equipo_loc'] == equipo
                loc_o_vis = 'loc' if is_equipo_loc else 'vis'

                # Si ya tengo los suficientes partidos para determinar la variable
                if len(l) == n_ult_part:

                    # Quito NaN de la lista para que no falle el cálculo
                    l_sin_nan = [x for x in l if not math.isnan(x)]

                    # Si no eliminé todos los elementos
                    if len(l_sin_nan) >= n_ult_part_min:

                        agregar_valor_promedio(df, idx, l_sin_nan, variable, loc_o_vis, str_adic)

                    # Elimino el valor más antiguo de la lista (para tener siempre los últimos n_part partidos)
                    l = l[1:]

                # Agrego el valor de la variable al final de la lista
                value = obtener_valor_variable(row, variable, is_equipo_loc, loc_o_vis)
                l.append(value)

    df = calcular_diferencia(df, variable, str_adic)
    return df

def agregar_valor_promedio(df, idx, l, variable, loc_o_vis, str_adic):  # cambiar nombre
    """
    Agrega el valor promedio de la variable al dataframe en la posición indicada por el índice.

    :param df: Dataframe.
    :param idx: Integer. Índice del partido.
    :param l: List. Lista de valores de la variable.
    :param variable: String. Nombre de variable a promediar.
    :param loc_o_vis: String. Etiqueta de local o visitante.
    :param str_adic: String. Sufijo adicional según localía.
    """
    if variable == 'goles' or variable == 'puntos':
        df.loc[idx, f'sum_{variable}_ult_part_{loc_o_vis}{str_adic}'] = sum(l)
    else:
        promedio = sum(l) / len(l)
        df.loc[idx, f'prom_{variable}_ult_part_{loc_o_vis}{str_adic}'] = promedio

def obtener_valor_variable(row, variable, is_equipo_loc, loc_o_vis):
    """
    Obtiene el valor de la variable según el partido y la localía.

    :param row: Series. Fila del dataframe correspondiente al partido.
    :param variable: String. Nombre de variable a obtener.
    :param is_equipo_loc: Boolean. Indica si el equipo es local en el partido.
    :param loc_o_vis: String. Etiqueta de local o visitante.
    :return: Valor de la variable.
    """
    if variable == 'goles':
        value = row['goles_loc'] - row['goles_vis'] if is_equipo_loc else row['goles_vis'] - row['goles_loc']
    elif variable == 'puntos':
        resultado = row['equipo_ganador']
        puntos_loc = lambda res: 3 if res == "Local" else (1 if res == "Empate" else 0)
        puntos_vis = lambda res: 3 if res == "Visitante" else (1 if res == "Empate" else 0)
        value = puntos_loc(resultado) if is_equipo_loc else puntos_vis(resultado)
    else:
        value = row[f'{variable}_{loc_o_vis}']
    return value

def calcular_diferencia(df, variable, str_adic):
    """
    Calcula la diferencia de la variable entre local y visitante y elimina las columnas temporales.

    :param df: Dataframe.
    :param variable: String. Nombre de variable a promediar.
    :param str_adic: String. Sufijo adicional según localía.
    """
    if variable == 'goles' or variable == 'puntos':
        df[f'dif_{variable}_segun_ult_part{str_adic}'] = df[f'sum_{variable}_ult_part_loc{str_adic}'] - df[f'sum_{variable}_ult_part_vis{str_adic}']

        # Elimino columnas utilizadas para calcular tanto el promedio como la diferencia
        df = df.drop(columns=[f'sum_{variable}_ult_part_loc{str_adic}', f'sum_{variable}_ult_part_vis{str_adic}'])
        # df = df.drop(columns=[f'sum_{variable}_ult_part_loc{str_adic}', f'sum_{variable}_ult_part_vis{str_adic}'])
    else:
        df[f'dif_{variable}_segun_ult_part{str_adic}'] = df[f'prom_{variable}_ult_part_loc{str_adic}'] - df[f'prom_{variable}_ult_part_vis{str_adic}']
        df = df.drop(columns=[f'{variable}_loc', f'{variable}_vis', f'prom_{variable}_ult_part_loc{str_adic}', f'prom_{variable}_ult_part_vis{str_adic}'])
        # df = df.drop(columns=[f'prom_{variable}_ult_part_loc{str_adic}', f'prom_{variable}_ult_part_vis{str_adic}'])

    return df

def historial_entre_si(df, n_ult_part, segun_loc=True):  # Garantizo que funciona
    """
    Crea columna 'equipo_ganador' donde se especifica el resultado de cada partido
    :param df: Dataframe. Unidad de analisis: partido. Columnas: al menos fecha, equipo_loc, equipo_vis y equipo_ganador
    :param n_ult_part: Integer. Numero de partidos a tener en cuenta para determinar el historial entre dos equipos.
    :param n_part_hist_min: Integer. Numero minimo de partidos a tener en cuenta para determinar el historial entre dos
    equipos.
    :return: Dataframe pasado por parametro con nueva columna, 'historial_entre_si', que permite a cual de los dos
    equipos de un partido lo favorece mas el historial entre ellos.
    """
    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Mas reciente a mas antiguo

    # Definicion de variables
    l_idxs_cargados = []
    col_name = 'historial_entre_si_segun_loc' if segun_loc else 'historial_entre_si'
    n_ult_part_min = 0.6 * n_ult_part

    # Por fila
    for i in range(len(df)):

        # Evito partidos a los cuales ya les cargue el historial
        if i not in l_idxs_cargados:

            # Definicion variables
            equipo_loc, equipo_vis = df.loc[i, 'equipo_loc'], df.loc[i, 'equipo_vis']
            # print(f' Historial {equipo_loc} - {equipo_vis} '.center(120, '+'))

            # Selecciono unicamente los partidos entre equipo1 y equipo2
            if segun_loc:
                df_historial = df[((df['equipo_loc'] == equipo_loc) & (df['equipo_vis'] == equipo_vis))]
            else:
                df_historial = df[((df['equipo_loc'] == equipo_loc) & (df['equipo_vis'] == equipo_vis)) | (
                            df['equipo_loc'] == equipo_vis) & (df['equipo_vis'] == equipo_loc)]

            # Por PARTIDO entre si (a c/u intentare asignarle un historial)
            for j in range(len(df_historial)):  # h toma [0, 1, 2, 3, 4, 5, 6, 7]

                # Definicion de variables
                l_historiales = []  # Reinicio historial
                l_idxs = df_historial.index  # Lista de indices de df_historial
                if not segun_loc:
                    equipo_loc, equipo_vis = df.loc[l_idxs[j], 'equipo_loc'], df.loc[l_idxs[j], 'equipo_vis']
                # print(f'Historial para el partido Nº{j+1} donde equipo local {equipo_loc} y equipo vis {equipo_vis}')

                # Por partido anterior al partido j (a tener en cuenta en historial)  # no hago que tome desde los 4 ult partidos del h anterior porque cambia el equipo local entre h y por ende el historial
                for k in range(j + 1, j + 1 + n_ult_part):

                    # Si aun hay <n_ult_part> anteriores a h
                    if k < len(df_historial):
                        # print(f"k={k}")

                        equipo_ganador = df.loc[l_idxs[k], 'equipo_ganador']  # {'Local', 'Empate', 'Visitante}


                        # Obtengo el equipo ganador y el equipo local del partido k
                        if segun_loc:
                            historial = +1 if equipo_ganador == 'Local' else 0 if equipo_ganador == 'Empate' else -1

                        else:
                            # Obtengo el equipo ganador y el equipo local del partido k
                            equipo_local = df.loc[l_idxs[k], 'equipo_loc']

                            # Si gano el local
                            if equipo_ganador == 'Local':
                                # Si el local del partido k es el equipo local en el partido h
                                historial = +1 if equipo_loc == equipo_local else -1

                            # Si empataron
                            elif equipo_ganador == 'Empate':
                                historial = 0

                            # Si gano el visitante
                            else:
                                # Si el local del partido k es el equipo local en el partido h
                                historial = -1 if equipo_loc == equipo_local else +1
                            # print(f"Equipo local en partido j={j}: {equipo_loc}")
                            # print(f"Equipo local en partido k={k}: {equipo_local}")
                            # print(f"Equipo ganador en partido k={k}: {equipo_ganador} --> historial = {historial}")

                        l_historiales.append(historial)
                        # print(l_historiales)

                    # Si ya no hay <n_ult_part> anteriores a h
                    else:
                        break

                l_idxs_cargados.append(list(l_idxs))

                # Quito nan de lista para que no falle la cuenta y de nan
                l_sin_nan = [x for x in l_historiales if not math.isnan(x)]

                # Guardo historial del partido j
                if len(l_sin_nan) >= n_ult_part_min:

                    df.loc[l_idxs[j], col_name] = sum(l_sin_nan)
                    # print(l_historiales)
                    # print(f"Historial cargado: {sum(l_historiales)}")

                # Si ya no hay al menos <n_part_hist_min> para determinar el historial
                else:
                    break
    return df

def rendimiento_equipo(df, n_ult_part, peso_puntos, por_localia=False):  # funciona perfecto la normalizacion
    """
    Calcula el rendimiento de un equipo utilizando la diferencia de goles y la cantidad de puntos obtenidos en los últimos partidos.
    :param df: DataFrame que contiene los datos del equipo.
    :param n_ult_part: Número de últimos partidos a considerar para calcular la diferencia de goles y puntos.
    :param peso_puntos: Peso que se le da a la cantidad de puntos en el rendimiento final.
    :return: DataFrame con la columna adicional de rendimiento del equipo.
    """
    # Definicion de variables
    str_adic = "_segun_loc" if por_localia else ""
    col_name_1 = f'dif_goles_segun_ult_part{str_adic}'
    col_name_2 = f'dif_puntos_segun_ult_part{str_adic}'

    # Calculo diferencia de gol
    df = promediar_var_en_ult_partidos(df, n_ult_part=n_ult_part, variable='goles', por_localia=por_localia)

    # Calculo diferencia de forma
    df = promediar_var_en_ult_partidos(df, n_ult_part=n_ult_part, variable='puntos', por_localia=por_localia)

    # Normalizar diferencia de goles y cantidad de puntos en el rango [-1, 1]
    df['dif_goles_norm'] = 2 * ((df[col_name_1] - df[col_name_1].min()) / (df[col_name_1].max() - df[col_name_1].min())) - 1
    df['dif_puntos_norm'] = 2 * ((df[col_name_2] - df[col_name_2].min()) / (df[col_name_2].max() - df[col_name_2].min())) - 1

    # Calcular variable de rendimiento
    df[f'dif_rendimiento_ult_part{str_adic}'] = (1-peso_puntos) * df['dif_goles_norm'] + peso_puntos * df['dif_puntos_norm']

    df = df.drop(['dif_goles_norm', 'dif_puntos_norm', col_name_1, col_name_2], axis=1)
    return df

def n_dias_ult_partido(df):  # Peopuesta por Chat GPT
    df = df.sort_values(by='fecha', ignore_index=True)

    df['dias_desde_ultimo_partido_loc'] = 0
    df['dias_desde_ultimo_partido_vis'] = 0

    for i, row in df.iterrows():
        equipo_loc = row['equipo_loc']
        equipo_vis = row['equipo_vis']

        # Calcula la diferencia de días desde el último partido jugado por equipo_loc
        mask_loc = ((df['equipo_loc'] == equipo_loc) | (df['equipo_vis'] == equipo_loc)) & (df['fecha'] < row['fecha'])
        ult_partido_loc = df.loc[mask_loc, 'fecha'].max()
        if pd.notnull(ult_partido_loc):
            dias_desde_ultimo_loc = (row['fecha'] - ult_partido_loc).days
            df.loc[i, 'dias_desde_ultimo_partido_loc'] = dias_desde_ultimo_loc

        # Calcula la diferencia de días desde el último partido jugado por equipo_vis
        mask_vis = ((df['equipo_loc'] == equipo_vis) | (df['equipo_vis'] == equipo_vis)) & (df['fecha'] < row['fecha'])
        ult_partido_vis = df.loc[mask_vis, 'fecha'].max()
        if pd.notnull(ult_partido_vis):
            dias_desde_ultimo_vis = (row['fecha'] - ult_partido_vis).days
            df.loc[i, 'dias_desde_ultimo_partido_vis'] = dias_desde_ultimo_vis

    df['dif_dias_ult_part'] = df['dias_desde_ultimo_partido_loc'] - df['dias_desde_ultimo_partido_vis']
    df = df.drop(['dias_desde_ultimo_partido_loc', 'dias_desde_ultimo_partido_vis'], axis=1)
    return df

def calculate_dif_col_jugadores(df):
    l_titularidad = ['tit', 'sup', 'aus']
    l_var_jug = ['edad', 'alt', 'rat', 'valor']

    for titularidad in l_titularidad:

        for var in l_var_jug:

            # Calculo suma de jugadores
            if (titularidad == 'aus') and (var == "rat"):

                nombre_col_loc = f'prom_{var}_jug_{titularidad}_loc'
                nombre_col_vis = f'prom_{var}_jug_{titularidad}_vis'

                nombre_col_loc_2 = f'n_jug_{titularidad}_loc'
                nombre_col_vis_2 = f'n_jug_{titularidad}_vis'

                nombre_col_dif = f'dif_sum_{var}_{titularidad}'
                df[nombre_col_dif] = df[nombre_col_loc] * df[nombre_col_loc_2] - df[nombre_col_vis] * df[nombre_col_vis_2]

                # Elimino variables utilizadas para calcular la diferencia
                df = df.drop([nombre_col_loc_2, nombre_col_vis_2], axis=1)

            else:
                # Obtengo nombre de variable local y visitante
                nombre_col_loc = f'prom_{var}_jug_{titularidad}_loc'
                nombre_col_vis = f'prom_{var}_jug_{titularidad}_vis'

                # Calculo diferencia entre local y visitante
                nombre_col_dif = f'dif_{var}_{titularidad}'
                df[nombre_col_dif] = df[nombre_col_loc] - df[nombre_col_vis]

            # Elimino variables utilizadas para calcular la diferencia
            df = df.drop([nombre_col_loc, nombre_col_vis], axis=1)
    return df


def prueba():
    # Definicion de variables
    N_ULT_PART = 5
    N_ULT_PART_LOC = 3
    peso_puntos = 0.6

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_integrated.xlsx')
    print(df.head())

    start = time.time()

    # Construyo variable respuesta: "equipo_ganador"
    df = determinar_equipo_ganador(df)

    # Construyo variables historicas
    df = historial_entre_si(df, n_ult_part=N_ULT_PART_LOC, segun_loc=True)
    df = historial_entre_si(df, n_ult_part=N_ULT_PART, segun_loc=False)

    df = rendimiento_equipo(df, n_ult_part=N_ULT_PART_LOC, peso_puntos=peso_puntos, por_localia=True)
    df = rendimiento_equipo(df, n_ult_part=N_ULT_PART, peso_puntos=peso_puntos, por_localia=False)

    l_estad_part = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp',
                    'offsides', 'ataques', 'ataques_pelig']
    for var in l_estad_part:
        df = promediar_var_en_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)
    # df = n_dias_ult_partido(df)  # Numero de dias desde ultimo partido

    # # Construyo variables de diferencias para las variables promedio de los jugadores
    df = calculate_dif_col_jugadores(df)

    # Elimino columnas usadas para construir datos
    df = df.drop(columns=['goles_loc', 'goles_vis'], axis=1)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba_2.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()