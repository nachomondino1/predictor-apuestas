import pandas as pd
import time
import math
from datetime import timedelta


# Construyo variables en dataframe "jugadores por partido"
def determine_min_played(df):
    """
    Determino minutos jugados por jugador en cada partido segun titularidad y el minuto de su cambio.

    :param df: Dataframe con datos de jugador por partido incluyendo titularidad y minuto de cambio. (DataFrame)
    :return: Dataframe con datos de jugador por partido agregando columna de minutos jugados. (DataFrame)
    """

    def calculate_minutes_played(row):
        if row['titularidad'] == 'titular':
            if pd.isna(row['min_cambio']) or row['min_cambio'] >= 90:
                return 90
            else:
                return row['min_cambio']
        elif row['titularidad'] == 'suplente':
            if pd.isna(row['min_cambio']) or row['min_cambio'] >= 90:
                return 0
            else:
                return 90 - row['min_cambio']
        else:
            return None

    # Aplicar la función a cada fila del DataFrame para calcular los minutos jugados.
    df['min_played'] = df.apply(calculate_minutes_played, axis=1)
    return df

def determine_edad(df_part, df_jug_part):
    """
    Determino edad de cada jugador en cada partido a partir de la fecha del partido y la fecha de nacimiento del jugador

    :param df_part: Dataframe con fecha de cada partido. (DataFrame)
    :param df_jug_part: Dataframe con datos de cada jugador en cada partido incluyendoo fecha de nacimiento. (DataFrame)
    :return: Dataframe con datos de cada jugador en cada partido agregando columna edad. (DataFrame)
    """
    # Filtrar las columnas necesarias de df_jug_part
    df_part_filtered = df_part[['id_part', 'fecha']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_merged = pd.merge(df_jug_part, df_part_filtered, on='id_part', how='left')

    # Calculo edad (fecha_hora - fecha nac)
    diferencia_dias = (df_merged['fecha'] - df_merged['fecha_nac']).dt.days  # Calcular la diferencia en días
    df_merged['edad'] = diferencia_dias // 365  # Calcular la edad en años
    # df_merged = df_merged.drop(['fecha_nac'], axis=1)  # Borro columnas fecha_hora y fecha_nac
    return df_merged

def determine_player_var_en_ult_partidos(df, variable, n_dias, type='mean', var_pond=None):  # Calcula bien.
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos

    :param df: Dataframe.
    :param variable: String. Nombre de variable a promediar
    :param type: String.
    :return: Dataframe con estadisticas promediadas
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por jugador
    for id_jug in df['id_jug'].unique():
        # print(f"ID JUGADOR: {id_jug}")

        # Obtengo los partidos que jugó el jugador
        df_filt = df[df['id_jug'] == id_jug]
        # print(df_filt)

        # Recorrer los partidos del jugador
        for idx, row in df_filt.iterrows():

            fecha_part = row['fecha']
            fecha_limite = fecha_part - timedelta(days=n_dias)

            # Filtro para seleccionar los ultimos partidos del jugador en los ultimos n dias
            df_seleccionados = df_filt.copy()
            df_seleccionados = df_seleccionados.dropna(subset=[variable])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)
            df_seleccionados = df_seleccionados.loc[(df_seleccionados['fecha'] >= fecha_limite) & (df_seleccionados['fecha'] < fecha_part)]

            if len(df_seleccionados) > 0:
                if type == "mean_pond":
                    df_seleccionados = df_seleccionados.dropna(subset=[var_pond])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)

                    if len(df_seleccionados) > 0:
                        promedio_ponderado = (df_seleccionados[var_pond] * df_seleccionados[variable]).sum() / df_seleccionados[var_pond].sum()
                        df.loc[idx, f'prom_pond_{variable}_ult_part'] = promedio_ponderado

                elif type == "mean":
                    promedio = df_seleccionados[variable].sum() / len(df_seleccionados[variable])  # df_seleccionados[variable].dropna().sum() / len(df_seleccionados[variable].dropna())
                    df.loc[idx, f'prom_{variable}_ult_part'] = promedio

                elif type == "sum":
                    suma = df_seleccionados[variable].sum()  # df_seleccionados[variable].dropna().sum()
                    df.loc[idx, f'sum_{variable}_ult_part'] = suma

    return df

# Construyo variables en dataframe "partido"
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

def determinar_dif_goles(df):

    df['dif_goles_loc'] = df['goles_loc'] - df['goles_vis']
    df['dif_goles_vis'] = df['goles_vis'] - df['goles_loc']
    return df

def determinar_puntos(df):
    """
    Determina los puntos obtenidos por cada equipo segun el resultado del juego
    :param df:
    :return:
    """
    df['puntos_loc'] = 0
    df['puntos_vis'] = 0

    df.loc[df['equipo_ganador'] == 'Local', 'puntos_loc'] = 3
    df.loc[df['equipo_ganador'] == 'Local', 'puntos_vis'] = 0

    df.loc[df['equipo_ganador'] == 'Empate', 'puntos_loc'] = 1
    df.loc[df['equipo_ganador'] == 'Empate', 'puntos_vis'] = 1

    df.loc[df['equipo_ganador'] == 'Visitante', 'puntos_loc'] = 0
    df.loc[df['equipo_ganador'] == 'Visitante', 'puntos_vis'] = 3
    return df

def determine_var_en_ult_partidos(df, n_dias, variable, type):
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos

    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :param variable: String. Nombre de variable a promediar
    :return: Dataframe con estadisticas promediadas
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugó el equipo
        df_filt = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        # print(f"Equipo: {equipo}")
        # print(df_filt)

        # Por partido del equipo
        for idx, row in df_filt.iterrows():

            loc_o_vis = 'loc' if row['equipo_loc'] == equipo else 'vis'
            # print(f"El equipo es {loc_o_vis} en el partido de indice {idx}")

            fecha_part = row['fecha']
            fecha_limite = fecha_part - timedelta(days=n_dias)
            # print(f"Fecha {fecha_part} Fecha limite: {fecha_limite}")

            # Selecciono los ultimos partidos del equipo
            df_seleccionados = df_filt.copy()
            df_seleccionados = df_seleccionados.dropna(subset=[f'{variable}_loc', f'{variable}_vis'])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)
            df_seleccionados = df_seleccionados.loc[(df_seleccionados['fecha'] >= fecha_limite) & (df_seleccionados['fecha'] < fecha_part)]
            # print(df_seleccionados)
            # print(df_seleccionados.shape)

            # Necesito la posesion segun si fue local o visitante en cada uno de esos partidos...
            l_valores_loc = df_seleccionados[df_seleccionados['equipo_loc'] == equipo][f'{variable}_loc']
            l_valores_vis = df_seleccionados[df_seleccionados['equipo_vis'] == equipo][f'{variable}_vis']
            l_valores = l_valores_loc.tolist() + l_valores_vis.tolist()
            # print(l_valores)

            if len(l_valores) > 0:
                if type == "mean":
                    promedio = sum(l_valores) / len(l_valores)
                    df.loc[idx, f'prom_{variable}_ult_part_{loc_o_vis}'] = promedio
                    # print(promedio)

                elif type == "sum":
                    suma = sum(l_valores)
                    df.loc[idx, f'sum_{variable}_ult_part_{loc_o_vis}'] = suma
                    # print(suma)

    df = calcular_diferencia(df, variable, type)
    return df

def determine_var_en_ult_partidos_localia(df, n_dias, variable, type):
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos

    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :param variable: String. Nombre de variable a promediar
    :return: Dataframe con estadisticas promediadas
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugó el equipo
        l_dfs = [df[df['equipo_loc'] == equipo], df[df['equipo_vis'] == equipo]]

        # Por localia
        for df_filt in l_dfs:

            # Por partido del equipo
            for idx, row in df_filt.iterrows():

                loc_o_vis = 'loc' if row['equipo_loc'] == equipo else 'vis'

                fecha_part = row['fecha']
                fecha_limite = fecha_part - timedelta(days=n_dias)

                # Selecciono los ultimos partidos del equipo
                df_seleccionados = df_filt.copy()
                df_seleccionados = df_seleccionados.dropna(subset=[f'{variable}_{loc_o_vis}'])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)
                df_seleccionados = df_seleccionados.loc[(df_seleccionados['fecha'] >= fecha_limite) & (df_seleccionados['fecha'] < fecha_part)]

                # Necesito la posesion seegun si fue local o visitante en cada uno de esos partidos...
                l_valores = df_seleccionados[f'{variable}_{loc_o_vis}'].tolist()

                if len(l_valores) > 0:
                    if type == "mean":
                        promedio = sum(l_valores) / len(l_valores)
                        df.loc[idx, f'prom_{variable}_ult_part_{loc_o_vis}_segun_loc'] = promedio

                    elif type == "sum":
                        suma = sum(l_valores)
                        df.loc[idx, f'sum_{variable}_ult_part_{loc_o_vis}_segun_loc'] = suma

    df = calcular_diferencia(df, variable, type, segun_loc=True)
    return df

def calcular_diferencia(df, variable, type, segun_loc=False):
    """
    Calcula la diferencia de la variable entre local y visitante y elimina las columnas temporales.

    :param df: Dataframe.
    :param variable: String. Nombre de variable a promediar.
    :param str_adic: String. Sufijo adicional según localía.
    """
    str_adic = "_segun_loc" if segun_loc else ""

    if type == 'mean':
        df[f'dif_{variable}_segun_ult_part{str_adic}'] = df[f'prom_{variable}_ult_part_loc{str_adic}'] - df[f'prom_{variable}_ult_part_vis{str_adic}']
        # df = df.drop(columns=[f'prom_{variable}_ult_part_loc{str_adic}', f'prom_{variable}_ult_part_vis{str_adic}'])

    elif type == "sum":
        df[f'dif_{variable}_segun_ult_part{str_adic}'] = df[f'sum_{variable}_ult_part_loc{str_adic}'] - df[f'sum_{variable}_ult_part_vis{str_adic}']
        # df = df.drop(columns=[f'sum_{variable}_ult_part_loc{str_adic}', f'sum_{variable}_ult_part_vis{str_adic}'])
    return df

def historial_entre_si_segun_fecha(df, n_dias=730, segun_loc=True):  # VERIFICAR Y HACER SEGUN LOCALIA....
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

    l_equipos = df['equipos_loc'].unique()

    for i in range(len(l_equipos)):

        for j in range(i+1, len(l_equipos)):

            eq1, eq2 = l_equipos[i], l_equipos[j]

            df_historial = df[((df['equipo_loc'] == eq1) & (df['equipo_vis'] == eq2)) | (
                            df['equipo_loc'] == eq1) & (df['equipo_vis'] == eq2)]

            # Por partido del historial
            for idx, row in df_historial.iterrows():

                equipo_local = row['equipo_loc']
                historial = 0

                fecha_part = row['fecha']
                fecha_limite = fecha_part - timedelta(days=n_dias)

                # Selecciono los ultimos partidos
                df_sel = df_historial.copy()
                df_sel = df_sel.loc[(df_sel['fecha'] >= fecha_limite) & (df_sel['fecha'] < fecha_part)]

                # Por ultimos partidos
                for index, fila in df_sel.iterrows():

                    if fila['equipo_ganador'] == "Local":
                        if fila['equipo_loc'] == equipo_local:
                            historial += 1
                        else:
                            historial += -1

                    elif fila['equipo_ganador'] == "Visitante":
                        if fila['equipo_loc'] == equipo_local:
                            historial -= 1
                        else:
                            historial += 1
                    else:
                        historial += 0

                # Guardo historial
                df.loc[idx, 'historial_entre_si'] = historial

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

def n_dias_ult_partido(df):  # Propuesta por Chat GPT. Creo que no la usare porque tengo mejores medidas del nivel de cansancio del equipo como sum_min_played y esta no requiere de eliminacion de outliers
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
    n_dias = 30  # 30 es como N_ULT_PART igual a 5...
    n_dias_loc = n_dias * 2  # 30 es como N_ULT_PART igual a 2

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/argentina/df_part_formated.xlsx')
    print(df.head())

    start = time.time()

    # Construyo variable respuesta: "equipo_ganador"
    df = determinar_equipo_ganador(df)

    # Determino diferencia de goles y puntos obtenidos
    df = determinar_dif_goles(df)
    df = determinar_puntos(df)
    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba_3.xlsx')

    # Construyo variables historicas
    # l_var = ['dif_goles', 'puntos', 'posesion', 'remates', 'remates_a_puerta', 'remates_palos', 'remates_fuera',
    #          'remates_block', 'porc_pases_comp', 'pases', 'pases_comp', 'pases_clave', 'amagues', 'duelos_aereos',
    #          'tackles', 'intercepciones', 'corners', 'offsides', 'faltas']
    l_var = ['dif_goles', 'puntos', 'posesion']

    for variable in l_var:
        df = determine_var_en_ult_partidos(df, n_dias=n_dias, variable=variable, type='mean')
        df = determine_var_en_ult_partidos_localia(df, n_dias=n_dias_loc, variable=variable, type='mean')

    # historial entre si
    # df = historial_entre_si(df, n_ult_part=N_ULT_PART_LOC, segun_loc=True)
    # df = historial_entre_si(df, n_ult_part=N_ULT_PART, segun_loc=False)

    # Construyo variables de diferencias para las variables promedio de los jugadores
    # df = calculate_dif_col_jugadores(df)

    # Eliminar columnas que usé para construir... lo puedeo hacer en select_data tambien..
    # df = df.drop(columns=['goles_loc', 'goles_vis'], axis=1)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba_2.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()