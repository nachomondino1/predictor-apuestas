import pandas as pd
import time
import math
from datetime import timedelta

# Construyo variables en dataframe "jugadores por partido"
def add_fecha(df_part, df_jug_part):
    """
    Agrega la fecha de cada partido al dataframe.

    :param df_part: Dataframe con fecha de cada partido. (DataFrame)
    :param df_jug_part: Dataframe con datos de jugadores en los partidos. (DataFrame)
    :return: Dataframe con datos de jugadores en los partidos incluyendo la fecha del partido. (DataFrame)
    """
    # Filtrar las columnas necesarias de df_jug_part
    df_part_filtered = df_part[['id_part', 'fecha']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_jug_part_with_date = pd.merge(df_jug_part, df_part_filtered, on='id_part', how='left')
    return df_jug_part_with_date

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

def determine_edad(df_jug_part):
    """
    Determino edad de cada jugador en cada partido a partir de la fecha del partido y la fecha de nacimiento del jugador

    :param df_jug_part: Dataframe con datos de cada jugador en cada partido incluyendoo fecha de nacimiento. (DataFrame)
    :return: Dataframe con datos de cada jugador en cada partido agregando columna edad. (DataFrame)
    """
    # Calculo edad (fecha_hora - fecha nac)
    diferencia_dias = (df_jug_part['fecha'] - df_jug_part['fecha_nac']).dt.days  # Calcular la diferencia en días
    df_jug_part['edad'] = diferencia_dias // 365  # Calcular la edad en años
    return df_jug_part

def determine_player_var_en_ult_partidos(df, variable, n_dias, tipo='mean', var_pond=None):
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos de cada jugador

    :param df: DataFrame.
    :param variable: String. Nombre de variable a promediar.
    :param n_dias: Integer. Numero de dias de los cuales obtener los datos.
    :param tipo: String. Tipo de cálculo a realizar ('mean_pond' para promedio ponderado, 'mean' para promedio, 'sum' para suma).
    :param var_pond: String. Nombre de la variable de ponderación en caso de promedio ponderado.
    :return: DataFrame con estadisticas promediadas.
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por jugador
    for id_jug in df['id_jug'].unique():
        # print(f"ID JUGADOR: {id_jug}")

        # Obtengo los partidos que jugó el jugador
        df_jugador = df[df['id_jug'] == id_jug]
        # print(df_filt)

        # Recorrer los partidos del jugador
        for idx, row in df_jugador.iterrows():

            fecha_part = row['fecha']
            fecha_limite = fecha_part - timedelta(days=n_dias)

            # Filtro para seleccionar los ultimos partidos del jugador en los ultimos n dias
            df_sel = df_jugador.loc[(df_jugador['fecha'] >= fecha_limite) & (df_jugador['fecha'] < fecha_part)]

            if len(df_sel) > 0:
                if tipo == "mean_pond":
                    df_sel = df_sel.dropna(subset=[var_pond, variable])
                    suma = df_sel[var_pond].sum()
                    if suma > 0:
                        df.loc[idx, f'prom_pond_{variable}_ult_part'] = (df_sel[var_pond] * df_sel[variable]).sum() / suma
                elif tipo == "mean":
                    df.loc[idx, f'prom_{variable}_ult_part'] = df_sel[variable].mean()

                elif tipo == "sum":
                    df.loc[idx, f'sum_{variable}_ult_part'] = df_sel[variable].sum()
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

def determinar_puntos(df):
    """
    Determina los puntos obtenidos por cada equipo segun el resultado del juego.

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

def determine_prom_en_ult_partidos(df, n_dias, variable, tipo):
    """
     Obtiene el promedio de las estadisticas en los ultimos partidos

     :param df: DataFrame.
     :param n_dias: Integer. Numero de dias de los cuales obtener los datos.
     :param variable: String. Nombre de la variable a promediar.
     :param tipo: String. Tipo de cálculo a realizar ('mean' para promedio, 'sum' para suma).
     :return: DataFrame con estadisticas promediadas
     """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugó el equipo
        df_equipo = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        # print(f"Equipo: {equipo}")
        # print(df_filt)

        # Por partido del equipo
        for idx, row in df_equipo.iterrows():

            loc_o_vis = 'loc' if row['equipo_loc'] == equipo else 'vis'
            fecha_part = row['fecha']
            fecha_limite = fecha_part - timedelta(days=n_dias)

            # Selecciono los ultimos partidos del equipo
            df_sel = df_equipo.loc[(df_equipo['fecha'] >= fecha_limite) & (df_equipo['fecha'] < fecha_part)]

            # Necesito la posesion segun si fue local o visitante en cada uno de esos partidos...
            s_valores_loc = df_sel.loc[df_sel['equipo_loc'] == equipo, f'dif_{variable}']
            s_valores_vis = df_sel.loc[df_sel['equipo_vis'] == equipo, f'dif_{variable}'] * -1  # -1 puesto que valores positivos en dif_variable es para el local y valores negativos es favor del visitante
            s_valores = pd.concat([s_valores_loc, s_valores_vis], ignore_index=True)

            if len(s_valores) > 0:
                if tipo == "mean":
                    # promedio =  sum(l_valores) / len(l_valores)
                    df.loc[idx, f'prom_ult_part_dif_{variable}_{loc_o_vis}'] = s_valores.mean()

                elif tipo == "sum":
                    # suma = sum(l_valores)
                    df.loc[idx, f'sum_ult_part_dif_{variable}_{loc_o_vis}'] = s_valores.sum()
    return df

def determine_prom_en_ult_partidos_localia(df, n_dias, variable, tipo):
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
        for df_equipo in l_dfs:
            # print('\n Iteracion')
            # print(df_equipo)

            # Por partido del equipo
            for idx, row in df_equipo.iterrows():

                loc_o_vis = 'loc' if row['equipo_loc'] == equipo else 'vis'
                fecha_part = row['fecha']
                fecha_limite = fecha_part - timedelta(days=n_dias)
                # print(fecha_part, fecha_limite)

                # Selecciono los ultimos partidos del equipo
                df_sel = df_equipo.loc[(df_equipo['fecha'] >= fecha_limite) & (df_equipo['fecha'] < fecha_part)]
                # print(df_sel)

                # Necesito la variable segun si fue local o visitante en cada uno de esos partidos...
                s_valores = df_sel[f'dif_{variable}'] if row['equipo_loc'] == equipo else df_sel[f'dif_{variable}'] * -1
                # print(s_valores)
                # print(s_valores.mean())

                if len(s_valores) > 0:
                    if tipo == "mean":
                        df.loc[idx, f'prom_ult_part_segun_localia_dif_{variable}_{loc_o_vis}'] = s_valores.mean()
                    elif tipo == "sum":
                        df.loc[idx, f'sum_ult_part_segun_localia_dif_{variable}_{loc_o_vis}'] = s_valores.sum()
    return df

def calcular_diferencia(df, variable, tipo, segun_loc=False):
    """
    Calcula la diferencia de la variable entre local y visitante y elimina las columnas temporales.

    :param df: Dataframe.
    :param variable: String. Nombre de variable a promediar.
    :param str_adic: String. Sufijo adicional según localía.
    """
    str_adic = "_segun_loc" if segun_loc else ""

    if tipo == 'mean':
        df[f'dif_prom_dif_{variable}_segun_ult_part{str_adic}'] = df[f'prom_dif_{variable}_ult_part_loc{str_adic}'] - df[f'prom_dif_{variable}_ult_part_vis{str_adic}']
        # df = df.drop(columns=[f'prom_dif_{variable}_ult_part_loc{str_adic}', f'prom_dif_{variable}_ult_part_vis{str_adic}'])

    elif tipo == "sum":
        df[f'dif_sum_dif_{variable}_segun_ult_part{str_adic}'] = df[f'sum_dif_{variable}_ult_part_loc{str_adic}'] - df[f'sum_dif_{variable}_ult_part_vis{str_adic}']
        # df = df.drop(columns=[f'sum_dif_{variable}_ult_part_loc{str_adic}', f'sum_dif_{variable}_ult_part_vis{str_adic}'])
    return df

def historial_entre_si_segun_fecha(df, n_anios):  # VERIFICAR (hecho)
    """
    Determina el historial entre los equipos que disputan el partido según los resultados en los últimos partidos entre ellos.

    :param df: DataFrame. Unidad de análisis: partido. Columnas: al menos fecha, equipo_loc, equipo_vis y equipo_ganador.
    :param n_anios: Integer. Número de años a tener en cuenta para determinar el historial entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'historial_entre_si_fecha', que permite determinar a cuál de los dos
    equipos de un partido le favorece más el historial entre ellos.
    """
    # Definicion de variables
    n_dias = 365 * n_anios
    l_equipos = df['equipo_loc'].unique()
    # print(f"Lista de equipos: {l_equipos}")

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Mas reciente a mas antiguo

    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]
        # print(f"Equipo 1: {l_equipos[i]}")

        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]
            # print(f"Equipo 2: {l_equipos[j]}")

            df_historial = df[((df['equipo_loc'] == eq1) & (df['equipo_vis'] == eq2)) | (
                            df['equipo_loc'] == eq2) & (df['equipo_vis'] == eq1)]
            # print(df_historial)

            # Por partido del historial
            for idx, row in df_historial.iterrows():

                equipo_local = row['equipo_loc']
                historial = 0
                fecha_part = row['fecha']
                fecha_limite = fecha_part - timedelta(days=n_dias)
                # print(f"Equipo local en partido {idx}: {equipo_local}".center(120))
                # print(f"Fecha: {fecha_part} ; Fecha limite: {fecha_limite}")

                # Selecciono los ultimos partidos
                df_sel = df_historial.loc[(df_historial['fecha'] >= fecha_limite) & (df_historial['fecha'] < fecha_part)]
                # print(df_sel)

                # Por ultimos partidos
                for index, fila in df_sel.iterrows():

                    if fila['equipo_ganador'] == "Local":
                        historial += +1 if fila['equipo_loc'] == equipo_local else -1

                    elif fila['equipo_ganador'] == "Visitante":
                        historial += -1 if fila['equipo_loc'] == equipo_local else +1

                    else:
                        historial += 0
                    # print(f"Index: {index} ; Equipo ganador: {fila['equipo_ganador']} ; Equipo local: {fila['equipo_loc']}")
                    # print(historial)

                # Guardo historial
                if len(df_sel) > 0:  # Para evitar guardar historial = 0 en partidos donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                    df.loc[idx, 'historial_entre_si_fecha'] = historial
    return df

def historial_entre_si_localia_segun_fecha(df, n_anios):  # VERIFICAR
    """
    Determina el historial entre los equipos que disputan el partido segun los resultados en los ultimos partidos entre
    ellos.

    :param df: Dataframe. Unidad de analisis: partido. Columnas: al menos fecha, equipo_loc, equipo_vis y equipo_ganador
    :param n_anios: Integer. Numero de años a tener en cuenta para determinar el historial entre dos equipos.
    :return: Dataframe pasado por parametro con nueva columna, 'historial_entre_si', que permite a cual de los dos
    equipos de un partido lo favorece mas el historial entre ellos.
    """
    # Definicion de variables
    n_dias = 365 * n_anios
    l_equipos = df['equipo_loc'].unique()

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Mas reciente a mas antiguo

    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]

        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]

            l_dfs = [df[(df['equipo_loc'] == eq1) & (df['equipo_vis'] == eq2)],
                     df[(df['equipo_loc'] == eq2) & (df['equipo_vis'] == eq1)]]

            for df_historial in l_dfs:

                # Por partido del historial
                for idx, row in df_historial.iterrows():

                    equipo_local = row['equipo_loc']
                    historial = 0
                    fecha_part = row['fecha']
                    fecha_limite = fecha_part - timedelta(days=n_dias)

                    # Selecciono los ultimos partidos
                    df_sel = df_historial.loc[(df_historial['fecha'] >= fecha_limite) & (df_historial['fecha'] < fecha_part)]

                    # Por ultimos partidos
                    for index, fila in df_sel.iterrows():

                        if fila['equipo_ganador'] == "Local":
                            historial += +1 if fila['equipo_loc'] == equipo_local else -1

                        elif fila['equipo_ganador'] == "Visitante":
                            historial += -1 if fila['equipo_loc'] == equipo_local else +1

                        else:
                            historial += 0

                    # Guardo historial
                    if len(df_sel) > 0:
                        df.loc[idx, 'historial_entre_si_segun_loc_fecha'] = historial
    return df

def calculate_dif_col_jugadores(df): # Funciona bien

    l_var_jug = ['prom_edad', 'prom_alt', 'sum_rat', 'sum_min']  # l_var_jug = ['prom_edad', 'prom_alt', 'prom_rat', 'sum_min']
    l_titularidad = ['titular', 'suplente']

    for titularidad in l_titularidad:

        for var in l_var_jug:
            # Calculo diferencia entre local y visitante
            df[f'dif_{var}_{titularidad}'] = df[f'{var}_home_{titularidad}'] - df[f'{var}_away_{titularidad}']

            # Elimino variables utilizadas para calcular la diferencia
            df = df.drop([f'{var}_home_{titularidad}', f'{var}_away_{titularidad}'], axis=1)
    return df

def prueba():
    # Definicion de variables
    n_dias = 30  # 30 es como N_ULT_PART igual a 5...
    n_dias_loc = n_dias * 2  # 30 es como N_ULT_PART igual a 2
    n_anios_historial = 2
    n_anios_historial_loc = n_anios_historial * 2
    l_estadisticas = ['goles', 'puntos', 'posesion', 'remates', 'remates_a_puerta', 'porc_pases_comp', 'pases',
                      'pases_clave', 'amagues', 'duelos_aereos', 'tackles', 'intercepciones', 'corners', 'offsides',
                      'faltas']  # 'remates_fuera', 'remates_block', 'pases_comp' 'remates_palos',
    # l_var = ['goles', 'ht_goles', 'puntos', 'rating', 'remates_a_puerta']

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/argentina_south_america/df_integrated.xlsx')
    print(df.head())

    start = time.time()

    # Construyo variables: "equipo_gandor", diferencia de goles y puntos obtenidos
    df = determinar_equipo_ganador(df)
    df = determinar_puntos(df)

    # Variables historicas
    df = historial_entre_si_segun_fecha(df, n_anios=n_anios_historial)
    df = historial_entre_si_localia_segun_fecha(df, n_anios=n_anios_historial_loc)

    # Por estadistica del partido
    for var in l_estadisticas:

        # Determinar la diferencia de la estadistica entre equipo local y visitante de cada partido
        df[f'dif_{var}'] = df[f'{var}_loc'] - df[f'{var}_vis']  # (e.g. dif_goles = goles_loc - goles_vis)
        df = df.drop([f'{var}_loc', f'{var}_vis'], axis=1)  # (e.g. borro goles_loc y goles_vis)

        # Determinar para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
        # if var in l_var:
        df = determine_prom_en_ult_partidos(df, n_dias=n_dias, variable=var, tipo='mean')

        df = determine_prom_en_ult_partidos_localia(df, n_dias=n_dias_loc, variable=var, tipo='mean')
        df = df.drop([f'dif_{var}'], axis=1)

        # Determinar la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_loc y prom_dif_goles_vis)
        # if var in l_var:
        df[f'dif_prom_ult_part_dif_{var}'] = df[f'prom_ult_part_dif_{var}_loc'] - df[f'prom_ult_part_dif_{var}_vis']

        df[f'dif_prom_ult_part_segun_localia_dif_{var}'] = df[f'prom_ult_part_segun_localia_dif_{var}_loc'] - df[f'prom_ult_part_segun_localia_dif_{var}_vis']

        # if var in l_var:
        df = df.drop(columns=[f'prom_ult_part_dif_{var}_loc', f'prom_ult_part_dif_{var}_vis'], axis=1)

        df = df.drop(columns=[f'prom_ult_part_segun_localia_dif_{var}_loc', f'prom_ult_part_segun_localia_dif_{var}_vis'], axis=1)

    # Construyo variables de diferencias para las variables promedio de los jugadores
    df = calculate_dif_col_jugadores(df)

    # Elimino variables que no construire
    df = df.drop(columns=['remates_palos_loc', 'remates_palos_vis', 'remates_fuera_loc', 'remates_fuera_vis', f'remates_block_loc', f'remates_block_vis', 'pases_comp_loc', 'pases_comp_vis'], axis=1)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba_2.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()