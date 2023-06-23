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

def promedio_ult_partidos(df, n_ult_part, l_var):  # Esta hecha para promediar la variable en los ultimos partidos, no para sumar..
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos

    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :param variable: String. Nombre de variable a promediar
    :return: Dataframe con estadisticas promediadas
    """
    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)
    n_ult_part_min = 0.6 * n_ult_part

    for variable in l_var:

        # Por equipo
        for equipo in df['equipo_loc'].unique():

            # Obtengo los partidos que jugo el equipo
            df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
            l = []
            # print(" Equipo: {} ".format(equipo).center(120, "#"))

            # Por partido que jugó el equipo
            for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

                # Definicion de variables
                is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo
                loc_o_vis = 'loc' if is_equipo_loc else 'vis'

                # Si ya tengo los suficientes partidos para determinar la variable
                if len(l) == n_ult_part:

                    # Quito nan de lista para que no falle la cuenta y de nan (ESTABA FALLANDO!!)
                    l_sin_nan = list(filter(lambda x: not math.isnan(x), l))  # Y si es None?

                    # Si no eliminé todos los elementos
                    if len(l_sin_nan) >= n_ult_part_min:  # No puede ser n_ult_part porque elimina nan... entonces cada vez que eliminan, no entraria...

                        # Guardo el valor promedio de la variable en los ultimos n_part
                        df.loc[idx, f'prom_{variable}_ult_part_{loc_o_vis}'] = sum(l_sin_nan) / len(l_sin_nan)
                        # print('Con valores agregados:', list(df.loc[idx]))

                    # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                    l = l[1:]

                # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
                l.append(df.loc[idx, f'{variable}_{loc_o_vis}'] if is_equipo_loc else df.loc[idx, f'{variable}_{loc_o_vis}'])
                # print(f'{variable} ultimos partidos: {l}')

        # Calculo diferencia entre variables promedio para el equipo local y para el equipo vis
        df[f'dif_{variable}_segun_ult_part'] = df[f'prom_{variable}_ult_part_loc'] - df[f'prom_{variable}_ult_part_vis']

        # Elimino columnas utilizadas para calcular tanto el promedio como la diferencia
        df = df.drop(columns=[f'{variable}_loc', f'{variable}_vis', f'prom_{variable}_ult_part_loc', f'prom_{variable}_ult_part_vis'])
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
                l_sin_nan = list(filter(lambda x: not math.isnan(x), l_historiales))

                # Guardo historial del partido j
                if len(l_sin_nan) >= n_ult_part_min:

                    df.loc[l_idxs[j], col_name] = sum(l_sin_nan)
                    # print(l_historiales)
                    # print(f"Historial cargado: {sum(l_historiales)}")

                # Si ya no hay al menos <n_part_hist_min> para determinar el historial
                else:
                    break
    return df

def rendimiento_equipo(df, n_ult_part, peso_puntos):  # funciona perfecto la normalizacion
    """
    Calcula el rendimiento de un equipo utilizando la diferencia de goles y la cantidad de puntos obtenidos en los últimos partidos.
    :param df: DataFrame que contiene los datos del equipo.
    :param n_ult_part: Número de últimos partidos a considerar para calcular la diferencia de goles y puntos.
    :param peso_puntos: Peso que se le da a la cantidad de puntos en el rendimiento final.
    :return: DataFrame con la columna adicional de rendimiento del equipo.
    """
    # Calculo diferencia de gol
    df = dif_gol_ult_part(df, n_ult_part)

    # Calculo diferencia de forma
    df = dif_cant_puntos_ult_part(df, n_ult_part)

    # Normalizar diferencia de goles y cantidad de puntos en el rango [-1, 1]
    df['dif_goles_norm'] = 2 * ((df['dif_gol_ult_part'] - df['dif_gol_ult_part'].min()) / (df['dif_gol_ult_part'].max() - df['dif_gol_ult_part'].min())) - 1
    df['dif_puntos_norm'] = 2 * ((df['dif_cant_punt_ult_part'] - df['dif_cant_punt_ult_part'].min()) / (df['dif_cant_punt_ult_part'].max() - df['dif_cant_punt_ult_part'].min())) - 1

    # Calcular variable de rendimiento
    df['dif_rendimiento_ult_part'] = (1-peso_puntos) * df['dif_goles_norm'] + peso_puntos * df['dif_puntos_norm']

    df = df.drop(['dif_goles_norm', 'dif_puntos_norm', 'dif_gol_ult_part', 'dif_cant_punt_ult_part'], axis=1)
    return df

def dif_gol_ult_part(df, n_ult_part):  # Con promedio_ult_part() funciona pero solo para goles_anotados... (verificado)
    """
    Determina la cantidad de goles anotados y recibidos en los ultimos partidos
    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :return: Dataframe con columnas goles_ult_part_loc y goles_ult_part_vis.
    """
    # Definicion de variables
    n_ult_part_min = 0.6 * n_ult_part

    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_dif_goles = []
        # print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo
            loc_o_vis = 'loc' if is_equipo_loc else 'vis'

            # Si ya tengo los suficientes partidos para determinar los goles del equipo
            if len(l_dif_goles) == n_ult_part:

                # Quito nan de lista para que no falle la cuenta y de nan (ESTABA FALLANDO!!)
                l_sin_nan = list(filter(lambda x: not math.isnan(x), l_dif_goles))  # Y si es None?

                # Si no eliminé todos los elementos
                if len(l_sin_nan) >= n_ult_part_min:  # No puede ser n_ult_part porque elimina nan... entonces cada vez que eliminan, no entraria...

                    # Guardo el valor promedio de la variable en los ultimos n_part
                    df.loc[idx, f'dif_gol_ult_part_{loc_o_vis}'] = sum(l_sin_nan)
                    # print('Con valores agregados:', list(df.loc[idx]))

                # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_dif_goles = l_dif_goles[1:]

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            dif_gol = df.loc[idx, 'goles_loc'] - df.loc[idx, 'goles_vis'] if is_equipo_loc else df.loc[idx, 'goles_vis'] - df.loc[idx, 'goles_loc']
            l_dif_goles.append(dif_gol)
            # print(f'Dif gol: {l_dif_goles}')

    # Calculo diferencia entre local y visitante
    df['dif_gol_ult_part'] = df['dif_gol_ult_part_loc'] - df['dif_gol_ult_part_vis']

    # Elimino variables
    # df = df.drop(columns=['goles_loc', 'goles_vis', 'dif_gol_ult_part_loc', 'dif_gol_ult_part_vis'], axis=1)
    df = df.drop(columns=['dif_gol_ult_part_loc', 'dif_gol_ult_part_vis'], axis=1)
    return df

def dif_cant_puntos_ult_part(df, n_ult_part):  # Es igual a promedio_ult_part no mas que tengo que ver el nombre de la variable...
    # Requiere de dataframe ordenado por fecha decreciente
    # Tengo que agregar forma de cada equipo y puntaje antes de cada partido.

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)  # Mas antiguo a mas reciente

    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()
    puntos_loc = lambda res: 3 if res == "Local" else (1 if res == "Empate" else 0)
    puntos_vis = lambda res: 3 if res == "Visitante" else (1 if res == "Empate" else 0)
    n_ult_part_min = 0.6 * n_ult_part

    # Por equipo
    for equipo in l_equipos:

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_puntos = []
        # print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugo el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            resultado = df_team.loc[idx, 'equipo_ganador']
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo
            loc_o_vis = 'loc' if is_equipo_loc else 'vis'

            # Si ya tengo los suficientes partidos para determinar los goles del equipo
            if len(l_puntos) == n_ult_part:

                # Quito nan de lista para que no falle la cuenta y de nan (ESTABA FALLANDO!!)
                l_sin_nan = list(filter(lambda x: not math.isnan(x), l_puntos))  # Y si es None?

                # Si no eliminé todos los elementos
                if len(l_sin_nan) >= n_ult_part_min:  # No puede ser n_ult_part porque elimina nan... entonces cada vez que eliminan, no entraria...

                    # Guardo el valor promedio de la variable en los ultimos n_part
                    df.loc[idx, f'forma_{loc_o_vis}'] = sum(l_sin_nan)
                    # print('Con valores agregados:', list(df.loc[idx]))

                # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_puntos = l_puntos[1:]

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            puntos = puntos_loc(resultado) if is_equipo_loc else puntos_vis(resultado)
            l_puntos.append(puntos)
            # print(l_puntos)

    # Calculo diferencia
    df['dif_cant_punt_ult_part'] = df['forma_loc'] - df['forma_vis']

    # Elimino variables
    df = df.drop(['forma_loc', 'forma_vis'], axis=1)
    return df

def rendimiento_equipo_segun_localia(df, n_ult_part, peso_puntos):  # funciona perfecto la normalizacion
    """
    Calcula el rendimiento de un equipo utilizando la diferencia de goles y la cantidad de puntos obtenidos en los últimos partidos.
    :param df: DataFrame que contiene los datos del equipo.
    :param n_ult_part: Número de últimos partidos a considerar para calcular la diferencia de goles y puntos.
    :param peso_puntos: Peso que se le da a la cantidad de puntos en el rendimiento final.
    :return: DataFrame con la columna adicional de rendimiento del equipo.
    """
    # Calculo diferencia de gol
    df = dif_gol_ult_part_segun_localia(df, n_ult_part)

    # Calculo diferencia de forma
    df = dif_cant_puntos_ult_part_segun_localia(df, n_ult_part)

    # Normalizar diferencia de goles y cantidad de puntos en el rango [-1, 1]
    df['dif_goles_norm'] = 2 * ((df['dif_gol_ult_part_segun_loc'] - df['dif_gol_ult_part_segun_loc'].min()) / (df['dif_gol_ult_part_segun_loc'].max() - df['dif_gol_ult_part_segun_loc'].min())) - 1
    df['dif_puntos_norm'] = 2 * ((df['dif_cant_punt_ult_part_segun_loc'] - df['dif_cant_punt_ult_part_segun_loc'].min()) / (df['dif_cant_punt_ult_part_segun_loc'].max() - df['dif_cant_punt_ult_part_segun_loc'].min())) - 1

    # Calcular variable de rendimiento
    df['dif_rendimiento_ult_part_segun_loc'] = (1-peso_puntos) * df['dif_goles_norm'] + peso_puntos * df['dif_puntos_norm']

    df = df.drop(['dif_goles_norm', 'dif_puntos_norm', 'dif_gol_ult_part_segun_loc', 'dif_cant_punt_ult_part_segun_loc'], axis=1)
    return df

def dif_gol_ult_part_segun_localia(df, n_ult_part):  # Con promedio_ult_part() funciona pero solo para goles_anotados... (verificado)
    """
    Determina la cantidad de goles anotados y recibidos en los ultimos partidos
    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :return: Dataframe con columnas goles_ult_part_loc y goles_ult_part_vis.
    """
    # Definicion de variables
    n_ult_part_min = 0.6 * n_ult_part

    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugo el equipo
        df_team_loc = df[df['equipo_loc'] == equipo]
        l_dif_goles = []
        # print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team_loc.index):  # for idx in l_idxs: NO HACER ESTO

            # Si ya tengo los suficientes partidos para determinar los goles del equipo
            if len(l_dif_goles) == n_ult_part:

                # Quito nan de lista para que no falle la cuenta y de nan (ESTABA FALLANDO!!)
                l_sin_nan = list(filter(lambda x: not math.isnan(x), l_dif_goles))  # Y si es None?

                # Si no eliminé todos los elementos
                if len(l_sin_nan) >= n_ult_part_min:  # No puede ser n_ult_part porque elimina nan... entonces cada vez que eliminan, no entraria...

                    # Guardo el valor promedio de la variable en los ultimos n_part
                    df.loc[idx, f'dif_gol_ult_part_loc_segun_loc'] = sum(l_sin_nan)
                    # print('Con valores agregados:', list(df.loc[idx]))

                # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_dif_goles = l_dif_goles[1:]

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            dif_gol = df.loc[idx, 'goles_loc'] - df.loc[idx, 'goles_vis']
            l_dif_goles.append(dif_gol)
            # print(f'Dif gol: {l_dif_goles}')



        # Obtengo los partidos que jugo el equipo
        df_team_vis = df[df['equipo_vis'] == equipo]
        l_dif_goles = []
        # print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team_vis.index):  # for idx in l_idxs: NO HACER ESTO

            # Si ya tengo los suficientes partidos para determinar los goles del equipo
            if len(l_dif_goles) == n_ult_part:

                # Quito nan de lista para que no falle la cuenta y de nan (ESTABA FALLANDO!!)
                l_sin_nan = list(filter(lambda x: not math.isnan(x), l_dif_goles))  # Y si es None?

                # Si no eliminé todos los elementos
                if len(l_sin_nan) >= n_ult_part_min:  # No puede ser n_ult_part porque elimina nan... entonces cada vez que eliminan, no entraria...

                    # Guardo el valor promedio de la variable en los ultimos n_part
                    df.loc[idx, f'dif_gol_ult_part_vis_segun_loc'] = sum(l_sin_nan)
                    # print('Con valores agregados:', list(df.loc[idx]))

                # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_dif_goles = l_dif_goles[1:]

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            dif_gol = df.loc[idx, 'goles_vis'] - df.loc[idx, 'goles_loc']
            l_dif_goles.append(dif_gol)
            # print(f'Dif gol: {l_dif_goles}')

    # Calculo diferencia entre local y visitante
    df['dif_gol_ult_part_segun_loc'] = df['dif_gol_ult_part_loc_segun_loc'] - df['dif_gol_ult_part_vis_segun_loc']

    # Elimino variables
    df = df.drop(columns=['dif_gol_ult_part_loc_segun_loc', 'dif_gol_ult_part_vis_segun_loc'], axis=1)
    return df

def dif_cant_puntos_ult_part_segun_localia(df, n_ult_part):  # Mejorar la hice asi no mas
    # Requiere de dataframe ordenado por fecha decreciente
    # Tengo que agregar forma de cada equipo y puntaje antes de cada partido.

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)  # Mas antiguo a mas reciente

    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()
    puntos_loc = lambda res: 3 if res == "Local" else (1 if res == "Empate" else 0)
    puntos_vis = lambda res: 3 if res == "Visitante" else (1 if res == "Empate" else 0)
    n_ult_part_min = 0.6 * n_ult_part

    # Por equipo
    for equipo in l_equipos:

        # Obtengo los partidos que jugó el equipo
        df_team_loc = df[df['equipo_loc'] == equipo]
        l_puntos = []
        # print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team_loc.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            equipo_ganador = df_team_loc.loc[idx, 'equipo_ganador']  # Local, Empate o Visitante

            # Si ya tengo los suficientes partidos para determinar los puntos del equipo
            if len(l_puntos) == n_ult_part:

                # Quito nan de lista para que no falle la cuenta y de nan (ESTABA FALLANDO!!)
                l_sin_nan = list(filter(lambda x: not math.isnan(x), l_puntos))  # Y si es None?

                # Si no eliminé todos los elementos
                if len(l_sin_nan) >= n_ult_part_min:  # No puede ser n_ult_part porque elimina nan... entonces cada vez que eliminan, no entraria...

                    # Guardo el valor promedio de la variable en los ultimos n_part
                    df.loc[idx, 'forma_loc_segun_loc'] = sum(l_sin_nan)
                    # print('Con valores agregados:', list(df.loc[idx]))

                # Elimino puntos del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_puntos = l_puntos[1:]

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            puntos = puntos_loc(equipo_ganador)
            l_puntos.append(puntos)
            # print(l_puntos)



        # Obtengo los partidos que jugó el equipo
        df_team_vis = df[df['equipo_vis'] == equipo]

        l_puntos = []
        # print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team_vis.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            equipo_ganador = df_team_vis.loc[idx, 'equipo_ganador']  # Local, Empate o Visitante

            # Si ya tengo los suficientes partidos para determinar los puntos del equipo
            if len(l_puntos) == n_ult_part:

                # Quito nan de lista para que no falle la cuenta y de nan (ESTABA FALLANDO!!)
                l_sin_nan = list(filter(lambda x: not math.isnan(x), l_puntos))  # Y si es None?

                # Si no eliminé todos los elementos
                if len(l_sin_nan) >= n_ult_part_min:  # No puede ser n_ult_part porque elimina nan... entonces cada vez que eliminan, no entraria...

                    # Guardo el valor promedio de la variable en los ultimos n_part
                    df.loc[idx, 'forma_vis_segun_loc'] = sum(l_sin_nan)
                    # print('Con valores agregados:', list(df.loc[idx]))

                # Elimino puntos del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_puntos = l_puntos[1:]

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            puntos = puntos_vis(equipo_ganador)
            l_puntos.append(puntos)
            # print(l_puntos)

    # Calculo diferencia
    df['dif_cant_punt_ult_part_segun_loc'] = df['forma_loc_segun_loc'] - df['forma_vis_segun_loc']

    # Elimino variables
    df = df.drop(['forma_loc_segun_loc', 'forma_vis_segun_loc'], axis=1)
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

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_integrated.xlsx')
    print(df.head())

    start = time.time()

    # Construyo variable respuesta: "equipo_ganador"
    df = determinar_equipo_ganador(df)

    # Construyo variables historicas
    df = historial_entre_si(df, n_ult_part=N_ULT_PART, segun_loc=True)
    df = historial_entre_si(df, n_ult_part=N_ULT_PART, segun_loc=False)

    df = rendimiento_equipo(df, n_ult_part=N_ULT_PART, peso_puntos=0.6)
    df = rendimiento_equipo_segun_localia(df, n_ult_part=N_ULT_PART, peso_puntos=0.6)

    # l_estad_part = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp',
    #                 'offsides', 'ataques', 'ataques_pelig']
    # df = promedio_ult_partidos(df, n_ult_part=N_ULT_PART, l_var=l_estad_part)  # Estadisticas del partido
    # df = n_dias_ult_partido(df)  # Numero de dias desde ultimo partido

    # # Construyo variables de diferencias para las variables promedio de los jugadores
    # df = calculate_dif_col_jugadores(df)

    # Elimino columnas usadas para construir datos
    df = df.drop(columns=['goles_loc', 'goles_vis'], axis=1)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba_2.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()