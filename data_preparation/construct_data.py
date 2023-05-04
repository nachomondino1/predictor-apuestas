import pandas as pd

def equipo_ganador(df):
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
        df.loc[i, 'equipo_ganador'] = 'Local' if ng1 > ng2 else 'Empate' if ng1==ng2 else "Visitante"  # 1 if ng1 > ng2 else 0 if ng1==ng2 else -1
    return df

def diferencia_col(df, col1, col2, nombre_nueva_col):  # Tengo que ver como implementarla. Algo no funciona
    """
    Para calcular una nueva columna cuyos valores son la diferencia entre otras dos columnas numericas.
    :param df:
    :param col1:
    :param col2:
    :param nombre_nueva_col:
    :return:
    """
    # Por registro (partido)
    for i in range(len(df)):

        # Obtengo diferencia entre valores numericos
        diferencia = df.loc[i, col1] - df.loc[i, col2]

        # Guardo diferencia en nueva columna
        df.loc[i, nombre_nueva_col] = diferencia
    return df

def promedio_ult_partidos(df, n_ult_part, variable):  # Esta hecha para promediar la variable en los ultimos partidos, no para sumar..
    """
    Determina la cantidad de goles anotados y recibidos en los ultimos partidos
    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :param variable: String. Nombre de variable a promediar
    :return: Dataframe con columnas goles_ult_part_loc y goles_ult_part_vis.
    """
    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l = []
        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo
            loc_o_vis = 'loc' if is_equipo_loc else 'vis'
            nombre_variable = f'{variable}_{loc_o_vis}'  # if variable != 'equipo_ganador' else variable
            nombre_nueva_col = f'{variable}_ult_part_{loc_o_vis}' # if variable != 'equipo_ganador' else f'forma_{loc_o_vis}'

            # Si ya tengo los suficientes partidos para determinar la variable
            if len(l) == n_ult_part:

                # Guardo el valor promedio de la variable en los ultimos n_part
                df.loc[idx, nombre_nueva_col] = sum(l) / n_ult_part
                print('Con valores agregados:', list(df.loc[idx]))

                # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l = l[1:]

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            l.append(df.loc[idx, nombre_variable] if is_equipo_loc else df.loc[idx, nombre_variable])
            print(f'{variable} ultimos partidos: {l}')

    # Calculo diferencia entre variables promedio para el equipo local y para el equipo vis
    df = diferencia_col(df, col1=f'{variable}_ult_part_loc', col2=f'{variable}_ult_part_vis', nombre_nueva_col=f'dif_{variable}_segun_ult_part')

    # Elimino columnas utilizadas para calcular tanto el promedio como la diferencia
    df = df.drop(columns=[f'{variable}_loc', f'{variable}_vis', f'{variable}_ult_part_loc', f'{variable}_ult_part_vis'])
    return df

def promedio_dif_gol_ult_part(df, n_ult_part):  # Con promedio_ult_part() funciona pero solo para goles_anotados... (verificado)
    """
    Determina la cantidad de goles anotados y recibidos en los ultimos partidos
    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :return: Dataframe con columnas goles_ult_part_loc y goles_ult_part_vis.
    """
    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_dif_goles = []
        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo
            loc_o_vis = 'loc' if is_equipo_loc else 'vis'
            nombre_nueva_col = f'dig_gol_ult_part_{loc_o_vis}'

            # Si ya tengo los suficientes partidos para determinar los goles del equipo
            if len(l_dif_goles) == n_ult_part:

                df.loc[idx, nombre_nueva_col] = sum(l_dif_goles)

                # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_dif_goles = l_dif_goles[1:]
                print('con valores agregados:', list(df.loc[idx]))

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            dif_gol = df.loc[idx, 'goles_loc'] - df.loc[idx, 'goles_vis'] if is_equipo_loc else df.loc[idx, 'goles_vis'] - df.loc[idx, 'goles_loc']
            l_dif_goles.append(dif_gol)
            print(f'Dif gol: {l_dif_goles}')

    # Elimino variables
    df = df.drop(columns=['goles_loc', 'goles_vis'])
    return df

def historial_entre_si(df, n_ult_part, n_part_hist_min = 3):
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
    l_equipos = df['equipo_loc'].unique()  # e.g. ['Aldosivi', 'Boca', 'Belgrano', ...]

    # Por equipo1
    for i in range(len(l_equipos)):

        # Por equipo2
        for j in range(i+1, len(l_equipos)):

            # Definicion variables
            eq1, eq2 = l_equipos[i], l_equipos[j]
            print(f' Historial entre {eq1} y {eq2} '.center(120, '+'))

            # Selecciono unicamente los partidos entre equipo1 y equipo2
            df_historial = df[((df['equipo_loc'] == eq1) & (df['equipo_vis'] == eq2)) | (df['equipo_loc'] == eq2) &
                              (df['equipo_vis'] == eq1)]
            print(f'Cantidad de partidos: {df_historial.shape[0]}')

            # Por PARTIDO entre si (a c/u intentare asignarle un historial)
            for h in range(len(df_historial)):  # h toma [0, 1, 2, 3, 4, 5, 6, 7]
                print(f"h={h}")

                # Definicion de variables
                l_historiales = []  # Reinicio historial
                l_idxs = df_historial.index  # Lista de indices de df_historial
                equipo_loc, equipo_vis = df.loc[l_idxs[h], 'equipo_loc'], df.loc[l_idxs[h], 'equipo_vis']
                print(f'Historial para el partido Nº{h+1} donde equipo local {equipo_loc} y equipo vis {equipo_vis}')

                # Por partido anterior al partido h (a tener en cuenta en historial)  # no hago que tome desde los 4 ult partidos del h anterior porque cambia el equipo local entre h y por ende el historial
                for k in range(h+1, h+1+n_ult_part):

                    # Si aun hay <n_ult_part> anteriores a h
                    if k < len(df_historial):
                        print(f"k={k}")

                        # Obtengo el equipo ganador y el equipo local del partido k
                        equipo_ganador = df.loc[l_idxs[k], 'equipo_ganador']  # {'Local', 'Empate', 'Visitante}
                        equipo_local = df.loc[l_idxs[k], 'equipo_loc']

                        # Si gano el local
                        if equipo_ganador == 'Local':

                            # Si el local del partido k es el equipo local en el partido h
                            if equipo_loc == equipo_local:
                                historial = +1
                            else:
                                historial = -1

                        # Si empataron
                        elif equipo_ganador == 'Empate':
                            historial = 0

                        # Si gano el visitante
                        else:  # equipo_ganador_sig_part == 'Visitante':

                            # Si el local del partido k es el equipo local en el partido h
                            if equipo_loc == equipo_local:
                                historial = -1
                            else:
                                historial = +1

                        l_historiales.append(historial)
                        print(l_historiales)

                    # Si ya no hay <n_ult_part> anteriores a h
                    else:
                        break

                # Guardo historial del partido h
                if len(l_historiales) >= n_part_hist_min:
                    df.loc[l_idxs[h], 'historial_entre_si'] = sum(l_historiales)
                    print(l_historiales)
                    print(f"Historial cargado: {sum(l_historiales)}")

                # Si ya no hay al menos <n_part_hist_min> para determinar el historial
                else:
                    break

                # Elimino el proximo partido h al cual definir el historial
                l_historiales.pop(0)
    return df

def numero_lesionados(df):
    """
    Obtener lista de lesionados y contarlos
    :param df:
    :return:
    """
    # Definicion de variables
    l_var = ['l_jug_lesionados_loc', 'l_jug_lesionados_vis']

    # Por fila (partido)
    for idx in df.index:

        # Por variable de lesionados (local y vis)
        for variable in l_var:

            # Convierto string a lista con eval()
            l_lesionados = eval(df.loc[idx, variable])

            # Cuento cantidad de lesionados y lo guardo (salvo cuando es [] pues es posible que no se tenga datos y no que no haya lesionados)
            df.loc[idx, variable] = len(l_lesionados) if len(l_lesionados) != 0 else None
    return df

def n_dias_ult_partido(df):
    pass

def forma_reciente(df, n_part):  # Es igual a variable_ult_part no mas que tengo que ver el nombre de la variable...
    # Requiere de dataframe ordenado por fecha decreciente
    # Tengo que agregar forma de cada equipo y puntaje antes de cada partido.

    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()
    puntos_loc = lambda res: 3 if res == "Local" else (1 if res == "Empate" else 0)
    puntos_vis = lambda res: 3 if res == "Visitante" else (1 if res == "Empate" else 0)

    # Por equipo
    for equipo in l_equipos:

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_puntos = []
        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugo el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            resultado = df_team.loc[idx, 'equipo_ganador']
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo

            # Si ya tengo los suficientes partidos para determinar la forma del equipo
            if len(l_puntos) == n_part:

                # Si el equipo es local
                if is_equipo_loc:
                    # Guardo forma del equipo local
                    df.loc[idx, 'forma_loc'] = sum(l_puntos)
                # Si el equipo es visitante
                else:
                    # Guardo forma del equipo visitante
                    df.loc[idx, 'forma_vis'] = sum(l_puntos)

                # Elimino puntos del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_puntos = l_puntos[1:]
                print(list(df.loc[idx]))

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            puntos = puntos_loc(resultado) if is_equipo_loc else puntos_vis(resultado)
            l_puntos.append(puntos)
            print(l_puntos)
    return df

def main():
    # Definicion de variables
    N_ULT_PART = 5
    # l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
    l_variables_a_prom = ['posesion', 'remates']

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Desktop/df_formated.xlsx', index_col=0)

    # Construct data
    df = equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis
    # df = numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados
    df = historial_entre_si(df, n_ult_part=N_ULT_PART)

    # df = promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos
    # df = diferencia_col(df, col1='dig_gol_ult_part_loc', col2='dig_gol_ult_part_vis', nombre_nueva_col='dif_gol')  # Calculo la dif de gol de loc - dif de gol de vis
    # df = forma_reciente(df, n_part=N_ULT_PART)

    # for var in l_variables_a_prom:
    #     df = promedio_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)

    print(df.head())
    df.to_excel('/Users/nachomondino/Desktop/df_constructed.xlsx')


main()



'''
def derive_dif_gol_last_matches(df, n_part):  # Ver si dejar o no. Uso esta o goles_anotados_y_recibidos()?
    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()
    dif_gol = lambda df, is_equipo_loc: df.loc[idx, 'goles_loc'] - df.loc[idx, 'goles_vis'] if is_equipo_loc else df.loc[idx, 'goles_vis'] - df.loc[idx, 'goles_loc']

    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Por equipo
    for equipo in l_equipos:

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_dif_gol = []
        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo

            # Si ya tengo los suficientes partidos para determinar la forma del equipo
            if len(l_dif_gol) == n_part:

                # Si el equipo es local
                if is_equipo_loc:
                    # Guardo goles del equipo local
                    df.loc[idx, 'dif_gol_loc'] = sum(l_dif_gol)
                # Si el equipo es visitante
                else:
                    # Guardo goles del equipo visitante
                    df.loc[idx, 'dif_gol_vis'] = sum(l_dif_gol)

                # Elimino puntos del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_dif_gol = l_dif_gol[1:]
                print(list(df.loc[idx]))

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            dif_gol2 = dif_gol(df, is_equipo_loc)
            l_dif_gol.append(dif_gol2)
            print(l_dif_gol)
    return df

def derive_forma_ponderada(df, n_part):  # Lo uso? o uso forma sola?
    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()
    puntos_loc = lambda res: 10 if res == "Local" else (5 if res == "Empate" else 1)
    puntos_vis = lambda res: 10 if res == "Visitante" else (5 if res == "Empate" else 1)

    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Calculo las columnas forma_loc y forma_vis
    df = derive_forma(df, n_part)

    # Por equipo
    for equipo in l_equipos:

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_puntos_pond = []
        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugo el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            resultado = df_team.loc[idx, 'equipo_ganador']
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo

            # Si ya tengo los puntos ponderados para <n_part> ultimos partidos del equipo
            if len(l_puntos_pond) == n_part:

                # Si el equipo es local
                if is_equipo_loc:
                    # Guardo forma del equipo local
                    df.loc[idx, 'puntaje_loc'] = sum(l_puntos_pond)
                # Si el equipo es visitante
                else:
                    # Guardo forma del equipo visitante
                    df.loc[idx, 'puntaje_vis'] = sum(l_puntos_pond)

                # Elimino puntos ponderados del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_puntos_pond = l_puntos_pond[1:]
                print(list(df.loc[idx]))

            # Obtengo puntos (segun si gano, empato o perdio) y la forma del rival (como llegaba el rival antes del partido)
            forma_rival = df.loc[idx, 'forma_vis'] if is_equipo_loc else df.loc[idx, 'forma_loc']  # Suma de puntos de los ult <n_part> del rival
            puntos = puntos_loc(resultado) if is_equipo_loc else puntos_vis(resultado)

            # Con puntos y forma del rival, determino la relevancia del resultado obtenido ante el rival
            puntos_pond = puntos * (1+forma_rival/(n_part*3))

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            l_puntos_pond.append(puntos_pond)
            print(l_puntos_pond)

    # Elimino columnas forma_loc y forma_vis (las hice para poder calcular los puntos ponderados)
    df = df.drop(['forma_loc'], axis=1)
    df = df.drop(['forma_vis'], axis=1)

    # Agrego columna "diferencia entre puntos ponderados" de los equipos que se enfrentan
    df['dif_forma_pond'] = df['puntaje_loc'] - df['puntaje_vis']

    # Elimino columnas puntaje_loc y puntaje_vis (las hice para poder calcular la diferencia entre los puntos ponderados de los equipos)
    df = df.drop(['puntaje_loc'], axis=1)
    df = df.drop(['puntaje_vis'], axis=1)
    return df
    
def goles_anotados_y_recibidos(df, n_ult_part):  # Con promedio_ult_part() funciona pero solo para goles_anotados... (verificado)
    """
    Determina la cantidad de goles anotados y recibidos en los ultimos partidos
    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :return: Dataframe con columnas goles_ult_part_loc y goles_ult_part_vis.
    """
    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()

    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Por equipo
    for equipo in l_equipos:

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_goles_anot, l_goles_recib = [], []
        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Por partido que jugó el equipo
        for idx in list(df_team.index):  # for idx in l_idxs: NO HACER ESTO

            print(list(df.loc[idx]))

            # Definicion de variables
            is_equipo_loc = df_team.loc[idx, 'equipo_loc'] == equipo

            # Si ya tengo los suficientes partidos para determinar los goles del equipo
            if len(l_goles_anot) == n_ult_part:

                # Si el equipo es local
                if is_equipo_loc:
                    # Guardo goles del equipo local
                    df.loc[idx, 'goles_anot_ult_part_loc'] = sum(l_goles_anot)
                    df.loc[idx, 'goles_recib_ult_part_loc'] = sum(l_goles_recib)

                # Si el equipo es visitante
                else:
                    # Guardo goles del equipo visitante
                    df.loc[idx, 'goles_anot_ult_part_vis'] = sum(l_goles_anot)
                    df.loc[idx, 'goles_recib_ult_part_vis'] = sum(l_goles_recib)

                # Elimino goles del ultimo partido (para tener siempre los ultimos <n_part> partidos)
                l_goles_anot, l_goles_recib = l_goles_anot[1:], l_goles_recib[1:]
                print('con valores agregados:', list(df.loc[idx]))

            # Si el equipo es local
            if is_equipo_loc:
                n_goles_anot, n_goles_recib = df.loc[idx, 'goles_loc'], df.loc[idx, 'goles_vis']
            # Si el equipo es visitante
            else:
                n_goles_anot, n_goles_recib = df.loc[idx, 'goles_vis'], df.loc[idx, 'goles_loc']

            # Agrego puntos del partido (para tener siempre los ultimos <n_part> partidos)
            l_goles_anot.append(n_goles_anot)
            l_goles_recib.append(n_goles_recib)
            print(f'Goles anotados: {l_goles_anot}')
            print(f'Goles recibidos: {l_goles_recib}')
    return df
'''

'''
def historial_entre_si(df, n_part, n_part_hist_min = 3):  # antes de cambios
    """
    Crea columna 'equipo_ganador' donde se especifica el resultado de cada partido
    :param df: Dataframe. Unidad de analisis: partido. Columnas: al menos fecha, equipo_loc, equipo_vis y equipo_ganador
    :param n_part: Integer. Numero de partidos a tener en cuenta para determinar el historial entre dos equipos.
    :param n_part_hist_min: Integer. Numero minimo de partidos a tener en cuenta para determinar el historial entre dos
    equipos.
    :return: Dataframe pasado por parametro con nueva columna, 'historial_entre_si', que permite a cual de los dos
    equipos de un partido lo favorece mas el historial entre ellos.
    """
    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Mas reciente a mas antiguo

    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()  # e.g. ['Aldosivi', 'Boca', 'Belgrano', ...]

    # Por equipo1
    for i in range(len(l_equipos)):

        # Por equipo2
        for j in range(i+1, len(l_equipos)):

            # Definicion variables
            eq1, eq2 = l_equipos[i], l_equipos[j]
            print(f' Historial entre {eq1} y {eq2} '.center(120, '+'))

            # Selecciono unicamente los partidos entre equipo1 y equipo2
            df_historial = df[((df['equipo_loc'] == eq1) & (df['equipo_vis'] == eq2)) | (df['equipo_loc'] == eq2) &
                              (df['equipo_vis'] == eq1)]
            print(f'Cantidad de partidos: {df_historial.shape[0]}')

            # Por PARTIDO entre si
            for h in range(len(df_historial)):  # h toma [0, 1, 2, 3, 4, 5, 6, 7]

                # Definicion de variables
                historial, n_part_hist = 0, 0 # Reinicio el historial por cada PARTIDO
                l_idxs = df_historial.index  # Lista de indices de df_historial
                equipo_loc, equipo_vis = df.loc[l_idxs[h], 'equipo_loc'], df.loc[l_idxs[h], 'equipo_vis']
                print('Equipo Local: {} \t Equipo Visitante: {}'.format(equipo_loc, equipo_vis))

                # Por partido a tener en cuenta en el historial del PARTIDO
                for k in range(h+1, h+1+n_part):  # k toma 1 a 6, 2 a 7, 3 a 8, etc.

                    # Si hay partido anterior (por ej, puede ser que quiera 5 partidos antes y solo haya 4)
                    if k < len(l_idxs):

                        # Obtengo el resultado y lo guardo en el historial
                        equipo_ganador = df.loc[l_idxs[k], 'equipo_ganador']  # {'Local', 'Empate', 'Visitante}
                        historial += 1 if equipo_ganador == "Local" else -1 if equipo_ganador == "Visitante" else 0
                        n_part_hist += 1
                        print(k, equipo_ganador)

                    # Si no hay partido anterior
                    else:
                        # Dejo de calcular el historial entre los equipos
                        break

                # Si el historial se basa en al menos <n_part_hist_min>
                if n_part_hist >= n_part_hist_min:
                    # Guardo el historial (segun rdos entre si de los ult <n_part> entre si) para el PARTIDO
                    df.loc[l_idxs[h], 'historial_entre_si'] = historial

                # Si no hay partidos (No hay historial)
                else:
                    df.loc[l_idxs[h], 'historial_entre_si'] = None
                    break  # Cambio de equipos y dejo de calcular el historial entre estos equipos
    return df
'''