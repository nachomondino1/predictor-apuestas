def derive_winning_team(df):
    """
    Crea columna 'equipo_ganador' donde se especifica el resultado de cada partido.
    :param df: Dataframe. Unidad de analisis: partido. Columnas: fecha, equipo_loc, equipo_vis, arbitro, dt_loc, dt_vis,
    goles_loc, goles_vis.
    :return: Dataframe pasado por parametro con nueva columna, 'equipo_ganador', que detalla el resultado del partido.
    """
    # Por partido
    for i in range(len(df)):

        # Obtengo goles de cada equipo
        ng1, ng2 = df.loc[i, 'goles_loc'], df.loc[i, 'goles_vis']

        # Si el equipo local hizo mas goles que el equipo visitante
        if ng1 > ng2:
            # Ganó equipo local
            df.loc[i, 'equipo_ganador'] = 'Local'
        # Si el equipo local hizo la misma cantidad de goles que el equipo visitante
        elif ng1 == ng2:
            # Empataron
            df.loc[i, 'equipo_ganador'] = 'Empate'
        # Si el equipo local hizo menos goles que el equipo visitante
        else:
            # Ganó equipo vis
            df.loc[i, 'equipo_ganador'] = 'Visitante'
    return df

def derive_historial_entre_si(df, n_part, n_part_hist_min = 1):
    """
    Crea columna 'equipo_ganador' donde se especifica el resultado de cada partido
    :param df: Dataframe. Unidad de analisis: partido. Columnas: fecha, equipo_loc, equipo_vis, arbitro, dt_loc, dt_vis,
    goles_loc, goles_vis, equipo_ganador
    :param n_part: Integer. Numero de partidos a tener en cuenta para determinar el historial entre dos equipos.
    :param n_part_hist_min: Integer. Numero minimo de partidos a tener en cuenta para determinar el historial entre dos
    equipos.
    :return: Dataframe pasado por parametro con nueva columna, 'historial_entre_si', que permite a cual de los dos
    equipos de un partido lo favorece mas el historial entre ellos.
    """
    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()  # e.g. ['Aldosivi', 'Boca', 'Belgrano', ...]

    # Por equipo1
    for i in range(len(l_equipos)):

        # Por equipo2
        for j in range(i+1, len(l_equipos)):

            # Definicion variables
            equipo1, equipo2 = l_equipos[i], l_equipos[j]
            print('\nEquipo1: {} \t Equipo2: {}'.format(equipo1, equipo2))

            # Selecciono partidos entre equipo1 y equipo2
            df_historial = df[((df['equipo_loc'] == equipo1) & (df['equipo_vis'] == equipo2)) | (df['equipo_loc'] == equipo2) & (df['equipo_vis'] == equipo1)]
            print(df_historial.head())

            # Por PARTIDO entre si
            for h in range(len(df_historial)):

                # Definicion de variables
                historial = 0  # Reinicio el hisotorial por cada PARTIDO
                l_idxs = df_historial.index  # Lista de indices de df_historial
                equipo_loc, equipo_vis = df.loc[l_idxs[h], 'equipo_loc'], df.loc[l_idxs[h], 'equipo_vis']
                n_part_hist = 0
                print('Equipo Local: {} \t Equipo Visitante: {}'.format(equipo_loc, equipo_vis))

                # Por partido a tener en cuenta en el historial del PARTIDO
                for k in range(h+1, h+1+n_part):

                    # Si hay partido anterior (por ej, puede ser que quiera 5 partidos antes y solo haya 4)
                    if k < len(l_idxs):

                        resultado = df.loc[l_idxs[k], 'equipo_ganador']
                        n_part_hist += 1
                        print(k, resultado)

                        # Si gano el equipo local (local en el PARTIDO)
                        if resultado == "Local":
                            historial += 1

                        # Si gano el equipo visitante (visitante en el PARTIDO)
                        elif resultado == "Visitante":
                            historial += -1

                    # Si no hay partido anterior
                    else:
                        # Dejo de calcular el historial entre los equipos
                        break

                print(historial)
                # Si el historial se basa en al menos <n_part_hist_min>
                if n_part_hist >= n_part_hist_min:
                    # Guardo el historial (segun rdos entre si de los ult <n_part> entre si) para el PARTIDO
                    df.loc[l_idxs[h], 'historial_entre_si'] = historial
                # Si no hay partidos
                else:
                    # No hay historial
                    df.loc[l_idxs[h], 'historial_entre_si'] = None

                    # Cambio de equipos y dejo de calcular el historial entre estos equipos
                    break

    return df

def derive_dif_gol_last_matches(df, n_part):

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

def derive_forma(df, n_part):
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

def derive_forma_ponderada(df, n_part):
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

def n_dias_ult_partido(df):
    pass