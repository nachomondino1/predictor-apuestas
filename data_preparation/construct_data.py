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
    df[f'dif_{variable}_segun_ult_part'] = df[f'{variable}_ult_part_loc'] - df[f'{variable}_ult_part_vis']

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
            nombre_nueva_col = f'dif_gol_ult_part_{loc_o_vis}'

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

    # Calculo diferencia entre local y visitante
    df['dif_gol'] = df['dif_gol_ult_part_loc'] - df['dif_gol_ult_part_vis']

    # Elimino variables
    df = df.drop(columns=['goles_loc', 'goles_vis', 'dif_gol_ult_part_loc', 'dif_gol_ult_part_vis'], axis=1)
    return df

def historial_entre_si_segun_localia(df, n_ult_part, n_part_hist_min = 2):  # quiero poner historial_entre_si segun localia. Es decir, para el partido River-Boca quiero poner el historial de los ultimos 5 River-Boca en el monumental (y no en los estadios)
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

    # Por fila
    for i in range(len(df)):

        # Evito partidos a los cuales ya les cargue el historial
        if i not in l_idxs_cargados:

            # Definicion variables
            equipo_loc, equipo_vis = df.loc[i, 'equipo_loc'], df.loc[i, 'equipo_vis']
            print(f' Historial {equipo_loc} - {equipo_vis} '.center(120, '+'))

            # Selecciono unicamente los partidos entre equipo1 y equipo2
            df_historial = df[((df['equipo_loc'] == equipo_loc) & (df['equipo_vis'] == equipo_vis))]

            # Por PARTIDO entre si (a c/u intentare asignarle un historial)
            for j in range(len(df_historial)):  # h toma [0, 1, 2, 3, 4, 5, 6, 7]

                # Definicion de variables
                l_historiales = []  # Reinicio historial
                l_idxs = df_historial.index  # Lista de indices de df_historial
                print(f'Historial para el partido Nº{j+1} donde equipo local {equipo_loc} y equipo vis {equipo_vis}')

                # Por partido anterior al partido j (a tener en cuenta en historial)  # no hago que tome desde los 4 ult partidos del h anterior porque cambia el equipo local entre h y por ende el historial
                for k in range(j + 1, j + 1 + n_ult_part):

                    # Si aun hay <n_ult_part> anteriores a h
                    if k < len(df_historial):
                        print(f"k={k}")

                        # Obtengo el equipo ganador y el equipo local del partido k
                        equipo_ganador = df.loc[l_idxs[k], 'equipo_ganador']  # {'Local', 'Empate', 'Visitante}
                        historial = +1 if equipo_ganador == 'Local' else 0 if equipo_ganador == 'Empate' else -1

                        l_historiales.append(historial)
                        print(l_historiales)

                    # Si ya no hay <n_ult_part> anteriores a h
                    else:
                        break

                l_idxs_cargados.append(list(l_idxs))

                # Guardo historial del partido j
                if len(l_historiales) >= n_part_hist_min:
                    df.loc[l_idxs[j], 'historial_entre_si'] = sum(l_historiales)
                    # print(l_historiales)
                    print(f"Historial cargado: {sum(l_historiales)}")

                # Si ya no hay al menos <n_part_hist_min> para determinar el historial
                else:
                    break
    return df

def forma_reciente(df, n_part):  # Es igual a promedio_ult_part no mas que tengo que ver el nombre de la variable...
    # Requiere de dataframe ordenado por fecha decreciente
    # Tengo que agregar forma de cada equipo y puntaje antes de cada partido.

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)  # Mas antiguo a mas reciente

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

    # Calculo diferencia
    df['dif_forma'] = df['forma_loc'] - df['forma_vis']

    # Elimino variables
    df = df.drop(columns=['forma_loc', 'forma_vis'], axis=1)
    return df

def n_dias_ult_partido(df):  # Peopuesta por Chat GPT
    df = df.sort_values('fecha')

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

    return df

def convert_odds_to_prob(df):  # No estoy seguro de que la vaya a usar... Para el caculo del ROI necesito las cuotas...
    ''' Converts bookkeeper odds to probabilities. '''

    # Por registro
    for i in range(len(df)):

        # Converts odds to prob
        prob_loc_with_over = 1 / df.loc[i, 'odds_loc']
        prob_emp_with_over = 1 / df.loc[i, 'odds_emp']
        prob_vis_with_over = 1 / df.loc[i, 'odds_vis']

        prob_total_with_over = prob_loc_with_over + prob_emp_with_over + prob_vis_with_over

        # Define output format and scale probs by sum over all probs
        df.loc[i, 'odds_loc'] = prob_loc_with_over / prob_total_with_over
        df.loc[:, 'odds_emp'] = prob_emp_with_over / prob_total_with_over
        df.loc[:, 'odds_vis'] = prob_vis_with_over / prob_total_with_over

    return df

def suma_overall_rating_ausentes(df):
    pass

def main():

    # Definicion de variables
    N_ULT_PART = 5
    l_variables_a_prom = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']

    # Levanto dataset
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/df_integrated.xlsx')
    print(df.head())

    # Construct data
    df = equipo_ganador(df)  # Determino columna "equipo_ganador" segun goles_loc y goles_vis

    df = historial_entre_si_segun_localia(df, n_ult_part=3)

    df = promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Determino diferencia de gol de cada uno  de los equipos en los ultimos partidos

    df = forma_reciente(df, n_part=N_ULT_PART)  # df = derive_forma_ponderada(df, n_part=N_ULT_PART)

    # Calculo diferencias para las variables promedio de los jugadores
    df['dif_rat_tit'] = df['l_jug_tit_loc_prom_rat'] - df['l_jug_tit_vis_prom_rat']
    df['dif_edad_tit'] = df['l_jug_tit_loc_prom_edad'] - df['l_jug_tit_vis_prom_edad']
    df['dif_alt_tit'] = df['l_jug_tit_loc_prom_alt'] - df['l_jug_tit_vis_prom_alt']
    df['dif_rat_sup'] = df['l_jug_sup_loc_prom_rat'] - df['l_jug_sup_vis_prom_rat']
    df['dif_edad_sup'] = df['l_jug_sup_loc_prom_edad'] - df['l_jug_sup_vis_prom_edad']
    df['dif_alt_sup'] = df['l_jug_sup_loc_prom_alt'] - df['l_jug_sup_vis_prom_alt']

    for var in l_variables_a_prom:
        df = promedio_ult_partidos(df, n_ult_part=N_ULT_PART, variable=var)

    df = n_dias_ult_partido(df)
    # df = convert_odds_to_prob(df)
    # df = numero_lesionados(df)  # Determino numero de lesionados segun cantidad de lesionados

    print(df.head())
    df.to_excel('./df_constructed.xlsx')


# main()

'''
def factor_ponderacion(df): # Me gusta la idea de ponderar las variables forma, posesion, remates, etc pues no es lo mismo hacerlo contra Boca de visitante que Local contra Olimpo. El factor tiene que ponderar, localidad, rival, su forma y que mas?
    # Cada equipo tendra un factor en cada partido. Un boca que viene de 6 partidos seguidos ganando no es lo mismo que un Boca perdiendo 3 seguidos de local.

    # Por registro (partido)
    for i in range(len(df)):

        # Obtengo factor de pond del local (analizo su rival, la forma del rival, la forma de ellos) Real Madrid - Osasuna, es cantado que el madrid lo gana, es de alta dificultad para el osasuna por mas que el madrid no venga en su mejor momento
        factor_pond_loc = 1 * prob_ganar_equipo_vis * forma_vis * forma_loc
        # A mayor factor,

        # Obtengo factor de pond del visitante
        factor_pond_vis = 1
    pass

def derive_forma_ponderada(df, n_part):  # Me gusta la idea de ponderar las variables forma, posesion, remates, etc pues no es lo mismo hacerlo contra Boca de visitante que Local contra Olimpo. El factor tiene que ponderar, localidad, rival, su forma y que mas?
    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()
    puntos_loc = lambda res: 10 if res == "Local" else (5 if res == "Empate" else 1)
    puntos_vis = lambda res: 10 if res == "Visitante" else (5 if res == "Empate" else 1)

    # Ordeno por fecha descendiente
    df = df.sort_values(by='fecha', ascending=True, ignore_index=True)

    # Calculo las columnas forma_loc y forma_vis
    df = forma_reciente(df, n_part)

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
    # df = df.drop(['forma_loc', 'forma_vis'], axis=1)

    # Agrego columna "diferencia entre puntos ponderados" de los equipos que se enfrentan
    df['dif_forma_pond'] = df['puntaje_loc'] - df['puntaje_vis']

    # Elimino columnas puntaje_loc y puntaje_vis (las hice para poder calcular la diferencia entre los puntos ponderados de los equipos)
    # df = df.drop(['puntaje_loc', 'puntaje_vis'], axis=1)
    return df

'''


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
