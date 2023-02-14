import pandas as pd

def calculate_dif_forma(df, n_part):
    # Definicion de variables
    puntos_loc = lambda res: 10 if res == "Local" else (5 if res == "Empate" else 1)
    puntos_vis = lambda res: 10 if res == "Visitante" else (5 if res == "Empate" else 1)

    # Por partido (registro o  fila) (e.g. Boca - Talleres)
    for i in range(len(df)):

        # Definicion de variables
        l_teams = df.loc[i, ['equipo_loc', 'equipo_vis']]  # Lista de equipos que juegan el partido
        l_formas = []

        # Por equipo que juega el partido (e.g. Boca)
        for j in range(len(l_teams)):

            # Definicion de variables
            forma = 0
            team = l_teams[j]

            # Selecciono solo los partidos en que jugo el equipo
            df_team = df[(df['equipo_loc'] == team) | (df['equipo_vis'] == team)]

            # Si ya tengo mas de n_part partidos viejos del equipo
            if i >= list(df_team.index)[n_part]:

                print('Partido:', list(df.loc[i]))
                print(df_team.head())

                # Definicion de variables
                pos_ini = list(df_team.index).index(i)
                l_idxs = list(df_team.index)[pos_ini-n_part: pos_ini]

                # Por partido de los ultimos N_PART del equipo (e.g. Por partido de los ultimos 5 part de Boca antes de jugar con Talleres)
                for idx in l_idxs:  # Sin incluir el partido i pues es la forma con que llegaba cada equipo al partido

                    # Definicion de variables
                    equipo_loc = df_team.loc[idx, 'equipo_loc']
                    resultado = df_team.loc[idx, 'equipo_ganador']

                    # si el equipo es local
                    if equipo_loc == team:
                        forma += puntos_loc(resultado)

                    # Si el equipo es visitante
                    else:
                        forma += puntos_vis(resultado)

                print("{}. Forma: {} ptos".format(team, forma))
                l_formas.append(forma)

            try:
                df.loc[i, "dif_forma"] = l_formas[0] - l_formas[1]
            except:
                pass
    return df

def historial_entre_si(df):
    pass

def main():
    # Levanto el dataframe formateado
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data understanding/format_data/df_formated.xlsx', index_col=0)

    # Ordeno por fecha descendiente
    df_by_fecha = df.sort_values(by='fecha', ascending=True, ignore_index=True)
    # print(df_by_fecha)
    df_by_fecha.to_excel('./df_by_fecha.xlsx')

    # Derive new data
    df = calculate_dif_forma(df=df_by_fecha, n_part=5)
    print(df.head())

    df.to_excel('./df_derived_data.xlsx')

main()


'''
def derive_new_columns(df):
    # Requiere de dataframe ordenado por fecha decreciente
    # Tengo que agregar forma de cada equipo y puntaje antes de cada partido.

    # Definicion de variables
    l_equipos = df['equipo_loc'].unique()

    # Por equipo
    for equipo in l_equipos:

        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_idxs = list(df_team.index)

        # Por partido que jugo el equipo
        for i in range(len(l_idxs)):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            df_aux = df.loc[l_idxs[i:len(l_idxs)]]  # quito partidos que ya se jugaron

            # Obtengo forma del equipo (antes de ese partido)
            forma = calculate_forma(df_aux, equipo, 10)

            # Guardo nuevos valores
            # Si el equipo es el local
            if equipo == df.loc[l_idxs[i], 'equipo_loc']:
                df.loc[l_idxs[i], 'forma_loc'] = forma
            else:
                df.loc[l_idxs[i], 'forma_vis'] = forma

    # ARREGLAR ESTE DESASTRE. DOS VECES LA MISMA ESTRUCTURA. VER COMO PUEDO JUNTARLAS
    for equipo in l_equipos:

        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        l_idxs = list(df_team.index)

        # Por partido que jugo el equipo
        for i in range(len(l_idxs)):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            df_aux = df.loc[l_idxs[i:len(l_idxs)]]  # quito partidos que ya se jugaron

            # Obtengo forma del equipo (antes de ese partido)
            forma_pond = forma_ponderada(df_aux, equipo, 10)

            # Guardo nuevos valores
            # Si el equipo es el local
            if equipo == df.loc[l_idxs[i], 'equipo_loc']:
                df.loc[l_idxs[i], 'dif_loc'] = forma_pond
            else:
                df.loc[l_idxs[i], 'dif_vis'] = forma_pond

    return df

def calculate_forma(df, equipo, n_part):
    # Definicion de variables
    forma = 0

    # Selecciono ultimos <N_PARTIDOS> resultados del equipo
    l_resultados = list(df.iloc[1:1+n_part, 3])  # Uso 3 en vez de "equipo_ganador" por que deberia reiniciar el indice de df_team antes..
    print("Ultimos resultados", l_resultados)

    # Por resultado
    for resultado in l_resultados:

        # Si el equipo ganó
        if resultado == equipo:
            forma += 3

        # Si el equipo empató
        elif resultado == "Empate":
            forma += 1

    print("Forma:", forma)
    return forma

def forma_ponderada(df, equipo, n_part):
    # El dataframe es por equipo

    # Definicion de variables
    puntaje = 0
    # dif_rival = lambda equipo, df, i: df.loc[i, 'forma_vis'] if equipo == df.loc[i, 'equipo_loc'] else df.loc[i, 'forma_loc']

    # Si el equipo es local
    if equipo == df.iloc[0, 1]:  # Uso 1 en vez de "equipo_loc" por que deberia reiniciar el indice de df_team antes..
        df = df[df['equipo_loc'] == equipo]
    else:
        df = df[df['equipo_vis'] == equipo]

    # Selecciono ultimos <N_PARTIDOS> resultados del equipo
    df_ult_partidos = df.iloc[1:1+n_part]
    l_idxs = list(df_ult_partidos.index)
    print("Ultimos resultado", df_ult_partidos)

    # Por partido
    for i in range(len(l_idxs)):

        # Defino variables (dif rival y resultado)
        # Si el equipo es local
        if equipo == df.loc[l_idxs[i], 'equipo_loc']:
            forma_rival = df_ult_partidos.loc[l_idxs[i], 'forma_vis']
        else:
            forma_rival = df_ult_partidos.loc[l_idxs[i], 'forma_loc']

        resultado = df_ult_partidos.loc[l_idxs[i], 'equipo_ganador']

        # Calculo puntaje del partido segun dif rival y resultado
        # Si el equipo ganó
        if resultado == equipo:
            puntaje += 10 * (1 + forma_rival /(n_part * 3))

        # Si el equipo empató
        elif resultado == "Empate":
            puntaje += 5 * (1 + forma_rival /(n_part * 3))

        else:
            puntaje += 1 * (1 + forma_rival /(n_part * 3))

    return puntaje
'''