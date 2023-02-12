import pandas as pd

def derive_new_columns(df):
    # Requiere de dataframe ordenado por fecha decreciente
    # Tengo que agregar forma de cada equipo y puntaje antes de cada partido.

    # Definicion de variables
    l_equipos = df['Equipo local'].unique()

    # Por equipo
    for equipo in l_equipos:

        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['Equipo local'] == equipo) | (df['Equipo Visitante'] == equipo)]
        l_idxs = list(df_team.index)

        # Por partido que jugo el equipo
        for i in range(len(l_idxs)):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            df_aux = df.loc[l_idxs[i:len(l_idxs)]]  # quito partidos que ya se jugaron

            # Obtengo forma del equipo (antes de ese partido)
            forma = calculate_forma(df_aux, equipo, 10)

            # Guardo nuevos valores
            # Si el equipo es el local
            if equipo == df.loc[l_idxs[i], 'Equipo local']:
                df.loc[l_idxs[i], 'forma_loc'] = forma
            else:
                df.loc[l_idxs[i], 'forma_vis'] = forma


    for equipo in l_equipos:

        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['Equipo local'] == equipo) | (df['Equipo Visitante'] == equipo)]
        l_idxs = list(df_team.index)

        # Por partido que jugo el equipo
        for i in range(len(l_idxs)):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            df_aux = df.loc[l_idxs[i:len(l_idxs)]]  # quito partidos que ya se jugaron

            # Obtengo forma del equipo (antes de ese partido)
            forma_pond = forma_ponderada(df_aux, equipo, 10)

            # Guardo nuevos valores
            # Si el equipo es el local
            if equipo == df.loc[l_idxs[i], 'Equipo local']:
                df.loc[l_idxs[i], 'dif_loc'] = forma_pond
            else:
                df.loc[l_idxs[i], 'dif_vis'] = forma_pond

    return df

def calculate_forma(df, equipo, n_part):
    # Definicion de variables
    forma = 0

    # Selecciono los partidos que jugo el equipo
    # df_team = df[(df['Equipo local'] == equipo) | (df['Equipo Visitante'] == equipo)]

    # Selecciono ultimos <N_PARTIDOS> resultados del equipo
    l_resultados = list(df.iloc[1:1+n_part, 3])  # Uso 3 en vez de "Resultado" por que deberia reiniciar el indice de df_team antes..
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
    # dif_rival = lambda equipo, df, i: df.loc[i, 'forma_vis'] if equipo == df.loc[i, 'Equipo local'] else df.loc[i, 'forma_loc']

    # Si el equipo es local
    if equipo == df.iloc[0, 1]:  # Uso 1 en vez de "Equipo local" por que deberia reiniciar el indice de df_team antes..
        df = df[df['Equipo local'] == equipo]
    else:
        df = df[df['Equipo Visitante'] == equipo]

    # Selecciono ultimos <N_PARTIDOS> resultados del equipo
    df_ult_partidos = df.iloc[1:1+n_part]  # Uso 3 en vez de "Resultado" por que deberia reiniciar el indice de df_team antes..
    l_idxs = list(df_ult_partidos.index)
    print("Ultimos resultado", df_ult_partidos)

    # Por partido
    for i in range(len(l_idxs)):

        # Defino variables (dif rival y resultado)
        # Si el equipo es local
        if equipo == df.loc[l_idxs[i], 'Equipo local']:
            forma_rival = df_ult_partidos.loc[l_idxs[i], 'forma_vis']
        else:
            forma_rival = df_ult_partidos.loc[l_idxs[i], 'forma_loc']

        resultado = df_ult_partidos.loc[l_idxs[i], 'Resultado']

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


def historial_entre_si(df):

    # Por partido
    for i in range(len(df)):

        print(" Equipo: {} ".format(equipo).center(120, "#"))

        # Obtengo los partidos que jugo el equipo
        df_team = df[(df['Equipo local'] == equipo) | (df['Equipo Visitante'] == equipo)]
        l_idxs = list(df_team.index)

        # Por partido que jugo el equipo
        for i in range(len(l_idxs)):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            df_aux = df.loc[l_idxs[i:len(l_idxs)]]  # quito partidos que ya se jugaron

            # Obtengo forma del equipo (antes de ese partido)
            forma_pond = forma_ponderada(df_aux, equipo, 10)

            # Guardo nuevos valores
            # Si el equipo es el local
            if equipo == df.loc[l_idxs[i], 'Equipo local']:
                df.loc[l_idxs[i], 'dif_loc'] = forma_pond
            else:
                df.loc[l_idxs[i], 'dif_vis'] = forma_pond



def main():
    # Levanto el dataframe formateado
    df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data understanding/df_formated.xlsx', index_col=0)

    # Ordeno por fecha descendiente
    df_by_fecha = df.sort_values(by='Fecha', ascending=False, ignore_index=True)
    # print(df_by_fecha)

    # Derive new data
    df = derive_new_columns(df_by_fecha)
    print(df.head())

    df.to_excel('./df_derived_data.xlsx')

main()