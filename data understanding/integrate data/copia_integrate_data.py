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
        print(df_team.head())

        # Por partido que jugo el equipo
        for i in range(len(l_idxs)):  # for idx in l_idxs: NO HACER ESTO

            # Definicion de variables
            idx = l_idxs[i]

            # Obtengo forma del equipo (antes de ese partido)
            forma = calculate_forma(df_team, equipo, i)

            # Obtengo puntaje segun rendimiento del equipo (antes de ese partido)
            forma_ponderada = forma_ponderada(df_team, equipo, i)

            # Guardo nuevos valores
            # Si el equipo es el local
            if equipo == df.loc[idx, 'Equipo local']:
                df.loc[idx, 'forma_loc'] = forma
            else:
                df.loc[idx, 'forma_vis'] = forma

    return df

def calculate_forma(df_team, equipo, i):
    # Definicion de variables
    forma = 0
    N_PARTIDOS = 5

    # Selecciono ultimos <N_PARTIDOS> resultados del equipo
    l_resultados = list(df_team.iloc[i + 1:i + 1 + N_PARTIDOS, 3])  # Uso 3 en vez de "Resultado" por que deberia reiniciar el indice de df_team antes..
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


def forma_ponderada(df, equipo, i):
    # El datafrmae es por equipo

    # Definicion de variables
    N_PART = 10
    puntaje = 0

    # Filtro dataframe segun localia
    # Si el equipo juega de local
    if equipo == df.loc[i, 'Equipo local']:
        df_aux = df[df['Equipo local'] == equipo]
    # Si el equipo juega de visitante
    else:
        df_aux = df[df['Equipo Visitante'] == equipo]

    # Selecciono ultimos <N_PARTIDOS> resultados del equipo
    # FALTA IMPLEMENTAR LA DIFICULTAD DEL RIVAL...
    l_resultados = list(df_aux.iloc[i + 1:i + 1 + N_PART,3])  # Uso 3 en vez de "Resultado" por que deberia reiniciar el indice de df_team antes..
    print("Ultimos resultados", l_resultados)

    # Por resultado
    for resultado in l_resultados:

        # Si el equipo ganó
        if resultado == equipo:
            puntaje += 10

        # Si el equipo empató
        elif resultado == "Empate":
            puntaje += 5

        else:
            puntaje += 1

    return puntaje


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