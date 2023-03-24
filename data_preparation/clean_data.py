def clean_teams(df):
    """
    Limpio string 'Vencedor' en el nombre de algunos equipos.
    :param df:
    :return:
    """
    # Definicion de variables
    n_reemplazos = 0

    # Por partido
    for i in range(len(df)):

        equipo_loc, equipo_vis = df.loc[i, 'equipo_loc'], df.loc[i, 'equipo_vis']

        if 'Vencedor' in equipo_loc:

            df.loc[i, 'equipo_loc'] = equipo_loc.replace('Vencedor', '')
            n_reemplazos += 1

        elif 'Vencedor' in equipo_vis:

            df.loc[i, 'equipo_vis'] = equipo_vis.replace('Vencedor', '')
            n_reemplazos += 1

    print("Se encontraron {} equipos con la palabra 'Vencedor'".format(n_reemplazos))
    return df