# Juntar entidades jugadores, partido y atrib_jugadores
import pandas as pd
import re
import time

def search_player_data(df_part, df_jug):  # Ver como guardar los datos de prom edad solo para titulares, solo suplentes y asi... si es que quiero asi...
    """
    Segun el nombre del jugador, la fecha y el equipo(?) en la entidad partido, busco sus atributos en la entidad jugador
    :param df:
    :return:
    """
    # Defino las columnas a procesar segun la variable
    pattern = 'jug_' # pattern = variable[2:] + '_[0-9]*'
    l_col_to_preprocess = df_part.filter(regex=pattern, axis=1).columns.tolist()  #LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...
    print(f'Columnas a procesar: {l_col_to_preprocess}')
    d = {}

    # Por partido (fila) en entidad partido
    for i in range(len(df_part)):

        # Obtengo equipo y año del partido
        year_part = df_part.loc[i, 'fecha'].year   # e.g. 2023
        print(f' Partido Nº: {i} '.center(120, '#'))

        # Reinicio variables
        l_prom_edad, l_prom_alt, l_prom_rating, l_prom_valor = [], [], [], []

        # Por jugador
        for jug in l_col_to_preprocess:

            # Busco nombre del jugador en la entidad partido
            nombre_jug_part = df_part.loc[i, jug]  # e.g. "Rodriguez D. (G) (C)"
            equipo_jug_part = df_part.loc[i, 'equipo_loc'] if 'loc' in jug else df_part.loc[i, 'equipo_vis']  # DEPENDERA DE SI ES LOCAL O VIS...

            # Si el jugador no es nan
            if isinstance(nombre_jug_part, str):

                print(f' {nombre_jug_part} '.center(100, '-'))
                print(f' JUGADOR EN ENTIDAD PARTIDO: ')
                print(f'\t- Nombre: {nombre_jug_part}')
                print(f'\t- Equipo: {equipo_jug_part}')
                print(f'\t- Año: {year_part}')
                print('\nBuscando jugador en entidad partido...')

                # Busco match entre df_part y df_jug solo si ya no lo busque para dicho jugador
                if f'{equipo_jug_part}{year_part}{nombre_jug_part}' in d.keys():
                    l = d[f'{equipo_jug_part}{year_part}{nombre_jug_part}']
                    edad, altura, overall_rating = l[0], l[1], l[2]
                else:
                    edad, altura, overall_rating = find_match(df_jug, nombre_jug_part, equipo_jug_part, year_part)

                # Obtengo datos del jugador (overall_rating, edad, altura, valor de mercado)
                print(f"\nMejor coincidencia: \n Edad: {edad}, Altura: {altura}, Overall rating: {overall_rating}")

                if edad is not None:
                    l_prom_edad.append(edad)
                    l_prom_alt.append(altura)
                    l_prom_rating.append(overall_rating)
                    # l_prom_valor.append(df_jug.loc[idx, 'valor_mercado'])  # es un string...

                    # Guardo coincidencia de jugador para evitar volver a buscarlo...
                    d[f'{equipo_jug_part}{year_part}{nombre_jug_part}'] = [edad, altura, overall_rating]
                else:
                    print("Fallo la busqueda")

        # Guardo promedios de edad, altura, overall_rating y valor de mercado
        try:
            df_part.loc[i, f'{variable}_prom_edad'] = sum(l_prom_edad) / len(l_prom_edad)
            df_part.loc[i, f'{variable}_prom_alt'] = sum(l_prom_alt) / len(l_prom_alt)
            df_part.loc[i, f'{variable}_prom_rat'] = sum(l_prom_rating) / len(l_prom_rating)
            print(f'Promedio de edad: {sum(l_prom_edad) / len(l_prom_edad)}')
            print(f'Promedio de altura: {sum(l_prom_alt) / len(l_prom_alt)}')
            print(f'Promedio de rating: {sum(l_prom_rating) / len(l_prom_rating)}')

        except ZeroDivisionError:
            print("Aparentemente no hay datos de jugadores para el partido")
    return df_part

def find_match(df_jug, nombre_jug_part, equipo_jug_part, year_part):

    # Defino variables
    match_data = {'edad': None, 'altura': None, 'overall_rating': None}
    match, best_match = 0, 0

    # Formateo campos para poder hacer la relacion entre entidades
    name, surname = separate_name_and_surname(nombre_jug_part)
    nombre_jug_part = f'{name}. {surname}'

    # Filtro dataset por año
    df_jug_filt = df_jug[(df_jug['fecha'].dt.year == year_part)]

    # Por partido (fila) en entidad jugador
    for j in df_jug_filt.index:

        # Obtengo entidad jugador por fecha, nombre del jugador, equipo? (OJO aqui, puede haber jugadores con mismo nombre...)
        nombre_comp_jug = df_jug_filt.loc[j, 'nombre']  # e.g. l. gonzalez pirez
        apellido = nombre_comp_jug[nombre_comp_jug.find(" "):].strip() if len(nombre_comp_jug.split()) > 1 else nombre_comp_jug  # nombre_comp_jug[nombre_comp_jug.find(". ")+2:] --> Falla en luis leal puesot que no tiene punto
        equipo_jug_comp = df_jug_filt.loc[j, 'equipo_actual']  # central cordoba
        # equipo_jug_corto = equipo_jug_comp.split()[0]  # central  # Selecciono solo Boca puesto que Boca jrs no es igual a Boca Juniors y es el misma equipo

        # Defino condiciones para que exista un match entre jugadores de cada entidad
        cond_nombre_jug_comp = (nombre_comp_jug in nombre_jug_part) or (nombre_jug_part in nombre_comp_jug)
        cond_apellido_jug = apellido in surname  # nombre_jug_2 == nombre_jug_part.split()[1]
        cond_nombre_equipo_comp = equipo_jug_comp in equipo_jug_part
        cond_nombre_equipo_corto = (equipo_jug_comp.split()[0] == equipo_jug_part.split()[0]) or (equipo_jug_comp.split()[-1] == equipo_jug_part.split()[-1])  # Para "Central Cordoba" en "Central cordoba SDE" y no caer en "Rosario central" # Para "atl. tucuman" y "atletico tucuman"

        # Si hay una coincidencia exacta
        if cond_nombre_jug_comp and cond_nombre_equipo_comp:
            match = 3
            print(f"Coincidencia del tipo {match}: {list(df_jug_filt.loc[j])}")

        # Si hay una gran coincidencia
        elif (cond_nombre_jug_comp and cond_nombre_equipo_corto) or (cond_apellido_jug and cond_nombre_equipo_comp):
            match = 2
            print(f"Coincidencia del tipo {match}: {list(df_jug_filt.loc[j])}")

        # Si hay una coincidencia posible
        elif (cond_apellido_jug and cond_nombre_equipo_corto):
            match = 1
            print(f"Coincidencia del tipo {match}: {list(df_jug_filt.loc[j])}")

        # Si es la mejor coincidencia, guardo sus datos
        if match > best_match:
            best_match = match
            match_data['edad'] = df_jug_filt.loc[j, 'edad']
            match_data['altura'] = df_jug_filt.loc[j, 'altura']
            match_data['overall_rating'] = df_jug_filt.loc[j, 'overall_rating']
    return match_data['edad'], match_data['altura'], match_data['overall_rating']

def separate_name_and_surname(name):
    """
    De un nombre completo separo en nombre y apellido
    :param name: String. Nombre completo (e.g. "L. Gonzalez", "Nacho Fernandez", "higuita")
    :return:
    """
    # Si contiene punto  (e.g. "Gonzalez L.", "De la cruz N.")
    if len(name.split(".")) > 1:

        pos_punto = name.find('.')
        nombre = name[pos_punto - 1:pos_punto]
        apellido = name[:pos_punto - 2]  # Si hay doble apellido, lo selecciono   # apellido = nombre_jug_part.split()[0]  # Si hay doble apellido, me quedo solo con el primero...
        return nombre, apellido

    # Si contiene punto  (e.g. "Nacho Fernandez")
    elif len(name.split()) > 1:
        nombre = name[:name.find(" ")][0]  # Solo la letra inicial
        apellido = name[name.find(" ")+1:]
        return nombre, apellido

    # Si es un solo nombre (suele ser solo el apellido)
    else:
        return str(), name

def main():
    # Definicion de variables
    l_var = ['l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc', 'l_jug_sup_vis', 'l_jug_ausentes_loc', 'l_jug_ausentes_vis']

    # Levanto datasets
    df_part = pd.read_excel("./df_formated.xlsx")
    df_jug = pd.read_excel("./df_jug_formated.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace falta el index_col=0
    print(df_part.head())

    # Integro datasets
    for var in l_var:
        df = search_player_data(df_part, df_jug, var)

    df.to_excel('./df_integrated.xlsx', index=False)


# main()