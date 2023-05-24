# Juntar entidades jugadores, partido y atrib_jugadores
import pandas as pd
import re

def clean_text(text):  # EFICIENTIZAR. La hice asi no mas para poder integrar datos...
    # Defino variables
    d = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}  # la ñ no seria acento...

    # Verificar si el elemento es un string
    if isinstance(text, str):

        # Hago minuscula
        new_text = str()

        # Por carecter
        for char in text.lower():

            # Si es una letra con tilde
            if char in d.keys():
                new_text += d[char]
            else:
                new_text += char
        return new_text
    return text

def separate_lists_in_columns(df, variable):  # Altamente ineficiente. Cuando extraiga cada jugador en vez de la lista, podre borrala
    """
    Convierto columnas que contienen listas en multiples columnas de un solo elemento
    :param df: Dataframe.
    :param variable: String. Nombre de la variable
    :return: Dataframe.
    """
    # Por registro
    for i in range(len(df)):

        # Obtengo lista
        string_with_list = df.loc[i, variable]

        # Verificar si el elemento es un string (Evito nan)
        if isinstance(string_with_list, str):

            # Convierto string a lista
            l_jug = eval(string_with_list)  # e.g. ["Dibu", ..., "Messi"]

            # Por elemento de la lista
            for j in range(len(l_jug)):

                # Guardo jugador en columna nueva
                df.loc[i, f'{variable[2:]}_{j+1}'] = l_jug[j]

    df = df.drop([variable], axis=1)
    return df

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

    # Si es un solo nombre
    else:
        return name, str()

def search_player_data(df_part, df_jug, variable):  # ojo con la comparacion si hay mayusculas...
    """
    Segun el nombre del jugador, la fecha y el equipo(?) en la entidad partido, busco sus atributos en la entidad jugador
    :param df:
    :return:
    """
    # Defino las columnas a procesar segun la variable
    pattern = variable[2:] + '_[0-9]*'
    l_jug_to_preprocess = df_part.filter(regex=pattern, axis=1).columns.tolist()  #LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...

    # Por partido (fila) en entidad partido
    for i in range(len(df_part)):

        # Obtengo equipo y año del partido
        equipo_jug_part = clean_text(df_part.loc[i, 'equipo_loc'] if 'loc' in variable else df_part.loc[i, 'equipo_vis'])  # DEPENDERA DE SI ES LOCAL O VIS...
        year_part = df_part.loc[i, 'fecha'].year   # e.g. 2023
        print(f' Partido Nº: {i} '.center(120, '#'))

        # Reinicio variables
        l_prom_edad, l_prom_alt, l_prom_rating, l_prom_valor = [], [], [], []

        # Por jugador
        for jug in l_jug_to_preprocess:

            # Busco nombre del jugador en la entidad partido
            nombre_jug_part = df_part.loc[i, jug]  # e.g. "Rodriguez D. (G) (C)" --> rodriguez d. (g) (c)

            # Si el jugador no es nan
            if isinstance(nombre_jug_part, str):

                # Formateo campos para poder hacer la relacion entre entidades
                name, surname = separate_name_and_surname(clean_text(nombre_jug_part))
                nombre_jug_part = f'{name}. {surname}'
                print(f' {nombre_jug_part} '.center(100, '-'))
                print(f' JUGADOR EN ENTIDAD PARTIDO: ')
                print(f'\t- Nombre: {nombre_jug_part}')
                print(f'\t- Equipo: {equipo_jug_part}')
                print(f'\t- Año: {year_part}')
                print('\nBuscando jugador en entidad partido...')

                # Filtro dataset por año
                df_juf_filtrado = df_jug[(df_jug['fecha'].dt.year == year_part)]

                edad, altura, overall_rating = None, None, None  # No se si es lo mejor asignar None a los jugadores que fallan su coincidencia...
                match, best_match = 0, 0

                # Por partido (fila) en entidad jugador
                for j in df_juf_filtrado.index:

                    # Obtengo entidad jugador por fecha, nombre del jugador, equipo? (OJO aqui, puede haber jugadores con mismo nombre...)
                    nombre_comp_jug = clean_text(df_juf_filtrado.loc[j, 'nombre'])  # e.g. l. gonzalez pirez
                    # nombre_jug_2 = re.split(r'. | ', nombre_jug_1)[1:] if len(nombre_jug_1.split()) > 1 else nombre_jug_1  # gonzalez # Selecciono solo el apellido C. Barrios no es igual a N.Barrios y es la misma persona
                    apellido = nombre_comp_jug[nombre_comp_jug.find(". ")+2:]
                    equipo_jug_comp = clean_text(df_juf_filtrado.loc[j, 'equipo_actual'])  # central cordoba
                    equipo_jug_corto = equipo_jug_comp.split()[0]  # central  # Selecciono solo Boca puesto que Boca jrs no es igual a Boca Juniors y es el misma equipo

                    # Defino condiciones para que exista un match entre jugadores de cada entidad
                    # Intento 1
                    # Problema puede haber mas de un match y no tengo como rankear que match es mejor...
                    # cond_nombre = (nombre_jug_1 in nombre_jug_part) or (nombre_jug_2 == nombre_jug_part.split()[1])
                    # cond_equipo = (equipo_jug_1 in equipo_jug_part) or (equipo_jug_2 == equipo_jug_part.split()[0])
                    # Si hay coincidencia
                    # if cond_nombre and cond_equipo:
                    # jug_ent_part = "L. González Pirez" ; " O. Opazo lara"
                    # jug_ent_jug = "L. González Pirez"; "Ó. Opazo"

                    # Intento 2
                    cond_nombre_jug_comp = nombre_comp_jug in nombre_jug_part
                    cond_apellido_jug = apellido in surname  # nombre_jug_2 == nombre_jug_part.split()[1]
                    cond_nombre_equipo_comp = equipo_jug_comp in equipo_jug_part
                    cond_nombre_equipo_corto = equipo_jug_corto == equipo_jug_part.split()[0]  # Para "Central Cordoba" en "Central cordoba SDE" y no caer en "Rosario central"

                    # Si hay una coincidencia exacta
                    if cond_nombre_jug_comp and cond_nombre_equipo_comp:
                        match = 3
                        print(f"Coincidencia del tipo {match} \n {list(df_juf_filtrado.loc[j])}")

                    # Si hay una gran coincidencia
                    elif (cond_nombre_jug_comp and cond_nombre_equipo_corto) or (cond_apellido_jug and cond_nombre_equipo_comp):
                        match = 2
                        print(f"Coincidencia del tipo {match} \n {list(df_juf_filtrado.loc[j])}")

                    # Si hay una coincidencia posible
                    elif (cond_apellido_jug and cond_nombre_equipo_corto):
                        match = 1
                        print(f"Coincidencia del tipo {match} \n {list(df_juf_filtrado.loc[j])}")

                    # Si no hay coincidencia
                    else:
                        pass

                    # Si es la mejor coincidencia, guardo sus datos
                    if match > best_match:
                        best_match = match
                        edad = df_juf_filtrado.loc[j, 'edad']
                        altura = df_juf_filtrado.loc[j, 'altura']
                        overall_rating = df_juf_filtrado.loc[j, 'overall_rating']

                # Obtengo datos del jugador (overall_rating, edad, altura, valor de mercado)
                print(f"\nSeleccionando la mejor coincidencia: \n Edad: {edad}, Altura: {altura}, Overall rating: {overall_rating}")

                if edad is not None:  # CUIADADO, FALLA GONZALEZ PIREZ
                    l_prom_edad.append(edad)
                    l_prom_alt.append(altura)
                    l_prom_rating.append(overall_rating)
                    # l_prom_valor.append(df_jug.loc[idx, 'valor_mercado'])  # es un string...

        # Guardo promedios de edad, altura, overall_rating y valor de mercado
        try:
            df_part.loc[i, f'{variable}_prom_edad'] = sum(l_prom_edad) / len(l_prom_edad)
            df_part.loc[i, f'{variable}_prom_alt'] = sum(l_prom_alt) / len(l_prom_alt)
            df_part.loc[i, f'{variable}_prom_rat'] = sum(l_prom_rating) / len(l_prom_rating)
            # df_part.loc[i, f'{variable}_prom_valor'] = sum(l_prom_valor) / len(l_prom_valor)
            print(f'Promedio de edad: {sum(l_prom_edad) / len(l_prom_edad)}')
            print(f'Promedio de altura: {sum(l_prom_alt) / len(l_prom_alt)}')
            print(f'Promedio de rating: {sum(l_prom_rating) / len(l_prom_rating)}')

        except ZeroDivisionError:
            print("Aparentemente no hay datos de jugadores para el partido")
    return df_part


def main():
    # Definicion de variables
    l_var = ['l_jug_tit_loc', 'l_jug_tit_vis', 'l_jug_sup_loc', 'l_jug_sup_vis', 'l_jug_ausentes_loc', 'l_jug_ausentes_vis']
    '''
    # Levanto datasets
    df = pd.read_excel("./df_formated.xlsx")
    print(df.head())

    # Separo columnas listas en multiples columnas
    for var in l_var:
        df = separate_lists_in_columns(df, var)
    df.to_excel('./prueba.xlsx', index=False)
    
    '''
    # Relaciono jugador de entidad partido con jugador de entidad jugador
    df = pd.read_excel('./prueba.xlsx')
    df_jug = pd.read_excel("./df_jug_formated.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace falta el index_col=0
    print(df_jug.head())

    df_part = search_player_data(df, df_jug, 'l_jug_tit_loc')
    df_part.to_excel('./df_integrated.xlsx', index=False)

main()


'''
def remove_accent(df, l_col_names):  # EFICIENTIZAR. La hice asi no mas para poder integrar datos...

    # Defino variables
    d = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}  # la ñ no seria acento...

    # Por registro
    for i in range(len(df)):

        # Por columna
        for col_name in l_col_names:

            string = df.loc[i, col_name]
            new_str = str()

            # Verificar si el elemento es un string
            if isinstance(string, str):

                # Por carecter
                for char in string:

                    # Si es una letra con tilde
                    if char in d.keys():
                        new_str += d[char]
                    else:
                        new_str += char

                # Guardo string sin acentos
                df.loc[i, col_name] = new_str
    return df

'''


'''Funcion antes de usar clean_text()
def search_player_data(df_part, df_jug, variable):  # ojo con la comparacion si hay mayusculas...
    """
    Segun el nombre del jugador, la fecha y el equipo(?) en la entidad partido, busco sus atributos en la entidad jugador
    :param df:
    :return:
    """
    # Defino las columnas a procesar segun la variable
    col_pattern = variable[2:] + '_[0-9]*'
    l_col_to_preprocess = df_part.filter(regex=col_pattern, axis=1).columns.tolist()  #LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...

    # Por partido
    for i in range(10):  # for i in range(len(df_part)):

        # Obtengo año del partido  # e.g. 2023
        year_part = df_part.loc[i, 'fecha'].year

        print(f' Partido Nº: {i} '.center(120, '#'))

        # Reinicio variables
        l_prom_edad, l_prom_alt, l_prom_rating, l_prom_valor = [], [], [], []

        # Por columna de jugador
        for col_jug in l_col_to_preprocess:

            # Busco datos (nombre, equipo y fecha) del jugador en la entidad partido
            nombre_jug_part = df_part.loc[i, col_jug]  # e.g. "Rodriguez D. (G) (C)"
            equipo_jug_part = df_part.loc[i, 'equipo_loc'] if 'loc' in variable else df_part.loc[i, 'equipo_vis']  # DEPENDERA DE SI ES LOCAL O VIS...

            # Si el jugador no es nan
            if isinstance(nombre_jug_part, str):

                # Formateo campos para poder hacer la relacion
                pos_punto = nombre_jug_part.find('.')
                nombre = nombre_jug_part[pos_punto-1:pos_punto]
                apellido = nombre_jug_part.split()[0]  # Si hay doble apellido, me quedo solo con el primero...
                nombre_jug_part = f'{nombre}. {apellido}'

                print(f' {nombre_jug_part} '.center(100, '-'))
                print(f' JUGADOR EN ENTIDAD PARTIDO: ')
                print(f'\t- Nombre: {nombre_jug_part}')
                print(f'\t- Equipo: {equipo_jug_part}')
                print(f'\t- Año: {year_part}')

                # Filtro entidad jugador por fecha, nombre del jugador, equipo? (OJO aqui, puede haber jugadores con mismo nombre...)
                # Busco match
                df_jug_filt = df_jug[(df_jug['nombre'] == nombre_jug_part) & (df_jug['fecha'].dt.year == year_part) & (df_jug['equipo_actual'].str.contains(equipo_jug_part)) ]
                print('\nBuscando jugador en entidad partido...')

                print('\nJUGADOR EN ENTIDAD JUGADOR:')
                print(f"Cantidad coincidencias en la busqueda: {len(df_jug_filt)}")

                # Si hay una sola coincidencia
                if len(df_jug_filt) == 1:

                    idx = df_jug_filt.index[0]
                    # print(list(df_jug.loc[idx]))

                    # Obtengo datos del jugador (overall_rating, edad, altura, valor de mercado)
                    l_prom_edad.append(df_jug.loc[idx, 'edad'])
                    l_prom_alt.append(df_jug.loc[idx, 'altura'])
                    l_prom_rating.append(df_jug.loc[idx, 'overall_rating'])
                    # l_prom_valor.append(df_jug.loc[idx, 'valor_mercado'])  # es un string...
                    print(f'\t- Edad: {df_jug.loc[idx, "edad"]}')
                    print(f'\t- Altura: {df_jug.loc[idx, "altura"]}')
                    print(f'\t- Rating: {df_jug.loc[idx, "overall_rating"]}', end='\n')

                else:
                    print("Se encontro mas de un jugador con dichos datos, o bien, ninguno")
                    print(df_jug_filt)

        # Guardo promedios de edad, altura, overall_rating y valor de mercado
        try:
            df_part.loc[i, f'{variable}_prom_edad'] = sum(l_prom_edad) / len(l_prom_edad)
            df_part.loc[i, f'{variable}_prom_alt'] = sum(l_prom_alt) / len(l_prom_alt)
            df_part.loc[i, f'{variable}_prom_rat'] = sum(l_prom_rating) / len(l_prom_rating)
            # df_part.loc[i, f'{variable}_prom_valor'] = sum(l_prom_valor) / len(l_prom_valor)
            print(f'Promedio de edad: {sum(l_prom_edad) / len(l_prom_edad)}')
            print(f'Promedio de altura: {sum(l_prom_alt) / len(l_prom_alt)}')
            print(f'Promedio de rating: {sum(l_prom_rating) / len(l_prom_rating)}')

        except ZeroDivisionError:
            print("Aparentemente no hay datos de jugadores para el partido")

    return df_part
'''