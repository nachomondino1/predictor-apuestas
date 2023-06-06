# Juntar entidades jugadores, partido y atrib_jugadores
import pandas as pd
from fuzzywuzzy import fuzz
import warnings


def player_data_in_match(df_part, df_jug):
    """
    Integra la entidad jugador en la entidad partido. Es decir, sintetiza los datos de los jugadores a cada partido en
    particular. Se determinan los promedios de edad, overall rating,  valor de mercado y altura del equipo titular,
    suplente y los ausentes para cada equipo.
    :param df:
    :return:
    """
    # Definicion de variables
    l_titularidad = ['tit', 'sup', 'aus']  # tengo que agregar 'sup_ing' pero se debe procesar con sup...
    l_condicion = ['loc', 'vis']
    df_jug_encontrados = pd.DataFrame(columns=['nombre', 'equipo', 'anio', 'edad', 'altura', 'overall_rating', 'valor_mercado', 'str_encont'])
    warnings.filterwarnings('ignore')

    # Por titularidad (Titular, suplente o ausente)
    for titularidad in l_titularidad:

        # Por condicion (Local o visitante)
        for condicion in l_condicion:

            # Defino pattern y con el, selecciono las variables a procesar
            pattern = f'jug_{titularidad}_[a-z]*[_]*{condicion}_[0-9]+'  # CAMBIE EL PATTERN PARA QUE SUP Y SUP_ING SEAN PROCESADOS JUNTOS. VERIFICAR QUE FUNCIONA..
            l_col_to_preprocess = df_part.filter(regex=pattern, axis=1).columns.tolist()  #LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...
            # print(f'Columnas a procesar: {l_col_to_preprocess}')

            # Por partido (fila) en entidad partido
            for i in range(len(df_part)):

                # Obtengo equipo y año del partido
                equipo_jug_ent_part = df_part.loc[i, 'equipo_loc'] if condicion == 'loc' else df_part.loc[i, 'equipo_vis']  # DEPENDERA DE SI ES LOCAL O VIS...
                year_ent_part = df_part.loc[i, 'fecha'].year   # e.g. 2023
                # print(f' Partido Nº: {i} '.center(120, '#'))

                # Reinicio variables
                l_prom_edad, l_prom_alt, l_prom_rating, l_prom_valor = [], [], [], []

                # Por jugador
                for jug in l_col_to_preprocess:

                    # Busco nombre del jugador en la entidad partido
                    nombre_jug_ent_part = df_part.loc[i, jug]  # gomez gaston # e.g. "Rodriguez D. (G) (C)"

                    # Si el jugador no es nan
                    if isinstance(nombre_jug_ent_part, str):

                        # QUITAR UNA VEZ QUE SE QUE FUNCIONA...
                        # print(f' {nombre_jug_ent_part} '.center(100, '-'))
                        # print(f' JUGADOR EN ENTIDAD PARTIDO: ')
                        # print(f'\t- Nombre: {nombre_jug_ent_part}')
                        # print(f'\t- Equipo: {equipo_jug_ent_part}')
                        # print(f'\t- Año: {year_ent_part}')
                        # print('\nBuscando jugador en entidad partido...')

                        jugador_encontrado = df_jug_encontrados[
                            (df_jug_encontrados['nombre'] == nombre_jug_ent_part) &
                            (df_jug_encontrados['equipo'] == equipo_jug_ent_part) &
                            (df_jug_encontrados['anio'] == year_ent_part)].head(1)

                        # Si el jugador no fue encontrado aun
                        if jugador_encontrado.empty:
                            # Busco coincidencia en df_jug
                            edad, altura, overall_rating, valor_mercado, str_encontrado = find_player_in_ent_jug(df_jug, nombre_jug_ent_part, equipo_jug_ent_part, year_ent_part)
                            df_jug_encontrados = df_jug_encontrados.append({'nombre': nombre_jug_ent_part, 'equipo': equipo_jug_ent_part, 'anio': year_ent_part, 'edad': edad, 'altura': altura, 'overall_rating': overall_rating, 'valor_mercado': valor_mercado, 'str_encont': str_encontrado}, ignore_index=True)

                        # Si el jugador ya fue encontrado
                        else:
                            # No lo vuelvo a buscar sino que llamo los resultados de la anterior busqueda
                            edad = jugador_encontrado['edad'].values[0]
                            altura = jugador_encontrado['altura'].values[0]
                            overall_rating = jugador_encontrado['overall_rating'].values[0]
                            valor_mercado = jugador_encontrado['valor_mercado'].values[0]
                            str_encontrado = jugador_encontrado['str_encont'].values[0]
                            # print('Evité nueva busqueda, uso datos ya buscados')

                        # Obtengo datos del jugador (overall_rating, edad, altura, valor de mercado)
                        # print(f"\nMejor coincidencia: \n Edad: {edad}, Altura: {altura}, Overall rating: {overall_rating}, Valor mercado: {valor_mercado}, Jugador encontrado: {str_encontrado}")

                        if edad is not None:
                            l_prom_edad.append(edad)
                            l_prom_alt.append(altura)
                            l_prom_rating.append(overall_rating)
                            l_prom_valor.append(valor_mercado)

                # Guardo promedios de edad, altura, overall_rating y valor de mercado
                try:
                    df_part.loc[i, f'prom_edad_jug_{titularidad}_{condicion}'] = sum(l_prom_edad) / len(l_prom_edad)
                    df_part.loc[i, f'prom_alt_jug_{titularidad}_{condicion}'] = sum(l_prom_alt) / len(l_prom_alt)
                    df_part.loc[i, f'prom_rat_jug_{titularidad}_{condicion}'] = sum(l_prom_rating) / len(l_prom_rating)
                    df_part.loc[i, f'prom_valor_jug_{titularidad}_{condicion}'] = sum(l_prom_valor) / len(l_prom_valor)
                    # print(f'\nPromedio de edad: {sum(l_prom_edad) / len(l_prom_edad)} \nPromedio de altura: {sum(l_prom_alt) / len(l_prom_alt)} \nPromedio de rating: {sum(l_prom_rating) / len(l_prom_rating)} \nPromedio de valor: {sum(l_prom_valor) / len(l_prom_valor)} ')

                except ZeroDivisionError:
                    # print("Aparentemente no hay datos de jugadores para el partido")
                    pass

            # Elimino variables
            df_part = df_part.drop(l_col_to_preprocess, axis=1)
            df_jug_encontrados.to_excel('./ver_jug_encontrados.xlsx', index=False)
    return df_part

def find_player_in_ent_jug(df_jug, nombre_jug_ent_part, equipo_jug_ent_part, year_ent_part):
    """
    Encuentra coincidencias de jugadores en un dataframe.
    :param df_jug: DataFrame que contiene los datos de los jugadores.
    :param nombre_jug_ent_part: Nombre del jugador a buscar.
    :param equipo_jug_ent_part: Nombre del equipo del jugador a buscar.
    :param year_ent_part: Año o temporada en el que se requiere el jugador a buscar.
    :return: Diccionario con los datos del jugador coincidente. Las claves son 'edad', 'altura', 'overall_rating',
             'valor_mercado' y 'str_encont'.
    """
    # Definicion de variables a extraer
    match_data = {'edad': None, 'altura': None, 'overall_rating': None, 'valor_mercado': None, 'str_encont': None}

    # Funcion que hace una busqueda aproximada de un string en una columna
    def buscar_coincidencias(row, palabras_clave, columna):
        for palabra in palabras_clave:
            if fuzz.token_set_ratio(palabra, row[columna]) < 60:
                return False
        return True

    # Selecciono los datos de jugadores del año del partido
    df_jug_filt = df_jug[df_jug['fecha'].dt.year == year_ent_part]

    # Selecciono los datos de jugadores segun los nombres de jugadores mas parecidos al buscado
    palabras_clave_nombre = nombre_jug_ent_part.split()
    df_jug_filt = df_jug_filt[df_jug_filt.apply(buscar_coincidencias, args=(palabras_clave_nombre, 'nombre'), axis=1)]

    # Si encontro al menos un jugador con nombre similar
    if len(df_jug_filt) > 0:

        # Selecciono los datos de jugadores segun el nombre del equipo mas similar al buscado
        palabras_clave_equipo = equipo_jug_ent_part.split()
        df_jug_filt = df_jug_filt[df_jug_filt.apply(buscar_coincidencias, args=(palabras_clave_equipo, 'equipo_actual'), axis=1)]

        # Si encontro al menos un equipo con nombre similar
        if len(df_jug_filt) > 0:

            # Extraigo datos del jugador
            match_data['edad'] = df_jug_filt.iloc[0]['edad']
            match_data['altura'] = df_jug_filt.iloc[0]['altura']
            match_data['overall_rating'] = df_jug_filt.iloc[0]['overall_rating']
            match_data['valor_mercado'] = df_jug_filt.iloc[0]['valor_mercado']
            match_data['str_encont'] = df_jug_filt.iloc[0]['nombre']

    return match_data['edad'], match_data['altura'], match_data['overall_rating'], match_data['valor_mercado'], match_data['str_encont']

def prueba():
    # Levanto datasets
    df_part = pd.read_excel("./data/df_part_cleaned.xlsx")
    df_jug = pd.read_excel("./data/df_jug_cleaned.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace falta el index_col=0
    print(df_part.head())

    # Integro datasets
    df_integrated = player_data_in_match(df_part, df_jug)
    df_integrated.to_excel('./df_integrated.xlsx', index=False)

# prueba()