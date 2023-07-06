import pandas as pd
import math
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz

# RELLENO DE DATOS DE WHOSCORED A PARTIR DE FLASHCORE
def buscar_coincidencias(str1, str2):
    if fuzz.token_set_ratio(str1, str2) < 50:
        return False
    return True

def rellenar_statistics_with_flashscore(df_part_who, df_part_flash):

    # Saco horas de fechas para poder compararlas
    df_part_who['fecha_sin_hora'] = df_part_who['fecha'].dt.date
    df_part_flash['fecha_sin_hora'] = df_part_flash['fecha'].dt.date

    # Selecciono partidos sin estadisticas
    df_filt = df_part_who[df_part_who['posesion_loc'].isnull()]

    print(df_filt)

    # Por partido sin estadisticas en Whoscored
    for idx, row in df_filt.iterrows():

        fecha_who = row['fecha_sin_hora']
        equipo_loc_who = row['equipo_loc']
        equipo_vis_who = row['equipo_vis']

        l_col = ['posesion_loc', 'posesion_vis', 'remates_loc', 'remates_vis', 'remates_a_puerta_loc', 'remates_a_puerta_vis', 'faltas_loc', 'faltas_vis', 'pases_loc', 'pases_vis', 'pases_comp_loc', 'pases_comp_vis', 'offsides_loc', 'offsides_vis']

        # Busco match entre dfs
        df_filt = df_part_flash[df_part_flash['fecha_sin_hora'] == fecha_who]

        for index, fila in df_filt.iterrows():

            equipo_loc_fs = fila['equipo_loc']
            equipo_vis_fs = fila['equipo_vis']

            # Si lo encontró
            if buscar_coincidencias(str1=equipo_loc_fs, str2=equipo_loc_who) and buscar_coincidencias(str1=equipo_vis_fs, str2=equipo_vis_who):

                # Relleno con columnas de df_part_flash en caso que tenga datos
                for col in l_col:
                    df_part_who.loc[idx, col] = df_part_flash.loc[index, col]

    return df_part_who


# Integracion entre datos de WHOSCORED
def map_data(df_jug, df_jug_part):
    # Filtrar las columnas necesarias de df_jug_part
    df_jug_filtered = df_jug[['id_jug', 'altura', 'fecha_nac']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_merged = pd.merge(df_jug_part, df_jug_filtered, on='id_jug', how='left')

    return df_merged

def integrate_player_to_part(df_jug_part, df_part):
    # De df_jug quiero prom_edad_{tit, sup}_{loc, vis}, prom_alt_{tit, sup}_{loc, vis}
    # De df_jug_part quiero prom_rat_{tit, sup}_{loc, vis} y ponderarlo por min played...

    # Calculo edad
    df_jug_part = determine_edad(df_part, df_jug_part)

    # Calculo min_played
    df_jug_part = determine_min_played(df_jug_part)

    # Por jugador, construyo sum_min_played y prom_rating en ultimos n partidos --> para no requerir equipo, uso fecha
    df_jug_part = determine_var_en_ult_partidos(df_jug_part, 'min_played', type='sum')
    df_jug_part = determine_var_en_ult_partidos(df_jug_part, 'rating', type='mean_pond', var_pond='min_played')
    df_jug_part.to_excel('/Users/nachomondino/Desktop/df_jug_part_rating_ult_part.xlsx')

    # df_jug_part = pd.read_excel('/Users/nachomondino/Desktop/df_jug_part_rating_ult_part.xlsx', index_col=0)

    # Construyo prom_edad_{tit, sup}_{loc, vis}, prom_alt_{tit, sup}_{loc, vis}     # Agregar construccion de rating
    df_part = construct_variables_jug_in_part(df_part, df_jug_part)
    df_part.to_excel('/Users/nachomondino/Desktop/df_part_int.xlsx')

def determine_min_played(df):

    def calculate_minutes_played(row):
        if row['titularidad'] == 'titular':
            if pd.isna(row['min_cambio']) or row['min_cambio'] >=90:
                return 90
            else:
                return row['min_cambio']
        elif row['titularidad'] == 'suplente':
            if pd.isna(row['min_cambio']) or row['min_cambio'] >=90:
                return 0
            else:
                return 90 - row['min_cambio']
        else:
            return None

    # Aplicar la función a cada fila del DataFrame para calcular los minutos jugados.
    df['min_played'] = df.apply(calculate_minutes_played, axis=1)
    return df

def determine_edad(df_part, df_jug_part):

    # Filtrar las columnas necesarias de df_jug_part
    df_part_filtered = df_part[['id_part', 'fecha']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_merged = pd.merge(df_jug_part, df_part_filtered, on='id_part', how='left')

    # Calculo edad (fecha_hora - fecha nac)
    diferencia_dias = (df_merged['fecha'] - df_merged['fecha_nac']).dt.days  # Calcular la diferencia en días
    df_merged['edad'] = diferencia_dias // 365  # Calcular la edad en años
    # df_merged = df_merged.drop(['fecha_nac'], axis=1)  # Borro columnas fecha_hora y fecha_nac

    return df_merged

def determine_var_en_ult_partidos(df, variable, type='mean', var_pond=None):  # Calcula bien.
    """
    Obtiene el promedio de las estadisticas en los ultimos partidos

    :param df: Dataframe.
    :param n_ult_part: Integer. Numero de partidos de los cuales obtener los goles
    :param variable: String. Nombre de variable a promediar
    :return: Dataframe con estadisticas promediadas
    """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por jugador
    for id_jug in df['id_jug'].unique():
        # print(f"ID JUGADOR: {id_jug}")

        # Obtengo los partidos que jugó el jugador
        df_filt = df[df['id_jug'] == id_jug]
        # print(df_filt)

        # Recorrer los partidos del jugador
        for idx, row in df_filt.iterrows():

            # Filtro para seleccionar los ultimos partidos del jugador en los ultimos n dias
            fecha_part = row['fecha']
            fecha_limite = fecha_part - timedelta(days=30)
            df_seleccionados = df_filt.copy()
            df_seleccionados = df_seleccionados.dropna(subset=[variable])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)
            df_seleccionados = df_seleccionados.loc[(df_seleccionados['fecha'] >= fecha_limite) & (df_seleccionados['fecha'] < fecha_part)]
            largo = len(df_seleccionados)

            if largo > 0:
                if type == "mean_pond":
                    df_seleccionados = df_seleccionados.dropna(subset=[var_pond])  # Elimino registros en que no se tiene la variable (evita que el prom o sum de nan)

                    if len(df_seleccionados) > 0:
                        promedio_ponderado = (df_seleccionados[var_pond] * df_seleccionados[variable]).sum() / df_seleccionados[var_pond].sum()
                        df.loc[idx, f'prom_pond_{variable}_ult_part'] = promedio_ponderado

                elif type == "mean":
                    promedio = df_seleccionados[variable].sum() / len(df_seleccionados[variable])  # df_seleccionados[variable].dropna().sum() / len(df_seleccionados[variable].dropna())
                    df.loc[idx, f'prom_{variable}_ult_part'] = promedio

                elif type == "sum":
                    suma = df_seleccionados[variable].sum()  # df_seleccionados[variable].dropna().sum()
                    df.loc[idx, f'sum_{variable}_ult_part'] = suma

    return df

def construct_variables_jug_in_part(df_part, df_jug_part):  # Agregar calculo de rating y min played

    l_condiciones = df_jug_part['condicion'].unique()
    l_titularidades = df_jug_part['titularidad'].unique()

    # Por partido en df_part
    for index, row in df_part.iterrows():

        df_jug_part_filt = df_jug_part.copy()
        df_jug_part_filt_1 = df_jug_part_filt[df_jug_part_filt['id_part'] == row['id_part']]

        # Por condicion (home, away)
        for condicion in l_condiciones:
            df_jug_part_filt_2 = df_jug_part_filt_1[df_jug_part_filt_1['condicion'] == condicion]

            # Por titularidad (tit, sup)
            for titularidad in l_titularidades:

                # Selecciono registros
                df_jug_part_filt_3 = df_jug_part_filt_2[df_jug_part_filt_2['titularidad'] == titularidad]
                largo = len(df_jug_part_filt_3)
                # print(df_jug_part_filt_3)
                # print(sum(df_jug_part_filt_3['edad']))
                # print(sum(df_jug_part_filt_3['altura']))
                # print(sum(df_jug_part_filt_3['prom_rating_ult_part']))
                # print(sum(df_jug_part_filt_3['sum_min_played_ult_part']))

                # Calculo promedio de edad y altura
                if largo > 0:
                    prom_edad = df_jug_part_filt_3['edad'].dropna().sum() / len(df_jug_part_filt_3['edad'].dropna())
                    prom_alt = df_jug_part_filt_3['altura'].dropna().sum() / len(df_jug_part_filt_3['altura'].dropna())
                    sum_rat = df_jug_part_filt_3['prom_pond_rating_ult_part'].dropna().sum()
                    sum_min_played = df_jug_part_filt_3['sum_min_played_ult_part'].dropna().sum()

                    # Guardo columna en df_part
                    df_part.loc[index, f'prom_edad_{condicion}_{titularidad}'] = prom_edad
                    df_part.loc[index, f'prom_alt_{condicion}_{titularidad}'] = prom_alt
                    df_part.loc[index, f'sum_rat_{condicion}_{titularidad}'] = sum_rat
                    df_part.loc[index, f'sum_min_{condicion}_{titularidad}'] = sum_min_played

    return df_part

def prueba():
    # Levanto datasets de prueba
    df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/argentina/df_part_formated.xlsx')
    df_jug_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/argentina/df_jug_part.xlsx')
    df_jug = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/argentina/df_jug_formated.xlsx')

    # Integro df_jug a df_jug_part
    df_jug_part_integ = map_data(df_jug, df_jug_part)
    df_jug_part_integ.to_excel('/Users/nachomondino/Desktop/df_prueba.xlsx')

    # Integro df_jug_part_integ (df_jug_part + df_jug) a df_part
    df_part_integ = integrate_player_to_part(df_jug_part_integ, df_part)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()












'''
import pandas as pd
from fuzzywuzzy import fuzz
import warnings
import time
from tqdm import tqdm


def player_data_in_match(df_part, df_jug, pais):
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
    warnings.filterwarnings('ignore')  # Ver el ignore, y solucionarlo en vez de ignorarlo...

    # Levanto dataset de jugadores ya buscados, o bien, lo creo
    try:  # cuidado que si mejoras la extraccion, el cambio puede que no se vea puesto que levanta el df_jug_encontrado viejo...
        df_jug_encontrados = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_integrated_jug_encontrados.xlsx')
    except:
        df_jug_encontrados = pd.DataFrame(columns=['nombre', 'equipo', 'fecha', 'edad', 'altura', 'overall_rating', 'valor_mercado', 'str_encont'])
    df_jug_buscados = df_jug_encontrados.copy()  # Razon de inclusion: evito que no se realicen cambios cuando en hago una mejora en integrate_data. No la hace puesto que df_jug_encontrados antes guaradaba a todos los jugadores a pesar de no habarle encontrado match. Si la nueva version encontraba match, no lo iba a usar puesto que el jugador ya aparecia en df_jug_encontrados...
    df_jug_no_encontrados = pd.DataFrame(columns=['nombre', 'equipo', 'fecha', 'edad', 'altura', 'overall_rating', 'valor_mercado', 'str_encont'])
    n_jug_encontrados_inicial = len(df_jug_buscados)

    # Por titularidad (Titular, suplente o ausente)
    for titularidad in l_titularidad:

        # Por condicion (Local o visitante)
        for condicion in l_condicion:

            # Defino pattern y con el, selecciono las variables a procesar
            pattern = f'jug_{titularidad}_[a-z]*[_]*{condicion}_[0-9]+'  # CAMBIE EL PATTERN PARA QUE SUP Y SUP_ING SEAN PROCESADOS JUNTOS. VERIFICAR QUE FUNCIONA..
            l_col_to_preprocess = df_part.filter(regex=pattern, axis=1).columns.tolist()  #LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...
            print(f'\nColumnas a procesar: {l_col_to_preprocess}')

            # Inicializo barra de progreso
            progress_bar = tqdm(total=len(df_part), ncols=80)

            # Por partido (fila) en entidad partido
            for i, row in df_part.iterrows():  # for i in range(len(df_part)):

                progress_bar.update(1)

                # Obtengo equipo y año del partido
                equipo_jug_ent_part = row['equipo_loc'] if condicion == 'loc' else row['equipo_vis']  # equipo_jug_ent_part = df_part.loc[i, 'equipo_loc'] if condicion == 'loc' else df_part.loc[i, 'equipo_vis']  # DEPENDERA DE SI ES LOCAL O VIS...
                fecha_ent_part = row['fecha']   # year_ent_part = df_part.loc[i, 'fecha'].year   # e.g. 2023
                # print(f' Partido Nº: {i} '.center(120, '#'))

                # Busco el fifa correspondiente segun la fecha del partido
                fecha_part_fifa = search_fecha_actualizacion(df_jug, fecha_ent_part)  # Pasarle lista  de posibles fechas en vez de df_jug...

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
                        # print(f'\t- Fecha: {fecha_ent_part}')
                        # print('\nBuscando jugador en entidad partido...')

                        jugador_buscado = df_jug_buscados[
                            (df_jug_buscados['nombre'] == nombre_jug_ent_part) &
                            (df_jug_buscados['equipo'] == equipo_jug_ent_part) &
                            (df_jug_buscados['fecha'] == fecha_part_fifa)].head(1)

                        # Si el jugador no fue buscado aun
                        if jugador_buscado.empty:

                            # Busco coincidencia en df_jug
                            edad, altura, overall_rating, valor_mercado, str_encontrado = find_player_in_ent_jug(df_jug, nombre_jug_ent_part, equipo_jug_ent_part, fecha_part_fifa)
                            d_data = {'nombre': nombre_jug_ent_part, 'equipo': equipo_jug_ent_part, 'fecha': fecha_part_fifa, 'edad': edad, 'altura': altura, 'overall_rating': overall_rating, 'valor_mercado': valor_mercado, 'str_encont': str_encontrado}

                            # Si encontró el jugador
                            if edad is not None:
                                # Lo guardo como encontrado
                                df_jug_encontrados = df_jug_encontrados.append(d_data, ignore_index=True)
                            # Si no encontró el jugador
                            else:
                                df_jug_no_encontrados = df_jug_no_encontrados.append(d_data, ignore_index=True)

                            # Guardo jugador como buscado por mas que no lo haya encontrado
                            df_jug_buscados = df_jug_buscados.append(d_data, ignore_index=True)

                        # Si el jugador ya fue buscado
                        else:
                            # No lo vuelvo a buscar sino que llamo los resultados de la anterior busqueda
                            edad = jugador_buscado['edad'].values[0]
                            altura = jugador_buscado['altura'].values[0]
                            overall_rating = jugador_buscado['overall_rating'].values[0]
                            valor_mercado = jugador_buscado['valor_mercado'].values[0]
                            str_encontrado = jugador_buscado['str_encont'].values[0]
                            # print('Evité nueva busqueda, uso datos ya buscados')

                        # Imprimo resultados de busqueda
                        # print(f"Mejor coincidencia: \n Edad: {edad}, Altura: {altura}, Overall rating: {overall_rating}, Valor mercado: {valor_mercado}, Jugador encontrado: {str_encontrado}")

                        # Solo si se encontró al jugador, guardo datos para calculo de promedio
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

                    # Calculo nro de jugadores lesionados
                    if titularidad == 'aus':
                        df_part.loc[i, f'n_jug_{titularidad}_{condicion}'] = len(l_prom_rating)

                        print("Numero de ausentes: ", len(l_prom_rating))
                    # print(f'\nPromedio de edad: {sum(l_prom_edad) / len(l_prom_edad)} \nPromedio de altura: {sum(l_prom_alt) / len(l_prom_alt)} \nPromedio de rating: {sum(l_prom_rating) / len(l_prom_rating)} \nPromedio de valor: {sum(l_prom_valor) / len(l_prom_valor)} ')

                except ZeroDivisionError:
                    # print("Aparentemente no hay datos de jugadores para el partido")
                    pass

            # Cerrar la barra de progreso al finalizar
            progress_bar.close()

            # Elimino variables recien procesadas
            df_part = df_part.drop(l_col_to_preprocess, axis=1)

    # Exporto dataframe de jugadores encontrados solo si se encontraron nuevos jugadores
    if len(df_jug_encontrados) > n_jug_encontrados_inicial:
        print(f"Reemplazo datos puesto que se tienen {len(df_jug_encontrados) - n_jug_encontrados_inicial} jugadores nuevos")
        df_jug_encontrados.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_integrated_jug_encontrados.xlsx', index=False)
    df_jug_no_encontrados.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_integrated_jug_no_encontrados.xlsx',index=False)
    return df_part

def find_player_in_ent_jug(df_jug, nombre_jug_ent_part, equipo_jug_ent_part, fecha_ent_part):
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
            if fuzz.token_set_ratio(palabra, row[columna]) < 50: # Lo baje porque jugadores como lautaro giaccone no son encontrados dado que aparece como "laurtaro giaccone"
                return False
        return True

    # Selecciono los datos de jugadores segun el fifa que se requiera
    df_jug_filt = df_jug[df_jug['fecha'] == fecha_ent_part]  # df_jug_filt = df_jug[df_jug['fecha'].dt.year == year_ent_part]

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

def search_fecha_actualizacion(df, fecha_part):  # Verificar funcionamiento

    # Si el partido es de Dic 2021, deberá escoger el fifa 22 puesto dic es posterior a Sept y luego como diciembre es anterior a ene, la primera actualizacion.
    # Si el partido es de Jun 2022, deberá escoger el fifa 22 puesto jun es anterior a Sept y luego como jun es posterio a ene, la ultima actualizacion.

    year_part = fecha_part.year  # e.g. "2021"
    mes_part = fecha_part.month  # e.g. "Dic"

    # Si es posterior a Septiembre
    if mes_part >= 9:  # Dic 2021 entraria aqui

        year_str = str(year_part+1)[-2:]  # Ultimos dos "22"
        fifa_str = f"fifa {year_str}"  # FIFA 22

        # Si existe un fifa para dicha fecha
        if fifa_str in df['fifa'].unique():

            # Selecciono posibles fechas de actualizion segun el fifa
            l_posibles_fechas = df[df['fifa'] == fifa_str]['fecha'].unique() # [18 ago 2022, 16 Ago 2021]

            # Elijo la primera fecha de actualizacion (pues es si o si es anterior a enero)
            fecha = min(l_posibles_fechas)

        else:
            # print(f"No hay fifa para la fecha {fecha_part}")
            return None

    # Si es anterior a Septiembre
    else:  # Jun 2021 entraria aqui

        year_str = str(year_part)[-2:]  # Ultimos dos "21"
        fifa_str = f"fifa {year_str}"  # FIFA 21

        # Si existe un fifa para dicha fecha
        if fifa_str in df['fifa'].unique():

            # Selecciono posibles fechas de actualizion segun el fifa
            l_posibles_fechas = df[df['fifa'] == fifa_str]['fecha'].unique()  # Cada elemento es del tipo numpy.datetime64... sin embargo, parece funcionar igual

            # Elijo la ultima fecha de actualizacion (pues si o si es posterior a enero)
            fecha = max(l_posibles_fechas)

        else:
            # print(f"No hay fifa para la fecha {fecha_part}")
            return None

    # Convertir a datetime (por algun motivo lo entiende como numpy.datetime64...
    # fecha_dt = datetime.datetime.utcfromtimestamp(fecha.astype('O') / 1e9)
    # print(f"Para el partido jugado en {mes_part}/{year_part}, el fifa que le corresponde es {fifa_str} y la actualizacion {fecha}")
    return fecha

def prueba():
    start = time.time()
    print("\nIntegrando los datos...")

    pais = 'argentina'

    # Levanto datasets
    df_part = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_part_cleaned.xlsx")
    df_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_jug_cleaned.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace falta el index_col=0
    print(df_part.head(1))
    print(df_jug.head(1))

    # Integro datasets
    df_integrated = player_data_in_match(df_part, df_jug)
    df_integrated.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=False)

    end = time.time()
    print(f"Integracion de datos en {(end - start) / 60:.1f} minutos")
'''