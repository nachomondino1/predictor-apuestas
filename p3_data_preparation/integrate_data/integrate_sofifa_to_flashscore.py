# Juntar entidades jugadores, partido y atrib_jugadores
import pandas as pd
from fuzzywuzzy import fuzz
import warnings
import time
from tqdm import tqdm
from p3_data_preparation import clean_data
from sklearn.preprocessing import StandardScaler


def player_data_in_match(df_jug_part, df_jug):
    """
    Integra la entidad jugador en la entidad partido. Es decir, sintetiza los datos de los jugadores a cada partido en
    particular. Se determinan los promedios de edad, overall rating,  valor de mercado y altura del equipo titular,
    suplente y los ausentes para cada equipo.

    :param df:
    :return:
    """
    # Definicion de variables
    warnings.filterwarnings('ignore')  # Ver el ignore, y solucionarlo en vez de ignorarlo...
    df_jug_buscados = pd.DataFrame(columns=['id_jug', 'nombre', 'equipo', 'fecha', 'edad', 'altura', 'overall_rating', 'valor_mercado', 'str_encont'])

    # Levanto dataset de jugadores ya buscados, o bien, lo creo
    try:  # cuidado que si mejoras la extraccion, el cambio puede que no se vea puesto que levanta el df_jug_encontrado viejo...
        df_jug_buscados = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/argentina/df_integrated_jug_buscados.xlsx')
    except:
        df_jug_buscados = pd.DataFrame(columns=['id_jug', 'nombre', 'equipo', 'fecha', 'edad', 'altura', 'overall_rating', 'valor_mercado', 'str_encont'])

    # Preparo datasets para facilitar integracion (Nombres de jugadores en minuscula, sin acentos y sin caracteres especiales)
    df_jug = preparate_to_integrate(df_jug)
    df_jug_part = clean_data.prepare_text_columns(df_jug_part, l_col_to_except=['temporada'])

    # Inicializo barra de progreso
    progress_bar = tqdm(total=len(df_jug_part), ncols=80)

    # Por jugador en df_jug_part
    for i, row in df_jug_part.iterrows():  # for i in range(len(df_part)):

        # Obtengo equipo y año del partido
        nombre = row['nombre_jug']
        equipo = row['equipo_actual']
        fecha = row['fecha']
        # print(f' Jugador Nº: {i} '.center(120, '#'))

        # Busco el fifa correspondiente segun la fecha del partido
        fecha_part_fifa = search_fecha_actualizacion(df_jug, fecha)  # Pasarle lista  de posibles fechas en vez de df_jug...

        # Si el jugador no es nan
        if isinstance(nombre, str):

            jugador_buscado = df_jug_buscados[
                (df_jug_buscados['id_jug'] == row['id_jug']) &
                (df_jug_buscados['fecha'] == fecha_part_fifa)].head(1)

            # Si el jugador no fue buscado aun
            if jugador_buscado.empty:

                # Busco coincidencia en df_jug
                edad, altura, overall_rating, valor_mercado, str_encontrado = find_player_in_ent_jug(df_jug, row, fecha_part_fifa)
                d_data = {'id_jug': row['id_jug'], 'nombre': nombre, 'equipo': equipo, 'fecha': fecha_part_fifa, 'edad': edad,
                          'altura': altura, 'overall_rating': overall_rating, 'valor_mercado': valor_mercado,
                          'str_encont': str_encontrado}

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

            # Guardo datos
            df_jug_part.loc[i, 'edad'] = edad
            df_jug_part.loc[i, 'altura'] = altura
            df_jug_part.loc[i, 'overall_rating'] = overall_rating
            df_jug_part.loc[i, 'valor_mercado'] = valor_mercado
            progress_bar.update(1)

    # Cerrar la barra de progreso al finalizar
    progress_bar.close()

    '''
    # Exporto dataframe de jugadores encontrados solo si se encontraron nuevos jugadores
    if len(df_jug_encontrados) > n_jug_encontrados_inicial:
        print(f"Reemplazo datos puesto que se tienen {len(df_jug_encontrados) - n_jug_encontrados_inicial} jugadores nuevos")
        df_jug_encontrados.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_integrated_jug_encontrados.xlsx', index=False)
    df_jug_no_encontrados.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_integrated_jug_no_encontrados.xlsx',index=False)
    '''
    # df_jug_buscados.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/argentina/df_integrated_jug_buscados.xlsx',index=False)
    return df_jug_part

def find_player_in_ent_jug(df_jug, row, fecha_fifa):
    """
    Encuentra coincidencias de jugadores en un dataframe.
    :param df_jug: DataFrame que contiene los datos de los jugadores.
    :param nombre_jug_ent_part: Nombre del jugador a buscar.
    :param year_ent_part: Año o temporada en el que se requiere el jugador a buscar.
    :return: Diccionario con los datos del jugador coincidente. Las claves son 'edad', 'altura', 'overall_rating',
             'valor_mercado' y 'str_encont'.
    """
    # Definicion de variables a extraer
    match_data = {'edad': None, 'altura': None, 'overall_rating': None, 'valor_mercado': None, 'str_encont': None}

    # Funcion que hace una busqueda aproximada de un string en una columna
    def buscar_coincidencias(row, palabra, columna, umbral):
        return fuzz.token_set_ratio(palabra, row[columna]) >= umbral

    # Selecciono los datos de jugadores segun el fifa que se requiera
    df_jug_filt = df_jug[df_jug['fecha'] == fecha_fifa]  # df_jug_filt = df_jug[df_jug['fecha'].dt.year == year_ent_part]

    # Selecciono los datos de jugadores segun los nombres de jugadores mas parecidos al buscado
    df_jug_filt = df_jug_filt[df_jug_filt.apply(buscar_coincidencias, args=(row['nombre_jug'], 'nombre', 90), axis=1)]

    # Si encontro al menos un jugador con nombre similar
    if len(df_jug_filt) > 0:

        # Selecciono los datos de jugadores segun el nombre del equipo mas similar al buscado
        df_jug_filt = df_jug_filt[df_jug_filt.apply(buscar_coincidencias, args=(row['equipo_actual'], 'equipo_actual', 60), axis=1)]

        # Si encontro al menos un equipo con nombre similar
        if len(df_jug_filt) > 0:

            # Extraigo datos del jugador
            match_data['edad'] = df_jug_filt.iloc[0]['edad']
            match_data['altura'] = df_jug_filt.iloc[0]['altura']
            match_data['overall_rating'] = df_jug_filt.iloc[0]['overall_rating']
            match_data['valor_mercado'] = df_jug_filt.iloc[0]['valor_mercado']
            match_data['str_encont'] = df_jug_filt.iloc[0]['nombre']

    return match_data['edad'], match_data['altura'], match_data['overall_rating'], match_data['valor_mercado'], match_data['str_encont']

def search_fecha_actualizacion(df, fecha_part):

    # Si el partido es de Dic 2021, deberá escoger el fifa 22 puesto dic es posterior a Sept y luego como diciembre es anterior a ene, la primera actualizacion.
    # Si el partido es de Jun 2022, deberá escoger el fifa 22 puesto jun es anterior a Sept y luego como jun es posterio a ene, la ultima actualizacion.

    # Definicion de variables
    year_part = fecha_part.year  # e.g. "2021"

    # Si el partido se jugo de Julio a Diciembre (post mercado de pases de invierno)
    if fecha_part.month >= 7:  #if (mes_part > 7) or (mes_part == 7 and fecha_part.day > 20):

        year_str = str(year_part+1)[-2:]  # Ultimos dos "22"
        fifa_str = f"fifa {year_str}"  # FIFA 22

        # Si existe un fifa para dicha fecha
        if fifa_str in df['fifa'].unique():

            # Selecciono posibles fechas de actualizion segun el fifa
            l_posibles_fechas = df[df['fifa'] == fifa_str]['fecha'].unique() # [18 ago 2022, 16 Ago 2021]

            # Elijo la primera fecha de actualizacion (pues es si o si es anterior a enero)
            fecha = min(l_posibles_fechas)
            return fecha

    # Si el partido se jugo de Enero a Julio (post mercado de pases de verano)
    else:  # Jun 2021 entraria aqui

        year_str = str(year_part)[-2:]  # Ultimos dos "21"
        fifa_str = f"fifa {year_str}"  # FIFA 21

        # Si existe un fifa para dicha fecha
        if fifa_str in df['fifa'].unique():

            # Selecciono posibles fechas de actualizion segun el fifa
            l_posibles_fechas = df[df['fifa'] == fifa_str]['fecha'].unique()  # Cada elemento es del tipo numpy.datetime64... sin embargo, parece funcionar igual

            # Elijo la ultima fecha de actualizacion (pues si o si es posterior a enero)
            fecha = max(l_posibles_fechas)
            return fecha
    return None

def preparate_to_integrate(df_jug):

    # Format data: fecha y posesion
    df_jug['fecha'] = pd.to_datetime(df_jug['fecha'], format='%b %d, %Y')  # ya lo voy a extraer datetime...
    df_jug = convert_valor_mercado_to_int(df_jug)

    # Clean data
    # Hago limpieza de variables object antes de integrar para facilitar la integracion de datos
    df_jug = clean_data.prepare_text_columns(df_jug, l_col_to_except=['id'])  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

    # Normalizo valor de mercado para evitar el error en entrenamiento de "ValueError: Solver produced non-finite parameter weights. The input data may contain large values and need to be preprocessed."
    scaler = StandardScaler()  # Crea un objeto StandardScaler
    df_jug['valor_mercado'] = scaler.fit_transform(df_jug['valor_mercado'].values.reshape(-1, 1))

    return df_jug

def convert_valor_mercado_to_int(df):
    """
     Transforma el valor de mercado de string a float.

     :param df: Dataframe con columna 'valor_mercado' cuyos valores son un string, por ejemplo, '€1.2M'.
     :return: Dataframe con la columna 'valor_mercado' interpretada como float, por ejemplo, 1.200.000.
     """
    d = {'M': 1000000, 'K': 1000}

    def convertir_valor_mercado(valor_mercado_str):

        # Si no se tiene el dato del valor de mercado
        if valor_mercado_str == "€0":
            return None

        # Si se tiene el dato del valor de mercado
        else:
            for elem in d.keys():
                if elem in valor_mercado_str:
                    valor_mercado_int = float(valor_mercado_str.replace("€", "").replace(elem, "")) * d[elem]
                    return valor_mercado_int
            return None

    df['valor_mercado'] = df['valor_mercado'].apply(convertir_valor_mercado)
    return df

def prueba():
    start = time.time()
    print("\nIntegrando los datos...")

    pais = 'England'

    # Levanto datasets
    df_part_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_part_jug_cleaned.xlsx")
    df_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_jug_cleaned.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace falta el index_col=0
    print(df_part_jug.head(1))
    print(df_jug.head(1))

    # Integro datasets
    df_integrated = player_data_in_match(df_part_jug, df_jug)
    df_integrated.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=False)

    end = time.time()
    print(f"Integracion de datos en {(end - start) / 60:.1f} minutos")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()