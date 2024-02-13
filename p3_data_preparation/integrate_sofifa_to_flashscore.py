# Juntar entidades jugadores, partido y atrib_jugadores
import pandas as pd
from fuzzywuzzy import fuzz
import warnings
import time
from tqdm import tqdm
import math


def unique_players_df_part_jug(df_part_jug):
    """
    Obtencion de listado de jugadores unicos de df_part_jug. Un mismo jugador se repite varias veces porque esta en mas
    de un partido.
    :param df_part_jug: Dataframe. Unidad de analisis: partido. Columnas: una por jugador segun formaciones.
    :return: Dataframe. Unidad de analisis: jugador. Una sola columna con nombres de los jugadores en df_part_jug
    (sin repetidos).
    """
    # Definicion de variables
    set_unique_players = set()

    # Seleccionar todas las columnas excepto "id_part"
    l_columnas_sin_id_part = df_part_jug.columns[df_part_jug.columns != 'id_part']

    # Por columna (e.g. jug_tit_loc_7)
    for col in l_columnas_sin_id_part:

        # Obtengo jugadores unicos y agrego al set
        l_unique_players = df_part_jug[col].unique()
        set_unique_players.update(l_unique_players)  #  Uso set puesto que un jugador puede estar en mas de una columna

    # Creo dataframe con listado de jugadores unicos
    df = pd.DataFrame(list(set_unique_players), columns=['nombre_jug'])
    df = df.dropna() # No se porque le queda un na
    return df

def unique_players_df_jug(df_jug):
    """
    Obtencion de listado de jugadores unicos de df_jug. Un mismo jugador se repite varias veces porque esta en mas
    de un fifa.
    :param df_jug: Dataframe. Unidad de analisis: jugador.
    :return: Dataframe. Unidad de analisis: jugador. Columnas id_jugador y nombre. La columna "nombre" tiene los nombres
    de los jugadores en df_jug (sin repetidos).
    """
    # Obtengo listado de nombres de los jugadores (sin repetidos)
    df_unique_players = df_jug.drop_duplicates(subset=['id_jugador'])

    # Selecciono solo las columnas id y nombre
    df_unique_players = df_unique_players.loc[:, ['id_jugador', 'nombre']]
    return df_unique_players

def integrate_players_by_name(df_part_jug_up, df_jug_up):
    """
    Vinculo datasets de los jugadores de Sofifa y los jugadores de Flashscore segun nombre de jugador.
    :param df_part_jug_up: Dataframe. Unidad de analisis: jugador. Una sola columna con nombre de los jugadores en
    df_part_jug (sin repetidos).
    :param df_jug_up: Dataframe. Unidad de analisis: jugador. Columnas id_jugador y nombre. La columna "nombre" tiene los
    nombres de los jugadores en df_jug (sin repetidos).
    :return: Dataframe. Unidad de analisis: jugador. df_part_jug pasado como parametro con columna "id_jugador" de
    df_jug gracias a voncular nombres de jugadores de sendos dataframes.
    """
    print("Vinculando df_jug de Sofifa y df_part_jug de Flashscore...")

    # Definicion de variables
    df_part_jug_with_id = df_part_jug_up.copy()  # Creo copia del dataframe df_part_jug en el que agregar la columna "id_jugador"
    l_umbrales = [95, 90, 85, 80, 75]
    n_pos_matchs, n_matchs = len(df_part_jug_up), 0

    # Funcion que hace una busqueda aproximada de un string en una columna
    def buscar_coincidencias(row, palabra, columna, umbral):
        return fuzz.token_set_ratio(palabra, row[columna]) >= umbral

    # Por jugador en df_part_jug
    for i, row in df_part_jug_up.iterrows():

        # Filtro inicial. Me quedo con los jugadores con nombre mas parecido (agiliza enormemente la funcion)
        df_jug_filt_ini = df_jug_up[df_jug_up.apply(buscar_coincidencias, args=(row['nombre_jug'], 'nombre', min(l_umbrales)), axis=1)]

        # Por umbral
        for umbral in l_umbrales:

            # Selecciono los jugadores con nombre mas parecido al buscado
            df_jug_filt = df_jug_filt_ini[df_jug_filt_ini.apply(buscar_coincidencias, args=(row['nombre_jug'], 'nombre', umbral), axis=1)]

            # Si hay al menos un posible match
            if len(df_jug_filt) >= 1:

                # Agrego id_jugador de Sofifa como columna en df_part_jug
                df_part_jug_with_id.loc[i, 'id_jugador'] = df_jug_filt.id_jugador.values[0]
                # df_part_jug_with_id.loc[i, 'nombre_sofifa'] = df_jug_filt.nombre.values[0]  # temporalmente para analizar calidad de match

                # Elimino jugador de df_jug que hizo match para agilizar la busqueda
                df_jug_up = df_jug_up.drop(df_jug_filt.index[0])
                n_matchs += 1
                break
    
    try:
        print(f"De los {n_pos_matchs} jugadores en df_part_jug, hizo match para {n_matchs/n_pos_matchs*100:.2f}% de ellos, es decir, para {n_matchs}.")
    except ZeroDivisionError:
        print("No se cuenta con las formaciones de ningun partido de df_part, por ende, df_part_jug no tiene que integrar a df_part.")

    return df_part_jug_with_id

def reemplazar_name_por_id(df_part_jug, df_part_jug_with_id):
    """
    Reemplazo nombre de jugadoores en df_part_jug por su id de manera de facilitar la integracion posterior
    :param df_part_jug: Dataframe. Unidad de analisis: partido. Columnas: una por jugador segun formaciones.
    :param df_part_jug_with_id: Dataframe. Unidad de analisis: jugador. Columnas: id_jugador y nombre.
    :return: Dataframe. df_part_jug pasado como parametro reemplazando los nombres de los jugadores por su id.
    """
    # Por jugador
    for i, row in df_part_jug_with_id.iterrows():

        # Reemplazo su nombre por su id en df_part_jug
        df_part_jug = df_part_jug.replace(row['nombre_jug'], row['id_jugador'])

    return df_part_jug

def player_data_in_match(df_part, df_part_jug, df_jug):
    """
    Integra la entidad jugador en la entidad partido. Es decir, sintetiza los datos de los jugadores a cada partido en
    particular. Se determinan los promedios de edad, overall rating, valor de mercado y altura del equipo titular,
    suplente y los ausentes para cada equipo.
    :param df_part: Dataframe. Unidad de analisis: partido. Columnas: equipos, arbitros, estadisticas del partido, etc.
    :param df_part_jug: Dataframe. Unidad de analisis: partido. Columnas: id_part y una por jugador segun formaciones.
    Celdas: id de jugador (en vez de nombre).
    :param df_jug: Dataframe. Unidad de analisis: jugador. Columnas: id_jugador y datos del jugador como edad y overall
    rating.
    :return: Dataframe. Dataframe con los datos de todos los dataframes pasados como parametro. Tod@ en un solo
    dataframe para poder entrenar un modelo con ellos.
    """
    # Definicion de variables
    l_titularidad = ['tit', 'sup', 'aus']  # tengo que agregar 'sup_ing' pero se debe procesar con sup...
    l_condicion = ['loc', 'vis']
    warnings.filterwarnings('ignore')  # Ver el ignore, y solucionarlo en vez de ignorarlo...

    # Agrego columna "fifa_year" quedandome solo con el año del fifa (e.g. "22" en vez de "FIFA 22")
    df_jug['fifa_year'] = df_jug['fifa'].str.split(' ').str[-1]

    # Por titularidad (Titular, suplente o ausente)
    for titularidad in l_titularidad:

        # Por condicion (Local o visitante)
        for condicion in l_condicion:

            # Defino pattern y con el, selecciono las variables a procesar
            pattern = f'jug_{titularidad}_[a-z]*[_]*{condicion}_[0-9]+'  # CAMBIE EL PATTERN PARA QUE SUP Y SUP_ING SEAN PROCESADOS JUNTOS. VERIFICAR QUE FUNCIONA..
            l_col_to_preprocess = df_part_jug.filter(regex=pattern, axis=1).columns.tolist()  # LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...
            print(f'Columnas a procesar: {l_col_to_preprocess}')

            # Inicializo barra de progreso
            progress_bar = tqdm(total=len(df_part_jug), ncols=80)

            # Por partido
            for i, row in df_part.iterrows():

                progress_bar.update(1)
                # print(f' Partido Nº: {i} '.center(120, '#'))

                # Busco el fifa correspondiente segun la fecha del partido
                year_fifa = search_fecha_fifa(row['fecha'])
                # print(f"Fecha partido: {row['fecha']} --> Fifa a buscar: {fecha_part_fifa}")

                # Reinicio variables
                l_prom_edad, l_prom_alt, l_prom_rating, l_prom_valor = [], [], [], []

                # Por jugador
                for col_jug in l_col_to_preprocess:

                    # Busco id del jugador en df_part_jug
                    id_jug_ent_part = df_part_jug.loc[i, col_jug]
                    # print(f"\t Id jugador a buscar en Sofifa: {id_jug_ent_part}")

                    # Si el id_jugador no es nan
                    if not math.isnan(id_jug_ent_part):  # hay mucho nan sobretodo columnas de jugadores ausentes (e.g. jug_aus_vis_12)

                        # Busco el id y la fecha en df_jug (Sofifa)
                        df_jug_filt = df_jug[df_jug['id_jugador'] == id_jug_ent_part]
                        df_jug_filt = df_jug_filt[df_jug_filt['fifa_year'] == year_fifa]

                        # Guardo datos del jugador
                        if len(df_jug_filt) > 0:
                            l_prom_edad.append(df_jug_filt.edad.values[0])
                            l_prom_alt.append(df_jug_filt.altura.values[0])
                            l_prom_rating.append(df_jug_filt.overall_rating.values[0])
                            l_prom_valor.append( df_jug_filt.valor_mercado.values[0])
                            # print("Ejemplo de lista promedio de edad: ", l_prom_edad)

                # Guardo promedios de edad, altura, overall_rating y valor de mercado
                try:
                    df_part.loc[i, f'prom_edad_jug_{titularidad}_{condicion}'] = sum(l_prom_edad) / len(l_prom_edad)
                    df_part.loc[i, f'prom_alt_jug_{titularidad}_{condicion}'] = sum(l_prom_alt) / len(l_prom_alt)
                    df_part.loc[i, f'prom_rat_jug_{titularidad}_{condicion}'] = sum(l_prom_rating) / len(l_prom_rating)
                    df_part.loc[i, f'prom_valor_jug_{titularidad}_{condicion}'] = sum(l_prom_valor) / len(l_prom_valor)

                    # Calculo nro de jugadores lesionados
                    if titularidad == 'aus':
                        df_part.loc[i, f'n_jug_{titularidad}_{condicion}'] = len(l_prom_rating)
                        # print("Numero de ausentes: ", len(l_prom_rating))

                except ZeroDivisionError:
                    # print("Aparentemente no hay datos de jugadores para el partido")
                    pass

            # Cerrar la barra de progreso al finalizar
            progress_bar.close()

    return df_part

def search_fecha_fifa(fecha_part):
    """
    Dado la fecha de un partido, busco el fifa que le corresponde.
    :param fecha_part: Datetime. Fecha del partido.
    :return: String. Año del fifa que corresponde segun la fecha pasada como parametro (e.g. partido del 23/12/2023 le
    corresponde FIFA 24)
    """
    # Definicion de variables
    year_part = fecha_part.year  # e.g. "2021"

    # Si el partido se jugo de Julio a Diciembre (post mercado de pases de invierno)
    if fecha_part.month >= 7:

        year_part_str = str(year_part + 1)[-2:]  # Ultimos dos "22"
        return year_part_str

    # Si el partido se jugo de Enero a Julio (post mercado de pases de verano)
    else:
        year_part_str = str(year_part)[-2:]  # Ultimos dos "21"
        return year_part_str

def prueba():
    start = time.time()
    print("\nIntegrando los datos...")
    pais = 'England'

    # Levanto datasets
    df_part = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_part_formated.xlsx")
    df_part_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_part_jug_formated.xlsx")
    df_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_jug_formated.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace falta el index_col=0
    print(f"df_part: \n{df_part.head(1)} \n\ndf_part_jug: \n{df_part_jug.head(1)} \n\n df_jug: \n{df_jug.head(1)}")

    # Obtengo listado de jugadores unicos tanto en df_jug (Sofifa) como en df_part_jug (Flashscore) para agilizar vinculacion
    df_part_jug_unique_players = unique_players_df_part_jug(df_part_jug)
    df_jug_unique_players = unique_players_df_jug(df_jug)

    # Vinculo con "id_jugador" a df_jug (Sofifa) y df_part_jug (Flashscore) utilizando los nombres de los jugadores
    df_part_jug_vinc_df_jug = integrate_players_by_name(df_part_jug_unique_players, df_jug_unique_players)
    # df_part_jug_vinc_df_jug.to_excel('/Users/nachomondino/Desktop/df_part_jug_vinc_df_jug.xlsx', index=False)

    # Reemplazo los nombres de los jugadores por su id en df_part_jug (Flashscore)
    df_part_jug = reemplazar_name_por_id(df_part_jug, df_part_jug_vinc_df_jug)
    # df_part_jug.to_excel('/Users/nachomondino/Desktop/df_part_jug_with_id.xlsx', index=False)

    # Sintetizar la data de df_jug (Sofifa) en df_part (Flashscore) gracias al vinculo con df_part_jug (Flashscore) -->   Aca dentro hago esto:  # Traer fecha, equipo y no se que mas de df_part (Flashscore) y agregar a df_part_jug (Flashscore) para poder saber en que momento traer la info del jugador (Sofifa tiene varias veces un mismo jugador porque es el jugador en ≠ fifas)
    df = player_data_in_match(df_part, df_part_jug, df_jug)
    df.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=False)

    end = time.time()
    print(f"Integracion de datos en {(end - start) / 60:.1f} minutos")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()