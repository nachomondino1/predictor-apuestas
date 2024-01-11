# Juntar entidades jugadores, partido y atrib_jugadores
import pandas as pd
from fuzzywuzzy import fuzz
import warnings
import time
from tqdm import tqdm
from p3_data_preparation import clean_data
from sklearn.preprocessing import StandardScaler


def unique_players_df_part_jug(df_part_jug):

    set_unique_players = set()

    # Seleccionar todas las columnas excepto "id_part"
    columnas_sin_id_part = df_part_jug.columns[df_part_jug.columns != 'id_part']
    print(df_part_jug.shape)

    for col in columnas_sin_id_part:

        l_unique_players = df_part_jug[col].unique()
        set_unique_players.update(l_unique_players)

    df = pd.DataFrame(list(set_unique_players), columns=['nombre_jug'])
    df = df.dropna() # No se porque le queda un na
    # print(df.shape)
    # print(len(df['nombre'].unique()))
    return df

def unique_players_df_jug(df_jug):

    # Obtengo listado de nombres de los jugadores (sin repetidos)
    df_unique_players = df_jug.drop_duplicates(subset=['id_jugador'])

    # Selecciono solo las columnas id y nombre
    df_unique_players = df_unique_players.loc[:, ['id_jugador', 'nombre']]

    return df_unique_players

def integrate_players_by_name(df_part_jug, df_jug):

    print("Vinculando df_jug de Sofifa y df_part_jug de Flashscore...")

    # Creo copia del dataframe df_part_jug en el que agregar la columna "id_jugador"
    df_part_jug_with_id = df_part_jug.copy()
    l_umbrales = [90, 80, 75]
    # l1, l2 = [], []   # Prueba para definir l_umbrales

    # Preparo nombre de jugadores para facilitar integracion
    df_jug = clean_data.prepare_text_columns(df_jug, l_col_to_except=['id_jugador'])  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

    # Funcion que hace una busqueda aproximada de un string en una columna
    def buscar_coincidencias(row, palabra, columna, umbral):
        return fuzz.token_set_ratio(palabra, row[columna]) >= umbral

    # Por fila en df_part_jug
    for i, row in df_part_jug.iterrows():

        # Por umbral
        for umbral in l_umbrales:

            # Selecciono los datos de jugadores segun los nombres de jugadores mas parecidos al buscado
            df_jug_filt = df_jug[df_jug.apply(buscar_coincidencias, args=(row['nombre_jug'], 'nombre', umbral), axis=1)]

            if len(df_jug_filt) >= 1:

                # Prueba para definir l_umbrales
                # l1.append(len(df_jug_filt))
                # l2.append(umbral)

                # Agrego id_jugador de Sofifa como columna en df_part_jug
                df_part_jug_with_id.loc[i, 'id_jugador'] = df_jug_filt.id_jugador.values[0]
                # df_part_jug_with_id.loc[i, 'nombre_sofifa'] = df_jug_filt.nombre.values[0]  # temporalmente para analizar calidad de match

                # Elimino jugador de df_jug que hizo match para agilizar la busqueda
                df_jug = df_jug.drop(df_jug_filt.index[0])
                break

    # Prueba para definir l_umbrales
    # df = pd.DataFrame({"Cantidad de posibles match": l1, "Umbral": l2})
    # df.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=False)

    return df_part_jug_with_id

def reemplazar_name_por_id(df_part_jug, df_part_jug_with_id):

    # Por jugador
    for i, row in df_part_jug_with_id.iterrows():

        # Reemplazo su nombre por su id en df_part_jug
        df_part_jug = df_part_jug.replace(row['nombre_jug'], row['id_jugador'])

    return df_part_jug

def player_data_in_match(df_part, df_part_jug, df_jug):
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

    # Normalizo valor de mercado para evitar el error en entrenamiento de "ValueError: Solver produced non-finite parameter weights. The input data may contain large values and need to be preprocessed."
    scaler = StandardScaler()  # Crea un objeto StandardScaler
    df_jug['valor_mercado'] = scaler.fit_transform(df_jug['valor_mercado'].values.reshape(-1, 1))

    # Inicializo barra de progreso
    progress_bar = tqdm(total=len(df_part_jug), ncols=80)

    # Por titularidad (Titular, suplente o ausente)
    for titularidad in l_titularidad:

        # Por condicion (Local o visitante)
        for condicion in l_condicion:

            # Defino pattern y con el, selecciono las variables a procesar
            pattern = f'jug_{titularidad}_[a-z]*[_]*{condicion}_[0-9]+'  # CAMBIE EL PATTERN PARA QUE SUP Y SUP_ING SEAN PROCESADOS JUNTOS. VERIFICAR QUE FUNCIONA..
            l_col_to_preprocess = df_part_jug.filter(regex=pattern, axis=1).columns.tolist()  # LISTA DE COLUMNAS QUE CONTIENEN NOMBRES DE JUGADOR # con regex las que dicen jug... VER CODIGO DE UNO DE LOS PROYECTOS DE KAGGLE...
            print(f'Columnas a procesar: {l_col_to_preprocess}')

            # Por partido
            for i, row in df_part.iterrows():  # el i es igual si es df_part o df_part_jug

                progress_bar.update(1)

                # Busco el fifa correspondiente segun la fecha del partido
                fecha_part_fifa = search_fecha_actualizacion(df_jug, row['fecha'])  # Pasarle lista de posibles fechas en vez de df_jug...
                print(f' Partido Nº: {i} '.center(120, '#'))
                print(f"Fecha partido: {row['fecha']} --> Fecha Actualizacion Fifa: {fecha_part_fifa}")

                # Reinicio variables
                l_prom_edad, l_prom_alt, l_prom_rating, l_prom_valor = [], [], [], []

                # Por jugador
                for col_jug in l_col_to_preprocess:

                    # Busco id del jugador en df_part_jug
                    id_jug_ent_part = df_part_jug.loc[i, col_jug]
                    print(f"\t Id jugador a buscar en Sofifa: {id_jug_ent_part}")

                    # Si el jugador no es nan
                    # if isinstance(id_jug_ent_part, float) or isinstance(id_jug_ent_part, int) : # no se si es int o float

                    # Busco el id y la fecha en df_jug (Sofifa)
                    print(df_jug.shape)
                    df_jug_filt = df_jug[df_jug['id_jugador'] == id_jug_ent_part]
                    print(df_jug_filt.shape)
                    df_jug_filt = df_jug_filt[df_jug_filt['fecha'] == fecha_part_fifa]  # FALLA
                    print(df_jug_filt.shape)
                    # df_jug_filt = df_jug[(df_jug['id_jugador'] == id_jug_ent_part) & (df_jug['fecha'] == fecha_part_fifa)]
                    print(df_jug_filt)

                    # Guardo datos del jugador
                    l_prom_edad.append(df_jug_filt.edad.values[0])
                    l_prom_alt.append(df_jug_filt.altura.values[0])
                    l_prom_rating.append(df_jug_filt.overall_rating.values[0])
                    l_prom_valor.append( df_jug_filt.valor_mercado.values[0])
                    print("Ejemplo de lista promedio de edad: ", l_prom_edad)

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

    return df_part

def search_fecha_actualizacion(df, fecha_part): # FALLA PORQUE EL FIFA 24 CAMBIÓ DE NOMBRE Y AHORA ES FC 24... ACTUALIZAR
    """
    Segun la fecha del partido, busca la fecha de actualizacion del fifa correspondiente.
    :param df: df_jug
    :param fecha_part:
    :return:
    """
    # Si el partido es de Dic 2021, deberá escoger el fifa 22 puesto dic es posterior a Sept y luego como diciembre es anterior a ene, la primera actualizacion.
    # Si el partido es de Jun 2022, deberá escoger el fifa 22 puesto jun es anterior a Sept y luego como jun es posterio a ene, la ultima actualizacion.

    # Definicion de variables
    year_part = fecha_part.year  # e.g. "2021"

    # Si el partido se jugo de Julio a Diciembre (post mercado de pases de invierno)
    if fecha_part.month >= 7:  #if (mes_part > 7) or (mes_part == 7 and fecha_part.day > 20):

        year_str = str(year_part + 1)[-2:]  # Ultimos dos "22"

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


def prueba():
    start = time.time()
    print("\nIntegrando los datos...")
    pais = 'England'


    # Levanto datasets
    df_part = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_part_formated.xlsx")
    # df_part_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_part_jug.xlsx")
    df_jug = pd.read_excel(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_jug_formated.xlsx", index_col=0)  # A pesar de correr format_data con index=False, hace falta el index_col=0
    # print(df_part.head(1))
    # print(df_part_jug.head(1))
    # print(df_jug.head(1))

    """
    # Obtengo listado unicos de jugadores en df_jug (Sofifa) y df_part_jug (Flashscore) para agilizar vinculacion
    df_part_jug_unique_players = unique_players_df_part_jug(df_part_jug)
    df_jug_unique_players = unique_players_df_jug(df_jug)

    # Vinculo con "id_jugador" a df_jug (Sofifa) y df_part_jug (Flashscore) utilizando los nombres de los jugadores
    df_part_jug_vinc_df_jug = integrate_players_by_name(df_part_jug_unique_players, df_jug_unique_players)
    df_part_jug_vinc_df_jug.to_excel('/Users/nachomondino/Desktop/df_part_jug_vinc_df_jug.xlsx', index=False)

    # Reemplazo los nombres de los jugadores por su id en df_part_jug (Flashscore)
    df_part_jug = reemplazar_name_por_id(df_part_jug, df_part_jug_vinc_df_jug)
    df_part_jug.to_excel('/Users/nachomondino/Desktop/df_part_jug_with_id.xlsx', index=False)
    """

    df_part_jug = pd.read_excel('/Users/nachomondino/Desktop/df_part_jug_with_id.xlsx')  # Pruebas de funcion de abajo
    # print(df_part_jug.head(1))

    # Sintetizar la data de df_jug (Sofifa) en df_part (Flashscore) gracias al vinculo con df_part_jug (Flashscore) -->   Aca dentro hago esto:  # Traer fecha, equipo y no se que mas de df_part (Flashscore) y agregar a df_part_jug (Flashscore) para poder saber en que momento traer la info del jugador (Sofifa tiene varias veces un mismo jugador porque es el jugador en ≠ fifas)
    df = player_data_in_match(df_part, df_part_jug, df_jug)
    df.to_excel('/Users/nachomondino/Desktop/df_integrated_prueba.xlsx', index=False)

    end = time.time()
    print(f"Integracion de datos en {(end - start) / 60:.1f} minutos")

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()