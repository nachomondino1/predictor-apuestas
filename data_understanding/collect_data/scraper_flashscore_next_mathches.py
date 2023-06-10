# Importo librerias
import pandas as pd
import datetime
import time
import warnings
import random
import pickle
from dspy.data_understanding.web_scraping.selenium import Crawler
from data_preparation import format_data, integrate_data, construct_data, select_data, clean_data
from dspy.data_understanding.describe_data import getting_to_know_data


def extract_flashscore(n_dias_max):
    """
    It contains all the extraction logic, i.e. it directs the bot on WHEN to perform each action. First initialize the
    driver, then enter the page, then accept cookies and so on.
    """
    # DEFINCION DE PARAMETROS & VARIABLES
    warnings.filterwarnings(
        "ignore")  # /Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/scraper_flashscore.py:175: FutureWarning: In a future version, object-dtype columns with all-bool values will not be included in reductions with bool_only=True. Explicitly cast to bool dtype instead. df_part = pd.concat([df_part, pd.DataFrame(d_nueva_fila, index=[0])])
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    fecha_act = datetime.datetime.now()
    crawler = Crawler(headless=True, path=None)  # Creo objeto de clase CrawlerActions()
    d_formaciones = {'Formaciones iniciales': 'tit', 'Suplentes': 'sup', 'Jugadores reemplazados': 'sup_ing',
                     'Jugadores ausentes': 'aus'}
    df_comp = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_competicion.xlsx')
    df = pd.DataFrame()  # No hace falta definir columnas por mas que no haya extraido partidos

    # POR PAIS
    for pais in df_comp['pais'].unique()[:1]:  # Solo arg?
        pais = 'argentina'

        # Filtro competiciones por pais
        # df_comp_pais = df_comp[df_comp['pais'] == pais]
        print(f' PAIS: {pais} '.center(120, '#'))
        l_comp = ['liga profesional']
        l_cat = ['Liga']

        # POR COMPETICION
        # for competicion, categoria in zip(df_comp_pais['nombre'], df_comp_pais['categoria']):
        for competicion, categoria in zip(l_comp, l_cat):

            # Obtengo datos de la competicion
            print(f' Competicion: {competicion} '.center(120, '+'))
            competicion_form = competicion.replace(" ", "-")  # formateo competicion para las rutas de archivo y urls

            # Ingreso a pagina
            # url = f'https://www.flashscore.es/futbol/{pais}/{competicion_form}/partidos/'
            url = f'https://www.flashscore.es/futbol/{pais}/{competicion_form}/resultados/'
            crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...
            print(url)

            # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
            crawler.click_boton(xpath='.//button[@id="onetrust-accept-btn-handler"]')

            # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
            # crawler.click_boton(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=SEC_WAIT_LONG*3, repeat_click=True)  # Si hace click, es None. Si falla, es un str  # A veces me tira (y no entiendo por qué): selenium.common.exceptions.StaleElementReferenceException: Message: stale element reference: stale element not found

            # Obtengo temporada
            # No tiene sentido extraer todas las temporadas puesto que solo necesito la ultima...
            temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)

            # Extraigo partidos (items) y sus ids
            l_items = crawler.extract_tags(xpath='.//div[@class="sportName soccer"]//div[@title="¡Haga click para detalles del partido!"]')
            l_ids = [item.get_attribute('id') for item in l_items]
            print(f"Cantidad de items (partidos): {len(l_items)}")

            # POR PARTIDO (c/u identificado con un id)
            cont_part = 0
            for id in l_ids:

                start = time.time()

                # Quito lo que no es del id
                id = id[id.rfind('_') + 1:]  # paso de "g_1_fshvzbls" a "fshvzbls"

                # Reinicio diccionario en el que guardar la nueva fila
                d_nueva_fila = {'id': id, 'competicion': competicion, 'temporada': temp_year, 'pais': pais,
                                'es_copa': 1 if categoria == "Copa" else 0}
                cont_part += 1
                print(f" Partido {cont_part} de {len(l_items)}. Recolectado el {cont_part / len(l_items) * 100:.0f}% ".center(120, "."))

                # Ingreso a pagina de informacion del partido
                crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

                # EXTRACCION DE CAMPOS
                fecha_str = crawler.extract_tag(xpath='.//div[@class="duelParticipant__startTime"]', text=True, sec_wait=SEC_WAIT_LONG)
                fecha_dt = datetime.datetime.strptime(fecha_str, "%d.%m.%Y %H:%M")
                print(fecha_dt, fecha_act)
                dif_fecha = (fecha_dt - fecha_act).days  # Ojo que si falta 1 dia y 23 hs, lo toma como 1...
                print(dif_fecha)

                # Si el partido aun no se jugo (extraia partidos de la Copa de la Liga profesional 2023 la cual aun no se jugo pero ya esta el fixture... tampoco es tan grave solo esta la jornada 1)
                if dif_fecha < n_dias_max:

                    # Extraigo campos de hoja "Resumen"
                    d_nueva_fila['fecha'] = fecha_dt
                    d_nueva_fila['equipo_loc'] = crawler.extract_tag(
                        xpath='.//div[starts-with(@class, "duelParticipant__home")]', text=True, sec_wait=SEC_WAIT)
                    d_nueva_fila['equipo_vis'] = crawler.extract_tag(
                        xpath='.//div[starts-with(@class, "duelParticipant__away")]', text=True, sec_wait=SEC_WAIT)
                    d_nueva_fila['arbitro'] = crawler.extract_tag(
                        xpath='.//div[@class="mi__data"]//span[contains(text(), "Árbitro")]/following-sibling::span',
                        text=True, sec_wait=SEC_WAIT)
                    d_nueva_fila['cancha'] = crawler.extract_tag(
                        xpath='.//div[@class="mi__data"]//span[contains(text(), "Estadio")]/following-sibling::span',
                        text=True, sec_wait=SEC_WAIT)

                    # No tiene sentido extraer estadisticas puesto que nunca abrá de un partido que aun no se jugo...

                    # Extraigo campos de hoja "Formaciones" --> Extriago de algun diario? https://www.ole.com.ar/futbol-primera/estudiantes-vs-barracas-central-hora-tv-posibles-formaciones-liga-profesional-2023_0_QdU0rRnj59.html
                    if crawler.click_boton(xpath='.//div[@class="tabs tabs__detail--nav"]//a[text()="Formaciones"]', sec_wait=SEC_WAIT) is not False:

                        time.sleep(random.uniform(SEC_WAIT+3, SEC_WAIT_LONG+3)) # Por posible falla en el primer campo a extraer

                        # Por seccion ("Formacion inicial", "Suplentes" y  "Ausentes")
                        for formacion in d_formaciones.keys():

                            # Si existe la seccion "Formaciones"
                            if crawler.extract_tag(xpath=f'.//div[text()="{formacion}"]', sec_wait=SEC_WAIT_LONG) is not None:

                                # Extraigo listado de jugadores
                                l_tags_jug_loc = crawler.extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][1]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT*2)  # Es lista de tags o None
                                l_tags_jug_vis = crawler.extract_tags(xpath=f'.//div[text()="{formacion}"]//following-sibling::div//div[@class="lf__side"][2]//*[@class="lf__participantName"]', sec_wait=SEC_WAIT*2)  # Es lista de tags o None

                                # Creo una columna por cada jugador local
                                if l_tags_jug_loc is not None:
                                    for i in range(len(l_tags_jug_loc)):
                                        # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "Genez Nahuel")
                                        try:
                                            url_with_nombre = l_tags_jug_loc[i].get_attribute('href')  # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                                            nombre = url_with_nombre.split('/')[4].replace('-', " ")
                                        # Extraigo nombre corto (e.g. "N.genez")
                                        except:
                                            nombre = l_tags_jug_loc[i].text

                                        # Extraigo nombre
                                        d_nueva_fila[f'jug_{d_formaciones[formacion]}_loc_{i+1}'] = nombre  # l_tags_jug_loc[i].text

                                # Creo una columna por cada jugador visitante
                                if l_tags_jug_vis is not None:
                                    for i in range(len(l_tags_jug_vis)):
                                        # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "genez nahuel")
                                        try:
                                            url_with_nombre = l_tags_jug_vis[i].get_attribute('href')  # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                                            nombre = url_with_nombre.split('/')[4].replace('-', " ")
                                        # Extraigo nombre corto (e.g. "N.genez")
                                        except:
                                            nombre = l_tags_jug_vis[i].text

                                        d_nueva_fila[f'jug_{d_formaciones[formacion]}_vis_{i+1}'] = nombre  # l_tags_jug_vis[i].text

                        # Extraigo entrenadores  # Sacar los try - except horribles
                        # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "genez nahuel")
                        try:
                            url_with_nombre = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]//a[@class="lf__participantName"]',  attribute='href', sec_wait=SEC_WAIT) # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                            d_nueva_fila['dt_loc'] = url_with_nombre.split('/')[4].replace('-', " ")
                        # Extraigo nombre corto (e.g. "N.genez")
                        except:
                            d_nueva_fila['dt_loc'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][1]', text=True, sec_wait=SEC_WAIT)

                        # Intento extraer nombre completo (solo si tiene link asociado) (e.g. "genez nahuel")
                        try:
                            url_with_nombre = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]//a[@class="lf__participantName"]', attribute='href', sec_wait=SEC_WAIT)  # href="https:/www.flashscore.com.ar/jugador/genez-nahuel/WtErcTOs/"
                            d_nueva_fila['dt_vis'] = url_with_nombre.split('/')[4].replace('-', " ")
                        # Extraigo nombre corto (e.g. "N.genez")
                        except:
                            d_nueva_fila['dt_vis'] = crawler.extract_tag(xpath='.//div[text()="Entrenadores"]//following-sibling::div//div[@class="lf__side"][2]',text=True, sec_wait=SEC_WAIT)

                    # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                    if crawler.extract_tag(
                            xpath='.//div[@class="oddsRow"]') is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...

                        d_nueva_fila['odds_loc'] = extract_cuota(crawler, SEC_WAIT, i=1)
                        d_nueva_fila['odds_emp'] = extract_cuota(crawler, SEC_WAIT, i=2)
                        d_nueva_fila['odds_vis'] = extract_cuota(crawler, SEC_WAIT, i=3)

                    # GUARDADO DE DATOS EN DATAFRAME
                    df = pd.concat([df, pd.DataFrame(d_nueva_fila, index=[0])])
                    print(df.iloc[-1])

                    end = time.time()
                    print(f"Partido recolectado en {(end - start):.1f} segundos")

                # Guardo partidos no extraidos de la temporada (por seguridad)
                # df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/{competicion_form}_{temp_year.replace("/","_")}_{pais}.xlsx', index=False)

        # Guardado de archivo excel en computadora
        # df.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/entidad_next_partido_{pais}.xlsx',index=False)
        df.to_excel('/Users/nachomondino/Desktop/entidad_next_partido.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df

def extract_cuota(crawler, SEC_WAIT, i):  # Puedo volver a la anterior, solo fallaron 41 cuotas por "Cuotas retiradas por la casa de apuestas."

    # Busco odd suponiendo que cambio durante el partido
    try:
        odds_str = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]', attribute='title',
                                       sec_wait=SEC_WAIT)  # 3.00 » 2.25
        cuota = float(odds_str.split('»')[0].strip())  # 3.00

    # Si la odd no cambio durante el partido, o bien, aparece "Cuotas retiradas por la casa de apuestas."
    except ValueError:  # could not convert string to float:

        try:
            odds_str = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]//span[@class="oddsValueInner"]',
                                           text=True, sec_wait=SEC_WAIT)
            cuota = float(odds_str)

        except (ValueError, TypeError):
            print("Fallo extraccion de la cuota")
            cuota = None
    return cuota

class DataPreparation:  # 17.4 min

    def __init__(self, var_resp):
        self.var_resp = var_resp

    def clean_data(self, df_part=None, export=False):  # 0.0 min
        """
        Limpia los datos de un dataframe.
        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe limpiado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe limpiado. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df_part = pd.read_excel('./data_preparation/data/df_part_formated.xlsx') if df_part is None else df_part

        print("\nLimpiando los datos...")

        # Hago limpieza de datos antes de integrar para facilitar la integracion de datos
        df_part = clean_data.prepare_text_columns(df_part, l_col_to_except=['id', 'temporada'])  # df_part = clean_data.prepare_text_columns(df_part)  # Nombre de equipos minuscula, sin acentos y sin caracteres especiales

        # Remuevo strings adicionales en los nombres de los equipos
        df_part = clean_data.clean_teams_names(df_part)

        if export:
            df_part.to_excel('/Users/nachomondino/Desktop/df_part_cleaned_next_matches.xlsx', index=False)

        return df_part

    def integrate_data(self, df_part=None, df_jug=None, export=False):  # 13.3 min (sin copa arg y otras comp)
        """
        Integra los datos de partidos y jugadores en un solo dataframe.
        :param df_part: Dataframe de los datos de los partidos. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_part_cleaned.xlsx') if df_part is None else df_part
        df_jug = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_jug_cleaned.xlsx') if df_jug is None else df_jug

        start = time.time()
        print("\nIntegrando los datos...")

        # Integro entidad partido y jugador
        df_integrated = integrate_data.player_data_in_match(df_part, df_jug)

        end = time.time()
        print(f"Integracion de datos en {(end - start) / 60:.1f} minutos")

        if export:
            df_integrated.to_excel('/Users/nachomondino/Desktop/df_integrated_next_matches.xlsx', index=False)

        return df_integrated

    def construct_data(self, df=None, N_ULT_PART=5, export=False):  # 2.7 minutos
        """
        Construye nuevos datos a partir de un dataframe existente.
        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_integrated.xlsx') if df is None else df

        # Construyo variables historicas
        l_estad_part = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases',
                        'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
        df = construct_data.historial_entre_si_segun_localia(df, n_ult_part=int(N_ULT_PART / 2))
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART,l_var=l_estad_part)  # Estadisticas del partido
        df = construct_data.promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Diferencia de gol
        df = construct_data.forma_reciente(df, n_part=N_ULT_PART)  # Rendimiento del equipo
        df = construct_data.n_dias_ult_partido(df)  # Numero de dias desde ultimo partido

        # Construyo variables de diferencias para las variables promedio de los jugadores
        df = construct_data.calculate_dif_col_jugadores(df)  # No tengo datos de jugadores...  df[nombre_col_dif] = df[nombre_col_loc] - df[nombre_col_vis]  KeyError: 'prom_edad_jug_tit_loc'

        if export:
            df.to_excel('/Users/nachomondino/Desktop/df_constructed_next_matches.xlsx', index=False)

        return df

    def select_data(self, df=None, treat_nan='drop', export=False):  # 1.3 minutos
        """
        Selecciona las variables relevantes del dataframe.
        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_constructed.xlsx') if df is None else df

        warnings.filterwarnings('ignore')
        print("\nSeleccionado datos...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df.index = df['id']
        df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df = format_data.convert_columns_to_int(df)

        # Selecciono las variables mas importantes (feature selection) --> Levanto df?
        df_test = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/modeling/data/df_test.xlsx')
        l_selected_features = df_test.columns
        df = df.loc[:, l_selected_features]
        print(df.shape)
        print(df.columns)

        # Tratamiento de NaN values (drop, fillna con moda, fillna con random forest)
        df = clean_data.treat_nan_values(df, type=treat_nan)

        if export:
            df.to_excel('/Users/nachomondino/Desktop/df_part_selected_next_matches.xlsx', index=False)

        return df

def rellenar_player_data(df_part, df_part_old):
    # Tal vez, para no rellenar automaticamente las variables de jugadores (como dif_rat_tit, dif_edad_sup, dif_rat_aus)
    # por no tener las formaciones antes del partido, podria tomar el rating de cada equipo segun su ultimo partido?
    # Y considerar bajas?

    l_var = ['dif_rat_tit', 'dif_edad_sup', 'dif_rat_aus']

    # Por partido
    for i in range(len(df_part)):

        l_equipos = [df_part.loc[i, 'equipo_loc'], df_part.loc[i, 'equipo_vis']]

        # Por equipo
        for equipo in l_equipos:

            # Busco el ultimo partido del equipo  # Suponiendo que df_part_old esta ordenado decrecientemente
            indice = df_part_old[(df_part_old['equipo_loc'] == equipo) | (df_part_old['equipo_vis'] == equipo)].index[0]

            df_part_old
            # Buscar ultimo partido del equipo
            # En dicho partido, extraer: ['dif_rat_tit', 'dif_edad_sup', 'dif_rat_aus']

            # Guardar ['dif_rat_tit', 'dif_edad_sup', 'dif_rat_aus'] en nuevo partido...
    pass

def prueba():

    # Definicion de variables
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    prepare = DataPreparation(var_resp)

    # Hiperparametros
    n_dias_a_prox_part = 3  # Numero de dias maximo para partido a recolectar
    n_anios_df_part = 7  # Numero de ultimos años a tomar de los partidos ya recolectados (si es muy bajo, por ej 3, no llega a construir la variable "historial_entre_si" pues hay un numero de part min...

    N_ULT_PART = 5  # Numero de partidos a tener en cuenta para variables historicas como posesion en ult partidos
    treat_nan = 'fillna_with_ml' # Tratamiento de nan values: dropna, fillna_with_mode, fillna_with_ml
    export = True

    ## DATA UNDERSTANDING
    print(" Data Understanding ".center(120, "#"))
    '''
    # Collect initial data
    try:
        df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data/argentina/entidad_next_partido.xlsx')
    except:
        df_part = extract_flashscore(n_dias_max=n_dias_a_prox_part)
    '''

    # Levanto dataset con los ultimos 15 partidos de la liga argentina
    df_part = pd.read_excel('/Users/nachomondino/Desktop/entidad_next_partido.xlsx')
    # l_ids = list(df_part['id'])

    # Describe data
    getting_to_know_data(df_part)

    ## DATA PREPARATION
    print(" Data preparation ".center(120, "#"))

    # no le hago format porque ya extraigo la fecha en formato datetime, la copa como 1 o 0 y no tengo posesion_loc ni posesion_vis
    df_part = prepare.clean_data(df_part, export=export)
    df_part = prepare.integrate_data(df_part, export=export)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc

    '''
    # TENGO QUE LEVANTAR DF_PART VIEJO PARA PODER CONSTRUIR VARIABLES HISTORICAS... --> lo voy a hacer dentro de construct_data...
    # Levanto dataset de partidos viejos (pues los necesito para construir variables historicas)  Y # Filtrar el DataFrame para seleccionar los registros dentro de los últimos 3 años # no deberia levantar todos los registros... es solo los ultumos 5 de cada equipo...
    df_part_old = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_integrated.xlsx')  # Lo tengo que levantar integrado para tener las variables de jugadores... como "prom_edad_jug_tit_loc"
    fecha_limite = datetime.datetime.now() - datetime.timedelta(days=365 * n_anios_df_part)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual
    df_part_old_filt = df_part_old[df_part_old['fecha'] >= fecha_limite]

    # Creo variable equipo_ganador para que poder calcular historial_entre_si y forma_reciente
    df_part_old_filt = construct_data.determinar_equipo_ganador(df_part_old_filt)  # --> a df_part no le construyo equipo_ganador...

    rellenar_player_data(df_part)

    # Agrego dataframe viejo para poder calcular variables historicas...
    df = pd.concat([df_part_old_filt, df_part], axis=0).reset_index(drop=True)  # Funciona bien
    
    

    df = prepare.construct_data(df, N_ULT_PART=N_ULT_PART, export=False)

    df = prepare.select_data(df, treat_nan=treat_nan, export=False)

    # Vuelvo a seleccionar solo los partidos a predecir  (despues de select para poder hacer treat_nan con ml basandome en los partidos viejos...)
    df = df[df.index.isin(l_ids)] # df = df[df.id.isin(l_ids)]  # Funciona... shape = (5, 119)
    print(df.shape)

    df = df.drop('equipo_ganador', axis=1)
    df.to_excel('/Users/nachomondino/Desktop/entidad_partido_argentina_next_matches.xlsx', index=False)

    ## Modeling  --> solo tengo que predecir... y despues analizar los resultados una vez concluida la fecha...
    df_test_without_odds = df.copy().drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)
    loaded_model = pickle.load(open("/Users/nachomondino/Documents/GitHub/predictor-apuestas/modeling/data/modelo.pkl", "rb"))
    y_pred = loaded_model.predict(df_test_without_odds)
    df_res = df.copy()
    df_res['y_pred'] = y_pred
    df_res.to_excel('/Users/nachomondino/Desktop/predicciones.xlsx')
    '''

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()

# Tal vez, para no rellenar automaticamente las variables de jugadores (como dif_rat_tit, dif_edad_sup, dif_rat_aus)
# por no tener las formaciones antes del partido, podria tomar el rating de cada equipo segun su ultimo partido?
# Y considerar bajas?