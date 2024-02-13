# Importo librerias
import sys
sys.path.append('/Users/nachomondino/Documents/GitHub/predictor-apuestas')  # Fallaba el import de main
import pandas as pd
from main import DataUnderstanding, DataPreparation
from p2_data_understanding.collect_initial_data.scraper_flashscore import FlashscoreCrawler
from tqdm import tqdm


def extract_data_faltante_flashcore(l_ids_already_collected, pais, competition, is_cup):
    """
    Extrae los partidos aun no extraidos de una competencia de un pais.
    """
  # DEFINCION DE PARAMETROS & VARIABLES
    path_driver_exe = "/Users/nachomondino/Documents/chrome_driver/chromedriver"
    crawler = FlashscoreCrawler(headless=True, path=path_driver_exe, browser="Chrome")
    df_part, df_part_jug = pd.DataFrame(), pd.DataFrame()
    pais_form = pais.lower().replace(' ', "_")
    competicion_form = competition.lower().replace(" ", "-")  # formateo competicion para las rutas de archivo y urls
    # ruta_base = f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/p6_deployment/data_next_matches/{pais_form}/data_understanding/"

    # Ingreso a pagina de competicion
    url = f'https://www.flashscore.com.ar/futbol/{pais_form}/{competicion_form}'
    print(url)
    crawler.driver.get(url)  # hasta que no se carga toda la pagina, no sigue...

    # Accept cookies (a veces no llega a cargar, igual creo que no afecta)
    crawler.accept_cookies()
    temp_year = crawler.extract_tag(xpath='.//div[@class="heading__info"]', text=True)
    temp_year_form = temp_year.replace("/", "_")
    print(f" {temp_year} ".center(120, "-"))

    # Click en boton "Mostrar mas partidos" (para ver no solo la jornada actual sino todas las jornadas de la temporada)
    while True:
        boton_mostrar = crawler.extract_tag(xpath='.//a[text()="Mostrar más partidos"]', sec_wait=crawler.SEC_WAIT_MAX) # boton_mostrar = crawler.extract_tag(xpath='.//div[@id="live-table"]//div[@class="tabs" and text()="Últimos Resultados"]/following_sibling()::div//a[@class="event__more event__more--static"]', sec_wait=crawler.SEC_WAIT_MAX * 15)
        if crawler.click_boton(boton_mostrar) is False:
            break

    # Extraigo partidos (items) y sus ids
    l_items = crawler.extract_tags(xpath='.//div[@id="live-table"]//div[@class="event__match event__match--static event__match--twoLine" or @title="¡Haga click para detalles del partido!"]', sec_wait=crawler.SEC_WAIT_MAX)
    l_ids = [item.get_attribute('id') for item in l_items]
    print(f"Partidos recolectados de la temporada {temp_year} (e.g. en premier league deberian ser 380): {len(l_ids)}")
    progress_bar = tqdm(total=len(l_ids), ncols=80)  # Inicializo barra de progreso

    # POR PARTIDO (c/u identificado con un id)
    for id_part in l_ids:

        # Limpio el id
        id_part = id_part[id_part.rfind('_') + 1:]  # Quito lo que no es del id (e.g. paso de "g_1_fshvzbls" a "fshvzbls")

        # Si el partido aun no fue extraido
        if id_part not in l_ids_already_collected:

            # Ingreso a pagina de informacion del partido
            url_partido = f'https://www.flashscore.com.ar/partido/{id_part}/#/resumen-del-partido/resumen-del-partido'
            crawler.driver.get(url_partido)

            # Reinicio diccionario en el que guardar datos del nuevo partido
            d_new_row_df_part = {'id_part': id_part, 'competicion': competition, 'temporada': temp_year, 'pais': pais, 'es_copa': is_cup}
            d_new_row_df_part_jug = {'id_part': id_part}

            # EXTRACCION DE CAMPOS
            ## Extraigo campos de hoja "Resumen"
            d_new_row_df_part.update(crawler.extract_basic_data_from_resumen())

            ## Si tiene hoja "Estadisticas", extraigo campos
            boton_estadisticas = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Estadísticas"]', sec_wait=crawler.SEC_WAIT_MED, print_fail=True)
            if crawler.click_boton(boton_estadisticas) is not False:
                d_new_row_df_part.update(crawler.extract_estadisticas())

            ## Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
            if crawler.extract_tag(xpath='.//div[@class="oddsRowContent"]', sec_wait=crawler.SEC_WAIT_MAX) is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...
                d_new_row_df_part.update(crawler.extract_cuota())

            ## Si tiene hoja "Formaciones", extraigo campos
            boton_formaciones = crawler.extract_tag(xpath='.//div[@class="filterOver filterOver--indent"]//button[text()="Formaciones" or text()="Alineaciones"]', sec_wait=crawler.SEC_WAIT_MED, print_fail=True)
            if crawler.click_boton(boton_formaciones) is not False:
                d_new_row_df_part_jug.update(crawler.extract_formacion())
                d_new_row_df_part.update(crawler.extract_dts())

            # GUARDADO DE DATOS EN DATAFRAME
            df_part = pd.concat([df_part, pd.DataFrame(d_new_row_df_part, index=[0])])
            df_part_jug = pd.concat([df_part_jug, pd.DataFrame(d_new_row_df_part_jug, index=[0])])
            progress_bar.update(1)
        
        else:
            print(f"El id {id_part} ya está en l_ids_already_collected, por ende, no se extrae")
            break

    # Cerrar la barra de progreso al finalizar
    progress_bar.close()

    # Guardo datos por seguridad
    # if export:
    #     # Guardo partidos
    #     df_part.to_excel(f'{ruta_base}/df_part_new_matches_{competicion_form}_{temp_year_form}_{pais_form}.xlsx', index=False)
    #     df_part_jug.to_excel(f'{ruta_base}/df_part_jug_new_matches_{competicion_form}_{temp_year_form}_{pais_form}.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()
    return df_part, df_part_jug

def collect_initial_data_faltante(pais, export=True):
    """
    Extraccion de varias competencias de un mismo pais.
    """
    print(" Collecting data... ")
    # Selecciono competencias del pais
    l_comp_a_evitar = ['Championship']
    df_comp = pd.read_excel('./p2_data_understanding/data/df_competencias.xlsx') # /Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/df_competencias.xlsx'
    df_comp_pais = df_comp[df_comp['pais_flashscore'] == pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
    print(f' PAIS: {pais} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_pais['competicion_flashscore']}")

    # Levanto datos viejos 
    df_part = pd.read_excel(f'./p2_data_understanding/data/{pais}/df_part.xlsx')

    # Definicion de variables
    df_part_updated, df_part_jug_updated = pd.DataFrame(), pd.DataFrame()

    # POR COMPETICION
    for i, row in df_comp_pais.iterrows():
        print(f" Competicion: {row['competicion_flashscore']} ".center(120, '+'))

        if row['competicion_flashscore'] not in l_comp_a_evitar:

            # Actualizo df_part y df_part_jug con los partidos faltantes
            df_part_miss_matches, df_part_jug_miss_matches = extract_data_faltante_flashcore(list(df_part['id_part']), row['pais_flashscore'], row['competicion_flashscore'], row['is_cup'])
            print(f"Cantidad de partidos faltantes en df_part: {df_part_miss_matches.shape[0]}")
            # Concateno dfs
            df_part_updated = pd.concat([df_part_updated, df_part_miss_matches], axis=0)
            df_part_jug_updated = pd.concat([df_part_jug_updated, df_part_jug_miss_matches], axis=0)

    # Exporto datasets
    if export:
        df_part_updated.to_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_missing.xlsx', index=False)
        df_part_jug_updated.to_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_jug_missing.xlsx', index=False)

    return df_part_updated, df_part_jug_updated

def main():
    """
    Extraer automaticamente los partidos que faltan en df_part y df_part_jug. Sobretodo es esencial para poder construir estadisticas en 
    partidos nuevos a predecir.
    """
     # Definicion de variables
    var_resp = 'equipo_ganador'
    pais = "Inglaterra"  # Ponelo en miniscula
    data_unders, data_prep = False, True
    export = True

    # Actualizo datasets
    if data_unders:
        # Creo instancia de clase de main.py (el procesamiento es el mismo)
        du = DataUnderstanding(pais=pais)

        # Collect missing data
        df_part_missing, df_part_jug_missing = collect_initial_data_faltante(pais, export=export)
        df_jug = pd.read_excel(f'./p2_data_understanding/data/{pais}/df_jug.xlsx') # Podria recolectar nueva version del ultimo fifa.

        # Describe data
        du.describe_data(df_part_missing, df_part_jug_missing, df_jug)
    else:
        df_part_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_missing.xlsx')
        df_part_jug_missing = pd.read_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_jug_missing.xlsx')
        df_jug = pd.read_excel(f'./p2_data_understanding/data/{pais}/df_jug.xlsx')

    if data_prep:
        # Creo instancia de clase de main.py (el procesamiento es el mismo)
        dp = DataPreparation(var_resp=var_resp, pais=pais)

        ## Data prep
        thr_nan_col = 1  # No quiero que elimine nada
        df_part_missing, df_part_jug_missing, df_jug = dp.format_data(df_part_missing, df_part_jug_missing, df_jug, export=False)
        df_part_missing, df_part_jug_missing, df_jug = dp.clean_data(df_part_missing, df_part_jug_missing, df_jug, thr_nan_col, export=False)
        df_integrated_missing = dp.integrate_data(df_part_missing, df_part_jug_missing, df_jug, export=False)
        df_integrated_missing.to_excel(f'./p6_deployment/data_next_matches/{pais}/data_preparation/df_integrated_missing.xlsx', index=False)

        # Concateno con datos viejos
        ## Levanto datos viejos (para rellenarlos)
        df = pd.read_excel(f'./p3_data_preparation/data/{pais}/df_integrated.xlsx')
        df_integrated_updated = pd.concat([df, df_integrated_missing], axis=0)
        df_integrated_updated.to_excel(f'./p6_deployment/data_next_matches/{pais}/data_preparation/df_integrated_updated.xlsx', index=False)

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()
