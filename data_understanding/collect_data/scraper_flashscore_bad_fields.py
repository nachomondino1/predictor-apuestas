# Importo librerias
import pandas as pd
from dspy.data_understanding.web_scraping.selenium import Crawler


def extract_bad_fields_flashscore():

    # Solucionar el tema de que cuando falla un campo, tengo que volver a extraer tod@... Dar la posibildiad de recorrer los ids ya extraidos y extraer de nuevo el campo que falló

    # DEFINCION DE PARAMETROS & VARIABLES
    SEC_WAIT, SEC_WAIT_LONG = 0.2, 1.5
    crawler = Crawler(headless=True, path=None) # Creo objeto de clase CrawlerActions()
    l_var = ['odds_loc', 'odds_emp', 'odds_vis']

    # Levanto dataset de partidos ya extraidos
    df_part = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_partido_argentina.xlsx')

    # Por partido
    for i in range(len(df_part)):

        print(f" Partido Nº{i + 1} ".center(120, "#"))

        # Por variable
        for var in l_var:

            # Convierto str a float
            try:
                float(df_part.loc[i, var])

            # Si falla la conversion a float (es un str)
            except ValueError:

                # Vuelvo a buscar la cuota
                id = df_part.loc[i, 'id']

                # Reinicio diccionario en el que guardar la nueva fila
                d_nueva_fila = {'odds_loc': None, 'odds_emp': None, 'odds_vis': None}

                # Ingreso a pagina de informacion del partido
                crawler.driver.get(f'https://www.flashscore.es/partido/{id}/#/resumen-del-partido')

                # Si existe la seccion "Cuotas pre-partido", extraigo cuotas de Bet365
                if crawler.extract_tag(xpath='.//div[@class="oddsRow"]') is not None:  # No sirve en algunos partidos en los que existe la seccion de las cuotas pero no hay valores...

                    d_nueva_fila['odds_loc'] = extract_cuota(crawler, SEC_WAIT, i=1)
                    d_nueva_fila['odds_emp'] = extract_cuota(crawler, SEC_WAIT, i=2)
                    d_nueva_fila['odds_vis'] = extract_cuota(crawler, SEC_WAIT, i=3)

                print(f" Partido Nº{i + 1} ".center(120, "#"))
                print(f'{id} \nOdds_loc: {df_part.loc[i, "odds_loc"]} -> {d_nueva_fila["odds_loc"]} '
                      f'\nOdds emp: {df_part.loc[i, "odds_emp"]} -> {d_nueva_fila["odds_emp"]}'
                      f'\nOdds vis: {df_part.loc[i, "odds_vis"]} -> {d_nueva_fila["odds_vis"]}')

                # GUARDADO DE DATOS EN DATAFRAME
                df_part.loc[i, 'odds_loc'] = d_nueva_fila['odds_loc']
                df_part.loc[i, 'odds_emp'] = d_nueva_fila['odds_emp']
                df_part.loc[i, 'odds_vis'] = d_nueva_fila['odds_vis']
                break

    # Guardado de archivo excel en computadora
    df_part['odds_loc'] = df_part['odds_loc'].apply(lambda x: float(x))
    df_part['odds_emp'] = df_part['odds_emp'].apply(lambda x: float(x))
    df_part['odds_vis'] = df_part['odds_vis'].apply(lambda x: float(x))

    df_part.to_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/collect_data/data_seg/entidad_partido_argentina_prueba.xlsx', index=False)

    # Finalizada la extraccion, cierro el web browser automático
    crawler.driver.close()

def extract_cuota(crawler, SEC_WAIT, i):  # Puedo volver a la anterior, solo fallaron 41 cuotas por "Cuotas retiradas por la casa de apuestas."
    try:
        odds_str = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]', attribute='title', sec_wait=SEC_WAIT)  # 3.00 » 2.25
        cuota = float(odds_str.split('»')[0].strip())  # 3.00
    except ValueError:
        try:
            odds_str = crawler.extract_tag(xpath=f'.//div[@class="cellWrapper"][{i}]//span[@class="oddsValueInner"]', text=True, sec_wait=SEC_WAIT)
            cuota = float(odds_str)
        except (ValueError, TypeError):
            cuota = None
    return cuota

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    extract_bad_fields_flashscore()