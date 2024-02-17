# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import datetime
import re
import pandas as pd
import os
from main import DataUnderstanding, DataPreparation
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_next_matches_flashscore
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation.construct_data import determine_mean_in_last_match, determine_l_stats
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
# Modeling
import pickle

import main_missing_matches

class DataUnderstandingNew():  # DataUnderstanding

    def __init__(self, id_country, country):
        #super().__init__(country)
        self.id_country = id_country
        self.country = country
        self.make_directories()

    def make_directories(self):
        
        directorio = f'./p6_deployment/data_next_matches/{self.country.lower()}/data_understanding'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

    def collect_initial_data_new(self, n_days: int = 7, export=True):

        print(" Collecting data... ")
        print(f' COUNTRY: {self.country} '.center(120, '#'))
        # Levanto datasets
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencies.xlsx')
        df_match = pd.read_excel(f'./p2_data_understanding/data/{self.country}/df_match.xlsx', index_col=0)

        # Definicion de variables
        df_match_concat, df_match_player_concat = pd.DataFrame(), pd.DataFrame()
        l_competencies = df_match['id_competition'].unique()
        print(f"Competiciones a extraer del pais {self.country}: {l_competencies}")

        # POR COMPETITION
        for id_competition in l_competencies:
            
            df_comp_filt = df_comp[(df_comp['id_country'] == self.id_country) & (df_comp['id_competition'] == id_competition)]  # Para extrar varios countryes?: df = df_comp[df_comp['country'].isin(l_countryes)]
            competition, is_cup = df_comp_filt['competition_flashscore'].values[0], df_comp_filt['is_cup'].values[0]
            print(f" Competition: {competition} ".center(120, '+'))

            # Extraigo proximos partidos
            df_match_next, df_match_player_next = extract_next_matches_flashscore(self.country, competition, n_days=n_days)

            # Add columns: id_country, is_cup and id_competition
            df_match_next['id_country'] = self.id_country
            df_match_next['id_competition'] = id_competition
            df_match_next['is_cup'] = is_cup

            # Guarda datos de competition
            df_match_concat = pd.concat([df_match_concat, df_match_next], axis=0)
            df_match_player_concat = pd.concat([df_match_player_concat, df_match_player_next], axis=0)
            
        # Supongamos que df es tu DataFrame original 
        df_match_odds = df_match_concat.loc[:, ['odds_home', 'odds_draw', 'odds_away']]
        df_match_concat = df_match_concat.drop(['odds_home', 'odds_draw', 'odds_away'], axis=1)

        # Exporto datasets
        if export:
            df_match_concat.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_understanding/df_match_next.xlsx', index=True)
            df_match_player_concat.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_understanding/df_match_player_next.xlsx', index=True)
            df_match_odds.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_understanding/df_match_next_odds.xlsx', index=True)

        return df_match_concat, df_match_player_concat

    def describe_data_new(self, df_match, df_match_player):  # Podria usar describe_data de main.py

        print(" Describiendo datos... ")
        describe_data.getting_to_know_data(df_match)
        describe_data.getting_to_know_data(df_match_player)

        # Verifico unicidad de registros segun campos id
        describe_data.verificar_unicidad_registros(df_match)

        # Verifico consistencia en campos que relacionan entidades
        describe_data.check_ids_in_both_dataframes(df_match, df_match_player)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de jugadores: verificar_relacion_entidades(df_player_part, df_match)


class DataPreparationNew(DataPreparation):

    def __init__(self, var_resp, id_country, country):

        super().__init__(var_resp, country)
        self.var_resp = var_resp
        self.id_country = id_country
        self.country = country.lower()
        self.make_directories()

    def make_directories(self):  # Pasarle direcotio o l_directorios como argumento...
        directorio = f'./p6_deployment/data_next_matches/{self.country.lower()}/data_preparation'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

    def format_data_new(self, df_match: pd.DataFrame, export: bool = True):
        """
        Arreglo el data type de algunas variables.
        :param df_match: Dataframe de los datos de los partidos. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormating data...")

        # Dataframe partido
        ## Fecha
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        
        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_match.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_match_form.xlsx')
        return df_match

    def clean_data_new(self, df_match: pd.DataFrame, df_match_player: pd.DataFrame, export: bool = True):
        """
        Limpieza inicial de los dataframes
        :param df_match:
        :param df_match_player:
        :param df_player:
        :param export:
        :return:
        """
        start = time.time()
        print("\nLimpiando los datos...")

        # Dataframe partido:
        ## Team_home y team_away
        df_match = clean_data.prepare_text_columns(df_match, l_cols_to_process=['team_home', 'team_away'])  # Preparacion texto para facilitar construccion de datos bassado en equipos
        df_match = clean_data.clean_teams_names(df_match)  # Eliminar strings adicionales en nombres de equipos

        # Dataframe partido jugador:
        ## jug_tit_home_1, jug_tit_home2, ..., jug_aus_sup_18
        df_match_player = clean_data.prepare_text_columns(df_match_player)

        end = time.time()
        print(f"Limpieza inicial de datos en {(end - start) / 60:.1f} minutos")

        if export:
            df_match.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_match_clean.xlsx')
            df_match_player.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_match_player_clean.xlsx')

        return df_match, df_match_player

    def integrate_data_new(self, df_match, df_match_player, export=True):
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        :param df_match: Dataframe de los datos de los partidos.
        :param df_match_player: Dataframe de los datos de los jugadores en cada partido.
        :param df_player: Dataframe de los datos de los jugadores.
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de
        lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        # Levanto df_player
        df_player = pd.read_excel(f"./p3_data_preparation/data/{self.country}/df_player_form_clean.xlsx")

        # Uso integracion de main.py
        df = self.integrate_data(df_match, df_match_player, df_player, export=False)  # Podria levantar df_match_player_vinc_df_player guardado para agilizar

        if export:
            df.to_excel(f'./p6_deployment/data_next_matches/{self.country.lower()}/data_preparation/df_integrated.xlsx')

        return df

    def construct_data_new(self, df: pd.DataFrame, n_days: int, n_years_h2h: int, export: bool = True): # actualizar df_old tiene que ser df_updated...
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        # Definicion de variables
        fecha_limite = datetime.datetime.now() - datetime.timedelta(days=365 * n_years_h2h)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual


        # 1) Levanto df_match actualizado
        ## Levanto dataset de partidos viejos para poder calcular variables historicas en el nuevo df
        df_match = pd.read_excel(f'p6_deployment/data_next_matches/{self.country}/data_preparation/df_integrated_updated.xlsx')

        ## Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada equipo...
        df_match = df_match[df_match['date'] >= fecha_limite]
        print(df_match.shape)

        ## Llamo a main_missing_matches para actualizar df_match hasta el ultimo partido jugado
        df_match_miss = main_missing_matches.main(self.id_country, self.country, self.var_resp)
        df_match_updated = pd.concat([df_match, df_match_miss], axis=1)

        ## Ordenar el DataFrame por la columna 'fecha' de forma descendente
        df_match_updated = df_match_updated.sort_values(by='date', ascending=False)


        # 2) Relleno datos no disponibles en partidos nuevos (rating formacion titular, etc) usando los partidos viejos
        df_match_next = self.rellenar_datos_no_disp_new_matches(df, df_match_updated) 
        df_match_next.to_excel("/Users/nachomondino/Desktop/prueba.xlsx")


        # 3) Construyo datos con df_match_updated y df_match_next
        # PODRIA HACER TODO ESTO EN OTRA VARIABLE LLAMADA RELLENAR...
        # Concateno df_new y df_old para construir variables
        df_concat = pd.concat([df_match_next, df_match_updated], axis=0)

        # Construyo datos normalmente como lo hago en main.py
        df_concat = self.construct_data(df_concat, n_days, n_years_h2h, export=False)
        df_concat.to_excel(f'/Users/nachomondino/Desktop/df_constructed_con_promedio.xlsx', index=False)


        # 4) Relleno variables de jugadores
        df_concat = self.rellenar_variables_de_jugadores()


        # 5) Selecciono solo los partidos nuevos de los datos construidos 
        df_match_next_constructed = df_concat[df_concat.index.isin(df.index)]
        # df_match_next_constructed.to_excel(f'/Users/nachomondino/Desktop/df_constructed_solo_next_matches.xlsx', index=False)
        
        # VERIFICAR QUE EL DATASET NO TIENEN NAN

        if export:
            df_match_next_constructed.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_constructed.xlsx', index=True)
        return df_match_next_constructed

    def rellenar_datos_no_disp_new_matches(self, df_new, df_old):
        """
        En los partidos nuevos, rellena los datos no disponibles (necesarios antes de construir datos) con los datos de partidos anteriores.
        
        Cosas a agregar:
        # Si los dts son nan en el partido anterior, entonces buuscar en el sigueinte y asi...
        """  
        print("Rellenando datos no disponibles...")
        # Tengo que agregar copiar los datos del ultimo partido para: dt
        l_var_to_copy = ['dt']

        # Por partido nuevo
        for i, row in df_new.iterrows():

            # print("\nFila nuevo partido:", row)
            l_equipos = [row['team_home'], row['team_away']]

            # Por equipo
            for equipo in l_equipos:

                # Busco el partido anterior del equipo
                fila_part_ant = df_old.loc[(df_old['team_home'] == equipo) | (df_old['team_away'] == equipo)].iloc[0]

                # Determino localidad en partido nuevo y partido anterior
                tit = "loc" if fila_part_ant['team_home'] == equipo else "vis"
                tit_new = "loc" if row['team_home'] == equipo else "vis"

                # Cargo datos a partido nuevo
                for elem in l_var_to_copy:
                    df_new.loc[i, f"{elem}_{tit_new}"] = fila_part_ant[f"{elem}_{tit}"]  # (e.g. df_new['dt_away'] = fila_part_ant['dt_home'])
            
        # VERIFICAR QUE EL DATASET NO TIENEN NAN.
        return df_new

    def rellenar_variables_de_jugadores(self, df_match_updated, df, df_concat, n_days):
        # Obtengo valor para columnas promedio de jugadores promediando el valor en ultimos partidos
        ## Obtengo automaticamente las columnas que faltan en df_next_matches
        l_columns_to_add = list(set(df_match_updated.columns) - set(df.columns))  # Puedo llamar a determine_estadisticas y quitarlas...
        l_columns_sin_suffix = {re.sub(r'_(home|away)$', '', elemento) for elemento in l_columns_to_add}
        # print("Columnas a agregar a df de next matches: ", l_columns_sin_suffix)
        l_var_prom_jug = [col for col in l_columns_sin_suffix if re.search(r'prom_.*_jug_', col)]  # r'_jug_'  --> tmb modificaba n_jug_home pero crasheaba para hacer promedio.
        print(f"Lista de columnas prom jug: {l_var_prom_jug}")

        # Por variable prom_jug
        for var in l_var_prom_jug:
            print(f"\tVariable a promediar: {var}")

            df_concat = determine_mean_in_last_match(df_concat, n_days=n_days, variable=f'dif_{var}', tipo="mean") # (e.g. mean_last_match_dif_prom_alt_jug_tit_home)
            df_concat = df_concat.drop([f'dif_{var}'], axis=1)  # (e.g. dif_prom_alt_jug_tit)

            # Determinar la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            df_concat[f'dif_{var}'] = df_concat[f'mean_last_match_dif_{var}_home'] - df_concat[f'mean_last_match_dif_{var}_away'] # (e.g. mean_last_match_dif_prom_alt_jug_tit)
            df_concat = df_concat.drop(columns=[f'mean_last_match_dif_{var}_home', f'mean_last_match_dif_{var}_away'], axis=1)
        # df.to_excel(f'/Users/nachomondino/Desktop/df_prom_jug.xlsx', index=False)

    def select_data_new(self, df, export=True):
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        start = time.time()
        print("\nSeleccionado datos...")

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        # n_col = len(df.columns)
        df = df.drop(['date'], axis=1)
        # print(f"Se eliminó {n_col - len(df.columns)} de {n_col} columnas puesto que no sirven para el analisis (e.g. id_match, fecha, etc).")

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')
        df, df_etiquetas = format_data.convert_columns_to_int(df, df_etiquetas)  # Si o si tengo que devolver df_etiquetas?
        df.to_excel('/Users/nachomondino/Desktop/df_codificacion_next_matches.xlsx')  #  Comprobé que codifica bien

        # Selecciono las variables que necesita el modelo
        X_test = pd.read_excel(f'./p4_modeling/data/{self.country}/X_test.xlsx')
        df = df[X_test.columns]
        print(f"Las siguientes {len(X_test.columns)} columnas son las seleccionadas: {list(X_test.columns)}")

        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_selected.xlsx')
        return df


def main():
    """
    Recoleccion de proximos partidos
    """
    # Definicion de variables
    var_resp, var_pred = 'result', 'predicted_result'
    data_unders, data_prep, modeling = False, True, False
    export = True

    # Selecciono country a extraer por terminal
    # df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    # country = str(input("Choose country to extract (e.g. England, Germany, etc): "))
    # id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]
    id_country, country = 48, "England"
    
    # DATA UNDERSTANDING
    if data_unders:
        print(" Data understanding ".center(120, "#"))
        du = DataUnderstandingNew(id_country, country) # Creo objeto de clase DataPreparation

        # Hiperparametro
        n_days = 7  # Numero de dias maximo desde hoy para extraer partidos

        # Extriago datos o los levanto
        df_match, df_match_player = du.collect_initial_data_new(n_days=n_days, export=export)

        # Describo datos
        du.describe_data_new(df_match, df_match_player)
    
    elif data_prep:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_player_next.xlsx', index_col=0)
        # print("\n DF MATCH \n", df_match.head(2))
        # print("\n DF MATCH PLAYER \n", df_match_player.head(2))

        #du = DataUnderstanding(country) # Creo objeto de clase DataPreparation
        # du.describe_data(df_match, df_match_player, df_player)

    # DATA PREPARATION
    if data_prep:
        # Definicion de variables
        print(" Data preparation ".center(120, "#"))
        dp = DataPreparationNew(var_resp, id_country, country) # Creo objeto de clase DataPreparation

        # Hiperparametros (tengo que usar los mismos que con los que construi los datos con los que entrene el modelo)
        df_hiper_prep = pd.read_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx')
        n_days = int(df_hiper_prep['n_days'].values[0])
        n_years_h2h = int(df_hiper_prep['n_years_h2h'].values[0])

        # Preparo el dataset para el analisis
        df_match = dp.format_data_new(df_match, export=False) # Campo "fecha"
        df_match, df_match_player = dp.clean_data_new(df_match, df_match_player, export=export)
        df = dp.integrate_data_new(df_match, df_match_player, export=export)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc
        # df = dp.construct_data_new(df, n_days=n_days, n_years_h2h=n_years_h2h, export=export)
        # df = dp.select_data_new(df, export=export)
    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_preparation/df_selected.xlsx', index_col=0)
        print(df.head(), df.shape)

    # MODELING
    if modeling:
        print(" Modeling ".center(120, "#"))
        # Levanto modelo ya entrenado
        loaded_model = pickle.load(open(f"./p4_modeling/data/{country}/modelo.pkl", "rb"))

        # Realizo predicciones sobre los nuevos partidos
        y_pred = loaded_model.predict(df)

        # Asignar las predicciones a una nueva columna
        df_res = df.copy()
        df_res['predicted_result'] = y_pred
    
        # Traduzco predicciones numericas a etiquetas
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{country}/df_etiquetas.xlsx')
        df = format_data.revert_columns_from_int(df_res, df_etiquetas, columns=['predicted_result'])
        df.to_excel(f'./p6_deployment/data_next_matches/{country}/modeling/predicciones.xlsx')

        # Genero df con id y prediccion y le agrego equipos y fecha? o ya es suficiente con predicciones.xlsx?

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()