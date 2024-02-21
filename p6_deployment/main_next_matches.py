# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import datetime
import re
import pandas as pd
import os
from main import DataUnderstanding, DataPreparation
## Data understanding
from p2_data_understanding.collect_initial_data.scraper_flashscore import extract_next_matches
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation.construct_data import determine_mean_in_last_matches, determine_stats_columns
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
            df_match_next, df_match_player_next = extract_next_matches(self.country, competition, n_days=n_days)

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

    def __init__(self, id_country, country):

        super().__init__(country)
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
        Arreglo el data fill_type de algunas variables.
        :param df_match: Dataframe de los datos de los partidos. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormating data...")

        # Dataframe partido
        ## Fecha
        df_match['date'] = pd.to_datetime(df_match['date'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        ## Capacity & Attendance  # Deberia fallar attendance porque aun no existe el dato...
        df_match = format_data.convert_capacity_to_int(df_match)

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
        print("\nCleaning new data...")

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
        print("\nConstructing new data...")
        # 1) Levanto df_integrated y lo actualizo (no puedo construir datos en nuevos partidos sin los partidos anteriores)
        # df_int_old_updated = self.get_updated_df(n_years_h2h)  # Podria hacerlo afuera de aqui. 
        df_int_old_updated = pd.read_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_integrated_updated.xlsx', index_col=0)
        print("\n df_int_old_updated \n")
        print(df_int_old_updated.head(2))
        df_int_old_updated.info()

        # print(df_int_old_updated.shape)  # 1149 (1141 VIEJOS + 8 MISSING)
        df_concat = pd.concat([df, df_int_old_updated], axis=0)
        print("\n DF_CONCAT \n")
        df_concat.info()
        # print(df_concat.shape)  # 1160 (1149 + 10 NEXT)
        df_concat.to_excel(f'/Users/nachomondino/Desktop/df_antes_de_construir.xlsx', index=True)

        # 2) Construyo datos normalmente como lo hago en main.py
        df_concat = self.construct_data(df_concat, n_days, n_years_h2h, export=False)  # FALLA PORQUE AL CONCATENAR LAS ESTADISTICAS PIERDEN EL DTYPE FLOAT
        df_concat.to_excel(f'/Users/nachomondino/Desktop/df_constructed.xlsx', index=True)
    
        # 3) Relleno datos no disponibles en partidos nuevos (rating formacion titular, etc) usando los partidos viejos
        df_concat = self.copy_last_matches_mean_value(df_concat, n_days)
        df_concat = self.copy_last_matches_value(df_concat, df.index) 
        df_concat.to_excel("/Users/nachomondino/Desktop/prueba.xlsx")

        # 4) Selecciono solo los partidos nuevos de los datos construidos 
        df_constructed = df_concat[df_concat.index.isin(df.index)]
        df_constructed.to_excel(f'/Users/nachomondino/Desktop/df_constructed_solo_next_matches.xlsx', index=False)
        
        # # VERIFICAR QUE EL DATASET NO TIENEN NAN  --> Aca o en select_data? Para eliminar solo si no tiene datos en las variables selected y no borrar mal.
        df_sin_dup = df_constructed.dropna()
        if len(df_sin_dup) > 0:
            text = f"Cuidado! No se hara la prediccion para {len(df_sin_dup)} partidos puesto que tienen al menos un valor NaN y el modelo no puede tener input NaN."
            warnings.warn(text)
    
        if export:
            df_constructed.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_constructed.xlsx', index=True)
        return df_constructed

    def get_updated_df(self, n_years_h2h):
        """
        Levanta dataframe integrado con el que se entreno el modelo y lo actualiza con los partidos mas recientes.
        """
        print("\nRaising integrated data with which the model was trained and updating it...")
         ## Levanto dataset de partidos viejos para poder calcular variables historicas en el nuevo df
        df_integrated = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_integrated.xlsx', index_col=0)

        ## Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada team...
        fecha_limite = datetime.datetime.now() - datetime.timedelta(days=365 * n_years_h2h)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual
        df_integrated = df_integrated[df_integrated['date'] >= fecha_limite]
        print(df_integrated.shape)

        ## Llamo a main_missing_matches para actualizar df_match hasta el ultimo partido jugado
        df_integrated_missing = main_missing_matches.main(self.id_country)
        print("\n df_integrated_missing \n")
        df_integrated_missing.info()
        df_integrated_missing = self.convert_columns_to_float(df_integrated_missing)
        
        print("\n df_integrated_missing post conversion \n")
        df_integrated_missing.info()

        df_integrated_missing.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_integrated_missing.xlsx', index=True)
        
        # Concateno
        df_integrated_updated = pd.concat([df_integrated, df_integrated_missing], axis=0)
        df_integrated_updated.to_excel(f'/Users/nachomondino/Desktop/df_integrated_updated.xlsx', index=True)

        ## Ordenar el DataFrame por la columna 'fecha' de forma descendente
        df_integrated_updated = df_integrated_updated.sort_values(by='date', ascending=False)
        df_integrated_updated.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_integrated_updated.xlsx')
        return df_integrated_updated
    
    def convert_columns_to_float(self, df: pd.DataFrame):
        """
        Intenta convertir las columnas object a float
        """
        # Selecciono las columnas object
        l_columnas_a_codificar = df.select_dtypes(include=['object']).columns

        # Por columna object
        for col in l_columnas_a_codificar:

            # Intento convertirla a float
            try:
                df[col] = df[col].astype(float)
                print(f"Se convirtio la columna {col} a float!")
            except:
                pass
        return df

    def copy_last_matches_value(self, df_concat, index_new_matches):
        """
        # En los partidos nuevos, rellena los datos no disponibles (necesarios antes de construir datos) con los datos de partidos anteriores.
        
        Cosas a agregar:
        - Si los dts son nan en el partido anterior, entonces buscar en el sigueinte y asi...
        """  
        print("Rellenando datos no disponibles...")
        # Tengo que agregar copiar los datos del ultimo partido para: dt
        l_var_to_copy = ['coach']
        df_int_new = df_concat[df_concat.index.isin(index_new_matches)]
        df_int_old = df_concat.loc[~df_concat.index.isin(index_new_matches)]

        # Por partido nuevo
        for idx, row in df_int_new.iterrows():

            # print("\nFila nuevo partido:", row)
            l_teams = [row['team_home'], row['team_away']]

            # Por team
            for team in l_teams:

                # Busco el partido anterior del team
                fila_part_ant = df_int_old.loc[(df_int_old['team_home'] == team) | (df_int_old['team_away'] == team)].iloc[0]

                # Determino localidad en partido nuevo y partido anterior
                tit = "home" if fila_part_ant['team_home'] == team else "away"
                tit_new = "home" if row['team_home'] == team else "away"

                # Cargo datos a partido nuevo
                for elem in l_var_to_copy:
                    df_int_new.loc[idx, f"{elem}_{tit_new}"] = fila_part_ant[f"{elem}_{tit}"]  # (e.g. df_new['dt_away'] = fila_part_ant['dt_home'])
            
        return df_int_new

    def copy_last_matches_mean_value(self, df_concat, n_days):
        """
        Copio el promedio de los valores en los ultimos partidos
        # Obtengo valor para columnas promedio de jugadores promediando el valor en ultimos partidos
        """
        # Selecciono las variables que corresponden a jugadores
        l_columns_sin_suffix = {re.sub(r'_(home|away)$', '', elemento) for elemento in df_concat.columns}
        l_var_mean_player = [col for col in l_columns_sin_suffix if re.search(r'mean_.*_player_', col)]  # r'_jug_'  --> tmb modificaba n_jug_home pero crasheaba para hacer promedio.
        print("Columnas a agregar a df de next matches: ", l_columns_sin_suffix)
        print(f"Lista de columnas prom jug: {l_var_mean_player}")

        # Por variable mean_player
        for var in l_var_mean_player:
            print(f"\tVariable a promediar: {var}")  # (e.g. dif_mean_val_player_sub)

            df_concat = determine_mean_in_last_matches(df_concat, n_days=n_days, variable=var, fill_type="mean") # (e.g. mean_last_match_dif_mean_val_player_sub)
            df_concat = df_concat.drop([var], axis=1)  # (e.g. dif_prom_alt_jug_tit)

            # Determinar la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_home y prom_dif_goles_away)
            df_concat[var] = df_concat[f'mean_last_match_{var}_home'] - df_concat[f'mean_last_match_{var}_away'] # (e.g. mean_last_match_dif_prom_alt_jug_tit)
            df_concat = df_concat.drop(columns=[f'mean_last_match_{var}_home', f'mean_last_match_{var}_away'], axis=1)
      
        df_concat.to_excel(f'/Users/nachomondino/Desktop/df_copy_last_matchs_mean_value.xlsx', index=False)
        return df_concat

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
        df = df.drop(['date'], axis=1)

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.country}/df_etiquetas.xlsx')
        df, df_etiquetas = format_data.convert_columns_to_int(df, df_etiquetas)  # Si o si tengo que devolver df_etiquetas?
        df.to_excel('/Users/nachomondino/Desktop/df_codificacion_next_matches.xlsx')  #  Comprobé que codifica bien

        # Selecciono las variables que necesita el modelo
        X_test = pd.read_excel(f'./p4_modeling/data/{self.country}/X_test.xlsx', index_col=0)
        df = df[X_test.columns]
        print(f"Las siguientes {len(X_test.columns)} columnas son las seleccionadas: {list(X_test.columns)}")

        end = time.time()
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p6_deployment/data_next_matches/{self.country}/data_preparation/df_selected.xlsx', index=True)
        return df
 
def main():
    """
    Recoleccion de proximos partidos
    """
    # Definicion de variables
    var_resp, var_pred = 'result', 'predicted_result'
    data_unders, data_prep, modeling = False, False, True
    export = True

    # Hiperparametro
    n_days = 7  # Numero de dias maximo desde hoy para extraer partidos

    # Selecciono country a extraer por terminal
    # df_countries = pd.read_excel('./p2_data_understanding/data/df_countries.xlsx')
    # country = str(input("Choose country to extract (e.g. England, Germany, etc): "))
    # id_country = df_countries[df_countries['country_name'] == country]['id_country'].values[0]
    id_country, country = 48, "England"
 
    # DATA UNDERSTANDING
    if data_unders:
        print(" Data understanding ".center(120, "#"))
        du = DataUnderstandingNew(id_country, country) # Creo objeto de clase DataPreparation

        # Extriago datos o los levanto
        df_match, df_match_player = du.collect_initial_data_new(n_days=n_days, export=export)

        # Describo datos
        du.describe_data_new(df_match, df_match_player)
    
    elif data_prep:
        # Levanto datos ya extraidos
        df_match = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_player = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_understanding/df_match_player_next.xlsx', index_col=0)
        print("\n DF MATCH \n", df_match.head(2))
        print("\n DF MATCH PLAYER \n", df_match_player.head(2))

        #du = DataUnderstanding(country) # Creo objeto de clase DataPreparation
        # du.describe_data(df_match, df_match_player, df_player)

    # DATA PREPARATION
    if data_prep:
        # Definicion de variables
        print(" Data preparation ".center(120, "#"))
        dp = DataPreparationNew(id_country, country) # Creo objeto de clase DataPreparation

        # Hiperparametros (tengo que usar los mismos que con los que construi los datos con los que entrene el modelo)
        df_hiper_prep = pd.read_excel(f'./p3_data_preparation/data/{country}/df_hiper_prep.xlsx')
        n_days = int(df_hiper_prep['n_days'].values[0])
        n_years_h2h = int(df_hiper_prep['n_years_h2h'].values[0])

        # Preparo el dataset para el analisis
        df_match = dp.format_data_new(df_match, export=False) # Campo "fecha"
        df_match, df_match_player = dp.clean_data_new(df_match, df_match_player, export=export)
        df = dp.integrate_data_new(df_match, df_match_player, export=export)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc
        df = dp.construct_data_new(df, n_days=n_days, n_years_h2h=n_years_h2h, export=export)
        df = dp.select_data_new(df, export=export)
    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p6_deployment/data_next_matches/{country}/data_preparation/df_selected.xlsx', index_col=0)
        print(df.head(), df.shape)

    # MODELING
    if modeling:
        directorio = f'./p6_deployment/data_next_matches/{country.lower()}/modeling'
        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

        print(" Modeling ".center(120, "#"))
        # Levanto modelo ya entrenado
        loaded_model = pickle.load(open(f"./p4_modeling/data/{country}/modelo.pkl", "rb"))

        # Realizo predicciones sobre los nuevos partidos
        y_pred = loaded_model.predict(df)
        y_pred_prob = loaded_model.predict_proba(df)

        # Asignar las predicciones a una nueva columna
        df_res = df.copy()
        df_res['predicted_result'] = y_pred
        df_res['probability_class_0'] = y_pred_prob[:, 0]
        df_res['probability_class_1'] = y_pred_prob[:, 1]
        df_res['probability_class_2'] = y_pred_prob[:, 2]
    
        # Traduzco predicciones numericas a etiquetas
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{country}/df_etiquetas.xlsx')
        # df_etiquetas_y = df_etiquetas[df_etiquetas['variable'] == var_resp]  # solo etiquetas de la var resp
        # from p4_modeling import asses_model
        # df = asses_model.convert_pred_int_to_str(df, name_var_int=var_resp, name_var_str=f'{var_resp}_str', df_etiquetas_y= df_etiquetas_y)

        df = format_data.revert_columns_from_int(df_res, df_etiquetas, columns=['predicted_result'])
        df.to_excel(f'./p6_deployment/data_next_matches/{country}/modeling/predicciones.xlsx')

        # Genero df con id y prediccion y le agrego equipos y fecha? o ya es suficiente con predicciones.xlsx?
        df_match_next = pd.read_excel(f'p6_deployment/data_next_matches/{country.lower()}/data_understanding/df_match_next.xlsx', index_col=0)
        df_match_odds_next = pd.read_excel(f'./p6_deployment/data_next_matches/{country.lower()}/data_understanding/df_match_next_odds.xlsx', index_col=0)

        df_concat = pd.concat([df_match_next, df_match_odds_next, df_res], axis=1)
        df_concat.to_excel(f'/Users/nachomondino/Desktop/predicciones.xlsx')


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()