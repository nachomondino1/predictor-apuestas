# Importo librerias
import sys
sys.path.append('.')  # Fallaba el import de main
import datetime
import re
import pandas as pd
import os
from main import DataUnderstanding, DataPreparation
## Data understanding
from p2_data_understanding import describe_data
## Data preparation
from p3_data_preparation import format_data, select_data, clean_data, construct_data
from p3_data_preparation.integrate_sofifa_to_flashscore import *
# Modeling
import pickle
# Deployment
from p6_deployment import collect_next_matches


class DataUnderstandingNew():  # DataUnderstanding

    def __init__(self, pais):
        #super().__init__(pais)
        self.pais = pais
        self.make_directories()

    def make_directories(self):
        
        directorio = f'./p6_deployment/data_next_matches/{self.pais.lower()}/data_understanding'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

    def collect_initial_data_new(self, export=True):

        print(" Collecting data... ")
        # Definicion de variables
        df_part_concat, df_part_jug_concat, df_jug_concat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        l_comp_a_evitar = ['Championship']

        # Selecciono competencias del pais
        df_comp = pd.read_excel('./p2_data_understanding/data/df_competencias.xlsx') # ./p2_data_understanding/data/df_competencias.xlsx'
        df_comp_pais = df_comp[df_comp['pais_flashscore'] == self.pais]  # Para extrar varios paises?: df = df_comp[df_comp['pais'].isin(l_paises)]
        print(f' PAIS: {self.pais} '.center(120, '#'), f"\nCompeticiones a extraer:\n{df_comp_pais['competicion_flashscore']}")

        # POR COMPETICION
        for i, row in df_comp_pais.iterrows():
            print(f" Competicion: {row['competicion_flashscore']} ".center(120, '+'))
    
            if row['competicion_flashscore'] not in l_comp_a_evitar:

                # Extraigo proximos partidos
                df_part_next_matches, df_part_jug_next_matches = collect_next_matches.extract_next_matches_flashscore(row['pais_flashscore'], row['competicion_flashscore'], row['is_cup'])
                print(f"Cantidad de proximos partidos: {df_part_next_matches.shape[0]}")
                df_part_concat = pd.concat([df_part_concat, df_part_next_matches], axis=0)
                df_part_jug_concat = pd.concat([df_part_jug_concat, df_part_jug_next_matches], axis=0)

        # Exporto datasets
        if export:
            df_part_concat.to_excel(f'./p6_deployment/data_next_matches/{self.pais}/data_understanding/df_part_next_matches.xlsx', index=False)
            df_part_jug_concat.to_excel(f'./p6_deployment/data_next_matches/{self.pais}/data_understanding/df_part_jug_next_matches.xlsx', index=False)

        return df_part_concat, df_part_jug_concat

    def describe_data_new(self, df_part, df_part_jug):  # Podria usar describe_data de main.py

        print(" Describiendo datos... ")
        describe_data.getting_to_know_data(df_part)
        describe_data.getting_to_know_data(df_part_jug)

        # Verifico unicidad de registros segun campos id
        describe_data.verificar_unicidad_registros(df_part, columns_id='id_part')

        # Verifico consistencia en campos que relacionan entidades
        describe_data.verificar_relacion_entidades(df_part, df_part_jug)  # si lo hago al reves si hay, pues no tod@ partido tiene datos de jugadores: verificar_relacion_entidades(df_jug_part, df_part)


class DataPreparationNew(DataPreparation):

    def __init__(self, var_resp, pais):

        super().__init__(var_resp, pais)
        self.var_resp = var_resp
        self.pais = pais
        self.make_directories()

    def make_directories(self):  # Pasarle direcotio o l_directorios como argumento...
        directorio = f'./p6_deployment/data_next_matches/{self.pais.lower()}/data_preparation'

        if not os.path.exists(directorio):
            # Si no existe, crear el directorio
            os.makedirs(directorio)

    def format_data_new(self, df_part, export=True):
        """
        Arreglo el data type de algunas variables.

        :param df_part: Dataframe de los datos de los partidos. (DataFrame)
        :param df_jug: Dataframe de los datos de los jugadores. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataset generado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe formateado. (DataFrame)
        """
        start = time.time()
        print("\nFormateando los datos...")

        # Dataframe partido
        ## Fecha
        df_part['fecha'] = pd.to_datetime(df_part['fecha'], format='%d.%m.%Y %H:%M') # Convierto fecha de object a datetime
        
        end = time.time()
        print(f"Formateo de datos en {(end - start)/60:.1f} minutos")

        if export:
            df_part.to_excel(f'./p6_deployment/data_next_matches/{self.pais.lower()}/data_preparation/df_part_form.xlsx')

        return df_part

    def clean_data_new(self, df_part, df_part_jug, export=True):
        """
        Limpieza inicial de los dataframes
        :param df_part:
        :param df_part_jug:
        :param df_jug:
        :param export:
        :return:
        """
        start = time.time()
        print("\nLimpiando los datos...")

        # Dataframe partido:
        ## Equipo_loc y equipo_vis
        df_part = clean_data.prepare_text_columns(df_part, l_cols_to_process=['equipo_loc', 'equipo_vis'])  # Preparacion texto para facilitar construccion de datos bassado en equipos
        df_part = clean_data.clean_teams_names(df_part)  # Eliminar strings adicionales en nombres de equipos

        # Dataframe partido jugador:
        ## jug_tit_loc_1, jug_tit_loc2, ..., jug_aus_sup_18
        df_part_jug = clean_data.prepare_text_columns(df_part_jug, l_col_to_except=['id_part'])

        end = time.time()
        print(f"Limpieza inicial de datos en {(end - start) / 60:.1f} minutos")

        if export:
            df_part.to_excel(f'./p6_deployment/data_next_matches/{self.pais.lower()}/data_preparation/df_part_clean.xlsx')
            df_part_jug.to_excel(f'./p6_deployment/data_next_matches/{self.pais.lower()}/data_preparation/df_part_jug_clean.xlsx')

        return df_part, df_part_jug

    def integrate_data_new(self, df_part, df_part_jug, df_jug, export=True):
        """
        Integra los datos de partidos y jugadores en un solo dataframe.

        :param df_part: Dataframe de los datos de los partidos.
        :param df_part_jug: Dataframe de los datos de los jugadores en cada partido.
        :param df_jug: Dataframe de los datos de los jugadores.
        :param export: Booleano para indicar si se debe exportar el dataframe integrado. True para exportar, False de
        lo contrario. (bool)
        :return: Dataframe integrado. (DataFrame)
        """
        df = self.integrate_data(df_part, df_part_jug, df_jug, export=False)  # Podria levantar df_part_jug_vinc_df_jug guardado para agilizar

        if export:
            df.to_excel(f'./p6_deployment/data_next_matches/{self.pais.lower()}/data_preparation/df_integrated.xlsx')

        return df

    def construct_data_new(self, df_new, n_dias, n_anios_historial, export=True): # actualizar df_old tiene que ser df_updated...
        """
        Construye nuevos datos a partir de un dataframe existente.

        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        # Guardo los ids de los partidos nuevos
        valores_id_part = df_new['id_part']

        # Levanto dataset de partidos viejos para poder calcular variables historicas en el nuevo df
        df_old = pd.read_excel(f'p6_deployment/data_next_matches/{self.pais}/data_preparation/df_integrated_updated.xlsx')
        # Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada equipo...
        fecha_limite = datetime.datetime.now() - datetime.timedelta(days=365 * n_anios_historial)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual
        df_old = df_old[df_old['fecha'] >= fecha_limite]
        print(df_old.shape)

        # Relleno datos no disponibles en partidos nuevos (rating formacion titular, etc) usando los partidos viejos
        df_new = self.rellenar_datos_no_disp_new_matches(df_new, df_old) 

        # Concateno df_new y df_old para construir variables
        df = pd.concat([df_new, df_old], axis=0).reset_index(drop=True)

        # Construyo datos llamando a construct_data de main.py
        df = self.construct_data(df, n_dias, n_anios_historial, export=False)
        df.to_excel(f'/Users/nachomondino/Desktop/df_constructed_entero.xlsx', index=False)

        # Vuelvo a seleccionar solo los partidos nuevos
        df_new_const = df[df['id_part'].isin(valores_id_part)]
        print(df_new_const)
        df_new_const.to_excel(f'/Users/nachomondino/Desktop/df_constructed_solo_next_matches.xlsx', index=False)


        if export:
            df_new_const.to_excel(f'./p6_deployment/data_next_matches/{self.pais}/data_preparation/df_constructed.xlsx', index=False)
        return df_new_const

    def rellenar_datos_no_disp_new_matches(self, df_new, df_old):
        """
        En los partidos nuevos, rellena los datos no disponibles (necesarios antes de construir datos) con los datos de partidos anteriores.

        ###### COSAS A MEJORAR: ######
       
        1) No queda claro que no me interesa rellenar las estadisticas ni goles... parece que es importante. De hecho, seria mas eficiente compu
        tacionalmente si las saco de entrada. Es dificil quitarlas sin que este hardcodeado (usar determine_estadisticas?)
        
        2) Rellena los datos solo coon los datos del ultimo partido. Si justo es un partido de copa o algo, puedo no tener datos o justo jugo con 
        suplentes. Por ello, para las variables numericas, deberias promediar el valor en los ultimos x partidos del equipo. --> PODRIA LLAMAR A
        determine_prom_en_ult_partidos() DE CONSTRUCT_DATA, NECESITO HACER ESO...

        """  
        print("Rellenando datos no disponibles...")
        # Ordenar el DataFrame por la columna 'fecha' de forma descendente
        df_old = df_old.sort_values(by='fecha', ascending=False)
        print(df_old.head(5))

        # Obtengo automaticamente las columnas que faltan en df_next_matches
        l_columns_to_add = list(set(df_old.columns) - set(df_new.columns))  # Puedo llamar a determine_estadisticas y quitarlas...
        # print("Columnas a agregar a df_new: ", l_columns_to_add)
        elementos_sin_suffix = {re.sub(r'_(loc|vis)$', '', elemento) for elemento in l_columns_to_add}
        print("Columnas a agregar a df de next matches: ", elementos_sin_suffix)

        # Por partido nuevo
        for i, row in df_new.iterrows():

            # print("\nFila nuevo partido:", row)
            l_equipos = [row['equipo_loc'], row['equipo_vis']]

            # Por equipo
            for equipo in l_equipos:

                # Busco el partido anterior del equipo
                fila_part_ant = df_old.loc[(df_old['equipo_loc'] == equipo) | (df_old['equipo_vis'] == equipo)].iloc[0]
                # print("Fila anterior partido:", fila_part_ant)

                # Determino localidad en partido nuevo y partido anterior
                tit = "loc" if fila_part_ant['equipo_loc'] == equipo else "vis"
                tit_new = "loc" if row['equipo_loc'] == equipo else "vis"

                # Cargo datos a partido nuevo
                for elem in elementos_sin_suffix:
                    # print(f"\t Columna: {elem}")
                    df_new.loc[i, f"{elem}_{tit_new}"] = fila_part_ant[f"{elem}_{tit}"]  # (e.g. df_new['dt_vis'] = fila_part_ant['dt_loc'])
                    # print(df_new.shape)
                
                    # ## prom_jugadores --> Deberia tomar promedio de ultimos x partidos pero bueno.
            
        # print(df_new)
        df_new.to_excel("/Users/nachomondino/Desktop/prueba.xlsx")
        return df_new

    def select_data_new(self, df, export=True):
        """
        Selecciona las variables relevantes del dataframe.

        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        start = time.time()
        print("\nSeleccionado datos...")

        # Seteo id_part como indice para poder reconocer que partido es cada uno luego de que el modelo prediga sus resultados.
        df = df.set_index("id_part")  

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        n_col = len(df.columns)
        df = df.drop(['pais', 'fecha'], axis=1)
        print(f"Se eliminó {n_col - len(df.columns)} de {n_col} columnas puesto que no sirven para el analisis (e.g. id_part, fecha, etc).")

        # Codifico variables categoricas a numericas con el mismo sistema que se uso en el dataframe original (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{self.pais}/df_etiquetas.xlsx')
        df, df_etiquetas = format_data.convert_columns_to_int(df, df_etiquetas)  # Si o si tengo que devolver df_etiquetas?
        df.to_excel('/Users/nachomondino/Desktop/df_codificacion_next_matches.xlsx')  #  Comprobé que codifica bien

        # Selecciono las variables que necesita el modelo
        X_test = pd.read_excel(f'./p4_modeling/data/{self.pais}/X_test.xlsx')
        df = df[X_test.columns]

        end = time.time()
        print(f"Las siguientes {len(df.columns)} columnas son las seleccionadas: {list(df.columns)}")
        print(f"Seleccion de datos en {(end - start)/60:.1f} minutos")

        if export:
            df.to_excel(f'./p6_deployment/data_next_matches/{self.pais}/data_preparation/df_selected.xlsx')
        return df


def main():

    # Definicion de variables
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    pais = "Inglaterra"  # Ponelo en miniscula
    data_unders, data_prep, modeling = False, True, False
    export = True

    if data_unders:
        print(" Data understanding ".center(120, "#"))
        du = DataUnderstandingNew(pais) # Creo objeto de clase DataPreparation

        # Extriago datos o los levanto
        df_part, df_part_jug = du.collect_initial_data_new(export=export)
        df_jug = pd.read_excel(f"./p3_data_preparation/data/{pais}/df_jug_form_clean.xlsx")

        # Describo datos
        du.describe_data_new(df_part, df_part_jug)

    # Si no extraigo datos
    else:
        # Levanto datos ya extraidos
        df_part = pd.read_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_next_matches.xlsx')
        df_part_jug = pd.read_excel(f'./p6_deployment/data_next_matches/{pais}/data_understanding/df_part_jug_next_matches.xlsx')
        df_jug = pd.read_excel(f"./p3_data_preparation/data/{pais}/df_jug_form_clean.xlsx")

        #du = DataUnderstanding(pais) # Creo objeto de clase DataPreparation
        # du.describe_data(df_part, df_part_jug, df_jug)

    if data_prep:
        # Definicion de variables
        print(" Data preparation ".center(120, "#"))
        dp = DataPreparationNew(var_resp, pais) # Creo objeto de clase DataPreparation

        # Hiperparametros --> Cuidado! tengo que usar los mismos que con los que construi los datos con los que entrene el modelo... 
        n_dias = 30  # 30 es como N_ULT_PART igual a 5... --> uso 90 porque no hay datos de partidos recientes...
        n_anios_historial = 3 
       
        # Preparo el dataset para el analisis
        df_part = dp.format_data_new(df_part, export=False) # Campo "fecha"
        df_part, df_part_jug = dp.clean_data_new(df_part, df_part_jug, export=export)
        df = dp.integrate_data_new(df_part, df_part_jug, df_jug, export=export)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc
        df = dp.construct_data_new(df, n_dias=n_dias, n_anios_historial=n_anios_historial, export=export)
        df = dp.select_data_new(df, export=export)

    elif not data_unders:
        # Levanto dataset para prueba
        df = pd.read_excel(f'./p6_deployment/data_next_matches/{pais}/data_preparation/df_selected.xlsx')
        print(df.head(1), df.shape)

    if modeling:

        # Definicion de variables
        print(" Modeling ".center(120, "#"))

        # Levanto modelo ya entrenado
        loaded_model = pickle.load(open(f"./p4_modeling/data/{pais}/modelo.pkl", "rb"))

        # Realizo predicciones sobre los nuevos partidos
        y_pred = loaded_model.predict(df)

        # Asignar las predicciones a una nueva columna en df_test (para poder calcular ROI)
        df_res = df.copy()
        df_res['y_pred'] = y_pred
    
        # Traduzco predicciones numericas a etiquetas
        df_etiquetas = pd.read_excel(f'./p3_data_preparation/data/{pais}/df_etiquetas.xlsx')
        df = format_data.revert_columns_from_int(df_res, df_etiquetas, columns=['y_pred'])
        df.to_excel(f'./p6_deployment/data_next_matches/{pais}/modeling/predicciones.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()