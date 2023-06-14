# Importo librerias
import pandas as pd
import datetime
import time
import warnings
import random
import pickle
from data_understanding import collect_initial_data
from data_preparation import format_data, integrate_data, construct_data, select_data, clean_data
from dspy.data_understanding.describe_data import getting_to_know_data


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

    def construct_data(self, df_new=None, N_ULT_PART=5, export=False):  # 2.7 minutos
        """
        Construye nuevos datos a partir de un dataframe existente.
        :param df: Dataframe con datos de partidos incluyendo datos de jugadores. Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param N_ULT_PART: Número de últimos partidos a considerar para el cálculo de variables. (int)
        :param export: Booleano para indicar si se debe exportar el dataframe construido. True para exportar, False de lo contrario. (bool)
        :return: Dataframe construido. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        # df_new = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_integrated.xlsx') if df is None else df

        n_reg = len(df_new)

        # Levanto dataset viejo para poder calcular variables historicas en el nuevo df
        df_old = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_integrated.xlsx')

        # Filtro dataset old por fecha para evitar levantar todos los datos y minimizar tiempo de computo. Solo requiero ultimos 5 part de cada equipo...
        # fecha_limite = datetime.datetime.now() - datetime.timedelta(days=365 * 5)  # Calcular la fecha límite retrocediendo 3 años a partir de la fecha actual
        # df_old = df_old[df_old['fecha'] >= fecha_limite]

        # Creo variable equipo_ganador para que poder calcular historial_entre_si y forma_reciente
        df_old = construct_data.determinar_equipo_ganador(df_old)  # --> a df_part no le construyo equipo_ganador...

        # Relleno formacion de proximos partidos
        # rellenar_player_data(df_part)

        # Concateno df_new y df_old
        df = pd.concat([df_old, df_new], axis=0).reset_index(drop=True)  # Funciona bien

        # Construyo variables historicas
        l_estad_part = ['posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas', 'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']
        df = construct_data.historial_entre_si_segun_localia(df, n_ult_part=int(N_ULT_PART / 2))
        df = construct_data.promedio_ult_partidos(df, n_ult_part=N_ULT_PART, l_var=l_estad_part)  # Estadisticas del partido
        df = construct_data.promedio_dif_gol_ult_part(df, n_ult_part=N_ULT_PART)  # Diferencia de gol
        df = construct_data.forma_reciente(df, n_part=N_ULT_PART)  # Rendimiento del equipo
        df = construct_data.n_dias_ult_partido(df)  # Numero de dias desde ultimo partido

        # Construyo variables de diferencias para las variables promedio de los jugadores
        df = construct_data.calculate_dif_col_jugadores(df)  # No tengo datos de jugadores...  df[nombre_col_dif] = df[nombre_col_loc] - df[nombre_col_vis]  KeyError: 'prom_edad_jug_tit_loc'

        # Vuelvo a seleccionar ultimos n registros
        df_new = df.tail(n_reg)
        if export:
            df_new.to_excel('/Users/nachomondino/Desktop/df_constructed_next_matches.xlsx', index=False)
        return df_new

    def select_data(self, df_new=None, treat_nan='drop', export=False):  # 1.3 minutos
        """
        Selecciona las variables relevantes del dataframe.
        :param df: Dataframe de los datos Si no se proporciona, se cargará desde un archivo. (DataFrame)
        :param export: Booleano para indicar si se debe exportar el dataframe seleccionado. True para exportar, False de lo contrario. (bool)
        :return: Dataframe con las variables seleccionadas. (DataFrame)
        """
        # Si no han pasado un dataset utilizo un dataframe guardado
        df_new = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/df_constructed.xlsx') if df_new is None else df_new
        df = df_new.copy()

        # Definicion de variables
        warnings.filterwarnings('ignore')
        print("\nSeleccionado datos...")
        # n_reg = len(df_new)


        # TENGO QUE USAR DF_OLD SOLO PARA CALCULAR DIF_RAT_TIT POR LOS EQUIPOS QUE NO ESTAN EN EL FIFA?

        # Levanto dataset viejo para poder calcular variables historicas en el nuevo df
        # df_old = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/argentina/df_constructed.xlsx')

        # Concateno df_new y df_old
        # df = pd.concat([df_old, df_new], axis=0).reset_index(drop=True)  # Funciona bien

        # Elimino variables que no usare en el modelo como id o fecha (la idea es usar todas las posibles)
        df.index = df['id']  # Despues lo saco?
        # df = df.drop(['id', 'fecha', 'cancha', 'competicion', 'temporada', 'pais'], axis=1)  # No hace falta, las quito al seleccionor despues...

        # Tratamiento de NaN values --> hace falta aqui? loo hago antes de concatenarlo?
        # df = select_data.eliminar_filas_nan(df, umbral=0.5)  # 1º elimino registros con muchos nan
        # df = select_data.eliminar_columnas_nan(df, umbral=0.2)  # 2º elimino columnas con mucho NaN
        # df = select_data.eliminar_filas_nan(df, umbral=0)  # Si es 0, funciona igual a dropna() pero ademas, imprime rdos
        # Ojo que por ahi elimino alguno/s de los proximos partidos

        # Codifico variables categoricas a numericas (es de format_data pero lo hago aca porque sino no puedo calcular la correlacion de las variables no numericas...)
        df = format_data.convert_columns_to_int(df)

        # Selecciono las variables utilizadas en el modelo (feature selection) --> Levanto df?
        df_test = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/modeling/data/argentina/df_test.xlsx')
        l_selected_features = list(df_test.drop(self.var_resp, axis=1).columns)
        df = df.loc[:, l_selected_features]

        # Borrro nan una vez que seleccione las columnas... (MOMENTANEAMENTE, PARA EVITAR TENER QUE LEVANTAR DF_OLD PARA RELLENAR NAN EN DIF_RAT_TIT...
        df = select_data.eliminar_filas_nan(df, umbral=0)  # Si es 0, funciona igual a dropna() pero ademas, imprime rdos

        # Vuelvo a seleccionar ultimos n registros
        # df_new = df.tail(n_reg)

        if export:
            df.to_excel('/Users/nachomondino/Desktop/df_part_selected_next_matches.xlsx')

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

def main():

    # Definicion de variables
    pais = 'argentina'
    var_resp, var_pred = 'equipo_ganador', 'y_pred'
    prepare = DataPreparation(var_resp)
    data_unders, data_prep, modeling = False, True, True

    ## DATA UNDERSTANDING
    if data_unders:
        # Hiperparametros
        n_dias_a_prox_part = 1  # Numero de dias maximo para partido a recolectar

        print(" Data Understanding ".center(120, "#"))
        # Collect initial data
        df_part = collect_initial_data.extract_proximos_partidos_flashcore([pais], n_dias_max=n_dias_a_prox_part)

        # Describe data
        getting_to_know_data(df_part)

    if data_prep:

        # Hiperparametros
        N_ULT_PART = 5  # Numero de partidos a tener en cuenta para variables historicas como posesion en ult partidos
        treat_nan = 'ml'  # Relleno de nan values: 'mode' o 'ml'
        export = False

        # Levanto dataset con los ultimos 15 partidos de la liga argentina
        df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_understanding/data/{pais}/entidad_next_partido.xlsx')
        df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/data_preparation/data/{pais}/df_jug_cleaned.xlsx')

        ## DATA PREPARATION
        print(" Data preparation ".center(120, "#"))

        # no le hago format porque ya extraigo la fecha en formato datetime, la copa como 1 o 0 y no tengo posesion_loc ni posesion_vis
        df_part = prepare.clean_data(df_part, export=export)
        df = prepare.integrate_data(df_part, df_jug, export=export)  # si no tengo formaciones, no tiene sentido integrar... Integrar en el fondo es reemplazar nombre de jugadores por su rating, edad, valor_mercado, etc
        df = prepare.construct_data(df, N_ULT_PART=N_ULT_PART, export=True)
        df = prepare.select_data(df, treat_nan=treat_nan, export=True)

    if modeling:
        # Vuelvo a seleccionar solo los partidos a predecir  (despues de select para poder hacer treat_nan con ml basandome en los partidos viejos...)

        # df = pd.read_excel('/Users/nachomondino/Desktop/df_part_selected_next_matches.xlsx', index_col=0)

        ## Modeling  --> solo tengo que predecir... y despues analizar los resultados una vez concluida la fecha...
        df_test_pred = df.copy().drop(['odds_loc', 'odds_emp', 'odds_vis'], axis=1)  # Esto esta ok
        print(df_test_pred.shape)

        loaded_model = pickle.load(open(f"/Users/nachomondino/Documents/GitHub/predictor-apuestas/modeling/data/{pais}/modelo.pkl", "rb"))

        y_pred = loaded_model.predict(df_test_pred)

        # Asignar las predicciones a una nueva columna en df_test (para poder calcular ROI)
        df_res = df.copy()
        df_res['y_pred'] = y_pred

        # Traduzco predicciones numericas a etiquetas
        df = format_data.target_to_object(df_res, pais)
        df.to_excel('/Users/nachomondino/Desktop/predicciones.xlsx')


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    main()

# Tal vez, para no rellenar automaticamente las variables de jugadores (como dif_rat_tit, dif_edad_sup, dif_rat_aus)
# por no tener las formaciones antes del partido, podria tomar el rating de cada equipo segun su ultimo partido?
# Y considerar bajas?