import pandas as pd
import numpy as np
import time
import math
from datetime import timedelta

# Construyo variables en dataframe "partido"
def determinar_equipo_ganador_segun_casa_apuesta(df):
    """
    Se determina el 'equipo_ganador' segun la casa de apuestas
    :param df: Dataframe. Unidad de analisis: partido. Columnas: entre ellas odds_loc, odds_emp, odds_vis
    :return: Dataframe pasado por parametro con nueva columna, 'equipo_ganador_ca', que detalla el resultado del partido
    segun la casa de apuesta.
    """
    # Por fila
    for i, row in df.iterrows():

        # Determino la cuota minima de las 3 posibles
        odds_min = min(row['odds_loc'], row['odds_emp'], row['odds_vis'])

        # Si la cuota minima es la del equipo local
        if row['odds_loc'] == odds_min:
            df.loc[i, 'equipo_ganador_ca'] = "Local"

        # Si la cuota minima es la del equipo visitante
        elif row['odds_vis'] == odds_min:
            df.loc[i, 'equipo_ganador_ca'] = "Visitante"

        # Si la cuota minima es la del empat
        else:
            df.loc[i, 'equipo_ganador_ca'] = "Empate"
    return df

def determinar_equipo_ganador(df):
    """
    Se determina el 'equipo_ganador' a partir de los goles que hizo cada equipo
    :param df: Dataframe. Unidad de analisis: partido. Columnas: entre ellas goles_loc y goles_vis
    :return: Dataframe pasado por parametro con nueva columna, 'equipo_ganador', que detalla el resultado del partido.
    """
    # Por fila (partido)
    for i in range(len(df)):

        # Obtengo goles de cada equipo
        ng1, ng2 = df.loc[i, 'goles_loc'], df.loc[i, 'goles_vis']

        # Guardo equipo ganador
        df.loc[i, 'equipo_ganador'] = 'Local' if ng1 > ng2 else 'Empate' if ng1==ng2 else "Visitante"
    return df

def determinar_puntos(df):
    """
    Determina los puntos obtenidos por cada equipo segun el resultado del juego.

    :param df:
    :return:
    """
    # Inicializo las columnas "puntos_loc" y "puntos_vis"
    df['puntos_loc'] = 0
    df['puntos_vis'] = 0

    df.loc[df['equipo_ganador'] == 'Local', 'puntos_loc'] = 3
    df.loc[df['equipo_ganador'] == 'Local', 'puntos_vis'] = 0

    df.loc[df['equipo_ganador'] == 'Empate', 'puntos_loc'] = 1
    df.loc[df['equipo_ganador'] == 'Empate', 'puntos_vis'] = 1

    df.loc[df['equipo_ganador'] == 'Visitante', 'puntos_loc'] = 0
    df.loc[df['equipo_ganador'] == 'Visitante', 'puntos_vis'] = 3
    return df

def determine_prom_en_ult_partidos(df, n_dias, variable, tipo):
    """
     Obtiene el promedio de las estadisticas en los ultimos partidos

     :param df: DataFrame.
     :param n_dias: Integer. Numero de dias de los cuales obtener los datos.
     :param variable: String. Nombre de la variable a promediar.
     :param tipo: String. Tipo de cálculo a realizar ('mean' para promedio, 'sum' para suma).
     :return: DataFrame con estadisticas promediadas
     """
    # Ordeno por fecha ascendente
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)

    # Por equipo
    for equipo in df['equipo_loc'].unique():

        # Obtengo los partidos que jugó el equipo
        df_equipo = df[(df['equipo_loc'] == equipo) | (df['equipo_vis'] == equipo)]
        # print(f"Equipo: {equipo}")
        # print(f"Las filas son la cantidad total de partidos del equipo: {df_equipo.shape}")

        # Por partido del equipo
        for idx, row in df_equipo.iterrows():

            loc_o_vis = 'loc' if row['equipo_loc'] == equipo else 'vis'
            fecha_limite = row['fecha'] - timedelta(days=n_dias)
            # print(f"Partido Nº: {idx}")

            # Selecciono los ultimos partidos del equipo
            df_equipo_last_matches = df_equipo.loc[(df_equipo['fecha'] >= fecha_limite) & (df_equipo['fecha'] < row['fecha'])]
            # print(f"Las filas son las cantidad de partido en ultimos {n_dias} dias: {df_equipo_last_matches.shape}")
            # df_equipo_last_matches.to_excel('/Users/nachomondino/Desktop/df_equipo_last_matches.xlsx')

            # Necesito la posesion segun si fue local o visitante en cada uno de esos partidos...
            s_valores_loc = df_equipo_last_matches.loc[df_equipo_last_matches['equipo_loc'] == equipo, f'dif_{variable}']
            s_valores_vis = df_equipo_last_matches.loc[df_equipo_last_matches['equipo_vis'] == equipo, f'dif_{variable}'] * -1  # -1 puesto que valores positivos en dif_variable es para el local y valores negativos es favor del visitante
            s_valores = pd.concat([s_valores_loc, s_valores_vis], ignore_index=True)
            # print(f"Valores del equipo en estadistica dif_{variable}: {s_valores}")

            if len(s_valores) > 0:
                if tipo == "mean":
                    df.loc[idx, f'prom_ult_part_dif_{variable}_{loc_o_vis}'] = s_valores.mean()
                    # print(f"Promedio: {s_valores.mean()} \n")

                elif tipo == "sum":
                    df.loc[idx, f'sum_ult_part_dif_{variable}_{loc_o_vis}'] = s_valores.sum()

    return df

def historial_entre_si_segun_fecha(df, n_anios):
    """
    Determina el historial entre los equipos que disputan el partido según los resultados en los últimos partidos entre ellos.

    :param df: DataFrame. Unidad de análisis: partido. Columnas: al menos fecha, equipo_loc, equipo_vis y equipo_ganador.
    :param n_anios: Integer. Número de años a tener en cuenta para determinar el historial entre dos equipos.
    :return: DataFrame pasado por parámetro con nueva columna, 'historial_entre_si_fecha', que permite determinar a cuál de los dos
    equipos de un partido le favorece más el historial entre ellos.
    """
    # Definicion de variables
    n_dias = 365 * n_anios
    l_equipos = df['equipo_loc'].unique()
    # print(f"Lista de equipos: {l_equipos}")

    # Ordeno por fecha descendiente (ya se extrae ordenado por fecha descendente pero por las dudas)
    df = df.sort_values(by='fecha', ascending=False, ignore_index=True)  # Mas reciente a mas antiguo

    for i in range(len(l_equipos)):
        eq1 = l_equipos[i]
        # print(f"Equipo 1: {l_equipos[i]}")

        for j in range(i+1, len(l_equipos)):
            eq2 = l_equipos[j]
            # print(f"Equipo 2: {l_equipos[j]}")

            df_historial = df[((df['equipo_loc'] == eq1) & (df['equipo_vis'] == eq2)) | (
                            df['equipo_loc'] == eq2) & (df['equipo_vis'] == eq1)]
            # print(df_historial)

            # Por partido del historial
            for idx, row in df_historial.iterrows():

                equipo_local = row['equipo_loc']
                historial = 0
                fecha_part = row['fecha']
                fecha_limite = fecha_part - timedelta(days=n_dias)
                # print(f"Equipo local en partido {idx}: {equipo_local}".center(120))
                # print(f"Fecha: {fecha_part} ; Fecha limite: {fecha_limite}")

                # Selecciono los ultimos partidos
                df_sel = df_historial.loc[(df_historial['fecha'] >= fecha_limite) & (df_historial['fecha'] < fecha_part)]
                # print(df_sel)

                # Por ultimos partidos
                for index, fila in df_sel.iterrows():

                    if fila['equipo_ganador'] == "Local":
                        historial += +1 if fila['equipo_loc'] == equipo_local else -1

                    elif fila['equipo_ganador'] == "Visitante":
                        historial += -1 if fila['equipo_loc'] == equipo_local else +1

                    else:
                        historial += 0
                    # print(f"Index: {index} ; Equipo ganador: {fila['equipo_ganador']} ; Equipo local: {fila['equipo_loc']}")
                    # print(historial)

                # Guardo historial
                if len(df_sel) > 0:  # Para evitar guardar historial = 0 en partidos donde df_sel no tiene registros porque no jugaron entre si en los ultimos años
                    df.loc[idx, 'historial_entre_si_fecha'] = historial
    return df

def calculate_dif_col_jugadores(df):
    """
    Calcula la diferencia entre local y visitante
    :param df: Dataframe. Unidad de analisis: partido
    :return:
    """
    # Defincion de variables
    l_var_jug = ['prom_edad', 'prom_alt', 'prom_rat', 'prom_valor']
    l_titularidad = ['tit', 'sup', 'aus']

    for titularidad in l_titularidad:

        for var in l_var_jug:

            # Calculo diferencia entre local y visitante
            df[f'dif_{var}_jug_{titularidad}'] = df[f'{var}_jug_{titularidad}_loc'] - df[f'{var}_jug_{titularidad}_vis']

            # Elimino variables utilizadas para calcular la diferencia
            df = df.drop([f'{var}_jug_{titularidad}_loc', f'{var}_jug_{titularidad}_vis'], axis=1)
    return df

def suma_rat_jug_aus(df):  # Ojo falla en calculo cuando uno de los dos equipos no tiene jugadores ausentes (o sea, las variables aus son nan) --> en ese caso tiene que hacer la diferencia igual...
    """
    Calculo la suma del rating de los jugadores ausentes dado que cada equipo tiene distinto numero de ausentes.
    :param df: Dataframe. Unidad de analisis: partido.
    :return:
    """
    df['prom_rat_jug_aus_loc'] = df['prom_rat_jug_aus_loc'] * df['n_jug_aus_loc']  # Lo guardo en la misma porque sino la cago con calculate_dif_col_jugadores()
    df['prom_rat_jug_aus_vis'] = df['prom_rat_jug_aus_vis'] * df['n_jug_aus_vis']

    # Elimino columnas que contaban jugadores ausentes en cada equipo
    df = df.drop(["n_jug_aus_loc", "n_jug_aus_vis"], axis=1)
    return df

def prueba():
    # Definicion de variables
    pais = 'England'
    n_dias = 30  # 30 es como N_ULT_PART igual a 5...
    n_anios_historial = 2
    l_estadisticas = ['goles', 'puntos', 'posesion', 'remates', 'remates_a_puerta', 'tarjetas_amarillas', 'faltas',
                      'pases', 'pases_comp', 'offsides', 'ataques', 'ataques_pelig']

    # Levanto dataset
    df = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_integrated.xlsx')
    print(df.head())

    start = time.time()

    # Construyo variables: "equipo_gandor", diferencia de goles y puntos obtenidos
    df = determinar_equipo_ganador(df)
    df = determinar_puntos(df)

    # Variables historicas
    df = historial_entre_si_segun_fecha(df, n_anios=n_anios_historial)

    # Por estadistica del partido
    for est in l_estadisticas:

        # Determinar la diferencia de la estadistica entre equipo local y visitante de cada partido
        df[f'dif_{est}'] = df[f'{est}_loc'] - df[f'{est}_vis']  # (e.g. dif_goles = goles_loc - goles_vis)
        df = df.drop([f'{est}_loc', f'{est}_vis'], axis=1)  # (e.g. borro goles_loc y goles_vis)

        # Determinar para cada equipo de un partido, el promedio en los ultimos partidos de dicha diferencia de la estadistica
        df = determine_prom_en_ult_partidos(df, n_dias=n_dias, variable=est, tipo='mean')
        df = df.drop([f'dif_{est}'], axis=1)

        # Determinar la diferencia entre promedio del local y del visitante (por ej, diferencia entre prom_dif_goles_loc y prom_dif_goles_vis)
        df[f'dif_prom_ult_part_dif_{est}'] = df[f'prom_ult_part_dif_{est}_loc'] - df[f'prom_ult_part_dif_{est}_vis']
        df = df.drop(columns=[f'prom_ult_part_dif_{est}_loc', f'prom_ult_part_dif_{est}_vis'], axis=1)

    # Construyo variables de diferencias para las variables promedio de los jugadores
    df = suma_rat_jug_aus(df)
    df = calculate_dif_col_jugadores(df)

    end = time.time()
    print(f"Construccion de datos en {(end - start) / 60:.1f} minutos")

    df.to_excel('/Users/nachomondino/Desktop/df_constructed_prueba.xlsx')

# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()