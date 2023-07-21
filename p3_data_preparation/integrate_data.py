import pandas as pd


# Integracion entre datos de WHOSCORED
def map_player_entities(df_jug, df_jug_part):
    """
    Integro entidad de jugador con la entidad de jugador por partido.

    :param df_jug: Dataframe de jugadores. (DataFrame)
    :param df_jug_part: Dataframe de jugadores por partido. (DataFrame)
    :return: Dataframe jugadores por partido con columnas altura y fecha_nac por jugador. (DataFrame)
    """
    # Filtrar las columnas necesarias de df_jug_part
    df_jug_filtered = df_jug[['id_jug', 'altura', 'fecha_nac']]

    # Combinar df_jug_part_filtered con df_jug usando el id_jug como clave
    df_merged = pd.merge(df_jug_part, df_jug_filtered, on='id_jug', how='left')
    return df_merged

def map_player_to_part(df_jug_part, df_part):  # Agregar calculo de rating y min played
    """
    Integro entidad de jugador por partido con la entidad partido.

    :param df_jug_part: Dataframe de jugadores por partido con columnas altura y fecha_nac por jugador. (DataFrame)
    :param df_part: Dataframe de partidos. (DataFrame)
    :return: Dataframe de partidos con nuevas columnas con los datos de los jugadores por cada partido. (DataFrame)
    """
    # Definicion de variables
    l_condiciones = df_jug_part['condicion'].unique()
    l_titularidades = df_jug_part['titularidad'].unique()

    # Por partido en df_part
    for index, row in df_part.iterrows():
        df_jug_part_filt_1 = df_jug_part[df_jug_part['id_part'] == row['id_part']]

        # Por condicion {home, away}
        for condicion in l_condiciones:
            df_jug_part_filt_2 = df_jug_part_filt_1[df_jug_part_filt_1['condicion'] == condicion]

            # Por titularidad {tit, sup}
            for titularidad in l_titularidades:

                # Selecciono registros
                df_jug_part_filt_3 = df_jug_part_filt_2[df_jug_part_filt_2['titularidad'] == titularidad]
                largo = len(df_jug_part_filt_3)

                # Calculo promedio de edad y altura
                if largo > 0:
                    prom_edad = df_jug_part_filt_3['edad'].mean()
                    prom_alt = df_jug_part_filt_3['altura'].mean()
                    sum_rat = df_jug_part_filt_3['prom_pond_rating_ult_part'].dropna().sum()  #  prom_rat = df_jug_part_filt_3['prom_pond_rating_ult_part'].mean()
                    sum_min_played = df_jug_part_filt_3['sum_min_played_ult_part'].dropna().sum()
                    prom_overall_rat = df_jug_part_filt_3['overall_rating'].mean()  # si hago suma, tengo que rellenar NaN values con min rating...
                    prom_valor_mercado = df_jug_part_filt_3['valor_mercado'].mean()

                    # Guardo columna en df_part
                    df_part.loc[index, f'prom_edad_{condicion}_{titularidad}'] = prom_edad
                    df_part.loc[index, f'prom_alt_{condicion}_{titularidad}'] = prom_alt
                    df_part.loc[index, f'sum_rat_{condicion}_{titularidad}'] = sum_rat  # df_part.loc[index, f'prom_rat_{condicion}_{titularidad}'] = prom_rat
                    df_part.loc[index, f'sum_min_{condicion}_{titularidad}'] = sum_min_played
                    df_part.loc[index, f'prom_ov_rat_{condicion}_{titularidad}'] = prom_overall_rat
                    df_part.loc[index, f'prom_val_mer_{condicion}_{titularidad}'] = prom_valor_mercado

    return df_part

def prueba():
    from p3_data_preparation import construct_data
    from p3_data_preparation.integrate_sofifa_to_whoscored import player_data_in_match
    from p3_data_preparation.integrate_flashscore_to_whoscored import fill_whoscored_with_flashscore

    # Definicion de variables
    pais = "argentina"
    # pais = "argentina_south_america"
    fill_data_with_flashcore = False
    n_dias_player_data = 30

    # Levanto datasets de prueba
    df_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_part_cleaned.xlsx')
    df_jug_part = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_jug_part.xlsx')
    df_jug = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p3_data_preparation/data/{pais}/df_jug_formated.xlsx')

    # Integro Flashscore a Whoscored para rellenar estadisticas en partidos de Whoscored
    if fill_data_with_flashcore:
        df_part_flash = pd.read_excel(f'/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/{pais}/df_part_fs.xlsx')
        df_part = fill_whoscored_with_flashscore(df_part, df_part_flash)

    # Integro df_jug a df_jug_part
    df_jug_part = map_player_entities(df_jug, df_jug_part)

    # Calculo edad y minutos jugados por jugador en cada partido
    df_jug_part = construct_data.add_fecha(df_part, df_jug_part)
    df_jug_part = construct_data.add_team(df_part, df_jug_part)
    df_jug_part = construct_data.determine_edad(df_jug_part)
    df_jug_part = construct_data.determine_min_played(df_jug_part)
    df_jug_part = construct_data.determine_player_var_en_ult_partidos(df_jug_part, 'min_played', n_dias=n_dias_player_data, tipo='sum')
    df_jug_part = construct_data.determine_player_var_en_ult_partidos(df_jug_part, 'rating', n_dias=n_dias_player_data, tipo='mean_pond', var_pond='min_played')
    # df_jug_part.to_excel('/Users/nachomondino/Desktop/df_jug_part_antes_de_int_sofifa.xlsx', index=False)

    # Integro sofifa a df_jug_part
    # df_jug_part = pd.read_excel('/Users/nachomondino/Desktop/df_jug_part_antes_de_int_sofifa.xlsx')
    df_jug_sofifa = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/argentina/entidad_jugadores.xlsx')
    df_jug_part = player_data_in_match(df_jug_part, df_jug_sofifa)
    # df_jug_part.to_excel('/Users/nachomondino/Desktop/df_jug_part_antes_int.xlsx', index=False)

    # Integro df_jug_part_integ (df_jug_part + df_jug) a df_part
    # df_jug_part = pd.read_excel('/Users/nachomondino/Desktop/df_jug_part_antes_int.xlsx')
    df = map_player_to_part(df_jug_part, df_part)
    df.to_excel('/Users/nachomondino/Desktop/df_integrated.xlsx', index=False)


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":
    prueba()