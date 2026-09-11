import pandas as pd
from predictor.utils.set_up_logging import logger
import os
from dotenv import load_dotenv
import predictor.utils.directories as directories


def concat_raw_data_by_country(d_countries, verbose: int = 1):
    """
    Concatenacion de datos extraidos de varios paises.
    """
    df_ct4, df_ct5, df_ct6 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Por pais
    for id_country, country in d_countries.items():

        # Levanto datasets de Sofifa
        df4 = pd.read_excel(f'data/{country}/p2_data_understanding/df_player_sofifa.xlsx', index_col=0)
        df5 = pd.read_excel(f'data/{country}/p2_data_understanding/df_player_fifa_sofifa.xlsx', index_col=0)
        df6 = pd.read_excel(f'data/{country}/p2_data_understanding/df_teams_sofifa.xlsx', index_col=0)

        if verbose >= 1:
            logger.info(f"Country: {country}. player: {df4.shape}. fifa: {df5.shape} team: {df6.shape}")

        df4_clean = df4[~df4.index.isin(df_ct4.index)]
        df5_clean = df5.drop_duplicates()

        # Concateno datos
        df_ct4 = pd.concat([df_ct4, df4_clean], axis=0)
        df_ct5 = pd.concat([df_ct5, df5_clean], axis=0)
        df_ct6 = pd.concat([df_ct6, df6], axis=0)

        if verbose >= 1:
            logger.info(f"player_ct: {df_ct4.shape}. fifa_ct: {df_ct5.shape} team_ct: {df_ct6.shape}")

    return df_ct4, df_ct5, df_ct6 

if __name__ == "__main__":

    # Definicion de parametros
    d_countries = {6: 'argentina', 48: 'england', 55: 'france', 59: 'germany', 77: 'italy', 148: 'spain', 167: 'usa'}

    # Flashscore
    df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = concat_raw_data_by_country(d_countries)
    
    # Exporto datos
    df_player_sofifa.to_excel('data/sofifa/df_player_sofifa.xlsx')
    df_player_fifa_sofifa.to_excel('data/sofifa/df_player_fifa_sofifa.xlsx')
    df_teams_sofifa.to_excel('data/sofifa/df_teams_sofifa.xlsx')