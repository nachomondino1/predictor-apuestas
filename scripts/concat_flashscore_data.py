import pandas as pd
from utils.set_up_logging import logger
import os
from dotenv import load_dotenv
import utils.directories as directories


def concat_raw_data_by_country(d_countries, missing: bool = True, verbose: int = 1):
    """
    Concatenacion de datos extraidos de varios paises.
    """
    df_ct1, df_ct2, df_ct3 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Por pais
    for id_country, country in d_countries.items():

        # Levanto datasets de Flashscore
        df1 = pd.read_excel(f'data/{country}/p2_data_understanding/df_match.xlsx', index_col=0)
        df2 = pd.read_excel(f'data/{country}/p2_data_understanding/df_match_player.xlsx', index_col=0)
        df3 = pd.read_excel(f'data/{country}/p2_data_understanding/df_match_odds.xlsx', index_col=0)

        if verbose >= 1:
            logger.info(f"Country: {country}")
            logger.info(f"df_match: {df1.shape}. df_match_player: {df2.shape} df_match_odds: {df3.shape}")

        # Missing data
        if missing:
            df4 = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/all/df_match_miss.xlsx', index_col=0)
            df5 = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/all/df_match_player_miss.xlsx', index_col=0)
            df6 = pd.read_excel(f'data/{country}/p6_deployment/missing/data_understanding/all/df_match_odds_miss.xlsx', index_col=0)

            if verbose >= 1:
                logger.info(f"Missing data: df_match: {df4.shape}. df_match_player: {df5.shape} df_match_odds: {df6.shape}")

            df4_cl = df4[~df4.index.isin(df1.index)] 
            df5_cl = df5[~df5.index.isin(df2.index)]
            df6_cl = df6[~df6.index.isin(df3.index)]

            # Concateno datos missing
            df1 = pd.concat([df1, df4_cl], axis=0)
            df2 = pd.concat([df2, df5_cl], axis=0)
            df3 = pd.concat([df3, df6_cl], axis=0)

        # Concateno datos
        df_ct1 = pd.concat([df_ct1, df1], axis=0)
        df_ct2 = pd.concat([df_ct2, df2], axis=0)
        df_ct3 = pd.concat([df_ct3, df3], axis=0)

        if verbose >= 1:
            logger.info(f"Concat data: df_match_ct: {df_ct1.shape}. df_match_player_ct: {df_ct2.shape} df_match_odds_ct: {df_ct3.shape}")

    return df_ct1, df_ct2, df_ct3

if __name__ == "__main__":

    # Definicion de parametros
    d_countries = {6: 'argentina', 48: 'england', 55: 'france', 59: 'germany', 77: 'italy', 148: 'spain', 167: 'usa'}

    # Flashscore
    df_match, df_match_player, df_match_odds = concat_raw_data_by_country(d_countries)
    
    # Exporto datos
    df_match.to_excel('data/flashscore/df_match.xlsx')
    df_match_player.to_excel('data/flashscore/df_match_player.xlsx')
    df_match_odds.to_excel('data/flashscore/df_match_odds.xlsx')
