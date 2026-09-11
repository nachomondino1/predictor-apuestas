import pandas as pd
import datetime
from predictor.utils import directories


def paths(country, iteration_date):
    # Defino paths
    d_paths = {
        'BASE_DIR_du': f"./data/{country}/data_understanding/old_updated/{iteration_date}",
        # BASE_DIR_flashscore = f'data/{country}/deployment/missing/old_updated'
        # BASE_DIR_sofifa = f'data/{country}/data_understanding/sofifa_update'
        'BASE_DIR_dp': f'data/{country}/data_preparation/{iteration_date}',
        'BASE_DIR_miss': f'data/{country}/deployment/missing/old_updated',
        'BASE_DIR_miss_du': f'data/{country}/deployment/missing/data_understanding/all',
        'BASE_DIR_miss_dp': f'data/{country}/deployment/missing/data_preparation/all',
    }

    directories.make_directories(l_directorios=d_paths.values())
    return d_paths

def main(d_countries, data_unders: bool = False, data_prep: bool = False, deployment: bool = False, sofifa: bool = False):

    df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16 = pd.DataFrame(), pd.DataFrame(),  pd.DataFrame(), pd.DataFrame(),  pd.DataFrame(), pd.DataFrame(),  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),  pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    date = datetime.datetime.now().date()
    d_paths_all = paths(country='all', iteration_date=date)

    # DATA UNDERS
    for id_country, lista in d_countries.items():
        country = lista[0]
        iteration_date = lista[1]
        print(f"Country: {country}")

        d_paths = paths(country, iteration_date)
        BASE_DIR_du = d_paths['BASE_DIR_du']
        BASE_DIR_dp = d_paths['BASE_DIR_dp']
        BASE_DIR_miss = d_paths['BASE_DIR_miss']

        if data_unders:
            
            # Levanto datos de data unders
            df_match = pd.read_excel(f'{BASE_DIR_du}/df_match.xlsx', index_col=0)
            df_match_player = pd.read_excel(f'{BASE_DIR_du}/df_match_player.xlsx', index_col=0)
            df_match_odds = pd.read_excel(f'{BASE_DIR_du}/df_match_odds.xlsx', index_col=0)
            
            if sofifa:
                df_player_sofifa = pd.read_excel(f'{BASE_DIR_du}/df_player_sofifa.xlsx', index_col=0)
                df_player_fifa_sofifa = pd.read_excel(f'{BASE_DIR_du}/df_player_fifa_sofifa.xlsx', index_col=0)
                print(df_player_sofifa.shape, df_player_fifa_sofifa.shape)

            print(df_match.shape, df_match_player.shape,df_match_odds.shape)

            # Concateno datos
            df1 = pd.concat([df1, df_match], axis=0)
            df2 = pd.concat([df2, df_match_player], axis=0)
            df3 = pd.concat([df3, df_match_odds], axis=0)
            
            if sofifa:
                df4 = pd.concat([df4, df_player_sofifa], axis=0)
                df5 = pd.concat([df5, df_player_fifa_sofifa], axis=0)
                print(df4.shape, df5.shape)

            print(df1.shape, df2.shape, df3.shape)

        if data_prep:
            # Levanto datos de data prep
            # df_integrated = pd.read_excel(f'{BASE_DIR_dp}/df_integrated.xlsx', index_col=0)
            df_map_players_fs_so =  pd.read_excel(f'{BASE_DIR_dp}/integrate_data/df_map_players_fs_so.xlsx', index_col=0)
            df_teams = pd.read_excel(f'{BASE_DIR_dp}/integrate_data/df_teams.xlsx', index_col=0)

            # df6 = pd.concat([df6, df_integrated], axis=0)
            df7 = pd.concat([df7, df_map_players_fs_so], axis=0)
            df8 = pd.concat([df8, df_teams], axis=0)

            # Eliminar duplicados? Jugadores que jugaron en mas de 1 pais.
            df7 = df7.drop_duplicates(subset=['id_player_fs'], keep='first')

        if deployment:
            # Levanto datos de deployment
            # df_match_miss = pd.read_excel(f'{BASE_DIR_miss}/df_match.xlsx', index_col=0)
            # df_match_player_miss = pd.read_excel(f'{BASE_DIR_miss}/df_match_player.xlsx', index_col=0)
            # df_match_odds_miss = pd.read_excel(f'{BASE_DIR_miss}/df_match_odds.xlsx', index_col=0)
            # df_integrated_miss = pd.read_excel(f'{BASE_DIR_miss}/df_integrated.xlsx', index_col=0)

            nose1 = pd.read_excel(f'{d_paths['BASE_DIR_miss_du']}/df_match_miss.xlsx', index_col=0)
            nose2 = pd.read_excel(f'{d_paths['BASE_DIR_miss_du']}/df_match_player_miss.xlsx', index_col=0)
            nose3 = pd.read_excel(f'{d_paths['BASE_DIR_miss_du']}/df_match_odds_miss.xlsx', index_col=0)

            nose4 = pd.read_excel(f'{d_paths['BASE_DIR_miss_dp']}/df_integrated_missing.xlsx', index_col=0)

            # Concateno datos
            # df9 = pd.concat([df9, df_match_miss], axis=0)
            # df10 = pd.concat([df10, df_match_player_miss], axis=0)
            # df11 = pd.concat([df11, df_match_odds_miss], axis=0)
            # df12 = pd.concat([df12, df_integrated_miss], axis=0)

            df13 = pd.concat([df13, nose1], axis=0)
            df14 = pd.concat([df14, nose2], axis=0)
            df15 = pd.concat([df15, nose3], axis=0)
            df16 = pd.concat([df16, nose4], axis=0)

    # Exporto datos
    if data_unders:
        df1.to_excel(f'{d_paths_all['BASE_DIR_du']}/df_match.xlsx')
        df2.to_excel(f'{d_paths_all['BASE_DIR_du']}/df_match_player.xlsx')
        df3.to_excel(f'{d_paths_all['BASE_DIR_du']}/df_match_odds.xlsx')
        
        if sofifa:
            df4.to_excel(f'{d_paths_all['BASE_DIR_du']}/df_player_sofifa.xlsx')
            df5.to_excel(f'{d_paths_all['BASE_DIR_du']}/df_player_fifa_sofifa.xlsx')
    
    if data_prep:
        # df6.to_excel(f'{d_paths_all['BASE_DIR_dp']}/df_integrated.xlsx')
        df7.to_excel(f'{d_paths_all['BASE_DIR_dp']}/integrate_data/df_map_players_fs_so.xlsx')
        df8.to_excel(f'{d_paths_all['BASE_DIR_dp']}/integrate_data/df_teams.xlsx')
    
    if deployment:
        # df9.to_excel(f'{d_paths_all['BASE_DIR_miss']}/df_match.xlsx')
        # df10.to_excel(f'{d_paths_all['BASE_DIR_miss']}/df_match_player.xlsx')
        # df11.to_excel(f'{d_paths_all['BASE_DIR_miss']}/df_match_odds.xlsx')
        # df12.to_excel(f'{d_paths_all['BASE_DIR_miss']}/df_integrated.xlsx')
        
        df13.to_excel(f'{d_paths_all['BASE_DIR_miss_du']}/df_match_miss.xlsx')
        df14.to_excel(f'{d_paths_all['BASE_DIR_miss_du']}/df_match_player_miss.xlsx')
        df15.to_excel(f'{d_paths_all['BASE_DIR_miss_du']}/df_match_odds_miss.xlsx')
        df16.to_excel(f'{d_paths_all['BASE_DIR_miss_dp']}/df_integrated_missing.xlsx')

if __name__ == "__main__":

    d_countries = {
        1000: ["europe", '2025-08-26'],
        48: ["england", '2025-08-26'],
        55: ["france", '2025-08-26'], 
        59: ["germany", '2025-08-26'],
        77: ["italy", '2025-08-26'],
        148: ["spain", '2025-08-26']
        }
    
    data_unders = True
    data_prep = False
    deployment = False

    main(d_countries, data_unders=data_unders, data_prep=data_prep, deployment=deployment)