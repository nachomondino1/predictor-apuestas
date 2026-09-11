import pandas as pd
from predictor.data_preparation import construct_data


def main(id_competition):

    d_rows_ho = {}
    d_rows_aw = {}

    # Levantar predicciones historicas
    # df = pd.read_excel("/Users/nachomondino/Desktop/historial_predicciones.xlsx", index_col=0)
    df = pd.read_excel('data/_dashboard/df.xlsx')
    print(df)

    # Determino expected result
    df = construct_data.determine_expected_result(df, verbose=1)

    # Filtrar por temporada y pais
    df_filt = df[df['id_competition'] == id_competition]
    print(df_filt.shape)

    # Por equipo
    l_teams = df_filt['id_team_home'].unique()

    for team in l_teams:
        # print(team)

        # Partidos que jugo de local
        df_team_home = df_filt[df_filt['id_team_home'] == team]
        df_team_away = df_filt[df_filt['id_team_away'] == team]
        # print(df_team_home.shape)
        # print(df_team_away.shape)

        # Calculo tablas
        d_rows_ho[team] = calculate_table(df=df_team_home, home=True)
        d_rows_aw[team] = calculate_table(df=df_team_away, home=False)

    table_home = pd.DataFrame.from_dict(d_rows_ho, orient='index')
    table_away = pd.DataFrame.from_dict(d_rows_aw, orient='index')

    table = table_home.add(table_away, fill_value=0)  # Sumar ambas tablas, rellenando con 0 donde falte información
    table_ct = table_home.join(table_away, how='outer', lsuffix='_home', rsuffix='_away')

    table_home = table_home.sort_values(by='points', ascending=False)
    table_away = table_away.sort_values(by='points', ascending=False)
    table = table.sort_values(by='points', ascending=False)

    table_home.to_excel(f"data/_expected_table/{id_competition}_home.xlsx")
    table_away.to_excel(f"data/_expected_table/{id_competition}_away.xlsx")
    table.to_excel(f"data/_expected_table/{id_competition}.xlsx")
    table_ct.to_excel(f"data/_expected_table/{id_competition}_ct.xlsx")

def calculate_table(df, home: bool):

    suffix = 'home' if home else 'away'
    suffix_ag = 'away' if home else 'home'
    res_win = 1 if home else 2
    res_loss = 2 if home else 1
    
    # Results
    n_wins = len(df[df['result'] == res_win])
    n_draws = len(df[df['result'] == 0])
    n_loss = len(df[df['result'] == res_loss])
    n_matches = len(df)
    points = 3 * n_wins + n_draws
    
    # Goals
    n_goals = df[f'goals_{suffix}'].sum()
    n_goals_against = df[f'goals_{suffix_ag}'].sum()
    dif_goals = n_goals - n_goals_against

    # Expected results
    x_n_wins = len(df[df['expected_result'] == res_win])
    x_n_draws = len(df[df['expected_result'] == 0])
    x_n_loss = len(df[df['expected_result'] == res_loss])
    x_n_matches = len(df)
    x_points = 3 * x_n_wins + x_n_draws
    
    # X_goals
    x_n_goals = df[f'expected_goals_(xg)_{suffix}'].sum()
    x_n_goals_against = df[f'expected_goals_(xg)_{suffix_ag}'].sum()
    dif_x_goals = x_n_goals - x_n_goals_against

    d = {"n_matches": n_matches, "n_wins": n_wins, "n_draws": n_draws, "n_loss": n_loss, "points": points, 
         'GF': n_goals, 'GC': n_goals_against, 'dif': dif_goals,
          "x_n_matches": x_n_matches, "x_n_wins": x_n_wins, "x_n_draws": x_n_draws, "x_n_loss": x_n_loss, "x_points": x_points, 
           'x_GF': x_n_goals, 'x_GC': x_n_goals_against, 'x_dif': dif_x_goals
         }
    return d


# Código que se ejecuta solo cuando el archivo se ejecuta directamente
if __name__ == "__main__":

    l_comps = [481, 551, 591, 771, 1481] 

    for comp in l_comps:
        main(id_competition=comp)