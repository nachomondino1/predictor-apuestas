import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd


def main(id_competition):

    rows = []
    rows_ho = []
    rows_aw = []

    # Levantar predicciones historicas
    # df = pd.read_excel("/Users/nachomondino/Desktop/historial_predicciones.xlsx", index_col=0)
    df = pd.read_excel('data/_dashboard/df.xlsx')
    print(df)

    # Filtrar por temporada y pais
    df_filt = df[df['id_competition'] == id_competition]
    print(df_filt.shape)

    # Por equipo
    l_teams = df_filt['id_team_home'].unique()

    for team in l_teams:
        print(team)

        # Partidos que jugo de local
        df_team_home = df_filt[df_filt['id_team_home'] == team]
        df_team_away = df_filt[df_filt['id_team_away'] == team]
        print(df_team_home.shape)
        print(df_team_away.shape)

        # Sumar puntos segun expected result
        n_wins_ho = len(df_team_home[df_team_home['expected_result'] == 1])
        n_draws_ho = len(df_team_home[df_team_home['expected_result'] == 0])
        n_loss_ho = len(df_team_home[df_team_home['expected_result'] == 2])
        n_matches_ho = len(df_team_home)
        points_ho = 3 * n_wins_ho + n_draws_ho

        n_wins_aw = len(df_team_away[df_team_away['expected_result'] == 2])
        n_draws_aw = len(df_team_away[df_team_away['expected_result'] == 0])
        n_loss_aw = len(df_team_away[df_team_away['expected_result'] == 1])
        n_matches_aw = len(df_team_away)
        points_aw = 3 * n_wins_aw + n_draws_aw

        n_wins = n_wins_ho + n_wins_aw
        n_draws = n_draws_ho + n_draws_aw
        n_loss = n_loss_ho + n_loss_aw
        n_matches = n_matches_ho + n_matches_aw

        points = points_ho + points_aw

        rows.append([team, n_matches, n_wins, n_draws, n_loss, points])
        rows_ho.append([team, n_matches_ho, n_wins_ho, n_draws_ho, n_loss_ho, points_ho])
        rows_aw.append([team, n_matches_aw, n_wins_aw, n_draws_aw, n_loss_aw, points_aw])

    table = pd.DataFrame(
        rows,
        columns=["team", "n_matches", "n_wins", "n_draws", "n_loss", "points"]
    )
    table_home = pd.DataFrame(
        rows_ho,
        columns=["team", 'n_matches_ho', 'n_wins_ho', 'n_draws_ho', 'n_loss_ho', 'points_ho']
    )
    table_away = pd.DataFrame(
        rows_aw,
        columns=["team", 'n_matches_aw', 'n_wins_aw', 'n_draws_aw', 'n_loss_aw', 'points_aw']
    )

    table = table.sort_values(by='points', ascending=False)
    table_ct = pd.merge(table, table_home, on='team')
    table_ct = pd.merge(table_ct, table_away, on='team')
    table_home = table_home.sort_values(by='points_ho', ascending=False)
    table_away = table_away.sort_values(by='points_aw', ascending=False)

    table.index = range(1, len(table) + 1)
    table_home.index = range(1, len(table_home) + 1)
    table_away.index = range(1, len(table_away) + 1)

    table.to_excel(f"data/_expected_table/{id_competition}.xlsx")
    table_home.to_excel(f"data/_expected_table/{id_competition}_home.xlsx")
    table_away.to_excel(f"data/_expected_table/{id_competition}_away.xlsx")
    table_ct.to_excel(f"data/_expected_table/{id_competition}_comp.xlsx")

for comp in [481, 551, 591, 771, 1481]:
    main(id_competition=comp)