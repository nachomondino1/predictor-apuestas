import sys
sys.path.append('.')  # Fallaba el import de mainimport pandas as pd
import pandas as pd


def main(id_competition):

    rows = []

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
        
        n_wins_aw = len(df_team_away[df_team_away['expected_result'] == 2])
        n_draws_aw = len(df_team_away[df_team_away['expected_result'] == 0])
        n_loss_aw = len(df_team_away[df_team_away['expected_result'] == 1])

        n_matches = len(df_team_home) + len(df_team_away)
        n_wins = n_wins_ho + n_wins_aw
        n_draws = n_draws_ho + n_draws_aw
        n_loss = n_loss_ho + n_loss_aw

        points = 3 * n_wins + n_draws

        rows.append([team, n_matches, n_wins, n_draws, n_loss, points])

    table = pd.DataFrame(
        rows,
        columns=["team", "n_matches", "n_wins", "n_draws", "n_loss", "points"]
    )
    table = table.sort_values(by='points', ascending=False)
    table.to_excel(f"/Users/nachomondino/Desktop/{id_competition}.xlsx")

main(id_competition=1481)