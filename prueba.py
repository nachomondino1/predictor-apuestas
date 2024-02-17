import pandas as pd


country = "england"
id_country = 48
id_competition = 101
is_cup = 0

df = pd.read_excel('./p2_data_understanding/data/england/df_match.xlsx', index_col=0)
print(df.head(1))
print(df.shape)

# Add columns: id_country, is_cup and id_competition
df['id_country'] = id_country
df['id_competition'] = id_competition
df['is_cup'] = is_cup
print(df.shape)

df.to_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index=True)

"""
# DATAFRAME MATCH
df = pd.read_excel('./p2_data_understanding/data/england/df_match.xlsx', index_col=0)
print(df.head(1))
print(df.shape)

# Elimino odds
df_match_odds = df.loc[:, ['odds_home', 'odds_draw', 'odds_away']]
df_match_concat = df.drop(['odds_home', 'odds_draw', 'odds_away'], axis=1)
print(df_match_concat.shape)

df_match_concat.to_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index=True)
df_match_odds.to_excel(f'./p2_data_understanding/data/{country}/df_match_odds.xlsx', index=True)
"""

"""
# DATAFRAME MATCH PLAYER
df = pd.read_excel('./p2_data_understanding/data/england/df_match_player.xlsx')
print(df.head(1))
print(df.shape)

df = df.set_index('id_match')
print(df.head(1))
print(df.shape)

df.to_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index=True)
"""

'''
df = pd.read_excel('./p2_data_understanding/data/england/df_match_odds.xlsx')
print(df.head(1))
print(df.shape)
'''

"""
# DATAFRAME MATCH
df = pd.read_excel('./p2_data_understanding/data/england/df_match_odds.xlsx')
print(df.head(1))
print(df.shape)

df = df.set_index('id_match')
print(df.head(1))
print(df.shape)

df.to_excel(f'./p2_data_understanding/data/{country}/df_match_odds.xlsx', index=True)
"""