import pandas as pd
import numpy as np
from p3_data_preparation import construct_data

country = "england"
id_country = 48

def replace_nan_with_zero(df, col1, col2):
    """
    Replace NaN values with 0 if one of the variables has an integer value and the other is NaN.
    If both variables are NaN, do not replace any values.
    If both variables take integer values, do not replace any values.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the columns.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.

    Returns:
    DataFrame: The DataFrame with NaN values replaced by 0 according to the specified conditions.
    """
    # Replace NaN with 0 if one variable has an integer value and the other is NaN
    condition_1 = df[col1].notnull() & df[col2].isnull()
    condition_2 = df[col2].notnull() & df[col1].isnull()

    df[col1] = np.where(condition_2, 0, df[col1])
    df[col2] = np.where(condition_1, 0, df[col2])
    return df

# Ejemplo de uso
import pandas as pd

# Crear un DataFrame de ejemplo
data = {
    'n_player_miss_home': [1, np.nan, 3, np.nan, np.nan, 5],
    'n_player_miss_away': [np.nan, 2, np.nan, 4, np.nan, 6]
}
df = pd.DataFrame(data)

# Llamar a la función
df = replace_nan_with_zero(df, 'n_player_miss_home', 'n_player_miss_away')
print(df)

# Salida deseada
# 'n_player_miss_home': [1, 0, 3, 0, np.nan, 5],
# 'n_player_miss_away': [0, 2, 0, 4, np.nan, 6]




"""
df = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
df2 = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index_col=0)
print(df.head(1))
print(df.shape)
print()
print(df2.head(1))
print(df2.shape)

df_season = df2.loc[:, "season"]
print(df_season.shape)

df2 = df2.drop(['season'], axis=1)
print(df2.shape)

df_concat = pd.concat([df, df_season], axis=1)
print(df_concat.shape)

df_concat.to_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index=True)
df2.to_excel(f'./p2_data_understanding/data/{country}/df_match_player.xlsx', index=True)
"""

"""
# DATAFRAME MATCH
df = pd.read_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index_col=0)
print(df.head(1))
print(df.shape)

# Elimino odds
df_match_odds = df.loc[:, ['odds_home', 'odds_draw', 'odds_away']]
df_match_concat = df.drop(['odds_home', 'odds_draw', 'odds_away'], axis=1)
print(df_match_concat.shape)

df_match_concat.to_excel(f'./p2_data_understanding/data/{country}/df_match.xlsx', index=True)
df_match_odds.to_excel(f'./p2_data_understanding/data/{country}/df_match_odds.xlsx', index=True)
"""