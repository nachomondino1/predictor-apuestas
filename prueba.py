import pandas as pd


df_match = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/usa/df_match.xlsx', index_col=0)
df_match_player = pd.read_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/usa/df_match_player.xlsx', index_col=0)

print(df_match.head(2))
print(df_match.shape)
print(df_match_player.head(2))
print(df_match_player.shape)

cont = 0
for idx in df_match.index:
    if idx not in df_match_player.index:
        cont += 1
        df_match_player.loc[idx] = None
        # print(df_match_player.tail(1))
    
print(cont)

df_match_player.to_excel('/Users/nachomondino/Documents/GitHub/predictor-apuestas/p2_data_understanding/data/usa/data_seg/df_match_player.xlsx')
print(df_match_player.shape)

'''
from main import DataPreparation


country = "england"
dp = DataPreparation(country=country) # Creo objeto de clase DataPreparation

df_match_miss = pd.read_excel(f'p6_deployment/data/{country}/missing/data_understanding/all/df_match_miss.xlsx', index_col=0)
df_match_player_miss = pd.read_excel(f'p6_deployment/data/{country}/missing/data_understanding/all/df_match_player_miss.xlsx', index_col=0) 
df_player_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_sofifa.xlsx', index_col=0) # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
df_player_fifa_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_player_fifa_sofifa.xlsx') # Podria recolectar nueva version del ultimo fifa. # ACTUALIZAR TAMBIEN
df_teams_sofifa = pd.read_excel(f'./p2_data_understanding/data/{country}/df_teams_sofifa.xlsx', index_col=0)

print(df_match_miss.shape)
print(df_match_player_miss.shape)
print(df_player_sofifa.shape)
print(df_player_fifa_sofifa.shape)

# Preparo datos
df_match_miss, df_match_player_miss, df_player_fifa_sofifa = dp.format_data(df_match_miss, df_match_player_miss, df_player_fifa_sofifa, export=False)
df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa = dp.clean_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False)
df_integrated_missing = dp.integrate_data(df_match_miss, df_match_player_miss, df_player_sofifa, df_player_fifa_sofifa, df_teams_sofifa, export=False) 


df_integrated_missing.to_excel('/Users/nachomondino/Desktop/df_integrated_missing.xlsx', index=True)
'''



'''
import re

def extract_id_from_href(href: str) -> str:
    """
    Extrae el identificador del atributo href.

    Parameters:
        href (str): La cadena href del cual se extraerá el identificador.

    Returns:
        str: El identificador extraído o None si no se encuentra.
    """
    pattern = r"/(team|player)/[^/]+/([^/]+)/?" # (e.g. "/player/raya-david/nkVV0IXb", "/player/raya-david/nkVV0IXb/", "/player/lionel-messi/erigoeriowgjwi", "/team/arsenal/asfjsiafjis/"
    match = re.search(pattern, href)
    if match:
        return match.group(2)
    return None

def extract_name_from_href(href: str) -> str:
    """
    Extrae el nombre del equipo del atributo href.

    Parameters:
        href (str): La cadena href del cual se extraerá el nombre del equipo.

    Returns:
        str: El nombre del equipo extraído o None si no se encuentra.
    """
    pattern = r"/(team|player)/([^/]+)/[^/]+/?"
    match = re.search(pattern, href)
    if match:
        name = match.group(2).replace('-', ' ')
        return name
    return None

# Ejemplos de uso:
hrefs = [
    "/team/arsenal/hA1Zm19f/",
    "/player/emiliano-martinez/KluSTr9s/",
    "/team/manchester-united/ppjDR086/"
]

for href in hrefs:
    print(f"Href: {href}")
    print(f"\tID: {extract_id_from_href(href)}")
    print(f"\tName: {extract_name_from_href(href)}")
'''