"""
Tests de las funciones puras de parseo de Flashscore (regex sobre hrefs / ids).
Son las más frágiles ante cambios de formato y las más baratas de testear.
"""
from p2_data_understanding.collect_initial_data.scraper_flashscore import (
    clean_id,
    extract_id_from_href,
    extract_name_from_href,
)


def test_clean_id_quita_prefijo():
    assert clean_id(["g_1_fshvzbls"]) == ["fshvzbls"]
    assert clean_id(["g_3_abc123", "g_1_def456"]) == ["abc123", "def456"]
    assert clean_id(["sinprefijo"]) == ["sinprefijo"]


def test_extract_id_from_href_player_y_team():
    assert extract_id_from_href("/player/raya-david/nkVV0IXb") == "nkVV0IXb"
    assert extract_id_from_href("/player/raya-david/nkVV0IXb/") == "nkVV0IXb"
    assert extract_id_from_href("https://www.flashscore.com/team/arsenal/hA1Zm19f/") == "hA1Zm19f"
    assert extract_id_from_href("/coach/whatever/xyz") is None  # patrón solo team|player


def test_extract_name_from_href():
    assert extract_name_from_href("/player/roerslev-rasmussen-mads/pp1zpsrr/") == "roerslev rasmussen mads"
    assert extract_name_from_href("/team/manchester-utd/abc123") == "manchester utd"
    assert extract_name_from_href("no-match") is None
