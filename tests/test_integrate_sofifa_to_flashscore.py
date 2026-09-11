"""
Tests de `integrate_player_data_in_match` (reescrita para ser vectorizada — ver
docs/REFACTOR.md). Datos sintéticos chicos, pensados para cubrir los casos que
la versión vieja manejaba con Python puro y que la vectorización tiene que
seguir respetando:

- match normal con datos completos -> mean/sum correctos.
- fallback al FIFA anterior cuando no hay datos para el año del partido.
- jugador sin mapeo Flashscore->Sofifa -> se ignora.
- NaN en un campo (p. ej. wage) -> "envenena" (propaga NaN) solo esa métrica,
  igual que `sum()`/división de Python puro sobre una lista con un NaN.
- un partido por debajo de `n_reg_min` no genera fila en `df_aux` ni columnas
  en `df_match` para esa combinación titularidad/condición.
"""
import numpy as np
import pandas as pd

from predictor.data_preparation.integrate_sofifa_to_flashscore import integrate_player_data_in_match

# n_reg_min de 'miss' es 1 -> alcanza con 1 jugador por partido para simplificar los fixtures.
# Los partidos de fixture son de octubre (mes >= 7 -> "post mercado de pases de invierno"),
# por eso el fifa "actual" es el del año siguiente (ver search_fecha_fifa / _fifa_years_vectorized).
FIFA_YEAR = 25        # fifa vigente para partidos jugados entre jul-2024 y jun-2025
FIFA_YEAR_ANT = 24


def _build_inputs():
    df_match = pd.DataFrame(
        {"date": pd.to_datetime(["2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04"])},
        index=["m1", "m2", "m3", "m4"],
    )
    df_match_player = pd.DataFrame(
        {
            # m1: jugador con datos completos en el fifa del año del partido
            # m2: jugador que solo existe en el fifa anterior -> fallback
            # m3: jugador con wage=NaN en la fuente -> debe propagar NaN solo en wage
            # m4: jugador sin mapeo Flashscore->Sofifa -> se ignora, la fila no debe existir en df_aux
            "id_player_miss_home_1": ["p1", "p2", "p3", "p4"],
        },
        index=["m1", "m2", "m3", "m4"],
    )
    df_map_fs_so = pd.DataFrame(
        {
            "id_player_fs": ["p1", "p2", "p3"],  # "p4" deliberadamente ausente
            "id_player_so": [10, 20, 30],
        }
    )
    df_player_sofifa = pd.DataFrame(
        {"height": [180, 175, 190]},
        index=["10", "20", "30"],
    )
    df_player_fifa_sofifa = pd.DataFrame(
        {
            "id_player": ["10", "20", "30"],
            "fifa_year": [FIFA_YEAR, FIFA_YEAR_ANT, FIFA_YEAR],  # "20" solo tiene el año anterior
            "age": [20, 22, 24],
            "overall_rating": [70.0, 65.0, 60.0],
            "wage": [1000.0, 2000.0, np.nan],  # NaN a propósito para el jugador de m3
            "value": [5_000_000.0, 3_000_000.0, 1_000_000.0],
            "potential": [75.0, 70.0, 60.0],
            "int_reputation": [1, 1, 2],
        }
    )
    return df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa


def test_match_directo_calcula_medias_y_sumas():
    df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa = _build_inputs()
    df_out, df_aux = integrate_player_data_in_match(df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa)

    assert df_out.loc["m1", "mean_age_player_miss_home"] == 20
    assert df_out.loc["m1", "mean_hei_player_miss_home"] == 180
    assert df_out.loc["m1", "sum_rat_player_miss_home"] == 70.0
    assert df_out.loc["m1", "sum_wage_player_miss_home"] == 1000.0
    assert df_aux.loc["m1", "n_player_miss_home"] == 1


def test_fallback_al_fifa_anterior():
    df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa = _build_inputs()
    df_out, _ = integrate_player_data_in_match(df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa)

    # "p2" -> id_so "20" solo tiene datos en FIFA_YEAR_ANT, no en FIFA_YEAR del partido
    assert df_out.loc["m2", "mean_age_player_miss_home"] == 22
    assert df_out.loc["m2", "sum_rat_player_miss_home"] == 65.0


def test_nan_en_una_columna_propaga_solo_esa_columna():
    df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa = _build_inputs()
    df_out, _ = integrate_player_data_in_match(df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa)

    # "p3" tiene wage=NaN en la fuente -> sum_wage debe ser NaN (no 0, no salteado)
    assert pd.isna(df_out.loc["m3", "sum_wage_player_miss_home"])
    # el resto de las métricas de ese mismo partido no deben verse afectadas
    assert df_out.loc["m3", "mean_age_player_miss_home"] == 24
    assert df_out.loc["m3", "sum_rat_player_miss_home"] == 60.0


def test_jugador_sin_mapeo_se_ignora_del_todo():
    df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa = _build_inputs()
    df_out, df_aux = integrate_player_data_in_match(df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa)

    # "p4" no está en df_map_fs_so -> m4 no cumple n_reg_min (0 < 1) para esta combinación
    assert "m4" not in df_aux.index
    if "mean_age_player_miss_home" in df_out.columns:
        assert pd.isna(df_out.loc["m4", "mean_age_player_miss_home"])


def test_debajo_de_n_reg_min_no_genera_fila_en_df_aux():
    """n_reg_min de 'start' es 8; con 1 solo jugador titular no debería computarse nada."""
    df_match = pd.DataFrame({"date": pd.to_datetime(["2024-10-01"])}, index=["m1"])
    df_match_player = pd.DataFrame({"id_player_start_home_1": ["p1"]}, index=["m1"])
    df_map_fs_so = pd.DataFrame({"id_player_fs": ["p1"], "id_player_so": [10]})
    df_player_sofifa = pd.DataFrame({"height": [180]}, index=["10"])
    df_player_fifa_sofifa = pd.DataFrame({
        "id_player": ["10"], "fifa_year": [FIFA_YEAR], "age": [20], "overall_rating": [70.0],
        "wage": [1000.0], "value": [5_000_000.0], "potential": [75.0], "int_reputation": [1],
    })

    df_out, df_aux = integrate_player_data_in_match(df_match, df_match_player, df_map_fs_so, df_player_sofifa, df_player_fifa_sofifa)

    assert df_aux.empty
    assert "mean_age_player_start_home" not in df_out.columns
