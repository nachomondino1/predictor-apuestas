"""
Tests de la estrategia de apuesta "sin ea" (p4-7, ver docs/REFACTOR.md): un
único valor fijo de hiperparámetros por contexto (train/prod), sin búsqueda
ni selección por país/resultado (eso duplicaba el overfitting de la
selección de modelo, según la conclusión de la iter4).
"""
import pandas as pd
import pytest

from predictor.modeling.betting_strategy import BettingStrategy


def make_bs():
    # Sin country/iteration_date no se tocan directorios (ver __init__).
    return BettingStrategy()


def test_define_hiperparameters_train():
    bs = make_bs()
    dic = bs.define_hiperparameters(strategy="train")
    assert dic == {"prob_dp": None, "curva": "linear", "m": 10, "b": 0}


def test_define_hiperparameters_prod():
    bs = make_bs()
    dic = bs.define_hiperparameters(strategy="prod")
    assert dic == {"prob_dp": None, "curva": "kelly_linear", "m": 10, "b": 0, "k": 1}


def test_define_hiperparameters_estrategia_invalida():
    bs = make_bs()
    with pytest.raises(ValueError):
        bs.define_hiperparameters(strategy="lo_que_sea")


def test_stake_reduction_no_apuesta_en_local():
    """La única salvedad 'de la realidad' sobre la estrategia fija: no apostar
    en local (result_to_bet == 1), independientemente del stake calculado."""
    bs = make_bs()
    df = pd.DataFrame({
        "result_to_bet": [1, 0, 2],
        "player_emergency_fill": [0, 0, 0],
        "stake_to_bet": [50.0, 50.0, 50.0],
    })
    out = bs.stake_reduction(df)
    assert out["stake_to_bet"].tolist() == [0.0, 50.0, 50.0]


def test_stake_reduction_player_emergency_fill():
    bs = make_bs()
    df = pd.DataFrame({
        "result_to_bet": [0, 0, 2],
        "player_emergency_fill": [0, 1, 0],
        "stake_to_bet": [50.0, 50.0, 50.0],
    })
    out = bs.stake_reduction(df)
    assert out["stake_to_bet"].tolist() == [50.0, 0.0, 50.0]


def test_stake_reduction_ambas_salvedades_combinadas():
    bs = make_bs()
    df = pd.DataFrame({
        "result_to_bet": [1, 0, 2, 1],
        "player_emergency_fill": [0, 1, 0, 1],
        "stake_to_bet": [50.0, 50.0, 50.0, 50.0],
    })
    out = bs.stake_reduction(df)
    assert out["stake_to_bet"].tolist() == [0.0, 0.0, 50.0, 0.0]
