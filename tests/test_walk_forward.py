"""
Tests del diseño de evaluación walk-forward (paso 0 del plan de experimentos,
ver docs/REFACTOR.md y predictor/config.py):

- `determine_walk_forward_folds`: folds consecutivos, disjuntos, y con una fecha
  de corte que garantiza que el train de un fold no contenga su futuro.
- `aggregate_folds`: colapsa los folds a 1 fila por modelo, promediando y
  agregando el desvío de las métricas clave.
"""
import numpy as np
import pandas as pd
import pytest

from predictor.modeling.main_train_models import aggregate_folds, determine_walk_forward_folds


def make_df_match(n=1500, id_competition=481):
    """Partidos diarios consecutivos, todos de una liga pública."""
    fechas = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": [f.strftime("%d.%m.%Y %H:%M") for f in fechas],
            "id_competition": [id_competition] * n,
        },
        index=[f"m{i}" for i in range(n)],
    )


def test_folds_son_consecutivos_disjuntos_y_del_tamano_pedido():
    folds = determine_walk_forward_folds(make_df_match(), n_folds=5, fold_size=200, n_reg_val=100)

    assert len(folds) == 5
    assert [len(f["index_test"]) for f in folds] == [200] * 5
    assert [len(f["index_val"]) for f in folds] == [100] * 5

    # Ningún partido aparece en dos folds de test, ni en test y val del mismo fold.
    tests = [set(f["index_test"]) for f in folds]
    for i in range(len(tests)):
        for j in range(i + 1, len(tests)):
            assert not (tests[i] & tests[j]), f"folds {i+1} y {j+1} comparten partidos"
    for f in folds:
        assert not (set(f["index_test"]) & set(f["index_val"]))


def test_fold_1_es_el_mas_reciente_y_los_folds_van_hacia_atras():
    df = make_df_match()
    fechas = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M")
    folds = determine_walk_forward_folds(df, n_folds=3, fold_size=200, n_reg_val=100)

    maximos = [fechas[f["index_test"]].max() for f in folds]
    assert maximos == sorted(maximos, reverse=True), "el fold 1 debería ser el más reciente"


def test_date_cutoff_deja_afuera_el_futuro_del_fold():
    """
    El train de un fold se define por fecha de corte: todo lo ANTERIOR. Así el
    fold 3 no entrena con los folds 1 y 2, que son su futuro.
    """
    df = make_df_match()
    fechas = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M")
    folds = determine_walk_forward_folds(df, n_folds=3, fold_size=200, n_reg_val=100)

    for f in folds:
        idx_train = fechas[fechas < f["date_cutoff"]].index
        # nada del test ni del val del fold puede estar en su train
        assert not (set(idx_train) & set(f["index_test"]))
        assert not (set(idx_train) & set(f["index_val"]))
        # y tampoco nada posterior al test del fold (su futuro)
        assert fechas[idx_train].max() < fechas[f["index_test"]].min()


def test_avisa_y_recorta_cuando_no_alcanzan_los_partidos():
    # 700 partidos no alcanzan para 5 folds de 200 + 100 de val + train
    folds = determine_walk_forward_folds(make_df_match(n=700), n_folds=5, fold_size=200, n_reg_val=100)
    assert 0 < len(folds) < 5


def test_error_si_no_alcanza_ni_para_un_fold():
    with pytest.raises(ValueError):
        determine_walk_forward_folds(make_df_match(n=120), n_folds=5, fold_size=200, n_reg_val=100)


def test_aggregate_folds_promedia_y_calcula_desvio():
    rows = [
        {"n_iteration": 1, "model_name": "LogisticRegression", "n_fold": 1, "f1_score": 40.0, "roi": -10.0},
        {"n_iteration": 1, "model_name": "LogisticRegression", "n_fold": 2, "f1_score": 50.0, "roi": 10.0},
    ]
    out = aggregate_folds(rows)

    assert len(out) == 1
    row = out[0]
    assert row["n_folds"] == 2
    assert row["f1_score"] == pytest.approx(45.0)          # promedio, con el nombre original
    assert row["std_f1_score"] == pytest.approx(5.0)        # ddof=0
    assert row["roi"] == pytest.approx(0.0)
    assert row["std_roi"] == pytest.approx(10.0)
    assert "n_fold" not in row                              # el nº de fold no sobrevive al promedio


def test_aggregate_folds_separa_por_modelo_y_conserva_no_numericas():
    rows = [
        {"model_name": "A", "n_fold": 1, "f1_score": 10.0, "model_hiper": {"C": 1}},
        {"model_name": "A", "n_fold": 2, "f1_score": 20.0, "model_hiper": {"C": 2}},
        {"model_name": "B", "n_fold": 1, "f1_score": 30.0, "model_hiper": {"C": 3}},
    ]
    out = aggregate_folds(rows)

    por_modelo = {r["model_name"]: r for r in out}
    assert set(por_modelo) == {"A", "B"}
    assert por_modelo["A"]["f1_score"] == pytest.approx(15.0)
    assert por_modelo["A"]["n_folds"] == 2
    assert por_modelo["B"]["n_folds"] == 1
    # las no numéricas (hiperparámetros elegidos) se toman del primer fold
    assert por_modelo["A"]["model_hiper"] == {"C": 1}


def test_aggregate_folds_con_lista_vacia():
    assert aggregate_folds([]) == []
