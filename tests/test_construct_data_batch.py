"""
Tests de `determine_mean_last_matches_difference_batch` (vectorización de
`determine_mean_last_matches_difference` para procesar todas las stats de una
sola pasada por equipo — ver docs/REFACTOR.md, ítem p3-1).

Comparo contra una implementación de referencia deliberadamente simple (loops
puros, sin vectorizar) que replica la semántica documentada:
- promedio ponderado por decaimiento exponencial, ANTIGÜEDAD entre los
  partidos del propio equipo dentro de una ventana de `n_days` (estrictamente
  antes de la fecha del partido);
- el signo se invierte si el equipo jugó de visitante en el partido histórico
  (la variable ya es una diferencia home-away);
- los NaN se excluyen del promedio, y el rank de decaimiento (0 = más
  reciente) se cuenta ENTRE LOS SOBREVIVIENTES de esa variable específica, no
  entre todos los partidos de la ventana (un partido puede tener NaN en una
  stat y no en otra) — este es el caso que la primera versión de la
  vectorización manejaba mal (ver changelog p3-1).
"""
import numpy as np
import pandas as pd

from predictor.data_preparation.construct_data import determine_mean_last_matches_difference_batch


def make_matches(rows):
    """rows: list of (id_match, date, home, away, **stat_values)"""
    records = []
    for r in rows:
        id_match, date, home, away, stats = r
        rec = {'id_match': id_match, 'date': pd.Timestamp(date), 'id_team_home': home, 'id_team_away': away}
        rec.update(stats)
        records.append(rec)
    return pd.DataFrame(records).set_index('id_match')


def test_promedio_ponderado_basico():
    df = make_matches([
        ('m1', '2024-01-01', 'A', 'X', {'dif_goals': 10}),
        ('m2', '2024-01-10', 'Y', 'A', {'dif_goals': 6}),
        ('m3', '2024-01-25', 'A', 'W', {'dif_goals': 4}),  # match a evaluar
    ])
    out = determine_mean_last_matches_difference_batch(
        df.copy(), n_days=30, variables=['dif_goals'], segun_localia=False, decay_rate=0.1, diff=False,
    )
    # A jugo m1 (home, +10) y m2 (away, -6). Orden mas-reciente-primero: m2(-6, rank0), m1(10, rank1).
    w0, w1 = np.exp(0), np.exp(-0.1 * 1)
    expected = (-6 * w0 + 10 * w1) / (w0 + w1)
    got = out.loc['m3', 'mean_last_30_days_dif_goals_home']
    assert np.isclose(got, expected, atol=1e-10), f"got={got} expected={expected}"


def test_rank_entre_sobrevivientes_con_nan_intercalado():
    """
    Caso que la primera version (con bug) manejaba mal: un NaN INTERCALADO
    entre dos partidos válidos corre el rank de decaimiento del segundo, no
    solo lo excluye. positions mas reciente->mas vieja: [10 (valido), NaN, 6 (valido)].
    El rank de "6" debe ser 1 (segundo sobreviviente), no 2 (tercera posicion cruda).
    """
    df = make_matches([
        ('m1', '2024-01-01', 'A', 'X', {'dif_goals': 6}),    # mas vieja, sobreviviente
        ('m2', '2024-01-10', 'A', 'X', {'dif_goals': np.nan}),  # intercalada, NaN
        ('m3', '2024-01-15', 'A', 'X', {'dif_goals': 10}),   # mas reciente, sobreviviente
        ('m4', '2024-01-25', 'A', 'W', {'dif_goals': 4}),    # match a evaluar (home para A)
    ])
    decay_rate = 0.3
    out = determine_mean_last_matches_difference_batch(
        df.copy(), n_days=30, variables=['dif_goals'], segun_localia=False, decay_rate=decay_rate, diff=False,
    )
    got = out.loc['m4', 'mean_last_30_days_dif_goals_home']

    # Referencia a mano: sobrevivientes en orden mas-reciente-primero = [10 (rank0), 6 (rank1)]
    w0, w1 = np.exp(0), np.exp(-decay_rate * 1)
    expected = (10 * w0 + 6 * w1) / (w0 + w1)
    assert np.isclose(got, expected, atol=1e-10), f"got={got} expected={expected}"


def test_ventana_vacia_da_nan():
    df = make_matches([
        ('m1', '2024-01-01', 'A', 'W', {'dif_goals': 4}),  # sin partidos previos
    ])
    out = determine_mean_last_matches_difference_batch(
        df.copy(), n_days=30, variables=['dif_goals'], segun_localia=False, decay_rate=0.1, diff=False,
    )
    assert pd.isna(out.loc['m1', 'mean_last_30_days_dif_goals_home'])


def test_fuera_de_ventana_no_cuenta():
    df = make_matches([
        ('m1', '2023-11-01', 'A', 'X', {'dif_goals': 100}),  # muy viejo, fuera de la ventana de 30 dias
        ('m2', '2024-01-20', 'A', 'X', {'dif_goals': 5}),
        ('m3', '2024-01-25', 'A', 'W', {'dif_goals': 4}),
    ])
    out = determine_mean_last_matches_difference_batch(
        df.copy(), n_days=30, variables=['dif_goals'], segun_localia=False, decay_rate=0.1, diff=False,
    )
    got = out.loc['m3', 'mean_last_30_days_dif_goals_home']
    assert np.isclose(got, 5.0, atol=1e-10)  # solo m2 entra en la ventana


def test_signo_se_invierte_de_visitante():
    df = make_matches([
        ('m1', '2024-01-01', 'X', 'A', {'dif_goals': 10}),  # A jugo de visitante -> se invierte a -10
        ('m2', '2024-01-25', 'A', 'W', {'dif_goals': 4}),
    ])
    out = determine_mean_last_matches_difference_batch(
        df.copy(), n_days=30, variables=['dif_goals'], segun_localia=False, decay_rate=0.1, diff=False,
    )
    got = out.loc['m2', 'mean_last_30_days_dif_goals_home']
    assert np.isclose(got, -10.0, atol=1e-10)


def test_segun_localia_separa_historiales():
    """
    Con segun_localia=True, el historial "de local" de A y el "de visitante"
    de A son independientes: un partido de A de visitante no debe aparecer en
    la ventana de un partido de A de local.
    """
    df = make_matches([
        ('m1', '2024-01-01', 'Y', 'A', {'dif_goals': 8}),   # A de visitante
        ('m2', '2024-01-25', 'A', 'W', {'dif_goals': 4}),   # A de local -> a evaluar (loc)
    ])
    out = determine_mean_last_matches_difference_batch(
        df.copy(), n_days=30, variables=['dif_goals'], segun_localia=True, decay_rate=0.1, diff=False,
    )
    # A no tiene ningun partido "de local" previo -> NaN, aunque tenga un partido de visitante en la ventana.
    assert pd.isna(out.loc['m2', 'loc_mean_last_30_days_dif_goals_home'])


def test_multiples_variables_batch_igual_a_una_por_una():
    """La llamada batcheada con N variables da lo mismo que N llamadas de a una."""
    rng = np.random.default_rng(42)
    n = 40
    teams = [f"T{i}" for i in range(6)]
    dates = pd.date_range("2024-01-01", periods=n, freq="3D")
    rows = []
    for i in range(n):
        home, away = rng.choice(teams, size=2, replace=False)
        stats = {
            "dif_a": rng.normal(),
            "dif_b": rng.normal() if rng.random() > 0.2 else np.nan,  # con NaN salpicado
        }
        rows.append((f"m{i}", dates[i], home, away, stats))
    df = make_matches(rows)

    out_batch = determine_mean_last_matches_difference_batch(
        df.copy(), n_days=20, variables=["dif_a", "dif_b"], segun_localia=False, decay_rate=0.2, diff=True,
    )

    for var in ["dif_a", "dif_b"]:
        out_single = determine_mean_last_matches_difference_batch(
            df.copy(), n_days=20, variables=[var], segun_localia=False, decay_rate=0.2, diff=True,
        )
        col = f"dif_mean_last_20_days_{var}"
        a = out_batch[col].to_numpy(dtype=float)
        b = out_single[col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.all(np.isclose(a, b, equal_nan=False, atol=1e-10) | both_nan)
