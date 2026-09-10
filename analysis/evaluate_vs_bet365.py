"""
Evaluación retrospectiva: ¿las predicciones del modelo le ganan a bet365?

Fuente: dump MySQL de producción (tabla `historial_predicciones`), que trae para
cada partido ya jugado: probabilidades del modelo, cuotas de bet365, la apuesta
que decidió la estrategia (`result_to_bet` / `stake_to_bet` / `odd_to_bet`), el
resultado real (`result`) y si se acertó (`acerte`).

NO scrapea nada. Solo lee el .sql.

Uso:
    python analysis/evaluate_vs_bet365.py [ruta_al_dump.sql] [--md salida.md]

Por defecto lee data/backup_predictor_apuestas.sql y escribe
docs/EVALUACION_VS_BET365.md
"""
from __future__ import annotations
import re
import argparse
import pathlib
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SQL = REPO / "data" / "backup_predictor_apuestas.sql"
DEFAULT_MD = REPO / "docs" / "EVALUACION_VS_BET365.md"
TABLE = "historial_predicciones"

NUM_COLS = [
    "result", "predicted_result", "result_to_bet", "acerte",
    "goals_home", "goals_away",
    "odds_home", "odds_draw", "odds_away", "odd_to_bet",
    "stake_to_bet", "stake_to_bet_raw", "bet_yield",
    "prob_class_0", "prob_class_1", "prob_class_2",
    "prob_home_bm", "prob_draw_bm", "prob_away_bm",
]


# --------------------------------------------------------------------------- #
# Parseo del dump
# --------------------------------------------------------------------------- #
def _columns(sql: str, table: str) -> list[str]:
    m = re.search(rf"CREATE TABLE `{re.escape(table)}` \((.*?)\n\) ENGINE", sql, re.S)
    if not m:
        raise ValueError(f"No encontré CREATE TABLE `{table}` en el dump")
    return re.findall(r"^\s*`([^`]+)`\s", m.group(1), re.M)


def _rows(sql: str, table: str) -> list[list]:
    m = re.search(rf"INSERT INTO `{re.escape(table)}` VALUES (.*?);\s*\n", sql, re.S)
    if not m:
        return []
    blob, i, n, out = m.group(1), 0, len(m.group(1)), []
    while i < n:
        assert blob[i] == "(", blob[i - 20:i + 20]
        i += 1
        vals, cur, inq = [], [], False
        while True:
            c = blob[i]
            if inq:
                if c == "\\":
                    cur.append(blob[i:i + 2]); i += 2; continue
                if c == "'":
                    if blob[i + 1:i + 2] == "'":
                        cur.append("''"); i += 2; continue
                    inq = False; cur.append(c); i += 1; continue
                cur.append(c); i += 1; continue
            if c == "'":
                inq = True; cur.append(c); i += 1; continue
            if c == ",":
                vals.append("".join(cur)); cur = []; i += 1; continue
            if c == ")":
                vals.append("".join(cur)); i += 1; break
            cur.append(c); i += 1
        out.append([_val(v) for v in vals])
        while i < n and blob[i] in ", \n":
            i += 1
    return out


def _val(v: str):
    v = v.strip()
    if v == "NULL":
        return None
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        return v[1:-1].replace("\\'", "'").replace("''", "'").replace('\\"', '"').replace("\\\\", "\\")
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def load_historial(sql_path: pathlib.Path) -> pd.DataFrame:
    sql = sql_path.read_text(encoding="utf-8", errors="replace")
    df = pd.DataFrame(_rows(sql, TABLE), columns=_columns(sql, TABLE))
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def bookie_favourite(odds_hda: np.ndarray) -> np.ndarray:
    """Índice de resultado (1=local, 0=empate, 2=visita) con menor cuota."""
    return np.array([1, 0, 2])[np.argmin(odds_hda, axis=1)]


def devig_probs(odds_hda: np.ndarray) -> np.ndarray:
    imp = 1.0 / odds_hda
    return imp / imp.sum(axis=1, keepdims=True)


def flat_roi(pick: np.ndarray, result: np.ndarray, odds_hda: np.ndarray) -> tuple[float, int]:
    """ROI apostando 1 unidad plana al `pick` de cada partido."""
    col = np.where(pick == 1, odds_hda[:, 0], np.where(pick == 0, odds_hda[:, 1], odds_hda[:, 2]))
    ok = np.isfinite(col)
    pnl = np.where(pick[ok] == result[ok], col[ok] - 1.0, -1.0)
    return float(pnl.sum() / ok.sum()), int(ok.sum())


def evaluate(df: pd.DataFrame) -> dict:
    s = df[df["acerte"].notna() & df["result"].notna()].copy()
    has_odds = s[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1)
    so = s[has_odds].copy()
    odds = so[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    fav = bookie_favourite(odds)
    res = so["result"].to_numpy()

    R: dict = {}
    R["n_total"] = int(len(df))
    R["n_settled"] = int(len(s))
    R["n_with_odds"] = int(len(so))
    R["date_min"] = s["date"].min()
    R["date_max"] = s["date"].max()

    # 1) accuracy 1X2
    R["acc_model"] = float((so["predicted_result"].to_numpy() == res).mean())
    R["acc_bet365"] = float((fav == res).mean())
    R["acc_bet_placed"] = float(s["acerte"].mean())          # acierto de la apuesta real
    R["acc_always_home"] = float((res == 1).mean())

    # 2) calibración
    P = so[["prob_class_1", "prob_class_0", "prob_class_2"]].to_numpy(float)   # home, draw, away
    imp = devig_probs(odds)
    y = so["result"].map({1: 0, 0: 1, 2: 2}).to_numpy()
    mask = np.isfinite(P).all(1) & (np.nansum(P, 1) > 0) & np.isfinite(y)
    Pn = np.clip(P[mask] / P[mask].sum(1, keepdims=True), 1e-6, 1)
    In = np.clip(imp[mask], 1e-6, 1)
    yy = y[mask].astype(int)
    oh = np.eye(3)[yy]
    R["n_calib"] = int(mask.sum())
    R["logloss_model"] = float(-np.log(Pn[np.arange(len(yy)), yy]).mean())
    R["logloss_bet365"] = float(-np.log(In[np.arange(len(yy)), yy]).mean())
    R["brier_model"] = float(((Pn - oh) ** 2).sum(1).mean())
    R["brier_bet365"] = float(((In - oh) ** 2).sum(1).mean())

    # 3) ROI de la estrategia real del sistema
    b = s[s["stake_to_bet"].fillna(0) > 0].copy()
    bb = b.dropna(subset=["odd_to_bet"])
    pnl = np.where(bb["acerte"] == 1, bb["stake_to_bet"] * (bb["odd_to_bet"] - 1), -bb["stake_to_bet"])
    R["n_bets"] = int(len(b))
    R["staked"] = float(bb["stake_to_bet"].sum())
    R["pnl_strategy"] = float(pnl.sum())
    R["roi_strategy"] = float(pnl.sum() / bb["stake_to_bet"].sum())
    R["roi_strategy_bet_yield"] = float(
        pd.to_numeric(b["bet_yield"], errors="coerce").sum() / b["stake_to_bet"].sum()
    )

    # 4) ROI flat 1u
    R["roi_flat_model"], _ = flat_roi(so["predicted_result"].to_numpy(), res, odds)
    R["roi_flat_bet365_fav"], _ = flat_roi(fav, res, odds)
    R["roi_flat_home"], _ = flat_roi(np.ones(len(so), int), res, odds)

    # 5) desagregado temporal y por país
    so = so.assign(
        q=so["date"].dt.to_period("Q").astype(str),
        hit_m=(so["predicted_result"].to_numpy() == res),
        hit_b=(fav == res),
    )
    R["by_quarter"] = so.groupby("q").agg(
        n=("result", "size"), acc_modelo=("hit_m", "mean"), acc_bet365=("hit_b", "mean")
    ).round(4)
    R["by_country"] = so.groupby("country").agg(
        n=("result", "size"), acc_modelo=("hit_m", "mean"), acc_bet365=("hit_b", "mean")
    ).round(4)
    return R


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def _df_to_md(df: pd.DataFrame) -> str:
    """Tabla markdown sin depender de `tabulate`. El índice va como 1ª columna."""
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


AUTO_MARKER = "<!-- === bloque autogenerado: todo lo de abajo se reescribe === -->"


def render_md(R: dict, sql_path: pathlib.Path) -> str:
    L = []
    L.append(AUTO_MARKER + "\n")
    L.append("## Cifras (autogeneradas)\n")
    try:
        src = sql_path.relative_to(REPO)
    except ValueError:
        src = sql_path
    L.append(f"- Fuente: `{src}` (tabla `{TABLE}`)")
    L.append(f"- Predicciones en el dump: **{R['n_total']}** · con resultado (settled): "
             f"**{R['n_settled']}** · con las 3 cuotas: **{R['n_with_odds']}**")
    L.append(f"- Rango: **{R['date_min'].date()} → {R['date_max'].date()}**\n")

    L.append("## 1. Acierto 1X2 (menor cuota = favorito de bet365)\n")
    L.append("| | acierto |")
    L.append("|---|---|")
    L.append(f"| Modelo (`predicted_result`) | **{R['acc_model']:.1%}** |")
    L.append(f"| bet365 (favorito por cuota) | **{R['acc_bet365']:.1%}** |")
    L.append(f"| Apuesta real del sistema (`acerte`) | {R['acc_bet_placed']:.1%} |")
    L.append(f"| Baseline «siempre local» | {R['acc_always_home']:.1%} |")
    L.append(f"\n→ El modelo **{'supera' if R['acc_model'] > R['acc_bet365'] else 'no supera'}** "
             f"a bet365 en acierto puro ({R['acc_model']:.1%} vs {R['acc_bet365']:.1%}).\n")

    L.append("## 2. Calidad de las probabilidades (menor = mejor)\n")
    L.append(f"Sobre {R['n_calib']} partidos, contra las probabilidades de bet365 "
             "de-vigadas (cuotas normalizadas).\n")
    L.append("| métrica | modelo | bet365 |")
    L.append("|---|---|---|")
    L.append(f"| log-loss | {R['logloss_model']:.4f} | {R['logloss_bet365']:.4f} |")
    L.append(f"| Brier | {R['brier_model']:.4f} | {R['brier_bet365']:.4f} |")
    L.append(f"\n→ Las probabilidades del modelo son **{'mejores' if R['logloss_model'] < R['logloss_bet365'] else 'peores'}** "
             "que las cuotas de-vigadas.\n")

    L.append("## 3. ROI de la estrategia de apuesta del sistema\n")
    L.append(f"- Apuestas con `stake_to_bet > 0`: **{R['n_bets']}** · total apostado: "
             f"{R['staked']:.0f} u")
    L.append(f"- P/L recomputado (`stake·(odd−1)` / `−stake`): **{R['pnl_strategy']:+.1f} u** "
             f"→ ROI **{R['roi_strategy']:+.2%}**")
    L.append(f"- ROI según la columna `bet_yield` del sistema: **{R['roi_strategy_bet_yield']:+.2%}**\n")

    L.append("## 4. ROI a stake plano (1 u), subset con cuotas\n")
    L.append("| estrategia | ROI |")
    L.append("|---|---|")
    L.append(f"| Seguir al modelo (`predicted_result`) | {R['roi_flat_model']:+.2%} |")
    L.append(f"| Seguir al favorito de bet365 | {R['roi_flat_bet365_fav']:+.2%} |")
    L.append(f"| Apostar siempre local | {R['roi_flat_home']:+.2%} |")
    L.append("")

    L.append("## 5. Acierto por trimestre\n")
    L.append(_df_to_md(R["by_quarter"]))
    L.append("\n## 6. Acierto por país\n")
    L.append(_df_to_md(R["by_country"]))
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sql", nargs="?", default=str(DEFAULT_SQL))
    ap.add_argument("--md", default=str(DEFAULT_MD))
    a = ap.parse_args(argv)

    sql_path = pathlib.Path(a.sql)
    df = load_historial(sql_path)
    R = evaluate(df)

    # consola
    print(f"settled={R['n_settled']}  con_cuotas={R['n_with_odds']}  "
          f"{R['date_min'].date()} → {R['date_max'].date()}")
    print(f"acierto  modelo {R['acc_model']:.1%}  |  bet365 {R['acc_bet365']:.1%}  "
          f"|  apuesta real {R['acc_bet_placed']:.1%}")
    print(f"log-loss modelo {R['logloss_model']:.4f}  |  bet365 {R['logloss_bet365']:.4f}")
    print(f"ROI estrategia  {R['roi_strategy']:+.2%}  (bet_yield {R['roi_strategy_bet_yield']:+.2%})")
    print(f"ROI flat modelo {R['roi_flat_model']:+.2%}  |  flat favorito {R['roi_flat_bet365_fav']:+.2%}")

    md_path = pathlib.Path(a.md)
    auto = render_md(R, sql_path)
    # Preserva la parte escrita a mano (encima del marcador); solo reescribe lo de abajo.
    prefix = ""
    if md_path.exists():
        existing = md_path.read_text(encoding="utf-8")
        if AUTO_MARKER in existing:
            prefix = existing.split(AUTO_MARKER)[0].rstrip() + "\n\n"
    md_path.write_text(prefix + auto, encoding="utf-8")
    print(f"\nreporte -> {md_path}")


if __name__ == "__main__":
    main()
