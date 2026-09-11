"""
Historial de entrenamientos. Cada corrida de `comprehensive_search`
(entrenamiento real o smoke test) agrega una fila a `data/_shared/logs/_training_log.xlsx`
para poder comparar entrenamientos entre sí (cuál salió mejor, cuánto tardó,
con qué código) y saber dónde quedaron guardados los modelos de esa corrida.

`data/_shared/logs/_training_log.xlsx` es un archivo chico pensado para abrirse a mano en
Excel (por eso queda en .xlsx, no Parquet — ver docs/REFACTOR.md ítem g-6).
"""
import datetime
import subprocess

import pandas as pd

LOG_PATH = "data/_shared/logs/_training_log.xlsx"

# Métricas de df_ite_test que vale la pena resumir en el log, si están presentes.
_SUMMARY_METRICS = ["f1_score", "test_accuracy", "roi", "expected_roi"]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def log_run(
    country: str,
    date,
    n_iter: int,
    l_modelos: list,
    duration_min: float,
    models_path: str,
    df_ite_test: pd.DataFrame = None,
    run_type: str = "train",
    notes: str = "",
) -> dict:
    """
    Agrega una fila al historial de entrenamientos (`data/_shared/logs/_training_log.xlsx`).

    # Parameters
        country: país entrenado. (str)
        date: `iteration_date` usado (carpeta de datos de esa corrida).
        n_iter: cantidad de combinaciones de hiperparámetros corridas. (int)
        l_modelos: instancias de modelos probados (se guardan sus nombres de clase).
        duration_min: minutos que tardó la corrida completa. (float)
        models_path: carpeta donde quedaron los `.pkl` de los modelos de esta corrida.
        df_ite_test: DataFrame de métricas de test, si está disponible, para
            resumir la mejor/promedio métrica lograda. (DataFrame u None)
        run_type: "smoke" | "train" | "retrain" — para distinguir pruebas de
            humo de entrenamientos reales al comparar el historial.
        notes: comentario libre (qué cambió en el código, por qué se corrió, etc).

    # Returns
        La fila agregada, como dict (por si se quiere loguear/imprimir).
    """
    row = {
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "run_type": run_type,
        "country": country,
        "iteration_date": str(date),
        "n_iteraciones": n_iter,
        "modelos": ", ".join(sorted({type(m).__name__ for m in l_modelos})),
        "duration_min": round(duration_min, 1),
        "models_path": models_path,
        "notes": notes,
    }

    if df_ite_test is not None and len(df_ite_test) > 0:
        for col in _SUMMARY_METRICS:
            if col in df_ite_test.columns:
                row[f"best_{col}"] = df_ite_test[col].max()
                row[f"mean_{col}"] = df_ite_test[col].mean()

    try:
        log = pd.read_excel(LOG_PATH)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        log = pd.DataFrame([row])

    log.to_excel(LOG_PATH, index=False)
    return row
