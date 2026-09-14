"""
Entrenamiento real (no smoke): corre `comprehensive_search` para 1 país con el
grid completo de `define_params_space` (128 combinaciones de datos x 3 modelos:
LogisticRegression, XGBoost, RandomForest) y la evaluación walk-forward
(5 folds x 200 partidos, ver `predictor/config.py`).

Uso:
    python scripts/real_train.py                                  # england, fecha de hoy
    python scripts/real_train.py 55                               # otro id_country
    python scripts/real_train.py 48 2026-09-11                    # fecha fija (reusa cache)
    python scripts/real_train.py 48 2026-09-11 "calibracion isotonica"

El 2º argumento (`iteration_date`) es lo que hace comparables dos corridas: con
una fecha nueva el pipeline rearma el mapeo global de sofifa y el formateo desde
los .xlsx crudos y tarda 3-8 h **antes** de empezar a entrenar; con una fecha ya
usada reusa esa cache. Para el ciclo "meto un cambio y veo si mejora la métrica"
hay que pasar SIEMPRE la misma fecha, si no se mezcla el efecto del cambio con el
de haber regenerado los datos. Ver docs/EXPERIMENTOS.md.

El 3º argumento es una nota libre que queda en `data/_shared/logs/_training_log.xlsx`:
poner qué cambió respecto de la corrida anterior.

Salidas (en `data/{country}/modeling/{date}/`):
    df_iteration.xlsx       1 fila por combinación x modelo, métricas promediadas
                            entre folds + `std_<métrica>` + `n_folds`. Insumo de
                            `predictor.modeling.main_select_model`.
    df_ite_test_folds.xlsx  el detalle sin promediar, 1 fila por fold.
"""
import sys
import datetime

import pandas as pd

import predictor.modeling.main_train_models as mtm
from predictor.modeling.main_train_models import define_params_space, concat_dataframes_on_iteration

ID_COUNTRY = int(sys.argv[1]) if len(sys.argv) > 1 else 48
D_COUNTRIES = {48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa", 6: "argentina"}
country = D_COUNTRIES[ID_COUNTRY]
if len(sys.argv) > 2:
    date = datetime.datetime.strptime(sys.argv[2], '%Y-%m-%d').date()
else:
    date = datetime.datetime.now().date()
notes = sys.argv[3] if len(sys.argv) > 3 else ""

# comprehensive_search usa `id_country` como global del módulo (no como parámetro)
mtm.id_country = ID_COUNTRY

d_params, l_modelos = define_params_space(ID_COUNTRY)

print(f"REAL TRAIN · country={country} ({ID_COUNTRY}) · date={date}" + (f" · notes={notes!r}" if notes else ""))
print(f"Grid: {d_params}")
print(f"Modelos: {[type(m).__name__ for m in l_modelos]}")
print(f"Iteraciones totales: {mtm.define_n_iterations(d_params)}")

df_params_ite, df_ite_train, df_ite_test = mtm.comprehensive_search(
    country=country,
    date=date,
    d_params=d_params,
    l_modelos=l_modelos,
    data_unders=True,
    update_missing=False,
    data_prep_int=True,
    data_prep_int_miss=False,
    update_sofifa=False,
    retrain=True,
    verbose=0,  # verbose>=1 dispara du.describe_data() -> sns.pairplot; evitarlo en corridas largas
    run_type="train",
    notes=notes,
)

df_iteration = concat_dataframes_on_iteration(df_params_ite, df_ite_train, df_ite_test)
df_iteration.to_excel(f"./data/{country}/modeling/{date}/df_iteration.xlsx", index=False)

print("\n================ RESULTADO ================")
print(f"df_iteration: {df_iteration.shape}")
cols_metricas = [c for c in ("f1_score", "std_f1_score", "test_accuracy", "std_test_accuracy",
                             "roi", "std_roi", "n_folds") if c in df_iteration.columns]
print(df_iteration.sort_values("f1_score", ascending=False).head(10)[cols_metricas + [
    c for c in ("model_name",) if c in df_iteration.columns]].to_string())
