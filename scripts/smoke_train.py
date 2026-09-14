"""
Smoke test de entrenamiento: corre `comprehensive_search` para 1 país con el grid
de hiperparámetros más chico posible (1 iteración, 1 modelo). Sirve para validar
que el pipeline de train corre end-to-end tras un refactor, no para obtener un
modelo bueno.

Uso:
    python scripts/smoke_train.py                       # england, fecha de hoy
    python scripts/smoke_train.py 55                    # otro id_country
    python scripts/smoke_train.py 48 2026-09-11         # fecha fija (reusa cache)
    python scripts/smoke_train.py 48 2026-09-11 "post p4-4"   # + nota en el historial

El 2º argumento (`iteration_date`) es clave para comparar corridas: con una
fecha nueva, el pipeline rearma el mapeo global de sofifa y el formateo desde
los .xlsx crudos y tarda 3-8 h; con una fecha ya usada reusa esa cache y tarda
minutos. Para el ciclo "meto un cambio y veo si mejora la métrica" hay que
pasar SIEMPRE la misma fecha, si no se mezcla el efecto del cambio con el de
haber regenerado los datos.
"""
import sys
import datetime

from sklearn.linear_model import LogisticRegression

import predictor.modeling.main_train_models as mtm
from predictor.config import SEED
from predictor.data_preparation.select_data import determine_country_competitions

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

d_comps = determine_country_competitions(ID_COUNTRY)

# Grid mínimo: 1 valor por hiperparámetro
d_params = {
    "clean_post_integrate": {
        "competencies_to_select": [d_comps["all_comp"]],
        "n_years_to_select": [5],
    },
    "construct": {
        "n_last_matches": [[60]],
        "n_years_h2h": [2],
        "segun_localia": [False],
        "calculate_dif": [True],
        "decay_rate": [0.1],
    },
    "select": {
        "thr_corr": [0.7],
        "thr_fs": [0.05],
        "fill_na": ["0"],
    },
    "modeling": {
        "n_reg_val": [100],
        "n_reg_test": [200],  # tamaño de cada fold del walk-forward (5 folds -> 1000 partidos de test)
        "bal_type": ["under"],
        "k": [5],
    },
}
l_modelos = [LogisticRegression(random_state=SEED)]

print(f"SMOKE TRAIN · country={country} ({ID_COUNTRY}) · date={date} · 1 iter · LogisticRegression"
      + (f" · notes={notes!r}" if notes else ""))

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
    verbose=0,  # verbose>=1 dispara du.describe_data() -> sns.pairplot sobre df de 121 cols -> se cuelga
    run_type="smoke",
    notes=notes,
)

print("\n================ RESULTADO ================")
print("df_ite_test:")
print(df_ite_test.to_string() if len(df_ite_test) else "(vacío — no se entrenó ningún modelo)")
