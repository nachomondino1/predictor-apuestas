"""
Smoke test de entrenamiento: corre `comprehensive_search` para 1 país con el grid
de hiperparámetros más chico posible (1 iteración, 1 modelo). Sirve para validar
que el pipeline de train corre end-to-end tras un refactor, no para obtener un
modelo bueno.

Uso:
    python scripts/smoke_train.py            # england
    python scripts/smoke_train.py 55         # otro id_country
"""
import sys
import datetime

from sklearn.linear_model import LogisticRegression

import predictor.modeling.main_train_models as mtm
from predictor.data_preparation.select_data import determine_country_competitions

ID_COUNTRY = int(sys.argv[1]) if len(sys.argv) > 1 else 48
D_COUNTRIES = {48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa", 6: "argentina"}
country = D_COUNTRIES[ID_COUNTRY]
date = datetime.datetime.now().date()

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
        "n_reg_test": [100],
        "bal_type": ["under"],
        "k": [5],
    },
}
l_modelos = [LogisticRegression()]

print(f"SMOKE TRAIN · country={country} ({ID_COUNTRY}) · date={date} · 1 iter · LogisticRegression")

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
)

print("\n================ RESULTADO ================")
print("df_ite_test:")
print(df_ite_test.to_string() if len(df_ite_test) else "(vacío — no se entrenó ningún modelo)")
