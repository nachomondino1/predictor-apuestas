"""
Primer entrenamiento real (no smoke): corre `comprehensive_search` para 1 país
con el grid completo de `define_params_space` (128 combinaciones de datos x 3
modelos: LogisticRegression, XGBoost, RandomForest). Pensado para correr en
background — puede tardar bastante (construct_data todavía sin vectorizar,
p3-1 pendiente).

Resultado: `data/{country}/p4_modeling/{date}/df_iteration.xlsx`, que es el
insumo de `predictor.modeling.main_select_model` para elegir qué modelo deployar
(ya selecciona por ROI real de test + expected_error, ver docs/REFACTOR.md
ítem p4-1).

Uso:
    python scripts/real_train.py            # england
    python scripts/real_train.py 55         # otro id_country
"""
import sys
import datetime

import pandas as pd

import predictor.modeling.main_train_models as mtm
from predictor.modeling.main_train_models import define_params_space, concat_dataframes_on_iteration

ID_COUNTRY = int(sys.argv[1]) if len(sys.argv) > 1 else 48
D_COUNTRIES = {48: "england", 55: "france", 59: "germany", 77: "italy", 148: "spain", 167: "usa", 6: "argentina"}
country = D_COUNTRIES[ID_COUNTRY]
date = datetime.datetime.now().date()

# comprehensive_search usa `id_country` como global del módulo (no como parámetro)
mtm.id_country = ID_COUNTRY

d_params, l_modelos = define_params_space(ID_COUNTRY)

print(f"REAL TRAIN · country={country} ({ID_COUNTRY}) · date={date}")
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
)

df_iteration = concat_dataframes_on_iteration(df_params_ite, df_ite_train, df_ite_test)
df_iteration.to_excel(f"./data/{country}/p4_modeling/{date}/df_iteration.xlsx", index=False)

print("\n================ RESULTADO ================")
print(f"df_iteration: {df_iteration.shape}")
print(df_iteration.sort_values("f1_score", ascending=False).head(10).to_string())
