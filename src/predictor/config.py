"""
Configuración transversal del pipeline.

# SEED / reproducibilidad

`SEED` es la semilla única de todo lo aleatorio del entrenamiento. Existe para
que **dos corridas del mismo código y la misma config den exactamente el mismo
resultado**, que es la precondición para poder comparar corridas entre sí
(ver docs/REFACTOR.md, "paso 0" del plan de experimentos).

Antes de centralizarla, cuatro cosas quedaban sin sembrar y hacían que el
mismo código diera métricas distintas en cada corrida — medido sobre 6 smokes
idénticos: **f1 entre 40.0 y 45.0, ROI entre −50.6 y −19.2**:

1. `RandomUnderSampler()` / `RandomOverSampler()` (`generate_test_design.py`)
   → cambiaba los datos de train de cada modelo.
2. `RandomForestClassifier()` / `XGBClassifier()` (`main_train_models.py`)
   → 2 de los 3 modelos del grid.
3. `DecisionTreeClassifier()` (`select_data.py`, feature selection)
   → cambiaba las importancias y por ende **qué features se seleccionan**.
4. `LogisticRegression()` (`select_data.py`, feature selection).

Los `train_test_split(...)` del repo ya usaban `random_state=42` literal desde
antes; se dejan así (mismo valor que `SEED`) para no tocar código que no hacía
falta tocar.

⚠️ Cambiar `SEED` cambia los números de todo el histórico: las corridas
anteriores a 2026-09-13 en `data/_shared/logs/_training_log.xlsx` **no son
comparables** con las de después (eran una muestra al azar de la distribución,
no un valor fijo).

Para medir la banda de ruido residual (útil para saber si una mejora de +2 de
f1 es real o no), correr la misma config con varias semillas distintas
pasando `random_state=` explícito, en vez de cambiar esta constante.
"""

SEED = 42


# # Diseño de evaluación: walk-forward

# Cantidad de folds y tamaño (en partidos) de cada fold de test. Los folds son
# bloques consecutivos de partidos ordenados por fecha: el fold 1 son los
# `WALK_FORWARD_FOLD_SIZE` partidos más recientes, el fold 2 los
# `WALK_FORWARD_FOLD_SIZE` anteriores, y así. Para cada fold, el modelo se
# entrena SOLO con partidos anteriores a ese fold (ventana expansiva), igual que
# en producción: nunca se predice con información del futuro.
#
# Reemplaza al split único de 100 partidos que había antes, que era demasiado
# chico para medir: elegir el mejor de 384 modelos por ROI sobre ~73-99 partidos
# es elegir ruido (12 de 384 daban ROI>0, ≈ lo esperable por azar), y explicaba
# la brecha entre el backtest (+74% de ROI) y producción (≈0%).
#
# 5 × 200 = 1000 partidos de evaluación (10x más que antes) y, sobre todo, un
# **desvío estándar entre folds**: sin eso no se puede decir si una mejora de
# +2 de f1 es real o es ruido.
WALK_FORWARD_N_FOLDS = 5
WALK_FORWARD_FOLD_SIZE = 200
