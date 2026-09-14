# Estado actual

> **Actualizado:** 2026-09-14 · **Commit base:** `8150568cc` · **Rama:** `claude-test`
>
> Este doc describe **el presente**: qué funciona, en qué números estamos, qué
> decisiones están abiertas y qué sigue. La historia de cómo llegamos acá está en
> [`REFACTOR.md`](REFACTOR.md); la estructura del código en
> [`ARQUITECTURA.md`](ARQUITECTURA.md). **Si algo de acá cambia, se edita este archivo.**

## 1. Qué funciona hoy

- **Pipeline de entrenamiento end-to-end**: verificado en los **5 países** (smoke
  del 13-sep, 0 errores). `scripts/smoke_train.py` y `scripts/real_train.py`.
- **Reproducibilidad**: dos corridas de la misma config dan resultados
  **bit-idénticos** (`predictor/config.py::SEED`).
- **Evaluación walk-forward**: 5 folds × 200 partidos, ventana expansiva con fecha
  de corte. Reporta promedio **y desvío entre folds**.
- **Historial de corridas**: `data/_shared/logs/_training_log.xlsx`, una fila por
  corrida (commit, país, duración, métricas, carpeta de modelos).
- **Tests**: `pytest` 56/56.
- **Producción / predicción semanal**: sin cambios de comportamiento durante todo
  el refactor (`predictor/deployment/main_next_matches.py`).

## 2. Dónde está el modelo

Medido sobre **2079 partidos reales de producción** (jul-2024 → sep-2025, ver
[`EVALUACION_VS_BET365.md`](EVALUACION_VS_BET365.md)):

| | modelo | bet365 | baseline "siempre local" |
|---|---|---|---|
| Acierto 1X2 | 49.0 % | 53.7 % | 42.9 % |
| log-loss | 1.03 | 0.96 | — |
| ROI stake plano | −0.5 % | −4.25 % | −12.2 % |

**~5 puntos abajo de bet365**, ~6 arriba del baseline naive, y peor calibrado.
Lo único a favor: a stake plano pierde mucho menos que seguir al favorito del
bookie → encuentra value en **no-favoritos**, pero no cubre el overround (~5-6 %).

**El backtest ahora coincide con producción** (accuracy 53.1 del modelo vs 57.3
del bookie en walk-forward). Antes el backtest decía 57.8 y producción 49.0: esa
contradicción era, en parte, un leakage de feature selection + scaler que ya se
arregló. Que cierren es la señal de que la medición es honesta.

**Baseline oficial para comparar mejoras** (england, smoke, 4 folds efectivos,
commit `8150568cc`):

| | promedio | desvío entre folds |
|---|---|---|
| f1 | 45.3 | ± 2.3 |
| accuracy | 53.1 | ± 3.4 |
| ROI | −25.6 | **± 28.7** |

## 3. Decisiones abiertas (las debe tomar el usuario)

### 3.1 Cuántos folds — los datos sostienen 4, no 5

La disponibilidad de features cae hacia atrás en el tiempo (`expected_goals` no
existe en partidos viejos) y `select_data` dropea toda fila con algún NaN. Filas
de train sin NaN por fold en england: **900 / 697 / 440 / 184 / 0**. Hoy el fold 5
se saltea con un error explícito y el promedio sale sobre 4.

Opciones: **(a)** bajar a 3 folds de 200 · **(b)** rellenar NaN antes de la feature
selection · **(c)** recortar las features históricamente ausentes.

### 3.2 Con qué métrica se selecciona el modelo

`main_select_model.py` selecciona por **ROI real de test** (decisión `p4-1`), pero
ROI tiene ±28.7 de desvío sobre una media de −25.6 (por fold: de −48.3 a +23.4).
**Rankear 384 modelos por ROI es rankear suerte.** f1 (±2.3) y accuracy (±3.4) sí
son estables.

Recomendación: f1 o accuracy como criterio primario, ROI solo como filtro grueso
de descarte. Contra-argumento válido del usuario: el negocio es el ROI, no el f1.
**`p4-1` queda para revisar.**

### 3.3 Qué modelo se deploya con walk-forward

Hoy se guardan los artefactos (`.pkl`, predicciones) **solo del fold 1**, para no
sobrescribir 5 veces el mismo nombre y que producción siga encontrando un modelo
por combinación. Lo correcto sería: seleccionar **la configuración** por
walk-forward y después **reentrenar con todos los datos**. No está hecho.

## 4. Próximos pasos

Plan acordado con el usuario para el ciclo de mejoras:

| Paso | Qué | Estado |
|---|---|---|
| **0** | Que la métrica se pueda comparar entre corridas (reproducibilidad + walk-forward) | ✅ `8150568cc` |
| **1** | Entrenamiento completo (grid 128×3) de los 5 países → línea de base real | pendiente |
| **2** | Armar el plan de mejoras ordenado por impacto esperado | pendiente |
| **3** | Meter **un cambio a la vez** y correr el mismo entrenamiento para ver si mejora | pendiente |

Ver el protocolo del paso 3 en [`EXPERIMENTOS.md`](EXPERIMENTOS.md).

**Costo del paso 1:** el compute se paga **una vez por (país, `iteration_date`)**,
no por corrida. Con fecha nueva son 3-8 h por país solo de preparación de datos
(mapeo de sofifa ~1h28 + formateo ~1h48, medido en germany), más el grid. Total
estimado ~20 h. Con la caché del día ya armada, minutos.

**Hipótesis de dónde buscar mejoras** (sin validar): el modelo está peor calibrado
que las cuotas pero pierde menos a stake plano → apuntaría a **calibración de
probabilidades** (Platt / isotónica) y a **features que el bookie no capture**,
antes que a más modelos o hiperparámetros. También está pendiente la palanca de
performance más grande que queda: pasar a Parquet el parseo de `.xlsx` crudos
(los dos bloques de 1h30 de arriba).

## 5. Backlog del refactor que queda

Detalle y criterios en [`REFACTOR.md`](REFACTOR.md) §2.

| # | Qué | Prio |
|---|---|---|
| p4-4 | Podar `assess_model.py` (métricas sin uso tras `p4-7`) | P2 |
| g-4 | Separar requirements (scrape / train) + pinear `mnm` | P2 |
| p6-1 | Partir `main_next_matches.main` (~370 líneas) + sacar los 11 `input()` | P1 |
| p3-2 | Domar el sprawl de `prod`: core compartido, train/prod como wrappers finos | P1 |
| p6-2 | Resolver la historia de workflows (predicción diaria: ¿CI o manual?) | P1 |
| g-8 | Más cobertura de tests por fase | 🟡 |
| — | Actualizar `update_results.yml` cuando `g-10`/`g-11` lleguen a `prod` | — |

**Congelado** ❄️: `p2-1`, `p2-2`, `p2-8` (modernización de scrapers) — no se
justifica invertir ahí hasta que el modelo muestre edge real sobre el bookie.
