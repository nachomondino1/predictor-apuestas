# Protocolo de experimentos

Cómo meter un cambio y saber si mejoró de verdad. Este doc es el "paso 3" del plan
de [`ESTADO.md`](ESTADO.md) §4.

La precondición ya está: el entrenamiento es **reproducible** (misma config →
resultado bit-idéntico) y la evaluación es **walk-forward** con desvío entre folds.
Sin eso, el ciclo no era medible; ver `predictor/config.py` y la entrada "paso 0"
de [`REFACTOR.md`](REFACTOR.md).

## Las cuatro reglas

### 1. Un cambio a la vez

Dos cambios juntos y la métrica deja de decir cuál sirvió. Si un cambio "obliga" a
otro, es un solo experimento — y hay que anotarlo así en la nota de la corrida.

### 2. Misma fecha, mismo país, misma config

```bash
python scripts/real_train.py 48 2026-09-11 "qué cambié"
```

- **Misma `iteration_date`** → reusa la caché de datos. Con fecha nueva el pipeline
  regenera el mapeo de sofifa y el formateo (3-8 h) y el "antes" y el "después"
  dejan de compartir los datos de entrada.
- **Mismo país** (england por defecto: es el que tiene la caché tibia y corre en
  minutos). Un cambio que mejora england puede no mejorar italy; eso se verifica
  después, no durante.
- **Mismo grid.** Si el cambio agrega hiperparámetros, el grid crece y el máximo
  sobre más modelos sube por azar. Comparar a igual tamaño de grid.

### 3. Comparar por fold, no solo el promedio

El detalle sin promediar queda en `data/{país}/modeling/{fecha}/df_ite_test_folds.xlsx`.

Como la corrida es determinista y los folds son los mismos, la comparación es
**pareada**: mirar el signo del delta en cada fold.

| Evidencia | Lectura |
|---|---|
| mejora en 4/4 folds | creíble |
| mejora en 3/4 | plausible, anotarlo como tal |
| mejora en 2/4 | ruido, aunque el promedio suba |

Guía de magnitud sobre el promedio (england, 4 folds): el desvío entre folds es
**f1 ±2.3** y **accuracy ±3.4** → el error estándar del promedio ronda **1.2 en f1**.
**Una diferencia de menos de ~1 punto de f1 en el promedio no alcanza para declarar
mejora.** (Los folds no son independientes — las ventanas de train están anidadas —
así que esto es una guía gruesa; el criterio de los signos por fold es más fuerte.)

**ROI no sirve como métrica de comparación**: ±28.7 sobre una media de −25.6. Se
mira, pero no decide. Ver la decisión abierta en [`ESTADO.md`](ESTADO.md) §3.2.

### 4. No elegir el mejor de 384 y llamarlo mejora

Quedarse con el máximo del grid es sobreajustar el test: con 384 modelos, alguno
siempre parece bueno. Para comparar dos versiones del código, mirar el **promedio o
la mediana del grid**, o una **config fija** (la del smoke). El máximo sirve para
elegir qué deployar, no para medir si un cambio funcionó.

## Checklist de un experimento

1. `git status` limpio y `pytest -q` en verde **antes** de tocar nada.
2. Anotar en `REFACTOR.md` qué se va a probar y por qué (hipótesis, antes del resultado).
3. Hacer el cambio. Un commit.
4. Correr con la **misma fecha y país** que el baseline, con nota:
   ```bash
   python scripts/real_train.py 48 2026-09-11 "calibracion isotonica sobre predict_proba"
   ```
5. Comparar contra el baseline: promedio, desvío, **y los 4 deltas por fold**.
6. Anotar el resultado en la tabla de abajo — **también si no mejoró**. Un
   experimento negativo bien registrado ahorra repetirlo.
7. Si no mejoró: revertir el cambio, no dejarlo "por si acaso".

## Dónde quedan los resultados

| Archivo | Qué tiene |
|---|---|
| `data/_shared/logs/_training_log.xlsx` | 1 fila por corrida: commit, país, fecha, duración, métricas resumidas, `run_type`, `notes`. Se escribe solo. |
| `data/{país}/modeling/{fecha}/df_iteration.xlsx` | 1 fila por combinación × modelo: métricas promediadas + `std_<métrica>` + `n_folds`. |
| `data/{país}/modeling/{fecha}/df_ite_test_folds.xlsx` | el detalle por fold, sin promediar. **Es el que se usa para comparar.** |

⚠️ Las corridas **anteriores al 2026-09-13** del historial no son comparables con
las de después: antes el resultado era una muestra al azar (sin `SEED`), y el split
de test era uno solo de ~73-99 partidos en vez de 4 ventanas de 200.

## Baseline vigente

england · `iteration_date` 2026-09-11 · smoke (1 config, LogisticRegression) ·
4 folds efectivos · commit `8150568cc`:

| | promedio | desvío | por fold |
|---|---|---|---|
| f1 | 45.3 | ± 2.3 | 44.92 / 45.52 / 48.64 / 42.12 |
| accuracy | 53.1 | ± 3.4 | — |
| ROI | −25.6 | ± 28.7 | −33.96 / −48.25 / +23.39 / −43.51 |

Referencia externa: el bookie en el mismo test da **accuracy 57.3**.

**Falta el baseline con grid completo** (paso 1 del plan) — el de arriba es un
smoke de una sola config.

## Registro de experimentos

| Fecha | Cambio | Commit | País/fecha | f1 (Δ) | acc (Δ) | folds a favor | ¿Queda? |
|---|---|---|---|---|---|---|---|
| 2026-09-13 | *baseline* — paso 0 (reproducibilidad + walk-forward) | `8150568cc` | england / 2026-09-11 | 45.3 | 53.1 | — | ✅ |

## Ideas a probar (sin validar, ordenadas por impacto esperado)

1. **Calibración de probabilidades** (Platt / isotónica). El modelo está peor
   calibrado que las cuotas (log-loss 1.03 vs 0.96) pero pierde menos a stake plano
   → la señal existe y está mal escalada. Es el candidato más barato.
2. **Features que el bookie no capture.** Acierto 1X2 no le gana a bet365, así que
   competir en el mismo terreno informativo no alcanza.
3. **Resolver el NaN que se come el fold 5** (decisión abierta §3.1 de `ESTADO.md`):
   más historia usable = más test y más train.
4. **Seleccionar por f1/accuracy en vez de ROI** (decisión abierta §3.2). No mejora
   el modelo, mejora la *elección* del modelo — que hoy es en buena parte azar.
5. **Reentrenar con todos los datos** la config elegida por walk-forward (§3.3).
6. Más modelos / más hiperparámetros: **lo último**. Con 384 modelos ya, el problema
   no es falta de búsqueda.
